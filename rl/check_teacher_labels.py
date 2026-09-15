#!/usr/bin/env python3
"""D1 — label-integrity harness for the Diffusion-Planner teacher label set.

Checks, in order of how cheap they are to run:

  1. SCHEMA      keys, shapes, dtypes, finiteness
  2. DEDUPE      one label per (scene, timestep) token; no collisions
  3. KINEMATICS  recomputes run_teacher.py's own audit_flags criteria from the
                 stored ego_future (NONFINITE / V_MAX>40 / ACCEL>12 / BACKWARDS)
  4. COVERAGE    scenes, cities, timestep stride, samples per scene
  5. ADE vs GT   (--ade) reconstructs the ground-truth ego future from the
                 trajdata cache and compares against the teacher's prediction

Why 5 is separate and opt-in: the stored labels contain NO ground truth and NO
inputs, so ADE cannot be recomputed from the label set alone — it needs the
trajdata cache. It is also the only check that depends on my reconstruction of
the frame convention being right, so it self-validates: a correct convention
yields sub-metre ADE, a wrong one yields tens of metres. If it comes out
nonsense the harness says so rather than reporting the number.

CPU-only, read-only. Usage:

    # fast checks over the whole set
    python3 rl/check_teacher_labels.py --labels rl/teacher_labels

    # add the GT comparison on a sample (needs the dataset)
    python3 rl/check_teacher_labels.py --labels rl/teacher_labels \\
        --ade --ade-sample 200 --root /datasets/eemcs/ps/alpasim-nuplan-2026
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np

# Expected schema, from run_teacher.py's writer
EXPECTED = {
    "ego_future": ((80, 4), np.float32),
    "neighbor_future": ((10, 80, 4), np.float32),
    "token": ((), None),
    "map_name": ((), None),
}
DT = 0.1                      # cache cadence (dt0.10)
V_MAX_FLAG = 40.0             # m/s   } same thresholds as
A_MAX_FLAG = 12.0             # m/s^2 } run_teacher.audit_flags
BACKWARDS_M = -1.0            # m


def kinematics(ego: np.ndarray) -> dict:
    """Speed/accel profile of an (80,4) ego_future in the ego frame."""
    xy = ego[:, :2].astype(np.float64)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    v = d / DT
    a = np.diff(v) / DT
    return {
        "v_max": float(v.max()) if len(v) else 0.0,
        "a_max": float(np.abs(a).max()) if len(a) else 0.0,
        "x_min": float(xy[:, 0].min()),
        "dist": float(d.sum()),
        "finite": bool(np.isfinite(ego).all()),
    }


def flags_for(k: dict) -> list[str]:
    f = []
    if not k["finite"]:
        f.append("NONFINITE")
    if k["v_max"] > V_MAX_FLAG:
        f.append("V_MAX>40")
    if k["a_max"] > A_MAX_FLAG:
        f.append("ACCEL>12")
    if k["x_min"] < BACKWARDS_M:
        f.append("BACKWARDS")
    return f


# ---------------------------------------------------------------------------
# ADE vs ground truth
# ---------------------------------------------------------------------------

MIN_GT_STEPS = 10             # 1.0 s — below this a comparison says nothing


def gt_future_ego_frame(scene_dir, ts: int, horizon: int = 80):
    """Reconstruct the GT ego future in the ego frame at `ts`.

    Mirrors trajdata_to_dp.se2_world_to_ego: translate by the ego position at
    `ts`, then rotate by -heading(ts).

    Returns as many steps as the cache actually holds, up to `horizon` — it does
    NOT pad or require the full horizon. The AlpaSim trajdata scenes are 56 ego
    timesteps (5.6 s) and labels are sampled at ts in 20..55, so the GT future
    available after `ts` is a median of ~17 steps and is ZERO at ts=55. The
    teacher predicts 80 steps (8 s), i.e. mostly beyond the recorded scene, so
    any ADE-vs-GT figure is only meaningful with its horizon stated.
    """
    import pandas as pd
    f = os.path.join(scene_dir, "agent_data_dt0.10.feather")
    if not os.path.exists(f):
        return None
    df = pd.read_feather(f)
    ego = df[df["agent_id"] == "ego"]
    if ego.empty:
        return None
    col_t = "scene_ts" if "scene_ts" in ego.columns else ego.columns[0]
    ego = ego.sort_values(col_t)
    t = ego[col_t].values
    i0 = int(np.searchsorted(t, ts))
    if i0 >= len(t) or t[i0] != ts:
        return None
    fut = ego.iloc[i0 + 1: i0 + 1 + horizon]
    if len(fut) < MIN_GT_STEPS:
        return None
    anchor = ego.iloc[i0]
    h = float(anchor["heading"]) if "heading" in ego.columns else 0.0
    c, s = math.cos(-h), math.sin(-h)
    rot = np.array([[c, -s], [s, c]])
    rel = fut[["x", "y"]].values - np.array(
        [float(anchor["x"]), float(anchor["y"])])
    return rel @ rot.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ade", action="store_true",
                    help="also compare against GT from the trajdata cache")
    ap.add_argument("--ade-sample", type=int, default=200)
    ap.add_argument("--root", default="/datasets/eemcs/ps/alpasim-nuplan-2026")
    ap.add_argument("--json", default="", help="write a machine-readable summary")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.labels, "*.npz")))
    if not files:
        print(f"no .npz under {args.labels}", file=sys.stderr)
        return 1

    print(f"=== D1 label integrity: {len(files)} files under {args.labels} ===\n")

    schema_bad, tokens, per_scene = [], Counter(), defaultdict(list)
    flagged, kin_all, cities = [], [], Counter()
    unreadable = []

    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as e:
            unreadable.append((os.path.basename(p), repr(e)))
            continue

        # 1. schema
        missing = [k for k in EXPECTED if k not in z.files]
        if missing:
            schema_bad.append((os.path.basename(p), f"missing {missing}"))
            continue
        for k, (shape, dtype) in EXPECTED.items():
            a = z[k]
            if shape and a.shape != shape:
                schema_bad.append((os.path.basename(p), f"{k} shape {a.shape} != {shape}"))
            if dtype and a.dtype != dtype:
                schema_bad.append((os.path.basename(p), f"{k} dtype {a.dtype} != {dtype}"))

        tok = str(z["token"])
        tokens[tok] += 1
        cities[str(z["map_name"])] += 1
        scene, _, ts = tok.rpartition("_")
        per_scene[scene].append(int(ts) if ts.isdigit() else -1)

        k = kinematics(z["ego_future"])
        kin_all.append(k)
        fl = flags_for(k)
        if fl:
            flagged.append((tok, fl, k))

    n_ok = len(kin_all)

    # --- report ---------------------------------------------------------
    print("1. SCHEMA")
    print(f"   readable      : {len(files)-len(unreadable)}/{len(files)}")
    print(f"   schema errors : {len(schema_bad)}")
    for name, why in (unreadable + schema_bad)[:8]:
        print(f"     !! {name}: {why}")
    print(f"   -> {'PASS' if not schema_bad and not unreadable else 'FAIL'}\n")

    dups = {t: c for t, c in tokens.items() if c > 1}
    print("2. DEDUPE")
    print(f"   unique tokens : {len(tokens)}")
    print(f"   duplicates    : {len(dups)}")
    for t, c in list(dups.items())[:5]:
        print(f"     !! {t} x{c}")
    print(f"   -> {'PASS' if not dups else 'FAIL'}\n")

    print("3. KINEMATICS (run_teacher.audit_flags criteria, recomputed)")
    rate = len(flagged) / max(n_ok, 1)
    print(f"   flagged       : {len(flagged)}/{n_ok}  ({rate:.4%})")
    hist = Counter(f for _, fl, _ in flagged for f in fl)
    for f, c in hist.most_common():
        print(f"     {c:5d}  {f}")
    for tok, fl, k in flagged[:5]:
        print(f"     e.g. {tok[-42:]} {fl} v_max={k['v_max']:.1f} a_max={k['a_max']:.1f}")
    if kin_all:
        print(f"   v_max  median {statistics.median(k['v_max'] for k in kin_all):6.2f} m/s"
              f"   p99 {sorted(k['v_max'] for k in kin_all)[int(.99*n_ok)]:6.2f}")
        print(f"   a_max  median {statistics.median(k['a_max'] for k in kin_all):6.2f} m/s^2"
              f" p99 {sorted(k['a_max'] for k in kin_all)[int(.99*n_ok)]:6.2f}")
        print(f"   8s dist median {statistics.median(k['dist'] for k in kin_all):6.2f} m")
    print()

    print("4. COVERAGE")
    print(f"   scenes        : {len(per_scene)}")
    print(f"   samples/scene : median {statistics.median(len(v) for v in per_scene.values()):.0f}"
          f"  min {min(len(v) for v in per_scene.values())}"
          f"  max {max(len(v) for v in per_scene.values())}")
    strides = set()
    for v in per_scene.values():
        sv = sorted(x for x in v if x >= 0)
        strides.update(b - a for a, b in zip(sv, sv[1:]))
    print(f"   ts strides    : {sorted(strides)[:8]}  (expect [5] for 2 Hz)")
    print(f"   cities        : {dict(cities)}")
    print()

    ade_summary = None
    print("5. ADE vs GROUND TRUTH")
    if not args.ade:
        print("   skipped (--ade not set).")
        print("   NOTE: the labels store no GT and no inputs, so ADE is NOT")
        print("   recomputable from the label set alone -- it needs the trajdata")
        print("   cache. run_teacher.py never computed it either.")
    else:
        import random
        rng = random.Random(0)
        sample = rng.sample(files, min(args.ade_sample, len(files)))
        ades, ade1s, horizons, missing_gt = [], [], [], 0
        for p in sample:
            z = np.load(p, allow_pickle=True)
            tok = str(z["token"])
            scene, _, ts = tok.rpartition("_")
            gt = gt_future_ego_frame(os.path.join(args.root, "nuplan_test", scene),
                                     int(ts))
            if gt is None:
                missing_gt += 1
                continue
            pred = z["ego_future"][:, :2].astype(np.float64)
            n = min(len(gt), len(pred))
            horizons.append(n)
            err = np.linalg.norm(pred[:n] - gt[:n], axis=1)
            ades.append(float(err.mean()))
            ade1s.append(float(err[:MIN_GT_STEPS].mean()))
        if not ades:
            print(f"   could not reconstruct GT for any of {len(sample)} sampled "
                  f"tokens ({missing_gt} missing). Cannot report ADE.")
        else:
            med = statistics.median(ades)
            ades_s = sorted(ades)
            med1 = statistics.median(ade1s)
            ade_summary = {"n": len(ades), "median": med,
                           "p90": ades_s[int(.9*len(ades))], "max": ades_s[-1],
                           "median_at_1s": med1,
                           "horizon_steps_median": statistics.median(horizons),
                           "horizon_steps_max": max(horizons),
                           "missing_gt": missing_gt,
                           "min_gt_steps_required": MIN_GT_STEPS}
            print(f"   compared      : {len(ades)}/{len(sample)}  "
                  f"({missing_gt} had <{MIN_GT_STEPS} GT steps available)")
            print(f"   GT horizon    : median {statistics.median(horizons):.0f} steps "
                  f"({statistics.median(horizons)*DT:.1f}s), max {max(horizons)} "
                  f"-- teacher predicts 80 (8.0s)")
            print(f"   ADE @1.0s     : {med1:.3f} m   <- the comparable window")
            print(f"   ADE @available: {med:.3f} m  (p90 {ade_summary['p90']:.3f}, "
                  f"max {ade_summary['max']:.3f})")
            if med > 10.0:
                print("   !! MEDIAN ADE IS IMPLAUSIBLE (>10 m). Treat this number as")
                print("      INVALID: it most likely means my reconstruction of the")
                print("      frame convention (anchor/heading/timestep alignment) is")
                print("      wrong, not that the teacher is bad. Do not report it.")
            elif med > 2.0:
                print("   !  higher than the ~0.58 m previously claimed -- worth")
                print("      confirming the convention before trusting either number.")
            else:
                print("   -> plausible; convention appears correct (sub-metre ADE is")
                print("      self-validating for this check)")
    print()

    verdict = "PASS" if (not schema_bad and not unreadable and not dups) else "FAIL"
    print(f"=== VERDICT: {verdict} "
          f"(schema/dedupe hard checks; kinematics flag-rate {rate:.4%}) ===")

    if args.json:
        json.dump({
            "n_files": len(files), "n_ok": n_ok,
            "schema_errors": len(schema_bad), "unreadable": len(unreadable),
            "unique_tokens": len(tokens), "duplicates": len(dups),
            "flag_rate": rate, "flag_hist": dict(hist),
            "scenes": len(per_scene), "cities": dict(cities),
            "ts_strides": sorted(strides), "ade": ade_summary,
            "verdict": verdict,
        }, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
