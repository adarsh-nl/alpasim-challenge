#!/usr/bin/env python3
"""R4 — per-route zero-list dashboard with root-cause buckets.

"Report the tail": for every route, emit score, a zero/near-zero flag, and a
root-cause bucket. Zeros are what DriveIRT punishes (zero-inflated, peer-
relative), so the tail is the only part of the distribution worth optimising —
and this makes it visible.

Bucketing follows the evaluator's own gate, read from `score_criteria` in the
results summary rather than hardcoded:

    collision_at_fault == 0  AND  offroad == 0   (hard gates)
    then progress_score = min(clamp(progress_clipped_rel,0,1)/0.8, 1.0)
    with progress_score = 1.0 when gt_dist_traveled_m < 5.0   (short-GT override)

So a zero is always one of: a gate failure (collision / excursion), an infra
failure (no usable rollout), or zero progress. The buckets are ordered to
attribute to the *first* cause that applies.

Usage:
    python3 tools/zero_dashboard.py <run_dir> [--near 0.8] [--md out.md]
    python3 tools/zero_dashboard.py ~/alpasim/runs/array-538078
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter

import numpy as np   # paired gate only; the single-run dashboard stays stdlib

ZERO = 0.01


def load_rollouts(run_dir: str) -> tuple[list[dict], dict]:
    """Collect rollouts from a run dir or an array dir of task-* subdirs."""
    paths = sorted(glob.glob(os.path.join(run_dir, "aggregate", "*.json")))
    paths += sorted(glob.glob(os.path.join(run_dir, "task-*", "aggregate", "*.json")))
    rollouts, criteria = [], {}
    seen = set()
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        criteria = criteria or d.get("score_criteria", {})
        for r in d.get("rollouts", []):
            key = r.get("rollout_id") or r.get("clipgt_id")
            if key in seen:
                continue
            seen.add(key)
            rollouts.append(r)
    return rollouts, criteria


def g(r: dict, key: str, default=0.0) -> float:
    v = (r.get("metrics") or {}).get(key, default)
    return default if v is None else float(v)


def bucket(r: dict) -> tuple[str, str]:
    """Return (bucket, one-line evidence). Ordered: first applicable cause wins."""
    status = r.get("status", "")
    reason = r.get("failure_reason") or ""

    # infra / no usable rollout -- not a driving failure, do not blame the policy
    if g(r, "img_is_black") > 0:
        return "infra:black_frames", "renderer produced black frames"
    if status not in ("pass", "fail"):
        return "infra:status", f"status={status!r}"

    # hard gates, in the evaluator's own order
    if g(r, "collision_at_fault") > 0:
        which = []
        if g(r, "collision_front") > 0:
            which.append("front")
        if g(r, "collision_lateral") > 0:
            which.append("lateral")
        obs = g(r, "min_distance_to_obstacle_m", 9.0)
        return ("collision",
                f"at-fault ({'+'.join(which) or 'unspecified'}), minObs={obs:.2f}m")
    if g(r, "offroad") > 0:
        return ("excursion",
                f"offroad, minLaneDist={g(r,'min_distance_to_lane_boundary_m',9.0):.2f}m")

    # gates passed -> any shortfall is progress
    prog = g(r, "progress_clipped_rel")
    if prog <= 0.02:
        return "no_progress", f"progress_clipped_rel={prog:.3f} (stuck/abstained)"
    if r.get("score", 1.0) < 1.0:
        return ("low_progress",
                f"progress_clipped_rel={prog:.3f} of the 0.80 cap")
    if reason:
        return "other", f"failure_reason={reason}"
    return "clean", "gates passed, progress capped out"


# ---------------------------------------------------------------------------
# paired NEAR-MISS MARGIN gate
# ---------------------------------------------------------------------------
# Why this exists: the 200-scene set is saturated on score (198/200 == 1.000), so
# a score gate returns INCONCLUSIVE for anything outside the 2 zeros. But
# min_distance_to_obstacle_m still varies widely on PASSING scenes (0.02 m to
# 9 m in that same sweep), and it is the axis the zeros live on -- both reached
# 0.00 m. So margin is an informative signal where score is dead.
#
# The trap this guards against: margin is trivially bought by driving slower.
# A driver that brakes everywhere improves every margin and is worse. So the
# gate is a CONJUNCTION -- margin must improve on the scenes that are actually
# tight, AND score/progress must not regress.

TAIL_M = 1.0          # a scene is "tight" if baseline margin is below this
MARGIN_FLOOR = 9.0    # metric is capped/absent above this; treat as "no obstacle"
# Minimum margin gain worth calling a win. Without this the gate ships noise:
# the gov-governor pair improved tight-scene margin by 0.002 m (2 mm) with
# p=0.49 and was greenlit by an earlier version of this verdict. 5 cm is still
# small but is at least physically meaningful next to a 2 m vehicle width.
MIN_GAIN_M = 0.05


def _bootstrap_ci(d, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    if len(d) == 0:
        return (float("nan"), float("nan"))
    m = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(m, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _perm_p(d, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    if len(d) == 0:
        return float("nan")
    obs = abs(float(np.mean(d)))
    c = sum(1 for _ in range(n_perm)
            if abs(float(np.mean(d * rng.choice([-1, 1], len(d))))) >= obs)
    return (c + 1) / (n_perm + 1)


def _by_scene(run_dir):
    rollouts, _ = load_rollouts(run_dir)
    out = {}
    for r in rollouts:
        sid = r.get("clipgt_id") or ""
        out[sid] = {
            "score": float(r.get("score", 0.0) or 0.0),
            "prog": g(r, "progress_clipped_rel"),
            "margin": g(r, "min_distance_to_obstacle_m", MARGIN_FLOOR),
            "dist": g(r, "dist_traveled_m"),
            "bucket": bucket(r)[0],
        }
    return out


def paired_margin_gate(baseline_dir, candidate_dir, tol=0.02) -> int:
    b, c = _by_scene(baseline_dir), _by_scene(candidate_dir)
    shared = sorted(set(b) & set(c))
    if not shared:
        print("NO SHARED SCENES", file=sys.stderr)
        return 1

    print("=== PAIRED NEAR-MISS MARGIN GATE ===")
    print(f"  baseline : {baseline_dir}")
    print(f"  candidate: {candidate_dir}")
    print(f"  shared scenes: {len(shared)}\n")

    zb = sum(1 for s in shared if b[s]["score"] < ZERO)
    zc = sum(1 for s in shared if c[s]["score"] < ZERO)
    print(f"0. ZEROS (the primary bar)   baseline {zb}  ->  candidate {zc}"
          f"   {'WORSE' if zc > zb else ('better' if zc < zb else 'unchanged')}")

    mb = np.array([b[s]["margin"] for s in shared])
    mc = np.array([c[s]["margin"] for s in shared])
    dm = mc - mb
    tight = mb < TAIL_M
    print(f"\n1. MARGIN  min_distance_to_obstacle_m")
    print(f"   all scenes      median {np.median(mb):5.2f} -> {np.median(mc):5.2f} m"
          f"   (delta {np.median(dm):+.2f})")
    print(f"   tight scenes (baseline < {TAIL_M} m): {int(tight.sum())}/{len(shared)}")
    if tight.any():
        dt_ = dm[tight]
        lo, hi = _bootstrap_ci(dt_)
        print(f"     margin on tight scenes: {mb[tight].mean():.2f} -> "
              f"{mc[tight].mean():.2f} m   delta {dt_.mean():+.3f} m")
        print(f"     95% CI [{lo:+.3f}, {hi:+.3f}]   p = {_perm_p(dt_):.4f}")
    else:
        print("     (no tight scenes -- this pair cannot exercise the margin gate)")

    sb = np.array([b[s]["score"] for s in shared])
    sc = np.array([c[s]["score"] for s in shared])
    pb = np.array([b[s]["prog"] for s in shared])
    pc = np.array([c[s]["prog"] for s in shared])
    ds, dp = sc - sb, pc - pb
    print(f"\n2. NON-REGRESSION GUARD (margin bought by braking must not pass)")
    print(f"   score    {sb.mean():.4f} -> {sc.mean():.4f}   delta {ds.mean():+.4f}")
    print(f"   progress {pb.mean():.4f} -> {pc.mean():.4f}   delta {dp.mean():+.4f}")
    dd = np.array([c[s]["dist"] - b[s]["dist"] for s in shared])
    print(f"   distance travelled delta: {dd.mean():+.2f} m "
          f"({'slower' if dd.mean() < 0 else 'further'})")

    gain = float(dm[tight].mean()) if tight.any() else 0.0
    ci_lo = _bootstrap_ci(dm[tight])[0] if tight.any() else 0.0
    # A win must be MATERIAL (>= MIN_GAIN_M) and RESOLVED (CI excludes zero).
    # Either alone is not enough: a 2 mm gain can have a CI excluding zero and
    # still be physically meaningless.
    margin_up = bool(tight.any() and gain >= MIN_GAIN_M and ci_lo > 0)
    margin_trivial = bool(tight.any() and 0 < gain < MIN_GAIN_M)
    score_ok = bool(ds.mean() >= -tol)
    prog_ok = bool(dp.mean() >= -tol)
    # Braking is detectable even when the margin gain is trivial -- the tell is
    # progress falling, not margin rising.
    braking = bool(gain > 0 and (not prog_ok or dp.mean() < -tol / 2))

    print(f"\n=== VERDICT (tolerance {tol}, min material gain {MIN_GAIN_M} m) ===")
    if zc > zb:
        print("   HOLD  candidate adds a zero. Nothing else matters.")
    elif braking:
        print("   HOLD  margin improved but progress regressed "
              f"({dp.mean():+.4f}) -- this is margin bought by braking, "
              "which is exactly what the guard exists to catch.")
    elif margin_up and score_ok and prog_ok:
        print("   SHIP-CANDIDATE  margin improved on tight scenes by "
              f"{gain:+.3f} m (CI excludes 0) with no score/progress regression.")
    elif margin_trivial:
        print(f"   INCONCLUSIVE  tight-scene margin gain {gain:+.3f} m is below "
              f"the {MIN_GAIN_M} m materiality floor -- statistically it may be "
              f"real, physically it is noise next to a 2 m vehicle width.")
    elif not tight.any():
        print("   INCONCLUSIVE  no tight scenes in the shared set; the margin "
              "signal is untested here. Use a scene set containing near-misses.")
    else:
        print("   INCONCLUSIVE  margin did not improve on tight scenes.")

    print(f"\n   worst tight scenes (baseline margin ascending):")
    for s in sorted(shared, key=lambda s: b[s]["margin"])[:6]:
        print(f"     {b[s]['margin']:5.2f} -> {c[s]['margin']:5.2f} m   "
              f"score {b[s]['score']:.2f} -> {c[s]['score']:.2f}   "
              f"prog {b[s]['prog']:.2f} -> {c[s]['prog']:.2f}   {s[-34:]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--baseline", default="", help="paired margin gate: baseline run")
    ap.add_argument("--candidate", default="", help="paired margin gate: candidate run")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="score/progress non-regression tolerance")
    ap.add_argument("--near", type=float, default=0.8,
                    help="near-zero threshold (default 0.8)")
    ap.add_argument("--md", default="", help="also write a markdown report here")
    args = ap.parse_args()

    if args.baseline and args.candidate:
        return paired_margin_gate(args.baseline, args.candidate, args.tol)
    if not args.run_dir:
        ap.error("give a run_dir, or --baseline and --candidate")

    rollouts, criteria = load_rollouts(args.run_dir)
    if not rollouts:
        print(f"no rollouts found under {args.run_dir}", file=sys.stderr)
        return 1

    rows = []
    for r in rollouts:
        b, ev = bucket(r)
        s = float(r.get("score", 0.0) or 0.0)
        rows.append({
            "scene": (r.get("clipgt_id") or "?"),
            "score": s,
            "zero": s < ZERO,
            "near": ZERO <= s < args.near,
            "bucket": b,
            "evidence": ev,
            "prog": g(r, "progress_clipped_rel"),
            "gt_m": g(r, "gt_dist_traveled_m"),
            "obs_m": g(r, "min_distance_to_obstacle_m", 9.0),
            "dgt": g(r, "dist_to_gt_trajectory", -1),
        })
    rows.sort(key=lambda x: (x["score"], x["prog"]))

    n = len(rows)
    scores = [x["score"] for x in rows]
    zeros = [x for x in rows if x["zero"]]
    nears = [x for x in rows if x["near"]]
    perfect = sum(1 for s in scores if s >= 0.9999)

    out = []
    out.append(f"run: {args.run_dir}")
    out.append(f"routes: {n}   mean score: {statistics.fmean(scores):.4f}")
    out.append(f"ZEROS (<{ZERO}): {len(zeros)}   "
               f"near-zeros (<{args.near}): {len(nears)}   "
               f"perfect (1.000): {perfect}  ({perfect/n:.1%})")
    out.append("")

    out.append("root-cause buckets:")
    for b, c in Counter(x["bucket"] for x in rows).most_common():
        out.append(f"  {c:4d}  {b}")
    out.append("")

    # rear collisions are NOT at fault and cost nothing -- never report them as
    # failures. Conflating rear with front is how a free event gets mistaken for
    # a zero (and vice versa).
    rear_free = sum(1 for r in rollouts
                    if g(r, "collision_rear") > 0 and g(r, "collision_at_fault") == 0)
    at_fault_front = sum(1 for r in rollouts if g(r, "collision_front") > 0)
    at_fault_lat = sum(1 for r in rollouts if g(r, "collision_lateral") > 0)
    out.append(f"collisions: at-fault front={at_fault_front}  "
               f"at-fault lateral={at_fault_lat}  "
               f"rear (NOT at fault, free)={rear_free}")
    out.append("")

    if zeros or nears:
        out.append(f"{'score':>6} {'bucket':<20} {'prog':>5} {'gt_m':>6} "
                   f"{'obs_m':>6}  scene  <- evidence")
        out.append("-" * 100)
        for x in zeros + nears:
            out.append(f"{x['score']:6.3f} {x['bucket']:<20} {x['prog']:5.2f} "
                       f"{x['gt_m']:6.1f} {x['obs_m']:6.2f}  "
                       f"{x['scene'][-44:]}  <- {x['evidence']}")
    else:
        out.append("no zeros and no near-zeros at these thresholds.")
    out.append("")

    # saturation warning: if nearly everything is perfect there is no gradient
    if perfect / n > 0.9:
        out.append(f"!! SATURATION: {perfect}/{n} routes scored exactly 1.000. "
                   f"This set provides no gradient outside the "
                   f"{len(zeros)+len(nears)} tail route(s); any tuning measured "
                   f"on it is measuring noise. A larger sweep is required.")
        out.append("")

    if criteria:
        out.append("evaluator score_criteria (from the run, not assumed):")
        for k, v in criteria.items():
            out.append(f"  {k}: {v}")

    text = "\n".join(out)
    print(text)

    if args.md:
        with open(args.md, "w") as f:
            f.write("# Per-route zero list\n\n```\n" + text + "\n```\n")
        print(f"\nwrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
