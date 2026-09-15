#!/usr/bin/env python3
"""Task 3 — is the teacher's caution predictable from what the SHIPPED driver sees?

The hedge against the render route dying: gate the existing geometric driver's
speed using the teacher's conservatism, learned offline. That only works if the
teacher's caution is predictable from the shipped driver's actual observation
space. This probe answers that before anything is built on it.

THE SHIPPED DRIVER'S OBSERVATION SPACE (submission/route_follower.py):
  - route waypoints in rig frame  (navigation goal; a fixed path, not traffic)
  - ego pose history              (PAST only)
  - current speed
It sees no agents, no images, no map beyond the route.

FEATURES USED HERE are restricted to leak-free analogues of exactly that:
  v_now, a_recent        from PAST ego only
  lane curvature ahead   at 10/20/30 m along the current lane (the driver gets
                         the equivalent via its route waypoints)
  lane_remaining_m       arc length to the end of the current lane chain
  n_lanes_ahead          branching -- a proxy for junction complexity

DELIBERATELY EXCLUDED as leakage: anything from the ego's FUTURE (the human's
realised path). The route-arc-length feature the abstention driver uses is a
legitimate signal in the sim but cannot be reconstructed offline without the
sim's route, and approximating it with the GT future path would leak the answer.
That limitation is reported, not hidden.

TARGET: the teacher's planned distance over the next 8 s (from ego_future).
Low planned distance = the teacher is being cautious.

CPU-only, read-only. Usage:
    ~/alpasim/.venv/bin/python rl/probe_abstention_gate.py \\
        --labels rl/teacher_labels --limit 1500
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from functools import lru_cache

import numpy as np

ROOT = os.environ.get("ALPASIM_NUPLAN_ROOT", "")
if not ROOT or not os.path.isdir(os.path.join(ROOT, "nuplan_test")):
    ROOT = "/datasets/eemcs/ps/alpasim-nuplan-2026"
DT = 0.1
PAST_STEPS = 10          # 1.0 s of ego history, as the driver's pose ring gives
V_MAX, A_LAT_MAX = 20.0, 7.0     # the rank-1 shipped config


@lru_cache(maxsize=8)
def get_map(location: str):
    from pathlib import Path
    from trajdata import MapAPI
    return MapAPI(Path(ROOT)).get_map(f"nuplan_test:{location}",
                                      incl_road_areas=True)


@lru_cache(maxsize=256)
def scene_ego(scene: str):
    import dill
    import pandas as pd
    d = os.path.join(ROOT, "nuplan_test", scene)
    md = dill.load(open(os.path.join(d, "scene_metadata_dt0.10.dill"), "rb"))
    df = pd.read_feather(os.path.join(d, "agent_data_dt0.10.feather"))
    ego = df[df["agent_id"] == "ego"].sort_values("scene_ts")
    return ego[["x", "y", "heading"]].values.astype(np.float64), str(md.location)


def lane_features(location: str, x: float, y: float, h: float) -> dict:
    """Curvature ahead + remaining lane length, from the map only."""
    vm = get_map(location)
    out = {"kappa_10": 0.0, "kappa_20": 0.0, "kappa_30": 0.0,
           "lane_remaining_m": 0.0, "n_lanes_ahead": 0, "lane_found": 0}
    lanes = vm.get_current_lane(np.array([x, y, 0.0, h]), max_dist=4.0,
                                max_heading_error=0.8)
    if not lanes:
        cl = vm.get_closest_lane(np.array([x, y, 0.0]))
        lanes = [cl] if cl is not None else []
    if not lanes:
        return out
    ln = lanes[0]
    out["lane_found"] = 1
    out["n_lanes_ahead"] = len(ln.next_lanes)

    # walk the lane chain forward, accumulating a centreline polyline
    pts, cur, hops = [], ln, 0
    while cur is not None and hops < 6:
        c = np.asarray(cur.center.xyz)[:, :2]
        pts.append(c)
        nxt = list(cur.next_lanes)
        cur = None
        if nxt:
            by_id = {str(l.id): l for l in vm.lanes}
            cur = by_id.get(str(nxt[0]))
        hops += 1
    poly = np.vstack(pts) if pts else None
    if poly is None or len(poly) < 4:
        return out

    d0 = np.linalg.norm(poly - np.array([x, y]), axis=1)
    i0 = int(d0.argmin())
    fwd = poly[i0:]
    if len(fwd) < 4:
        return out
    seg = np.linalg.norm(np.diff(fwd, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out["lane_remaining_m"] = float(s[-1])

    dx = np.gradient(fwd[:, 0]); dy = np.gradient(fwd[:, 1])
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    kap = np.abs(dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-6)
    for tag, dist in (("kappa_10", 10.0), ("kappa_20", 20.0), ("kappa_30", 30.0)):
        if s[-1] >= dist:
            j = int(np.searchsorted(s, dist))
            out[tag] = float(np.nanmax(kap[:max(j, 2)]))
    return out


def follower_planned_dist(v_now: float, kap_ahead: float, horizon_s: float = 8.0
                          ) -> float:
    """What the rank-1 curvature-limited profile would travel, blind to traffic.

    Mirrors submission/route_follower.py: v_cap = min(V_MAX, sqrt(A_LAT/kappa)),
    approached from v_now. Traffic-blind by construction -- that is the point.
    """
    v_cap = min(V_MAX, math.sqrt(A_LAT_MAX / max(kap_ahead, 1e-3)))
    v = v_now
    d = 0.0
    for _ in range(int(horizon_s / DT)):
        v += np.clip(v_cap - v, -6.0 * DT, 6.0 * DT)
        d += v * DT
    return float(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.labels, "*.npz")))
    if args.limit and args.limit < len(files):
        step = max(1, len(files) // args.limit)
        files = files[::step][:args.limit]
    print(f"=== abstention-gate predictability probe: {len(files)} labels ===\n")

    rows, skipped = [], 0
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        tok = str(z["token"])
        scene, _, ts_s = tok.rpartition("_")
        ts = int(ts_s)
        try:
            ego, loc = scene_ego(scene)
            if ts >= len(ego) or ts < PAST_STEPS:
                skipped += 1
                continue
            past = ego[ts - PAST_STEPS: ts + 1]
            seg = np.linalg.norm(np.diff(past[:, :2], axis=0), axis=1)
            v_now = float(seg[-1] / DT)
            a_recent = float((seg[-1] - seg[0]) / (PAST_STEPS * DT * DT))
            lf = lane_features(loc, ego[ts][0], ego[ts][1], ego[ts][2])
            fut = z["ego_future"][:, :2].astype(np.float64)
            d_teacher = float(np.linalg.norm(np.diff(fut, axis=0), axis=1).sum())
            kap = max(lf["kappa_10"], lf["kappa_20"])
            d_follower = follower_planned_dist(v_now, kap)
            rows.append({**lf, "v_now": v_now, "a_recent": a_recent,
                         "d_teacher": d_teacher, "d_follower": d_follower,
                         "overshoot": d_follower - d_teacher, "token": tok})
        except Exception as e:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  skip {tok[-36:]}: {type(e).__name__}: {e}")
        if (i + 1) % 300 == 0:
            print(f"  ... {i+1}/{len(files)}", flush=True)

    if not rows:
        print("no usable rows", file=sys.stderr)
        return 1
    n = len(rows)
    ov = np.array([r["overshoot"] for r in rows])
    dt_ = np.array([r["d_teacher"] for r in rows])
    df_ = np.array([r["d_follower"] for r in rows])

    print(f"\nusable rows: {n} ({skipped} skipped)\n")
    print("1. DOES THE BLIND FOLLOWER OVERSHOOT THE TEACHER?")
    print(f"   teacher 8 s distance : median {np.median(dt_):6.2f} m")
    print(f"   follower 8 s distance: median {np.median(df_):6.2f} m")
    print(f"   overshoot (f - t)    : median {np.median(ov):+6.2f} m  "
          f"p90 {np.percentile(ov,90):+6.2f}  max {ov.max():+6.2f}")
    frac = float((ov > 5.0).mean())
    print(f"   overshoots by >5 m   : {frac:.1%} of samples")

    print("\n2. IS THAT OVERSHOOT PREDICTABLE FROM ROUTE-GEOMETRY + EGO ALONE?")
    feats = ["v_now", "a_recent", "kappa_10", "kappa_20", "kappa_30",
             "lane_remaining_m", "n_lanes_ahead"]
    X = np.array([[r[k] for k in feats] for r in rows], dtype=np.float64)
    y = ov
    print("   univariate |Pearson r| vs overshoot:")
    for j, k in enumerate(feats):
        c = X[:, j]
        if c.std() < 1e-9:
            print(f"     {k:18s}   (constant)")
            continue
        r = float(np.corrcoef(c, y)[0, 1])
        print(f"     {k:18s} r = {r:+.3f}")
    # least-squares R^2 with intercept -- deliberately linear so it stays
    # interpretable and cannot flatter itself the way a deep model would
    Xa = np.hstack([X, np.ones((n, 1))])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    pred = Xa @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    print(f"\n   linear R^2 (all features)          : {r2:.3f}")
    print(f"   residual std                        : {math.sqrt(ss_res/n):.2f} m"
          f"   (overshoot std {y.std():.2f} m)")

    # the decisive slice: can it flag the cautious cases?
    cautious = dt_ < np.percentile(dt_, 10)
    print(f"\n3. THE CASE THAT MATTERS -- teacher in its most cautious decile "
          f"(d_teacher < {np.percentile(dt_,10):.1f} m, n={int(cautious.sum())})")
    thr = np.percentile(pred, 90)
    tp = int((cautious & (pred > thr)).sum())
    print(f"   flagged by a route+ego predictor at its own 90th pct: "
          f"{tp}/{int(cautious.sum())} ({tp/max(cautious.sum(),1):.1%})")
    print(f"   (random baseline would be ~10%)")

    if args.json:
        json.dump({"n": n, "overshoot_median": float(np.median(ov)),
                   "overshoot_gt5m_frac": frac, "linear_r2": r2,
                   "cautious_recall_at_p90": tp / max(float(cautious.sum()), 1),
                   "features": feats,
                   "beta": {k: float(b) for k, b in zip(feats + ["intercept"], beta)}},
                  open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
