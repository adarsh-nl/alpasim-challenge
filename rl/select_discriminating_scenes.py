#!/usr/bin/env python
"""Select DISCRIMINATING scenes for RL training (the flat-landscape fix).

DriveIRT trims uninformative routes -- ones everyone passes or everyone fails.
For a single policy we proxy 'informative' as scenes with ROOM to improve:
the follower's score is NOT saturated at 1.0 and NOT a hard 0. On those scenes
a change in the speed multiplier actually MOVES the score, giving PPO gradient.

Selection from the 200-scene metrics (array-538078):
  - EXCLUDE scenes the follower already aces (dist_to_gt < d_tight AND full progress)
    -> saturated, no gradient (these dominated the old flat batches)
  - EXCLUDE hard failures we can't fix from this observation space
    (collision -- proven information-limited)  [optional, via --keep_fail]
  - KEEP the interior: intermediate dist_to_gt / partial progress
    -> the routes where speed modulation has leverage

Writes full scene ids to --out, one per line.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import pandas as pd


def collapse(df, name):
    sub = df[df["name"] == name]
    if sub.empty:
        return np.nan
    agg = sub["time_aggregation"].iloc[0]
    v = sub["values"].astype(float)
    if agg == "last":
        return sub.sort_values("timestamps_us")["values"].astype(float).iloc[-1]
    return {"max": v.max, "min": v.min, "mean": v.mean}.get(agg, lambda: v.iloc[-1])()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/home/nanjaiyalathaa/alpasim/runs/array-538078")
    ap.add_argument("--out", default="train_scenes_discriminating.txt")
    ap.add_argument("--d_tight", type=float, default=0.4,
                    help="dist_to_gt below this = already tight (saturated)")
    ap.add_argument("--d_loose", type=float, default=6.0,
                    help="dist_to_gt above this = hopeless/route-absent, skip")
    ap.add_argument("--keep_fail", action="store_true",
                    help="also include at-fault-collision scenes (default: exclude, info-limited)")
    args = ap.parse_args()

    pqs = glob.glob(f"{args.run}/task-*/rollouts/*/*/metrics.parquet")
    rows = []
    for p in pqs:
        try:
            df = pd.read_parquet(p)
            sid = df["clipgt_id"].iloc[0]
            rows.append({
                "scene": sid,
                "dgt": collapse(df, "dist_to_gt_trajectory"),
                "prog": collapse(df, "progress_rel_to_total"),
                "coll": collapse(df, "collision_front"),
                "offroad": collapse(df, "offroad"),
            })
        except Exception:
            pass
    R = pd.DataFrame(rows)
    print(f"loaded {len(R)} scenes", file=sys.stderr)

    # discriminating = interior room to improve
    sel = R[
        (R["dgt"] >= args.d_tight) &          # not already saturated-tight
        (R["dgt"] <= args.d_loose)            # not hopeless
    ].copy()
    if not args.keep_fail:
        sel = sel[(sel["coll"].fillna(0) == 0) & (sel["offroad"].fillna(0) == 0)]

    sel = sel.sort_values("dgt")
    with open(args.out, "w") as f:
        for s in sel["scene"]:
            f.write(s + "\n")
    print(f"selected {len(sel)} discriminating scenes -> {args.out}", file=sys.stderr)
    print(f"  dist_to_gt range: {sel['dgt'].min():.2f} - {sel['dgt'].max():.2f}", file=sys.stderr)
    print(f"  (excluded {len(R)-len(sel)}: saturated-tight, hopeless, or failures)", file=sys.stderr)


if __name__ == "__main__":
    main()
