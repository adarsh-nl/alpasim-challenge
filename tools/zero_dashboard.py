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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--near", type=float, default=0.8,
                    help="near-zero threshold (default 0.8)")
    ap.add_argument("--md", default="", help="also write a markdown report here")
    args = ap.parse_args()

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
