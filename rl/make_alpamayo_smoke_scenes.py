#!/usr/bin/env python3
"""Build the reproducible 20-scene list for the Alpamayo smoke test.

Composition, chosen so the run answers one question:
  2  the known zeros (dd162898..., 8cc8fe63...) -- both at-fault FRONT
     collisions from the 200-scene sweep; these are the whole point
 18  renderable tail scenes -- lowest-margin non-zero scenes we have evidence
     for, so "tail" means measured, not guessed

Every scene must be renderable (Alpamayo is a VLA and needs camera frames).

CPU-only, read-only. Usage:
    ~/alpasim/.venv/bin/python rl/make_alpamayo_smoke_scenes.py \\
        --out rl/alpamayo_smoke_scenes.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ZEROS = ("dd162898ecf351d4", "8cc8fe63227d5c07")
SWEEP = os.path.expanduser(
    "~/alpasim/runs/array-538078/aggregate/results-summary.json")
ROOT = os.environ.get("ALPASIM_NUPLAN_ROOT", "")
if not ROOT or not os.path.isdir(os.path.join(ROOT, "navtest", "assets")):
    ROOT = "/datasets/eemcs/ps/alpasim-nuplan-2026"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tail", type=int, default=18)
    args = ap.parse_args()

    renderable = set(os.listdir(os.path.join(ROOT, "navtest", "assets")))

    picked, why = [], {}

    # 1. the zeros
    if os.path.exists(SWEEP):
        d = json.load(open(SWEEP))
        rollouts = d["rollouts"]
        for r in rollouts:
            sid = r.get("clipgt_id", "")
            if any(z in sid for z in ZEROS):
                if sid in renderable:
                    picked.append(sid)
                    why[sid] = (f"KNOWN ZERO score={r['score']:.3f} "
                                f"reason={r.get('failure_reason')}")
                else:
                    print(f"  WARNING: zero {sid[-16:]} is NOT renderable; "
                          f"Alpamayo cannot be run on it", file=sys.stderr)

        # 2. "tail" cannot be selected by score: the 200-scene sweep is
        #    SATURATED (198/200 score exactly 1.000), so ranking by score just
        #    returns arbitrary perfect scenes. Rank instead by a MEASURED risk
        #    proxy that still varies on passing scenes:
        #      min_distance_to_obstacle_m  -- how close we came to a collision
        #    Both known zeros reached 0.00 m, so this is the axis the zeros live
        #    on and the one where a tail-specialist teacher should differ.
        def obs_m(r):
            v = (r.get("metrics") or {}).get("min_distance_to_obstacle_m")
            return 9.0 if v is None else float(v)

        cand = [(obs_m(r), r["score"], r["clipgt_id"]) for r in rollouts
                if r["clipgt_id"] in renderable
                and r["clipgt_id"] not in picked
                and r["score"] >= 0.01]
        cand.sort()                      # closest near-misses first
        for om, sc, sid in cand[:args.n_tail]:
            picked.append(sid)
            why[sid] = f"near-miss tail: minObs={om:.2f} m, score={sc:.3f}"
        if len(cand) < args.n_tail:
            print(f"  NOTE: only {len(cand)} renderable non-zero scenes in the "
                  f"sweep; asked for {args.n_tail}", file=sys.stderr)
    else:
        print(f"  {SWEEP} missing -- cannot select by measured score",
              file=sys.stderr)

    # 3. top up from renderable scenes that have teacher labels, if short
    if len(picked) < 2 + args.n_tail:
        labelled = set()
        for f in glob.glob(os.path.expanduser(
                "~/alpasim-challenge/rl/teacher_labels/*.npz")):
            import numpy as np
            labelled.add(str(np.load(f, allow_pickle=True)["token"]
                             ).rpartition("_")[0])
        for sid in sorted(labelled & renderable):
            if len(picked) >= 2 + args.n_tail:
                break
            if sid not in picked:
                picked.append(sid)
                why[sid] = "top-up: renderable + has DP labels (no sweep score)"

    with open(args.out, "w") as f:
        for sid in picked:
            f.write(sid + "\n")

    print(f"wrote {len(picked)} scenes -> {args.out}")
    for sid in picked:
        tag = "ZERO" if any(z in sid for z in ZEROS) else "tail"
        print(f"  [{tag}] {sid[-46:]}   {why.get(sid,'')}")
    return 0 if picked else 1


if __name__ == "__main__":
    sys.exit(main())
