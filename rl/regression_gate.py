#!/usr/bin/env python
"""Regression gate: is a CANDIDATE container actually better than the BASELINE?

Inspired by Josh Laubach's e2e-regression-gate. Answers the question that
protects scarce submissions: does the candidate beat the baseline with
statistical confidence, or does it just look better on a noisy average while
regressing the tail?

Method (all on the SAME held-out scenes, paired):
  1. Per-scene score in [0,1], DriveIRT-aligned (reward_v2): catastrophic->0,
     else progress-gated * fidelity.
  2. RELIABILITY vs QUALITY decomposition (the key idea from the harness):
       - 'valid' = produced a usable drive (no at-fault collision, no offroad,
         made real progress). A route-absent fall-through or a crash is INVALID.
       - Split the score gap into: (a) change in valid-rate (reliability), and
         (b) change in quality AMONG scenes both models drove validly.
     A gap that's all reliability points to a different fix than a driving-quality gap.
  3. Paired comparison: per-scene score difference, bootstrap 95% CI on the mean
     difference, paired permutation p-value. No scipy (numpy-only, robust).
  4. Verdict: SHIP / HOLD / INCONCLUSIVE against a tolerance (default 0.02),
     with INCONCLUSIVE reported honestly when the CI can't resolve the tolerance.

Usage:
  python regression_gate.py --baseline <run_dir> --candidate <run_dir> [--tol 0.02]
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_v2 import episode_reward_v2, RewardV2Weights


def _scene_scores(run_dir, w):
    """scene_id -> (score, valid, metrics) from a run's aggregate JSONs."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "aggregate", "*.json")):
        d = json.load(open(f))
        for r in d.get("rollouts", []):
            m = r.get("metrics", {})
            sid = r.get("clipgt_id", "").split("-")[-1]
            score, b = episode_reward_v2(m, w)
            coll = float(m.get("collision_at_fault", 0) or 0)
            off = float(m.get("offroad", 0) or 0)
            prog = float(m.get("progress_rel_to_total", m.get("progress", 0)) or 0)
            valid = (coll == 0 and off == 0 and prog > 0.3)   # drove the route without failing
            out[sid] = (score, valid, m)
    return out


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.array([rng.choice(diffs, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return float(lo), float(hi)


def perm_pvalue(diffs, n_perm=10000, seed=0):
    """Paired permutation test: H0 = no difference (sign-flip symmetry)."""
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], n)
        if abs((diffs * signs).mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--tol", type=float, default=0.02, help="ship tolerance (score units)")
    args = ap.parse_args()
    w = RewardV2Weights.from_env()

    base = _scene_scores(args.baseline, w)
    cand = _scene_scores(args.candidate, w)
    shared = sorted(set(base) & set(cand))
    if not shared:
        print("NO SHARED SCENES between the two runs"); return
    print(f"=== REGRESSION GATE ===")
    print(f"  baseline:  {args.baseline}")
    print(f"  candidate: {args.candidate}")
    print(f"  shared scenes: {len(shared)}  (baseline {len(base)}, candidate {len(cand)})\n")

    bs = np.array([base[s][0] for s in shared])
    cs = np.array([cand[s][0] for s in shared])
    diffs = cs - bs                                   # candidate - baseline

    print(f"  mean score  baseline={bs.mean():.4f}  candidate={cs.mean():.4f}  "
          f"diff={diffs.mean():+.4f}")

    # RELIABILITY vs QUALITY decomposition
    bv = np.array([base[s][1] for s in shared])
    cv = np.array([cand[s][1] for s in shared])
    print(f"\n  --- reliability (valid-drive rate) ---")
    print(f"    baseline valid: {bv.sum()}/{len(shared)} ({bv.mean():.1%})   "
          f"candidate valid: {cv.sum()}/{len(shared)} ({cv.mean():.1%})")
    both_valid = bv & cv
    if both_valid.sum() > 0:
        q = (cs[both_valid] - bs[both_valid])
        print(f"\n  --- quality (score gap among {both_valid.sum()} scenes BOTH drove validly) ---")
        print(f"    quality diff = {q.mean():+.4f}   "
              f"(if ~0, the gap is RELIABILITY not driving quality)")

    # paired stats on the full shared set
    lo, hi = bootstrap_ci(diffs)
    p = perm_pvalue(diffs)
    print(f"\n  --- paired comparison (n={len(shared)}) ---")
    print(f"    mean diff = {diffs.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p = {p:.4f}")

    # verdict
    print(f"\n  === VERDICT (tolerance ±{args.tol}) ===")
    if lo > args.tol:
        print(f"    SHIP  candidate is better by a resolved margin (CI above +{args.tol})")
    elif hi < -args.tol:
        print(f"    HOLD  candidate REGRESSES by a resolved margin (CI below -{args.tol})")
    elif lo > 0 and p < 0.05:
        print(f"    SHIP (marginal)  significant improvement but within tolerance band")
    elif hi < 0 and p < 0.05:
        print(f"    HOLD (marginal)  significant regression but within tolerance band")
    else:
        print(f"    INCONCLUSIVE  CI [{lo:+.4f},{hi:+.4f}] spans the tolerance band;")
        print(f"                  not enough evidence to resolve a {args.tol}-size effect.")
        print(f"                  (honest headline: need more scenes, not a false 'cleared')")

    # worst regressions (where candidate lost most) — the tail the average hides
    order = np.argsort(diffs)
    print(f"\n  --- worst 5 regressions (candidate lost most) ---")
    for i in order[:5]:
        s = shared[i]
        print(f"    {s}: base={bs[i]:.3f} cand={cs[i]:.3f} diff={diffs[i]:+.3f} "
              f"(base_valid={base[s][1]} cand_valid={cand[s][1]})")


if __name__ == "__main__":
    main()
