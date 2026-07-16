#!/usr/bin/env python
"""Milestone 3a: balanced episode reward from an AlpaSim rollout's metrics.

Reward = w_progress * progress
       + w_fidelity * fidelity_closeness          (dense: how tightly we track GT)
       - w_safety   * (collision_at_fault or offroad)   (sparse guardrail)

All weights are config knobs (env or dict) so we retune empirically after the
first runs. Starting point (reasoned from our metrics):
    w_progress = 1.0   dense primary driver
    w_fidelity = 0.5   dense tiebreak refinement, secondary to progress
    w_safety   = 5.0   large guardrail: one zero > any polish

fidelity_closeness maps dist_to_gt (0=perfect, grows unbounded) to (0,1]:
    closeness = 1 / (1 + dist_to_gt / d0),  d0 = 0.56 (the rival's tracking level)
so tracking at rival level gives 0.5, perfect gives 1.0, far gives ->0.

The reward is the ENVIRONMENT's feedback (sim score), not GT fed to the policy.
Legit: the policy never sees GT; the trainer scores the rollout with it.
"""
from __future__ import annotations
import json, glob, os
from dataclasses import dataclass


@dataclass
class RewardWeights:
    w_progress: float = 1.0
    w_fidelity: float = 0.5
    w_safety: float = 5.0
    d0: float = 0.56           # fidelity scale (rival dist_to_gt)

    @classmethod
    def from_env(cls):
        return cls(
            w_progress=float(os.environ.get("RL_W_PROGRESS", "1.0")),
            w_fidelity=float(os.environ.get("RL_W_FIDELITY", "0.5")),
            w_safety=float(os.environ.get("RL_W_SAFETY", "5.0")),
            d0=float(os.environ.get("RL_FID_D0", "0.56")),
        )


def episode_reward(metrics: dict, w: RewardWeights) -> tuple[float, dict]:
    """Compute scalar reward + a breakdown dict from one rollout's metrics."""
    progress = float(metrics.get("progress", 0.0) or 0.0)
    dist_gt = float(metrics.get("dist_to_gt_trajectory", 10.0) or 10.0)
    coll = float(metrics.get("collision_at_fault", 0.0) or 0.0)
    offroad = float(metrics.get("offroad", 0.0) or 0.0)

    fidelity = 1.0 / (1.0 + dist_gt / w.d0)                 # (0,1], 1=perfect
    unsafe = 1.0 if (coll > 0 or offroad > 0) else 0.0

    r = (w.w_progress * progress
         + w.w_fidelity * fidelity
         - w.w_safety * unsafe)
    breakdown = {
        "reward": r, "progress": progress, "dist_to_gt": dist_gt,
        "fidelity": fidelity, "unsafe": unsafe,
        "r_progress": w.w_progress * progress,
        "r_fidelity": w.w_fidelity * fidelity,
        "r_safety": -w.w_safety * unsafe,
    }
    return r, breakdown


def load_rollout_metrics(run_dir: str) -> dict[str, dict]:
    """Map rollout_uuid -> metrics dict from a run's aggregate JSON."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, "aggregate", "*.json")):
        d = json.load(open(f))
        for r in d.get("rollouts", []):
            # rollout uuid is the session id; match the transition-log filename
            uuid = r.get("rollout_id") or r.get("session_uuid") or r.get("clipgt_id")
            out[uuid] = r.get("metrics", {})
    return out


if __name__ == "__main__":
    import sys
    w = RewardWeights.from_env()
    run = sys.argv[1] if len(sys.argv) > 1 else "."
    mets = load_rollout_metrics(run)
    print(f"weights: progress={w.w_progress} fidelity={w.w_fidelity} safety={w.w_safety} d0={w.d0}")
    for uuid, m in mets.items():
        r, b = episode_reward(m, w)
        print(f"  {str(uuid)[:20]}: reward={r:+.3f}  "
              f"[prog={b['r_progress']:+.2f} fid={b['r_fidelity']:+.2f} safe={b['r_safety']:+.2f}]  "
              f"dist_to_gt={b['dist_to_gt']:.2f}")
