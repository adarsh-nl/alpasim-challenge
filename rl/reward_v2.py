#!/usr/bin/env python
"""DriveIRT-aligned reward (v2): bounded [0,1], hard zero-spike for failure.

Mirrors the leaderboard's per-route score structure (from the DriveIRT docs +
the eval spec we confirmed):
  - CATASTROPHIC (collision_at_fault or offroad) -> hard 0.0  (the zero-spike)
  - otherwise -> bounded interior score in (0,1] combining:
      * progress_score: min(clamp(progress_rel,0,1)/0.8, 1.0)  [leaderboard gating]
      * fidelity:       1/(1 + dist_to_gt/d0)                   [tiebreak, keeps gradient]

Unlike reward v1 (unbounded additive, soft collision penalty), this:
  1. is bounded [0,1] like the actual score,
  2. makes failure a HARD zero (the sharpest possible "never cross this line"),
  3. keeps a fidelity gradient even on passing scenes (so tighter tracking always
     pays), which -- combined with discriminating-scene selection -- gives PPO
     the signal it lacked.

reward = 0                              if unsafe
       = progress * (base + (1-base)*fidelity)   otherwise, in [0,1]
with base=0.5: full-progress + perfect-fidelity -> 1.0;
               full-progress + poor-fidelity   -> 0.5;
               partial progress scales linearly (the discriminating middle).
"""
from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class RewardV2Weights:
    d0: float = 0.56          # fidelity scale (rival tracking level)
    base: float = 0.5         # floor of the interior score at full progress, poor fidelity
    progress_gate: float = 0.8

    @classmethod
    def from_env(cls):
        return cls(
            d0=float(os.environ.get("RL_FID_D0", "0.56")),
            base=float(os.environ.get("RL_BASE", "0.5")),
            progress_gate=float(os.environ.get("RL_PROG_GATE", "0.8")),
        )


def episode_reward_v2(metrics: dict, w: RewardV2Weights) -> tuple[float, dict]:
    coll = float(metrics.get("collision_at_fault", 0.0) or 0.0)
    offroad = float(metrics.get("offroad", 0.0) or 0.0)
    if coll > 0 or offroad > 0:
        return 0.0, {"reward": 0.0, "gate": "FAIL", "progress": 0.0, "fidelity": 0.0}

    prog_rel = float(metrics.get("progress_rel_to_total",
                     metrics.get("progress", 0.0)) or 0.0)
    prog = min(max(prog_rel, 0.0) / w.progress_gate, 1.0)     # leaderboard progress gating
    dist_gt = float(metrics.get("dist_to_gt_trajectory", 10.0) or 10.0)
    fid = 1.0 / (1.0 + dist_gt / w.d0)                        # (0,1]

    r = prog * (w.base + (1.0 - w.base) * fid)               # bounded [0,1]
    return r, {"reward": r, "gate": "pass", "progress": prog,
               "fidelity": fid, "dist_to_gt": dist_gt}


if __name__ == "__main__":
    w = RewardV2Weights()
    print("=== reward_v2 sanity (bounded [0,1], hard-gate) ===")
    for name, m in [
        ("clear, perfect track",   {"progress_rel_to_total":1.0,"dist_to_gt_trajectory":0.05,"collision_at_fault":0}),
        ("clear, rival track",     {"progress_rel_to_total":1.0,"dist_to_gt_trajectory":0.56,"collision_at_fault":0}),
        ("passes, LOOSE track",    {"progress_rel_to_total":1.0,"dist_to_gt_trajectory":3.0,"collision_at_fault":0}),
        ("partial progress",       {"progress_rel_to_total":0.5,"dist_to_gt_trajectory":1.0,"collision_at_fault":0}),
        ("COLLISION (hard zero)",  {"progress_rel_to_total":1.0,"dist_to_gt_trajectory":0.05,"collision_at_fault":1}),
    ]:
        r, b = episode_reward_v2(m, w)
        print(f"  {name:24s} reward={r:.3f}  [{b['gate']}] "
              f"prog={b.get('progress',0):.2f} fid={b.get('fidelity',0):.2f}")
