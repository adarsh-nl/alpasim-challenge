#!/usr/bin/env python
"""Milestone 3b: assemble a PPO training batch from a completed rollout run.

After a batch of rollouts finishes (policy in SAMPLING mode, RL_DETERMINISTIC=0),
each rollout produced:
  - a transition .npz  (img, route, ego, pre_action, logprob, value, t_us)  per step
  - an entry in the run's aggregate JSON  (metrics -> reward)

This joins them: for every rollout, compute the episode reward, assign it to that
rollout's transitions (with per-step return via reward-to-go + GAE-ready value),
and stack everything into ONE batch .npz that the PPO trainer consumes.

Key design choice for a residual speed policy with EPISODE-level reward:
we use the episode reward as the return for every step in that episode (each
step contributed to the outcome), and let PPO's advantage = return - value.
This is standard for short-horizon episodic control with a terminal reward.
GAE with per-step rewards is overkill here (10 steps, one outcome).
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os
from reward import load_rollout_metrics
if _os.environ.get("RL_REWARD", "v1") == "v2":
    from reward_v2 import episode_reward_v2 as episode_reward, RewardV2Weights as RewardWeights
else:
    from reward import episode_reward, RewardWeights



def _perstep_rewards(run_dir, uuid, t_us, w, safe):
    """Dense per-step reward from the rollout's per-timestep metrics.parquet.
    reward_t = 0 if episode unsafe else fidelity(dist_to_gt_at_t).
    Falls back to episode reward broadcast if the parquet/timeseries is missing.
    """
    import glob as _g
    import pandas as _pd
    import numpy as _np
    hits = _g.glob(f"{run_dir}/rollouts/*/{uuid}/metrics.parquet")
    if not hits or not safe:
        return None
    df = _pd.read_parquet(hits[0])
    sub = df[df["name"] == "dist_to_gt_trajectory"].sort_values("timestamps_us")
    if sub.empty:
        return None
    tt = sub["timestamps_us"].astype(float).values
    vv = sub["values"].astype(float).values
    d0 = getattr(w, "d0", 0.56)
    out = []
    for t in t_us:
        dgt = float(_np.interp(float(t), tt, vv))
        out.append(1.0 / (1.0 + dgt / d0))     # per-step fidelity in (0,1]
    return _np.array(out, dtype=_np.float32)


def build_batch(run_dir: str, w: RewardWeights) -> dict:
    metrics_by_uuid = load_rollout_metrics(run_dir)
    log_files = sorted(glob.glob(os.path.join(run_dir, "rl-logs", "*.npz")))
    if not log_files:
        raise RuntimeError(f"no transition logs in {run_dir}/rl-logs")

    imgs, routes, egos, pres, logps, vals, rets, ep_ids = [], [], [], [], [], [], [], []
    ep_rewards = []
    matched = 0
    for ei, lf in enumerate(log_files):
        uuid = os.path.basename(lf)[:-4]           # strip .npz
        z = np.load(lf)
        n = len(z["pre_action"])
        if n == 0:
            continue
        # find this rollout's metrics (uuid match; aggregate may key differently)
        m = metrics_by_uuid.get(uuid)
        if m is None:
            # fall back: if only one rollout, take the only metrics
            if len(metrics_by_uuid) == 1:
                m = next(iter(metrics_by_uuid.values()))
            else:
                # try matching by any key containing the uuid prefix
                for k, v in metrics_by_uuid.items():
                    if uuid[:8] in str(k):
                        m = v; break
        if m is None:
            print(f"  WARN no metrics for {uuid[:12]}, skipping")
            continue
        r, rb = episode_reward(m, w)
        matched += 1
        ep_rewards.append(r)
        safe = (rb.get("gate", "pass") != "FAIL") if isinstance(rb, dict) else True
        imgs.append(z["img"]); routes.append(z["route"]); egos.append(z["ego"])
        pres.append(z["pre_action"]); logps.append(z["logprob"]); vals.append(z["value"])
        ps = _perstep_rewards(run_dir, uuid, z["t_us"], w, safe) if _os.environ.get("RL_PERSTEP","0")=="1" else None
        if ps is not None and len(ps)==n:
            rets.append(ps)                              # dense per-step reward
        else:
            rets.append(np.full(n, r, dtype=np.float32)) # fallback: episode reward
        ep_ids.append(np.full(n, ei, dtype=np.int64))

    if matched == 0:
        raise RuntimeError("no rollouts matched to metrics")

    batch = {
        "img": np.concatenate(imgs).astype(np.uint8),
        "route": np.concatenate(routes).astype(np.float32),
        "ego": np.concatenate(egos).astype(np.float32),
        "pre_action": np.concatenate(pres).astype(np.float32),
        "old_logprob": np.concatenate(logps).astype(np.float32),
        "old_value": np.concatenate(vals).astype(np.float32),
        "ret": np.concatenate(rets).astype(np.float32),
        "ep_id": np.concatenate(ep_ids).astype(np.int64),
    }
    # normalize advantage baseline info; PPO computes adv = ret - value
    batch["ep_reward_mean"] = np.float32(np.mean(ep_rewards))
    batch["ep_reward_std"] = np.float32(np.std(ep_rewards))
    batch["n_episodes"] = np.int64(matched)
    batch["n_transitions"] = np.int64(len(batch["ret"]))
    return batch, ep_rewards


if __name__ == "__main__":
    run = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, "batch.npz")
    w = RewardWeights.from_env()
    batch, ep_rewards = build_batch(run, w)
    np.savez_compressed(out, **batch)
    print(f"=== batch built: {run} ===")
    print(f"  episodes matched:  {int(batch['n_episodes'])}")
    print(f"  transitions:       {int(batch['n_transitions'])}")
    print(f"  ep reward: mean={float(batch['ep_reward_mean']):+.3f} "
          f"std={float(batch['ep_reward_std']):.3f} "
          f"min={min(ep_rewards):+.3f} max={max(ep_rewards):+.3f}")
    print(f"  saved -> {out}")
