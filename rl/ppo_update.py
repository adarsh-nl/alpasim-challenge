#!/usr/bin/env python
"""Milestone 4: PPO clipped-surrogate update for the residual speed policy.

Takes a batch .npz (from collect_batch.py) and the current policy, runs K epochs
of minibatch PPO, and writes the updated policy checkpoint.

Standard PPO:
  advantage = normalize(return - value_old)
  ratio     = exp(logprob_new - logprob_old)
  L_clip    = -min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv)
  L_value   = (value_new - return)^2
  L_entropy = -entropy            (encourage exploration)
  loss = L_clip + c_v*L_value - c_e*L_entropy

Short episodes + 1-D action -> we use episode-return as the target (no GAE);
advantage = return - value_baseline. This is appropriate for terminal-reward
episodic control and keeps the update simple and stable.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_policy import ResidualSpeedPolicy

N_ROUTE = 10


def ppo_update(policy: ResidualSpeedPolicy, batch: dict, *,
               epochs=4, minibatch=256, clip=0.2, c_value=0.5, c_entropy=0.01,
               lr=3e-4, device="cuda", max_grad_norm=0.5) -> dict:
    dev = device
    policy = policy.to(dev).train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    img = torch.from_numpy(batch["img"].astype(np.float32) / 255.0).to(dev)   # N,3,H,W
    route = torch.from_numpy(batch["route"]).to(dev)
    ego = torch.from_numpy(batch["ego"]).to(dev)
    pre = torch.from_numpy(batch["pre_action"]).to(dev).unsqueeze(-1)         # N,1
    old_logp = torch.from_numpy(batch["old_logprob"]).to(dev)
    ret = torch.from_numpy(batch["ret"]).to(dev)
    old_val = torch.from_numpy(batch["old_value"]).to(dev)

    # advantage = return - baseline(value), normalized
    adv = ret - old_val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = img.shape[0]
    idx = np.arange(N)
    stats = {"pg_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": []}

    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, N, minibatch):
            b = idx[s:s + minibatch]
            bt = torch.as_tensor(b, device=dev)
            logp, ent, val = policy.evaluate_actions(
                img[bt], route[bt], ego[bt], pre[bt])
            ratio = torch.exp(logp - old_logp[bt])
            a = adv[bt]
            l_clip = -torch.min(ratio * a,
                                torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            l_value = F.mse_loss(val, ret[bt])
            l_ent = ent.mean()
            loss = l_clip + c_value * l_value - c_entropy * l_ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()

            with torch.no_grad():
                kl = (old_logp[bt] - logp).mean().item()
                clipfrac = ((ratio - 1.0).abs() > clip).float().mean().item()
            stats["pg_loss"].append(l_clip.item())
            stats["v_loss"].append(l_value.item())
            stats["entropy"].append(l_ent.item())
            stats["kl"].append(kl)
            stats["clipfrac"].append(clipfrac)

    return {k: float(np.mean(v)) for k, v in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--in_ckpt", default="")
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--c_entropy", type=float, default=0.01)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ResidualSpeedPolicy(n_route=N_ROUTE)
    if args.in_ckpt and os.path.exists(args.in_ckpt):
        policy.load_state_dict(torch.load(args.in_ckpt, map_location=dev))
        print(f"loaded in_ckpt {args.in_ckpt}")
    else:
        print("starting from random-init policy")

    z = np.load(args.batch)
    batch = {k: z[k] for k in z.files}
    print(f"batch: {int(batch['n_transitions'])} transitions, "
          f"{int(batch['n_episodes'])} episodes, "
          f"ep_reward mean={float(batch['ep_reward_mean']):+.3f}")

    stats = ppo_update(policy, batch, epochs=args.epochs, clip=args.clip,
                       c_entropy=args.c_entropy, lr=args.lr, device=dev)
    torch.save(policy.state_dict(), args.out_ckpt)
    print("=== PPO update ===")
    for k, v in stats.items():
        print(f"  {k}: {v:+.4f}")
    print(f"  saved -> {args.out_ckpt}")


if __name__ == "__main__":
    main()
