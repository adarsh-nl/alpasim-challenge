#!/usr/bin/env python
"""Instrument ONE PPO update to find where the gradient dies.

Loads a real batch from a completed iteration, runs one update with full
diagnostics: advantage stats, per-parameter gradient norms, ratio distribution,
and whether the mean_head (the actor) receives ANY gradient.
"""
import sys, os, glob
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_policy import ResidualSpeedPolicy

batch_path = sys.argv[1]
z = np.load(batch_path)
batch = {k: z[k] for k in z.files}
dev = "cuda" if torch.cuda.is_available() else "cpu"

print(f"=== batch: {batch_path} ===")
print(f"  n_transitions={int(batch['n_transitions'])}  n_episodes={int(batch['n_episodes'])}")
print(f"  ret:        mean={batch['ret'].mean():.4f} std={batch['ret'].std():.4f} "
      f"min={batch['ret'].min():.4f} max={batch['ret'].max():.4f}")
print(f"  old_value:  mean={batch['old_value'].mean():.4f} std={batch['old_value'].std():.4f}")
print(f"  pre_action: mean={batch['pre_action'].mean():.4f} std={batch['pre_action'].std():.4f}")
print(f"  old_logprob:mean={batch['old_logprob'].mean():.4f} std={batch['old_logprob'].std():.4f}")

adv = batch['ret'] - batch['old_value']
print(f"\n  RAW advantage (ret - old_value): mean={adv.mean():.5f} std={adv.std():.5f}")
adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
print(f"  NORM advantage: mean={adv_n.mean():.5f} std={adv_n.std():.5f}  "
      f"(|mean|<1e-6 => mean-subtraction killed directional signal)")

# now run one update step with grad instrumentation
policy = ResidualSpeedPolicy().to(dev).train()
img = torch.from_numpy(batch["img"].astype(np.float32)/255.0).to(dev)
route = torch.from_numpy(batch["route"]).to(dev)
ego = torch.from_numpy(batch["ego"]).to(dev)
pre = torch.from_numpy(batch["pre_action"]).to(dev).unsqueeze(-1)
old_logp = torch.from_numpy(batch["old_logprob"]).to(dev)
ret = torch.from_numpy(batch["ret"]).to(dev)
old_val = torch.from_numpy(batch["old_value"]).to(dev)
advt = torch.from_numpy(adv_n.astype(np.float32)).to(dev)

logp, ent, val = policy.evaluate_actions(img, route, ego, pre)
ratio = torch.exp(logp - old_logp)
print(f"\n  logp(new): mean={logp.mean().item():.4f} std={logp.std().item():.4f}")
print(f"  ratio: mean={ratio.mean().item():.4f} std={ratio.std().item():.4f} "
      f"min={ratio.min().item():.4f} max={ratio.max().item():.4f}")

l_clip = -torch.min(ratio*advt, torch.clamp(ratio,0.8,1.2)*advt).mean()
l_value = F.mse_loss(val, ret)
l_ent = ent.mean()
loss = l_clip + 0.5*l_value - 0.01*l_ent
print(f"\n  l_clip={l_clip.item():.6f}  l_value={l_value.item():.4f}  l_ent={l_ent.item():.4f}")

policy.zero_grad()
loss.backward()
print("\n  === per-parameter gradient norms (is the ACTOR getting gradient?) ===")
for name, p in policy.named_parameters():
    g = p.grad.norm().item() if p.grad is not None else 0.0
    tag = "  <-- ACTOR mean_head" if "mean_head" in name else ("  <-- log_std" if "log_std" in name else "")
    if "mean_head" in name or "log_std" in name or "value_head" in name or g==0:
        print(f"    {name:32s} grad_norm={g:.6e}{tag}")

# the key diagnostic
mh_grad = sum(p.grad.norm().item() for n,p in policy.named_parameters() if "mean_head" in n and p.grad is not None)
print(f"\n  >>> ACTOR (mean_head) total grad norm: {mh_grad:.6e}")
print(f"  >>> if ~0: policy can't learn (advantage or ratio path is dead)")
print(f"  >>> if >0: gradient flows; freeze was advantage-magnitude, fixable with adv scaling")
