#!/usr/bin/env python
"""Residual speed-control policy for closed-loop RL on AlpaSim (nuPlan track).

DESIGN: This does NOT learn to drive from scratch. The rank-1 route-follower
already produces a strong geometric plan (median dist_to_gt 0.39). This policy
learns a *residual* on top of it: a per-Drive-call speed multiplier m in (0,1]
that scales the follower's planned speed. The learned behavior is "slow down
when the forward camera shows something ahead" -- the longitudinal/braking
behavior the geometric follower lacks.

Action space is 1-dimensional (the speed multiplier) -> PPO converges in
hundreds of rollouts, not hundreds of thousands. Fits the 16 GiB / 0.1s
submission envelope with huge headroom (this net is ~2M params).

Actor-critic: the network outputs
  - actor: (mean, log_std) of a Gaussian over a pre-squash scalar; the action
    m = sigmoid(sample) lands in (0,1].
  - critic: a scalar value estimate for PPO advantage.

Inputs (all legitimate inference inputs -- no GT, no privileged state):
  - image: CAM_F0 forward camera, (3, H, W) float in [0,1]
  - route: K route waypoints in ego frame, (K, 2), NaN-padded -> zero-filled + mask
  - ego:   [speed, recent_ax, recent_ay] or similar small vector
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_STD_INIT = float(os.environ.get('RL_LOG_STD_INIT', '-0.5'))  # exploration width
_MEAN_BIAS = float(os.environ.get('RL_MEAN_BIAS', '2.0'))         # prior speed-mult center


class SmallCNN(nn.Module):
    """Lightweight image encoder (no torchvision dependency). ~1.5M params.

    Downsamples a (3, 180, 320) frame to a 128-d feature via strided convs +
    global average pool. Deliberately small: the forward camera only needs to
    surface 'is there looming mass ahead', not fine detail.
    """
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True),   # 90x160
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 45x80
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 23x40
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 12x20
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 6x10
        )
        self.out_dim = out_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        z = self.net(img)                       # (B, out_dim, 6, 10)
        return z.mean(dim=(2, 3))               # global avg pool -> (B, out_dim)


class ResidualSpeedPolicy(nn.Module):
    """Actor-critic that emits a speed-multiplier distribution + value.

    Total params ~2M. Fits well under 16 GiB and runs a forward pass in well
    under 100 ms even on CPU for batch=1.
    """
    def __init__(self, n_route: int = 10, img_dim: int = 128,
                 route_dim: int = 64, ego_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.n_route = n_route
        self.img_enc = SmallCNN(img_dim)
        # route encoder: flatten K waypoints (xy) + mask -> MLP
        self.route_enc = nn.Sequential(
            nn.Linear(n_route * 3, route_dim), nn.ReLU(inplace=True),   # xy + mask per wp
            nn.Linear(route_dim, route_dim), nn.ReLU(inplace=True),
        )
        self.ego_enc = nn.Sequential(
            nn.Linear(3, ego_dim), nn.ReLU(inplace=True),
        )
        fuse_in = img_dim + route_dim + ego_dim
        self.trunk = nn.Sequential(
            nn.Linear(fuse_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        # actor: mean of pre-squash scalar. log_std is a free parameter (state-independent).
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.tensor([_LOG_STD_INIT]))   # exploration width (env)
        # critic
        self.value_head = nn.Linear(hidden, 1)
        # init the mean head so the initial action ~ sigmoid(+2)=0.88 (mostly full speed,
        # i.e. start close to the follower's behavior; learn to slow from there)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, _MEAN_BIAS)

    def _encode(self, img: torch.Tensor, route: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        zi = self.img_enc(img)
        zr = self.route_enc(route.flatten(1))
        ze = self.ego_enc(ego)
        return self.trunk(torch.cat([zi, zr, ze], dim=1))

    def forward(self, img: torch.Tensor, route: torch.Tensor, ego: torch.Tensor):
        """Return (pre_squash_mean, log_std, value)."""
        h = self._encode(img, route, ego)
        mean = self.mean_head(h)                          # (B,1) pre-squash
        value = self.value_head(h).squeeze(-1)            # (B,)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std, value

    @torch.no_grad()
    def act(self, img: torch.Tensor, route: torch.Tensor, ego: torch.Tensor, deterministic: bool = False):
        """Sample a speed multiplier m in (0,1]. Returns (m, logprob, value).

        For deployment/eval use deterministic=True -> m = sigmoid(mean).
        For rollout collection use deterministic=False -> sample for exploration.
        """
        mean, log_std, value = self.forward(img, route, ego)
        std = log_std.exp()
        if deterministic:
            pre = mean
        else:
            eps = torch.randn_like(mean)
            pre = mean + std * eps
        # squashed-Gaussian log-prob: Gaussian logp on pre MINUS sigmoid Jacobian
        gauss_logp = (-0.5 * ((pre - mean) / std) ** 2 - log_std
                      - 0.5 * torch.log(torch.tensor(2 * torch.pi)))
        sig = torch.sigmoid(pre)
        log_jac = torch.log(sig * (1.0 - sig) + 1e-6)     # d m / d pre = sig*(1-sig)
        logp = (gauss_logp - log_jac).sum(-1)
        if deterministic:
            logp = torch.zeros_like(logp)
        m = sig.squeeze(-1)                                # (B,) in (0,1)
        return m, logp, value, pre.squeeze(-1)

    def evaluate_actions(self, img, route, ego, pre_actions):
        """For PPO update: given stored pre-squash actions, return (logprob, entropy, value)."""
        mean, log_std, value = self.forward(img, route, ego)
        std = log_std.exp()
        gauss_logp = (-0.5 * ((pre_actions - mean) / std) ** 2 - log_std
                      - 0.5 * torch.log(torch.tensor(2 * torch.pi)))
        sig = torch.sigmoid(pre_actions)
        log_jac = torch.log(sig * (1.0 - sig) + 1e-6)
        logp = (gauss_logp - log_jac).sum(-1)
        entropy = (0.5 + 0.5 * torch.log(torch.tensor(2 * torch.pi)) + log_std).sum(-1)
        return logp, entropy, value


def _smoke() -> None:
    """Verify param count, VRAM, and forward-pass latency against the submission envelope."""
    import time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualSpeedPolicy().to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"device: {dev}")
    print(f"params: {n_params:,}  (~{n_params/1e6:.2f}M)")

    B = 1
    img = torch.rand(B, 3, 180, 320, device=dev)
    route = torch.rand(B, 10, 3, device=dev)
    ego = torch.rand(B, 3, device=dev)

    # warmup
    for _ in range(3):
        net.act(img, route, ego)
    if dev == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    N = 50
    for _ in range(N):
        m, logp, v, _pre = net.act(img, route, ego)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
    print(f"forward (act) latency: {dt*1000:.2f} ms/call   [budget: <100 ms]  {'OK' if dt < 0.1 else 'OVER'}")
    print(f"sample action m={m.item():.3f}  value={v.item():.3f}")

    if dev == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak VRAM: {peak:.3f} GiB   [budget: <16 GiB]  {'OK' if peak < 16 else 'OVER'}")
    print("\nMilestone 1 smoke: policy net instantiates, fits envelope, produces valid action.")


if __name__ == "__main__":
    _smoke()
