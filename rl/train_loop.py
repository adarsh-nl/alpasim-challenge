#!/usr/bin/env python
"""Milestone 5: the closed-loop PPO training orchestrator.

Repeats:  collect rollouts (policy sampling) -> build batch -> PPO update -> repeat.
Periodically evaluates (deterministic) vs the follower baseline and logs a curve.

This is the CONTROL PLANE. It runs on a GPU node and, each iteration:
  1. writes the current policy checkpoint
  2. launches a batch of rollouts via the sim (bare-process services), policy in
     SAMPLING mode reading that checkpoint, logging transitions
  3. joins transitions+rewards into a batch
  4. runs a PPO update -> new checkpoint
  5. logs iteration stats (mean reward, dist_to_gt, collision rate, entropy)

All sizes are config knobs. Defaults = SMALL/FAST (does it learn at all?).

IMPORTANT: rollout execution is delegated to a helper that runs the 3 sim
services + policy driver for a given scene list and checkpoint. That helper is
`run_rollouts()` below; it reuses the exact bare-process pattern validated in
the M2 smoke. On a multi-GPU allocation it shards scenes across GPUs.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, glob
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_policy import ResidualSpeedPolicy
from reward import RewardWeights, episode_reward, load_rollout_metrics
from collect_batch import build_batch
from ppo_update import ppo_update

N_ROUTE = 10


def run_rollouts(scenes, ckpt, run_dir, *, deterministic=False, gpu=0,
                 repo, chal, nuplan_root, hf_home):
    """Launch sim services + policy driver, run the given scenes, return run_dir.

    Reuses the validated bare-process pattern: wizard(run_method=NONE) to gen
    configs, then renderer/controller/policy-driver as background procs, then
    the runtime simulate over the scene list. Blocks until done.
    """
    os.makedirs(f"{run_dir}/txt-logs", exist_ok=True)
    os.makedirs(f"{run_dir}/rl-logs", exist_ok=True)
    scene_arg = "scenes.scene_ids=[" + ",".join(scenes) + "]"
    env = dict(os.environ)
    env.update({
        "ALPASIM_NUPLAN_ROOT": nuplan_root, "HF_HOME": hf_home,
        "LC_ALL": "C.UTF-8", "CUDA_VISIBLE_DEVICES": str(gpu),
        "RL_LOG_DIR": f"{run_dir}/rl-logs",
        "RL_DETERMINISTIC": "1" if deterministic else "0",
        "RL_POLICY_CKPT": ckpt,
        "RF_V_MAX": "20", "RF_A_LON": "6.0", "RF_A_LAT": "7.0",
    })
    # 1. generate configs (no container launch)
    subprocess.run(
        ["uv","run","alpasim_wizard","+e2e_challenge_nuplan=dev",
         "wizard.run_method=NONE", f"wizard.log_dir={run_dir}",
         "scenes.limit_to_first_n=0", scene_arg, "eval.video.render_video=false"],
        cwd=repo, env=env, check=True,
        stdout=open(f"{run_dir}/txt-logs/wizard.log","w"), stderr=subprocess.STDOUT)

    procs = []
    def bg(cmd, log):
        return subprocess.Popen(cmd, cwd=repo, env=env,
                                stdout=open(f"{run_dir}/txt-logs/{log}","w"),
                                stderr=subprocess.STDOUT)
    try:
        procs.append(bg(["uv","run","python",f"{chal}/rl/policy_driver.py"], "driver.log"))
        procs.append(bg(["uv","run","python","-m","alpasim_controller.server",
                         "--port=6001", f"--log_dir={run_dir}/controller",
                         "--log-level=INFO", f"--config={run_dir}/controller-config.yaml"],
                        "controller.log"))
        procs.append(bg(["uv","run","alpasim-mtgs-server",
                         f"--user-config={run_dir}/generated-user-config-0.yaml",
                         "--host=0.0.0.0","--port=6000","--cache-size=5","--log-level=INFO"],
                        "renderer.log"))
        # wait for ports
        if not _wait_ports([6789,6001,6000], timeout=900):
            raise RuntimeError("services did not come up")
        # 2. run the rollouts
        subprocess.run(
            ["uv","run","python","-m","alpasim_runtime.simulate",
             f"--user-config={run_dir}/generated-user-config-0.yaml",
             f"--network-config={run_dir}/generated-network-config.yaml",
             f"--log-dir={run_dir}","--log-level=INFO",
             f"--array-job-dir={run_dir}",f"--eval-config={run_dir}/eval-config.yaml"],
            cwd=repo, env=env, check=True,
            stdout=open(f"{run_dir}/txt-logs/simulate.log","w"), stderr=subprocess.STDOUT)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try: p.wait(timeout=10)
            except Exception: p.kill()
    return run_dir


def _wait_ports(ports, timeout=900):
    import socket
    t0 = time.time()
    for port in ports:
        while time.time()-t0 < timeout:
            with socket.socket() as s:
                s.settimeout(1)
                if s.connect_ex(("localhost", port)) == 0:
                    break
            time.sleep(5)
        else:
            return False
    return True


def evaluate(policy, scenes, ckpt_path, work, cfg):
    """Deterministic eval over scenes -> mean dist_to_gt, collision rate, mean reward."""
    torch.save(policy.state_dict(), ckpt_path)
    rd = f"{work}/eval-{int(time.time())}"
    run_rollouts(scenes, ckpt_path, rd, deterministic=True, **cfg["sim"])
    mets = load_rollout_metrics(rd)
    w = cfg["weights"]
    ds, colls, rs = [], [], []
    for m in mets.values():
        r,_ = episode_reward(m, w)
        ds.append(float(m.get("dist_to_gt_trajectory",10) or 10))
        colls.append(1.0 if float(m.get("collision_at_fault",0) or 0)>0 else 0.0)
        rs.append(r)
    return {"dist_to_gt": float(np.mean(ds)), "collision_rate": float(np.mean(colls)),
            "reward": float(np.mean(rs)), "n": len(ds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--episodes_per_iter", type=int, default=16)
    ap.add_argument("--scene_file", required=True, help="txt of scene ids, one per line")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--eval_scene_file", default="")
    ap.add_argument("--work", default="/home/nanjaiyalathaa/alpasim/runs/rl-train")
    ap.add_argument("--repo", default="/home/nanjaiyalathaa/alpasim")
    ap.add_argument("--chal", default="/home/nanjaiyalathaa/alpasim-challenge")
    ap.add_argument("--nuplan_root", default="/datasets/eemcs/ps/alpasim-nuplan-2026")
    ap.add_argument("--hf_home", default="/home/nanjaiyalathaa/data/hf-cache")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.work, exist_ok=True)
    scenes_all = [l.strip() for l in open(args.scene_file) if l.strip()]
    eval_scenes = ([l.strip() for l in open(args.eval_scene_file) if l.strip()]
                   if args.eval_scene_file else scenes_all[:8])
    cfg = {"sim": dict(repo=args.repo, chal=args.chal, nuplan_root=args.nuplan_root,
                       hf_home=args.hf_home, gpu=0),
           "weights": RewardWeights.from_env()}

    policy = ResidualSpeedPolicy(n_route=N_ROUTE).to(dev)
    if args.resume and os.path.exists(args.resume):
        policy.load_state_dict(torch.load(args.resume, map_location=dev))
        print(f"resumed from {args.resume}")

    ckpt = f"{args.work}/policy.pt"
    curve = []
    for it in range(args.iters):
        t0 = time.time()
        torch.save(policy.state_dict(), ckpt)
        # sample scenes for this iteration's batch (with replacement if needed)
        pick = list(np.random.choice(scenes_all,
                    size=min(args.episodes_per_iter, len(scenes_all)), replace=False))
        rd = f"{args.work}/iter{it:03d}"
        run_rollouts(pick, ckpt, rd, deterministic=False, **cfg["sim"])
        batch, ep_rewards = build_batch(rd, cfg["weights"])
        stats = ppo_update(policy, batch, lr=args.lr, device=dev)
        torch.save(policy.state_dict(), ckpt)

        row = {"iter": it, "ep_reward_mean": float(np.mean(ep_rewards)),
               "ep_reward_std": float(np.std(ep_rewards)),
               "n_ep": int(batch["n_episodes"]), **stats,
               "sec": round(time.time()-t0,1)}
        # periodic eval
        if args.eval_every and (it % args.eval_every == 0 or it == args.iters-1):
            ev = evaluate(policy, eval_scenes, f"{args.work}/eval.pt", args.work, cfg)
            row["eval"] = ev
        curve.append(row)
        json.dump(curve, open(f"{args.work}/curve.json","w"), indent=2)
        ev_s = (f"  EVAL dist_to_gt={row['eval']['dist_to_gt']:.3f} "
                f"coll={row['eval']['collision_rate']:.2f} rew={row['eval']['reward']:+.2f}"
                if "eval" in row else "")
        print(f"[iter {it:3d}] ep_reward={row['ep_reward_mean']:+.3f}±{row['ep_reward_std']:.2f} "
              f"kl={stats['kl']:+.3f} ent={stats['entropy']:.2f} "
              f"clipfrac={stats['clipfrac']:.2f} ({row['sec']}s){ev_s}", flush=True)

    print(f"\ndone. curve -> {args.work}/curve.json  final ckpt -> {ckpt}")


if __name__ == "__main__":
    main()
