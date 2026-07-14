# AlpaSim E2E Closed-Loop Challenge 2026 — nuPlan track

Target: 1st place + Innovative Solution, nuPlan track.
Leaderboard closes 2026-10-31. Rules freeze 2026-09-15.

## Cluster: UT EEMCS HPC

Partitions: `ps` (priority: hpc-node11/12 = 4x L40S 48GB, hpc-node14 = 3x L40),
plus `main-gpu` (shared). No H100/H200/Blackwell. No node has >4 GPUs.
Interconnect is 10GbE -> single-node training only.

### The Docker problem, and the workaround
No Docker, no enroot/pyxis on the cluster, so both the wizard's
DOCKER_COMPOSE and SLURM deployment backends are unusable.

Fix: `wizard.run_method=NONE` generates all configs and exits. Then run
renderer / controller / runtime / driver as bare `uv run` processes inside a
single sbatch job on one node, talking over localhost. See `jobs/smoke2.sbatch`.

Always pin `--gres=gpu:lovelace:1` -- gsplat kernels are built for sm_89, and
an A100 (sm_80) will fail.

Docker is still needed to build the final submission image; do that on a
laptop, not the cluster.

## Scoring -- confirmed from results-summary.json, not inferred

    score_criteria:
      collision_at_fault: "== 0"        # front OR lateral. REAR IS NOT AT FAULT.
      offroad:            "== 0"
      progress_score:     "min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1.0)"

Gated, not weighted. Fail either gate -> score 0 for the scene.

Implications:
- No reward for progress beyond 80% of GT distance.
- Being rear-ended costs nothing -> braking is cheap; conservatism is cheap.
- Leaderboard aggregates via zero-inflated IRT (DriveIRT), so zeros are
  disproportionately costly and easy scenes are trimmed. Optimize the failure
  tail, not the mean.

## Eval spec (from configs/e2e_challenge_nuplan_common/base.yaml)

8 cameras (F0, L0-L2, R0-R2, B0) @ 1920x1080, 2 Hz.
control_timestep = 500ms. n_sim_steps = 200 -> 100s rollouts.
Official (`ec2`): navtest_full, 8gpu_32rollouts_mtgs,
driver n_concurrent_rollouts=2 -> per-rollout state isolation is mandatory.

## Baseline

Starter driver (straight line @ 5 m/s), navtest_dev, 1 scene:
    score 0.427, collision_at_fault 0, offroad 0, progress_rel 0.64
    driver_drive_rpc_duration_mean_s = 0.0028  (budget 0.1) -> 35x headroom

Latency is NOT the bottleneck. Deprioritise FP8/TensorRT/dual-rate work.

## Environment

    export ALPASIM_NUPLAN_ROOT=$HOME/data/alpasim-nuplan
    export ALPASIM_NUPLAN_HF=$HOME/data/alpasim-nuplan-hf
    export HF_HOME=$HOME/data/hf-cache
    export SINGULARITY_CACHEDIR=/local/$USER/singularity
    export LC_ALL=C.UTF-8

Data: HF `OpenDriveLab/AlpasimChallenge2026_nuplan_track` (459 GiB, 15 shards).
Only `part001` + configs + trajdata cache downloaded so far (smoke test).
