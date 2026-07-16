#!/bin/bash
#SBATCH --job-name=rl-m2-smoke
#SBATCH --partition=ps
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nanjaiyalathaa/alpasim/logs/%x-%j.out
set -euo pipefail
hostname; nvidia-smi --query-gpu=name --format=csv,noheader

export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export ALPASIM_NUPLAN_ROOT=/datasets/eemcs/ps/alpasim-nuplan-2026
export HF_HOME=/home/nanjaiyalathaa/data/hf-cache
export LC_ALL=C.UTF-8
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST="8.9"
module load nvidia/cuda-12.4

REPO=/home/nanjaiyalathaa/alpasim
CHAL=/home/nanjaiyalathaa/alpasim-challenge
cd $REPO

RUN=$REPO/runs/rl-m2-$SLURM_JOB_ID
mkdir -p $RUN/txt-logs $RUN/rl-logs

# one clear scene, deterministic policy (eval mode) so behavior is reproducible
SCENES='scenes.scene_ids=[2021.05.25.14.16.10_veh-35_01100_01664-f02b15cc225b5d9a]'

uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.run_method=NONE wizard.log_dir=$RUN \
  scenes.limit_to_first_n=0 "$SCENES" eval.video.render_video=false

PIDS=()
cleanup(){ for p in "${PIDS[@]:-}"; do kill $p 2>/dev/null || true; done; }
trap cleanup EXIT

# *** policy driver runs from MAIN .venv (has torch), NOT .venv-driver ***
ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 \
  RL_LOG_DIR=$RUN/rl-logs RL_DETERMINISTIC=1 \
  RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
  uv run python $CHAL/rl/policy_driver.py > $RUN/txt-logs/driver.log 2>&1 &
PIDS+=($!)

uv run python -m alpasim_controller.server --port=6001 --log_dir=$RUN/controller \
  --log-level=INFO --config=$RUN/controller-config.yaml > $RUN/txt-logs/controller.log 2>&1 &
PIDS+=($!)
uv run alpasim-mtgs-server --user-config=$RUN/generated-user-config-0.yaml \
  --host=0.0.0.0 --port=6000 --cache-size=5 --log-level=INFO > $RUN/txt-logs/renderer.log 2>&1 &
PIDS+=($!)

for p in 6789 6001 6000; do
  for i in $(seq 1 180); do
    (echo > /dev/tcp/localhost/$p) 2>/dev/null && { echo "port $p UP"; break; }
    sleep 5; [ $i -eq 180 ] && { echo "PORT $p NEVER UP"; tail -40 $RUN/txt-logs/*.log; exit 1; }
  done
done

uv run python -m alpasim_runtime.simulate \
  --user-config=$RUN/generated-user-config-0.yaml \
  --network-config=$RUN/generated-network-config.yaml \
  --log-dir=$RUN --log-level=INFO --array-job-dir=$RUN --eval-config=$RUN/eval-config.yaml

echo "=== M2 CHECKS ==="
echo "--- driver log (policy loaded? camera arriving? actions?) ---"
grep -iE "policy|ckpt|flushed|drive failed|CAM_F0|error" $RUN/txt-logs/driver.log | tail -20

echo "--- did it drive + score? ---"
python3 -c "
import json,glob
f=glob.glob('$RUN/aggregate/*.json')
if f:
  d=json.load(open(f[0]))
  for r in d['rollouts']:
    m=r['metrics']
    print(f\"  score={r['score']:.2f} dist_to_gt={m['dist_to_gt_trajectory']:.2f} trav={m['dist_traveled_m']:.1f} coll={m['collision_at_fault']:.0f} {r['status']}\")
" 2>/dev/null || echo "  no aggregate json"

echo "--- transition log written? (the PPO data) ---"
ls -la $RUN/rl-logs/ 2>/dev/null
python3 -c "
import numpy as np, glob
f=glob.glob('$RUN/rl-logs/*.npz')
if f:
  z=np.load(f[0])
  print(f'  logged {len(z[\"pre_action\"])} transitions')
  print(f'  img shape {z[\"img\"].shape}  (nonzero: {np.any(z[\"img\"])})')
  print(f'  actions (speed-mult pre-squash): mean={z[\"pre_action\"].mean():.2f}')
  print(f'  values: mean={z[\"value\"].mean():.3f}')
else:
  print('  NO transition log written')
" 2>/dev/null || echo "  transition log check failed"
