#!/bin/bash
#SBATCH --job-name=nuplan-rf
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
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
RUN=$REPO/runs/rf-$SLURM_JOB_ID
cd $REPO

# --- 1. generate configs only, no docker ---
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.run_method=NONE \
  wizard.log_dir=$RUN

mkdir -p $RUN/controller

# --- 2. launch services as bare processes ---
PIDS=()
cleanup() { echo "--- cleanup ---"; for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done; }
trap cleanup EXIT

source $REPO/.venv-driver/bin/activate
ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 \
  python $REPO/../alpasim-challenge/driver/route_follower.py > $RUN/txt-logs/driver.log 2>&1 &
PIDS+=($!); deactivate

uv run python -m alpasim_controller.server \
  --port=6001 --log_dir=$RUN/controller --log-level=INFO \
  --config=$RUN/controller-config.yaml > $RUN/txt-logs/controller.log 2>&1 &
PIDS+=($!)

uv run alpasim-mtgs-server \
  --user-config=$RUN/generated-user-config-0.yaml \
  --host=0.0.0.0 --port=6000 --cache-size=5 --log-level=INFO \
  > $RUN/txt-logs/renderer.log 2>&1 &
PIDS+=($!)

# --- 3. wait for all three to listen ---
for p in 6789 6001 6000; do
  for i in $(seq 1 180); do
    (echo > /dev/tcp/localhost/$p) 2>/dev/null && { echo "port $p UP"; break; }
    sleep 5
    [ $i -eq 180 ] && { echo "PORT $p NEVER CAME UP"; tail -30 $RUN/txt-logs/*.log; exit 1; }
  done
done

# --- 4. run the sim ---
uv run python -m alpasim_runtime.simulate \
  --user-config=$RUN/generated-user-config-0.yaml \
  --network-config=$RUN/generated-network-config.yaml \
  --log-dir=$RUN --log-level=INFO \
  --array-job-dir=$RUN \
  --eval-config=$RUN/eval-config.yaml

echo "=== RESULTS ==="
cat $RUN/aggregate/results-summary.json 2>/dev/null || echo "no summary written"
