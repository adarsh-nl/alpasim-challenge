#!/bin/bash
#SBATCH --job-name=rf-array
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-19%6
#SBATCH --output=/home/nanjaiyalathaa/alpasim/logs/%x-%A_%a.out

set -x
hostname; nvidia-smi --query-gpu=name --format=csv,noheader

export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export ALPASIM_NUPLAN_ROOT=/home/nanjaiyalathaa/data/alpasim-nuplan
export HF_HOME=/home/nanjaiyalathaa/data/hf-cache
export LC_ALL=C.UTF-8
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST="8.9"
module load nvidia/cuda-12.4

REPO=/home/nanjaiyalathaa/alpasim
CHAL=/home/nanjaiyalathaa/alpasim-challenge
ARRAY_DIR=$REPO/runs/array-$SLURM_ARRAY_JOB_ID
RUN=$ARRAY_DIR/task-$SLURM_ARRAY_TASK_ID
SHARD=$(printf "shard%02d" $SLURM_ARRAY_TASK_ID)
PORT=$((6789 + SLURM_ARRAY_TASK_ID * 10))
cd $REPO

ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=$PORT \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  nuplan_scenes=$SHARD scenes.limit_to_first_n=0 \
  wizard.run_method=NONE \
  wizard.log_dir=$RUN wizard.array_job_dir=$ARRAY_DIR \
  eval.video.render_video=false

rm -rf $RUN/txt-logs
mkdir -p $RUN/controller $RUN/txt-logs
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; }
trap cleanup EXIT

source $REPO/.venv-driver/bin/activate
ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=$PORT \
  RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
  python $CHAL/driver/route_follower.py > $RUN/txt-logs/driver.log 2>&1 &
PIDS+=($!)
deactivate

uv run python -m alpasim_controller.server --port=$((PORT+1)) \
  --log_dir=$RUN/controller --log-level=INFO \
  --config=$RUN/controller-config.yaml > $RUN/txt-logs/controller.log 2>&1 &
PIDS+=($!)

uv run alpasim-mtgs-server --user-config=$RUN/generated-user-config-0.yaml \
  --host=0.0.0.0 --port=$((PORT+2)) --cache-size=5 --log-level=INFO \
  > $RUN/txt-logs/renderer.log 2>&1 &
PIDS+=($!)

echo "[$(date +%H:%M:%S)] services launched; sleeping 600s for renderer indexing"
sleep 600
echo "[$(date +%H:%M:%S)] proceeding"

for attempt in 1 2 3; do
  echo "[$(date +%H:%M:%S)] runtime attempt $attempt"
  uv run python -m alpasim_runtime.simulate \
    --user-config=$RUN/generated-user-config-0.yaml \
    --network-config=$RUN/generated-network-config.yaml \
    --log-dir=$RUN --log-level=INFO --array-job-dir=$ARRAY_DIR \
    --eval-config=$RUN/eval-config.yaml && break
  echo "attempt $attempt failed; waiting 120s"
  sleep 120
done
