#!/bin/bash
#SBATCH --job-name=nuplan-gov-ab
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nanjaiyalathaa/alpasim/logs/%x-%j.out
set -euo pipefail
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
cd $REPO

# The 6 scenes: 2 collisions + 4 clear (from navtest_dev)
SCENES='scenes.scene_ids=[2021.09.16.15.47.30_veh-45_01199_01391-dd162898ecf351d4,2021.09.16.19.27.01_veh-45_00472_00711-8cc8fe63227d5c07,2021.09.09.14.18.22_veh-48_00322_00895-52d3f15d6ca75701,2021.05.25.14.16.10_veh-35_01100_01664-f02b15cc225b5d9a,2021.05.25.14.16.10_veh-35_01690_02183-0b57b00279885fd4,2021.05.25.14.16.10_veh-35_01690_02183-19488eb3301f5d26]'

run_one () {
  local TAG=$1 GOV=$2
  local RUN=$REPO/runs/gov-${TAG}-$SLURM_JOB_ID
  echo "================ RUN $TAG (GOV_ENABLE=$GOV) ================"
  mkdir -p $RUN/txt-logs $RUN/controller

  ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
  uv run alpasim_wizard +e2e_challenge_nuplan=dev \
    wizard.run_method=NONE wizard.log_dir=$RUN \
    scenes.limit_to_first_n=0 "$SCENES" \
    eval.video.render_video=false

  PIDS=()
  cleanup() { for p in "${PIDS[@]:-}"; do kill $p 2>/dev/null || true; done; }
  trap cleanup RETURN

  source $REPO/.venv-driver/bin/activate
  ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 \
    RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
    GOV_ENABLE=$GOV GOV_GAIN=8.0 GOV_FLOOR=0.15 \
    python $CHAL/driver/route_follower_gov.py > $RUN/txt-logs/driver.log 2>&1 &
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

  for p in 6789 6001 6000; do
    for i in $(seq 1 180); do
      (echo > /dev/tcp/localhost/$p) 2>/dev/null && { echo "port $p UP"; break; }
      sleep 5
      [ $i -eq 180 ] && { echo "PORT $p NEVER CAME UP"; tail -30 $RUN/txt-logs/*.log; return 1; }
    done
  done

  uv run python -m alpasim_runtime.simulate \
    --user-config=$RUN/generated-user-config-0.yaml \
    --network-config=$RUN/generated-network-config.yaml \
    --log-dir=$RUN --log-level=INFO \
    --array-job-dir=$RUN \
    --eval-config=$RUN/eval-config.yaml

  echo "=== RESULTS [$TAG] ==="
  cat $RUN/aggregate/*.json 2>/dev/null || echo "no summary"
  cleanup; trap - RETURN
  sleep 5   # let ports free before next run
}

run_one baseline 0
run_one governor 1

echo "================ DONE. Compare gov-baseline-* vs gov-governor-* ================"
