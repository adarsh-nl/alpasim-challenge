#!/bin/bash
#SBATCH --job-name=nuplan-render-rearend
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
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

# Just the rear-end scene
SCENES='scenes.scene_ids=[2021.09.16.15.47.30_veh-45_01199_01391-dd162898ecf351d4]'

run_render () {
  local TAG=$1 CREEP=$2
  local RUN=$REPO/runs/render-${TAG}-$SLURM_JOB_ID
  echo "================ RENDER $TAG (RF_DEGEN_CREEP=$CREEP) ================"
  mkdir -p $RUN/txt-logs $RUN/controller

  # dev preset already has render_video=true and limit_to_first_n=1; we override scenes + turn on the combined video
  ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
  uv run alpasim_wizard +e2e_challenge_nuplan=dev \
    wizard.run_method=NONE wizard.log_dir=$RUN \
    scenes.limit_to_first_n=0 "$SCENES" \
    eval.video.render_video=true \
    eval.video.generate_combined_video=true

  PIDS=()
  cleanup() { for p in "${PIDS[@]:-}"; do kill $p 2>/dev/null || true; done; }
  trap cleanup RETURN

  source $REPO/.venv-driver/bin/activate
  ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 \
    RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
    GOV_ENABLE=0 RF_DEGEN_CREEP=$CREEP \
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
      [ $i -eq 180 ] && { echo "PORT $p NEVER UP"; tail -30 $RUN/txt-logs/*.log; return 1; }
    done
  done

  uv run python -m alpasim_runtime.simulate \
    --user-config=$RUN/generated-user-config-0.yaml \
    --network-config=$RUN/generated-network-config.yaml \
    --log-dir=$RUN --log-level=INFO --array-job-dir=$RUN \
    --eval-config=$RUN/eval-config.yaml

  echo "=== [$TAG] scene result ==="
  python3 -c "
import json,glob
f=glob.glob('$RUN/aggregate/*.json')
if f:
  d=json.load(open(f[0]))
  for r in d.get('rollouts',[]):
    m=r['metrics']
    print(f\"  coll_at_fault={m['collision_at_fault']:.0f} dist_to_gt={m['dist_to_gt_trajectory']:.2f} dist_traveled={m['dist_traveled_m']:.2f} score={r['score']:.2f} {r['status']}\")
" 2>/dev/null || echo "no json"
  echo "=== [$TAG] video files ==="
  find $RUN -name "*.mp4" -o -name "*.webm" 2>/dev/null | while read v; do echo "  $v ($(du -h "$v" | cut -f1))"; done
  cleanup; trap - RETURN
  sleep 5
}

# Render baseline-behavior (charge, reproduces the ORIGINAL collision) AND the fix (stop-0.3)
# so we can watch them side by side. Use the ORIGINAL route_follower for true baseline:
echo "########## TRUE BASELINE (original route_follower.py, no fix) ##########"
RUN=$REPO/runs/render-truebase-$SLURM_JOB_ID
mkdir -p $RUN/txt-logs $RUN/controller
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.run_method=NONE wizard.log_dir=$RUN \
  scenes.limit_to_first_n=0 "$SCENES" \
  eval.video.render_video=true eval.video.generate_combined_video=true
PIDS=()
cleanup2() { for p in "${PIDS[@]:-}"; do kill $p 2>/dev/null || true; done; }
trap cleanup2 EXIT
source $REPO/.venv-driver/bin/activate
ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 \
  RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
  python $CHAL/driver/route_follower.py > $RUN/txt-logs/driver.log 2>&1 &
PIDS+=($!); deactivate
uv run python -m alpasim_controller.server --port=6001 --log_dir=$RUN/controller --log-level=INFO --config=$RUN/controller-config.yaml > $RUN/txt-logs/controller.log 2>&1 &
PIDS+=($!)
uv run alpasim-mtgs-server --user-config=$RUN/generated-user-config-0.yaml --host=0.0.0.0 --port=6000 --cache-size=5 --log-level=INFO > $RUN/txt-logs/renderer.log 2>&1 &
PIDS+=($!)
for p in 6789 6001 6000; do for i in $(seq 1 180); do (echo > /dev/tcp/localhost/$p) 2>/dev/null && { echo "port $p UP"; break; }; sleep 5; [ $i -eq 180 ] && { echo "PORT $p NEVER UP"; exit 1; }; done; done
uv run python -m alpasim_runtime.simulate --user-config=$RUN/generated-user-config-0.yaml --network-config=$RUN/generated-network-config.yaml --log-dir=$RUN --log-level=INFO --array-job-dir=$RUN --eval-config=$RUN/eval-config.yaml
echo "=== [truebase] video ==="; find $RUN -name "*.mp4" | while read v; do echo "  $v ($(du -h "$v"|cut -f1))"; done
cleanup2; trap - EXIT; sleep 5

run_render stop03 0.3

echo "================ DONE. Two videos: truebase (collision) vs stop03 (fix) ================"
