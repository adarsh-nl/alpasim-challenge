#!/bin/bash
#SBATCH --job-name=nuplan-degen9
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

# All 9 degenerate-route scenes (1 valid waypoint each)
SCENES='scenes.scene_ids=[2021.06.28.21.47.53_veh-35_00280_00424-9ad434ec99685d35,2021.09.09.18.04.06_veh-40_00743_01071-6d2b59f78d995d9c,2021.09.16.15.12.03_veh-42_01037_01434-16ee6602bb5c5e99,2021.06.28.20.24.43_veh-38_03385_04952-1786ef7edac65502,2021.09.16.15.47.30_veh-45_01199_01391-dd162898ecf351d4,2021.09.16.19.49.00_veh-42_00990_01609-458e833803315b4f,2021.08.30.16.16.44_veh-40_01099_01351-0f24ec666e145cab,2021.09.09.18.29.25_veh-39_00569_00903-46cd5885206b5c6d,2021.09.16.14.39.34_veh-42_01111_01448-a09e4156f577568a]'

run_one () {
  local TAG=$1 DRIVER=$2 CREEP=$3
  local RUN=$REPO/runs/degen9-${TAG}-$SLURM_JOB_ID
  echo "================ RUN $TAG (driver=$DRIVER creep=$CREEP) ================"
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
    GOV_ENABLE=0 RF_DEGEN_CREEP=$CREEP \
    python $DRIVER > $RUN/txt-logs/driver.log 2>&1 &
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

  echo "=== RESULTS [$TAG] ==="
  python3 -c "
import json,glob
f=glob.glob('$RUN/aggregate/*.json')
if f:
  d=json.load(open(f[0]))
  agg=d['metrics_results'][0]
  print(f\"  AGG: coll_at_fault={agg['collision_at_fault']:.2f} dist_to_gt={agg['dist_to_gt_trajectory']:.2f} progress={agg['progress']:.2f}\")
  for r in sorted(d.get('rollouts',[]), key=lambda x:x['clipgt_id']):
    sid=r['clipgt_id'].split('-')[-1]
    m=r['metrics']
    print(f\"    {sid}  coll={m['collision_at_fault']:.0f} dist_to_gt={m['dist_to_gt_trajectory']:.2f} prog={m['progress']:.2f} gtdist={m['gt_dist_traveled_m']:.1f} score={r['score']:.2f} {r['status']}\")
" 2>/dev/null || cat $RUN/aggregate/*.json | head -5
  cleanup; trap - RETURN
  sleep 5
}

# Baseline = original route_follower (charges straight on degenerate route)
run_one baseline $CHAL/driver/route_follower.py 1.5
# Fix = governor driver with stop-0.3 degenerate fallback
run_one stop03   $CHAL/driver/route_follower_gov.py 0.3

echo "================ DONE. baseline vs stop03 across all 9 degenerate scenes ================"
