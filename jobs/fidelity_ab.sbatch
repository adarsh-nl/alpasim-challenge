#!/bin/bash
#SBATCH --job-name=nuplan-fidelity
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

# 5 TAIL scenes (bad dist_to_gt, short routes) + 3 TIGHT scenes (already <0.85, regression check)
SCENES='scenes.scene_ids=[2021.09.16.15.47.30_veh-45_01199_01391-dd162898ecf351d4,2021.05.25.14.16.10_veh-35_01100_01664-f02b15cc225b5d9a,2021.05.25.14.16.10_veh-35_01690_02183-0b57b00279885fd4,2021.05.25.14.16.10_veh-35_01690_02183-19488eb3301f5d26,2021.09.09.14.18.22_veh-48_00322_00895-52d3f15d6ca75701]'
# NOTE: using scenes we have confirmed full IDs for. dd162898 is the key tail/collision scene;
# the other 4 are tight/clear regression controls. (Full IDs for the other tail scenes need lookup;
# this run proves the mechanism on dd162898 + confirms zero regression on the 4 controls first.)

run_one () {
  local TAG=$1 DRIVER=$2 TERM=$3 VEND=$4
  local RUN=$REPO/runs/fidelity-${TAG}-$SLURM_JOB_ID
  echo "================ RUN $TAG (driver=$(basename $DRIVER) TERM_DECEL=$TERM V_END=$VEND) ================"
  mkdir -p $RUN/txt-logs $RUN/controller

  ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
  uv run alpasim_wizard +e2e_challenge_nuplan=dev \
    wizard.run_method=NONE wizard.log_dir=$RUN \
    scenes.limit_to_first_n=0 "$SCENES" eval.video.render_video=false

  PIDS=()
  cleanup() { for p in "${PIDS[@]:-}"; do kill $p 2>/dev/null || true; done; }
  trap cleanup RETURN

  source $REPO/.venv-driver/bin/activate
  ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 \
    RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
    RF_TERM_DECEL=$TERM RF_TERM_V_END=$VEND \
    python $DRIVER > $RUN/txt-logs/driver.log 2>&1 &
  PIDS+=($!); deactivate

  uv run python -m alpasim_controller.server --port=6001 --log_dir=$RUN/controller \
    --log-level=INFO --config=$RUN/controller-config.yaml > $RUN/txt-logs/controller.log 2>&1 &
  PIDS+=($!)
  uv run alpasim-mtgs-server --user-config=$RUN/generated-user-config-0.yaml \
    --host=0.0.0.0 --port=6000 --cache-size=5 --log-level=INFO > $RUN/txt-logs/renderer.log 2>&1 &
  PIDS+=($!)

  for p in 6789 6001 6000; do
    for i in $(seq 1 180); do
      (echo > /dev/tcp/localhost/$p) 2>/dev/null && { echo "port $p UP"; break; }
      sleep 5; [ $i -eq 180 ] && { echo "PORT $p NEVER UP"; tail -30 $RUN/txt-logs/*.log; return 1; }
    done
  done

  uv run python -m alpasim_runtime.simulate \
    --user-config=$RUN/generated-user-config-0.yaml \
    --network-config=$RUN/generated-network-config.yaml \
    --log-dir=$RUN --log-level=INFO --array-job-dir=$RUN --eval-config=$RUN/eval-config.yaml

  echo "=== RESULTS [$TAG] ==="
  python3 -c "
import json,glob
f=glob.glob('$RUN/aggregate/*.json')
if f:
  d=json.load(open(f[0])); agg=d['metrics_results'][0]
  print(f\"  AGG coll_at_fault={agg['collision_at_fault']:.2f} dist_to_gt={agg['dist_to_gt_trajectory']:.3f} dist_to_gt_loc={agg['dist_to_gt_location']:.2f} progress={agg['progress']:.2f}\")
  for r in sorted(d['rollouts'],key=lambda x:x['clipgt_id']):
    m=r['metrics']; sid=r['clipgt_id'].split('-')[-1]
    print(f\"    {sid}  traj={m['dist_to_gt_trajectory']:.2f} loc={m['dist_to_gt_location']:.2f} trav={m['dist_traveled_m']:.1f} gt={m['gt_dist_traveled_m']:.1f} coll={m['collision_at_fault']:.0f} collany={m['collision_any']:.0f} score={r['score']:.2f} {r['status']}\")
" 2>/dev/null || cat $RUN/aggregate/*.json | head -3
  cleanup; trap - RETURN; sleep 5
}

run_one baseline $CHAL/driver/route_follower.py      0 0.0   # your REAL rank-1 follower, no change
run_one term     $CHAL/driver/route_follower_term.py 1 0.0   # terminal decel, stop at route end

echo "================ DONE. baseline vs term ================"
