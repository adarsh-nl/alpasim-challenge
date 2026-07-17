#!/bin/bash
#SBATCH --job-name=abstain-ab
#SBATCH --partition=ps
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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

# route-absent collision scene + clear controls + a couple of tail scenes
SCENES='scenes.scene_ids=[2021.09.16.15.47.30_veh-45_01199_01391-dd162898ecf351d4,2021.05.25.14.16.10_veh-35_01100_01664-f02b15cc225b5d9a,2021.05.25.14.16.10_veh-35_01690_02183-0b57b00279885fd4,2021.05.25.14.16.10_veh-35_01690_02183-19488eb3301f5d26,2021.09.09.14.18.22_veh-48_00322_00895-52d3f15d6ca75701]'

run_one () {
  local TAG=$1 ABST=$2
  local RUN=$REPO/runs/abstain-${TAG}-$SLURM_JOB_ID
  echo "======== $TAG (ABSTAIN=$ABST) ========"
  mkdir -p $RUN/txt-logs
  uv run alpasim_wizard +e2e_challenge_nuplan=dev wizard.run_method=NONE \
    wizard.log_dir=$RUN scenes.limit_to_first_n=0 "$SCENES" eval.video.render_video=false
  PIDS=(); cleanup(){ for p in "${PIDS[@]:-}"; do kill $p 2>/dev/null||true; done; }
  trap cleanup RETURN
  ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=6789 RF_ABSTAIN=$ABST \
    RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
    uv run python $CHAL/driver/abstention_driver.py > $RUN/txt-logs/driver.log 2>&1 &
  PIDS+=($!)
  uv run python -m alpasim_controller.server --port=6001 --log_dir=$RUN/controller \
    --log-level=INFO --config=$RUN/controller-config.yaml > $RUN/txt-logs/controller.log 2>&1 &
  PIDS+=($!)
  uv run alpasim-mtgs-server --user-config=$RUN/generated-user-config-0.yaml \
    --host=0.0.0.0 --port=6000 --cache-size=5 --log-level=INFO > $RUN/txt-logs/renderer.log 2>&1 &
  PIDS+=($!)
  for p in 6789 6001 6000; do for i in $(seq 1 180); do
    (echo > /dev/tcp/localhost/$p) 2>/dev/null && break; sleep 5
    [ $i -eq 180 ] && { echo "PORT $p FAIL"; return 1; }; done; done
  uv run python -m alpasim_runtime.simulate \
    --user-config=$RUN/generated-user-config-0.yaml \
    --network-config=$RUN/generated-network-config.yaml \
    --log-dir=$RUN --log-level=INFO --array-job-dir=$RUN --eval-config=$RUN/eval-config.yaml
  echo "=== $TAG results ==="
  python3 -c "
import json,glob
d=json.load(open(glob.glob('$RUN/aggregate/*.json')[0]))
for r in sorted(d['rollouts'],key=lambda x:x['clipgt_id']):
  m=r['metrics']; print(f\"  {r['clipgt_id'].split('-')[-1]}: score={r['score']:.2f} dist_to_gt={m['dist_to_gt_trajectory']:.2f} trav={m['dist_traveled_m']:.1f} coll={m['collision_at_fault']:.0f} collany={m['collision_any']:.0f}\")
" 2>/dev/null
  cleanup; trap - RETURN; sleep 5
}
run_one follower 0
run_one abstain  1
echo "======== compare follower vs abstain above ========"
