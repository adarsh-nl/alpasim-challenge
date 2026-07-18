#!/bin/bash
#SBATCH --job-name=v4-gate
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
cd $CHAL/rl

# eval scenes = the held-out 12 the v4 run used
SCENES="scenes.scene_ids=[$(paste -sd, eval_scenes_v2.txt)]"
RUN=$REPO/runs/follower-on-v4eval-$SLURM_JOB_ID
mkdir -p $RUN/txt-logs

cd $REPO
uv run alpasim_wizard +e2e_challenge_nuplan=dev wizard.run_method=NONE \
  wizard.log_dir=$RUN scenes.limit_to_first_n=0 "$SCENES" eval.video.render_video=false

PIDS=(); cleanup(){ for p in "${PIDS[@]:-}"; do kill $p 2>/dev/null||true; done; }
trap cleanup EXIT
# FOLLOWER baseline (route_follower, no abstention, no policy)
RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
  uv run python $CHAL/driver/route_follower.py > $RUN/txt-logs/driver.log 2>&1 &
PIDS+=($!)
uv run python -m alpasim_controller.server --port=6001 --log_dir=$RUN/controller \
  --log-level=INFO --config=$RUN/controller-config.yaml > $RUN/txt-logs/controller.log 2>&1 &
PIDS+=($!)
uv run alpasim-mtgs-server --user-config=$RUN/generated-user-config-0.yaml \
  --host=0.0.0.0 --port=6000 --cache-size=5 --log-level=INFO > $RUN/txt-logs/renderer.log 2>&1 &
PIDS+=($!)
for p in 6789 6001 6000; do for i in $(seq 1 180); do
  (echo > /dev/tcp/localhost/$p) 2>/dev/null && break; sleep 5
  [ $i -eq 180 ] && { echo "PORT $p FAIL"; exit 1; }; done; done
uv run python -m alpasim_runtime.simulate \
  --user-config=$RUN/generated-user-config-0.yaml \
  --network-config=$RUN/generated-network-config.yaml \
  --log-dir=$RUN --log-level=INFO --array-job-dir=$RUN --eval-config=$RUN/eval-config.yaml
cleanup; trap - EXIT; sleep 5

echo "=== follower baseline done: $RUN ==="
# find v4's last eval dir (the candidate)
V4EVAL=$(ls -dt $REPO/runs/rl-train-v4-*/eval-* 2>/dev/null | head -1)
echo "=== v4 candidate eval: $V4EVAL ==="
echo ""
echo "############ REGRESSION GATE: v4 policy vs follower ############"
cd $CHAL/rl
RL_REWARD=v2 python regression_gate.py --baseline $RUN --candidate $V4EVAL --tol 0.02
