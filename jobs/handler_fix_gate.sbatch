#!/bin/bash
# STAGED -- NOT SUBMITTED. Paired regression gate for the drive-handler
# fallback fix (branch fix/drive-handler-fallback vs main).
#
# Design note: both arms run BACK-TO-BACK IN THE SAME ARRAY TASK, on the same
# node, against the same /local-staged assets and the same scene list. That
# removes node-to-node and staging variation from the paired comparison, which
# matters here because the expected effect size is exactly zero -- any drift
# would otherwise be indistinguishable from the thing we are measuring.
#
# Submit with:  sbatch --array=0-31%8 jobs/handler_fix_gate.sbatch
#SBATCH --job-name=handler-fix-gate
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nanjaiyalathaa/alpasim/logs/%x-%A_%a.out

set -x
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
ARRAY_DIR=$REPO/runs/handlerfix-$SLURM_ARRAY_JOB_ID
TID=$SLURM_ARRAY_TASK_ID
LOCAL=/local/$SLURM_JOB_ID
cd $REPO

# --- the two drivers under test -------------------------------------------
# BASELINE = the shipped rank-1 driver as it exists on main (pre-fix).
# CANDIDATE = the same file on fix/drive-handler-fallback.
mkdir -p $ARRAY_DIR
git -C $CHAL show main:submission/route_follower.py           > $ARRAY_DIR/driver_baseline.py
git -C $CHAL show fix/drive-handler-fallback:submission/route_follower.py \
                                                              > $ARRAY_DIR/driver_candidate.py
# Sanity: they must differ ONLY in the handler fix.
diff $ARRAY_DIR/driver_baseline.py $ARRAY_DIR/driver_candidate.py > $ARRAY_DIR/driver.diff || true

# --- stage this task's scene subset once, reuse for both arms -------------
run_arm () {
  local ARM=$1 DRIVER=$2 PORT=$3
  local RUN=$ARRAY_DIR/$ARM/task-$TID
  mkdir -p $RUN/controller $RUN/txt-logs

  ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=$PORT \
  uv run alpasim_wizard +e2e_challenge_nuplan=dev \
    nuplan_scenes=navtest_full scenes.limit_to_first_n=0 \
    wizard.run_method=NONE \
    wizard.log_dir=$RUN wizard.array_job_dir=$ARRAY_DIR/$ARM \
    eval.video.render_video=false || { echo "WIZARD FAILED ($ARM)"; return 1; }

  local CFG=$RUN/generated-user-config-$TID.yaml
  [ -f "$CFG" ] || { echo "CONFIG MISSING: $CFG"; return 1; }
  sed -i "s|/datasets/eemcs/ps/alpasim-nuplan-2026|$LOCAL|g" $CFG
  sed -i "s|localhost:6001|localhost:$((PORT+1))|; s|localhost:6000|localhost:$((PORT+2))|" \
    $RUN/generated-network-config.yaml

  PIDS=()
  cleanup_arm(){ for p in "${PIDS[@]:-}"; do kill $p 2>/dev/null; done; }

  RF_V_MAX=20 RF_A_LON=6.0 RF_A_LAT=7.0 \
    ALPASIM_DRIVER_HOST=0.0.0.0 ALPASIM_DRIVER_PORT=$PORT \
    uv run python $DRIVER > $RUN/txt-logs/driver.log 2>&1 &
  PIDS+=($!)
  uv run python -m alpasim_controller.server --port=$((PORT+1)) \
    --log_dir=$RUN/controller --log-level=INFO \
    --config=$RUN/controller-config.yaml > $RUN/txt-logs/controller.log 2>&1 &
  PIDS+=($!)
  uv run alpasim-mtgs-server --user-config=$CFG \
    --host=0.0.0.0 --port=$((PORT+2)) --cache-size=5 --log-level=INFO \
    > $RUN/txt-logs/renderer.log 2>&1 &
  PIDS+=($!)

  for p in $PORT $((PORT+1)) $((PORT+2)); do
    for i in $(seq 1 180); do
      (echo > /dev/tcp/localhost/$p) 2>/dev/null && break
      sleep 5
      [ $i -eq 180 ] && { echo "PORT $p NEVER CAME UP ($ARM)"; cleanup_arm; return 1; }
    done
  done

  uv run python -m alpasim_runtime.simulate \
    --user-config=$CFG \
    --network-config=$RUN/generated-network-config.yaml \
    --log-dir=$RUN --log-level=INFO --array-job-dir=$ARRAY_DIR/$ARM \
    --eval-config=$RUN/eval-config.yaml
  local rc=$?
  cleanup_arm
  sleep 10
  return $rc
}

# stage assets for this task's scenes (shared by both arms)
BASE_PORT=$((6789 + TID * 20))
mkdir -p $LOCAL/navtest/assets
cp -r $ALPASIM_NUPLAN_ROOT/navtest/configs $LOCAL/navtest/
cp -r $ALPASIM_NUPLAN_ROOT/nuplan_test $LOCAL/
trap 'rm -rf $LOCAL' EXIT

# arm 1 generates the config we harvest the scene list from
run_arm baseline  $ARRAY_DIR/driver_baseline.py  $BASE_PORT
run_arm candidate $ARRAY_DIR/driver_candidate.py $((BASE_PORT + 10))

echo "=== task $TID done: $ARRAY_DIR/{baseline,candidate}/task-$TID ==="
