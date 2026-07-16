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
export ALPASIM_NUPLAN_ROOT=/datasets/eemcs/ps/alpasim-nuplan-2026
export HF_HOME=/home/nanjaiyalathaa/data/hf-cache
export LC_ALL=C.UTF-8
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST="8.9"
module load nvidia/cuda-12.4

REPO=/home/nanjaiyalathaa/alpasim
CHAL=/home/nanjaiyalathaa/alpasim-challenge
ARRAY_DIR=$REPO/runs/array-$SLURM_ARRAY_JOB_ID
RUN=$ARRAY_DIR/task-$SLURM_ARRAY_TASK_ID
TID=$SLURM_ARRAY_TASK_ID
PORT=$((6789 + TID * 10))
LOCAL=/local/$SLURM_JOB_ID
cd $REPO

# 1. wizard: same 200-scene list for every task; wizard splits by array index
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=$PORT \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  nuplan_scenes=sample200 scenes.limit_to_first_n=0 \
  wizard.run_method=NONE \
  wizard.log_dir=$RUN wizard.array_job_dir=$ARRAY_DIR \
  eval.video.render_video=false \
  || { echo "WIZARD FAILED"; exit 1; }

CFG=$RUN/generated-user-config-$TID.yaml
[ -f "$CFG" ] || { echo "CONFIG MISSING: $CFG"; ls $RUN; exit 1; }

# 2. stage this task's data to local NVMe (admin request + kills NFS bottleneck)
mkdir -p $LOCAL/navtest/assets
cp -r $ALPASIM_NUPLAN_ROOT/navtest/configs $LOCAL/navtest/
cp -r $ALPASIM_NUPLAN_ROOT/nuplan_test $LOCAL/
grep -oE '2021\.[0-9.]+_veh-[0-9]+_[0-9]+_[0-9]+-[a-f0-9]+' $CFG | sort -u > $LOCAL/scene_list.txt
echo "task $TID scene count: $(wc -l < $LOCAL/scene_list.txt)"
while read sid; do
  cp -r $ALPASIM_NUPLAN_ROOT/navtest/assets/$sid $LOCAL/navtest/assets/ || echo "WARN missing asset $sid"
done < $LOCAL/scene_list.txt

# 3. point generated configs at the local copy
sed -i "s|/datasets/eemcs/ps/alpasim-nuplan-2026|$LOCAL|g" $CFG
# 4. rewrite managed-service ports to per-task ports
sed -i "s|localhost:6001|localhost:$((PORT+1))|; s|localhost:6000|localhost:$((PORT+2))|" \
  $RUN/generated-network-config.yaml
grep -n "address" $RUN/generated-network-config.yaml

rm -rf $RUN/txt-logs
mkdir -p $RUN/controller $RUN/txt-logs
PIDS=()
cleanup(){ for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; rm -rf $LOCAL; }
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

uv run alpasim-mtgs-server --user-config=$CFG \
  --host=0.0.0.0 --port=$((PORT+2)) --cache-size=5 --log-level=INFO \
  > $RUN/txt-logs/renderer.log 2>&1 &
PIDS+=($!)

echo "[$(date +%H:%M:%S)] services launched; sleeping 240s (local NVMe indexing)"
sleep 240

for attempt in 1 2 3; do
  echo "[$(date +%H:%M:%S)] runtime attempt $attempt"
  uv run python -m alpasim_runtime.simulate \
    --user-config=$CFG \
    --network-config=$RUN/generated-network-config.yaml \
    --log-dir=$RUN --log-level=INFO --array-job-dir=$ARRAY_DIR \
    --eval-config=$RUN/eval-config.yaml && break
  echo "attempt $attempt failed; waiting 120s"
  sleep 120
done
