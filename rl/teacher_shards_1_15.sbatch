#!/bin/bash
# D1 — STAGED, NOT SUBMITTED. Teacher labels for shards 1..15.
#
# Shard 0 is already complete and VERIFIED (6080 samples, 760 scenes, 0 schema
# errors, 0 duplicate tokens, 0.033% kinematic flag rate) — see
# reports/d1_label_integrity.md. This job does the remaining 15 of 16 shards.
#
# Submit with:   sbatch --array=1-15%4 rl/teacher_shards_1_15.sbatch
#   %4 throttles to 4 concurrent tasks — shard 0 took ~12 min on one L40S, so
#   15 shards at 4-way is ~1 hour. Raise the throttle if the partition is idle.
#
# NOTE: run_teacher.py does NOT skip existing outputs (no exists() check before
# savez), so this array deliberately starts at 1. Re-running 0 would redo ~12
# min of work and rewrite identical files. Do not use --array=0-15.
#
#SBATCH --job-name=dp-teacher-1-15
#SBATCH --partition=ps
#SBATCH --account=ps
#SBATCH --qos=research
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nanjaiyalathaa/alpasim-challenge/logs/teacher_%A_%a.out

set -euo pipefail

SHARD="${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"
if [ "$SHARD" -eq 0 ]; then
  echo "shard 0 is already complete and verified; refusing to redo it." >&2
  exit 1
fi

echo "=== host=$(hostname) shard=$SHARD/16 start=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export ALPASIM_NUPLAN_ROOT=/datasets/eemcs/ps/alpasim-nuplan-2026
export LC_ALL=C.UTF-8
module load nvidia/cuda-12.4

CHAL=/home/nanjaiyalathaa/alpasim-challenge
# Per the cluster rules: compute on node-local NVMe, copy results back at the
# end. Never write intermediate job data straight into /home.
SCRATCH="/local/$SLURM_JOB_ID"
mkdir -p "$SCRATCH/labels"
trap 'rm -rf "$SCRATCH"' EXIT

source /home/nanjaiyalathaa/alpasim/.venv/bin/activate
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

cd "$CHAL/rl"
python run_teacher.py \
  --root "$ALPASIM_NUPLAN_ROOT" \
  --repo ./dp_repo \
  --ckpt ./dp_ckpt \
  --out  "$SCRATCH/labels" \
  --shard "$SHARD" \
  --n-shards 16 \
  --batch-size 64 \
  --device cuda \
  --audit-every 500

N=$(ls "$SCRATCH/labels" | wc -l)
echo "=== shard $SHARD produced $N files; copying back ==="
rsync -a "$SCRATCH/labels/" "$CHAL/rl/teacher_labels/"

echo "=== done=$(date -Is) shard=$SHARD files=$N ==="
echo "After ALL shards finish, re-run the integrity harness over the full set:"
echo "  ~/alpasim/.venv/bin/python rl/check_teacher_labels.py \\"
echo "      --labels rl/teacher_labels --ade --ade-sample 400 \\"
echo "      --json reports/d1_label_integrity_full.json"
