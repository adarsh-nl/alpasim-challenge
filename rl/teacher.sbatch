#!/bin/bash
#SBATCH --job-name=dp-teacher
#SBATCH --partition=ps
#SBATCH --account=ps
#SBATCH --qos=research
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/teacher_%A_%a.out

set -euo pipefail

echo "=== host=$(hostname) shard=${SLURM_ARRAY_TASK_ID:-0} start=$(date) ==="

export ALPASIM_NUPLAN_ROOT=/datasets/eemcs/ps/alpasim-nuplan-2026
source /home/nanjaiyalathaa/alpasim/.venv/bin/activate

cd /home/nanjaiyalathaa/alpasim-challenge/rl
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

python run_teacher.py \
  --root "$ALPASIM_NUPLAN_ROOT" \
  --repo ./dp_repo \
  --ckpt ./dp_ckpt \
  --out  ./teacher_labels \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" \
  --n-shards "${N_SHARDS:-16}" \
  --batch-size 64 \
  --device cuda \
  --audit-every 500

echo "=== done=$(date) files=$(ls ./teacher_labels | wc -l) ==="
