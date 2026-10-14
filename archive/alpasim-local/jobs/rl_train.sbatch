#!/bin/bash
#SBATCH --job-name=rl-train
#SBATCH --partition=ps
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/nanjaiyalathaa/alpasim/logs/%x-%j.out
set -euo pipefail
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C.UTF-8
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST="8.9"
module load nvidia/cuda-12.4

REPO=/home/nanjaiyalathaa/alpasim
CHAL=/home/nanjaiyalathaa/alpasim-challenge
cd $CHAL/rl

# generate scene lists (idempotent)
python make_scene_lists.py

# reward weights (tunable knobs) — reasoned-guess starting point
export RL_W_PROGRESS=1.0
export RL_W_FIDELITY=0.5
export RL_W_SAFETY=5.0

# SMALL/FAST config: does it learn at all?
cd $REPO
source .venv/bin/activate
cd $CHAL/rl
python train_loop.py \
  --iters 20 \
  --episodes_per_iter 16 \
  --scene_file train_scenes.txt \
  --eval_scene_file eval_scenes.txt \
  --eval_every 5 \
  --work /home/nanjaiyalathaa/alpasim/runs/rl-train-$SLURM_JOB_ID \
  --lr 3e-4

echo "=== TRAINING DONE — curve: ==="
cat /home/nanjaiyalathaa/alpasim/runs/rl-train-$SLURM_JOB_ID/curve.json
