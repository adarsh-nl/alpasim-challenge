#!/bin/bash
#SBATCH --job-name=rl-train-v2
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

# regenerate discriminating scene set (idempotent)
python select_discriminating_scenes.py --out train_scenes_discriminating.txt
# eval set: hold out a fixed slice of the discriminating scenes
head -12 train_scenes_discriminating.txt > eval_scenes_v2.txt
tail -n +13 train_scenes_discriminating.txt > train_scenes_v2.txt

# === v2 reward (DriveIRT-aligned, hard-gate) ===
export RL_REWARD=v2
export RL_FID_D0=0.56
export RL_BASE=0.5
export RL_PROG_GATE=0.8

cd $REPO
source .venv/bin/activate
cd $CHAL/rl
python train_loop.py \
  --iters 20 \
  --episodes_per_iter 16 \
  --scene_file train_scenes_v2.txt \
  --eval_scene_file eval_scenes_v2.txt \
  --eval_every 5 \
  --work /home/nanjaiyalathaa/alpasim/runs/rl-train-v2-$SLURM_JOB_ID \
  --lr 3e-4

echo "=== V2 TRAINING DONE ==="
grep EVAL /home/nanjaiyalathaa/alpasim/logs/rl-train-v2-$SLURM_JOB_ID.out || true
