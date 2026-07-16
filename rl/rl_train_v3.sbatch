#!/bin/bash
#SBATCH --job-name=rl-train-v3
#SBATCH --partition=ps
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
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

python select_discriminating_scenes.py --out train_scenes_discriminating.txt
head -12 train_scenes_discriminating.txt > eval_scenes_v2.txt
tail -n +13 train_scenes_discriminating.txt > train_scenes_v2.txt

# === THE FIX: wider exploration + per-step reward + bigger batch ===
export RL_REWARD=v2
export RL_LOG_STD_INIT=0.5      # wide exploration: m in ~[0.19, 0.97] (was [0.80,0.93])
export RL_MEAN_BIAS=1.0         # center prior lower so slowing is explored
export RL_PERSTEP=1             # dense per-step fidelity reward
export RL_FID_D0=0.56
export RL_BASE=0.5
export RL_PROG_GATE=0.8

cd $REPO
source .venv/bin/activate
cd $CHAL/rl
python train_loop.py \
  --iters 20 \
  --episodes_per_iter 24 \
  --scene_file train_scenes_v2.txt \
  --eval_scene_file eval_scenes_v2.txt \
  --eval_every 4 \
  --work /home/nanjaiyalathaa/alpasim/runs/rl-train-v3-$SLURM_JOB_ID \
  --lr 3e-4

echo "=== V3 DONE ==="
grep EVAL /home/nanjaiyalathaa/alpasim/logs/rl-train-v3-$SLURM_JOB_ID.out || true
