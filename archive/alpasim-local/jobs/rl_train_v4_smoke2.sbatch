#!/bin/bash
#SBATCH --job-name=rl-v4-smoke2
#SBATCH --partition=ps
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nanjaiyalathaa/alpasim/logs/%x-%j.out
set -euo pipefail
hostname; nvidia-smi --query-gpu=name --format=csv,noheader
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C.UTF-8
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST="8.9"
module load nvidia/cuda-12.4
REPO=/home/nanjaiyalathaa/alpasim
CHAL=/home/nanjaiyalathaa/alpasim-challenge
cd $CHAL/rl
python select_discriminating_scenes.py --out train_scenes_discriminating.txt
head -8 train_scenes_discriminating.txt > eval_scenes_v2.txt
tail -n +9 train_scenes_discriminating.txt > train_scenes_v2.txt
export RL_REWARD=v2
export RL_LOG_STD_INIT=-0.3
export RL_MEAN_BIAS=1.5
export RL_PERSTEP=1
cd $REPO
source .venv/bin/activate
cd $CHAL/rl
# LR bumped 3e-4 -> 3e-3; ppo_update now does 10 epochs, minibatch 64
python train_loop.py \
  --iters 3 \
  --episodes_per_iter 8 \
  --scene_file train_scenes_v2.txt \
  --eval_scene_file eval_scenes_v2.txt \
  --eval_every 99 \
  --work /home/nanjaiyalathaa/alpasim/runs/rl-v4smoke2-$SLURM_JOB_ID \
  --lr 3e-3
echo "=== V4 SMOKE2: did KL move with higher LR + more epochs? ==="
grep "iter " /home/nanjaiyalathaa/alpasim/logs/rl-v4-smoke2-$SLURM_JOB_ID.out
