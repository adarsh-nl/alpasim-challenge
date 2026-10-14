#!/bin/bash
#SBATCH --job-name=nuplan-smoke
#SBATCH --partition=ps,main-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nanjaiyalathaa/alpasim/logs/%x-%j.out

set -x
hostname
nvidia-smi

export ALPASIM_NUPLAN_ROOT=/datasets/eemcs/ps/alpasim-nuplan-2026
export HF_HOME=/home/nanjaiyalathaa/data/hf-cache
export LC_ALL=C.UTF-8
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH

DRIVER_HOST=hpc-head1.ewi.utwente.nl
timeout 5 bash -c "</dev/tcp/$DRIVER_HOST/6789" \
  && echo "DRIVER REACHABLE" || echo "DRIVER NOT REACHABLE"

cd /home/nanjaiyalathaa/alpasim

ALPASIM_DRIVER_HOST=$DRIVER_HOST ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge_nuplan=dev \
  wizard.log_dir=/home/nanjaiyalathaa/alpasim/runs/smoke-$SLURM_JOB_ID
