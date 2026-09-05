#!/bin/bash
#SBATCH --job-name=ux-phase3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ux_phase3_%j.out
#SBATCH --error=logs/ux_phase3_%j.err

set -euo pipefail

source .venv/bin/activate

python -m pip install -r phase3/requirements.txt

python phase3/scripts/train_multitask.py \
  --epochs 3 \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --lr 2e-5 \
  --max-length 256 \
  --output-dir phase3/models/ux_multi_task_deberta
