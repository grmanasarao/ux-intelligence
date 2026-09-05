#!/bin/bash
#SBATCH --job-name=ux_transformer
#SBATCH --qos=bbgpu
#SBATCH --account=YOUR_PROJECT_CODE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=phase3/logs/%x-%j.out
#SBATCH --error=phase3/logs/%x-%j.err

set -euo pipefail

module purge
module load bluebear
module load bear-apps/2023a

# Adjust the Python/PyTorch module to the currently supported BlueBEAR version
# shown by: module spider PyTorch
module load PyTorch

python -V
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python -m pip install --user --upgrade \
  "transformers>=4.57,<5.0" \
  "datasets>=3.0,<5.0" \
  "evaluate>=0.4,<1.0" \
  "accelerate>=1.0,<2.0" \
  "scikit-learn>=1.5,<2.0" \
  "pandas>=2.2,<3.0" \
  "pyyaml>=6.0,<7.0" \
  "sentencepiece>=0.2,<1.0"

python -m phase3.src.train \
  --data phase3/data/annotations.csv \
  --output-dir phase3/models/ux-deberta-v3 \
  --model-name microsoft/deberta-v3-base \
  --epochs 4 \
  --learning-rate 2e-5 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --gradient-accumulation 2 \
  --max-length 256 \
  --bf16 \
  --gradient-checkpointing
