#!/bin/bash
set -euo pipefail

python -m phase3.src.train \
  --data phase3/data/annotations.csv \
  --output-dir phase3/models/ux-distilbert \
  --model-name distilbert/distilbert-base-uncased \
  --epochs 3 \
  --learning-rate 3e-5 \
  --train-batch-size 2 \
  --eval-batch-size 4 \
  --gradient-accumulation 4 \
  --max-length 192 \
  --gradient-checkpointing
