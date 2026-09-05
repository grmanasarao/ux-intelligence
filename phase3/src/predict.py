from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run UX model inference.")
    parser.add_argument("--model", default="phase3/models/ux-transformer")
    parser.add_argument("--text", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def main():
    args = parse_args()

    if not args.text and not args.file:
        raise ValueError("Provide either --text or --file.")
    if args.text and args.file:
        raise ValueError("Use either --text or --file, not both.")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1.")

    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Train a model first."
        )

    text = args.text
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        text = path.read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(model.config, "max_position_embeddings", 256),
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits.detach().cpu().numpy()[0]

    probabilities = sigmoid(logits)
    labels = getattr(model.config, "id2label", {})
    ranked = sorted(
        [
            {
                "label": labels.get(int(idx), str(idx)),
                "probability": float(prob),
            }
            for idx, prob in enumerate(probabilities)
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )

    selected = [item for item in ranked if item["probability"] >= args.threshold][: args.top_k]

    result = {
        "text": text,
        "threshold": args.threshold,
        "predictions": selected,
        "device": str(device),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
