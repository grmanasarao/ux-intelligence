from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained UX classifier.")
    parser.add_argument("--model", default="phase3/models/ux-transformer")
    parser.add_argument("--data", default="phase3/data/annotations.csv")
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args()


def labels_to_vector(labels, label2id):
    vector = np.zeros(len(label2id), dtype=int)
    for label in labels:
        if label in label2id:
            vector[label2id[label]] = 1
    return vector


def main():
    args = parse_args()
    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    df = pd.read_csv(args.data)
    required = {"text", "labels"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model.eval()

    label2id = {str(k): int(v) for k, v in model.config.label2id.items()}
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true = []
    y_pred = []
    probability_rows = []

    for _, row in df.iterrows():
        text = str(row["text"])
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(model.config, "max_position_embeddings", 256),
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits.detach().cpu().numpy()[0]

        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        pred = (probs >= args.threshold).astype(int)

        labels = [x.strip().lower() for x in str(row["labels"]).split(";") if x.strip()]
        truth = labels_to_vector(labels, label2id)

        y_true.append(truth)
        y_pred.append(pred)
        probability_rows.append(probs)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    print(json.dumps({
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "samples": int(len(df)),
    }, indent=2))

    report = classification_report(
        y_true,
        y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True,
        zero_division=0,
    )
    report_path = model_dir / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved per-label classification report to {report_path}")


if __name__ == "__main__":
    main()
