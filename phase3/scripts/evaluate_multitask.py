from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import UXDataset, collate_batch
from model import UXMultiTaskModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=ROOT / "data" / "uxpid" / "training" / "test.jsonl",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    labels = json.loads((args.model_dir / "labels.json").read_text(encoding="utf-8"))
    topics = labels["topic_labels"]
    sentiments = labels["sentiment_labels"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = UXMultiTaskModel.from_pretrained(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = UXDataset(
        args.test_jsonl,
        topics,
        sentiments,
        tokenizer,
        max_length=args.max_length,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    topic_true, topic_pred = [], []
    sev_true, sev_pred = [], []
    sent_true, sent_pred = [], []

    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            probs = torch.sigmoid(out["topic_logits"]).cpu().numpy()
            targets = batch["topic_targets"].numpy()
            for target, prob in zip(targets, probs):
                valid = target >= 0
                if valid.any():
                    topic_true.append(target[valid])
                    topic_pred.append((prob[valid] >= 0.5).astype(np.float32))

            true = batch["severity_targets"].numpy()
            pred = out["severity_logits"].argmax(-1).cpu().numpy()
            valid = true >= 0
            sev_true.extend((true[valid] + 1).tolist())
            sev_pred.extend((pred[valid] + 1).tolist())

            true = batch["sentiment_targets"].numpy()
            pred = out["sentiment_logits"].argmax(-1).cpu().numpy()
            valid = true >= 0
            sent_true.extend(true[valid].tolist())
            sent_pred.extend(pred[valid].tolist())

    report = {}

    if topic_true:
        report["topic_micro_f1"] = float(
            f1_score(np.vstack(topic_true), np.vstack(topic_pred), average="micro", zero_division=0)
        )
        report["topic_macro_f1"] = float(
            f1_score(np.vstack(topic_true), np.vstack(topic_pred), average="macro", zero_division=0)
        )

    if sev_true:
        report["severity_accuracy"] = float(accuracy_score(sev_true, sev_pred))
        report["severity_mae"] = float(mean_absolute_error(sev_true, sev_pred))

    if sent_true:
        report["sentiment_accuracy"] = float(accuracy_score(sent_true, sent_pred))
        report["sentiment_macro_f1"] = float(
            f1_score(sent_true, sent_pred, average="macro", zero_division=0)
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
