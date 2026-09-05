from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict_multitask import predict_text
from transformers import AutoTokenizer
from model import UXMultiTaskModel
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="evidence_id")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")
        if args.text_column not in reader.fieldnames:
            raise ValueError(
                f"Missing text column '{args.text_column}'. Available: {reader.fieldnames}"
            )
        rows = list(reader)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = UXMultiTaskModel.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    for row in rows:
        pred = predict_text(
            model, tokenizer, row[args.text_column], device, args.threshold
        )
        row["phase3_predicted_topics"] = " | ".join(pred["ux_topics"])
        row["phase3_predicted_severity"] = str(pred["severity"])
        row["phase3_predicted_sentiment"] = pred["sentiment"]
        row["phase3_topic_scores"] = str(pred["topic_scores"])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["phase3_prediction"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV predictions: {args.output_csv}")


if __name__ == "__main__":
    main()
