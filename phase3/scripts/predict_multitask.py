from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import UXMultiTaskModel


DEFAULT_MODEL = ROOT / "models" / "ux_multi_task_deberta"


def predict_text(model, tokenizer, text: str, device, threshold: float):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    probs = torch.sigmoid(out["topic_logits"])[0].cpu().tolist()
    severity = int(out["severity_logits"].argmax(-1)[0].item()) + 1
    sentiment_id = int(out["sentiment_logits"].argmax(-1)[0].item())
    sentiment = model.config.sentiment_labels[sentiment_id]

    topics = [
        label
        for label, prob in zip(model.config.topic_labels, probs)
        if prob >= threshold
    ]

    # Guarantee at least one topic if all probabilities fall below threshold.
    if not topics and probs:
        topics = [model.config.topic_labels[int(max(range(len(probs)), key=probs.__getitem__))]]

    return {
        "text": text,
        "ux_topics": topics,
        "topic_scores": {
            label: round(float(prob), 4)
            for label, prob in zip(model.config.topic_labels, probs)
        },
        "severity": severity,
        "sentiment": sentiment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--text")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.text and not args.input_jsonl:
        raise ValueError("Provide --text or --input-jsonl")
    if args.input_jsonl and not args.output_jsonl:
        raise ValueError("--output-jsonl is required with --input-jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = UXMultiTaskModel.from_pretrained(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if args.text:
        print(
            json.dumps(
                predict_text(
                    model, tokenizer, args.text, device, args.threshold
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    with args.input_jsonl.open("r", encoding="utf-8") as f, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as out:
        for line in f:
            obj = json.loads(line)
            result = predict_text(
                model,
                tokenizer,
                str(obj.get("text", "")),
                device,
                args.threshold,
            )
            result["branch_id"] = obj.get("branch_id")
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote predictions: {args.output_jsonl}")


if __name__ == "__main__":
    main()
