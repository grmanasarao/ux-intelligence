from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import UXDataset, collate_batch
from model import MultiTaskConfig, UXMultiTaskModel


DEFAULT_DATA = ROOT / "data" / "uxpid" / "training"
DEFAULT_OUTPUT = ROOT / "models" / "ux_multi_task_deberta"

TOPIC_LABELS = [
    "accessibility",
    "content_clarity",
    "error_recovery",
    "feedback",
    "learnability",
    "performance",
    "trust",
    "workflow",
]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_validation(dataset: UXDataset, validation_fraction: float, seed: int):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * validation_fraction))
    return Subset(dataset, indices[val_count:]), Subset(dataset, indices[:val_count])


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "topic_targets": batch["topic_targets"].to(device),
        "severity_targets": batch["severity_targets"].to(device),
        "sentiment_targets": batch["sentiment_targets"].to(device),
    }


def evaluate(model, loader, device):
    model.eval()

    topic_true, topic_pred = [], []
    sev_true, sev_pred = [], []
    sent_true, sent_pred = [], []
    loss_sum = 0.0
    loss_steps = 0

    with torch.no_grad():
        for batch in loader:
            out = model(**move_batch(batch, device))

            if out["loss"] is not None:
                loss_sum += float(out["loss"].item())
                loss_steps += 1

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

    metrics = {"loss": loss_sum / max(loss_steps, 1)}

    if topic_true:
        yt = np.vstack(topic_true)
        yp = np.vstack(topic_pred)
        metrics["topic_micro_f1"] = float(
            f1_score(yt, yp, average="micro", zero_division=0)
        )
        metrics["topic_macro_f1"] = float(
            f1_score(yt, yp, average="macro", zero_division=0)
        )

    if sev_true:
        metrics["severity_accuracy"] = float(accuracy_score(sev_true, sev_pred))
        metrics["severity_mae"] = float(mean_absolute_error(sev_true, sev_pred))

    if sent_true:
        metrics["sentiment_accuracy"] = float(
            accuracy_score(sent_true, sent_pred)
        )
        metrics["sentiment_macro_f1"] = float(
            f1_score(sent_true, sent_pred, average="macro", zero_division=0)
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full training")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        use_amp = True
        amp_dtype = torch.float16
    else:
        print("CPU mode: real training will be slow.")
        use_amp = False
        amp_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    full_train = UXDataset(
        args.data_dir / "train.jsonl",
        TOPIC_LABELS,
        SENTIMENT_LABELS,
        tokenizer,
        max_length=args.max_length,
    )
    test_ds = UXDataset(
        args.data_dir / "test.jsonl",
        TOPIC_LABELS,
        SENTIMENT_LABELS,
        tokenizer,
        max_length=args.max_length,
    )

    train_ds, val_ds = split_train_validation(
        full_train, args.validation_fraction, args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    config = MultiTaskConfig(
        encoder_name=args.model_name,
        num_topics=len(TOPIC_LABELS),
        topic_labels=TOPIC_LABELS,
        num_severities=5,
        num_sentiments=len(SENTIMENT_LABELS),
        sentiment_labels=SENTIMENT_LABELS,
    )
    model = UXMultiTaskModel(config).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    effective_steps = max(
        1,
        (len(train_loader) * args.epochs + args.gradient_accumulation - 1)
        // args.gradient_accumulation,
    )
    if args.max_steps > 0:
        effective_steps = min(effective_steps, args.max_steps)

    warmup_steps = int(effective_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, effective_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader, 1):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                out = model(**move_batch(batch, device))
                loss = out["loss"]

            if loss is None:
                continue

            loss = loss / args.gradient_accumulation
            scaler.scale(loss).backward()

            if batch_index % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    print(
                        f"epoch={epoch} step={global_step}/{effective_steps} "
                        f"loss={loss.item() * args.gradient_accumulation:.4f}"
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        val_metrics = evaluate(model, val_loader, device)
        print("Validation:")
        print(json.dumps(val_metrics, indent=2))

        score = val_metrics.get("topic_macro_f1", 0.0)
        history.append({"epoch": epoch, **val_metrics})

        if score > best_score:
            best_score = score
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            (args.output_dir / "labels.json").write_text(
                json.dumps(
                    {
                        "topic_labels": TOPIC_LABELS,
                        "severity_labels": [1, 2, 3, 4, 5],
                        "sentiment_labels": SENTIMENT_LABELS,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (args.output_dir / "validation_metrics.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            print(f"Saved best model: {args.output_dir}")

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if not (args.output_dir / "config.json").exists():
        raise RuntimeError("No checkpoint was saved during training.")

    # Final, untouched UXPID test evaluation.
    best_model = UXMultiTaskModel.from_pretrained(args.output_dir).to(device)
    test_metrics = evaluate(best_model, test_loader, device)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    print("Final UXPID test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Model directory: {args.output_dir}")


if __name__ == "__main__":
    main()
