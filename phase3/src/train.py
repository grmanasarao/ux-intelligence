from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .common import (
    MODELS_DIR,
    load_ontology,
    ontology_labels,
    set_seed,
)
from .data import load_annotations, split_by_participant, validate_ontology_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a UX-domain Transformer.")
    parser.add_argument("--data", default="phase3/data/annotations.csv")
    parser.add_argument("--output-dir", default="phase3/models/ux-transformer")
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "microsoft/deberta-v3-base"),
        help="HF encoder. Use bert-base-uncased if you specifically want vanilla BERT.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    return parser.parse_args()


def make_multihot(labels: list[str], label2id: dict[str, int]) -> list[float]:
    vector = [0.0] * len(label2id)
    for label in labels:
        if label not in label2id:
            raise ValueError(f"Unknown training label: {label}")
        vector[label2id[label]] = 1.0
    return vector


def make_dataset(frame, label2id):
    records = []
    for row in frame.itertuples(index=False):
        records.append({
            "text": row.text,
            "labels": make_multihot(row.labels, label2id),
            "participant_id": row.participant_id,
            "source_name": row.source_name,
        })
    return Dataset.from_list(records)


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_precision": float(
            precision_score(labels, predictions, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(labels, predictions, average="micro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(labels, predictions)),
    }


class WeightedTrainer(Trainer):
    """
    Optional weighted BCE loss to reduce domination by common labels.
    Weights are provided through model.config.pos_weight.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        pos_weight = getattr(model.config, "pos_weight", None)

        if pos_weight is not None:
            weight_tensor = torch.tensor(
                pos_weight, dtype=logits.dtype, device=logits.device
            )
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        loss = loss_fn(logits, labels.to(dtype=logits.dtype))
        return (loss, outputs) if return_outputs else loss


def calculate_pos_weight(train_frame, label2id):
    counts = np.zeros(len(label2id), dtype=np.float64)
    for labels in train_frame["labels"]:
        for label in labels:
            counts[label2id[label]] += 1

    total = max(1, len(train_frame))
    negatives = total - counts
    counts = np.maximum(counts, 1.0)

    # Cap weights so very rare labels do not destabilize training.
    weights = np.clip(negatives / counts, 1.0, 10.0)
    return weights.tolist()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")
    if args.bf16 and not torch.cuda.is_available():
        raise ValueError("--bf16 requires a CUDA-capable GPU.")
    if args.fp16 and not torch.cuda.is_available():
        raise ValueError("--fp16 requires a CUDA-capable GPU.")

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.max_length < 32:
        raise ValueError("--max-length should be at least 32.")

    ontology = load_ontology()
    df = load_annotations(args.data)
    validate_ontology_labels(df, ontology)

    all_labels = ontology_labels(ontology)
    label2id = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_df, validation_df, test_df = split_by_participant(
        df,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    train_dataset = make_dataset(train_df, label2id)
    validation_dataset = make_dataset(validation_df, label2id)
    test_dataset = make_dataset(test_df, label2id)

    datasets = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = datasets.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text", "participant_id", "source_name"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(all_labels),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    # Let the model carry the training-time class weights for reproducibility.
    model.config.pos_weight = calculate_pos_weight(train_df, label2id)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=25,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=True,
        seed=args.seed,
    )

    training_args = TrainingArguments(**training_kwargs)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Training device: {trainer.args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Unique participants: {df['participant_id'].nunique()}")
    print(f"Labels: {len(all_labels)}")

    trainer.train()

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized["test"],
        metric_key_prefix="test",
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": args.model_name,
        "num_labels": len(all_labels),
        "labels": all_labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "max_length": args.max_length,
        "seed": args.seed,
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "train_participants": sorted(train_df["participant_id"].unique().tolist()),
        "validation_participants": sorted(validation_df["participant_id"].unique().tolist()),
        "test_participants": sorted(test_df["participant_id"].unique().tolist()),
        "test_metrics": {
            key: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for key, value in test_metrics.items()
            if key.startswith("test_")
        },
    }

    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFinal test metrics:")
    for key, value in sorted(test_metrics.items()):
        if key.startswith("test_"):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
