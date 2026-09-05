from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class Example:
    branch_id: str
    text: str
    topics: list[str]
    severity: int | None
    sentiment: str | None


class UXDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        topic_labels: list[str],
        sentiment_labels: list[str],
        tokenizer,
        max_length: int = 256,
    ):
        self.path = Path(path)
        self.topic_to_id = {x: i for i, x in enumerate(topic_labels)}
        self.sentiment_to_id = {x: i for i, x in enumerate(sentiment_labels)}
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: list[Example] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} of {self.path}"
                    ) from exc

                targets = obj.get("targets") or {}
                self.examples.append(
                    Example(
                        branch_id=str(obj["branch_id"]),
                        text=str(obj["text"]).strip(),
                        topics=[str(x) for x in (targets.get("ux_topics") or [])],
                        severity=targets.get("severity"),
                        sentiment=targets.get("sentiment"),
                    )
                )

        if not self.examples:
            raise ValueError(f"No examples found in {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        topic_target = torch.full(
            (len(self.topic_to_id),), -1.0, dtype=torch.float
        )
        valid_topic_labels = [
            topic for topic in ex.topics if topic in self.topic_to_id
        ]
        if valid_topic_labels:
            topic_target.fill_(0.0)
            for topic in valid_topic_labels:
                topic_target[self.topic_to_id[topic]] = 1.0

        severity_target = (
            torch.tensor(int(ex.severity) - 1, dtype=torch.long)
            if ex.severity in {1, 2, 3, 4, 5}
            else torch.tensor(-1, dtype=torch.long)
        )

        sentiment_target = torch.tensor(
            self.sentiment_to_id[ex.sentiment]
            if ex.sentiment in self.sentiment_to_id
            else -1,
            dtype=torch.long,
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "topic_targets": topic_target,
            "severity_targets": severity_target,
            "sentiment_targets": sentiment_target,
            "branch_id": ex.branch_id,
            "text": ex.text,
        }


def collate_batch(batch: list[dict]) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    n = len(batch)

    input_ids = torch.zeros((n, max_len), dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "topic_targets": torch.stack([x["topic_targets"] for x in batch]),
        "severity_targets": torch.stack([x["severity_targets"] for x in batch]),
        "sentiment_targets": torch.stack([x["sentiment_targets"] for x in batch]),
        "branch_ids": [x["branch_id"] for x in batch],
        "texts": [x["text"] for x in batch],
    }
