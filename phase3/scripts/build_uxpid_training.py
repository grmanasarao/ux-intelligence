from __future__ import annotations

import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "uxpid" / "uxpid_branch_dataset.csv"
MAPPING_FILE = BASE_DIR / "config" / "uxpid_label_mapping.json"
OUT_DIR = DATA_DIR / "uxpid" / "training"


def parse_labels(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split("||") if item.strip()]


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")

    mapping_cfg = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    topic_mapping = mapping_cfg.get("topic_mapping", {})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.jsonl"
    test_path = OUT_DIR / "test.jsonl"
    report_path = OUT_DIR / "training_data_report.json"

    split_rows = {"train": [], "test": []}
    topic_counts = {"train": {}, "test": {}}
    severity_counts = {"train": {}, "test": {}}
    sentiment_counts = {"train": {}, "test": {}}

    total_rows = 0
    skipped_no_text = 0
    skipped_no_task_label = 0
    unmapped_topic_occurrences = 0
    mapped_topic_occurrences = 0
    multi_label_rows = 0
    severity_rows = 0
    sentiment_rows = 0

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"branch_id", "split", "text", "topic_labels", "severity", "sentiment"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            text = (row.get("text") or "").strip()
            split = (row.get("split") or "").strip().lower()

            if not text:
                skipped_no_text += 1
                continue

            if split not in {"train", "test"}:
                continue

            mapped_topics = []
            for source_label in parse_labels(row.get("topic_labels", "")):
                spec = topic_mapping.get(source_label)
                if not spec:
                    raise ValueError(f"Topic label missing from mapping: {source_label}")

                task = spec.get("task")
                target = spec.get("target")

                if task == "ux_topic" and target:
                    mapped_topics.append(target)
                    mapped_topic_occurrences += 1
                elif task == "unmapped":
                    unmapped_topic_occurrences += 1

            mapped_topics = dedupe(mapped_topics)

            severity_raw = (row.get("severity") or "").strip()
            severity = None
            if severity_raw in {"1", "2", "3", "4", "5"}:
                severity = int(severity_raw)
                severity_rows += 1
                severity_counts[split][str(severity)] = (
                    severity_counts[split].get(str(severity), 0) + 1
                )

            sentiment = (row.get("sentiment") or "").strip().lower()
            if sentiment not in {"positive", "neutral", "negative"}:
                sentiment = None
            else:
                sentiment_rows += 1
                sentiment_counts[split][sentiment] = (
                    sentiment_counts[split].get(sentiment, 0) + 1
                )

            if mapped_topics:
                for label in mapped_topics:
                    topic_counts[split][label] = topic_counts[split].get(label, 0) + 1
                if len(mapped_topics) > 1:
                    multi_label_rows += 1

            # A row is usable for at least one supervised task if it has a mapped topic,
            # severity, or sentiment target. We keep missing targets as null.
            if not mapped_topics and severity is None and sentiment is None:
                skipped_no_task_label += 1
                continue

            record = {
                "branch_id": row["branch_id"],
                "split": split,
                "text": text,
                "targets": {
                    "ux_topics": mapped_topics,
                    "severity": severity,
                    "sentiment": sentiment,
                },
                "metadata": {
                    "thread_id": row.get("thread_id", ""),
                    "publication_year": row.get("publication_year", ""),
                    "branch_status": row.get("branch_status", ""),
                    "branch_type": row.get("branch_type", ""),
                },
            }
            split_rows[split].append(record)

    for split, rows in split_rows.items():
        output_path = train_path if split == "train" else test_path
        with output_path.open("w", encoding="utf-8") as f:
            for record in rows:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "dataset": "UXPID",
        "input_rows": total_rows,
        "output_rows": {
            "train": len(split_rows["train"]),
            "test": len(split_rows["test"]),
            "total": len(split_rows["train"]) + len(split_rows["test"]),
        },
        "skipped": {
            "no_text": skipped_no_text,
            "no_task_label": skipped_no_task_label,
        },
        "task_coverage": {
            "mapped_topic_occurrences": mapped_topic_occurrences,
            "unmapped_topic_occurrences": unmapped_topic_occurrences,
            "severity_rows": severity_rows,
            "sentiment_rows": sentiment_rows,
            "multi_label_rows": multi_label_rows,
        },
        "topic_counts": topic_counts,
        "severity_counts": severity_counts,
        "sentiment_counts": sentiment_counts,
        "notes": [
            "Official UXPID train/test split is preserved.",
            "UX topic targets use only labels explicitly mapped to task=ux_topic.",
            "Unmapped UXPID topic labels are excluded from the UX-topic target.",
            "Severity and sentiment are retained independently and may be null.",
            "No synthetic labels are created in this preparation step.",
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Training examples written : {len(split_rows['train'])}")
    print(f"Test examples written     : {len(split_rows['test'])}")
    print(f"Skipped (no task label)   : {skipped_no_task_label}")
    print(f"Mapped topic occurrences  : {mapped_topic_occurrences}")
    print(f"Unmapped topic occurrences: {unmapped_topic_occurrences}")
    print(f"Severity rows             : {severity_rows}")
    print(f"Sentiment rows            : {sentiment_rows}")
    print(f"Multi-label topic rows    : {multi_label_rows}")
    print(f"Train JSONL               : {train_path}")
    print(f"Test JSONL                : {test_path}")
    print(f"Report                    : {report_path}")


if __name__ == "__main__":
    main()
