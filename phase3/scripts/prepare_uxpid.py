from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = BASE_DIR / "data" / "external"
UXPID_DIR = EXTERNAL_DIR / "UXPID"
UXPID_ZIP = EXTERNAL_DIR / "UXPID.zip"

OUT_DIR = BASE_DIR / "data" / "uxpid"
OUT_CSV = OUT_DIR / "uxpid_branch_dataset.csv"
OUT_REPORT = OUT_DIR / "uxpid_label_report.json"


def read_split_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def locate_dataset() -> tuple[str, Path | zipfile.ZipFile]:
    if UXPID_DIR.exists() and (UXPID_DIR / "jsons").exists():
        return "folder", UXPID_DIR
    if UXPID_ZIP.exists():
        return "zip", zipfile.ZipFile(UXPID_ZIP, "r")
    raise FileNotFoundError(
        f"Could not find UXPID in {UXPID_DIR} or {UXPID_ZIP}."
    )


def build_branch_text(content: list[dict]) -> str:
    parts = []
    for item in content:
        body = str(item.get("comment_body") or "").strip()
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def normalise_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_type, source = locate_dataset()

    if source_type == "folder":
        root = source
        train_ids = read_split_file(root / "splits" / "train_branches.txt")
        test_ids = read_split_file(root / "splits" / "test_branches.txt")
        json_paths = sorted((root / "jsons").glob("*.json"))

        def load_json(p: Path) -> dict:
            return json.loads(p.read_text(encoding="utf-8"))

        items = ((p.stem, load_json(p)) for p in json_paths)
    else:
        zf = source
        train_ids = {
            x.strip()
            for x in zf.read("UXPID/splits/train_branches.txt")
            .decode("utf-8")
            .splitlines()
            if x.strip()
        }
        test_ids = {
            x.strip()
            for x in zf.read("UXPID/splits/test_branches.txt")
            .decode("utf-8")
            .splitlines()
            if x.strip()
        }
        json_names = sorted(
            n for n in zf.namelist()
            if n.startswith("UXPID/jsons/") and n.endswith(".json")
        )

        def load_json(name: str) -> dict:
            return json.loads(zf.read(name))

        items = ((Path(n).stem, load_json(n)) for n in json_names)

    rows = []
    topic_counts = Counter()
    sentiment_counts = Counter()
    severity_counts = Counter()
    split_counts = Counter()
    missing_sentiment = 0

    for branch_id, data in items:
        metadata = data.get("metadata") or {}
        analysis = data.get("analysis") or {}
        topics = data.get("topics") or {}
        content = data.get("content") or []

        topic_labels = sorted(
            set(str(v).strip() for v in topics.values() if str(v).strip())
        )
        sentiment = str(analysis.get("overall_branch_sentiment") or "").strip()
        severity = str(analysis.get("severity_expectation_level") or "").strip()

        if branch_id in train_ids:
            split = "train"
        elif branch_id in test_ids:
            split = "test"
        else:
            split = "unspecified"

        pain_keywords = normalise_list(analysis.get("pain_keywords"))
        feature_keywords = normalise_list(analysis.get("feature_keywords"))
        gain_keywords = normalise_list(analysis.get("gain_keywords"))

        if sentiment:
            sentiment_counts[sentiment] += 1
        else:
            missing_sentiment += 1

        if severity:
            severity_counts[severity] += 1

        topic_counts.update(topic_labels)
        split_counts[split] += 1

        rows.append(
            {
                "branch_id": branch_id,
                "thread_id": metadata.get("thread_id", ""),
                "publication_year": metadata.get("publication_year", ""),
                "branch_status": metadata.get("branch_status", ""),
                "branch_type": metadata.get("branch_type", ""),
                "split": split,
                "text": build_branch_text(content),
                "topic_labels": " || ".join(topic_labels),
                "sentiment": sentiment,
                "severity": severity,
                "pain_keywords": " || ".join(pain_keywords),
                "feature_keywords": " || ".join(feature_keywords),
                "gain_keywords": " || ".join(gain_keywords),
                "insight_summary": str(analysis.get("insight_summary") or "").strip(),
                "user_expectations": str(analysis.get("user_expectations") or "").strip(),
            }
        )

    if not rows:
        raise RuntimeError("No UXPID JSON branches were found.")

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "dataset": "UXPID",
        "branches": len(rows),
        "splits": dict(split_counts),
        "topic_class_count": len(topic_counts),
        "topic_counts": dict(topic_counts.most_common()),
        "sentiment_counts_non_missing": dict(sentiment_counts),
        "missing_sentiment_count": missing_sentiment,
        "severity_counts": dict(severity_counts),
        "notes": [
            "Supplied UXPID train/test split is preserved.",
            "Topic labels are branch-level and multi-label.",
            "Sentiment is missing for many branches and is not imputed.",
            "This step prepares data only; it does not create human gold labels.",
        ],
    }
    OUT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if source_type == "zip":
        source.close()

    print(f"UXPID branches prepared : {len(rows)}")
    print(f"Train branches          : {split_counts.get('train', 0)}")
    print(f"Test branches           : {split_counts.get('test', 0)}")
    print(f"Topic classes           : {len(topic_counts)}")
    print(f"Missing sentiment       : {missing_sentiment}")
    print(f"Output CSV              : {OUT_CSV}")
    print(f"Report                  : {OUT_REPORT}")


if __name__ == "__main__":
    main()
