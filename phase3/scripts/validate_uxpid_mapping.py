from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
UXPID_DIR = DATA_DIR / "external" / "UXPID"
UXPID_ZIP = DATA_DIR / "external" / "UXPID.zip"
MAPPING_FILE = BASE_DIR / "config" / "uxpid_label_mapping.json"


def get_topic_counts() -> Counter:
    counts = Counter()

    if UXPID_DIR.exists() and (UXPID_DIR / "jsons").exists():
        for path in sorted((UXPID_DIR / "jsons").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            topics = data.get("topics") or {}
            counts.update(
                set(str(v).strip() for v in topics.values() if str(v).strip())
            )
        return counts

    if UXPID_ZIP.exists():
        with zipfile.ZipFile(UXPID_ZIP) as zf:
            names = sorted(
                n for n in zf.namelist()
                if n.startswith("UXPID/jsons/") and n.endswith(".json")
            )
            for name in names:
                data = json.loads(zf.read(name))
                topics = data.get("topics") or {}
                counts.update(
                    set(str(v).strip() for v in topics.values() if str(v).strip())
                )
        return counts

    raise FileNotFoundError(
        f"UXPID not found at {UXPID_DIR} or {UXPID_ZIP}"
    )


def main() -> None:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")

    config = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    mapping = config.get("topic_mapping", {})
    observed = get_topic_counts()

    missing = sorted(set(observed) - set(mapping))
    invalid_targets = []
    allowed_tasks = {"ux_topic", "signal", "unmapped"}

    for label, spec in mapping.items():
        if not isinstance(spec, dict):
            invalid_targets.append((label, "mapping must be an object"))
            continue
        task = spec.get("task")
        target = spec.get("target")
        if task not in allowed_tasks:
            invalid_targets.append((label, f"invalid task: {task}"))
        if task == "unmapped" and target is not None:
            invalid_targets.append((label, "unmapped label must have null target"))
        if task != "unmapped" and not target:
            invalid_targets.append((label, "mapped label must have a target"))

    if missing or invalid_targets:
        print("VALIDATION FAILED")
        if missing:
            print("\nMissing observed labels:")
            for label in missing:
                print(f"  - {label}")
        if invalid_targets:
            print("\nInvalid mapping entries:")
            for label, reason in invalid_targets:
                print(f"  - {label}: {reason}")
        raise SystemExit(1)

    mapped = Counter()
    for label, count in observed.items():
        task = mapping[label]["task"]
        target = mapping[label]["target"]
        mapped[f"{task}:{target}"] += count

    total_topic_occurrences = sum(observed.values())
    mapped_occurrences = sum(
        count for key, count in mapped.items()
        if not key.startswith("unmapped:")
    )
    unmapped_occurrences = total_topic_occurrences - mapped_occurrences

    print("UXPID mapping validation PASSED")
    print(f"Observed topic classes : {len(observed)}")
    print(f"Mapped topic classes   : {sum(1 for v in mapping.values() if v['task'] != 'unmapped')}")
    print(f"Unmapped topic classes : {sum(1 for v in mapping.values() if v['task'] == 'unmapped')}")
    print(f"Topic occurrences      : {total_topic_occurrences}")
    print(f"Mapped occurrences     : {mapped_occurrences}")
    print(f"Unmapped occurrences   : {unmapped_occurrences}")

    print("\nMapped target counts:")
    for key, count in mapped.most_common():
        print(f"  {count:5d}  {key}")


if __name__ == "__main__":
    main()
