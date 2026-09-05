from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..src.common import load_ontology, set_seed
from ..src.data import load_annotations, split_by_participant, validate_ontology_labels


def main():
    parser = argparse.ArgumentParser(
        description="Create leakage-safe train/validation/test CSV files."
    )
    parser.add_argument("--input", default="phase3/data/annotations.csv")
    parser.add_argument("--output-dir", default="phase3/data/splits")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    df = load_annotations(args.input)
    ontology = load_ontology()
    validate_ontology_labels(df, ontology)

    train, validation, test = split_by_participant(
        df,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train.to_csv(out / "train.csv", index=False)
    validation.to_csv(out / "validation.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    print("Participant-safe dataset split created.")
    print(f"Train: {len(train)} rows / {train['participant_id'].nunique()} participants")
    print(f"Validation: {len(validation)} rows / {validation['participant_id'].nunique()} participants")
    print(f"Test: {len(test)} rows / {test['participant_id'].nunique()} participants")


if __name__ == "__main__":
    main()
