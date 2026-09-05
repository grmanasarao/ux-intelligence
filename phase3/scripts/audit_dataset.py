from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from ..src.common import load_ontology, normalize_labels, ontology_labels
from ..src.data import load_annotations, validate_ontology_labels


def main():
    parser = argparse.ArgumentParser(description="Audit UX annotation quality.")
    parser.add_argument("--data", default="phase3/data/annotations.csv")
    args = parser.parse_args()

    ontology = load_ontology()
    df = load_annotations(args.data)
    validate_ontology_labels(df, ontology)

    counts = Counter(label for labels in df["labels"] for label in labels)

    print(f"Rows: {len(df)}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Sources: {df['source_name'].nunique()}")
    print("\nLabel counts:")
    for label in ontology_labels(ontology):
        print(f"{label:35s} {counts.get(label, 0)}")

    rare = [label for label in ontology_labels(ontology) if 0 < counts.get(label, 0) < 5]
    unused = [label for label in ontology_labels(ontology) if counts.get(label, 0) == 0]

    print("\nWarnings:")
    if rare:
        print("Rare labels (<5 examples):", ", ".join(rare))
    if unused:
        print("Unused ontology labels:", ", ".join(unused))
    if not rare and not unused:
        print("No label-coverage warnings.")
