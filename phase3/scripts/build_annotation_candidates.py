from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from src.analysis import sentiment_label, sentiment_score
from src.parser import parse_transcript


def make_participant_id(filename: str, index: int) -> str:
    # One transcript = one participant for this Phase 3 dataset.
    # We can change this later if a file contains multiple participants.
    return f"P{index:03d}"


def suggest_labels(text: str) -> list[str]:
    """
    Generate conservative candidate labels from the UX ontology.

    IMPORTANT:
    These are only suggestions. They are NOT gold-standard labels.
    A human must review them before training.
    """
    t = text.lower()
    labels: list[str] = []

    groups = {
        "domain:discoverability": [
            "find", "found", "locate", "hidden",
            "where is", "search around", "hard to find"
        ],
        "issue:discoverability": [
            "find", "locate", "hidden",
            "where is", "search around", "hard to find"
        ],

        "domain:navigation": [
            "navigate", "navigation", "menu", "click",
            "where do i", "where should i"
        ],
        "issue:navigation": [
            "navigate", "navigation", "menu", "click",
            "where do i", "where should i"
        ],

        "domain:performance": [
            "slow", "loading", "load", "lag", "wait", "freeze"
        ],
        "issue:performance": [
            "slow", "loading", "load", "lag", "wait", "freeze"
        ],

        "domain:onboarding": [
            "onboarding", "getting started", "setup",
            "sign up", "signup"
        ],
        "issue:onboarding": [
            "onboarding", "getting started", "setup",
            "sign up", "signup"
        ],

        "domain:content_clarity": [
            "unclear", "instructions", "not clear",
            "don't understand", "do not understand"
        ],
        "issue:content_clarity": [
            "unclear", "instructions", "not clear",
            "don't understand", "do not understand"
        ],

        "domain:error_handling": [
            "error", "failed", "failure", "broken",
            "didn't work", "did not work"
        ],
        "issue:error_recovery": [
            "error", "failed", "failure", "broken",
            "try again", "didn't work", "did not work"
        ],

        "domain:workflow": [
            "workflow", "process", "task", "steps",
            "complete", "completed"
        ],
        "issue:workflow": [
            "workflow", "process", "task", "steps"
        ],
    }

    for label, keywords in groups.items():
        if any(keyword in t for keyword in keywords):
            labels.append(label)

    # Basic response label
    labels.append(
        f"response:{sentiment_label(sentiment_score(text)).lower()}"
    )

    # More specific response labels
    if any(keyword in t for keyword in [
        "confused", "confusing", "not sure", "unsure", "unclear"
    ]):
        labels.append("response:confusion")

    if any(keyword in t for keyword in [
        "frustrated", "frustrating", "annoying",
        "annoyed", "struggle", "stuck"
    ]):
        labels.append("response:frustration")

    if any(keyword in t for keyword in [
        "satisfied", "happy", "love", "liked",
        "great", "excellent", "smooth"
    ]):
        labels.append("response:satisfaction")

    # Outcome candidates
    if any(keyword in t for keyword in [
        "abandoned", "gave up", "quit", "left"
    ]):
        labels.append("outcome:abandonment")

    elif any(keyword in t for keyword in [
        "couldn't complete",
        "could not complete",
        "failed"
    ]):
        labels.append("outcome:task_failure")

    elif any(keyword in t for keyword in [
        "completed",
        "finished successfully",
        "worked successfully"
    ]):
        labels.append("outcome:task_success")

    elif any(keyword in t for keyword in [
        "try again",
        "again",
        "several times",
        "multiple times"
    ]):
        labels.append("outcome:repeat_attempt")

    elif any(keyword in t for keyword in [
        "wait",
        "waiting",
        "delay",
        "delayed"
    ]):
        labels.append("outcome:delay")

    elif any(keyword in t for keyword in [
        "frustrated",
        "confusing",
        "hard to find",
        "struggle",
        "stuck",
        "unclear",
        "difficult"
    ]):
        labels.append("outcome:task_friction")

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(labels))


def make_segments(
    utterances: Iterable,
    window: int = 3,
):
    """
    Convert participant speech into evidence segments.

    A small context window is used so that a short utterance such as
    'Yes, that was confusing' retains nearby context.
    """
    participant_utterances = [
        utterance
        for utterance in utterances
        if utterance.speaker == "participant"
    ]

    for i, utterance in enumerate(participant_utterances):
        start = max(0, i - window // 2)
        end = min(
            len(participant_utterances),
            i + window // 2 + 1,
        )

        context = " ".join(
            item.text
            for item in participant_utterances[start:end]
        ).strip()

        # Ignore extremely short segments.
        if len(context.split()) >= 5:
            yield (
                context,
                utterance.text,
                utterance.timestamp or "",
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build human-reviewable candidate UX annotations "
            "from TXT transcripts."
        )
    )

    parser.add_argument(
        "--input-dir",
        default="phase3/data/transcripts",
        help="Directory containing TXT transcripts.",
    )

    parser.add_argument(
        "--output",
        default="phase3/data/candidate_annotations.csv",
        help="Output candidate annotation CSV.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Transcript directory does not exist: {input_dir}"
        )

    transcript_files = sorted(input_dir.glob("*.txt"))

    if not transcript_files:
        raise FileNotFoundError(
            f"No .txt transcript files found in: {input_dir}"
        )

    rows = []

    for index, path in enumerate(transcript_files, start=1):
        participant_id = make_participant_id(
            path.name,
            index,
        )

        raw_text = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

        transcript = parse_transcript(
            raw_text=raw_text,
            participant_name=participant_id,
            source_name=path.name,
        )

        for segment_number, (
            context,
            quote,
            timestamp,
        ) in enumerate(
            make_segments(transcript.utterances),
            start=1,
        ):
            rows.append({
                "text": context,
                "participant_id": participant_id,
                "source_name": path.name,
                "evidence_id": (
                    f"{participant_id}_E{segment_number:03d}"
                ),
                "quote": quote,
                "timestamp": timestamp,
                "suggested_labels": ";".join(
                    suggest_labels(context)
                ),
                "gold_labels": "",
                "review_status": "pending",
                "review_comment": "",
            })

    if not rows:
        raise ValueError(
            "No participant evidence segments were found. "
            "Check the transcript speaker formatting."
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "text",
        "participant_id",
        "source_name",
        "evidence_id",
        "quote",
        "timestamp",
        "suggested_labels",
        "gold_labels",
        "review_status",
        "review_comment",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    participants = sorted(
        {row["participant_id"] for row in rows}
    )

    print(f"Transcripts processed : {len(transcript_files)}")
    print(f"Candidate segments    : {len(rows)}")
    print(f"Participants          : {len(participants)}")
    print(f"Output                : {output}")
    print()
    print("IMPORTANT:")
    print("1. suggested_labels are automated candidates only.")
    print("2. Review every row.")
    print("3. Put final human-approved labels in gold_labels.")
    print("4. Training will use gold_labels, not suggested_labels.")


if __name__ == "__main__":
    main()