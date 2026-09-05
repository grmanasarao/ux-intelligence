from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional


MODERATOR_NAMES = {
    "moderator", "interviewer", "researcher", "facilitator",
    "mod", "int", "interviewer/researcher", "researcher/moderator"
}


@dataclass
class Utterance:
    speaker: str
    text: str
    timestamp: Optional[str] = None


@dataclass
class Transcript:
    participant_id: str
    participant_name: str
    source_name: str
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def participant_utterances(self) -> list[Utterance]:
        return [u for u in self.utterances if u.speaker == "participant"]

    @property
    def full_text(self) -> str:
        return "\n".join(u.text for u in self.participant_utterances)


def _normalize_speaker(speaker: str) -> str:
    value = re.sub(r"\s+", " ", speaker.strip().lower())
    if value in MODERATOR_NAMES or value.startswith(("mod", "int", "researcher", "facilitator")):
        return "moderator"
    return "participant"


def _clean_vtt_line(line: str) -> str:
    line = re.sub(r"<[^>]+>", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def parse_transcript(raw_text: str, participant_name: str, source_name: str = "transcript.txt") -> Transcript:
    if not raw_text or not raw_text.strip():
        raise ValueError("Transcript is empty.")

    participant_name = participant_name.strip() or "Participant 1"
    participant_id = re.sub(r"[^A-Za-z0-9_-]+", "_", participant_name).strip("_") or "participant_1"

    utterances: list[Utterance] = []

    timestamped = re.compile(
        r"^\[?(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\]?\s+([^:]{1,60}):\s*(.+)$"
    )
    labelled = re.compile(r"^([^:]{1,60}):\s*(.+)$")

    for raw_line in raw_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()
        if upper in {"WEBVTT", "NOTE", "STYLE", "REGION"}:
            continue
        if "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue

        line = _clean_vtt_line(line)
        if not line:
            continue

        match = timestamped.match(line)
        if match:
            timestamp, speaker, text = match.groups()
            utterances.append(
                Utterance(_normalize_speaker(speaker), text.strip(), timestamp)
            )
            continue

        match = labelled.match(line)
        if match:
            speaker, text = match.groups()
            utterances.append(
                Utterance(_normalize_speaker(speaker), text.strip())
            )
            continue

        # Untagged lines are treated as participant speech.
        utterances.append(Utterance("participant", line))

    if not utterances:
        raise ValueError("No readable transcript lines were found.")

    return Transcript(
        participant_id=participant_id,
        participant_name=participant_name,
        source_name=Path(source_name).name,
        utterances=utterances,
    )
