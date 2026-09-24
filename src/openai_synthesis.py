from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _get_secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    return default


OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

OPENAI_MODEL = _get_secret("OPENAI_MODEL")

if OPENAI_MODEL:
    os.environ["OPENAI_MODEL"] = OPENAI_MODEL
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

SYSTEM_INSTRUCTIONS = """You are an expert UX research synthesis assistant.

Analyze the structured evidence produced by a UX Research Intelligence system.

Non-negotiable rules:
1. Never invent participant facts, quotes, evidence, or study results.
2. Every finding must be traceable to supplied evidence or clearly labelled as an interpretation.
3. Do not generalize from one participant to all users.
4. Preserve contradictions rather than smoothing them away.
5. Treat confidence as evidence-dependent.
6. Recommendations must be grounded in the supplied findings.
7. Be concise, specific, and useful to UX researchers, designers, and product teams.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "key_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "evidence_theme_id": {"type": "string"},
                    "confidence": {"type": "string"},
                },
                "required": [
                    "finding",
                    "why_it_matters",
                    "evidence_theme_id",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
        "research_implications": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recommendation": {"type": "string"},
                    "rationale": {"type": "string"},
                    "linked_theme_id": {"type": "string"},
                    "priority": {"type": "string"},
                },
                "required": [
                    "recommendation",
                    "rationale",
                    "linked_theme_id",
                    "priority",
                ],
                "additionalProperties": False,
            },
        },
        "caveats": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "executive_summary",
        "key_findings",
        "research_implications",
        "recommendations",
        "caveats",
    ],
    "additionalProperties": False,
}


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in your environment and restart Streamlit."
        )
    return OpenAI(api_key=api_key)


def generate_synthesis(
    *,
    study_purpose: str,
    research_questions: list[str],
    themes: list[dict[str, Any]],
    pain_points: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    model: str | None = None,
) -> dict[str, Any]:
    client = get_client()
    selected_model = model or DEFAULT_MODEL

    payload = {
        "study_purpose": study_purpose,
        "research_questions": research_questions,
        "themes": [
            {
                "theme_id": theme["theme_id"],
                "name": theme["name"],
                "frequency": theme["frequency"],
                "participant_count": theme["participant_count"],
                "coverage_pct": theme["coverage_pct"],
                "evidence_strength": theme["evidence_strength"],
                "evidence_level": theme["evidence_level"],
                "evidence": theme["evidence"][:8],
            }
            for theme in themes
        ],
        "pain_points": pain_points,
        "contradictions": contradictions,
        "research_gaps": gaps,
    }

    response = client.responses.create(
        model=selected_model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=(
            "Synthesize the following structured UX research evidence.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "ux_research_synthesis",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        store=False,
    )

    if not response.output_text:
        raise RuntimeError("OpenAI returned an empty response.")

    try:
        result = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "OpenAI returned text that did not conform to the expected JSON structure."
        ) from exc

    return result
