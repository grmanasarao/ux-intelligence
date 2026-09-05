from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors


def build_report(
    study_purpose: str,
    research_questions: list[str],
    participant_name: str,
    source_name: str,
    segments: list[dict[str, Any]],
    themes: list[dict[str, Any]],
    pain_points: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    sentiment: dict[str, Any],
) -> dict[str, Any]:
    negative_count = sentiment["counts"].get("Negative", 0)
    top_theme = themes[0]["name"] if themes else "No dominant theme detected"

    executive_summary = (
        f"The study contains {len(segments)} evidence segments from "
        f"one participant ({participant_name}). "
        f"The strongest detected theme is '{top_theme}'. "
        f"{negative_count} evidence segments show negative user-response signals. "
        f"The analysis identified {len(themes)} themes, {len(pain_points)} pain-point groups, "
        f"{len(contradictions)} potential contradictions, and {len(gaps)} research gaps."
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_purpose": study_purpose,
        "participant": {
            "name": participant_name,
            "source": source_name,
        },
        "research_questions": research_questions,
        "executive_summary": executive_summary,
        "sentiment": sentiment,
        "themes": themes,
        "pain_points": pain_points,
        "question_tracking": questions,
        "research_gaps": gaps,
        "contradictions": contradictions,
        "recommendations": recommendations,
        "all_evidence": segments,
    }


def report_json(report: dict[str, Any]) -> bytes:
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")


def report_markdown(report: dict[str, Any]) -> bytes:
    lines = [
        "# UX Research Intelligence Report",
        "",
        f"**Generated:** {report['generated_at']}",
        "",
        "## Study Purpose",
        report["study_purpose"],
        "",
        "## Executive Summary",
        report["executive_summary"],
        "",
        "## Sentiment",
        f"- Positive: {report['sentiment']['percentages'].get('Positive', 0)}%",
        f"- Neutral: {report['sentiment']['percentages'].get('Neutral', 0)}%",
        f"- Negative: {report['sentiment']['percentages'].get('Negative', 0)}%",
        "",
        "## Themes",
    ]

    for theme in report["themes"]:
        lines += [
            f"### {theme['name']}",
            f"- Frequency: {theme['frequency']}",
            f"- Participant coverage: {theme['coverage_pct']}%",
            f"- Evidence strength: {theme['evidence_strength']} ({theme['evidence_level']})",
            f"- Representative quote: \"{theme['representative_quote']}\"",
            "",
        ]

    lines.append("## Pain Points")
    for item in report["pain_points"]:
        lines += [
            f"- **{item['pain_point']}** — priority {item['priority']} "
            f"({item['priority_score']}/10), {item['participant_count']} participant(s)",
        ]

    lines.append("")
    lines.append("## Research Gaps")
    for gap in report["research_gaps"]:
        lines.append(f"- **{gap['gap_type']}:** {gap['description']} — {gap['suggested_follow_up']}")

    lines.append("")
    lines.append("## Recommendations")
    for rec in report["recommendations"]:
        lines.append(f"- **{rec['title']}** — {rec['description']}")

    return "\n".join(lines).encode("utf-8")


def report_csv(report: dict[str, Any]) -> bytes:
    rows = []
    for theme in report["themes"]:
        rows.append({
            "type": "theme",
            "name": theme["name"],
            "frequency": theme["frequency"],
            "participant_count": theme["participant_count"],
            "evidence_strength": theme["evidence_strength"],
            "priority": "",
            "description": theme["description"],
        })
    for item in report["pain_points"]:
        rows.append({
            "type": "pain_point",
            "name": item["pain_point"],
            "frequency": item["frequency"],
            "participant_count": item["participant_count"],
            "evidence_strength": "",
            "priority": item["priority"],
            "description": f"Severity {item['average_severity']}/10",
        })
    for rec in report["recommendations"]:
        rows.append({
            "type": "recommendation",
            "name": rec["title"],
            "frequency": "",
            "participant_count": "",
            "evidence_strength": rec["evidence_strength"],
            "priority": rec["priority"],
            "description": rec["description"],
        })

    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def report_pdf(report: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11))
    story = [
        Paragraph("UX Research Intelligence Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"<b>Study Purpose:</b> {report['study_purpose']}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("<b>Executive Summary</b>", styles["Heading2"]),
        Paragraph(report["executive_summary"], styles["BodyText"]),
        Spacer(1, 10),
    ]

    story.append(Paragraph("Themes", styles["Heading2"]))
    theme_data = [["Theme", "Frequency", "Participants", "Evidence", "Level"]]
    for theme in report["themes"]:
        theme_data.append([
            theme["name"],
            str(theme["frequency"]),
            str(theme["participant_count"]),
            f"{theme['evidence_strength']}",
            theme["evidence_level"],
        ])

    if len(theme_data) == 1:
        theme_data.append(["No themes detected", "-", "-", "-", "-"])

    table = Table(theme_data, repeatRows=1, colWidths=[2.5*inch, .7*inch, .8*inch, .8*inch, 1.0*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Pain Points", styles["Heading2"]))
    for item in report["pain_points"]:
        story.append(Paragraph(
            f"<b>{item['pain_point']}</b> — {item['priority']} priority, "
            f"score {item['priority_score']}/10. "
            f"{item['participant_count']} participant(s), {item['frequency']} evidence segment(s).",
            styles["Small"],
        ))
        story.append(Spacer(1, 4))

    story.append(Paragraph("Research Gaps", styles["Heading2"]))
    for gap in report["research_gaps"]:
        story.append(Paragraph(
            f"<b>{gap['gap_type']}:</b> {gap['description']} "
            f"Follow-up: {gap['suggested_follow_up']}",
            styles["Small"],
        ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Recommendations", styles["Heading2"]))
    for rec in report["recommendations"]:
        story.append(Paragraph(
            f"<b>{rec['title']}</b> — {rec['priority']} priority. {rec['description']}",
            styles["Small"],
        ))

    doc.build(story)
    return buffer.getvalue()
