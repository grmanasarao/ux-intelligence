from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src.analysis import (
    build_segments,
    cluster_segments,
    clustering_coordinates,
    contradiction_results,
    pain_points,
    recommendation_results,
    research_gaps,
    research_question_results,
    sentiment_overview,
    theme_results,
)
from src.embeddings import EmbeddingEngine
from src.feedback import save_feedback
from src.parser import parse_transcript
from src.openai_synthesis import DEFAULT_MODEL, generate_synthesis
from src.reporting import report_csv, report_json, report_markdown, report_pdf, build_report


st.set_page_config(
    page_title="UX Research Intelligence — Phase 2",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the local embedding model...")
def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()


def parse_questions(raw: str) -> list[str]:
    questions: list[str] = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", line).strip()
        if line:
            questions.append(line)
    return questions


def run_phase1_pipeline(
    transcript_text: str,
    participant_name: str,
    source_name: str,
    study_purpose: str,
    questions: list[str],
) -> dict:
    transcript = parse_transcript(
        transcript_text,
        participant_name=participant_name,
        source_name=source_name,
    )

    segments = build_segments([transcript])
    if not segments:
        raise ValueError("Not enough participant speech was found for analysis.")

    engine = get_embedding_engine()
    segment_texts = [segment["text"] for segment in segments]
    embeddings = engine.encode(segment_texts, batch_size=16)

    labels, cluster_info = cluster_segments(segments, embeddings)
    themes = theme_results(segments, labels)
    pains = pain_points(segments, labels, themes)
    recommendations = recommendation_results(pains, themes)
    sentiment = sentiment_overview(segments)

    if questions:
        question_embeddings = engine.encode(questions, batch_size=16)
        question_tracking = research_question_results(
            questions,
            question_embeddings,
            segments,
            embeddings,
        )
    else:
        question_tracking = []

    gaps = research_gaps(question_tracking)
    contradictions = contradiction_results(segments, labels, themes)
    coords = clustering_coordinates(embeddings, labels)

    report = build_report(
        study_purpose=study_purpose,
        research_questions=questions,
        participant_name=participant_name,
        source_name=source_name,
        segments=segments,
        themes=themes,
        pain_points=pains,
        questions=question_tracking,
        gaps=gaps,
        contradictions=contradictions,
        recommendations=recommendations,
        sentiment=sentiment,
    )

    return {
        "transcript": transcript,
        "segments": segments,
        "embeddings": embeddings,
        "labels": labels,
        "cluster_info": cluster_info,
        "themes": themes,
        "pain_points": pains,
        "recommendations": recommendations,
        "sentiment": sentiment,
        "question_tracking": question_tracking,
        "gaps": gaps,
        "contradictions": contradictions,
        "coords": coords,
        "report": report,
    }


def save_validation(target_type: str, target_id: str, original_text: str) -> None:
    with st.expander("Researcher validation", expanded=False):
        decision = st.radio(
            "Decision",
            ["Accept", "Reject", "Edit"],
            horizontal=True,
            key=f"{target_type}_{target_id}_decision",
        )
        edited = st.text_area(
            "Correction",
            value=original_text,
            key=f"{target_type}_{target_id}_edited",
            disabled=decision != "Edit",
        )
        comment = st.text_input(
            "Comment",
            key=f"{target_type}_{target_id}_comment",
        )

        if st.button(
            "Save validation",
            key=f"{target_type}_{target_id}_save",
        ):
            save_feedback(
                {
                    "target_type": target_type,
                    "target_id": target_id,
                    "original_text": original_text,
                    "decision": decision.lower(),
                    "edited_text": edited if decision == "Edit" else None,
                    "comment": comment,
                }
            )
            st.success("Validation saved.")


def main() -> None:
    st.title("🔎 UX Research Intelligence")
    st.caption("Phase 2 — Phase 1 local analysis + optional OpenAI synthesis")

    with st.sidebar:
        st.header("Study setup")
        study_purpose = st.text_area(
            "Study purpose",
            value="Understand the main user needs, pain points, and usability issues.",
        )
        participant_name = st.text_input(
            "Participant name / ID",
            value="Participant 1",
        )
        questions_raw = st.text_area(
            "Research questions",
            placeholder="One research question per line",
        )
        questions = parse_questions(questions_raw)

        st.divider()
        st.header("OpenAI")
        enable_openai = st.checkbox(
            "Enable OpenAI synthesis",
            value=False,
        )
        model = st.text_input(
            "OpenAI model",
            value=DEFAULT_MODEL,
            disabled=not enable_openai,
        )

        if enable_openai and not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY is missing from the current environment.")

    default_text = """Moderator: Tell me about your experience using the product.
Participant: The product is useful, but onboarding was confusing.
Moderator: What was confusing?
Participant: I could not find the settings and had to search around.
Moderator: How did that feel?
Participant: It was frustrating and wasted time.
Moderator: Was anything easy?
Participant: Once I understood the workflow, the main task was simple and fast.
"""

    uploaded = st.file_uploader(
        "Upload one transcript",
        type=["txt", "vtt", "text"],
    )

    if uploaded:
        transcript_text = uploaded.getvalue().decode("utf-8", errors="replace")
        source_name = uploaded.name
    else:
        transcript_text = st.text_area(
            "Or paste transcript",
            value=default_text,
            height=250,
        )
        source_name = "pasted_transcript.txt"

    if st.button("🚀 Run analysis", type="primary", use_container_width=True):
        if not transcript_text.strip():
            st.error("Please upload or paste a transcript.")
            return

        try:
            with st.spinner("Running Phase 1 local analysis..."):
                result = run_phase1_pipeline(
                    transcript_text=transcript_text,
                    participant_name=participant_name,
                    source_name=source_name,
                    study_purpose=study_purpose,
                    questions=questions,
                )

            if enable_openai:
                with st.spinner("Running OpenAI synthesis..."):
                    result["openai_synthesis"] = generate_synthesis(
                        study_purpose=study_purpose,
                        research_questions=questions,
                        themes=result["themes"],
                        pain_points=result["pain_points"],
                        contradictions=result["contradictions"],
                        gaps=result["gaps"],
                        model=model.strip() or DEFAULT_MODEL,
                    )
            else:
                result["openai_synthesis"] = None

            st.session_state["phase2_result"] = result
            st.success("Analysis complete.")
        except Exception as exc:
            st.error(str(exc))
            return

    result = st.session_state.get("phase2_result")
    if not result:
        st.info("Run the analysis to see the Phase 1 and optional Phase 2 results.")
        return

    # Phase 2 output
    synthesis = result.get("openai_synthesis")
    if synthesis:
        st.divider()
        st.header("🤖 OpenAI Research Synthesis")

        st.subheader("Executive Summary")
        st.write(synthesis["executive_summary"])

        st.subheader("Key Findings")
        for index, finding in enumerate(synthesis["key_findings"], start=1):
            with st.container(border=True):
                st.markdown(f"**{index}. {finding['finding']}**")
                st.write(finding["why_it_matters"])
                st.caption(
                    f"Evidence theme: {finding['evidence_theme_id']} · "
                    f"Confidence: {finding['confidence']}"
                )
                save_validation(
                    "llm_finding",
                    f"finding_{index}",
                    finding["finding"],
                )

        st.subheader("Research Implications")
        for implication in synthesis["research_implications"]:
            st.markdown(f"- {implication}")

        st.subheader("LLM Recommendations")
        for index, recommendation in enumerate(
            synthesis["recommendations"],
            start=1,
        ):
            with st.container(border=True):
                st.markdown(
                    f"**{index}. {recommendation['recommendation']}**"
                )
                st.write(recommendation["rationale"])
                st.caption(
                    f"Theme: {recommendation['linked_theme_id']} · "
                    f"Priority: {recommendation['priority']}"
                )
                save_validation(
                    "llm_recommendation",
                    f"recommendation_{index}",
                    recommendation["recommendation"],
                )

        if synthesis["caveats"]:
            st.subheader("Caveats")
            for caveat in synthesis["caveats"]:
                st.warning(caveat)

    st.divider()
    st.header("📊 Phase 1 Results")

    summary_cols = st.columns(5)
    summary_cols[0].metric("Evidence segments", len(result["segments"]))
    summary_cols[1].metric("Themes", len(result["themes"]))
    summary_cols[2].metric("Pain points", len(result["pain_points"]))
    summary_cols[3].metric("Research gaps", len(result["gaps"]))
    summary_cols[4].metric("Contradictions", len(result["contradictions"]))

    st.subheader("Themes")
    if result["themes"]:
        theme_df = pd.DataFrame(
            [
                {
                    "Theme": theme["name"],
                    "Frequency": theme["frequency"],
                    "Participants": theme["participant_count"],
                    "Coverage %": theme["coverage_pct"],
                    "Evidence strength": theme["evidence_strength"],
                    "Evidence level": theme["evidence_level"],
                }
                for theme in result["themes"]
            ]
        )
        st.dataframe(theme_df, use_container_width=True, hide_index=True)

        for theme in result["themes"]:
            with st.container(border=True):
                st.markdown(f"### {theme['name']}")
                st.write(theme["description"])
                st.caption(
                    f"Evidence: {theme['evidence_strength']}/100 · "
                    f"{theme['evidence_level']} · "
                    f"Coverage: {theme['coverage_pct']}%"
                )
                st.markdown(
                    f"> “{theme['representative_quote']}” — "
                    f"{theme['representative_participant']}"
                )
                with st.expander("Supporting evidence"):
                    for evidence in theme["evidence"]:
                        st.markdown(
                            f"- **{evidence['participant']}**: "
                            f"“{evidence['quote']}”"
                        )

    st.subheader("User Response / Sentiment")
    sentiment_percentages = result["sentiment"]["percentages"]
    if sentiment_percentages:
        df = pd.DataFrame(
            {
                "Sentiment": list(sentiment_percentages),
                "Percentage": list(sentiment_percentages.values()),
            }
        )
        fig = px.bar(df, x="Sentiment", y="Percentage", text="Percentage")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pain Points & Severity")
    if result["pain_points"]:
        pain_df = pd.DataFrame(
            [
                {
                    "Pain point": item["pain_point"],
                    "Frequency": item["frequency"],
                    "Participants": item["participant_count"],
                    "Severity": item["average_severity"],
                    "Priority score": item["priority_score"],
                    "Priority": item["priority"],
                }
                for item in result["pain_points"]
            ]
        )
        st.dataframe(pain_df, use_container_width=True, hide_index=True)

    st.subheader("Theme Clustering")
    if result["cluster_info"]["k"] > 1:
        coords = result["coords"].copy()
        coords["Participant"] = [
            segment["participant_name"] for segment in result["segments"]
        ]
        coords["Quote"] = [
            segment["quote"][:120] for segment in result["segments"]
        ]
        fig = px.scatter(
            coords,
            x="x",
            y="y",
            color="cluster",
            hover_data=["Participant", "Quote"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Research Questions")
    if result["question_tracking"]:
        for item in result["question_tracking"]:
            st.markdown(
                f"**{item['status']}** — {item['question']} "
                f"(evidence participants: {item['participant_count']})"
            )
    else:
        st.caption("No research questions were supplied.")

    st.subheader("Research Gaps")
    for item in result["gaps"]:
        st.warning(
            f"{item['gap_type']}: {item['description']} — "
            f"{item['suggested_follow_up']}"
        )

    st.subheader("Contradictions")
    if result["contradictions"]:
        for item in result["contradictions"]:
            with st.container(border=True):
                st.markdown(f"**{item['topic']}**")
                st.write(item["description"])
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Positive evidence**")
                    for evidence in item["positive_evidence"]:
                        st.markdown(
                            f"- {evidence['participant']}: "
                            f"“{evidence['quote']}”"
                        )
                with col_b:
                    st.markdown("**Negative evidence**")
                    for evidence in item["negative_evidence"]:
                        st.markdown(
                            f"- {evidence['participant']}: "
                            f"“{evidence['quote']}”"
                        )

    st.subheader("Recommendation Prioritization")
    if result["recommendations"]:
        rec_df = pd.DataFrame(
            [
                {
                    "Recommendation": item["title"],
                    "Impact": item["impact_score"],
                    "Effort": item["effort"],
                    "Priority": item["priority"],
                    "Evidence strength": item["evidence_strength"],
                }
                for item in result["recommendations"]
            ]
        )
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.subheader("Semantic Evidence Explorer")
    query = st.text_input(
        "Search the interview by meaning",
        placeholder="Example: Why was onboarding frustrating?",
    )
    if query.strip():
        engine = get_embedding_engine()
        query_vector = engine.encode([query], batch_size=1)
        scores = cosine_similarity(query_vector, result["embeddings"])[0]
        for index in np.argsort(scores)[::-1][:5]:
            segment = result["segments"][int(index)]
            st.markdown(
                f"**{segment['participant_name']}** · "
                f"similarity {scores[index]:.3f}"
            )
            st.markdown(f"> “{segment['quote']}”")

    st.subheader("Downloads")
    report = result["report"]
    cols = st.columns(4)
    with cols[0]:
        st.download_button(
            "JSON",
            report_json(report),
            "ux_research_report.json",
            "application/json",
        )
    with cols[1]:
        st.download_button(
            "CSV",
            report_csv(report),
            "ux_research_report.csv",
            "text/csv",
        )
    with cols[2]:
        st.download_button(
            "Markdown",
            report_markdown(report),
            "ux_research_report.md",
            "text/markdown",
        )
    with cols[3]:
        st.download_button(
            "PDF",
            report_pdf(report),
            "ux_research_report.pdf",
            "application/pdf",
        )


if __name__ == "__main__":
    main()
