from __future__ import annotations

import io
import re
from pathlib import Path

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
    sentiment_label,
    sentiment_overview,
    theme_results,
)
from src.embeddings import EmbeddingEngine
from src.feedback import save_feedback
from src.parser import parse_transcript
from src.reporting import (
    build_report,
    report_csv,
    report_json,
    report_markdown,
    report_pdf,
)


st.set_page_config(
    page_title="UX Research Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 0.8rem 1rem;
        border: 1px solid #D1D5DB;
        border-radius: 0.6rem;
        background: #FFFFFF;
    }
    .quote {
        padding: 0.75rem 1rem;
        border-left: 4px solid #4B5563;
        background: #F9FAFB;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading local embedding model...")
def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()


def parse_questions(raw: str) -> list[str]:
    questions = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s*", "", line).strip()
        if line:
            questions.append(line)
    return questions


def prepare_analysis(
    transcript_text: str,
    participant_name: str,
    source_name: str,
    study_purpose: str,
    questions: list[str],
):
    transcript = parse_transcript(
        transcript_text,
        participant_name=participant_name,
        source_name=source_name,
    )

    segments = build_segments([transcript])
    if not segments:
        raise ValueError(
            "The transcript was parsed, but there were not enough participant utterances "
            "to create analysis segments."
        )

    engine = get_embedding_engine()
    texts = [s["text"] for s in segments]
    segment_embeddings = engine.encode(texts, batch_size=16)

    labels, cluster_info = cluster_segments(segments, segment_embeddings)
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
            segment_embeddings,
        )
    else:
        question_tracking = []

    gaps = research_gaps(question_tracking)
    contradictions = contradiction_results(segments, labels, themes)
    coords = clustering_coordinates(segment_embeddings, labels)

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
        "embeddings": segment_embeddings,
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


def render_theme(theme: dict):
    with st.container(border=True):
        st.subheader(theme["name"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frequency", theme["frequency"])
        c2.metric("Participants", theme["participant_count"])
        c3.metric("Coverage", f"{theme['coverage_pct']}%")
        c4.metric("Evidence", f"{theme['evidence_strength']}/100")

        st.write(theme["description"])
        st.caption(
            f"Evidence level: {theme['evidence_level']} · "
            f"Negative response share: {theme['negative_share_pct']}%"
        )
        st.markdown(
            f'<div class="quote">“{theme["representative_quote"]}” '
            f'— {theme["representative_participant"]}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("View supporting evidence"):
            for evidence in theme["evidence"]:
                timestamp = f" [{evidence['timestamp']}]" if evidence.get("timestamp") else ""
                st.markdown(
                    f"**{evidence['participant']}**{timestamp}: "
                    f"“{evidence['quote']}”"
                )


def render_feedback(item_type: str, item_id: str, original_text: str):
    key = f"feedback_{item_type}_{item_id}"
    with st.expander("Researcher validation"):
        rating = st.radio(
            "Decision",
            ["Accept", "Reject", "Edit"],
            horizontal=True,
            key=f"{key}_rating",
        )
        edited = ""
        if rating == "Edit":
            edited = st.text_area(
                "Corrected version",
                value=original_text,
                key=f"{key}_edited",
            )
        comment = st.text_input(
            "Comment",
            key=f"{key}_comment",
            placeholder="Why is this finding correct, incorrect, or incomplete?",
        )
        if st.button("Save validation", key=f"{key}_save"):
            save_feedback({
                "target_type": item_type,
                "target_id": item_id,
                "original_text": original_text,
                "rating": rating.lower(),
                "edited_text": edited if rating == "Edit" else None,
                "comment": comment,
            })
            st.success("Validation saved to reports/feedback.json.")


def main():
    st.title("🔎 UX Research Intelligence")
    st.caption(
        "Phase 1 — local, evidence-driven UX research analysis without an LLM API."
    )

    with st.sidebar:
        st.header("Study setup")
        study_purpose = st.text_area(
            "Study purpose",
            placeholder="Example: Understand why users struggle with onboarding.",
            height=110,
        )
        participant_name = st.text_input("Participant name / ID", value="Participant 1")
        questions_raw = st.text_area(
            "Research questions",
            placeholder="One question per line",
            height=150,
        )
        questions = parse_questions(questions_raw)

        st.divider()
        st.info(
            "Phase 1 intentionally uses local NLP/ML. "
            "LLM/API integration is reserved for Phase 2."
        )

    st.subheader("1. Provide interview transcript")
    uploaded = st.file_uploader(
        "Upload one transcript",
        type=["txt", "vtt", "text"],
        accept_multiple_files=False,
        help="Multi-file/multi-participant upload is intentionally reserved for a later phase.",
    )

    default_text = """Moderator: Can you tell me about your experience using the product?
Participant: The product is useful, but I found the onboarding confusing.
Moderator: What was confusing?
Participant: I could not find the settings and I had to search around.
Moderator: How did that make you feel?
Participant: It was frustrating and it wasted a lot of time.
Moderator: Was anything easy?
Participant: Once I understood the workflow, the main task was simple and fast.
"""

    if uploaded is not None:
        transcript_text = uploaded.getvalue().decode("utf-8", errors="replace")
        source_name = uploaded.name
    else:
        transcript_text = st.text_area(
            "Or paste transcript text",
            value=default_text,
            height=280,
        )
        source_name = "pasted_transcript.txt"

    run = st.button("🚀 Run UX Analysis", type="primary", use_container_width=True)

    if run:
        if not transcript_text.strip():
            st.error("Please upload or paste a transcript.")
            return

        try:
            with st.spinner("Running local UX analysis..."):
                result = prepare_analysis(
                    transcript_text,
                    participant_name,
                    source_name,
                    study_purpose.strip() or "UX interview analysis",
                    questions,
                )
            st.session_state["analysis_result"] = result
            st.success("Analysis completed successfully.")
        except Exception as exc:
            st.error(f"Analysis could not be completed: {exc}")
            st.exception(exc)
            return

    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Provide a transcript and click **Run UX Analysis** to begin.")
        return

    report = result["report"]

    # Overview
    st.divider()
    st.header("2. Research Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Evidence segments", len(result["segments"]))
    c2.metric("Themes", len(result["themes"]))
    c3.metric("Pain points", len(result["pain_points"]))
    c4.metric("Research gaps", len(result["gaps"]))
    c5.metric("Contradictions", len(result["contradictions"]))

    st.subheader("Executive Summary")
    st.write(report["executive_summary"])

    # Sentiment
    st.header("3. User Response / Sentiment")
    sentiment = result["sentiment"]
    sentiment_df = pd.DataFrame({
        "Sentiment": list(sentiment["percentages"].keys()),
        "Percentage": list(sentiment["percentages"].values()),
    })
    if not sentiment_df.empty:
        fig = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Percentage",
            text="Percentage",
            title="Detected response distribution",
        )
        fig.update_layout(yaxis_title="Percentage", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Themes
    st.header("4. Themes — Frequency, Distribution & Evidence")
    if result["themes"]:
        theme_df = pd.DataFrame([
            {
                "Theme": t["name"],
                "Frequency": t["frequency"],
                "Participants": t["participant_count"],
                "Coverage %": t["coverage_pct"],
                "Evidence": t["evidence_strength"],
            }
            for t in result["themes"]
        ])
        st.dataframe(theme_df, use_container_width=True, hide_index=True)

        for theme in result["themes"]:
            render_theme(theme)
            render_feedback("theme", theme["theme_id"], theme["name"])
    else:
        st.warning("No themes were detected.")

    # Clustering
    st.header("5. Theme Clustering Visualization")
    cluster_info = result["cluster_info"]
    if cluster_info["k"] > 1:
        coords = result["coords"].copy()
        coords["Evidence"] = [s["quote"][:120] for s in result["segments"]]
        coords["Participant"] = [s["participant_name"] for s in result["segments"]]
        fig = px.scatter(
            coords,
            x="x",
            y="y",
            color="cluster",
            hover_data=["Participant", "Evidence"],
            title=f"Semantic evidence clusters ({cluster_info['k']} clusters)",
        )
        fig.update_layout(xaxis_title="Component 1", yaxis_title="Component 2")
        st.plotly_chart(fig, use_container_width=True)
        if cluster_info["silhouette"] is not None:
            st.caption(f"Clustering silhouette score: {cluster_info['silhouette']:.3f}")
    else:
        st.info("Not enough evidence segments for meaningful clustering.")

    # Pain points
    st.header("6. Pain Points & UX Severity")
    if result["pain_points"]:
        pain_df = pd.DataFrame([
            {
                "Pain point": p["pain_point"],
                "Frequency": p["frequency"],
                "Participants": p["participant_count"],
                "Severity": p["average_severity"],
                "Priority score": p["priority_score"],
                "Priority": p["priority"],
            }
            for p in result["pain_points"]
        ])
        st.dataframe(pain_df, use_container_width=True, hide_index=True)

        for item in result["pain_points"]:
            with st.expander(f"{item['priority']} — {item['pain_point']}"):
                st.write(
                    f"Average severity: **{item['average_severity']}/10** · "
                    f"Priority score: **{item['priority_score']}/10**"
                )
                for evidence in item["evidence"]:
                    st.markdown(
                        f"**{evidence['participant']}**: “{evidence['quote']}”"
                    )
    else:
        st.success("No strong pain-point signals were detected.")

    # Research questions
    st.header("7. Research Question Tracking")
    if result["question_tracking"]:
        for item in result["question_tracking"]:
            status = item["status"]
            icon = "🟢" if status == "Answered" else "🟡" if status == "Partially answered" else "🔴"
            with st.container(border=True):
                st.markdown(f"### {icon} {item['question']}")
                st.write(
                    f"**Status:** {status} · "
                    f"**Participant evidence:** {item['participant_count']} · "
                    f"**Top similarity:** {item['top_similarity']}"
                )
                for evidence in item["evidence"]:
                    st.markdown(
                        f"- **{evidence['participant']}** — "
                        f"“{evidence['quote']}” "
                        f"(similarity {evidence['similarity']})"
                    )
    else:
        st.info("Add research questions in the sidebar to enable question tracking.")

    # Gaps
    st.header("8. Research Gaps")
    if result["gaps"]:
        for gap in result["gaps"]:
            with st.container(border=True):
                st.markdown(f"**{gap['gap_type']}**")
                st.write(gap["description"])
                st.caption(
                    f"Participant evidence: {gap['participant_count']} · "
                    f"Suggested follow-up: {gap['suggested_follow_up']}"
                )
    else:
        st.success("No research gaps detected from the supplied research questions.")

    # Contradictions
    st.header("9. Contradiction Analysis")
    if result["contradictions"]:
        for contradiction in result["contradictions"]:
            with st.container(border=True):
                st.subheader(contradiction["topic"])
                st.write(contradiction["description"])
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Positive evidence**")
                    for e in contradiction["positive_evidence"]:
                        st.markdown(f"- {e['participant']}: “{e['quote']}”")
                with col2:
                    st.markdown("**Negative evidence**")
                    for e in contradiction["negative_evidence"]:
                        st.markdown(f"- {e['participant']}: “{e['quote']}”")
    else:
        st.success("No strong sentiment-based contradictions were detected.")

    # Recommendations
    st.header("10. Recommendation Prioritization")
    if result["recommendations"]:
        rec_df = pd.DataFrame([
            {
                "Recommendation": r["title"],
                "Impact": r["impact_score"],
                "Effort": r["effort"],
                "Priority": r["priority"],
                "Evidence strength": r["evidence_strength"],
            }
            for r in result["recommendations"]
        ])
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        fig = px.scatter(
            rec_df,
            x="Effort",
            y="Impact",
            size="Impact",
            text="Recommendation",
            color="Priority",
            title="Recommendation impact / effort view",
            category_orders={"Effort": ["Low", "Medium", "High"]},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

        for rec in result["recommendations"]:
            with st.container(border=True):
                st.subheader(rec["title"])
                st.write(rec["description"])
                st.caption(
                    f"Priority: {rec['priority']} · Impact: {rec['impact_score']}/10 · "
                    f"Effort: {rec['effort']} · Evidence: {rec['evidence_strength']}"
                )
                render_feedback("recommendation", rec["recommendation_id"], rec["description"])
    else:
        st.info("No recommendations could be generated from the detected pain points.")

    # Evidence explorer / semantic search
    st.header("11. Evidence Explorer")
    query = st.text_input(
        "Search the transcript by meaning",
        placeholder="Example: What frustrated users about onboarding?",
    )
    if query.strip():
        engine = get_embedding_engine()
        q_vector = engine.encode([query], batch_size=1)
        scores = cosine_similarity(q_vector, result["embeddings"])[0]
        top_indices = np.argsort(scores)[::-1][:5]

        for i in top_indices:
            segment = result["segments"][int(i)]
            st.markdown(
                f"**{segment['participant_name']}** · similarity **{scores[i]:.3f}**"
            )
            st.markdown(
                f'<div class="quote">“{segment["quote"]}”</div>',
                unsafe_allow_html=True,
            )

    # Export
    st.header("12. Exportable Research Report")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button(
            "Download JSON",
            data=report_json(report),
            file_name="ux_research_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download CSV",
            data=report_csv(report),
            file_name="ux_research_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "Download Markdown",
            data=report_markdown(report),
            file_name="ux_research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col4:
        st.download_button(
            "Download PDF",
            data=report_pdf(report),
            file_name="ux_research_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.caption(
        "Phase 1 uses local embeddings, clustering, deterministic scoring, and researcher validation. "
        "No LLM API call is made."
    )


if __name__ == "__main__":
    main()
