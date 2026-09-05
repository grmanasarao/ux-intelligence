
from __future__ import annotations

from pathlib import Path
import json
import re
import html

import pandas as pd
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="UX Research Intelligence — Phase 3 Annotation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

CANDIDATE_FILE = DATA_DIR / "candidate_annotations.csv"
GOLD_FILE = DATA_DIR / "annotations.csv"
ONTOLOGY_FILE = CONFIG_DIR / "ux_ontology.json"


# ============================================================
# Helpers
# ============================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def load_ontology() -> dict:
    if not ONTOLOGY_FILE.exists():
        raise FileNotFoundError(f"Ontology file not found: {ONTOLOGY_FILE}")

    with ONTOLOGY_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Current ontology format:
    # {"version": ..., "namespaces": {"domain": [...], ...}}
    namespaces = data.get("namespaces", {})
    if not isinstance(namespaces, dict):
        raise ValueError("ux_ontology.json does not contain a valid 'namespaces' object.")

    return namespaces


def normalize_labels(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    return list(dict.fromkeys(
        item.strip()
        for item in text.split(";")
        if item.strip()
    ))


def labels_to_string(labels: list[str]) -> str:
    return ";".join(dict.fromkeys(labels))


def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def initialize_dataframe() -> pd.DataFrame:
    """
    Candidate file is the source of truth.

    We create a separate gold/review dataset from candidates and merge any
    previously saved review decisions by evidence_id.
    """
    candidates = read_csv_safe(CANDIDATE_FILE)

    required = {
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
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            "candidate_annotations.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = candidates.copy()

    # Always normalize review columns.
    for col in ["gold_labels", "review_status", "review_comment"]:
        df[col] = df[col].fillna("").astype(str)

    # If a prior annotations.csv exists and has evidence_id, restore reviews.
    if GOLD_FILE.exists():
        try:
            previous = read_csv_safe(GOLD_FILE)
            if "evidence_id" in previous.columns:
                previous = previous.drop_duplicates("evidence_id", keep="last")
                previous = previous.set_index("evidence_id")

                for col in ["gold_labels", "labels", "review_status", "review_comment"]:
                    if col in previous.columns:
                        mapping = previous[col].to_dict()
                        mapped = df["evidence_id"].map(mapping)

                        if col in {"gold_labels", "labels"}:
                            # Prefer reviewed data only when it is actually present.
                            nonempty = mapped.notna() & mapped.astype(str).str.strip().ne("")
                            df.loc[nonempty, "gold_labels"] = mapped[nonempty].astype(str)

                        elif col == "review_status":
                            nonempty = mapped.notna() & mapped.astype(str).str.strip().ne("")
                            df.loc[nonempty, "review_status"] = mapped[nonempty].astype(str)

                        else:
                            nonempty = mapped.notna()
                            df.loc[nonempty, "review_comment"] = mapped[nonempty].fillna("").astype(str)
        except Exception:
            # An older incompatible annotations.csv should not break annotation.
            pass

    df["labels"] = df["gold_labels"].fillna("").astype(str)
    df["review_status"] = df["review_status"].replace({"": "pending"}).fillna("pending").astype(str)
    df["review_comment"] = df["review_comment"].fillna("").astype(str)

    return df


def save_gold_dataset(df: pd.DataFrame) -> None:
    GOLD_FILE.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "text",
        "participant_id",
        "source_name",
        "evidence_id",
        "quote",
        "timestamp",
        "suggested_labels",
        "gold_labels",
        "labels",
        "review_status",
        "review_comment",
    ]
    df = df.copy()

    # Guarantee expected columns exist.
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    # Atomic-ish write: write temp then replace.
    temp = GOLD_FILE.with_name(GOLD_FILE.name + ".tmp")
    df[columns].to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(GOLD_FILE)


def review_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["review_status"].value_counts().to_dict()
    return {
        "pending": int(counts.get("pending", 0)),
        "approved": int(counts.get("approved", 0)),
        "corrected": int(counts.get("corrected", 0)),
        "skipped": int(counts.get("skipped", 0)),
    }


def is_candidate_valid(row: pd.Series) -> bool:
    text = safe_text(row.get("text", "")).strip()

    if len(text.split()) < 4:
        return False

    # Obvious header / metadata patterns.
    bad_patterns = [
        r"^={4,}",
        r"^interview\s+\d+\s*[-—]",
        r"^moderated usability test",
        r"^\d{1,2}[-/][A-Za-z0-9]{3,9}[-/]\d{2,4}",
        r"^(participant profile|participant information|demographics)\b",
    ]

    return not any(re.search(pattern, text, flags=re.I) for pattern in bad_patterns)


def make_label_options(namespaces: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}

    for namespace in ["domain", "issue", "response", "outcome"]:
        values = namespaces.get(namespace, [])
        if not isinstance(values, list):
            values = []

        groups[namespace] = [
            f"{namespace}:{str(value).strip()}"
            for value in values
            if str(value).strip()
        ]

    return groups


def format_label(label: str) -> str:
    if ":" not in label:
        return label
    _, value = label.split(":", 1)
    return value.replace("_", " ").title()


def extract_selected_for_namespace(
    selected: list[str],
    namespace: str,
) -> list[str]:
    prefix = namespace + ":"
    return [label for label in selected if label.startswith(prefix)]


def clear_widget_keys(evidence_id: str) -> None:
    for key in [
        f"{evidence_id}_domain",
        f"{evidence_id}_issue",
        f"{evidence_id}_response",
        f"{evidence_id}_outcome",
        f"{evidence_id}_status",
        f"{evidence_id}_comment",
    ]:
        st.session_state.pop(key, None)


def current_index(df: pd.DataFrame) -> int:
    if "annotation_index" not in st.session_state:
        st.session_state.annotation_index = 0

    idx = int(st.session_state.annotation_index)
    return max(0, min(idx, max(0, len(df) - 1)))


def move_to(df: pd.DataFrame, new_index: int) -> None:
    new_index = max(0, min(new_index, len(df) - 1))
    old_idx = current_index(df)

    if 0 <= old_idx < len(df):
        old_evidence = safe_text(df.iloc[old_idx]["evidence_id"])
        if old_evidence:
            clear_widget_keys(old_evidence)

    st.session_state.annotation_index = new_index


# ============================================================
# Main
# ============================================================

def main() -> None:
    st.title("🧠 UX Research Intelligence — Phase 3 Annotation")
    st.caption(
        "This screen creates the human-reviewed dataset that will train our UX NLP model."
    )

    # ---------- Load files ----------
    try:
        namespaces = load_ontology()
        df = initialize_dataframe()
    except Exception as exc:
        st.error(f"Could not load the annotation workspace: {exc}")
        st.stop()

    if df.empty:
        st.warning("candidate_annotations.csv contains no rows.")
        st.stop()

    label_groups = make_label_options(namespaces)

    # ---------- Progress ----------
    counts = review_counts(df)
    total = len(df)
    reviewed = total - counts["pending"]
    progress = reviewed / total if total else 0

    st.progress(progress)
    st.markdown(
        f"**Progress:** {reviewed} / {total} reviewed  ·  "
        f"Pending: {counts['pending']}  ·  "
        f"Approved: {counts['approved']}  ·  "
        f"Corrected: {counts['corrected']}  ·  "
        f"Skipped: {counts['skipped']}"
    )

    st.divider()

    # ---------- Instructions ----------
    with st.expander("📖 How to annotate — read this first", expanded=False):
        st.markdown(
            """
**Your job:** decide which UX concepts are genuinely supported by the participant's statement.

For each evidence segment:

1. Read the **Participant Evidence**.
2. Use the suggested labels only as a starting point.
3. Select the correct **Domain**, **UX Issue**, **User Response**, and/or **Outcome**.
4. You do **not** have to select a label from every group.
5. You may select multiple labels in a group when the evidence supports them.
6. Choose **Approve** when the suggestions are correct.
7. Choose **Correct** when you change/add/remove labels.
8. Choose **Skip** when the text is not useful training evidence.
9. Click **Save & Next**.

Examples:

`"The settings were hard to find."`

→ `domain:discoverability`
→ `issue:discoverability`

`"I couldn't find it and became frustrated."`

→ `issue:discoverability`
→ `response:frustration`
→ `outcome:task_friction`

`"The product page looks clean."`

→ possibly `domain:visual_design`
→ `response:positive`

Do not invent an issue or outcome just because a label exists.
"""
        )

    # ---------- Navigation ----------
    left, middle, right = st.columns([1, 3, 1])

    with left:
        if st.button("← Previous", use_container_width=True, disabled=current_index(df) == 0):
            move_to(df, current_index(df) - 1)
            st.rerun()

    with middle:
        pending_positions = df.index[df["review_status"] == "pending"].tolist()

        options = {
            f"{i+1:03d} · {safe_text(df.iloc[i]['participant_id'])} · {safe_text(df.iloc[i]['evidence_id'])}": i
            for i in range(len(df))
        }

        selected_label = st.selectbox(
            "Jump to evidence",
            options=list(options.keys()),
            index=current_index(df),
            key="jump_to_evidence",
        )
        selected_position = options[selected_label]

        if selected_position != current_index(df):
            move_to(df, selected_position)
            st.rerun()

    with right:
        if st.button(
            "Next →",
            use_container_width=True,
            disabled=current_index(df) >= len(df) - 1,
        ):
            move_to(df, current_index(df) + 1)
            st.rerun()

    # ---------- Current row ----------
    idx = current_index(df)
    row = df.iloc[idx]

    evidence_id = safe_text(row["evidence_id"])
    participant_id = safe_text(row["participant_id"])
    source_name = safe_text(row["source_name"])
    text = safe_text(row["text"])
    quote = safe_text(row["quote"])
    timestamp = safe_text(row["timestamp"])
    suggested = normalize_labels(row.get("suggested_labels", ""))
    existing_gold = normalize_labels(row.get("gold_labels", ""))
    status = safe_text(row.get("review_status", "pending")) or "pending"
    comment = safe_text(row.get("review_comment", ""))

    st.markdown(f"## Evidence {idx + 1} of {total}")

    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("Participant", participant_id)
    meta2.metric("Evidence ID", evidence_id)
    meta3.metric("Status", status.title())

    st.markdown(
        f"**Source:** `{source_name}`"
        + (f" · **Timestamp:** `{timestamp}`" if timestamp and timestamp.lower() != "nan" else "")
    )

    st.subheader("1. Participant Evidence")
    escaped_text = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div style="
            background:#F8FAFC;
            border:1px solid #D1D5DB;
            border-left:6px solid #4F46E5;
            border-radius:10px;
            padding:18px 20px;
            font-size:18px;
            line-height:1.6;
            margin-bottom:10px;
        ">
        {escaped_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if quote and quote != text:
        st.caption(f"Original utterance: “{quote}”")

    # ---------- Suggested labels ----------
    st.subheader("2. Automated Suggestions")

    if suggested:
        suggestion_text = " · ".join(format_label(label) for label in suggested)
        st.info(
            f"{suggestion_text}\n\n"
            "These are hints only. They are not ground truth."
        )
    else:
        st.info("No automated suggestions were generated.")

    # Obvious invalid evidence warning.
    if not is_candidate_valid(row):
        st.warning(
            "This row looks like metadata/header text or is too short to be useful. "
            "Use **Skip** unless you have a strong reason to retain it."
        )

    # ---------- Annotation form ----------
    st.subheader("3. Human Annotation")

    suggested_groups = {
        namespace: extract_selected_for_namespace(suggested, namespace)
        for namespace in ["domain", "issue", "response", "outcome"]
    }
    gold_groups = {
        namespace: extract_selected_for_namespace(existing_gold, namespace)
        for namespace in ["domain", "issue", "response", "outcome"]
    }

    with st.form(key=f"annotation_form_{evidence_id}"):

        st.markdown("**Domain — What UX area does this relate to?**")
        domain_value = st.multiselect(
            "Domain",
            options=label_groups["domain"],
            default=gold_groups["domain"] or suggested_groups["domain"],
            format_func=format_label,
            key=f"{evidence_id}_domain",
            label_visibility="collapsed",
        )

        st.markdown("**UX Issue — What problem or usability characteristic is present?**")
        issue_value = st.multiselect(
            "UX Issue",
            options=label_groups["issue"],
            default=gold_groups["issue"] or suggested_groups["issue"],
            format_func=format_label,
            key=f"{evidence_id}_issue",
            label_visibility="collapsed",
        )

        st.markdown("**User Response — How did the participant respond/feel?**")
        response_value = st.multiselect(
            "User Response",
            options=label_groups["response"],
            default=gold_groups["response"] or suggested_groups["response"],
            format_func=format_label,
            key=f"{evidence_id}_response",
            label_visibility="collapsed",
        )

        st.markdown("**Outcome — What happened as a result?**")
        outcome_value = st.multiselect(
            "Outcome",
            options=label_groups["outcome"],
            default=gold_groups["outcome"] or suggested_groups["outcome"],
            format_func=format_label,
            key=f"{evidence_id}_outcome",
            label_visibility="collapsed",
        )

        st.markdown("**Researcher Decision**")
        status_value = st.radio(
            "Decision",
            options=["approved", "corrected", "skipped"],
            index=(
                ["approved", "corrected", "skipped"].index(status)
                if status in {"approved", "corrected", "skipped"}
                else 0
            ),
            horizontal=True,
            key=f"{evidence_id}_status",
            format_func=lambda x: x.title(),
        )

        comment_value = st.text_area(
            "Researcher comment (optional)",
            value=comment,
            placeholder="Explain any important correction or why this evidence was skipped.",
            key=f"{evidence_id}_comment",
        )

        st.markdown("---")

        save_col, skip_col, save_only_col = st.columns(3)

        save_next = save_col.form_submit_button(
            "💾 Save & Next",
            type="primary",
            use_container_width=True,
        )

        skip_next = skip_col.form_submit_button(
            "⏭️ Skip & Next",
            use_container_width=True,
        )

        save_only = save_only_col.form_submit_button(
            "💾 Save Only",
            use_container_width=True,
        )

    # ---------- Handle form submission ----------
    if save_next or save_only or skip_next:
        if skip_next:
            final_labels: list[str] = []
            final_status = "skipped"
            final_comment = comment_value or "Skipped during annotation."
        else:
            final_labels = list(
                dict.fromkeys(
                    domain_value
                    + issue_value
                    + response_value
                    + outcome_value
                )
            )
            final_status = status_value
            final_comment = comment_value

            if final_status != "skipped" and not final_labels:
                st.error(
                    "No labels selected. Either choose at least one label "
                    "or select **Skipped**."
                )
                st.stop()

        # Write annotation into the current DataFrame row.
        df.at[df.index[idx], "gold_labels"] = labels_to_string(final_labels)
        df.at[df.index[idx], "labels"] = labels_to_string(final_labels)
        df.at[df.index[idx], "review_status"] = final_status
        df.at[df.index[idx], "review_comment"] = final_comment

        try:
            save_gold_dataset(df)
        except Exception as exc:
            st.error(f"Could not save the annotation: {exc}")
            st.stop()

        if save_next or skip_next:
            next_index = min(idx + 1, len(df) - 1)
            move_to(df, next_index)
            st.success(
                f"{evidence_id} saved as **{final_status}**."
            )
            st.rerun()

        st.success(
            f"{evidence_id} saved as **{final_status}**."
        )

    # ---------- Bottom explanation ----------
    st.divider()

    gold_df = df[
        df["review_status"].isin(["approved", "corrected"])
        & df["labels"].fillna("").astype(str).str.strip().ne("")
    ]

    st.subheader("4. Current Gold Dataset")

    c1, c2, c3 = st.columns(3)
    c1.metric("Gold examples", len(gold_df))
    c2.metric("Participants", gold_df["participant_id"].nunique() if not gold_df.empty else 0)

    unique_labels = {
        label
        for value in gold_df["labels"].tolist()
        for label in normalize_labels(value)
    } if not gold_df.empty else set()

    c3.metric("Unique labels", len(unique_labels))

    if not gold_df.empty:
        label_counts: dict[str, int] = {}
        for value in gold_df["labels"]:
            for label in normalize_labels(value):
                label_counts[label] = label_counts.get(label, 0) + 1

        label_summary = (
            pd.DataFrame(
                [
                    {"Label": label, "Examples": count}
                    for label, count in sorted(
                        label_counts.items(),
                        key=lambda x: (-x[1], x[0]),
                    )
                ]
            )
        )
        st.dataframe(
            label_summary,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Training has not started. This application only creates the reviewed gold dataset."
    )


if __name__ == "__main__":
    main()
