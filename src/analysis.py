from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from .parser import Transcript


POSITIVE_WORDS = {
    "easy", "simple", "clear", "helpful", "useful", "great", "good",
    "love", "like", "satisfied", "smooth", "fast", "intuitive", "enjoy",
    "convenient", "efficient", "happy", "comfortable", "excellent"
}

NEGATIVE_WORDS = {
    "bad", "difficult", "hard", "confusing", "confused", "frustrated",
    "frustrating", "annoying", "slow", "hate", "problem", "problems",
    "issue", "issues", "error", "errors", "broken", "unclear", "poor",
    "pain", "painful", "struggle", "struggled", "struggling", "cannot",
    "can't", "unable", "waste", "wasted", "overwhelming", "disappointed"
}

PAIN_KEYWORDS = {
    "difficult", "hard", "confusing", "confused", "frustrated", "frustrating",
    "annoying", "slow", "problem", "problems", "issue", "issues", "error",
    "errors", "broken", "unclear", "poor", "pain", "painful", "struggle",
    "struggled", "struggling", "unable", "waste", "wasted", "overwhelming",
    "blocked", "stuck", "missing", "lack", "fail", "failed"
}

STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "has", "had",
    "for", "are", "was", "were", "they", "them", "their", "there", "here",
    "about", "would", "could", "should", "what", "when", "where", "which",
    "while", "into", "just", "really", "very", "also", "because", "then",
    "than", "been", "being", "you", "your", "our", "out", "but", "not",
    "can", "will", "all", "one", "two", "use", "using", "used", "like",
    "think", "feel", "felt", "know", "want", "get", "got", "make", "made"
}


def sentiment_score(text: str) -> float:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if not tokens:
        return 0.0
    positive = sum(t in POSITIVE_WORDS for t in tokens)
    negative = sum(t in NEGATIVE_WORDS for t in tokens)
    return (positive - negative) / max(1.0, math.sqrt(len(tokens)))


def sentiment_label(score: float) -> str:
    if score >= 0.35:
        return "Positive"
    if score <= -0.35:
        return "Negative"
    return "Neutral"


def _keyword_terms(texts: list[str], top_n: int = 5) -> list[str]:
    counts = Counter()
    for text in texts:
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
        counts.update(t for t in tokens if t not in STOPWORDS)
    return [term for term, _ in counts.most_common(top_n)]


def build_segments(transcripts: list[Transcript], window_size: int = 3) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for transcript in transcripts:
        utterances = transcript.participant_utterances
        for i, utterance in enumerate(utterances):
            start = max(0, i - window_size // 2)
            end = min(len(utterances), i + window_size // 2 + 1)
            context = " ".join(u.text for u in utterances[start:end]).strip()
            if len(context.split()) < 4:
                continue
            segments.append({
                "segment_id": f"{transcript.participant_id}_{i+1}",
                "participant_id": transcript.participant_id,
                "participant_name": transcript.participant_name,
                "source_name": transcript.source_name,
                "text": context,
                "quote": utterance.text,
                "timestamp": utterance.timestamp,
                "sentiment_score": sentiment_score(context),
            })
    return segments


def choose_cluster_count(n: int) -> int:
    if n < 4:
        return 1
    return max(2, min(8, int(round(math.sqrt(n / 2)))))


def cluster_segments(
    segments: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(segments)
    if n == 0:
        return np.array([], dtype=int), {"k": 0, "silhouette": None, "points": []}
    if n < 4:
        labels = np.zeros(n, dtype=int)
        return labels, {"k": 1, "silhouette": None, "points": []}

    k = min(choose_cluster_count(n), n - 1)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)

    score = None
    if len(set(labels)) > 1 and n > k:
        try:
            score = float(silhouette_score(embeddings, labels, metric="cosine"))
        except Exception:
            score = None

    return labels, {"k": k, "silhouette": score, "points": []}


def theme_results(
    segments: list[dict[str, Any]],
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for segment, label in zip(segments, labels):
        grouped[int(label)].append(segment)

    results = []
    total_participants = len({s["participant_id"] for s in segments})

    for label, items in sorted(grouped.items()):
        texts = [x["text"] for x in items]
        terms = _keyword_terms(texts, top_n=4)
        name = " / ".join(t.title() for t in terms[:3]) if terms else f"Theme {label + 1}"

        participants = sorted({x["participant_id"] for x in items})
        frequency = len(items)
        coverage = len(participants) / max(1, total_participants)

        scores = [x["sentiment_score"] for x in items]
        negative_share = sum(s < -0.35 for s in scores) / max(1, len(scores))

        evidence_strength = round(
            min(100.0, 45 * min(1.0, frequency / 5) + 35 * coverage + 20 * (1 - negative_share * 0.35)),
            1,
        )

        representative = sorted(
            items,
            key=lambda x: abs(x["sentiment_score"]),
            reverse=True,
        )[0]

        results.append({
            "theme_id": f"theme_{label + 1}",
            "cluster_label": int(label),
            "name": name,
            "description": (
                f"This theme contains {frequency} evidence segments from "
                f"{len(participants)} participant(s)."
            ),
            "frequency": frequency,
            "participant_count": len(participants),
            "participants": participants,
            "coverage_pct": round(coverage * 100, 1),
            "evidence_strength": evidence_strength,
            "evidence_level": (
                "Very Strong" if evidence_strength >= 90 else
                "Strong" if evidence_strength >= 75 else
                "Moderate" if evidence_strength >= 50 else "Weak"
            ),
            "negative_share_pct": round(negative_share * 100, 1),
            "representative_quote": representative["quote"],
            "representative_participant": representative["participant_name"],
            "evidence": [
                {
                    "participant": x["participant_name"],
                    "quote": x["quote"],
                    "timestamp": x["timestamp"],
                    "segment_id": x["segment_id"],
                }
                for x in items[:8]
            ],
        })

    return sorted(results, key=lambda x: (x["evidence_strength"], x["frequency"]), reverse=True)


def pain_points(segments: list[dict[str, Any]], labels: np.ndarray, themes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    theme_lookup = {
        int(theme["cluster_label"]): theme
        for theme in themes
    }

    candidates = []
    for segment, label in zip(segments, labels):
        text_lower = segment["text"].lower()
        hits = [kw for kw in PAIN_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", text_lower)]
        if segment["sentiment_score"] <= -0.35 or hits:
            severity = min(
                10.0,
                3.0
                + min(3.0, len(hits) * 0.7)
                + min(2.0, abs(segment["sentiment_score"]) * 2)
                + 1.0,
            )
            candidates.append({
                "theme": theme_lookup.get(int(label), {}).get("name", "Uncategorized"),
                "severity": round(severity, 1),
                "participant": segment["participant_name"],
                "quote": segment["quote"],
                "keywords": hits[:6],
                "segment_id": segment["segment_id"],
            })

    # Aggregate by theme.
    grouped = defaultdict(list)
    for item in candidates:
        grouped[item["theme"]].append(item)

    results = []
    for theme, items in grouped.items():
        participants = len({x["participant"] for x in items})
        avg_severity = sum(x["severity"] for x in items) / len(items)
        frequency = len(items)
        priority_score = min(10.0, avg_severity * 0.55 + min(10, participants) * 0.35 + min(10, frequency) * 0.10)
        results.append({
            "pain_point": theme,
            "frequency": frequency,
            "participant_count": participants,
            "average_severity": round(avg_severity, 1),
            "priority_score": round(priority_score, 1),
            "priority": "Critical" if priority_score >= 8 else "High" if priority_score >= 6.5 else "Medium",
            "evidence": items[:6],
        })

    return sorted(results, key=lambda x: x["priority_score"], reverse=True)


def recommendation_results(pain: list[dict[str, Any]], themes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations = []
    for item in pain[:10]:
        title = item["pain_point"]
        effort = "Medium"
        if item["priority_score"] >= 8:
            effort = "Medium"
        elif item["frequency"] <= 2:
            effort = "Low"

        recommendations.append({
            "recommendation_id": f"rec_{len(recommendations)+1}",
            "title": f"Address {title.lower()}",
            "description": (
                f"Investigate and reduce the user friction associated with {title.lower()}. "
                f"The issue appears in {item['participant_count']} participant(s) and "
                f"{item['frequency']} evidence segment(s)."
            ),
            "impact_score": round(item["priority_score"], 1),
            "effort": effort,
            "priority": item["priority"],
            "linked_theme": title,
            "evidence_strength": next(
                (t["evidence_strength"] for t in themes if t["name"] == title),
                item["priority_score"] * 10,
            ),
        })

    return recommendations


def research_question_results(
    questions: list[str],
    question_embeddings: np.ndarray,
    segments: list[dict[str, Any]],
    segment_embeddings: np.ndarray,
) -> list[dict[str, Any]]:
    results = []
    if not questions:
        return results

    similarities = cosine_similarity(question_embeddings, segment_embeddings) if len(segments) else np.empty((len(questions), 0))

    for idx, question in enumerate(questions):
        scores = similarities[idx] if similarities.size else np.array([])
        top_indices = np.argsort(scores)[::-1][:5] if len(scores) else []
        strong = [int(i) for i in top_indices if scores[i] >= 0.45]
        participants = sorted({segments[i]["participant_id"] for i in strong})

        if len(participants) >= 3:
            status = "Answered"
        elif len(participants) >= 1:
            status = "Partially answered"
        else:
            status = "Insufficient evidence"

        results.append({
            "question": question,
            "status": status,
            "participant_count": len(participants),
            "top_similarity": round(float(scores[top_indices[0]]) if len(top_indices) else 0.0, 3),
            "evidence": [
                {
                    "participant": segments[i]["participant_name"],
                    "quote": segments[i]["quote"],
                    "similarity": round(float(scores[i]), 3),
                    "segment_id": segments[i]["segment_id"],
                }
                for i in strong
            ],
        })
    return results


def research_gaps(question_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = []
    for item in question_results:
        if item["status"] != "Answered":
            gaps.append({
                "gap_type": "Unanswered research question" if item["status"] == "Insufficient evidence" else "Partial evidence",
                "description": item["question"],
                "participant_count": item["participant_count"],
                "suggested_follow_up": (
                    "Conduct targeted follow-up interviews or usability testing focused on this question."
                ),
            })
    return gaps


def contradiction_results(
    segments: list[dict[str, Any]],
    labels: np.ndarray,
    themes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_label = defaultdict(list)
    for segment, label in zip(segments, labels):
        by_label[int(label)].append(segment)

    theme_names = {i: theme["name"] for i, theme in enumerate(themes)}
    contradictions = []

    for label, items in by_label.items():
        positive = [x for x in items if x["sentiment_score"] >= 0.35]
        negative = [x for x in items if x["sentiment_score"] <= -0.35]
        if positive and negative:
            contradictions.append({
                "topic": theme_names.get(label, f"Theme {label+1}"),
                "description": (
                    "Participants show materially different responses within this theme: "
                    "some evidence is positive while other evidence is negative."
                ),
                "positive_evidence": [
                    {"participant": x["participant_name"], "quote": x["quote"]}
                    for x in positive[:4]
                ],
                "negative_evidence": [
                    {"participant": x["participant_name"], "quote": x["quote"]}
                    for x in negative[:4]
                ],
            })

    return contradictions


def sentiment_overview(segments: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [sentiment_label(x["sentiment_score"]) for x in segments]
    counts = Counter(labels)
    total = len(labels)
    return {
        "counts": dict(counts),
        "percentages": {
            key: round(value / max(1, total) * 100, 1)
            for key, value in counts.items()
        },
        "average_score": round(
            sum(x["sentiment_score"] for x in segments) / max(1, total), 3
        ),
    }


def clustering_coordinates(embeddings: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    if len(embeddings) == 0:
        return pd.DataFrame(columns=["x", "y", "cluster"])

    if embeddings.shape[0] == 1:
        coords = np.array([[0.0, 0.0]])
    elif embeddings.shape[0] == 2:
        coords = np.column_stack([embeddings[:, 0], np.zeros(2)])
    else:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)

    return pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": [f"Cluster {int(x)+1}" for x in labels],
    })
