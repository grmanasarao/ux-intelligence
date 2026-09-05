# Phase 3 annotation dataset

Do NOT treat `annotations_template.csv` as a real training dataset. It is only a format example.

Create `annotations.csv` with one row per evidence segment.

Required columns:
- text
- participant_id
- source_name
- labels

`labels` is a semicolon-separated list using the ontology namespace:
- domain:*
- issue:*
- response:*
- outcome:*

Examples:

text:
"I could not find the export button."

labels:
"domain:discoverability;issue:discoverability;response:confusion;outcome:task_friction"

text:
"The workflow was simple and I finished quickly."

labels:
"domain:workflow;response:satisfaction;outcome:task_success"

## Annotation principles

1. Annotate evidence, not interpretations.
2. Assign only labels supported by the text.
3. Multiple labels are allowed.
4. Prefer the most specific available label.
5. Do not use participant identity as a label.
6. Keep the same participant's related evidence in the same split.
7. Do not mix researcher-written summaries with raw participant evidence.

## Recommended initial dataset

Aim for at least:
- 500 labelled segments for a pilot
- 1,000–3,000 for a useful first fine-tune
- 5,000+ for a stronger multi-label system

These are practical targets, not hard scientific thresholds. More important than raw row count is label coverage, consistency, and participant-level separation.
