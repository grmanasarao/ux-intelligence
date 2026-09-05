# UX Research Intelligence — Phase 1

Phase 1 converts the existing notebook-based UX research pipeline into a Streamlit application.

## Important scope

- Single transcript at a time.
- No LLM/API calls.
- Local `all-MiniLM-L6-v2` embeddings.
- Semantic evidence search.
- K-Means semantic clustering.
- Theme frequency/distribution.
- Evidence-backed themes and quotes.
- Evidence strength scoring.
- Local user-response/sentiment analysis.
- Pain-point detection.
- UX severity and priority scoring.
- Research question tracking.
- Research gap detection.
- Contradiction analysis.
- Recommendation prioritization.
- Human-in-the-loop validation.
- JSON/CSV/Markdown/PDF export.

Multi-file upload and participant-metadata analysis are intentionally not included yet.

## Windows + VS Code setup

Assume the repository is on the D: drive.

```powershell
D:
cd D:\ux-research-intelligence
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py -3.11` is unavailable, use another supported Python 3.10/3.11 environment. For this project, Python 3.11 is the recommended baseline because it has broad wheel availability for the scientific Python stack.

Open the local URL printed by Streamlit.

## Test transcript

A sample transcript is provided at:

`data/sample_transcript.txt`

You can also paste it into the application.

## Architecture

```text
app.py
  |
  +-- parser.py
  +-- embeddings.py
  +-- analysis.py
  +-- feedback.py
  +-- reporting.py
```

The application intentionally does not use the current notebook's in-memory Qdrant search path. The existing notebook currently contains a Qdrant API mismatch in its old demonstration cell (`QdrantClient.search`), while its main retrieval implementation uses `query_points`. Phase 1 therefore uses normalized MiniLM embeddings + cosine similarity directly. This is lighter and avoids that version-sensitive failure.

## Phase 2

Phase 2 will add an LLM provider/API on top of the Phase 1 evidence and retrieval layer. The Phase 1 local analysis should remain available as a non-LLM baseline.
