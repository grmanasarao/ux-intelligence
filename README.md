# UX Research Intelligence

An AI-powered platform for turning raw UX research transcripts into evidence-backed themes, pain points, and recommendations — combining local NLP, LLM reasoning, and a domain-specific UX classification model.

Researchers spend hours manually coding interview transcripts to find themes, pain points, and contradictions. This project automates that first pass: it clusters evidence semantically, scores severity and confidence, and surfaces gaps and contradictions — while keeping a human reviewer in the loop and every claim traceable back to a quote.

The project is built in three additive phases, each usable on its own.

```text
Transcript
  -> parsing (src/parser.py)
  -> local MiniLM embeddings (src/embeddings.py)
  -> semantic clustering + theme detection (src/analysis.py)
  -> evidence-backed pain points, gaps, contradictions
  -> [optional] LLM synthesis (src/openai_synthesis.py)
  -> [optional] domain-tuned classification (phase3/)
  -> human review (src/feedback.py)
  -> export: JSON / CSV / Markdown / PDF (src/reporting.py)
```

## Project layout

```text
app.py                  Phase 1 Streamlit app (local, no API)
app_phase2.py            Phase 2 Streamlit app (adds LLM synthesis)
src/                      Shared pipeline: parsing, embeddings, analysis, feedback, reporting
phase3/                   UX-domain transformer: ontology, training, evaluation, inference
data/                     Sample transcript for local testing
```

---

## Phase 1 — Local pipeline (no API calls)

Converts the original notebook-based UX research pipeline into a Streamlit app. Everything runs locally with `all-MiniLM-L6-v2` embeddings and cosine similarity — no external API calls.

**Included:** single-transcript analysis, semantic evidence search, K-Means semantic clustering, theme frequency/distribution, evidence-backed themes and quotes, evidence strength scoring, local sentiment analysis, pain-point detection, UX severity/priority scoring, research question tracking, research gap detection, contradiction analysis, recommendation prioritization, human-in-the-loop validation, and JSON/CSV/Markdown/PDF export.

**Not yet included:** multi-file upload, participant-metadata analysis.

### Setup (Windows + VS Code)

```powershell
D:
cd D:\ux-research-intelligence
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python - m pip install -r requirements.txt
python - m streamlit run app.py
```

Python 3.11 is the recommended baseline (broadest wheel availability for the scientific Python stack). Open the local URL Streamlit prints. Try it with the sample transcript at `data/sample_transcript.txt`, or paste your own.

**Why not Qdrant:** the original notebook's demo cell calls a mismatched `QdrantClient.search` API, while its real retrieval path uses `query_points`. Phase 1 sidesteps that version-sensitive failure entirely by using normalized MiniLM embeddings + cosine similarity directly — lighter, and no external vector DB dependency.

## Phase 2 — Optional LLM synthesis

An overlay on top of Phase 1 (`app_phase2.py`) that adds OpenAI-powered synthesis without touching the Phase 1 files. The app always runs the full Phase 1 local pipeline first (parsing → embeddings → clustering → themes → evidence → pain points → gaps → contradictions → recommendations); when "Enable OpenAI synthesis" is checked, that structured evidence is then sent to OpenAI through the Responses API.

- Structured Outputs / JSON Schema constrain the response to predictable fields, not free-form text.
- Requests use `store=False`.
- **Only structured evidence + selected supporting quotes are sent — never the raw transcript.** This keeps token usage down and the architecture evidence-first.
- Unchecking the option falls back to Phase 1 local-only analysis.

### Setup

```powershell
D:
cd D:\ux-research-intelligence
.\.venv\Scripts\Activate.ps1
python - m pip install -r requirements_phase2.txt
$env:OPENAI_API_KEY="YOUR_API_KEY"
$env:OPENAI_MODEL="gpt-5.6-luna"   # optional, this is the default
python - m streamlit run app_phase2.py
```

Get an API key from your OpenAI account (see the [official quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request)). Model choice can also be changed in the Streamlit sidebar.

**Privacy:** interview transcripts can contain sensitive personal or company information. Do not send real research data to an external API unless your project's privacy requirements permit it — review your organization's data-retention and data-processing settings first.

## Phase 3 — UX-domain classification model

A fine-tuned transformer encoder for multi-label UX classification — domain specialization of a pretrained encoder (e.g. `microsoft/deberta-v3-base`), not pretraining from scratch. This is deliberately additive and does not modify Phase 1/2.

**Input:** a raw UX evidence segment, e.g. *"I couldn't find the export button and had to search around."*
**Output:** a multi-label vector over a UX ontology — `domain:*`, `issue:*`, `response:*`, `outcome:*` — e.g. `issue:discoverability`, `response:confusion`, `outcome:task_friction`. Multi-label because a single sentence often carries several concepts at once (e.g. both performance *and* discoverability).

Includes: an ontology + label system, participant-level train/validation/test splitting (leakage-safe), class-weighted training, micro/macro-F1 + precision/recall/Hamming-loss evaluation, a saved reusable Hugging Face model, standalone inference, dataset auditing, and a BlueBEAR SLURM job example.

### Setup

```powershell
D:
cd D:\ux-research-intelligence
.\.venv\Scripts\Activate.ps1
pip install -r phase3\requirements.txt

copy phase3\data\annotations_template.csv phase3\data\annotations.csv
# replace the template row with real annotations

python -m phase3.scripts.audit_dataset --data phase3\data\annotations.csv
python -m phase3.scripts.prepare_dataset --input phase3\data\annotations.csv

python -m phase3.src.train `
  --data phase3\data\annotations.csv `
  --output-dir phase3\models\ux-distilbert `
  --model-name distilbert/distilbert-base-uncased `
  --epochs 1 --train-batch-size 1 --eval-batch-size 1 `
  --gradient-accumulation 4 --max-length 128

python -m phase3.src.predict `
  --model phase3\models\ux-distilbert `
  --text "I could not find the export button and I became frustrated."
```

The command above is a CPU smoke test, not the final training run. Recommended dataset scale: 500 segments for a pilot, 1,000–3,000 for a useful first fine-tune, 5,000+ for a stronger system — annotation quality and participant-level independence matter more than raw row count.

**Recommended experimental sequence** (compare on the same held-out test participants, primary metrics macro/micro-F1):
1. TF-IDF + Logistic Regression/SVM baseline
2. MiniLM embeddings + classifier
3. DistilBERT
4. BERT-base-uncased
5. DeBERTa-v3-base
6. Ontology expansion/ablation

**BlueBEAR (GPU training):** recommended for the main DeBERTa run — A100-40GB/80GB GPUs, 512GB system memory per node. Replace `YOUR_PROJECT_CODE` in `phase3/scripts/train_bluebear.sh`, confirm `bbgpu` QOS access, then `sbatch phase3/scripts/train_bluebear.sh`. Use batch jobs, not login nodes, for real training.

**Future integration:** once a Phase 3 model passes held-out evaluation, the intended path is `phase3 model → inference API/function → Phase 1 evidence explorer → researcher validation → feedback dataset`. Phase 1/2 should not be modified until that point.

**Ethics/data:** do not upload confidential transcripts to public model hubs or repos. Keep participant identifiers out of training text where possible, and follow your institution's research ethics and consent requirements.

---

## Notes on data and privacy

Phase 1 keeps all analysis local. Phase 2 only sends structured evidence to an external API when explicitly enabled. Phase 3 training data should never include participant identifiers or be uploaded to public model hubs. See `.env.example` for required environment variables.
