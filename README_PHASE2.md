# Phase 2 overlay — OpenAI API

This package is designed to be copied into the already-working Phase 1 project.

## IMPORTANT: Phase 1 is not overwritten

Add these files to the existing repository:

- `app_phase2.py`
- `src/openai_synthesis.py`
- `.env.example`
- `requirements_phase2.txt`

Do not replace your existing `app.py` or your Phase 1 `src/*.py` files.

## Windows + VS Code

From the existing project folder:

```powershell
D:
cd D:\ux-research-intelligence
.\.venv\Scripts\Activate.ps1
pip install -r requirements_phase2.txt
```

Create an API key in your OpenAI API account. OpenAI's official Python quickstart uses the SDK and the Responses API, with the API key supplied through the environment. See the official documentation:
https://platform.openai.com/docs/quickstart/make-your-first-api-request

For the current PowerShell session:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

Optional:

```powershell
$env:OPENAI_MODEL="gpt-5.6-luna"
```

Then run:

```powershell
streamlit run app_phase2.py
```

## What Phase 2 does

The app first runs the Phase 1 local pipeline:

Transcript
  -> parsing
  -> local MiniLM embeddings
  -> semantic clustering
  -> themes
  -> evidence
  -> pain points
  -> research gaps
  -> contradictions
  -> recommendations

When `Enable OpenAI synthesis` is checked, the app then sends the structured evidence to OpenAI through the Responses API.

The OpenAI response is constrained with Structured Outputs / JSON Schema, so the application gets predictable fields instead of free-form text.

`store=False` is used in the request.

## Why the raw transcript is not sent by default

Phase 2 deliberately sends the structured Phase 1 evidence plus selected supporting quotes instead of the entire transcript. This reduces unnecessary token usage and keeps the architecture evidence-first.

## Important privacy note

Interview transcripts can contain sensitive personal or company information. Do not send real research data to an external API unless your project/privacy requirements permit it. Review your organization's OpenAI data-retention and data-processing settings before using production research data.

## Phase 1 fallback

Uncheck `Enable OpenAI synthesis` and the application runs the Phase 1 local analysis only.

## Model selection

The default is read from `OPENAI_MODEL`, falling back to `gpt-5.6-luna`. Change it in the Streamlit sidebar or environment as appropriate for your API account.
