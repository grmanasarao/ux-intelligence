# Phase 3 — UX-domain model

This phase is deliberately additive. It does not replace Phase 1 or Phase 2 files.

## What this implementation does

1. Defines a first UX ontology and namespace:value label system.
2. Uses human-labelled evidence segments as the training dataset.
3. Enforces participant-level train/validation/test separation to prevent leakage.
4. Fine-tunes a Transformer encoder for multi-label UX classification.
5. Uses class-weighted binary cross entropy so frequent labels do not dominate training.
6. Evaluates micro-F1, macro-F1, precision, recall and Hamming loss.
7. Saves a reusable Hugging Face model directory.
8. Provides standalone inference.
9. Provides dataset auditing and leakage-safe splitting.
10. Provides a BlueBEAR SLURM example.

## Important scientific point

This is NOT "pretraining an LLM from scratch."

It is domain specialization of a pretrained encoder. That is the right first experiment for a UX-specific model. Hugging Face's current sequence-classification workflow supports loading a pretrained encoder with `AutoModelForSequenceClassification`, tokenization, dynamic padding, `Trainer`, evaluation and model saving.

## Phase 3 model design

Input:
    raw UX evidence segment

Encoder:
    default = microsoft/deberta-v3-base

Target:
    multi-label vector over the UX ontology:
    domain:* + issue:* + response:* + outcome:*

Example:

Input:
    "I couldn't find the export button and had to search around."

Prediction:
    issue:discoverability
    response:confusion
    outcome:task_friction
    ...

The ontology is therefore part of the model's output space. It is not just documentation.

## Why multi-label?

A UX sentence can contain several valid concepts at once. For example:
"Search was slow and I couldn't find the filter."
can describe both performance and discoverability. Multi-label prediction preserves this information.

## Dataset requirements

Create:

`phase3/data/annotations.csv`

using:

`phase3/data/annotations_template.csv`

Do not use the template row as a real evaluation dataset.

Recommended scale:
- 500 labeled segments: pilot
- 1,000–3,000: useful first fine-tune
- 5,000+: stronger first system

These are engineering targets, not guarantees. Annotation quality and participant-level independence matter more than raw row count.

## Windows / VS Code local setup

From the existing project root:

```powershell
D:
cd D:\ux-research-intelligence
.\.venv\Scripts\Activate.ps1
pip install -r phase3\requirements.txt
```

Create your dataset:

```powershell
copy phase3\data\annotations_template.csv phase3\data\annotations.csv
```

Then replace the example row with your real annotations.

Audit:

```powershell
python -m phase3.scripts.audit_dataset --data phase3\data\annotations.csv
```

Create leakage-safe splits:

```powershell
python -m phase3.scripts.prepare_dataset --input phase3\data\annotations.csv
```

Local smoke-test training:

```powershell
python -m phase3.src.train `
  --data phase3\data\annotations.csv `
  --output-dir phase3\models\ux-distilbert `
  --model-name distilbert/distilbert-base-uncased `
  --epochs 1 `
  --train-batch-size 1 `
  --eval-batch-size 1 `
  --gradient-accumulation 4 `
  --max-length 128
```

This verifies the full training pipeline on CPU. It is NOT the final high-quality training run.

Run inference:

```powershell
python -m phase3.src.predict `
  --model phase3\models\ux-distilbert `
  --text "I could not find the export button and I became frustrated."
```

## BlueBEAR

BlueBEAR is the recommended location for the main DeBERTa experiment because its GPU service currently provides A100-40GB and A100-80GB GPUs; GPU nodes have 512 GB system memory. A single A100 is sufficient for the first fine-tuning experiment described here.

Before running:
1. Replace `YOUR_PROJECT_CODE` in `phase3/scripts/train_bluebear.sh`.
2. Confirm your project has access to the `bbgpu` QOS.
3. Check available module names with `module spider PyTorch`.
4. Put your source and annotated data in your BlueBEAR working area.
5. Submit with `sbatch phase3/scripts/train_bluebear.sh`.

BlueBEAR currently recommends using batch jobs for significant CPU/memory/GPU work rather than the login nodes.

## First BlueBEAR command sequence

```bash
ssh YOUR_USERNAME@bluebear.bham.ac.uk

mkdir -p ~/ux-research-intelligence
cd ~/ux-research-intelligence

# Copy the phase3 folder into this directory from your local machine.
# Then:
module spider PyTorch
```

Test GPU availability with:

```bash
module purge
module load bluebear
module load bear-apps/2023a
module load PyTorch
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Then submit the training job:

```bash
sbatch phase3/scripts/train_bluebear.sh
```

Check:

```bash
squeue -u $USER
```

and after completion:

```bash
cat phase3/logs/ux_transformer-<JOBID>.out
```

## Recommended experimental sequence

Do not declare DeBERTa "the best model" immediately.

Run these in order:

Experiment A:
    TF-IDF + Logistic Regression/SVM baseline

Experiment B:
    MiniLM embeddings + classifier

Experiment C:
    DistilBERT

Experiment D:
    BERT-base-uncased

Experiment E:
    DeBERTa-v3-base

Experiment F:
    ontology expansion/ablation

Compare them on the SAME held-out test participants.

Primary metrics:
    macro-F1
    micro-F1

Secondary:
    precision
    recall
    Hamming loss
    per-label F1

The final chosen model should be the model that gives the best meaningful UX performance, not necessarily the largest model.

## Future integration with Phase 1/2

Do not modify the Phase 1/2 application yet.

Once the Phase 3 model passes the held-out evaluation:
    phase3 model
        ↓
    inference API/function
        ↓
    Phase 1 evidence explorer
        ↓
    researcher validation
        ↓
    feedback dataset

That integration should be the last Phase 3 step, not the first.

## Ethical/data note

Do not upload confidential interview transcripts to public model hubs or public repositories. Keep participant identifiers out of training text where possible. Respect your institution's research ethics, consent, and data-retention requirements.
