# VERIS Incident Classifier

Convert plain-English security incident descriptions into structured VERIS output.

Live app:
- https://huggingface.co/spaces/vibesecurityguy/veris-classifier

If you just want to try it, open the live app and paste an incident description.

## What This App Does

- Classifies an incident into VERIS dimensions (`actor`, `action`, `asset`, `attribute`)
- Supports both `JSON` view (raw model output) and `Table` view (flattened rows)
- Validates model output against known VERIS enum sets
- Shows validation results (`Passed`, warnings, errors)
- Supports table filtering by dimension and `Errors Only`
- Exports filtered table rows to CSV
- Answers VERIS framework questions in a separate tab

## How Production Works (Hugging Face Space)

- Runs on Hugging Face ZeroGPU (`zero-a10g`)
- Uses the fine-tuned model only in production
- Requires user sign-in with Hugging Face so requests are tied to user quota
- Shows a session status card so users can see whether their login session is attached
- Retries queue timeouts once before returning an error

Important:
- Production does not use OpenAI keys.
- Local development still supports optional OpenAI fallback.

## Why We Made These Decisions

**ZeroGPU + HF OAuth**
- We wanted no paid per-request API dependency in production.
- ZeroGPU gives burst A10G access, but quota is user-scoped.
- OAuth login is required so requests count against the user, not an anonymous pool.

**Fine-tuned open model (Mistral-7B + LoRA adapter)**
- Keeps inference under our control in HF Spaces.
- Avoids vendor lock-in for runtime inference.
- LoRA adapter keeps model artifact smaller and easier to deploy.

**JSON-first output with table toggle**
- JSON is best for analysts and integrations.
- Table view is better for quick review and reporting.
- CSV export makes downstream handoff easier.

**Built-in validation**
- LLM outputs can drift in structure or enum values.
- Runtime validation catches bad/missing values immediately.
- Users can review warnings/errors without leaving the app.

**Retry/backoff on ZeroGPU queue errors**
- Queue contention is common with shared GPUs.
- A short automatic retry improves reliability without hiding real failures.

## Current Architecture

- UI: Gradio (`app.py`)
- Runtime inference backend: Hugging Face model pipeline (`src/veris_classifier/classifier.py`)
- Output validation: `src/veris_classifier/validator.py`
- Deployment: `scripts/11_deploy_spaces.py`

## Quick Start (Local)

### 1. Clone and install

```bash
git clone https://github.com/petershamoon/veris-classifier.git
cd veris-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment

```bash
cp .env.example .env
```

### 3. Run

```bash
python app.py
```

### 4. Backend selection

- To run local HF model inference, set `VERIS_USE_HF=true` and use a local GPU.
- If no local GPU is available, local mode can use OpenAI fallback only if you explicitly configure an OpenAI key.

## Deploy to Hugging Face Space

```bash
python scripts/11_deploy_spaces.py
```

This uploads:
- `app.py`
- `requirements.txt`
- `spaces/README.md` as Space `README.md`
- `src/veris_classifier/*`

## Repository Layout

```text
app.py
src/veris_classifier/
  classifier.py
  enums.py
  validator.py
scripts/
  01_ingest_vcdb.py
  02_generate_dataset_fast.py
  06_evaluate.py
  07_validate_dataset.py
  08_prepare_autotrain.py
  09_generate_qa.py
  11_deploy_spaces.py
  13_launch_training_space.py
FINE_TUNING.md
LESSONS_LEARNED.md
spaces/README.md
```

## Data and Model Links

- Model: https://huggingface.co/vibesecurityguy/veris-classifier-v2
- Training dataset: https://huggingface.co/datasets/vibesecurityguy/veris-classifier-training
- Incident classification dataset: https://huggingface.co/datasets/vibesecurityguy/veris-incident-classifications
- VERIS framework: https://verisframework.org/
- VCDB source: https://github.com/vz-risk/VCDB

## Documentation

- Fine-tuning and training path: [`FINE_TUNING.md`](FINE_TUNING.md)
- Engineering lessons and deployment notes: [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md)

## License

MIT
