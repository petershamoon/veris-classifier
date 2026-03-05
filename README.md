# VERIS Incident Classifier

AI-powered security incident classification using the [VERIS](https://verisframework.org/) (Vocabulary for Event Recording and Incident Sharing) framework.

Converts natural language incident reports into structured VERIS taxonomy — the same framework behind the [Verizon Data Breach Investigations Report (DBIR)](https://www.verizon.com/business/resources/reports/dbir/).

**[Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/vibesecurityguy/veris-classifier)** — no API key required!

## What It Does

**Classify incidents** — Describe a security incident in plain English, get a structured VERIS JSON classification covering actors, actions, assets, and attributes.

**Ask about VERIS** — Query the framework's taxonomy, enumerations, and classification best practices.

### Example

**Input:**
> "Russian organized crime group exploited a vulnerability in our MOVEit file transfer server, deployed ransomware, and exfiltrated 50,000 customer records."

**Output:**
```json
{
  "actor": {
    "external": {
      "variety": ["Organized crime"],
      "motive": ["Financial"]
    }
  },
  "action": {
    "hacking": {
      "variety": ["Exploit vuln"],
      "vector": ["Web application"]
    },
    "malware": {
      "variety": ["Ransomware"],
      "vector": ["Direct install"]
    }
  },
  "asset": {
    "variety": ["S - File", "S - Web application"]
  },
  "attribute": {
    "confidentiality": {
      "data_disclosure": "Yes",
      "data_variety": ["Personal"]
    },
    "availability": {
      "variety": ["Interruption"]
    }
  }
}
```

## Architecture

```
                                    VERIS Incident Classifier
                                    ========================

    ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
    │   VCDB (GitHub)  │────>│  Dataset Generation  │────>│  HuggingFace Hub    │
    │  10,037 incidents│     │  GPT-4o-mini synth   │     │  10K+ train pairs   │
    └─────────────────┘     │  + 311 Q&A pairs     │     │  + 517 eval pairs   │
                            └──────────────────────┘     └─────────┬───────────┘
                                                                   │
                                                                   v
    ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
    │   Gradio App     │<────│  Fine-tuned Model    │<────│  Custom Docker Space│
    │  ZeroGPU (A10G)  │     │  Qwen2.5-7B-Instruct│     │  trl SFTTrainer     │
    │  No API key!     │     │  vibesecurityguy/    │     │  QLoRA, 3 epochs    │
    └─────────────────┘     │  veris-classifier-v1 │     │  A10G GPU           │
                            └──────────────────────┘     └─────────────────────┘
```

## Model & Dataset

### Fine-tuned Model

| Property | Value |
|----------|-------|
| Base Model | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Method | QLoRA (4-bit quantized LoRA, r=16, alpha=32) |
| Training Data | 10,019 classification + 311 Q&A pairs |
| Epochs | 3 |
| Effective Batch Size | 8 (2 x 4 gradient accumulation) |
| Learning Rate | 2e-4 with cosine scheduler |
| Inference | HF ZeroGPU (A10G burst, requires HF Pro $9/mo) |

**Model on HF Hub:** [`vibesecurityguy/veris-classifier-v1`](https://huggingface.co/vibesecurityguy/veris-classifier-v1)

### Training Dataset

Built from **10,000+ real incidents** in the [VERIS Community Database (VCDB)](https://github.com/vz-risk/VCDB):

1. Ingested all 10,037 validated VCDB incident records from GitHub
2. Generated synthetic natural language descriptions for each using GPT-4o-mini
3. Paired descriptions with real VERIS classifications as training data
4. Added 311 VERIS Q&A pairs across 10 knowledge categories
5. Combined into unified chat-format dataset (95/5 train/eval split)

**Datasets on HF Hub:**
- [`vibesecurityguy/veris-incident-classifications`](https://huggingface.co/datasets/vibesecurityguy/veris-incident-classifications) — raw classification pairs
- [`vibesecurityguy/veris-classifier-training`](https://huggingface.co/datasets/vibesecurityguy/veris-classifier-training) — AutoTrain-ready format

## Project Structure

```
.
├── app.py                              # Gradio web application (ZeroGPU + OpenAI fallback)
├── src/veris_classifier/
│   ├── classifier.py                   # Dual-mode inference (HF model + OpenAI)
│   ├── enums.py                        # Complete VERIS enumerations (300+ values)
│   └── validator.py                    # Schema validation for classifications
├── scripts/
│   ├── 01_ingest_vcdb.py               # Pull & parse VCDB incidents from GitHub
│   ├── 02_generate_dataset_fast.py     # Async dataset generation (GPT-4o-mini)
│   ├── 03_push_to_hf.py               # Push dataset to Hugging Face Hub
│   ├── 04_finetune.py                  # Prepare fine-tuning data (chat + completion formats)
│   ├── 05_deploy_spaces.sh            # Bash deployment script
│   ├── 06_evaluate.py                  # Evaluation harness (HF vs OpenAI comparison)
│   ├── 07_validate_dataset.py          # Quality validation (dupes, jargon, enums)
│   ├── 08_prepare_autotrain.py         # Format data for HF AutoTrain
│   ├── 09_generate_qa.py              # Generate VERIS Q&A training pairs
│   ├── 10_launch_finetune.py          # AutoTrain configuration & launch
│   ├── 11_deploy_spaces.py            # Python deployment to HF Spaces
│   ├── 12_train_via_api.py            # HF API training launcher (attempted)
│   └── 13_launch_training_space.py    # Docker training Space (final approach)
├── spaces/
│   └── README.md                       # HF Spaces metadata
├── data/                               # Generated data (gitignored)
│   ├── dataset/                        # Raw training data
│   ├── autotrain/                      # AutoTrain-formatted data
│   └── evaluation/                     # Evaluation results
├── FINE_TUNING.md                      # Detailed fine-tuning methodology
├── LESSONS_LEARNED.md                  # Technical notes & gotchas
├── pyproject.toml
└── requirements.txt
```

## Setup

### Use the Live Demo (Easiest)
Visit **[huggingface.co/spaces/vibesecurityguy/veris-classifier](https://huggingface.co/spaces/vibesecurityguy/veris-classifier)** — no setup needed.

### Run Locally

```bash
# Clone
git clone https://github.com/pshamoon/veris-classifier.git
cd veris-classifier

# Install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys (optional — only needed for OpenAI fallback)

# Run
python app.py
# Opens at http://localhost:7860
```

### Python API

```python
from src.veris_classifier.classifier import classify_incident, answer_question

# Using the fine-tuned model (requires GPU)
result = classify_incident(description="Employee laptop stolen from their car", use_hf=True)
print(result)

# Using OpenAI (requires API key)
from openai import OpenAI
client = OpenAI()
result = classify_incident(client=client, description="Employee laptop stolen from their car")
print(result)

# Ask about VERIS
answer = answer_question(question="What's the difference between hacking and misuse?", use_hf=True)
print(answer)
```

### Rebuild the Dataset

```bash
# 1. Pull VCDB incidents
python scripts/01_ingest_vcdb.py

# 2. Generate descriptions (requires OpenAI API key)
python scripts/02_generate_dataset_fast.py

# 3. Generate Q&A pairs
python scripts/09_generate_qa.py

# 4. Prepare for AutoTrain & push
python scripts/08_prepare_autotrain.py --push

# 5. Evaluate
python scripts/06_evaluate.py --compare --sample-size 50
```

## VERIS Overview

VERIS classifies incidents across four dimensions (the "4 A's"):

| Dimension | Categories | Description |
|-----------|-----------|-------------|
| **Actors** | External, Internal, Partner | Who caused it |
| **Actions** | Malware, Hacking, Social, Misuse, Physical, Error, Environmental | What they did |
| **Assets** | Server, Network, User Device, Terminal, Media, People | What was affected |
| **Attributes** | Confidentiality, Integrity, Availability | How it was affected |

These create **315 possible combinations** (3 x 7 x 5 x 3) in the A4 Grid.

## Tech Stack

- **Qwen2.5-7B-Instruct** — fine-tuned classification model (QLoRA)
- **HuggingFace** — model hosting, dataset hosting, Spaces deployment, ZeroGPU
- **Gradio** — web interface with dark theme
- **OpenAI GPT-4o-mini** — synthetic dataset generation
- **VCDB** — 10,000+ real incident records
- **Python 3.10+** — async data pipeline, evaluation harness

## Documentation

- **[FINE_TUNING.md](FINE_TUNING.md)** — Detailed fine-tuning methodology, model selection rationale, platform debugging journey, and hyperparameter choices with reasoning
- **[LESSONS_LEARNED.md](LESSONS_LEARNED.md)** — Technical lessons learned: OpenAI rate limits, HF Spaces deployment gotchas, dependency conflicts, architecture decisions and why they were made

## License

MIT
