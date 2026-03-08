# Fine-Tuning Methodology

Detailed write-up of how the VERIS Classifier model was fine-tuned, from data preparation through evaluation.

## Overview

The goal was to fine-tune an open-source LLM to classify security incidents into the [VERIS framework](https://verisframework.org/) — eliminating the need for an OpenAI API key and enabling free inference on HuggingFace Spaces.

**Result:** A fine-tuned Mistral-7B-Instruct-v0.3 model that runs on ZeroGPU (HF Pro), handling both incident classification and VERIS Q&A.

## 1. Training Data Preparation

### Source Data: VERIS Community Database (VCDB)

The [VCDB](https://github.com/vz-risk/VCDB) contains 10,037 publicly disclosed security incidents, each classified in the VERIS framework. These are real incidents with validated classifications — the same data behind the Verizon DBIR.

### Synthetic Description Generation

VCDB records are structured JSON, not natural language. To create training pairs, we generated synthetic incident descriptions using GPT-4o-mini:

- **Input:** VCDB JSON record (actor, action, asset, attribute fields + summary if available)
- **Output:** 2-4 sentence natural language description written as an incident report
- **Volume:** 10,019 descriptions generated (99.8% of VCDB)
- **Quality:** Zero jargon leakage, 15 near-duplicates, 3 invalid asset values out of 10K+

The generation pipeline used async Python with semaphore-controlled concurrency, resume support (JSONL append), and tiered retry logic for OpenAI rate limits (per-minute vs per-day detection).

### Q&A Training Pairs (311 examples)

Fine-tuning teaches a model behavior patterns (input/output mappings), not new knowledge. Since the base model may not have deep VERIS domain knowledge, we generated 311 Q&A pairs across 10 categories:

| Category | Count | Description |
|----------|-------|-------------|
| Framework Basics | 40 | What is VERIS, who created it, purpose |
| The Four A's | 50 | How actors/actions/assets/attributes work together |
| Actors | 50 | Actor types, varieties, motives |
| Actions | 60 | 7 action categories and their varieties |
| Assets | 40 | Asset prefixes and varieties |
| Attributes | 40 | CIA triad in VERIS context |
| Classification Guidance | 70 | How-to classify specific scenarios |
| VCDB & DBIR | 30 | Community database and annual report |
| Advanced Concepts | 40 | A4 Grid, discovery methods, timeline |
| Enumeration Details | 80 | Specific enum values and meanings |

### Combined Training Dataset

Both datasets were combined into a unified chat-format JSONL:
- **Training split:** 9,813 examples (95%)
- **Evaluation split:** 517 examples (5%)
- **Format:** Chat-format messages (system/user/assistant)
- **Shuffled** with seed=42 for reproducibility

**Dataset on HF Hub:** [`vibesecurityguy/veris-classifier-training`](https://huggingface.co/datasets/vibesecurityguy/veris-classifier-training)

## 2. Model Selection

### Why Mistral-7B-Instruct-v0.3?

| Model | Params | Pros | Cons |
|-------|--------|------|------|
| Qwen2.5-7B-Instruct | 7B | Strong structured output, ChatML | Chinese company (Alibaba) |
| Llama-3.1-8B-Instruct | 8B | Most popular, huge community | Gated model (requires Meta license approval) |
| **Mistral-7B-Instruct-v0.3** | **7B** | **Apache 2.0, great JSON output, no gating** | **Slightly older than Llama 3.1** |
| Gemma 2-9B-IT | 9B | Strong benchmarks | Slightly larger, tighter license |

Mistral-7B-Instruct-v0.3 was chosen because:
- **Fully open** (Apache 2.0 license) — no gating, no approval wait, no usage restrictions
- **Strong structured JSON output** — critical for VERIS classification
- **Same 7B size** — fits on A10G GPU with 4-bit quantization (QLoRA)
- **Well-supported** in transformers/trl/peft ecosystem
- **French company** (Mistral AI) — independent from Chinese and US big tech

**Note:** v1 of the classifier used Qwen2.5-7B-Instruct. We switched to Mistral for licensing transparency — no data privacy concerns with open-weight models (all inference runs locally), but having a fully open license simplifies things.

## 3. Training Configuration

### QLoRA (Quantized Low-Rank Adaptation)

QLoRA enables fine-tuning a 7B model on a single GPU by:
1. **Quantizing** the base model to 4-bit (NF4 format)
2. **Adding** small trainable LoRA adapters (rank 16)
3. **Training** only the adapters (~1% of total parameters)

### Hyperparameters

```
Base Model:         mistralai/Mistral-7B-Instruct-v0.3
Method:             QLoRA (4-bit NF4 quantization)
LoRA Rank (r):      16
LoRA Alpha:         32
LoRA Dropout:       0.05
Learning Rate:      2e-4
LR Scheduler:       Cosine
Warmup Ratio:       0.1
Epochs:             3
Batch Size:         2
Gradient Accum:     4 (effective batch size = 8)
Max Sequence Len:   2048
Optimizer:          AdamW (torch)
Chat Template:      Mistral (auto-detected by tokenizer)
Hardware:           NVIDIA A10G (22GB VRAM)
Training Time:      ~1-2 hours
Cost:               ~$3-5 (HF AutoTrain)
```

### Why These Settings?

- **r=16, alpha=32:** Standard LoRA config. Higher rank captures more task-specific knowledge, alpha=2r is a common heuristic.
- **LR 2e-4:** Higher than typical fine-tuning (1e-5 to 5e-5) because QLoRA adapters are small and need stronger updates.
- **Cosine scheduler:** Smooth decay prevents sudden learning rate drops that can hurt convergence.
- **3 epochs:** Enough passes over 10K examples to learn the mapping without overfitting.
- **Batch size 2 x 4 accum = 8:** Fits in A10G VRAM while maintaining stable gradients.
- **Max seq len 2048:** Covers even the longest incident descriptions + full JSON classifications.

## 4. Training Platform — The Journey

### Attempt 1: AutoTrain Python Package
First tried installing `autotrain-advanced` locally. **Failed** — wheel builds for Pillow, pydantic-core, sentencepiece, and tiktoken all crashed on Python 3.14. The package doesn't support bleeding-edge Python versions.

### Attempt 2: AutoTrain Web UI
Configured everything through the HuggingFace AutoTrain web interface:
- Set base model, dataset, chat template (chatml), all hyperparameters
- **409 error** on "Start Training" — the output model repo already existed (we'd pre-created it)
- Deleted the repo, tried again — **400 error**
- Checked Space logs: OAuth authentication to HF Hub was timing out (`httpx.ReadTimeout` on `/oauth/userinfo`). The form submissions were received (`handle_form` logged 4 times) but training never actually started — the Space couldn't verify identity to download the dataset.

**Lesson:** AutoTrain's web UI depends on OAuth working end-to-end. If HF Hub has intermittent issues, AutoTrain silently fails.

### Attempt 3: Custom Gradio Training Space
Created our own HF Space with a Gradio UI and `trl SFTTrainer`. Two failures:
1. **Python 3.13 (default Gradio SDK)** — `audioop` module removed from stdlib, Gradio's `pydub` dependency crashes on import
2. **Docker Space with Python 3.10** — Gradio imported fine but `from huggingface_hub import HfFolder` failed — `HfFolder` was removed in newer `huggingface_hub` versions, creating an incompatibility with Gradio's OAuth module

**Lesson:** Gradio's dependency chain is fragile across version boundaries. For a training job, you don't need a UI.

### Attempt 4: Headless Docker Training Space (Partial — `formatting_func` bug)
Stripped out Gradio entirely. Built a Docker-based Space that:
- Uses `nvidia/cuda:12.1.1-devel-ubuntu22.04` base image with Python 3.10
- Installs only training dependencies (transformers, trl, peft, bitsandbytes)
- Runs training **automatically on container boot** — no button to click
- Serves a tiny `http.server` health check on port 7860 (required by Spaces)

Two fixes needed before training could start:
1. `trl` requires `rich` as a transitive dependency but doesn't declare it. Added `rich` to requirements.
2. Used `formatting_func` to convert nested `messages` → text at batch time. **Crashed** with `"Unable to create tensor... excessive nesting"` — the data collator tried to tensorize the nested `messages` column before the formatting function could process it.

### Attempt 5: Pre-Processed Dataset (Success!)
Fixed the `formatting_func` issue by pre-processing the dataset with `.map()` before training:
- Load tokenizer first, then use `tokenizer.apply_chat_template()` to convert nested messages → flat text
- Store result in a new `text` column, remove original nested columns
- Pass `dataset_text_field="text"` to SFTTrainer instead of `formatting_func`
- This avoids the collator ever seeing nested data — everything is flat strings by the time SFTTrainer touches it

**Script:** `scripts/13_launch_training_space.py` — creates the entire Space programmatically via `huggingface_hub` API.

### Actual Training Stats (from container logs)
```
Dataset:       9,813 train / 517 eval
Model memory:  7.2 GB (4-bit quantized)
Trainable:     40,370,176 / 4,393,342,464 (0.92%)
Hardware:      NVIDIA A10G (22GB VRAM)
Total steps:   3,678 (3 epochs × 9,813 examples ÷ batch 8)
Output:        162 MB LoRA adapter (adapter_model.safetensors)
Status:        ✅ Completed — model pushed to vibesecurityguy/veris-classifier-v2
```

### Platform Comparison (What We Learned)

| Platform | Tried? | Result | Notes |
|----------|--------|--------|-------|
| AutoTrain package | Yes | Failed | Python 3.14 incompatible |
| AutoTrain web UI | Yes | Failed | OAuth timeouts, silent failures |
| Gradio training Space | Yes | Failed | Dependency hell (audioop, HfFolder) |
| Headless Docker + `formatting_func` | Yes | Failed | Data collator tensorizes nested columns before formatting |
| **Headless Docker + pre-processed data** | **Yes** | **Success** | **`.map()` + `dataset_text_field` avoids nested data** |
| OpenAI Fine-tuning | No | N/A | Would lock us to OpenAI ecosystem |
| Local GPU | No | N/A | No local GPU available |

**Key takeaway:** For reliability, minimize dependencies. A training job doesn't need a web framework — just PyTorch, transformers, and a health check endpoint.

## 5. Inference Setup

### HuggingFace Spaces + ZeroGPU

The fine-tuned model runs on [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) — HuggingFace's free GPU sharing service:

- **HF Pro required** ($9/month) — ZeroGPU is not available on the free tier (returns `403 Forbidden` without Pro)
- **No per-request GPU costs** — the Pro subscription covers ZeroGPU usage
- **A10G GPU** — allocated on-demand per request via `@spaces.GPU` decorator
- **Cold start:** ~30-60s for first request (model loading + GPU allocation)
- **Warm inference:** ~5-10s per classification

### Dual-Mode Architecture

The app supports two backends:

1. **Fine-tuned HF model (primary)** — runs on ZeroGPU, no API key needed
2. **OpenAI GPT-4o (fallback)** — if user provides their own API key

Current production policy (Space):
- Hugging Face model only
- No OpenAI key path in production requests
- OpenAI fallback is kept for local development/debug use only

```python
# ZeroGPU decorator for free GPU allocation
@spaces.GPU(duration=120)
def _classify_gpu(description: str) -> dict:
    return classify_incident(description=description, use_hf=True)
```

The `@spaces.GPU` decorator handles GPU allocation transparently — the model loads into GPU memory, runs inference, and releases the GPU for other users.

### Deployment Gotchas

Three issues required fixing during deployment to Spaces:

1. **HfFolder ImportError** — Gradio OAuth internals import `HfFolder` from `huggingface_hub`, but `HfFolder` was removed in `huggingface_hub >= 0.24`. The Spaces base image pre-installs the newer version, and pinning in `requirements.txt` doesn't override it. **Fix:** Add a monkey-patch shim at the top of `app.py` (before `import gradio`) that creates a minimal `HfFolder` class wrapping `get_token()` and `login()`.

2. **Theme/CSS TypeError** — Theme/CSS belong in the `gr.Blocks()` constructor, not `app.launch()`. Using `launch(theme=..., css=...)` raises `TypeError: unexpected keyword argument`. **Fix:** Move `theme=THEME, css=CUSTOM_CSS` from `launch()` to `gr.Blocks()`.

3. **LoRA adapter loading** — The model repo only contains adapter weights (162 MB), not a full model. Loading it directly with `AutoModelForCausalLM.from_pretrained()` fails. **Fix:** Load the base model (`Mistral-7B-Instruct-v0.3`) first, apply the adapter with `PeftModel.from_pretrained()`, then merge with `merge_and_unload()`.

## 6. Evaluation

### Methodology

Evaluation uses set-based precision/recall/F1 across the four VERIS dimensions:

- **Actor Type:** Did the model identify the correct actor types? (external, internal, partner)
- **Action Category:** Did it identify the right action categories? (malware, hacking, social, etc.)
- **Asset Variety:** Did it identify the right asset types? (S - Web application, U - Laptop, etc.)
- **Attribute Type:** Did it identify the right impact types? (confidentiality, integrity, availability)

**Exact Match Rate:** Percentage of examples where ALL four dimensions match exactly.

### Running the Evaluation

```bash
# Evaluate fine-tuned model only
python scripts/06_evaluate.py --use-hf --sample-size 50

# Evaluate OpenAI baseline only
python scripts/06_evaluate.py --model gpt-4o --sample-size 50

# Side-by-side comparison (recommended)
python scripts/06_evaluate.py --compare --sample-size 50
```

## 7. Key Decisions & Trade-offs

### Fine-tuning Teaches Behavior, Not Knowledge

A common misconception: fine-tuning doesn't "teach" the model new facts. It teaches input/output patterns. The model learns:
- "When I see an incident description, I should output VERIS JSON"
- "The JSON should follow this specific schema with these field names"
- "These are the valid enumeration values for each field"

This is why we added Q&A training pairs — to teach the model the behavior of answering VERIS questions, not just classifying incidents.

### Why Not RAG?

Retrieval-Augmented Generation (RAG) is better for dynamic knowledge bases. VERIS is a relatively static taxonomy with ~300 enum values. Fine-tuning bakes this knowledge directly into the model weights, resulting in:
- Faster inference (no retrieval step)
- More consistent outputs (no retrieval noise)
- Simpler deployment (no vector database needed)

### Dataset Quality Over Quantity

With 10K training examples and 300+ enum values, each value appears ~33 times on average. This is sufficient for fine-tuning because:
- The model already understands English and JSON from pre-training
- We're teaching a specific mapping, not general reasoning
- QLoRA adapters are small and efficient learners

## 8. Reproducibility

All scripts, data, and configuration are version-controlled:

```bash
# Full pipeline from scratch
python scripts/01_ingest_vcdb.py              # Ingest VCDB (10,037 incidents)
python scripts/02_generate_dataset_fast.py     # Generate descriptions (GPT-4o-mini)
python scripts/09_generate_qa.py              # Generate Q&A pairs (311 pairs)
python scripts/08_prepare_autotrain.py --push  # Combine, split, push to HF
python scripts/13_launch_training_space.py     # Launch training on HF (A10G GPU)
# Wait ~1-2 hours for training to complete...
python scripts/06_evaluate.py --compare       # Evaluate HF model vs GPT-4o
python scripts/11_deploy_spaces.py            # Deploy Gradio app to HF Spaces
```

Random seeds are fixed (42) for dataset shuffling and evaluation sampling.

### Script Inventory

| Script | Purpose | Decision/Why |
|--------|---------|-------------|
| `01_ingest_vcdb.py` | Clone vz-risk/VCDB, parse 10,037 incidents to JSONL | VCDB is the gold standard for VERIS-classified incidents |
| `02_generate_dataset_fast.py` | Async GPT-4o-mini to generate natural language descriptions | Async + semaphore for rate limit control, resume support for reliability |
| `03_push_to_hf.py` | Push raw dataset to HF Hub | Separate from generation so we can re-push without regenerating |
| `04_finetune.py` | Format data in chat + completion formats | Created early; superseded by 08 for the actual training |
| `05_deploy_spaces.sh` | Bash deployment script | Simple but cross-platform issues; superseded by 11 |
| `06_evaluate.py` | Evaluation harness with `--compare` flag | Set-based P/R/F1 across 4 VERIS dimensions, supports both backends |
| `07_validate_dataset.py` | Quality checks (dupes, jargon, invalid enums) | Caught 15 near-dupes, 3 invalid assets before training |
| `08_prepare_autotrain.py` | Combine classification + Q&A, split 95/5, push | Unified format with separate system prompts per task type |
| `09_generate_qa.py` | Generate 311 VERIS Q&A pairs via GPT-4o-mini | Fine-tuning teaches behavior not knowledge; Q&A pairs teach answer behavior |
| `10_launch_finetune.py` | AutoTrain config + instructions | Created for AutoTrain approach; kept for reference |
| `11_deploy_spaces.py` | Python deployment to HF Spaces via API | More reliable than bash; handles Space creation + file upload |
| `12_train_via_api.py` | Attempt to launch via HF AutoTrain API | Explored but API endpoints were unreliable |
| `13_launch_training_space.py` | Create headless Docker training Space | **Final working approach** — creates Space, uploads Dockerfile + train script + requirements programmatically |

## Links

- **Live Demo:** [huggingface.co/spaces/vibesecurityguy/veris-classifier](https://huggingface.co/spaces/vibesecurityguy/veris-classifier)
- **Model:** [huggingface.co/vibesecurityguy/veris-classifier-v2](https://huggingface.co/vibesecurityguy/veris-classifier-v2)
- **Training Data:** [huggingface.co/datasets/vibesecurityguy/veris-classifier-training](https://huggingface.co/datasets/vibesecurityguy/veris-classifier-training)
- **Source Code:** [github.com/petershamoon/veris-classifier](https://github.com/petershamoon/veris-classifier)
- **VERIS Framework:** [verisframework.org](https://verisframework.org/)
