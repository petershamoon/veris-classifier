---
title: VERIS Incident Classifier
emoji: "\U0001F512"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.8.0"
python_version: "3.10"
app_file: app.py
pinned: false
hf_oauth: true
license: mit
short_description: AI-powered VERIS incident classification
models:
  - vibesecurityguy/veris-classifier-v2
datasets:
  - vibesecurityguy/veris-classifier-training
  - vibesecurityguy/veris-incident-classifications
---

# VERIS Incident Classifier

Classify incident descriptions into VERIS using a fine-tuned Mistral-7B model.

## Key Features

- Incident classification to structured VERIS output
- Output toggle: `JSON` or `Table`
- Validation summary for enum/schema quality
- Table filters and CSV export
- VERIS Q&A mode
- Hugging Face login + session status for ZeroGPU quota routing

## Runtime

- Hardware: ZeroGPU (`zero-a10g`)
- Auth: HF OAuth enabled (`hf_oauth: true`)
- Inference policy in Space: Hugging Face model only (no OpenAI key usage)

## Links

- Live Space: https://huggingface.co/spaces/vibesecurityguy/veris-classifier
- Model: https://huggingface.co/vibesecurityguy/veris-classifier-v2
- Training dataset: https://huggingface.co/datasets/vibesecurityguy/veris-classifier-training
- Source code: https://github.com/petershamoon/veris-classifier
- VERIS framework: https://verisframework.org/
