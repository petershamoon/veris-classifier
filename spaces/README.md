---
title: VERIS Incident Classifier
emoji: "\U0001F512"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered VERIS incident classification
models:
  - vibesecurityguy/veris-classifier-v2
datasets:
  - vibesecurityguy/veris-classifier-training
  - vibesecurityguy/veris-incident-classifications
---

# VERIS Incident Classifier

Classify security incidents into the [VERIS framework](https://verisframework.org/) using a fine-tuned Mistral-7B-Instruct model. **No API key required.**

## Features

- **Classify Incident** — Describe a security incident in plain English, get a structured VERIS JSON classification
- **Ask About VERIS** — Ask questions about the VERIS taxonomy, enumerations, and best practices
- **Free Inference** — Runs on HF ZeroGPU (A10G)

## Model

Fine-tuned [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) using QLoRA (4-bit quantization) on 10,000+ real security incidents from the [VERIS Community Database](https://github.com/vz-risk/VCDB) plus 300+ VERIS Q&A pairs.

## Links

- [Model](https://huggingface.co/vibesecurityguy/veris-classifier-v2)
- [Training Dataset](https://huggingface.co/datasets/vibesecurityguy/veris-classifier-training)
- [VERIS Framework](https://verisframework.org/)
- [Source Code](https://github.com/petershamoon/veris-classifier)
