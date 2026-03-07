---
library_name: peft
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
  - veris
  - cybersecurity
  - incident-classification
  - lora
  - qlora
  - mistral
datasets:
  - vibesecurityguy/veris-classifier-training
  - vibesecurityguy/veris-incident-classifications
language:
  - en
pipeline_tag: text-generation
---

# VERIS Classifier v2

A fine-tuned [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model that classifies cybersecurity incident descriptions into the [VERIS](http://veriscommunity.net/) (Vocabulary for Event Recording and Incident Sharing) framework.

Given a plain-English incident description, the model outputs structured JSON with the correct VERIS categories for **action**, **actor**, **asset**, and **attribute**.

**[Try the live demo](https://huggingface.co/spaces/vibesecurityguy/veris-classifier)** — no API key required, runs on ZeroGPU.

## Example

**Input:**
> An employee at a hospital clicked a phishing email, which installed ransomware that encrypted patient records.

**Output:**
```json
{
  "action": {"hacking": {"variety": ["Ransomware"]}, "social": {"variety": ["Phishing"]}},
  "actor": {"external": {"variety": ["Unaffiliated"], "motive": ["Financial"]}},
  "asset": {"assets": [{"variety": "S - Database"}]},
  "attribute": {"availability": {"variety": ["Obscuration"]}}
}
```

## Training Details

| Parameter | Value |
|-----------|-------|
| **Base model** | mistralai/Mistral-7B-Instruct-v0.3 |
| **Method** | QLoRA (4-bit NF4 quantization + LoRA) |
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.05 |
| **Target modules** | All linear (q, k, v, o, gate, up, down) |
| **Training examples** | 9,813 train / 517 eval |
| **Epochs** | 3 |
| **Batch size** | 2 x 4 gradient accumulation = 8 effective |
| **Learning rate** | 2e-4 (cosine schedule, 10% warmup) |
| **Precision** | bf16 |
| **Optimizer** | AdamW |
| **Max sequence length** | 2,048 tokens |
| **Hardware** | NVIDIA A10G (24GB VRAM) |

## Training Data

Fine-tuned on [vibesecurityguy/veris-classifier-training](https://huggingface.co/datasets/vibesecurityguy/veris-classifier-training), which contains:

- **10,019 classification examples** — synthetic incident descriptions generated from real [VCDB](https://github.com/vz-risk/VCDB) (Verizon Community Database) records, paired with their ground-truth VERIS classifications
- **311 Q&A pairs** — questions and answers about the VERIS framework itself

The source classifications come from 8,559 real-world incidents in VCDB, spanning healthcare, finance, retail, government, and other industries.

## How to Use

### With Transformers + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
adapter = "vibesecurityguy/veris-classifier-v2"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

messages = [
    {"role": "system", "content": "You are a VERIS classification expert..."},
    {"role": "user", "content": "Classify this incident: An employee lost a laptop containing unencrypted customer data."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

This model is designed for:
- Classifying cybersecurity incidents into the VERIS framework
- Answering questions about VERIS categories and taxonomy
- Assisting incident response teams with structured data entry

## Limitations

- **VCDB bias**: Training data over-represents healthcare (HIPAA mandatory disclosure) and US-based incidents
- **Schema version**: Trained primarily on VERIS 1.3.x schema; may not cover all 1.4 additions
- **Not a replacement for human analysis**: Output should be reviewed by a security analyst
- **English only**: Trained on English-language incident descriptions

## Links

- **Live Demo:** [huggingface.co/spaces/vibesecurityguy/veris-classifier](https://huggingface.co/spaces/vibesecurityguy/veris-classifier)
- **Training Data:** [huggingface.co/datasets/vibesecurityguy/veris-classifier-training](https://huggingface.co/datasets/vibesecurityguy/veris-classifier-training)
- **Source Code:** [github.com/petershamoon/veris-classifier](https://github.com/petershamoon/veris-classifier)
- **VERIS Framework:** [verisframework.org](https://verisframework.org/)

## Model Card Authors

Peter Shamoon ([@vibesecurityguy](https://huggingface.co/vibesecurityguy))
