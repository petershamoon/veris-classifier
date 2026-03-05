"""Core VERIS classification logic — dual-mode inference.

Supports two backends:
1. Fine-tuned HF model (primary) — runs on ZeroGPU in HF Spaces
2. OpenAI API (fallback) — for local dev or if HF model not available
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── System prompts ────────────────────────────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = (
    "You are a VERIS (Vocabulary for Event Recording and Incident Sharing) classifier. "
    "Given a security incident description, output a JSON classification using the VERIS framework. "
    "Include actor (external/internal/partner with variety and motive), "
    "action (malware/hacking/social/misuse/physical/error/environmental with variety and vector), "
    "asset (with variety like 'S - Web application', 'U - Laptop'), "
    "and attribute (confidentiality/integrity/availability with relevant sub-fields). "
    "Return ONLY valid JSON."
)

QA_SYSTEM_PROMPT = (
    "You are a VERIS (Vocabulary for Event Recording and Incident Sharing) expert. "
    "Answer questions about the VERIS framework accurately and thoroughly. "
    "Reference specific VERIS terminology, enumeration values, and concepts. "
    "Be helpful and educational."
)

# ── HF Model Backend ─────────────────────────────────────────────────────

HF_MODEL_ID = "vibesecurityguy/veris-classifier-v1"   # LoRA adapter repo
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"            # Base model
_hf_pipeline = None
_hf_tokenizer = None


def load_hf_model():
    """Load the base model + LoRA adapter from HF Hub. Called once on first request.

    The model repo only contains LoRA adapter weights (162 MB), not a full model.
    We load the base Qwen2.5-7B-Instruct model, then merge the adapter on top.
    """
    global _hf_pipeline, _hf_tokenizer

    if _hf_pipeline is not None:
        return _hf_pipeline, _hf_tokenizer

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    _hf_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if _hf_tokenizer.pad_token is None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Applying LoRA adapter: {HF_MODEL_ID}")
    model = PeftModel.from_pretrained(model, HF_MODEL_ID)
    model = model.merge_and_unload()  # Merge adapter into base for faster inference
    logger.info("Adapter merged successfully")

    _hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=_hf_tokenizer,
        return_full_text=False,
    )

    logger.info("Model loaded and ready for inference")
    return _hf_pipeline, _hf_tokenizer


def _generate_hf(messages: list[dict], max_new_tokens: int = 1024) -> str:
    """Generate a response using the fine-tuned HF model."""
    pipe, tokenizer = load_hf_model()

    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )

    return outputs[0]["generated_text"].strip()


# ── OpenAI Backend ────────────────────────────────────────────────────────


def _generate_openai(
    client,
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 1000,
    json_mode: bool = False,
) -> str:
    """Generate a response using the OpenAI API."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


# ── Public API ────────────────────────────────────────────────────────────


def classify_incident(
    client=None,
    description: str = "",
    model: str = "gpt-4o",
    use_hf: bool = False,
) -> dict:
    """Classify a security incident into the VERIS framework.

    Args:
        client: OpenAI client (required if use_hf=False)
        description: Plain-text incident description
        model: OpenAI model name (only used if use_hf=False)
        use_hf: If True, use the fine-tuned HF model instead of OpenAI

    Returns:
        dict: VERIS classification JSON
    """
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this security incident:\n\n{description}"},
    ]

    if use_hf:
        raw = _generate_hf(messages, max_new_tokens=1024)
    else:
        if client is None:
            raise ValueError("OpenAI client required when use_hf=False")
        raw = _generate_openai(
            client, messages, model=model, temperature=0.2, json_mode=True
        )

    # Parse JSON from response (handle markdown fences if present)
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` wrapper
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()

    return json.loads(text)


def answer_question(
    client=None,
    question: str = "",
    model: str = "gpt-4o",
    use_hf: bool = False,
) -> str:
    """Answer a question about the VERIS framework.

    Args:
        client: OpenAI client (required if use_hf=False)
        question: User's question about VERIS
        model: OpenAI model name (only used if use_hf=False)
        use_hf: If True, use the fine-tuned HF model instead of OpenAI

    Returns:
        str: Answer text
    """
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    if use_hf:
        return _generate_hf(messages, max_new_tokens=800)
    else:
        if client is None:
            raise ValueError("OpenAI client required when use_hf=False")
        return _generate_openai(
            client, messages, model=model, temperature=0.3, max_tokens=800
        )
