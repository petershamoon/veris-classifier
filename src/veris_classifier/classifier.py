"""Core VERIS classification logic — dual-mode inference.

Supports two backends:
1. Fine-tuned HF model (primary) — runs on ZeroGPU in HF Spaces
2. OpenAI API (fallback) — for local dev or if HF model not available
"""

import json
import logging
import re

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
    "Be helpful and educational. "
    "Answer only the user's question. "
    "Do not ask follow-up questions. "
    "Do not append additional Q&A prompts."
)

# ── HF Model Backend ─────────────────────────────────────────────────────

HF_MODEL_ID = "vibesecurityguy/veris-classifier-v2"   # LoRA adapter repo
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"   # Base model
_hf_pipeline = None
_hf_tokenizer = None


def load_hf_model():
    """Load the base model + LoRA adapter from HF Hub. Called once on first request.

    The model repo only contains LoRA adapter weights (162 MB), not a full model.
    We load the base Mistral-7B-Instruct model, then merge the adapter on top.
    """
    global _hf_pipeline, _hf_tokenizer

    if _hf_pipeline is not None:
        return _hf_pipeline, _hf_tokenizer

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    # This model path expects GPU execution (ZeroGPU on Spaces). On CPU-only
    # runtimes, transformers can fail with opaque disk offload errors.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Fine-tuned model requires GPU. This Space appears to be on CPU-only "
            "(no CUDA device available). Request ZeroGPU (A10G) or provide an "
            "OpenAI API key to use fallback inference."
        )

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
    return _generate_hf_with_options(messages, max_new_tokens=max_new_tokens)


def _generate_hf_with_options(
    messages: list[dict],
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """Generate a response using the fine-tuned HF model with explicit sampling controls."""
    pipe, tokenizer = load_hf_model()

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    outputs = pipe(messages, **generate_kwargs)

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


def _parse_json_response(raw: str) -> dict:
    """Parse model output into JSON with light recovery for wrapped text."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Recover when the model prepends/appends prose around a JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found in model output", text, 0)


def _clean_qa_response(answer: str) -> str:
    """Remove model-appended follow-up question chains from QA output."""
    text = answer.strip()
    match = re.search(r"(?:\n|[.!?]\s+)(What|How|Why|When|Where|Who)\b", text)
    if match and match.start() > 0:
        text = text[: match.start()].rstrip()
    return text


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
        raw = _generate_hf_with_options(messages, max_new_tokens=1024, do_sample=False)
    else:
        if client is None:
            raise ValueError("OpenAI client required when use_hf=False")
        raw = _generate_openai(
            client, messages, model=model, temperature=0.2, json_mode=True
        )

    return _parse_json_response(raw)


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
        raw = _generate_hf_with_options(
            messages,
            max_new_tokens=320,
            do_sample=False,
        )
        return _clean_qa_response(raw)
    else:
        if client is None:
            raise ValueError("OpenAI client required when use_hf=False")
        raw = _generate_openai(
            client, messages, model=model, temperature=0.3, max_tokens=800
        )
        return _clean_qa_response(raw)
