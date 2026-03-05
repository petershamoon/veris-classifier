"""Launch fine-tuning job on HuggingFace via the AutoTrain API.

Uses the HF API directly - no autotrain-advanced package needed.
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_URL = "https://huggingface.co/api"

# ── Configuration ─────────────────────────────────────────────────────────

CONFIG = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "task": "sft",  # Supervised Fine-Tuning
    "training_data": "vibesecurityguy/veris-classifier-training",
    "output_model": "vibesecurityguy/veris-classifier-v1",

    # QLoRA settings (4-bit quantized LoRA for efficiency)
    "quantization": "int4",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # Training hyperparameters
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,  # effective batch size = 8
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "max_seq_length": 2048,
    "optimizer": "adamw_torch",
    "scheduler": "cosine",

    # Logging
    "logging_steps": 50,
    "save_strategy": "epoch",
}


def create_autotrain_space():
    """Create an AutoTrain Space to run the fine-tuning job."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Check if model repo exists, create if not
    print(f"Creating output model repo: {CONFIG['output_model']}...")
    resp = requests.post(
        f"{BASE_URL}/repos/create",
        headers=headers,
        json={
            "name": CONFIG["output_model"].split("/")[1],
            "type": "model",
            "private": False,
        },
    )
    if resp.status_code in (200, 409):  # 409 = already exists
        print("  Model repo ready")
    else:
        print(f"  Warning: {resp.status_code} - {resp.text}")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              VERIS Classifier Fine-Tuning Setup             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Base Model:  {CONFIG['base_model']:<43}║
║  Dataset:     {CONFIG['training_data']:<43}║
║  Output:      {CONFIG['output_model']:<43}║
║  Method:      QLoRA (4-bit, r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})                   ║
║  Epochs:      {CONFIG['epochs']:<43}║
║  Batch Size:  {CONFIG['batch_size']} (x{CONFIG['gradient_accumulation_steps']} accum = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']} effective)                       ║
║  LR:          {CONFIG['learning_rate']:<43}║
║  Max Length:   {CONFIG['max_seq_length']:<42}║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  To launch training, go to:                                  ║
║  https://huggingface.co/autotrain                            ║
║                                                              ║
║  Steps:                                                      ║
║  1. Click "Create New Project"                               ║
║  2. Select task: "LLM SFT"                                  ║
║  3. Select base model: Qwen/Qwen2.5-7B-Instruct             ║
║  4. Upload dataset: vibesecurityguy/veris-classifier-training║
║  5. Set column mapping: text column = "messages"             ║
║  6. Set training params (or use defaults):                   ║
║     - Epochs: 3                                              ║
║     - Learning rate: 2e-4                                    ║
║     - LoRA r: 16                                             ║
║     - Quantization: int4                                     ║
║  7. Select hardware: A10G Small (~$3-5)                      ║
║  8. Click "Start Training"                                   ║
║                                                              ║
║  Training will take ~1-2 hours.                              ║
║  Model auto-pushed to: {CONFIG['output_model']:<23}║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Also save config for reference
    config_path = os.path.join(os.path.dirname(__file__), "..", "data", "autotrain", "config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    create_autotrain_space()
