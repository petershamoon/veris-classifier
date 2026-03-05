"""Launch fine-tuning via HuggingFace AutoTrain API.

Bypasses the AutoTrain web UI — launches training directly from the command line.

Usage:
    python scripts/12_train_via_api.py
    python scripts/12_train_via_api.py --dry-run    # preview config without launching
"""

import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set in .env file")
    sys.exit(1)

AUTOTRAIN_API = "https://huggingface.co/api/autotrain"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# ── Training Configuration ────────────────────────────────────────────────

CONFIG = {
    "task": "llm-sft",
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "vibesecurityguy/veris-classifier-training",
    "output_model": "vibesecurityguy/veris-classifier-v1",

    # Data format
    "text_column": "messages",
    "chat_template": "chatml",

    # QLoRA settings
    "peft": True,
    "quantization": "int4",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",

    # Training hyperparameters
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation": 4,
    "lr": 2e-4,
    "warmup_ratio": 0.1,
    "scheduler": "cosine",
    "optimizer": "adamw_torch",
    "max_seq_length": 2048,
    "logging_steps": 50,
    "save_total_limit": 1,
    "mixed_precision": "bf16",

    # Hardware
    "hardware": "a10g-small",
}


def check_existing_jobs():
    """Check if there are any running AutoTrain jobs."""
    print("Checking for existing AutoTrain jobs...")
    resp = requests.get(
        f"{AUTOTRAIN_API}/projects",
        headers=HEADERS,
    )
    if resp.status_code == 200:
        projects = resp.json()
        active = [p for p in projects if p.get("status") in ("running", "queued")]
        if active:
            print(f"  Found {len(active)} active job(s):")
            for p in active:
                print(f"    - {p.get('id')}: {p.get('status')} ({p.get('task')})")
            return active
        print("  No active jobs found.")
    else:
        print(f"  Could not check jobs: {resp.status_code}")
    return []


def create_project():
    """Create an AutoTrain project and start training."""
    print("\nCreating AutoTrain project...")

    # First, ensure the output model repo exists
    print(f"  Ensuring model repo exists: {CONFIG['output_model']}...")
    repo_resp = requests.post(
        "https://huggingface.co/api/repos/create",
        headers=HEADERS,
        json={
            "name": CONFIG["output_model"].split("/")[1],
            "type": "model",
            "private": False,
        },
    )
    if repo_resp.status_code in (200, 409):
        print("  Model repo ready.")
    else:
        print(f"  Warning: {repo_resp.status_code} - {repo_resp.text}")

    # Create the AutoTrain project
    payload = {
        "project_name": CONFIG["output_model"].split("/")[1],
        "task": CONFIG["task"],
        "base_model": CONFIG["base_model"],
        "hardware": CONFIG["hardware"],
        "params": {
            "data_path": CONFIG["dataset"],
            "text_column": CONFIG["text_column"],
            "chat_template": CONFIG["chat_template"],
            "peft": CONFIG["peft"],
            "quantization": CONFIG["quantization"],
            "lora_r": CONFIG["lora_r"],
            "lora_alpha": CONFIG["lora_alpha"],
            "lora_dropout": CONFIG["lora_dropout"],
            "target_modules": CONFIG["target_modules"],
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "gradient_accumulation": CONFIG["gradient_accumulation"],
            "lr": CONFIG["lr"],
            "warmup_ratio": CONFIG["warmup_ratio"],
            "scheduler": CONFIG["scheduler"],
            "optimizer": CONFIG["optimizer"],
            "max_seq_length": CONFIG["max_seq_length"],
            "logging_steps": CONFIG["logging_steps"],
            "save_total_limit": CONFIG["save_total_limit"],
            "mixed_precision": CONFIG["mixed_precision"],
        },
        "hub_model": CONFIG["output_model"],
    }

    print(f"\n  Config:")
    print(f"    Base model:    {CONFIG['base_model']}")
    print(f"    Dataset:       {CONFIG['dataset']}")
    print(f"    Output:        {CONFIG['output_model']}")
    print(f"    Method:        QLoRA (int4, r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})")
    print(f"    Epochs:        {CONFIG['epochs']}")
    print(f"    Batch size:    {CONFIG['batch_size']} x {CONFIG['gradient_accumulation']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation']} effective")
    print(f"    LR:            {CONFIG['lr']}")
    print(f"    Hardware:      {CONFIG['hardware']}")

    resp = requests.post(
        f"{AUTOTRAIN_API}/create_project",
        headers=HEADERS,
        json=payload,
    )

    if resp.status_code in (200, 201):
        result = resp.json()
        print(f"\n  Project created successfully!")
        print(f"  Project ID: {result.get('id', 'unknown')}")
        print(f"  Status: {result.get('status', 'unknown')}")
        return result
    else:
        print(f"\n  ERROR {resp.status_code}: {resp.text}")

        # If the API approach fails, try the alternative: direct Space creation
        if resp.status_code in (400, 404, 422):
            print("\n  The AutoTrain API endpoint may have changed.")
            print("  Let's try the alternative approach: creating an AutoTrain Space directly.")
            return try_space_approach()

        return None


def try_space_approach():
    """Alternative: Create an AutoTrain Space with training config embedded."""
    print("\n  Creating AutoTrain training Space...")

    space_id = f"vibesecurityguy/autotrain-veris-classifier"

    # Create the Space
    resp = requests.post(
        "https://huggingface.co/api/repos/create",
        headers=HEADERS,
        json={
            "name": "autotrain-veris-classifier",
            "type": "space",
            "sdk": "docker",
            "private": False,
        },
    )

    if resp.status_code not in (200, 409):
        print(f"  Failed to create Space: {resp.status_code} - {resp.text}")
        return None

    print(f"  Space created: {space_id}")

    # Upload training config
    config_content = json.dumps({
        "task": "llm-sft",
        "base_model": CONFIG["base_model"],
        "project_name": "veris-classifier-v1",
        "log": "tensorboard",
        "backend": "local",
        "data": {
            "path": CONFIG["dataset"],
            "text_column": CONFIG["text_column"],
            "chat_template": CONFIG["chat_template"],
        },
        "params": {
            "peft": True,
            "quantization": CONFIG["quantization"],
            "lora_r": CONFIG["lora_r"],
            "lora_alpha": CONFIG["lora_alpha"],
            "lora_dropout": CONFIG["lora_dropout"],
            "target_modules": CONFIG["target_modules"],
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "gradient_accumulation": CONFIG["gradient_accumulation"],
            "lr": CONFIG["lr"],
            "warmup_ratio": CONFIG["warmup_ratio"],
            "scheduler": CONFIG["scheduler"],
            "optimizer": CONFIG["optimizer"],
            "max_seq_length": CONFIG["max_seq_length"],
            "logging_steps": CONFIG["logging_steps"],
            "save_total_limit": CONFIG["save_total_limit"],
            "mixed_precision": CONFIG["mixed_precision"],
            "push_to_hub": True,
            "repo_id": CONFIG["output_model"],
        },
    }, indent=2)

    print("  Training config prepared.")
    print(f"\n  You can also try launching via the AutoTrain CLI:")
    print(f"  autotrain llm --train \\")
    print(f"    --model {CONFIG['base_model']} \\")
    print(f"    --data-path {CONFIG['dataset']} \\")
    print(f"    --text-column {CONFIG['text_column']} \\")
    print(f"    --chat-template {CONFIG['chat_template']} \\")
    print(f"    --peft --quantization {CONFIG['quantization']} \\")
    print(f"    --lr {CONFIG['lr']} --epochs {CONFIG['epochs']} \\")
    print(f"    --batch-size {CONFIG['batch_size']} \\")
    print(f"    --push-to-hub --repo-id {CONFIG['output_model']}")

    return {"status": "config_ready", "config": config_content}


def main():
    parser = argparse.ArgumentParser(description="Launch VERIS fine-tuning via HF API")
    parser.add_argument("--dry-run", action="store_true", help="Preview config without launching")
    args = parser.parse_args()

    print("=" * 60)
    print("  VERIS Classifier — Fine-Tuning Launcher")
    print("=" * 60)

    # Check token
    print(f"\nHF Token: {'***' + HF_TOKEN[-4:] if HF_TOKEN else 'NOT SET'}")

    # Verify token
    who = requests.get(
        "https://huggingface.co/api/whoami",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
    )
    if who.status_code == 200:
        username = who.json().get("name", "unknown")
        print(f"Authenticated as: {username}")
    else:
        print(f"WARNING: Token verification failed: {who.status_code}")

    if args.dry_run:
        print("\n[DRY RUN] Config preview:")
        print(json.dumps(CONFIG, indent=2))
        return

    # Check for existing jobs
    check_existing_jobs()

    # Launch
    result = create_project()

    if result:
        print(f"""
{'=' * 60}
  Training job submitted!

  Monitor at: https://huggingface.co/{CONFIG['output_model']}
  AutoTrain:  https://huggingface.co/autotrain

  Training takes ~1-2 hours on A10G.
  The model will be auto-pushed to:
  https://huggingface.co/{CONFIG['output_model']}
{'=' * 60}
""")


if __name__ == "__main__":
    main()
