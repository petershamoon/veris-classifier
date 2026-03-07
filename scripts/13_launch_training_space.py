"""Launch a dedicated training Space on HuggingFace.

Creates a Docker-based Space that runs training automatically on startup.
No Gradio — monitor via container logs in the HF Spaces UI.

Usage:
    python scripts/13_launch_training_space.py
"""

import os
import textwrap

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_ID = "vibesecurityguy/veris-train"
MODEL_REPO = "vibesecurityguy/veris-classifier-v2"

api = HfApi(token=HF_TOKEN)


def create_training_space():
    """Create a Docker Space that runs fine-tuning on startup."""

    print(f"Creating training Space: {SPACE_ID}...")

    # Delete existing space first to start clean
    try:
        api.delete_repo(SPACE_ID, repo_type="space")
        print("  Deleted existing Space.")
    except Exception:
        pass

    # Create Space with Docker SDK + GPU
    api.create_repo(
        SPACE_ID,
        repo_type="space",
        space_sdk="docker",
        space_hardware="a10g-small",
        exist_ok=True,
        private=False,
    )
    print("  Space created (Docker SDK).")

    # Set HF token as secret
    api.add_space_secret(SPACE_ID, key="HF_TOKEN", value=HF_TOKEN)
    print("  Secret set.")

    # ── Dockerfile ───────────────────────────────────────────────────────

    dockerfile = textwrap.dedent('''\
    FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

    RUN apt-get update && apt-get install -y \\
        python3.10 python3.10-venv python3-pip git \\
        && rm -rf /var/lib/apt/lists/*

    RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \\
        && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

    WORKDIR /app

    RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY train.py .

    EXPOSE 7860

    CMD ["python", "train.py"]
    ''')

    # ── Training script (no Gradio — just trains and logs) ───────────────

    train_script = textwrap.dedent('''\
    """VERIS Classifier Fine-Tuning — runs automatically on startup.

    Key fix (attempt 2): Pre-process dataset with .map() to convert nested
    messages into flat text BEFORE passing to SFTTrainer. The formatting_func
    approach fails because trl's data collator tries to tensorize the nested
    messages column before the formatting function can process it.
    """

    import json
    import os
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
    DATASET_ID = "vibesecurityguy/veris-classifier-training"
    OUTPUT_REPO = "vibesecurityguy/veris-classifier-v2"
    HF_TOKEN = os.getenv("HF_TOKEN")

    status = "starting"

    # ── Tiny health-check server (Spaces needs port 7860 open) ────────────

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h1>VERIS Training</h1>"
                f"<p>Status: <b>{status}</b></p>"
                f"<p>Check container logs for detailed progress.</p>"
                f"</body></html>".encode()
            )
        def log_message(self, format, *args):
            pass  # suppress access logs

    def start_health_server():
        server = HTTPServer(("0.0.0.0", 7860), Handler)
        server.serve_forever()

    threading.Thread(target=start_health_server, daemon=True).start()
    print("Health server running on :7860", flush=True)

    # ── Training ──────────────────────────────────────────────────────────

    def main():
        global status

        print("=" * 60, flush=True)
        print("  VERIS Classifier Fine-Tuning (v2 — pre-processed text)", flush=True)
        print("=" * 60, flush=True)

        # 1. Load tokenizer FIRST (needed for dataset pre-processing)
        status = "loading tokenizer"
        print(f"\\nLoading tokenizer: {BASE_MODEL}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("  Tokenizer loaded.", flush=True)

        # 2. Load and PRE-PROCESS dataset
        #    Convert nested messages → flat text using chat template
        #    This avoids the "excessive nesting" tensor error in SFTTrainer
        status = "loading dataset"
        print(f"\\nLoading dataset: {DATASET_ID}...", flush=True)
        dataset = load_dataset(DATASET_ID)

        if "train" in dataset:
            train_data = dataset["train"]
        else:
            split_name = list(dataset.keys())[0]
            train_data = dataset[split_name]
        print(f"  Train examples: {len(train_data)}", flush=True)

        eval_data = dataset.get("eval", dataset.get("validation", None))
        if eval_data:
            print(f"  Eval examples: {len(eval_data)}", flush=True)

        # Pre-process: messages → text (the key fix!)
        status = "pre-processing dataset"
        print("\\nPre-processing dataset (messages → text)...", flush=True)

        def format_to_text(example):
            messages = example["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        train_data = train_data.map(
            format_to_text,
            remove_columns=train_data.column_names,
            desc="Formatting train",
        )
        print(f"  Train pre-processed: {len(train_data)} examples", flush=True)
        print(f"  Sample (first 200 chars): {train_data[0]['text'][:200]}...", flush=True)

        if eval_data:
            eval_data = eval_data.map(
                format_to_text,
                remove_columns=eval_data.column_names,
                desc="Formatting eval",
            )
            print(f"  Eval pre-processed: {len(eval_data)} examples", flush=True)

        # 3. Load model with 4-bit quantization
        status = "loading model (4-bit)"
        print("\\nLoading model with 4-bit quantization...", flush=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"  Model loaded! GPU memory: {mem_gb:.1f} GB", flush=True)

        # 4. Apply LoRA
        status = "applying LoRA"
        print("\\nApplying LoRA (r=16, alpha=32)...", flush=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)

        # 5. Training args
        status = "configuring training"
        print("\\nConfiguring training...", flush=True)
        training_args = TrainingArguments(
            output_dir="./veris-finetune",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=1,
            bf16=True,
            optim="adamw_torch",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            push_to_hub=False,
        )

        # 6. Trainer — using dataset_text_field instead of formatting_func
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            dataset_text_field="text",
            max_seq_length=2048,
            packing=False,
        )

        total_steps = len(train_data) * 3 // 8
        print(f"\\n{'=' * 60}", flush=True)
        print(f"  TRAINING STARTED!", flush=True)
        print(f"  Epochs: 3  |  Batch: 2x4=8  |  LR: 2e-4", flush=True)
        print(f"  Estimated steps: ~{total_steps}", flush=True)
        print(f"{'=' * 60}\\n", flush=True)

        # 7. Train!
        status = "training"
        trainer.train()

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\\n  Training complete! Peak GPU: {peak_mem:.1f} GB", flush=True)

        # 8. Push to Hub
        status = "pushing to hub"
        print(f"\\nPushing model to {OUTPUT_REPO}...", flush=True)

        # Ensure model repo exists
        hf_api = HfApi(token=HF_TOKEN)
        hf_api.create_repo(OUTPUT_REPO, exist_ok=True, private=False)

        model.push_to_hub(OUTPUT_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(OUTPUT_REPO, token=HF_TOKEN)

        status = "DONE"
        print(f"\\n{'=' * 60}", flush=True)
        print(f"  DONE! Model at: https://huggingface.co/{OUTPUT_REPO}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"\\nYou can now delete this Space to stop GPU charges.", flush=True)

        # Keep the health server alive so you can see the "DONE" status
        import time
        while True:
            time.sleep(60)


    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            status = f"ERROR: {e}"
            print(f"\\nFATAL ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Keep alive so you can read the error in the status page
            import time
            while True:
                time.sleep(60)
    ''')

    # ── Requirements (no Gradio!) ────────────────────────────────────────

    requirements = textwrap.dedent('''\
    transformers>=4.40.0,<4.46.0
    datasets>=2.0.0
    accelerate>=0.28.0
    peft>=0.10.0
    trl>=0.8.0,<0.12.0
    bitsandbytes>=0.43.0
    huggingface_hub>=0.20.0,<0.25.0
    rich
    scipy
    sentencepiece
    protobuf
    ''')

    # ── README ───────────────────────────────────────────────────────────

    readme = textwrap.dedent('''\
    ---
    title: VERIS Classifier Training
    emoji: "\\U0001F3CB"
    colorFrom: blue
    colorTo: purple
    sdk: docker
    pinned: false
    ---

    # VERIS Classifier Training

    Fine-tuning Mistral-7B-Instruct-v0.3 with QLoRA for VERIS incident classification.
    Training starts automatically — check container logs for progress.
    ''')

    # ── Upload all files ─────────────────────────────────────────────────

    files = {
        "Dockerfile": dockerfile,
        "train.py": train_script,
        "requirements.txt": requirements,
        "README.md": readme,
    }

    for name, content in files.items():
        print(f"  Uploading {name}...")
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=name,
            repo_id=SPACE_ID,
            repo_type="space",
        )

    print(f"""
{'=' * 60}
  Training Space created!

  URL: https://huggingface.co/spaces/{SPACE_ID}

  Docker image will build (~5-10 min), then training
  starts AUTOMATICALLY. No button to click!

  Monitor progress:
  1. Go to the Space URL above
  2. Click the "Logs" tab (or the three dots > Logs)
  3. Watch the container output for training progress

  Training takes ~1-2 hours on A10G.

  When done, model will be at:
  https://huggingface.co/{MODEL_REPO}

  IMPORTANT: Delete this Space after training completes
  to stop GPU charges. (Settings > Delete this Space)
{'=' * 60}
""")


if __name__ == "__main__":
    create_training_space()
