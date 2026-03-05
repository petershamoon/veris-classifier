"""Prepare and launch fine-tuning on Hugging Face with a VERIS classification model."""

import json
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = Path(__file__).parent.parent / "data" / "dataset" / "veris_train.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "finetune"


def format_for_chat(row: dict) -> dict:
    """Convert a dataset row into chat-style training format.

    This produces the format needed for fine-tuning instruction-following models:
    system message + user message + assistant response.
    """
    system_msg = (
        "You are a VERIS (Vocabulary for Event Recording and Incident Sharing) classifier. "
        "Given a security incident description, output a JSON classification using the VERIS framework. "
        "Include actor (external/internal/partner with variety and motive), "
        "action (malware/hacking/social/misuse/physical/error/environmental with variety and vector), "
        "asset (with variety like 'S - Web application', 'U - Laptop'), "
        "and attribute (confidentiality/integrity/availability with relevant sub-fields). "
        "Return ONLY valid JSON."
    )

    classification = row["classification"]
    if isinstance(classification, str):
        classification = json.loads(classification)

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Classify this security incident:\n\n{row['description']}"},
            {"role": "assistant", "content": json.dumps(classification, indent=2)},
        ]
    }


def format_for_completion(row: dict) -> dict:
    """Format as a simple text completion for models that don't support chat format."""
    classification = row["classification"]
    if isinstance(classification, str):
        classification = json.loads(classification)

    text = (
        f"### Instruction: Classify this security incident into the VERIS framework.\n\n"
        f"### Input:\n{row['description']}\n\n"
        f"### Output:\n{json.dumps(classification, indent=2)}"
    )
    return {"text": text}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the generated dataset
    rows = []
    with open(DATA_FILE) as f:
        for line in f:
            row = json.loads(line)
            # Parse classification_json back if needed
            if "classification_json" in row:
                row["classification"] = json.loads(row.pop("classification_json"))
            rows.append(row)

    print(f"Loaded {len(rows)} examples")

    # Create both formats
    chat_rows = [format_for_chat(r) for r in rows]
    completion_rows = [format_for_completion(r) for r in rows]

    # Save chat format (for Mistral, Llama, etc.)
    chat_file = OUTPUT_DIR / "veris_chat.jsonl"
    with open(chat_file, "w") as f:
        for row in chat_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Chat format: {chat_file} ({len(chat_rows)} examples)")

    # Save completion format (for simpler models)
    completion_file = OUTPUT_DIR / "veris_completion.jsonl"
    with open(completion_file, "w") as f:
        for row in completion_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Completion format: {completion_file} ({len(completion_rows)} examples)")

    # Save a train/test split for evaluation
    split_idx = int(len(rows) * 0.9)
    for fmt_name, fmt_rows in [("chat", chat_rows), ("completion", completion_rows)]:
        train = fmt_rows[:split_idx]
        test = fmt_rows[split_idx:]

        train_file = OUTPUT_DIR / f"veris_{fmt_name}_train.jsonl"
        test_file = OUTPUT_DIR / f"veris_{fmt_name}_test.jsonl"

        with open(train_file, "w") as f:
            for row in train:
                f.write(json.dumps(row) + "\n")
        with open(test_file, "w") as f:
            for row in test:
                f.write(json.dumps(row) + "\n")

        print(f"  {fmt_name} train: {len(train)}, test: {len(test)}")

    # Print a sample
    print("\n--- Sample Chat Format ---")
    sample = chat_rows[0]
    for msg in sample["messages"]:
        print(f"[{msg['role']}]: {msg['content'][:150]}...")

    print("\n--- Sample Completion Format ---")
    print(completion_rows[0]["text"][:300] + "...")

    print(f"""
========================================
Fine-tuning data is ready!

Next steps - pick one approach:

1. HUGGING FACE (free, recommended):
   - Upload to HF Hub
   - Use AutoTrain or a training notebook
   - Suggested base models: mistralai/Mistral-7B-Instruct-v0.3
                             meta-llama/Llama-3.1-8B-Instruct

2. OPENAI (paid, easier):
   - Upload {chat_file} to OpenAI
   - Fine-tune gpt-4o-mini (~$3-5 for this dataset)
   - openai api fine_tuning.jobs.create -t {chat_file} -m gpt-4o-mini-2024-07-18

3. LOCAL (free, needs GPU):
   - Use unsloth or axolotl for efficient local fine-tuning
   - Works on a single consumer GPU (24GB VRAM)
========================================
""")


if __name__ == "__main__":
    main()
