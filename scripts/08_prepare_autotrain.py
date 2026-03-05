"""Prepare unified training data for HF AutoTrain fine-tuning.

Combines:
- 10K classification pairs (incident description → VERIS JSON)
- ~300 Q&A pairs (VERIS questions → answers)

Outputs chat-format JSONL ready for AutoTrain.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data" / "dataset"
CLASSIFICATION_FILE = DATA_DIR / "veris_train.jsonl"
QA_FILE = DATA_DIR / "veris_qa.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "autotrain"

CLASSIFY_SYSTEM = (
    "You are a VERIS (Vocabulary for Event Recording and Incident Sharing) classifier. "
    "Given a security incident description, output a JSON classification using the VERIS framework. "
    "Include actor (external/internal/partner with variety and motive), "
    "action (malware/hacking/social/misuse/physical/error/environmental with variety and vector), "
    "asset (with variety like 'S - Web application', 'U - Laptop'), "
    "and attribute (confidentiality/integrity/availability with relevant sub-fields). "
    "Return ONLY valid JSON."
)

QA_SYSTEM = (
    "You are a VERIS (Vocabulary for Event Recording and Incident Sharing) expert. "
    "Answer questions about the VERIS framework accurately and thoroughly. "
    "Reference specific VERIS terminology, enumeration values, and concepts. "
    "Be helpful and educational."
)


def load_classification_pairs() -> list[dict]:
    """Load and format the 10K classification training pairs."""
    rows = []
    with open(CLASSIFICATION_FILE) as f:
        for line in f:
            raw = json.loads(line)
            classification = raw.get("classification", {})
            if isinstance(classification, str):
                classification = json.loads(classification)

            rows.append({
                "messages": [
                    {"role": "system", "content": CLASSIFY_SYSTEM},
                    {"role": "user", "content": f"Classify this security incident:\n\n{raw['description']}"},
                    {"role": "assistant", "content": json.dumps(classification, indent=2)},
                ]
            })
    return rows


def load_qa_pairs() -> list[dict]:
    """Load and format the VERIS Q&A pairs."""
    rows = []
    with open(QA_FILE) as f:
        for line in f:
            raw = json.loads(line)
            rows.append({
                "messages": [
                    {"role": "system", "content": QA_SYSTEM},
                    {"role": "user", "content": raw["question"]},
                    {"role": "assistant", "content": raw["answer"]},
                ]
            })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load both datasets
    print("Loading classification pairs...")
    classify_rows = load_classification_pairs()
    print(f"  {len(classify_rows)} classification examples")

    print("Loading Q&A pairs...")
    qa_rows = load_qa_pairs()
    print(f"  {len(qa_rows)} Q&A examples")

    # Combine
    all_rows = classify_rows + qa_rows
    print(f"\nTotal: {len(all_rows)} training examples")

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(all_rows)

    # Split 95/5 train/eval (most data for training, small eval set)
    split_idx = int(len(all_rows) * 0.95)
    train = all_rows[:split_idx]
    eval_set = all_rows[split_idx:]

    print(f"Train: {len(train)}, Eval: {len(eval_set)}")

    # Count Q&A in eval to make sure we have some
    eval_qa = sum(1 for r in eval_set if "VERIS expert" in r["messages"][0]["content"])
    eval_classify = len(eval_set) - eval_qa
    print(f"  Eval breakdown: {eval_classify} classification, {eval_qa} Q&A")

    # Save train
    train_file = OUTPUT_DIR / "train.jsonl"
    with open(train_file, "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")

    # Save eval
    eval_file = OUTPUT_DIR / "eval.jsonl"
    with open(eval_file, "w") as f:
        for row in eval_set:
            f.write(json.dumps(row) + "\n")

    print(f"\nFiles saved:")
    print(f"  {train_file}")
    print(f"  {eval_file}")

    # Show samples
    print("\n--- Sample Classification ---")
    sample_c = next(r for r in train if "classifier" in r["messages"][0]["content"])
    print(f"User: {sample_c['messages'][1]['content'][:150]}...")
    print(f"Assistant: {sample_c['messages'][2]['content'][:150]}...")

    print("\n--- Sample Q&A ---")
    sample_q = next(r for r in train if "expert" in r["messages"][0]["content"])
    print(f"User: {sample_q['messages'][1]['content']}")
    print(f"Assistant: {sample_q['messages'][2]['content'][:200]}...")

    # Push to HF for AutoTrain
    print("\n" + "=" * 60)
    print("Ready for AutoTrain!")
    print(f"Upload {train_file} and {eval_file} to AutoTrain")
    print("Or push to HF with: python scripts/08_prepare_autotrain.py --push")
    print("=" * 60)

    # Auto-push if --push flag
    import sys
    if "--push" in sys.argv:
        push_to_hf(train_file, eval_file)


def push_to_hf(train_file: Path, eval_file: Path):
    """Push training data to HF Hub for AutoTrain."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = "vibesecurityguy/veris-classifier-training"

    # Create dataset repo
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)

    # Upload files
    api.upload_file(
        path_or_fileobj=str(train_file),
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(eval_file),
        path_in_repo="eval.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\nPushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
