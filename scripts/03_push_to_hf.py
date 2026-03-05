"""Package the generated dataset and push to Hugging Face Hub."""

import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

DATA_FILE = Path(__file__).parent.parent / "data" / "dataset" / "veris_train.jsonl"

# Change this to your HF username/repo
HF_REPO = "vibesecurityguy/veris-incident-classifications"


def load_dataset(path: Path) -> list[dict]:
    """Load the generated JSONL dataset."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            # Flatten classification to a JSON string for HF storage
            row["classification_json"] = json.dumps(row.pop("classification"))
            rows.append(row)
    print(f"Loaded {len(rows)} examples")
    return rows


def build_hf_dataset(rows: list[dict]) -> DatasetDict:
    """Build a HF DatasetDict with train/test split."""
    ds = Dataset.from_list(rows)

    # 90/10 train/test split
    split = ds.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])}, Test: {len(split['test'])}")
    return split


def push_to_hub(dataset_dict: DatasetDict):
    """Push dataset to Hugging Face Hub."""
    dataset_dict.push_to_hub(
        HF_REPO,
        private=False,
    )
    print(f"\nDataset pushed to https://huggingface.co/datasets/{HF_REPO}")


def main():
    if not DATA_FILE.exists():
        print(f"Dataset file not found: {DATA_FILE}")
        print("Run 02_generate_dataset.py first.")
        return

    rows = load_dataset(DATA_FILE)
    if len(rows) < 10:
        print(f"Only {len(rows)} examples - wait for more generation to complete.")
        return

    dataset_dict = build_hf_dataset(rows)

    print("\nSample entry:")
    sample = dataset_dict["train"][0]
    print(f"  Description: {sample['description'][:120]}...")
    print(f"  Classification: {sample['classification_json'][:120]}...")
    print(f"  Industry: {sample['victim_industry']}")

    push_to_hub(dataset_dict)


if __name__ == "__main__":
    main()
