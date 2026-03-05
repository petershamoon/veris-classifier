"""Evaluate the VERIS classifier against a held-out test set.

Loads the last 10% of the training dataset as a test split, runs the
classifier on each description, and computes per-field and aggregate
metrics using set-based precision / recall / F1.

Supports both OpenAI API and fine-tuned HF model backends.

Usage:
    python scripts/06_evaluate.py                           # GPT-4o (default)
    python scripts/06_evaluate.py --sample-size 20          # quick smoke test
    python scripts/06_evaluate.py --model gpt-4o-mini       # cheaper model
    python scripts/06_evaluate.py --use-hf                  # fine-tuned HF model
    python scripts/06_evaluate.py --compare                 # HF vs GPT-4o side-by-side
"""

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "dataset" / "veris_train.jsonl"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

# Make the project importable so we can pull in the classifier prompt
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from veris_classifier.classifier import CLASSIFY_SYSTEM_PROMPT  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Async classifier — OpenAI backend
# ---------------------------------------------------------------------------
CONCURRENCY = 10  # parallel API calls


async def classify_incident_openai(
    client,
    description: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict | None:
    """Classify a single incident description via OpenAI, with retry + backoff."""
    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Classify this security incident:\n\n{description}",
                        },
                    ],
                    temperature=0.2,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content.strip()
                return json.loads(text)
            except Exception as exc:
                if "429" in str(exc) and attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                if attempt == max_retries - 1:
                    print(f"\n  [ERROR] Classification failed: {exc}")
                    return None


# ---------------------------------------------------------------------------
# Sync classifier — HF model backend
# ---------------------------------------------------------------------------


def classify_incident_hf(description: str) -> dict | None:
    """Classify a single incident description using the fine-tuned HF model."""
    try:
        from veris_classifier.classifier import classify_incident

        return classify_incident(description=description, use_hf=True)
    except Exception as exc:
        print(f"\n  [ERROR] HF classification failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _to_set(value) -> set[str]:
    """Normalise a value (str, list, or None) into a lowercase frozen set."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if isinstance(value, list):
        return {str(v).strip().lower() for v in value if str(v).strip()}
    return set()


def set_precision_recall_f1(
    predicted: set[str], ground_truth: set[str]
) -> dict[str, float]:
    """Compute precision, recall, and F1 for two sets of labels."""
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Field extractors (ground-truth & prediction share the same schema)
# ---------------------------------------------------------------------------


def extract_actor_types(classification: dict) -> set[str]:
    """Return the set of top-level actor types (external, internal, partner)."""
    actor = classification.get("actor", {})
    return {k.lower() for k in actor if isinstance(actor[k], dict)}


def extract_action_categories(classification: dict) -> set[str]:
    """Return the set of top-level action categories."""
    action = classification.get("action", {})
    return {k.lower() for k in action if isinstance(action[k], dict)}


def extract_asset_varieties(classification: dict) -> set[str]:
    """Return the set of asset variety strings."""
    asset = classification.get("asset", {})
    varieties = asset.get("variety", [])
    return _to_set(varieties)


def extract_attribute_types(classification: dict) -> set[str]:
    """Return the set of attribute categories (confidentiality, integrity, availability)."""
    attribute = classification.get("attribute", {})
    return {k.lower() for k in attribute if isinstance(attribute[k], dict)}


def classifications_match_exactly(pred: dict, truth: dict) -> bool:
    """Check whether every field matches exactly (strict)."""
    return (
        extract_actor_types(pred) == extract_actor_types(truth)
        and extract_action_categories(pred) == extract_action_categories(truth)
        and extract_asset_varieties(pred) == extract_asset_varieties(truth)
        and extract_attribute_types(pred) == extract_attribute_types(truth)
    )


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

FIELD_EXTRACTORS = {
    "actor_type": extract_actor_types,
    "action_category": extract_action_categories,
    "asset_variety": extract_asset_varieties,
    "attribute_type": extract_attribute_types,
}


def evaluate_all(
    predictions: list[dict | None],
    ground_truths: list[dict],
) -> dict:
    """Compute per-field and aggregate metrics across the test set."""
    per_field: dict[str, list[dict[str, float]]] = {
        name: [] for name in FIELD_EXTRACTORS
    }
    exact_matches = 0
    errors = 0

    for pred, truth in zip(predictions, ground_truths):
        if pred is None:
            errors += 1
            continue

        for name, extractor in FIELD_EXTRACTORS.items():
            pred_set = extractor(pred)
            truth_set = extractor(truth)
            per_field[name].append(set_precision_recall_f1(pred_set, truth_set))

        if classifications_match_exactly(pred, truth):
            exact_matches += 1

    total_evaluated = len(predictions) - errors

    # Macro-average per field
    field_results: dict[str, dict[str, float]] = {}
    for name, scores in per_field.items():
        if not scores:
            field_results[name] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        field_results[name] = {
            metric: sum(s[metric] for s in scores) / len(scores)
            for metric in ("precision", "recall", "f1")
        }

    # Overall F1 is the macro-average of per-field F1 scores
    overall_f1 = (
        sum(fr["f1"] for fr in field_results.values()) / len(field_results)
        if field_results
        else 0.0
    )

    return {
        "total_examples": len(predictions),
        "evaluated": total_evaluated,
        "errors": errors,
        "exact_match_rate": (
            exact_matches / total_evaluated if total_evaluated else 0.0
        ),
        "exact_matches": exact_matches,
        "overall_macro_f1": overall_f1,
        "fields": field_results,
    }


# ---------------------------------------------------------------------------
# Main evaluation pipelines
# ---------------------------------------------------------------------------


async def run_evaluation_openai(
    model: str,
    sample_size: int,
    seed: int,
    test_set: list[dict],
) -> dict:
    """Run evaluation using OpenAI API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    descriptions = [row["description"] for row in test_set]
    ground_truths = _parse_ground_truths(test_set)

    print(f"\nClassifying {len(descriptions)} examples with {model} (OpenAI) ...")
    start = time.time()

    completed_count = 0
    total = len(descriptions)

    async def classify_with_progress(desc: str) -> dict | None:
        nonlocal completed_count
        result = await classify_incident_openai(client, desc, model, semaphore)
        completed_count += 1
        if completed_count % 10 == 0 or completed_count == total:
            print(f"  {completed_count}/{total} done", end="\r")
        return result

    predictions = list(
        await asyncio.gather(*[classify_with_progress(d) for d in descriptions])
    )

    elapsed = time.time() - start
    print(f"\n  Classification complete in {elapsed:.1f}s")

    results = evaluate_all(predictions, ground_truths)
    results["model"] = model
    results["backend"] = "openai"
    results["sample_size"] = sample_size
    results["seed"] = seed
    results["elapsed_seconds"] = round(elapsed, 2)
    results["timestamp"] = datetime.now(timezone.utc).isoformat()

    return results


def run_evaluation_hf(
    sample_size: int,
    seed: int,
    test_set: list[dict],
) -> dict:
    """Run evaluation using the fine-tuned HF model."""
    from veris_classifier.classifier import HF_MODEL_ID

    descriptions = [row["description"] for row in test_set]
    ground_truths = _parse_ground_truths(test_set)

    print(f"\nClassifying {len(descriptions)} examples with {HF_MODEL_ID} (HF) ...")
    print("  Loading model...")
    start = time.time()

    predictions = []
    for i, desc in enumerate(descriptions):
        pred = classify_incident_hf(desc)
        predictions.append(pred)
        if (i + 1) % 5 == 0 or (i + 1) == len(descriptions):
            print(f"  {i + 1}/{len(descriptions)} done", end="\r")

    elapsed = time.time() - start
    print(f"\n  Classification complete in {elapsed:.1f}s")

    results = evaluate_all(predictions, ground_truths)
    results["model"] = HF_MODEL_ID
    results["backend"] = "hf"
    results["sample_size"] = sample_size
    results["seed"] = seed
    results["elapsed_seconds"] = round(elapsed, 2)
    results["timestamp"] = datetime.now(timezone.utc).isoformat()

    return results


def _parse_ground_truths(test_set: list[dict]) -> list[dict]:
    """Parse ground truths, handling string-encoded JSON."""
    ground_truths = [row["classification"] for row in test_set]
    for i, gt in enumerate(ground_truths):
        if isinstance(gt, str):
            ground_truths[i] = json.loads(gt)
    return ground_truths


def load_test_set(sample_size: int, seed: int) -> list[dict]:
    """Load and optionally subsample the test set."""
    rows: list[dict] = []
    with open(DATA_FILE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    split_idx = int(len(rows) * 0.9)
    test_set = rows[split_idx:]
    print(f"Dataset size : {len(rows)}")
    print(f"Test split   : {len(test_set)} (last 10 %)")

    if sample_size and sample_size < len(test_set):
        random.seed(seed)
        test_set = random.sample(test_set, sample_size)
        print(f"Sampled      : {sample_size} examples (seed={seed})")
    else:
        print(f"Using full test split ({len(test_set)} examples)")

    return test_set


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

FIELD_DISPLAY_NAMES = {
    "actor_type": "Actor Type",
    "action_category": "Action Category",
    "asset_variety": "Asset Variety",
    "attribute_type": "Attribute Type",
}


def print_report(results: dict, label: str = "") -> None:
    """Pretty-print the evaluation results to stdout."""
    w = 60
    header = f"  VERIS Classifier Evaluation — {label}" if label else "  VERIS Classifier Evaluation Report"
    print("\n" + "=" * w)
    print(header)
    print("=" * w)
    print(f"  Model           : {results['model']}")
    print(f"  Backend         : {results.get('backend', 'openai')}")
    print(f"  Timestamp       : {results['timestamp']}")
    print(f"  Test examples   : {results['total_examples']}")
    print(f"  Evaluated       : {results['evaluated']}")
    print(f"  Errors          : {results['errors']}")
    print(f"  Elapsed         : {results['elapsed_seconds']}s")
    print("-" * w)

    print(f"\n{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * w)
    for key, scores in results["fields"].items():
        label_name = FIELD_DISPLAY_NAMES.get(key, key)
        print(
            f"  {label_name:<18} {scores['precision']:>9.1%} {scores['recall']:>9.1%} {scores['f1']:>9.1%}"
        )
    print("-" * w)
    print(f"  {'Overall Macro F1':<18} {'':>10} {'':>10} {results['overall_macro_f1']:>9.1%}")
    print(f"  {'Exact Match Rate':<18} {'':>10} {'':>10} {results['exact_match_rate']:>9.1%}")
    print("=" * w + "\n")


def print_comparison(hf_results: dict, openai_results: dict) -> None:
    """Print a side-by-side comparison of HF vs OpenAI results."""
    w = 72
    print("\n" + "=" * w)
    print("  VERIS Classifier — Model Comparison")
    print("=" * w)
    print(f"  {'':>22} {'Fine-tuned (HF)':>20} {'GPT-4o (OpenAI)':>20}")
    print("-" * w)

    for key in FIELD_EXTRACTORS:
        label = FIELD_DISPLAY_NAMES.get(key, key)
        hf_f1 = hf_results["fields"][key]["f1"]
        openai_f1 = openai_results["fields"][key]["f1"]
        diff = hf_f1 - openai_f1
        arrow = "+" if diff > 0 else ""
        print(f"  {label:<22} {hf_f1:>19.1%} {openai_f1:>19.1%}  ({arrow}{diff:.1%})")

    print("-" * w)
    hf_overall = hf_results["overall_macro_f1"]
    openai_overall = openai_results["overall_macro_f1"]
    diff = hf_overall - openai_overall
    arrow = "+" if diff > 0 else ""
    print(f"  {'Overall Macro F1':<22} {hf_overall:>19.1%} {openai_overall:>19.1%}  ({arrow}{diff:.1%})")

    hf_exact = hf_results["exact_match_rate"]
    openai_exact = openai_results["exact_match_rate"]
    diff = hf_exact - openai_exact
    arrow = "+" if diff > 0 else ""
    print(f"  {'Exact Match Rate':<22} {hf_exact:>19.1%} {openai_exact:>19.1%}  ({arrow}{diff:.1%})")

    hf_time = hf_results["elapsed_seconds"]
    openai_time = openai_results["elapsed_seconds"]
    print(f"\n  {'Time':<22} {hf_time:>18.1f}s {openai_time:>18.1f}s")
    print("=" * w + "\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the VERIS classifier on a held-out test split."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Max test examples to evaluate (0 = use full test set). Default: 50.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for classification. Default: gpt-4o.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42.",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use the fine-tuned HF model instead of OpenAI.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both HF and OpenAI models and compare results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_size = args.sample_size if args.sample_size > 0 else 0

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    test_set = load_test_set(sample_size, args.seed)

    if args.compare:
        # Run both models on the same test set
        print("\n" + "=" * 60)
        print("  Running comparison: Fine-tuned HF model vs OpenAI")
        print("=" * 60)

        hf_results = run_evaluation_hf(sample_size, args.seed, test_set)
        print_report(hf_results, label="Fine-tuned Model")

        openai_results = asyncio.run(
            run_evaluation_openai(args.model, sample_size, args.seed, test_set)
        )
        print_report(openai_results, label="GPT-4o Baseline")

        print_comparison(hf_results, openai_results)

        # Save both
        comparison = {
            "hf": hf_results,
            "openai": openai_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        output_path = EVAL_DIR / "comparison_results.json"
        with open(output_path, "w") as fh:
            json.dump(comparison, fh, indent=2)
        print(f"Comparison saved to {output_path}")

    elif args.use_hf:
        results = run_evaluation_hf(sample_size, args.seed, test_set)
        print_report(results, label="Fine-tuned Model")

        output_path = EVAL_DIR / "eval_results_hf.json"
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results saved to {output_path}")

    else:
        results = asyncio.run(
            run_evaluation_openai(args.model, sample_size, args.seed, test_set)
        )
        print_report(results)

        output_path = EVAL_DIR / "eval_results.json"
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
