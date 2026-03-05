"""Validate the generated VERIS training dataset and produce a quality report."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# --- Paths -------------------------------------------------------------------

DATASET_PATH = Path(__file__).parent.parent / "data" / "dataset" / "veris_train.jsonl"
REPORT_PATH = Path(__file__).parent.parent / "data" / "dataset" / "quality_report.json"

# --- Valid VERIS enums (imported inline to keep the script self-contained) -----

VALID_ACTOR_TYPES = {"external", "internal", "partner"}

VALID_ACTION_CATEGORIES = {
    "malware", "hacking", "social", "misuse", "physical", "error", "environmental",
}

VALID_ASSET_PREFIXES = {"S", "N", "U", "T", "M", "P"}

VALID_ATTRIBUTE_TYPES = {"confidentiality", "integrity", "availability"}

# Patterns that should NOT appear in natural-language descriptions.
VERIS_JARGON_PATTERNS = [
    r"action\.\w+\.variety",
    r"actor\.\w+\.variety",
    r"actor\.\w+\.motive",
    r"asset\.variety",
    r"attribute\.\w+\.variety",
    r"data_disclosure",
    r"data_variety",
    r"schema_version",
    r'"variety"\s*:',
    r'"vector"\s*:',
    r'"motive"\s*:',
]


# --- Helpers ------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_jargon(text: str) -> list[str]:
    """Return list of jargon patterns found in *text*."""
    hits = []
    for pat in VERIS_JARGON_PATTERNS:
        if re.search(pat, text):
            hits.append(pat)
    return hits


# --- Validation ---------------------------------------------------------------

def validate(rows: list[dict]) -> dict:
    report: dict = {}

    # ---- 1. Basic stats ------------------------------------------------------
    total = len(rows)
    has_desc = sum(1 for r in rows if r.get("description"))
    has_class = sum(1 for r in rows if r.get("classification"))
    desc_lengths = [len(r["description"]) for r in rows if r.get("description")]
    avg_desc_len = round(sum(desc_lengths) / len(desc_lengths), 1) if desc_lengths else 0
    min_desc_len = min(desc_lengths) if desc_lengths else 0
    max_desc_len = max(desc_lengths) if desc_lengths else 0

    report["basic_stats"] = {
        "total_rows": total,
        "rows_with_description": has_desc,
        "rows_with_classification": has_class,
        "avg_description_length": avg_desc_len,
        "min_description_length": min_desc_len,
        "max_description_length": max_desc_len,
    }

    # ---- 2. Classification coverage ------------------------------------------
    actor_type_counts: Counter = Counter()
    action_cat_counts: Counter = Counter()
    asset_prefix_counts: Counter = Counter()
    attribute_type_counts: Counter = Counter()

    for r in rows:
        cls = r.get("classification", {})

        # Actors
        for atype in cls.get("actor", {}):
            actor_type_counts[atype] += 1

        # Actions
        for acat in cls.get("action", {}):
            action_cat_counts[acat] += 1

        # Assets
        for variety in cls.get("asset", {}).get("variety", []):
            prefix = variety.split(" - ")[0].strip() if " - " in variety else variety
            asset_prefix_counts[prefix] += 1

        # Attributes
        for attr in cls.get("attribute", {}):
            attribute_type_counts[attr] += 1

    report["classification_coverage"] = {
        "actor_types": dict(actor_type_counts.most_common()),
        "action_categories": dict(action_cat_counts.most_common()),
        "asset_prefixes": dict(asset_prefix_counts.most_common()),
        "attribute_types": dict(attribute_type_counts.most_common()),
    }

    # ---- 3. Quality checks ---------------------------------------------------
    issues: dict = {}

    # 3a. Empty descriptions
    empty_desc = [r["incident_id"] for r in rows if not r.get("description")]
    issues["empty_descriptions"] = {"count": len(empty_desc), "incident_ids": empty_desc[:20]}

    # 3b. Empty classifications
    empty_class = [
        r["incident_id"]
        for r in rows
        if not r.get("classification")
        or (
            not r["classification"].get("actor")
            and not r["classification"].get("action")
            and not r["classification"].get("asset")
            and not r["classification"].get("attribute")
        )
    ]
    issues["empty_classifications"] = {"count": len(empty_class), "incident_ids": empty_class[:20]}

    # 3c. Duplicate incident_ids
    id_counts = Counter(r.get("incident_id", "") for r in rows)
    dupes = {iid: cnt for iid, cnt in id_counts.items() if cnt > 1}
    issues["duplicate_incident_ids"] = {"count": len(dupes), "ids": dupes}

    # 3d. Suspiciously short or long descriptions
    short_descs = [
        {"incident_id": r["incident_id"], "length": len(r["description"])}
        for r in rows
        if r.get("description") and len(r["description"]) < 50
    ]
    long_descs = [
        {"incident_id": r["incident_id"], "length": len(r["description"])}
        for r in rows
        if r.get("description") and len(r["description"]) > 1000
    ]
    issues["short_descriptions"] = {"count": len(short_descs), "threshold": "<50 chars", "samples": short_descs[:10]}
    issues["long_descriptions"] = {"count": len(long_descs), "threshold": ">1000 chars", "samples": long_descs[:10]}

    # 3e. VERIS jargon leaking into descriptions
    jargon_hits = []
    for r in rows:
        desc = r.get("description", "")
        found = check_jargon(desc)
        if found:
            jargon_hits.append({"incident_id": r["incident_id"], "patterns_found": found})
    issues["veris_jargon_in_descriptions"] = {"count": len(jargon_hits), "samples": jargon_hits[:10]}

    # 3f. Unknown actor / action / asset / attribute values
    unknown_actors = []
    unknown_actions = []
    unknown_assets = []
    unknown_attrs = []

    for r in rows:
        cls = r.get("classification", {})
        for atype in cls.get("actor", {}):
            if atype not in VALID_ACTOR_TYPES:
                unknown_actors.append({"incident_id": r["incident_id"], "value": atype})
        for acat in cls.get("action", {}):
            if acat not in VALID_ACTION_CATEGORIES:
                unknown_actions.append({"incident_id": r["incident_id"], "value": acat})
        for variety in cls.get("asset", {}).get("variety", []):
            prefix = variety.split(" - ")[0].strip() if " - " in variety else variety
            if prefix not in VALID_ASSET_PREFIXES and variety not in ("Unknown", "Other"):
                unknown_assets.append({"incident_id": r["incident_id"], "value": variety})
        for attr in cls.get("attribute", {}):
            if attr not in VALID_ATTRIBUTE_TYPES:
                unknown_attrs.append({"incident_id": r["incident_id"], "value": attr})

    issues["invalid_actor_types"] = {"count": len(unknown_actors), "samples": unknown_actors[:10]}
    issues["invalid_action_categories"] = {"count": len(unknown_actions), "samples": unknown_actions[:10]}
    issues["invalid_asset_values"] = {"count": len(unknown_assets), "samples": unknown_assets[:10]}
    issues["invalid_attribute_types"] = {"count": len(unknown_attrs), "samples": unknown_attrs[:10]}

    report["quality_checks"] = issues
    return report


# --- Formatted output ---------------------------------------------------------

def print_report(report: dict) -> None:
    sep = "=" * 72

    print(f"\n{sep}")
    print("  VERIS DATASET QUALITY REPORT")
    print(sep)

    # Basic stats
    bs = report["basic_stats"]
    print("\n--- Basic Stats ---")
    print(f"  Total rows:                {bs['total_rows']}")
    print(f"  Rows with description:     {bs['rows_with_description']}")
    print(f"  Rows with classification:  {bs['rows_with_classification']}")
    print(f"  Avg description length:    {bs['avg_description_length']} chars")
    print(f"  Min description length:    {bs['min_description_length']} chars")
    print(f"  Max description length:    {bs['max_description_length']} chars")

    # Classification coverage
    cc = report["classification_coverage"]

    print("\n--- Actor Types ---")
    for k, v in cc["actor_types"].items():
        pct = v / bs["total_rows"] * 100
        print(f"  {k:<20s} {v:>6d}  ({pct:5.1f}%)")

    print("\n--- Action Categories ---")
    for k, v in cc["action_categories"].items():
        pct = v / bs["total_rows"] * 100
        print(f"  {k:<20s} {v:>6d}  ({pct:5.1f}%)")

    print("\n--- Asset Prefixes ---")
    for k, v in cc["asset_prefixes"].items():
        pct = v / bs["total_rows"] * 100
        print(f"  {k:<20s} {v:>6d}  ({pct:5.1f}%)")

    print("\n--- Attribute Types ---")
    for k, v in cc["attribute_types"].items():
        pct = v / bs["total_rows"] * 100
        print(f"  {k:<20s} {v:>6d}  ({pct:5.1f}%)")

    # Quality checks
    qc = report["quality_checks"]

    print(f"\n--- Quality Checks ---")
    all_good = True

    def flag(label: str, count: int, samples: list | dict | None = None):
        nonlocal all_good
        status = "PASS" if count == 0 else "WARN"
        if count > 0:
            all_good = False
        print(f"  [{status}] {label}: {count}")
        if count > 0 and samples:
            items = samples if isinstance(samples, list) else list(samples.items())
            for item in items[:5]:
                print(f"         - {item}")

    flag("Empty descriptions", qc["empty_descriptions"]["count"], qc["empty_descriptions"]["incident_ids"])
    flag("Empty classifications", qc["empty_classifications"]["count"], qc["empty_classifications"]["incident_ids"])
    flag("Duplicate incident IDs", qc["duplicate_incident_ids"]["count"], qc["duplicate_incident_ids"]["ids"])
    flag("Short descriptions (<50 chars)", qc["short_descriptions"]["count"], qc["short_descriptions"]["samples"])
    flag("Long descriptions (>1000 chars)", qc["long_descriptions"]["count"], qc["long_descriptions"]["samples"])
    flag("VERIS jargon in descriptions", qc["veris_jargon_in_descriptions"]["count"], qc["veris_jargon_in_descriptions"]["samples"])
    flag("Invalid actor types", qc["invalid_actor_types"]["count"], qc["invalid_actor_types"]["samples"])
    flag("Invalid action categories", qc["invalid_action_categories"]["count"], qc["invalid_action_categories"]["samples"])
    flag("Invalid asset values", qc["invalid_asset_values"]["count"], qc["invalid_asset_values"]["samples"])
    flag("Invalid attribute types", qc["invalid_attribute_types"]["count"], qc["invalid_attribute_types"]["samples"])

    print()
    if all_good:
        print("  All quality checks passed.")
    else:
        print("  Some quality checks flagged issues -- review the report above.")

    print(f"\n{sep}\n")


# --- Main ---------------------------------------------------------------------

def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {DATASET_PATH} ...")
    rows = load_dataset(DATASET_PATH)

    if not rows:
        print("ERROR: Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    report = validate(rows)
    print_report(report)

    # Save JSON report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
