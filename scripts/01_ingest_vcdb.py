"""Pull all validated VCDB incident JSON files from GitHub."""

import json
import os
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "vcdb_raw"


def clone_vcdb():
    """Clone or update the VCDB repo."""
    vcdb_path = Path(__file__).parent.parent / "data" / "VCDB"
    if vcdb_path.exists():
        print("VCDB repo already cloned, pulling latest...")
        subprocess.run(["git", "-C", str(vcdb_path), "pull"], check=True)
    else:
        print("Cloning VCDB repo (this may take a minute)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/vz-risk/VCDB.git", str(vcdb_path)],
            check=True,
        )
    return vcdb_path


def extract_incidents(vcdb_path: Path) -> list[dict]:
    """Parse all validated incident JSON files."""
    validated_dir = vcdb_path / "data" / "json" / "validated"
    if not validated_dir.exists():
        raise FileNotFoundError(f"Validated directory not found: {validated_dir}")

    incidents = []
    errors = []
    for json_file in sorted(validated_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                incident = json.load(f)
            incidents.append(incident)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            errors.append((json_file.name, str(e)))

    print(f"Parsed {len(incidents)} incidents ({len(errors)} errors)")
    if errors:
        for name, err in errors[:5]:
            print(f"  Error in {name}: {err}")
    return incidents


def summarize_incidents(incidents: list[dict]):
    """Print stats about the ingested incidents."""
    has_summary = sum(1 for i in incidents if i.get("summary"))
    has_reference = sum(1 for i in incidents if i.get("reference"))

    actors = {"external": 0, "internal": 0, "partner": 0}
    for i in incidents:
        for actor_type in actors:
            if actor_type in i.get("actor", {}):
                actors[actor_type] += 1

    actions = {"malware": 0, "hacking": 0, "social": 0, "misuse": 0, "physical": 0, "error": 0, "environmental": 0}
    for i in incidents:
        for action_type in actions:
            if action_type in i.get("action", {}):
                actions[action_type] += 1

    print(f"\n--- VCDB Dataset Summary ---")
    print(f"Total incidents:    {len(incidents)}")
    print(f"With summary text:  {has_summary}")
    print(f"With reference URL: {has_reference}")
    print(f"\nActor distribution:")
    for k, v in actors.items():
        print(f"  {k:12s}: {v:>5d} ({v/len(incidents)*100:.1f}%)")
    print(f"\nAction distribution:")
    for k, v in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v:>5d} ({v/len(incidents)*100:.1f}%)")


def save_incidents(incidents: list[dict]):
    """Save parsed incidents to a single JSONL file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "vcdb_all.jsonl"
    with open(output, "w") as f:
        for incident in incidents:
            f.write(json.dumps(incident) + "\n")
    print(f"\nSaved to {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    vcdb_path = clone_vcdb()
    incidents = extract_incidents(vcdb_path)
    summarize_incidents(incidents)
    save_incidents(incidents)


if __name__ == "__main__":
    main()
