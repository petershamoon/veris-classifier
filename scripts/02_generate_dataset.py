"""Generate synthetic incident descriptions from VCDB records using OpenAI."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

RAW_DATA = Path(__file__).parent.parent / "data" / "vcdb_raw" / "vcdb_all.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "dataset"

SYSTEM_PROMPT = """You are a cybersecurity analyst writing incident reports. Given a structured VERIS JSON
classification of a security incident, write a realistic 2-4 sentence incident report in natural language
that a security analyst might write or that might appear in a news article.

Rules:
- Write as if you're describing the incident to someone unfamiliar with VERIS
- Include enough detail that someone could classify it back into VERIS from your description
- Mention the type of attacker, what they did, what was affected, and the impact
- Use natural language, not VERIS terminology
- Vary your writing style - sometimes formal, sometimes more casual
- If the VERIS record has a 'summary' field, use it as inspiration but expand it into a fuller description
- Do NOT reference VERIS, the schema, or any classification terms directly"""


def extract_veris_fields(incident: dict) -> dict:
    """Extract the key VERIS classification fields from a full incident record."""
    return {
        "actor": incident.get("actor", {}),
        "action": incident.get("action", {}),
        "asset": incident.get("asset", {}),
        "attribute": incident.get("attribute", {}),
        "victim": {
            "industry": incident.get("victim", {}).get("industry", "Unknown"),
            "employee_count": incident.get("victim", {}).get("employee_count", "Unknown"),
            "country": incident.get("victim", {}).get("country", ["Unknown"]),
        },
        "timeline": incident.get("timeline", {}),
        "discovery_method": incident.get("discovery_method", {}),
        "targeted": incident.get("targeted", "Unknown"),
        "summary": incident.get("summary", ""),
    }


def build_classification_target(incident: dict) -> dict:
    """Build the structured classification output that the model should learn to produce."""
    target = {}

    # Actor
    actor = incident.get("actor", {})
    actor_info = {}
    for actor_type in ["external", "internal", "partner"]:
        if actor_type in actor:
            a = actor[actor_type]
            actor_info[actor_type] = {
                "variety": a.get("variety", []),
                "motive": a.get("motive", []),
            }
    target["actor"] = actor_info

    # Action
    action = incident.get("action", {})
    action_info = {}
    for action_type in ["malware", "hacking", "social", "misuse", "physical", "error", "environmental"]:
        if action_type in action:
            a = action[action_type]
            action_info[action_type] = {
                "variety": a.get("variety", []),
                "vector": a.get("vector", []),
            }
    target["action"] = action_info

    # Asset
    assets = incident.get("asset", {}).get("assets", [])
    target["asset"] = {"variety": [a.get("variety", "") for a in assets]}

    # Attribute
    attribute = incident.get("attribute", {})
    attr_info = {}
    for attr_type in ["confidentiality", "integrity", "availability"]:
        if attr_type in attribute:
            a = attribute[attr_type]
            info = {}
            if "data_disclosure" in a:
                info["data_disclosure"] = a["data_disclosure"]
            if "data" in a:
                info["data_variety"] = [d.get("variety", "") for d in a["data"]]
            if "variety" in a:
                info["variety"] = a["variety"]
            attr_info[attr_type] = info
    target["attribute"] = attr_info

    return target


def generate_description(client: OpenAI, incident: dict, model: str = "gpt-4o") -> str | None:
    """Generate a synthetic incident description from VERIS JSON."""
    veris_fields = extract_veris_fields(incident)

    user_msg = f"""Generate a realistic incident report for this security incident:

{json.dumps(veris_fields, indent=2)}

Write 2-4 sentences describing what happened."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.8,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating for {incident.get('incident_id', 'unknown')}: {e}")
        return None


def load_progress(output_file: Path) -> set:
    """Load already-processed incident IDs to support resume."""
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                row = json.loads(line)
                done.add(row["incident_id"])
    return done


def main():
    client = OpenAI()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "veris_train.jsonl"

    # Load incidents
    incidents = []
    with open(RAW_DATA) as f:
        for line in f:
            incidents.append(json.loads(line))
    print(f"Loaded {len(incidents)} incidents")

    # Resume support
    done = load_progress(output_file)
    remaining = [i for i in incidents if i.get("incident_id", "") not in done]
    print(f"Already processed: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("All incidents already processed!")
        return

    with open(output_file, "a") as f:
        for incident in tqdm(remaining, desc="Generating descriptions"):
            incident_id = incident.get("incident_id", "unknown")

            description = generate_description(client, incident)
            if description is None:
                continue

            classification = build_classification_target(incident)

            row = {
                "incident_id": incident_id,
                "description": description,
                "classification": classification,
                "original_summary": incident.get("summary", ""),
                "victim_industry": incident.get("victim", {}).get("industry", ""),
                "schema_version": incident.get("schema_version", ""),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

            # Rate limiting for gpt-4o
            time.sleep(0.15)

    # Final count
    done = load_progress(output_file)
    print(f"\nDataset complete: {len(done)} training examples")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
