"""Fast async dataset generation - runs 15 concurrent requests to OpenAI."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

RAW_DATA = Path(__file__).parent.parent / "data" / "vcdb_raw" / "vcdb_all.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "dataset"

CONCURRENCY = 10  # conservative for overnight run - stays well under 200K TPM / 500 RPM limits

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
    target = {}
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

    assets = incident.get("asset", {}).get("assets", [])
    target["asset"] = {"variety": [a.get("variety", "") for a in assets]}

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


def load_progress(output_file: Path) -> set:
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add(row["incident_id"])
                except json.JSONDecodeError:
                    continue
    return done


async def generate_one(
    client: AsyncOpenAI,
    incident: dict,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o-mini",
    max_retries: int = 10,
) -> dict | None:
    incident_id = incident.get("incident_id", "unknown")
    veris_fields = extract_veris_fields(incident)
    user_msg = f"""Generate a realistic incident report for this security incident:

{json.dumps(veris_fields, indent=2)}

Write 2-4 sentences describing what happened."""

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.8,
                    max_tokens=300,
                )
                description = response.choices[0].message.content.strip()
                return {
                    "incident_id": incident_id,
                    "description": description,
                    "classification": build_classification_target(incident),
                    "original_summary": incident.get("summary", ""),
                    "victim_industry": incident.get("victim", {}).get("industry", ""),
                    "schema_version": incident.get("schema_version", ""),
                }
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    if "per day" in err or "RPD" in err:
                        # Daily limit hit - sleep 10 min and retry
                        print(f"\n  Daily limit hit, sleeping 10 min... ({incident_id})")
                        await asyncio.sleep(600)
                    else:
                        # Per-minute limit - shorter backoff
                        await asyncio.sleep(3 * (2 ** attempt))
                    continue
                if attempt == max_retries - 1:
                    print(f"\nFailed after {max_retries} retries for {incident_id}: {e}")
                return None


async def main():
    client = AsyncOpenAI()
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

    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Process in batches and write results as they come
    batch_size = 100
    with open(output_file, "a") as f:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start : batch_start + batch_size]
            tasks = [generate_one(client, inc, semaphore) for inc in batch]
            results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"Batch {batch_start // batch_size + 1}/{(len(remaining) + batch_size - 1) // batch_size}",
            )
            for result in results:
                if result is not None:
                    f.write(json.dumps(result) + "\n")
            f.flush()

            completed = len(done) + batch_start + len(batch)
            print(f"  Progress: {completed}/{len(incidents)} total")

    done = load_progress(output_file)
    print(f"\nDataset complete: {len(done)} training examples")


if __name__ == "__main__":
    asyncio.run(main())
