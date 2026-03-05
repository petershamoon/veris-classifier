"""Generate VERIS Q&A training pairs for fine-tuning the Ask About VERIS feature."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "dataset"
OUTPUT_FILE = OUTPUT_DIR / "veris_qa.jsonl"
CONCURRENCY = 8  # conservative to avoid rate limits

# ── VERIS knowledge base for Q&A generation ──────────────────────────────

VERIS_KNOWLEDGE = """
VERIS (Vocabulary for Event Recording and Incident Sharing) is a structured framework
created by the Verizon RISK Team for consistently describing security incidents. It is
the framework behind the annual Verizon Data Breach Investigations Report (DBIR).

The VERIS framework uses the "4 A's" to describe any incident:

1. ACTORS - Who is behind the incident?
   Types: External, Internal, Partner
   External varieties: Activist, Auditor, Competitor, Customer, Force majeure, Former employee,
     Nation-state, Organized crime, Acquaintance, State-affiliated, Terrorist, Unaffiliated
   Internal varieties: Auditor, Call center, Cashier, End-user, Executive, Finance, Helpdesk,
     Human resources, Maintenance, Manager, Guard, Developer, System admin
   Motives: NA, Espionage, Fear, Financial, Fun, Grudge, Ideology, Convenience

2. ACTIONS - What techniques were used?
   Categories: Malware, Hacking, Social, Misuse, Physical, Error, Environmental

   Malware varieties: Adware, Backdoor, Brute force, Capture app data, Capture stored data,
     C2, Destroy data, Disable controls, DoS, Downloader, Exploit vuln, Export data,
     Ransomware, Rootkit, Spyware/Keylogger, SQL injection, Ram scraper, Worm

   Hacking varieties: Abuse of functionality, Brute force, Buffer overflow, DoS, MitM,
     SQLi, XSS, Use of stolen creds, Use of backdoor or C2, Path traversal, RFI,
     OS commanding, Offline cracking, Forced browsing, Session replay

   Social varieties: Baiting, Bribery, Elicitation, Extortion, Forgery, Influence,
     Phishing, Pretexting, Propaganda, Scam, Spam

   Misuse varieties: Data mishandling, Email misuse, Illicit content, Knowledge abuse,
     Net misuse, Privilege abuse, Unapproved hardware, Unapproved software,
     Unapproved workaround, Possession abuse, Embezzlement

   Physical varieties: Assault, Destruction, Disabled controls, Skimmer, Surveillance,
     Tampering, Theft, Snooping

   Error varieties: Classification error, Data entry error, Disposal error, Gaffe,
     Loss, Malfunction, Misconfiguration, Misdelivery, Misinformation, Omission,
     Physical accident, Printing error, Programming error, Publishing error,
     Capacity shortage

3. ASSETS - What was affected?
   Prefixes: S (Server), N (Network), U (User device), M (Media), P (Person),
     T (Terminal/Kiosk), E (Embedded)

   Server examples: S - Web application, S - Database, S - Mail, S - File, S - DNS,
     S - DHCP, S - Directory (LDAP, AD), S - Auth, S - Backup
   User device examples: U - Desktop, U - Laptop, U - Tablet, U - Phone, U - POS terminal
   Media examples: M - Documents, M - Flash drive, M - Disk, M - Tapes, M - Payment card
   Network examples: N - Router or switch, N - Firewall, N - IDS, N - WAP, N - Private WAN

4. ATTRIBUTES - What was the impact?
   Types: Confidentiality, Integrity, Availability

   Confidentiality data varieties: Medical, Payment, Bank, Credentials, Personal,
     Internal, System, Classified, Copyrighted, Unknown
   Data disclosure levels: Yes, Potentially, No, Unknown

   Integrity varieties: Alter behavior, Created account, Defacement, Fraudulent transaction,
     Hardware tampering, Log tampering, Misrepresentation, Modify configuration,
     Modify data, Modify privileges, Software installation, Unknown

   Availability varieties: Acceleration, Degradation, Destruction, Interruption, Loss,
     Obscuration, Unknown

Additional VERIS concepts:
- The A4 Grid: 315 possible combinations of Actor × Action types
- Discovery methods: Int - antivirus, Int - incident response, Int - log review,
  Ext - customer, Ext - law enforcement, Ext - fraud detection, Prt - audit, etc.
- Timeline: compromise, exfiltration, discovery, containment timestamps
- Victim demographics: industry (NAICS codes), organization size, country
- VCDB: VERIS Community Database with 10,000+ publicly disclosed incidents
- Schema versions: 1.2, 1.3, 1.3.1, 1.3.2, 1.3.3, 1.3.4, 1.3.5, 1.3.6, 1.3.7, 1.4
"""

# ── Q&A prompt categories ────────────────────────────────────────────────

QA_CATEGORIES = [
    {
        "category": "framework_basics",
        "count": 40,
        "description": "Basic questions about what VERIS is, its purpose, history, and structure",
        "examples": [
            "What is VERIS?",
            "What does VERIS stand for?",
            "Who created the VERIS framework?",
            "What is the purpose of VERIS?",
            "How is VERIS used in the real world?",
        ],
    },
    {
        "category": "four_as",
        "count": 50,
        "description": "Questions about the 4 A's framework (Actors, Actions, Assets, Attributes)",
        "examples": [
            "What are the 4 A's in VERIS?",
            "How do the 4 A's work together to describe an incident?",
            "Can an incident have multiple action types?",
            "What's the difference between an Actor and an Action?",
        ],
    },
    {
        "category": "actors",
        "count": 50,
        "description": "Questions about actor types, varieties, and motives",
        "examples": [
            "What are the three actor types in VERIS?",
            "What's the difference between External and Internal actors?",
            "What motives does VERIS track?",
            "Give me examples of nation-state actor incidents",
            "When would you classify someone as a Partner actor?",
        ],
    },
    {
        "category": "actions",
        "count": 60,
        "description": "Questions about the 7 action categories and their varieties",
        "examples": [
            "What are the 7 action categories?",
            "What's the difference between Hacking and Malware?",
            "What does the Social action category cover?",
            "When should I use Misuse vs Error?",
            "What is a RAM scraper in VERIS malware terms?",
        ],
    },
    {
        "category": "assets",
        "count": 40,
        "description": "Questions about asset types and varieties",
        "examples": [
            "What asset prefixes does VERIS use?",
            "What does 'S - Web application' mean?",
            "How do I classify a stolen laptop?",
            "What's the difference between S and N prefixes?",
        ],
    },
    {
        "category": "attributes",
        "count": 40,
        "description": "Questions about the CIA triad in VERIS (confidentiality, integrity, availability)",
        "examples": [
            "How does VERIS handle confidentiality impact?",
            "What are the data disclosure levels?",
            "When is integrity affected vs confidentiality?",
            "What data varieties does VERIS track?",
        ],
    },
    {
        "category": "classification_guidance",
        "count": 70,
        "description": "How-to questions about classifying specific scenarios",
        "examples": [
            "How would I classify a phishing attack that led to ransomware?",
            "How do I classify an insider who stole customer data?",
            "What if an employee accidentally emailed PII to the wrong person?",
            "How do I handle a DDoS attack in VERIS?",
            "How should I classify a lost laptop with unencrypted data?",
        ],
    },
    {
        "category": "vcdb_and_dbir",
        "count": 30,
        "description": "Questions about the VERIS Community Database and DBIR",
        "examples": [
            "What is VCDB?",
            "How many incidents are in VCDB?",
            "What is the DBIR?",
            "How does VERIS relate to the DBIR?",
            "What industries are most represented in VCDB?",
        ],
    },
    {
        "category": "advanced_concepts",
        "count": 40,
        "description": "Advanced topics like A4 grid, discovery methods, timeline, comparisons",
        "examples": [
            "What is the A4 grid?",
            "How many combinations are in the A4 grid?",
            "How does VERIS handle discovery methods?",
            "How does VERIS compare to MITRE ATT&CK?",
            "What timeline information does VERIS capture?",
        ],
    },
    {
        "category": "enumeration_details",
        "count": 80,
        "description": "Specific questions about individual enumeration values and their meanings",
        "examples": [
            "What does Pretexting mean in VERIS?",
            "What's the difference between SQLi and SQL injection in VERIS?",
            "What does 'Use of stolen creds' mean?",
            "What is Privilege abuse under Misuse?",
            "What does data disclosure 'Potentially' mean?",
        ],
    },
]

GENERATION_PROMPT = """You are generating training data for a VERIS (Vocabulary for Event Recording
and Incident Sharing) expert Q&A system. Generate {count} unique question-answer pairs about: {description}

Use this VERIS knowledge as your reference:
{knowledge}

Here are some example questions for this category (generate DIFFERENT ones, more diverse):
{examples}

Requirements:
- Each question should be something a security professional would realistically ask
- Answers should be accurate, detailed (2-5 sentences), and reference specific VERIS terms/values
- Vary question complexity: some basic, some intermediate, some advanced
- Include specific enum values and examples in answers where relevant
- DON'T repeat the example questions - generate new, unique ones
- Cover the full breadth of the category

Return a JSON array of objects with "question" and "answer" fields.
Return ONLY the JSON array, no other text."""


async def generate_qa_batch(
    client: AsyncOpenAI,
    category: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 10,
) -> list[dict]:
    """Generate Q&A pairs for a category."""
    prompt = GENERATION_PROMPT.format(
        count=category["count"],
        description=category["description"],
        knowledge=VERIS_KNOWLEDGE,
        examples="\n".join(f"  - {q}" for q in category["examples"]),
    )

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a VERIS framework expert generating high-quality Q&A training data."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.9,
                    max_tokens=8000,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()
                data = json.loads(content)

                # Handle both {"pairs": [...]} and direct [...] formats
                if isinstance(data, list):
                    pairs = data
                elif isinstance(data, dict):
                    # Find the array in the dict
                    for v in data.values():
                        if isinstance(v, list):
                            pairs = v
                            break
                    else:
                        pairs = []
                else:
                    pairs = []

                # Tag each pair with the category
                for pair in pairs:
                    pair["category"] = category["category"]

                return pairs

            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    if "per day" in err or "RPD" in err:
                        print(f"\n  Daily limit hit, sleeping 10 min... ({category['category']})")
                        await asyncio.sleep(600)
                    else:
                        await asyncio.sleep(3 * (2 ** attempt))
                    continue
                if attempt == max_retries - 1:
                    print(f"\nFailed for {category['category']}: {e}")
                return []


async def main():
    client = AsyncOpenAI()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_target = sum(c["count"] for c in QA_CATEGORIES)
    print(f"Generating ~{total_target} Q&A pairs across {len(QA_CATEGORIES)} categories...")

    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Generate all categories concurrently
    tasks = [generate_qa_batch(client, cat, semaphore) for cat in QA_CATEGORIES]
    results = await asyncio.gather(*tasks)

    # Flatten and save
    all_pairs = []
    for pairs in results:
        all_pairs.extend(pairs)

    print(f"\nGenerated {len(all_pairs)} Q&A pairs")

    # Print distribution
    from collections import Counter
    dist = Counter(p.get("category", "unknown") for p in all_pairs)
    for cat, count in dist.most_common():
        print(f"  {cat}: {count}")

    # Save as JSONL
    with open(OUTPUT_FILE, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nSaved to {OUTPUT_FILE}")

    # Show a few samples
    print("\n--- Samples ---")
    import random
    for pair in random.sample(all_pairs, min(5, len(all_pairs))):
        print(f"\nQ: {pair['question']}")
        print(f"A: {pair['answer'][:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
