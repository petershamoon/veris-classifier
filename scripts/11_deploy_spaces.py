"""Deploy the VERIS Classifier to HuggingFace Spaces.

Uploads app.py, requirements.txt, src/, and README to a Gradio Space
with ZeroGPU support.

Usage:
    python scripts/11_deploy_spaces.py              # deploy to vibesecurityguy/veris-classifier
    python scripts/11_deploy_spaces.py --dry-run    # preview what would be uploaded
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPACE_ID = "vibesecurityguy/veris-classifier"

# Files to upload to the Space
FILES_TO_UPLOAD = {
    "app.py": PROJECT_ROOT / "app.py",
    "requirements.txt": PROJECT_ROOT / "requirements.txt",
    "README.md": PROJECT_ROOT / "spaces" / "README.md",
    "src/veris_classifier/__init__.py": PROJECT_ROOT / "src" / "veris_classifier" / "__init__.py",
    "src/veris_classifier/classifier.py": PROJECT_ROOT / "src" / "veris_classifier" / "classifier.py",
    "src/veris_classifier/enums.py": PROJECT_ROOT / "src" / "veris_classifier" / "enums.py",
    "src/veris_classifier/validator.py": PROJECT_ROOT / "src" / "veris_classifier" / "validator.py",
}


def main():
    parser = argparse.ArgumentParser(description="Deploy VERIS Classifier to HF Spaces")
    parser.add_argument("--dry-run", action="store_true", help="Preview files without uploading")
    parser.add_argument("--space-id", default=SPACE_ID, help=f"Space ID (default: {SPACE_ID})")
    args = parser.parse_args()

    print(f"\nDeploying to: https://huggingface.co/spaces/{args.space_id}")
    print(f"Files to upload ({len(FILES_TO_UPLOAD)}):\n")

    for repo_path, local_path in FILES_TO_UPLOAD.items():
        exists = local_path.exists()
        size = local_path.stat().st_size if exists else 0
        status = f"{size:,} bytes" if exists else "MISSING"
        print(f"  {repo_path:<50} {status}")

    # Check for missing files
    missing = [p for p in FILES_TO_UPLOAD.values() if not p.exists()]
    if missing:
        print(f"\nERROR: {len(missing)} files missing!")
        for p in missing:
            print(f"  - {p}")
        return

    if args.dry_run:
        print("\n[DRY RUN] No files uploaded.")
        return

    # Create Space and upload
    api = HfApi()

    print(f"\nCreating Space: {args.space_id}...")
    api.create_repo(
        args.space_id,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",  # ZeroGPU — free A10G burst (requires HF Pro)
        exist_ok=True,
        private=False,
    )

    print("Uploading files...")
    for repo_path, local_path in FILES_TO_UPLOAD.items():
        print(f"  {repo_path}...", end=" ")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=args.space_id,
            repo_type="space",
        )
        print("done")

    print(f"""
{'=' * 60}
  Deployment complete!

  Space URL: https://huggingface.co/spaces/{args.space_id}
  Model:     vibesecurityguy/veris-classifier-v1
  Hardware:  ZeroGPU (A10G)

  The Space will build and start automatically.
  First request may take 30-60s to load the model.
{'=' * 60}
""")


if __name__ == "__main__":
    main()
