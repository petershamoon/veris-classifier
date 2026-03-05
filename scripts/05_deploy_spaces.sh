#!/bin/bash
# Deploy the Gradio app to Hugging Face Spaces
# Usage: ./scripts/05_deploy_spaces.sh [hf-username]

set -e

HF_USER="${1:-vibesecurityguy}"
SPACE_NAME="veris-classifier"
REPO_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

echo "Deploying to ${REPO_URL}..."

# Create a temp directory for the space
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Copy required files
cp app.py "$TMPDIR/"
cp requirements.txt "$TMPDIR/"
cp -r src/ "$TMPDIR/"
cp spaces/README.md "$TMPDIR/README.md"

# Init git and push
cd "$TMPDIR"
git init
git add .
git commit -m "Deploy VERIS Classifier — fine-tuned Qwen2.5-7B + ZeroGPU"
git remote add space "https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"
git push space main --force

echo ""
echo "Deployed! Visit: ${REPO_URL}"
echo ""
echo "The app uses the fine-tuned model on ZeroGPU — no API key needed!"
echo "Users can optionally provide an OpenAI key for GPT-4o fallback."
