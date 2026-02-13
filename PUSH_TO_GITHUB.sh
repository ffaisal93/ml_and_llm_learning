#!/bin/bash

# Script to push ml_and_llm_learning repository to GitHub
# Usage: ./PUSH_TO_GITHUB.sh

set -e  # Exit on error

echo "=========================================="
echo "Pushing ml_and_llm_learning to GitHub"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "Error: README.md not found. Are you in the correct directory?"
    exit 1
fi

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git already initialized"
fi

# Check if remote exists
if git remote | grep -q "origin"; then
    echo "✓ Remote 'origin' already exists"
    REMOTE_URL=$(git remote get-url origin)
    echo "  Current remote: $REMOTE_URL"
else
    echo "Adding remote 'origin'..."
    git remote add origin https://github.com/ffaisal93/ml_and_llm_learning.git
    echo "✓ Remote added"
fi

# Add all files
echo ""
echo "Adding files to git..."
git add .
echo "✓ Files added"

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "No changes to commit"
else
    echo "Creating commit..."
    git commit -m "Initial commit: ML & LLM learning repository with 32+ topics

- Classical ML algorithms (from scratch)
- Transformers and attention mechanisms
- LLM inference techniques (KV cache, quantization)
- Training techniques (RLHF, DPO)
- Business use cases with detailed solutions
- System design for ML
- A/B testing and experimentation
- Neural networks from scratch
- Anomaly detection (Isolation Forest)
- PyTorch fundamentals
- And much more!"
    echo "✓ Commit created"
fi

# Set main branch
echo ""
echo "Setting main branch..."
git branch -M main
echo "✓ Branch set to 'main'"

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo "Note: You may be prompted for GitHub credentials"
git push -u origin main

echo ""
echo "=========================================="
echo "✓ Successfully pushed to GitHub!"
echo "=========================================="
echo ""
echo "Repository URL:"
echo "https://github.com/ffaisal93/ml_and_llm_learning"
echo ""
echo "Next steps:"
echo "1. Go to the repository URL above"
echo "2. Verify all files are there"
echo "3. Add topics/tags on GitHub for discoverability"
echo ""

