#!/bin/bash
# Setup script for local development without Docker

echo "Setting up local OCR pipeline development environment..."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required -- install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Creates .venv/ and installs runtime + dev dependencies from uv.lock
# (pyproject.toml + uv.lock are the source of truth; requirements.txt is
# generated from them for the Dockerfile's benefit -- see README).
echo "Syncing dependencies with uv..."
uv sync

# Create local environment file
if [ ! -f ".env.local" ]; then
    echo "Creating local environment file..."
    cat > .env.local << EOF
# Local development environment (no Docker services)
PIPELINE_PUBLISH_ENABLED=false
PIPELINE_USE_MINIO=false
SAVE_CSVS=true

# Disable database connections
PG_URL=
MONGO_URI=
MINIO_ENDPOINT=

# Local directories
PYTHONPATH=$(pwd)
EOF
fi

echo "Local development environment ready!"
echo ""
echo "To use:"
echo "1. export \$(cat .env.local | xargs)"
echo "2. Run scripts via 'uv run python ...' (no activation needed), or"
echo "   'source .venv/bin/activate' if you'd rather activate the venv directly"
echo ""
echo "Example workflow:"
echo "# Step 1: Convert PDFs to images"
echo "uv run python scripts/data_prep/convert_pdfs_to_images.py --input_dir data/pdf/2025-04/document1 --output_dir data/processed --dpi 300"
echo ""
echo "# Step 2: Run OCR"
echo "uv run python scripts/ocr/run_benchmark.py --config configs/engines/tesseract.yaml --data_dir data/processed --output_dir results/tesseract"
echo ""
echo "# Step 3: Extract fields"
echo "uv run python scripts/ocr/extract_from_json.py --engine tesseract --month 2025-04 --results_dir results"
