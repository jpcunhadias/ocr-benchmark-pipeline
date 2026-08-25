#!/bin/bash
# Setup script for local development without Docker

echo "Setting up local OCR pipeline development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

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
echo "1. source venv/bin/activate"
echo "2. export \$(cat .env.local | xargs)"
echo "3. Run individual scripts directly with python"
echo ""
echo "Example workflow:"
echo "# Step 1: Convert PDFs to images"
echo "python scripts/data_prep/convert_pdfs_to_images.py --input_dir data/pdf/2025-04/document1 --output_dir data/processed --dpi 300"
echo ""
echo "# Step 2: Run OCR"
echo "python scripts/ocr/run_benchmark.py --config configs/engines/tesseract.yaml --data_dir data/processed --output_dir results/tesseract"
echo ""
echo "# Step 3: Extract fields"
echo "python scripts/ocr/extract_from_json.py --engine tesseract --month 2025-04 --results_dir results"
