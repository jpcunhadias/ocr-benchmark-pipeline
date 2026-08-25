#!/usr/bin/env python
"""Local OCR pipeline runner - no Docker, no databases, just file processing."""

import argparse
import json
import os
import sys
from pathlib import Path

# Set up environment for local development
os.environ["PIPELINE_PUBLISH_ENABLED"] = "false"
os.environ["PIPELINE_USE_MINIO"] = "false"
os.environ["SAVE_CSVS"] = "true"

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_directories():
    """Create necessary local directories."""
    dirs = ["data/pdf", "data/processed", "results"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def convert_pdfs(input_dir: str, output_dir: str, dpi: int = 300):
    """Convert PDFs to images."""
    print(f"Converting PDFs from {input_dir} to {output_dir}")

    from scripts.data_prep.convert_pdfs_to_images import batch_convert_pdfs

    batch_convert_pdfs(input_dir, output_dir, dpi=dpi)
    print("PDF conversion completed")


def run_ocr(config_path: str, data_dir: str, output_dir: str):
    """Run OCR on processed images."""
    print(f"Running OCR with config {config_path}")

    from src.ocr_engines.utils import load_engine

    engine = load_engine(config_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_files = list(Path(data_dir).rglob("*.png"))
    print(f"Found {len(image_files)} images to process")

    results = []
    for img_path in image_files:
        try:
            result = engine.predict(str(img_path))
            results.append({"image_path": str(img_path), "ocr_result": result})
            print(f"Processed: {img_path.name}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    results_file = Path(output_dir) / "ocr_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"OCR completed. Results saved to {results_file}")
    return results


def extract_fields(results_dir: str, engine: str):
    """Clean the raw OCR text for each page and write it to a text file."""
    print(f"Extracting text from {results_dir}")

    results_file = Path(results_dir) / "ocr_results.json"

    if not results_file.exists():
        print(f"No results file found at {results_file}")
        return

    from src.data.preprocess import preprocess_ocr_text

    with open(results_file) as f:
        results = json.load(f)

    extracted_file = Path(results_dir) / "extracted_text.txt"
    with open(extracted_file, "w") as f:
        for result in results:
            raw_text = (result.get("ocr_result") or {}).get("text", "")
            f.write(f"=== {result['image_path']} ===\n")
            f.write(f"{preprocess_ocr_text(raw_text)}\n\n")

    print(f"Text extraction completed. Saved to {extracted_file}")


def main():
    parser = argparse.ArgumentParser(description="Local OCR Pipeline (no Docker)")
    parser.add_argument(
        "--step",
        choices=["convert", "ocr", "extract", "all"],
        default="all",
        help="Which step to run",
    )
    parser.add_argument("--input", help="Input directory containing PDFs")
    parser.add_argument("--engine", default="tesseract", help="OCR engine")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI")

    args = parser.parse_args()
    setup_directories()

    if not args.input:
        pdf_dirs = list(Path("data/pdf").rglob("*"))
        pdf_dirs = [d for d in pdf_dirs if d.is_dir() and list(d.glob("*.pdf"))]

        if not pdf_dirs:
            print("No PDFs found in data/pdf/. Please:")
            print("  1. Put PDFs in data/pdf/<period>/<document>/")
            print("  2. Or specify --input <path_to_pdf_dir>")
            return

        args.input = str(pdf_dirs[0])
        print(f"Using PDF directory: {args.input}")

    input_dir = Path(args.input)
    processed_dir = Path("data/processed") / input_dir.name
    results_dir = Path("results") / args.engine / input_dir.name

    config_path = f"configs/engines/{args.engine}.yaml"

    if args.step in ["convert", "all"]:
        convert_pdfs(str(input_dir), str(processed_dir), args.dpi)

    if args.step in ["ocr", "all"]:
        if not processed_dir.exists():
            print(f"No processed images at {processed_dir}. Run convert step first.")
            return
        run_ocr(config_path, str(processed_dir), str(results_dir))

    if args.step in ["extract", "all"]:
        extract_fields(str(results_dir), args.engine)

    print("Local pipeline completed!")
    print(f"Results available in: {results_dir}")


if __name__ == "__main__":
    main()
