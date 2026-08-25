import logging
from pathlib import Path

from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_pdf_to_images(
    pdf_path: str | Path,
    output_base_dir: str | Path,
    dpi: int = 300,
    input_root: str | Path = "data/pdf",
) -> None:
    """
    Convert a single PDF file to PNG images, preserving relative path under input_root.
    """
    pdf_path = Path(pdf_path).resolve()
    input_root = Path(input_root).resolve()
    output_base_dir = Path(output_base_dir).resolve()

    try:
        relative_path = pdf_path.relative_to(input_root)
    except ValueError:
        # fallback: just use the pdf stem in a flat folder
        relative_path = Path(pdf_path.name)

    doc_out_dir = output_base_dir / relative_path.parent / pdf_path.stem
    doc_out_dir.mkdir(parents=True, exist_ok=True)

    if pdf_path.stat().st_size == 0:
        logger.warning("[WARN] Empty PDF: %s — skipping...", pdf_path.name)
        return

    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        logger.error("[ERROR] Failed to convert %s: %s", pdf_path.name, str(e))
        return

    for idx, page in enumerate(pages, start=1):
        image_filename = f"{pdf_path.stem}_{idx:03}.png"
        image_path = doc_out_dir / image_filename
        page.save(image_path, "PNG")
        logger.info("Saved → %s", image_path)


def batch_convert_pdfs(
    input_dir: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
) -> None:
    """
    Recursively convert all PDF files in a directory into PNG images.
    Each PDF will generate a subfolder in the output directory, mirroring the input structure.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    pdf_files = list(input_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in %s.", input_dir)
        return

    for pdf_file in pdf_files:
        logger.info("Processing → %s", pdf_file)
        convert_pdf_to_images(pdf_file, output_dir, dpi, input_root=input_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs to Images for OCR")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/pdf",
        help="Directory with PDF files (default: data/pdf)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save output images (default: data/processed)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300)",
    )

    args = parser.parse_args()
    batch_convert_pdfs(args.input_dir, args.output_dir, args.dpi)
