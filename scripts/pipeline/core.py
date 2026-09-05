"""
Core OCR pipeline functions shared between interactive and non-interactive modes.
Extracted from run_ocr_pipeline.py and pipeline_loop.py to eliminate duplication.
"""

import os
import shutil
import time
from pathlib import Path

from minio import Minio

from scripts.data_prep.convert_pdfs_to_images import batch_convert_pdfs
from scripts.ocr.extract_from_json import run_extraction_stage
from scripts.ocr.run_benchmark import run_ocr_benchmark_and_publish
from scripts.validation.validate_pdf_pages import validate_folder
from src.data.minio_client import download_files, list_files
from src.io.publish import register_document
from src.ocr_engines.utils import load_engine
from src.utils.create_ids import make_doc_id
from src.utils.logger import get_logger
from src.utils.monitor_and_save_resources import monitor_pipeline_resources
from src.utils.runtime import pipeline_use_minio

logger = get_logger(__name__)

BUCKET_NAME = os.getenv("MINIO_BUCKET", "ocr-artifacts")
LOCAL_PDF_DIR = Path("data/pdf")
LOCAL_IMG_DIR = Path("data/processed/_tmp_images")
RESULTS_ROOT = Path("results")


def clean_temp_dirs() -> None:
    """Remove only temp files/dirs inside our scratch dirs (preserve .gitkeep)."""
    for folder in (LOCAL_PDF_DIR, LOCAL_IMG_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def list_subfolders(minio_client: Minio, base_prefix: str) -> list[str]:
    """
    List document folders under base_prefix.
    Returns sorted list of folder names (not full paths).
    """
    seen: set[str] = set()
    for obj in minio_client.list_objects(
        BUCKET_NAME, prefix=base_prefix, recursive=True
    ):
        if obj.object_name is None:
            continue
        remainder = obj.object_name[len(base_prefix) :]
        top = remainder.split("/", 1)[0]
        if top:
            seen.add(top)
    return sorted(seen)


def process_folder(
    minio_client: Minio | None,
    base_prefix: str,
    folder: str,
    config_path: str,
    month: str,
    run_id: str,
    engine_name: str,
    validate: bool = False,
    keep_json: bool = False,
) -> None:
    """
    Process a single document folder through the OCR pipeline.

    Args:
        minio_client: MinIO client instance
        base_prefix: Base path on MinIO (e.g., "acme/2025-04/")
        folder: Document folder name
        config_path: Path to OCR engine config file
        month: Period string (e.g., "2025-04")
        run_id: Pipeline run identifier
        engine_name: OCR engine name
        validate: Whether to validate PDF pages
        keep_json: Whether to keep intermediate JSON files
    """
    start_time = time.time()
    logger.info(f"Processing document: {folder}")

    clean_temp_dirs()

    # Extract source label from base_prefix
    source = base_prefix.split("/")[0]

    # Create document ID
    document_id = make_doc_id(document_name=folder, period=month)

    doc_row = {
        "document_id": document_id,
        "document_name": folder,
        "month": month,
        "source": source,
        "department": None,
        "source_path": f"{base_prefix}{folder}/",
        "engine": engine_name,
        "pipeline_version": "1",
        "run_id": run_id,
    }

    register_document(doc_row)

    prefix_on_minio = f"{base_prefix}{folder}/"
    local_pdf_root = LOCAL_PDF_DIR / month / folder
    local_img_root = LOCAL_IMG_DIR / month / folder

    local_pdf_root.mkdir(parents=True, exist_ok=True)
    local_img_root.mkdir(parents=True, exist_ok=True)

    # Download PDFs
    source_objects: dict[str, str] = {}
    if pipeline_use_minio():
        if minio_client is None:
            raise RuntimeError(
                "MinIO usage enabled but client is None. Check configuration."
            )
        logger.info(f"Downloading PDFs from {prefix_on_minio}")
        download_files(minio_client, BUCKET_NAME, prefix_on_minio, local_pdf_root)

        # Record exactly which MinIO object each PDF came from, keyed by
        # filename stem (== the `document` value run_benchmark() later
        # produces) -- lets the dashboard's box-overlay preview fetch a
        # page's source PDF directly by object key instead of re-deriving
        # it later via documents.source_path + a filename-stem search.
        source_objects = {
            Path(name).stem: name
            for name in list_files(minio_client, BUCKET_NAME, prefix_on_minio)
            if name.lower().endswith(".pdf")
        }
    else:
        logger.info(
            "MinIO download disabled; using existing PDFs at %s", local_pdf_root
        )

    # Optional: Validate PDF pages
    if validate:
        logger.info("Validating PDF pages...")
        validate_folder(local_pdf_root, month, folder)

    # Convert PDFs to images
    logger.info("Converting PDFs to images")
    batch_convert_pdfs(str(local_pdf_root), str(local_img_root), dpi=300)

    # Load OCR engine
    logger.debug(f"Loading OCR engine from {config_path}")
    ocr_engine = load_engine(config_path)

    # Results directory
    results_dir = RESULTS_ROOT / engine_name / month / folder
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check for images
    png_files = list(local_img_root.rglob("*.png"))
    if not png_files:
        logger.warning(f"No images found for {folder}. Skipping OCR.")
        return

    logger.info(f"Found {len(png_files)} images to process")

    # Run OCR
    logger.info("Running OCR engine")
    run_ocr_benchmark_and_publish(
        ocr_engine=ocr_engine,
        engine_name=engine_name,
        data_dir=str(local_img_root),
        output_path=results_dir,
        month=month,
        run_id=run_id,
        document_name=folder,
        source_objects=source_objects,
    )

    # Clean and store extracted text
    logger.info("Cleaning and storing extracted text")
    emit_csv = os.getenv("SAVE_CSVS", "false").lower() in ("true", "1", "yes")
    run_extraction_stage(
        run_id=run_id,
        engine=engine_name,
        month=month,
        results_dir=RESULTS_ROOT,
        document_folder=folder,
        emit_csv=emit_csv,
    )

    # Monitor resources
    monitor_pipeline_resources(
        folder_name=folder, engine_name=engine_name, start_time=start_time
    )

    # Cleanup
    if not keep_json:
        json_file = results_dir / f"{folder}.json"
        if json_file.exists():
            json_file.unlink()
            logger.debug(f"Removed temporary JSON file: {json_file}")

    clean_temp_dirs()
    elapsed = time.time() - start_time
    logger.info(f"✓ Completed {folder} in {elapsed:.2f}s")
    logger.info(f"✓ Saved results to folder: {results_dir}")


def get_engine_configs(config_dir: str = "configs/engines") -> dict[str, str]:
    """
    Discover available OCR engine configurations.

    Returns:
        Dict mapping engine name to config path
    """
    return {p.stem: str(p.resolve()) for p in Path(config_dir).glob("*.yaml")}
