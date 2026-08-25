"""
Interactive OCR pipeline with user prompts for source/period selection.
Loops through each folder inside a source/month on MinIO and processes them.
"""

import argparse
import difflib
import os

from minio import Minio

from scripts.pipeline.core import (
    BUCKET_NAME,
    LOCAL_PDF_DIR,
    get_engine_configs,
    list_subfolders,
    process_folder,
)
from src.config import load_env
from src.data.minio_client import connect_to_minio
from src.io.publish import finish_run, register_run, reset_run
from src.utils.create_ids import make_run_id
from src.utils.logger import get_logger
from src.utils.runtime import pipeline_use_minio, reset_runtime_caches
from src.utils.time_utils import extract_month_from_prefix

logger = get_logger(__name__)


def select_source_folder(minio_client: Minio) -> str:
    """
    List top-level folders (sources/batches) in the bucket.
    Returns e.g. 'acme', 'xpto'
    """
    seen = set()
    for obj in minio_client.list_objects(BUCKET_NAME, prefix="", recursive=False):
        folder = (obj.object_name or "").split("/", 1)[0]
        if folder and not folder.startswith("."):
            seen.add(folder)

    sorted_sources = sorted(seen)
    if not sorted_sources:
        raise RuntimeError("No source folders found in bucket.")

    logger.info("\nAvailable sources:")
    for i, folder in enumerate(sorted_sources, 1):
        logger.info(f"{i:2d}. {folder}")

    while True:
        choice = input("\nSelect source number to process: ")
        if choice.isdigit() and 1 <= int(choice) <= len(sorted_sources):
            return sorted_sources[int(choice) - 1]
        logger.info("Invalid choice. Please enter a valid number.")


def select_base_folder(minio_client: Minio, root: str) -> str:
    """
    Discover available folders directly under <root>/ on MinIO
    and let the user pick one.
    Returns full prefix, e.g. "rm/2025-04/"
    """
    candidates = set()
    for obj in minio_client.list_objects(
        BUCKET_NAME, prefix=f"{root}/", recursive=False
    ):
        if obj.object_name is None:
            continue
        remainder = obj.object_name[len(root) + 1 :]  # strip "root/"
        folder = remainder.split("/", 1)[0]
        if folder and not folder.startswith("."):
            candidates.add(folder)

    sorted_folders = sorted(candidates)
    if not sorted_folders:
        raise RuntimeError(f"No folders found under '{root}/' on MinIO.")

    logger.info("\nAvailable folders:")
    for i, folder in enumerate(sorted_folders, 1):
        logger.info(f"{i:2d}. {folder}")

    while True:
        choice = input("\nSelect folder number to process: ")
        if choice.isdigit() and 1 <= int(choice) <= len(sorted_folders):
            selected = sorted_folders[int(choice) - 1]
            return f"{root}/{selected}/"
        logger.info("Invalid choice. Please enter a valid number.")


def parse_args(engines):
    p = argparse.ArgumentParser(description="Run OCR pipeline interactively.")
    p.add_argument(
        "--engine",
        choices=engines,
        default="tesseract",
        help="OCR engine to use (default: tesseract)",
    )
    p.add_argument(
        "--doc", help="Process a single document folder under the selected month"
    )
    p.add_argument(
        "--profile",
        choices=["local", "prod"],
        default="local",
        help="Which env profile to load (default: local)",
    )
    p.add_argument(
        "--keep-json",
        action="store_true",
        help="Keep the raw JSON files in the results directory.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate PDF pages before processing",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Disable MinIO/downloads and database publishing. "
            "Requires PDFs under data/pdf/<period>/<document>."
        ),
    )
    return p.parse_args()


def main():
    engine_configs = get_engine_configs()
    args = parse_args(engine_configs.keys())
    load_env(args.profile)

    if args.offline:
        os.environ["PIPELINE_PUBLISH_ENABLED"] = "false"
        os.environ["PIPELINE_USE_MINIO"] = "false"

    reset_runtime_caches()

    use_minio = pipeline_use_minio()

    if use_minio:
        client = connect_to_minio()
        source = select_source_folder(client)
        base_prefix = select_base_folder(client, root=source)
    else:
        client = None
        source = input(
            "Enter source identifier (used for run metadata, e.g. 'acme'): "
        ).strip()
        if not source:
            raise SystemExit("Source identifier is required in offline mode.")

        if not LOCAL_PDF_DIR.exists():
            raise SystemExit(
                f"Local PDF directory '{LOCAL_PDF_DIR}' not found. "
                "Download PDFs first."
            )

        periods = [p.name for p in LOCAL_PDF_DIR.iterdir() if p.is_dir()]
        if not periods:
            raise SystemExit(
                f"No local periods found under {LOCAL_PDF_DIR}. "
                "Download PDFs first."
            )

        periods = sorted(periods)
        logger.info("\nAvailable local periods:")
        for idx, period in enumerate(periods, 1):
            logger.info(f"{idx:2d}. {period}")

        while True:
            choice = input("\nSelect period number to process: ")
            if choice.isdigit() and 1 <= int(choice) <= len(periods):
                selected_period = periods[int(choice) - 1]
                break
            logger.info("Invalid choice. Please enter a valid number.")

        base_prefix = f"{source}/{selected_period}/"

    logger.info(f"\n▶ Selected folder: {base_prefix}")

    month_str = extract_month_from_prefix(base_prefix)
    selected_period = month_str
    selected_config = engine_configs[args.engine]
    engine_name = args.engine

    logger.info(f"Using engine: {engine_name} ({selected_config})")

    run_id = ""

    try:
        if use_minio:
            folders = list_subfolders(client, base_prefix)
        else:
            local_base = LOCAL_PDF_DIR / selected_period
            if not local_base.exists():
                raise SystemExit(
                    f"Expected local PDFs at {local_base}, but directory is missing."
                )
            folders = sorted([p.name for p in local_base.iterdir() if p.is_dir()])

        logger.info(f"Found {len(folders)} subfolders under '{base_prefix}'")

        # If --doc is provided, restrict to a single folder
        if args.doc:
            if args.doc in folders:
                folders = [args.doc]
                logger.info(f"Restricting to single document: {args.doc}")
            else:
                # Optional: nearest-match help
                suggestion = difflib.get_close_matches(args.doc, folders, n=1)
                msg = f"[ERROR] Doc '{args.doc}' not found under {base_prefix}."
                if suggestion:
                    msg += f" Did you mean: {suggestion[0]} ?"
                raise SystemExit(msg)

        # Create deterministic run_id and register the run
        run_id = make_run_id(
            source_id=source,
            period=month_str,
            engine=engine_name,
            pipeline_version="1",
        )

        register_run(
            {
                "run_id": run_id,
                "source_id": source,
                "period": month_str,
                "engine": engine_name,
                "pipeline_version": "1",
                "config_hash": None,
                "code_version": None,
                "status": "STARTED",
                "num_documents": len(folders),
                "notes": f"base_prefix={base_prefix}",
            }
        )

        reset_run(
            run_id,
            [
                "ocr_document_stats",
                "ocr_page_metrics",
                "extractions",
            ],
        )

        # Process each folder
        for folder in folders:
            process_folder(
                minio_client=client,
                base_prefix=base_prefix,
                folder=folder,
                config_path=selected_config,
                month=month_str,
                run_id=run_id,
                engine_name=engine_name,
                validate=args.validate,
                keep_json=args.keep_json,
            )

        finish_run(run_id, status="OK", notes=f"Processed {len(folders)} docs")
        logger.info("=" * 60)
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"✅ Run ID: {run_id}")
        logger.info("=" * 60)

    except Exception as e:
        if run_id:
            finish_run(run_id, status="FAILED", notes=str(e)[:500])
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
