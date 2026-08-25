"""
Non-interactive wrapper for the OCR pipeline.
Allows running the OCR pipeline programmatically from Streamlit or other automation.
"""

import argparse
import os

from scripts.pipeline.core import (
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

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run OCR pipeline non-interactively")
    parser.add_argument(
        "--source", required=True, help="Source/batch identifier (e.g., acme)"
    )
    parser.add_argument(
        "--period", required=True, help="Period YYYY-MM (e.g., 2025-04)"
    )
    parser.add_argument("--engine", default="tesseract", help="OCR engine to use")
    parser.add_argument("--doc", help="Process single document folder")
    parser.add_argument("--profile", default="local", choices=["local", "prod"])
    parser.add_argument("--keep-json", action="store_true", help="Keep JSON files")
    parser.add_argument("--validate", action="store_true", help="Validate PDF pages")
    parser.add_argument("--pipeline-version", default="1", help="Pipeline version")
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Disable MinIO/downloads and database publishing. "
            "Requires PDFs under data/pdf/<period>/<document>."
        ),
    )

    args = parser.parse_args()

    # Load environment
    load_env(args.profile)

    if args.offline:
        os.environ["PIPELINE_PUBLISH_ENABLED"] = "false"
        os.environ["PIPELINE_USE_MINIO"] = "false"

    reset_runtime_caches()

    # Connect to MinIO
    use_minio = pipeline_use_minio()
    if use_minio:
        logger.info("Connecting to MinIO")
        client = connect_to_minio()
    else:
        client = None
        logger.info("MinIO disabled; using local PDFs from %s", LOCAL_PDF_DIR)

    # Construct base prefix
    base_prefix = f"{args.source}/{args.period}/"
    month_str = args.period

    # Get engine configs and validate
    engine_configs = get_engine_configs()
    if args.engine not in engine_configs:
        raise ValueError(
            f"Engine '{args.engine}' not found. Available: {list(engine_configs.keys())}"
        )
    config_path = engine_configs[args.engine]

    logger.info("=" * 60)
    logger.info("OCR Pipeline Starting")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source}")
    logger.info(f"Period: {args.period}")
    logger.info(f"Engine: {args.engine}")
    logger.info(f"Base prefix: {base_prefix}")

    run_id = ""

    try:
        # List folders based on mode
        if use_minio:
            folders = list_subfolders(client, base_prefix)
        else:
            local_base = LOCAL_PDF_DIR / args.period
            if not local_base.exists():
                raise FileNotFoundError(
                    f"No local PDFs found at {local_base}. Run download first or "
                    "point data/pdf to the desired documents."
                )
            folders = sorted([p.name for p in local_base.iterdir() if p.is_dir()])

        logger.info(f"Found {len(folders)} document folders")

        # Filter to single doc if specified
        if args.doc:
            if args.doc in folders:
                folders = [args.doc]
                logger.info(f"Processing single document: {args.doc}")
            else:
                raise ValueError(
                    f"Document '{args.doc}' not found under {base_prefix}"
                )

        # Create run
        run_id = make_run_id(
            source_id=args.source,
            period=month_str,
            engine=args.engine,
            pipeline_version=args.pipeline_version,
        )

        logger.info(f"Run ID: {run_id}")

        register_run(
            {
                "run_id": run_id,
                "source_id": args.source,
                "period": month_str,
                "engine": args.engine,
                "pipeline_version": args.pipeline_version,
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
        for i, folder in enumerate(folders, 1):
            logger.info(f"[{i}/{len(folders)}] Processing {folder}")
            process_folder(
                minio_client=client,
                base_prefix=base_prefix,
                folder=folder,
                config_path=config_path,
                month=month_str,
                run_id=run_id,
                engine_name=args.engine,
                validate=args.validate,
                keep_json=args.keep_json,
            )

        finish_run(run_id, status="OK", notes=f"Processed {len(folders)} documents")
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
