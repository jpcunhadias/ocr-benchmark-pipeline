import argparse
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.evaluate.benchmark import run_benchmark
from src.io.publish import append_df, publish_document_to_mongo, push_parquet
from src.ocr_engines.base_engine import BaseOCREngine
from src.ocr_engines.utils import load_engine
from src.utils.create_ids import make_doc_id
from src.utils.stats import save_document_result
from src.utils.time_utils import extract_month_from_prefix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PAGE_RE = re.compile(r"(?:_page_)?_(\d+)$")


def get_image_paths_by_document(
    data_dir: str, extensions: list[str] = None
) -> dict[str, list[str]]:
    """Group images by document prefix, removing suffixes like '_001' or '_page_001'."""
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    image_paths = [
        str(p) for ext in extensions for p in data_path.rglob(f"*{ext}") if p.is_file()
    ]
    if not image_paths:
        logger.warning("No image files found in directory: %s", data_dir)
        return {}

    doc_groups = defaultdict(list)
    for path in image_paths:
        name = Path(path).stem
        m = re.match(r"^(.*?)(?:_page_)?_\d+$", name)
        doc_prefix = m.group(1) if m else name
        doc_groups[doc_prefix].append(path)

    for doc_name in doc_groups:
        doc_groups[doc_name].sort(key=extract_page_number)
    return doc_groups


def extract_page_number(path: str) -> int:
    """Extract the page number from the file name; returns -1 if not found."""
    name = Path(path).stem
    m = _PAGE_RE.search(name)
    return int(m.group(1)) if m else -1


def _build_page_metrics_df(
    results: list[dict], doc_name: str, engine_name: str
) -> pd.DataFrame:
    now = datetime.utcnow()
    rows = []
    for r in results:
        text = (r.get("text") or "").strip()
        rows.append(
            {
                "timestamp": now,  # TIMESTAMPTZ
                "document": doc_name,  # TEXT
                "engine": engine_name,  # TEXT
                "page": extract_page_number(r["image_path"]),  # INT
                "elapsed_sec": round(float(r.get("inference_time") or 0.0), 2),
                "avg_confidence": (
                    round(float(r["confidence"]), 2)
                    if r.get("confidence") is not None
                    else None
                ),
                "char_count": len(text),
            }
        )
    return pd.DataFrame(rows)


def run_ocr_benchmark(
    ocr_engine: BaseOCREngine,
    engine_name: str,
    data_dir: str,
    output_path: Path,
    month: str | None = None,
    return_dataframes: bool = False,
):
    """
    Run OCR on grouped images and write one JSON per document into:
    results/<engine>/<month>/<CONTRACT>/<DOC>.json
    """
    if month is None:
        try:
            month = extract_month_from_prefix(str(data_dir))
        except ValueError:
            logger.error("No --month and cannot infer from data_dir.")
            raise

    grouped_images = get_image_paths_by_document(data_dir)
    if not grouped_images:
        logger.info("No images found in %s", data_dir)
        return (pd.DataFrame(), pd.DataFrame(), []) if return_dataframes else []

    doc_rows = []
    page_rows: list[pd.DataFrame] = []
    doc_contents: list[dict] = []

    for doc_name, img_paths in grouped_images.items():
        img_paths = sorted(img_paths, key=extract_page_number)

        # IMPORTANT: output_path is already results/<engine>/<month>/<CONTRACT>
        out_dir = output_path / doc_name
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        results = run_benchmark(img_paths, ocr_engine)  # list[dict] per page
        elapsed = round(time.time() - t0, 2)

        # Write JSON for the document and get the content back
        doc_content = save_document_result(doc_name, results, out_dir, month=month)
        doc_contents.append(doc_content)

        if return_dataframes:
            # Doc-level row
            doc_rows.append(
                {
                    "document": doc_name,
                    "engine": engine_name,
                    "num_pages": len(img_paths),
                    "elapsed_sec": elapsed,
                    "cpu_percent": None,
                    "memory_percent": None,
                }
            )

            # Page metrics rows for this doc
            page_rows.append(_build_page_metrics_df(results, doc_name, engine_name))

    if not return_dataframes:
        return doc_contents

    df_stats_doc = pd.DataFrame(doc_rows)
    df_page_metrics = (
        pd.concat(page_rows, ignore_index=True) if page_rows else pd.DataFrame()
    )
    return df_stats_doc, df_page_metrics, doc_contents


def run_ocr_benchmark_and_publish(
    ocr_engine,
    engine_name: str,
    data_dir: str,
    output_path: Path,
    month: str,
    run_id: str,
    document_name: str | None = None,
):
    # Run and get in-memory DFs (JSONs are also written by run_ocr_benchmark)
    result = run_ocr_benchmark(
        ocr_engine=ocr_engine,
        engine_name=engine_name,
        data_dir=data_dir,
        output_path=output_path,
        month=month,
        return_dataframes=True,
    )

    # Handle both return types: tuple (with dataframes) or None
    if result is None:
        df_stats_doc, df_page_metrics, doc_contents = pd.DataFrame(), pd.DataFrame(), []
    elif isinstance(result, tuple) and len(result) == 3:
        df_stats_doc, df_page_metrics, doc_contents = result
    else:
        # Should not happen with return_dataframes=True, but handle gracefully
        logger.warning(f"Unexpected return type from run_ocr_benchmark: {type(result)}")
        df_stats_doc, df_page_metrics, doc_contents = pd.DataFrame(), pd.DataFrame(), []

    # Attach run_id and persist to Postgres
    for df in (df_stats_doc, df_page_metrics):
        if df is not None and not df.empty:
            df["run_id"] = run_id

    if df_stats_doc is not None and not df_stats_doc.empty:
        append_df("ocr_document_stats", df_stats_doc)
    if df_page_metrics is not None and not df_page_metrics.empty:
        append_df("ocr_page_metrics", df_page_metrics)

    # Publish to Mongo
    for doc in doc_contents:
        if not isinstance(doc, dict):
            logger.warning("Skipping non-dict document payload: %s", type(doc))
            continue

        doc_name = doc.get("document_name")
        if not doc_name:
            logger.warning("Document payload missing 'document_name': %s", doc)
            continue

        # Ensure deterministic document ID aligned with database records
        doc_id = make_doc_id(doc_name, month)

        enriched_doc = doc.copy()
        enriched_doc.update(
            {
                "doc_id": doc_id,
                "run_id": run_id,
                "engine": engine_name,
                "month": doc.get("month", month),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )

        publish_document_to_mongo(doc_id, enriched_doc)

    # Snapshots to MinIO
    bucket = os.getenv("MINIO_BUCKET", "ocr-artifacts")
    base_key = document_name or "batch"
    if df_stats_doc is not None and not df_stats_doc.empty:
        push_parquet(
            df_stats_doc,
            bucket,
            f"runs/{run_id}/stats/{base_key}_document_stats.parquet",
        )
    if df_page_metrics is not None and not df_page_metrics.empty:
        push_parquet(
            df_page_metrics,
            bucket,
            f"runs/{run_id}/stats/{base_key}_page_metrics.parquet",
        )


def main(config_path: str, data_dir: str, output_dir: str, month: str | None = None):
    logger.info("Loading OCR engine from config: %s", config_path)
    ocr_engine = load_engine(config_path)
    engine_name = Path(config_path).stem
    logger.info("Running OCR benchmark with engine: %s", engine_name)
    run_ocr_benchmark(ocr_engine, engine_name, data_dir, Path(output_dir), month=month)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OCR benchmark per document (grouped images)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to OCR YAML config"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with image files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output JSONs (one per document)",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Period YYYY-MM (default: inferred from --data_dir)",
    )
    args = parser.parse_args()
    main(args.config, args.data_dir, args.output_dir, month=args.month)
