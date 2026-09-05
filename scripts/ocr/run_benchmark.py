import argparse
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB

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
    data_dir: str, extensions: list[str] | None = None
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


class BenchmarkResult(NamedTuple):
    """Dataframes + doc payloads from run_ocr_benchmark(return_dataframes=True).

    A real type instead of a positional tuple: this already grew once (3
    dataframes -> 4) with a brittle isinstance/len sniff at its one call
    site to cope; a NamedTuple gives attribute access and a real signature
    instead of extending that sniffing further as fields keep being added.
    """

    doc_stats: pd.DataFrame
    page_metrics: pd.DataFrame
    field_results: pd.DataFrame
    localization_results: pd.DataFrame
    doc_contents: list[dict]


def _empty_benchmark_result() -> BenchmarkResult:
    return BenchmarkResult(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    )


def _build_page_metrics_df(
    results: list[dict],
    doc_name: str,
    engine_name: str,
    source_object: str | None = None,
) -> pd.DataFrame:
    """Build the ocr_page_metrics row for each page result from run_benchmark().

    Keys here (elapsed_sec/avg_confidence/n_chars/cer/wer/fields_*/avg_iou/
    localization_fields_*) must match the dict shape produced by
    src.evaluate.benchmark.run_benchmark.

    `source_object` is the exact MinIO object key this document's PDF came
    from (None for offline runs, or when the caller doesn't know it) --
    lets the box-overlay preview fetch a page's source PDF directly instead
    of re-deriving the object key later via documents.source_path plus a
    filename-stem search.
    """
    now = datetime.utcnow()
    rows = []
    for r in results:
        rows.append(
            {
                "timestamp": now,  # TIMESTAMPTZ
                "document": doc_name,  # TEXT
                "engine": engine_name,  # TEXT
                "page": extract_page_number(r["image_path"]),  # INT
                "elapsed_sec": round(float(r.get("elapsed_sec") or 0.0), 2),
                "avg_confidence": (
                    round(float(r["avg_confidence"]), 2)
                    if r.get("avg_confidence") is not None
                    else None
                ),
                "char_count": r.get("n_chars", 0),
                "cer": r.get("cer"),
                "wer": r.get("wer"),
                "fields_total": r.get("fields_total"),
                "fields_correct": r.get("fields_correct"),
                "field_accuracy": r.get("field_accuracy"),
                "avg_iou": r.get("avg_iou"),
                "localization_fields_total": r.get("localization_fields_total"),
                "localization_fields_correct": r.get("localization_fields_correct"),
                "source_object": source_object,
            }
        )
    return pd.DataFrame(rows)


def _build_field_results_df(
    results: list[dict], doc_name: str, engine_name: str
) -> pd.DataFrame:
    """Explode each labeled page's field_details into one ocr_field_results
    row per field. Pages without field ground truth (field_details is None)
    contribute no rows."""
    rows = []
    for r in results:
        field_details = r.get("field_details")
        if not field_details:
            continue
        page = extract_page_number(r["image_path"])
        for field_name, detail in field_details.items():
            rows.append(
                {
                    "document": doc_name,
                    "engine": engine_name,
                    "page": page,
                    "field_name": field_name,
                    "expected_value": detail["expected"],
                    "extracted_value": detail["extracted"],
                    "correct": detail["correct"],
                }
            )
    return pd.DataFrame(rows)


def _build_localization_results_df(
    results: list[dict], doc_name: str, engine_name: str
) -> pd.DataFrame:
    """Explode each labeled page's localization_details into one
    ocr_localization_results row per field. Pages without box ground truth
    (localization_details is None) contribute no rows.

    Box coordinates are cast to native float here -- EasyOCR's regions
    carry numpy scalars upstream, which psycopg2's JSONB adapter cannot
    serialize; casting only at the source (in the engine) isn't enough
    once boxes have passed through dict/tuple operations, so cast again
    right before they go into the DataFrame that gets inserted.
    """

    def _native_bbox(bbox: dict | None) -> dict | None:
        if bbox is None:
            return None
        return {k: float(v) for k, v in bbox.items()}

    rows = []
    for r in results:
        localization_details = r.get("localization_details")
        if not localization_details:
            continue
        page = extract_page_number(r["image_path"])
        for field_name, detail in localization_details.items():
            rows.append(
                {
                    "document": doc_name,
                    "engine": engine_name,
                    "page": page,
                    "field_name": field_name,
                    "iou": float(detail["iou"]),
                    "located": detail["located"],
                    "correct": detail["correct"],
                    "gt_bbox": _native_bbox(detail["gt_bbox"]),
                    "predicted_bbox": _native_bbox(detail["predicted_bbox"]),
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
    source_objects: dict[str, str] | None = None,
):
    """
    Run OCR on grouped images and write one JSON per document into:
    results/<engine>/<month>/<CONTRACT>/<DOC>.json

    `source_objects` maps a document's filename stem to the exact MinIO
    object key its PDF came from (see scripts.pipeline.core.process_folder);
    None/absent for offline runs.
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
        return _empty_benchmark_result() if return_dataframes else []

    doc_rows = []
    page_rows: list[pd.DataFrame] = []
    field_rows: list[pd.DataFrame] = []
    localization_rows: list[pd.DataFrame] = []
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
            source_object = (source_objects or {}).get(doc_name)
            page_rows.append(
                _build_page_metrics_df(results, doc_name, engine_name, source_object)
            )
            field_rows.append(_build_field_results_df(results, doc_name, engine_name))
            localization_rows.append(
                _build_localization_results_df(results, doc_name, engine_name)
            )

    if not return_dataframes:
        return doc_contents

    df_stats_doc = pd.DataFrame(doc_rows)
    df_page_metrics = (
        pd.concat(page_rows, ignore_index=True) if page_rows else pd.DataFrame()
    )
    df_field_results = (
        pd.concat(field_rows, ignore_index=True) if field_rows else pd.DataFrame()
    )
    df_localization_results = (
        pd.concat(localization_rows, ignore_index=True)
        if localization_rows
        else pd.DataFrame()
    )
    return BenchmarkResult(
        df_stats_doc,
        df_page_metrics,
        df_field_results,
        df_localization_results,
        doc_contents,
    )


def run_ocr_benchmark_and_publish(
    ocr_engine,
    engine_name: str,
    data_dir: str,
    output_path: Path,
    month: str,
    run_id: str,
    document_name: str | None = None,
    source_objects: dict[str, str] | None = None,
):
    # Run and get in-memory DFs (JSONs are also written by run_ocr_benchmark)
    result: BenchmarkResult = run_ocr_benchmark(
        ocr_engine=ocr_engine,
        engine_name=engine_name,
        data_dir=data_dir,
        output_path=output_path,
        month=month,
        return_dataframes=True,
        source_objects=source_objects,
    )
    df_stats_doc = result.doc_stats
    df_page_metrics = result.page_metrics
    df_field_results = result.field_results
    df_localization_results = result.localization_results
    doc_contents = result.doc_contents

    # Attach run_id and persist to Postgres
    for df in (
        df_stats_doc,
        df_page_metrics,
        df_field_results,
        df_localization_results,
    ):
        if not df.empty:
            df["run_id"] = run_id

    if not df_stats_doc.empty:
        append_df("ocr_document_stats", df_stats_doc)
    if not df_page_metrics.empty:
        append_df("ocr_page_metrics", df_page_metrics)
    if not df_field_results.empty:
        append_df("ocr_field_results", df_field_results)
    if not df_localization_results.empty:
        # Explicit JSONB dtype: to_sql's default type inference for
        # dict-valued pandas columns against a jsonb DDL column isn't
        # reliable without this.
        append_df(
            "ocr_localization_results",
            df_localization_results,
            dtype={"gt_bbox": JSONB(), "predicted_bbox": JSONB()},
        )

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
    if not df_stats_doc.empty:
        push_parquet(
            df_stats_doc,
            bucket,
            f"runs/{run_id}/stats/{base_key}_document_stats.parquet",
        )
    if not df_page_metrics.empty:
        push_parquet(
            df_page_metrics,
            bucket,
            f"runs/{run_id}/stats/{base_key}_page_metrics.parquet",
        )
    if not df_field_results.empty:
        push_parquet(
            df_field_results,
            bucket,
            f"runs/{run_id}/stats/{base_key}_field_results.parquet",
        )
    if not df_localization_results.empty:
        push_parquet(
            df_localization_results,
            bucket,
            f"runs/{run_id}/stats/{base_key}_localization_results.parquet",
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
