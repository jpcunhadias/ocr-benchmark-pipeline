import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_ocr_text
from src.io.publish import append_df, push_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _iter_doc_jsons(
    root: Path, engine: str, month: str, document_folder: str | None
) -> list[Path]:
    base = root / engine / month

    if document_folder:
        # Search recursively within the document folder
        json_files = sorted((base / document_folder).rglob("*.json"))
    else:
        # Search recursively within the base directory
        json_files = sorted(base.rglob("*.json"))

    return json_files


def _normalize_page_no(k: Any) -> int:
    # handles "page_1" or integer keys
    try:
        s = str(k)
        if s.startswith("page_"):
            return int(s.split("_", 1)[1])
        return int(s)
    except Exception:
        return -1


def _sanitize_extractions_df(df: pd.DataFrame, month: str) -> pd.DataFrame:
    df = df.copy()

    # page as INT
    df["page"] = pd.to_numeric(df["page"], errors="coerce").fillna(0).astype(int)

    for c in ["document", "raw_text", "cleaned_text"]:
        df[c] = df[c].fillna("").astype(str)

    if "period" not in df.columns:
        df["period"] = month

    # De-dup at PK grain to avoid accidental doubles within the same run
    df = df.drop_duplicates(subset=["document", "page"])

    expected_cols = [
        "run_id",
        "timestamp",
        "document",
        "page",
        "raw_text",
        "cleaned_text",
        "char_count",
        "period",
        "doc_id",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[expected_cols]


def build_extractions_df(
    *,
    engine: str,
    month: str,
    results_dir: Path,
    document_folder: str | None = None,
) -> pd.DataFrame:
    """
    Load raw OCR output JSON produced by the benchmark step, apply the
    generic text-cleanup pass, and build one row per page.
    """
    rows: list[dict[str, Any]] = []
    json_files = _iter_doc_jsons(results_dir, engine, month, document_folder)

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            doc = json.load(f)

        pages = doc.get("pages", {})
        if isinstance(pages, dict):
            items = sorted(pages.items(), key=lambda kv: _normalize_page_no(kv[0]))
            page_iter = ((_normalize_page_no(k), v) for k, v in items)
        elif isinstance(pages, list):
            page_iter = (
                (p.get("page_no") or p.get("page_number") or i + 1, p)
                for i, p in enumerate(pages)
            )
        else:
            page_iter = []

        subfolder = jf.parent.name
        doc_name = jf.stem
        document_id = f"{subfolder}/{doc_name}"

        for page_no, pdata in page_iter:
            raw_text = (pdata or {}).get("raw_text", "")
            cleaned = preprocess_ocr_text(raw_text)

            rows.append(
                {
                    "timestamp": pd.Timestamp.utcnow(),
                    "document": document_id,
                    "page": int(page_no) if page_no not in (None, -1) else None,
                    "raw_text": raw_text,
                    "cleaned_text": cleaned,
                    "char_count": len(cleaned.strip()),
                    "period": month,
                    "doc_id": document_id,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "page" in df.columns:
        df["page"] = pd.to_numeric(df["page"], errors="coerce").astype("Int64")

    return df


def run_extraction_stage(
    *,
    run_id: str,
    engine: str,
    month: str,
    results_dir: Path | str = "results",
    document_folder: str | None = None,
    emit_csv: bool | None = None,
):
    results_dir = Path(results_dir)
    df = build_extractions_df(
        engine=engine,
        month=month,
        results_dir=results_dir,
        document_folder=document_folder,
    )
    if df is None or df.empty:
        logger.info("No extractions found to publish.")
        return df

    df["run_id"] = run_id
    df = _sanitize_extractions_df(df, month=month)
    append_df("extractions", df)

    bucket = os.getenv("MINIO_BUCKET", "ocr-artifacts")
    key = f"runs/{run_id}/extractions/{(document_folder or 'all').replace('/', '_')}.parquet"
    push_parquet(df, bucket, key)

    # Check environment variable if emit_csv not explicitly set
    if emit_csv is None:
        emit_csv = os.getenv("SAVE_CSVS", "false").lower() in ("true", "1", "yes")

    if emit_csv:
        out = results_dir / engine / month / "extracted"
        out.mkdir(parents=True, exist_ok=True)
        (out / "extracted_text.csv").write_text(
            df.to_csv(index=False), encoding="utf-8"
        )

    return df
