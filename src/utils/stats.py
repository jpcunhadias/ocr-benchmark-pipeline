from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


def _json_default(o: Any):
    # Datetimes
    if isinstance(o, datetime | date):
        return o.isoformat()
    if pd is not None and isinstance(o, pd.Timestamp):
        return o.to_pydatetime().isoformat()

    # NumPy scalars / arrays
    if np is not None:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()

    # Paths / sets
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set | frozenset):
        return list(o)

    # Final fallback
    return str(o)


def save_document_result(
    doc_name: str, results: list[dict], output_dir: Path, month: str
) -> dict:
    """
    Build a JSON document with per-page OCR results, write it to disk,
    and return the document dictionary.
    `results` should be a list of page dicts (text/confidence/engine/whatever).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{doc_name}.json"

    document = {
        "document_name": doc_name,
        "month": month,
        "pages": results,
    }

    output_file.write_text(
        json.dumps(document, indent=4, ensure_ascii=False, default=_json_default)
    )
    return document
