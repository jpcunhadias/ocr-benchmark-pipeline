import os
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ocr_engines.base_engine import BaseOCREngine
from src.utils.logger import get_logger

logger = get_logger(__name__)
_PAGE_RE = re.compile(r"(?:_page_)?_(\d+)$")


def parse_doc_and_page(image_path: str) -> tuple[str, int]:
    """
    Given '.../<DOC>/<DOC>_001.png' (or ..._page_001.png), return (DOC, 1).
    Falls back to (-1) when no page number is found.
    """
    p = Path(image_path)
    doc = p.parent.name
    m = _PAGE_RE.search(p.stem)
    page_no = int(m.group(1)) if m else -1
    return doc, page_no


def run_benchmark(image_paths: list[str], ocr_engine: BaseOCREngine) -> list[dict]:
    """
    Run OCR on a list of images and return one dict per page with metrics.
    The caller will decide how to persist (PG/Parquet/CSV).
    """
    results: list[dict] = []

    for image_path in tqdm(
        image_paths, desc=f"Running {ocr_engine.__class__.__name__}"
    ):
        if not os.path.exists(image_path):
            logger.info(f"Warning: {image_path} does not exist. Skipping.")
            continue

        start_time = time.time()
        try:
            ocr_result = ocr_engine.predict(
                image_path
            )  # expected keys: text, confidence, engine
            elapsed_time = time.time() - start_time

            text = ocr_result.get("text") or ""
            avg_conf = ocr_result.get(
                "confidence"
            )  # keep your scale (0–100 or 0–1), just be consistent

            doc, page_no = parse_doc_and_page(image_path)

            results.append(
                {
                    "timestamp": pd.Timestamp.now("UTC"),
                    "document": doc,
                    "engine": ocr_result.get("engine") or "",
                    "page": page_no,
                    "elapsed_sec": round(elapsed_time, 2),
                    "avg_confidence": None if avg_conf is None else float(avg_conf),
                    "n_chars": len(text.strip()),
                    # keep raw hooks if you want later:
                    "image_path": image_path,
                    "raw_text": text,
                }
            )

        except Exception as e:
            logger.info(f"Error processing {image_path}: {e}")
            continue

    return results
