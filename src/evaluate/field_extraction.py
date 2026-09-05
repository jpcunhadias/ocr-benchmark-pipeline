"""Structured field-extraction accuracy for OCR benchmarking.

Complements src.evaluate.metrics (raw-text CER/WER) with a metric that
reflects real-world usefulness: did the engine get the *values* right
(Report ID, Date, Route, ...), not just the characters. Works purely off
the flat text every engine already returns via BaseOCREngine.predict(), so
it applies to any engine with no interface changes.

Ground truth lives under ``data/labels/<document>/<document>_<page>.fields.json``,
mirroring metrics.py's ``.txt`` layout -- a page without a matching file
simply has no field ground truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LABELS_ROOT = Path("data/labels")
_WHITESPACE_RE = re.compile(r"\s+")


def field_ground_truth_path(
    image_path: str | Path, labels_root: Path = LABELS_ROOT
) -> Path:
    """Map a processed page image to its expected fields ground-truth file."""
    image_path = Path(image_path)
    return labels_root / image_path.parent.name / f"{image_path.stem}.fields.json"


def load_field_ground_truth(
    image_path: str | Path, labels_root: Path = LABELS_ROOT
) -> dict[str, str] | None:
    """Return the {label: value} ground truth for a page image, or None if unlabeled."""
    gt_path = field_ground_truth_path(image_path, labels_root)
    if not gt_path.exists():
        return None
    return json.loads(gt_path.read_text(encoding="utf-8"))


def extract_fields(text: str, field_names: list[str]) -> dict[str, str | None]:
    """For each field label, find the first line containing it (case-insensitive)
    and return the remainder of that line as the value; None if not found.

    No layout/positional assumptions -- works on whatever flat text an
    engine returns, so it applies to any BaseOCREngine implementation.
    """
    lines = (text or "").splitlines()
    extracted: dict[str, str | None] = {}
    for label in field_names:
        value = None
        label_lower = label.lower()
        for line in lines:
            idx = line.lower().find(label_lower)
            if idx != -1:
                value = line[idx + len(label) :].strip()
                break
        extracted[label] = value
    return extracted


def _normalize(value: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", (value or "").strip()).casefold()


def field_accuracy(
    ground_truth: dict[str, str], extracted: dict[str, str | None]
) -> dict:
    """Compare extracted field values to ground truth (whitespace/case-insensitive).

    Returns {"fields_total", "fields_correct", "field_accuracy", "per_field"}
    where per_field maps each label to {"expected", "extracted", "correct"}.
    """
    per_field = {
        label: {
            "expected": expected,
            "extracted": extracted.get(label),
            "correct": _normalize(expected) == _normalize(extracted.get(label)),
        }
        for label, expected in ground_truth.items()
    }
    total = len(per_field)
    correct = sum(1 for f in per_field.values() if f["correct"])
    return {
        "fields_total": total,
        "fields_correct": correct,
        "field_accuracy": (correct / total) if total else 0.0,
        "per_field": per_field,
    }
