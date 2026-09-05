"""Ground-truth accuracy metrics (CER/WER) for OCR benchmarking.

Ground truth lives under ``data/labels/<document>/<document>_<page>.txt``,
mirroring the ``<document>/<document>_<page>.png`` layout that
``convert_pdfs_to_images.py`` always produces for a processed page image
(the immediate parent directory is the PDF stem, regardless of which
period/temp directories sit above it -- see scripts/data_prep/convert_pdfs_to_images.py).

A page without a matching label file simply has no ground truth -- callers
should treat that as "accuracy unknown", not an error.
"""

from __future__ import annotations

from pathlib import Path

import jiwer

LABELS_ROOT = Path("data/labels")


def ground_truth_path(image_path: str | Path, labels_root: Path = LABELS_ROOT) -> Path:
    """Map a processed page image to its expected ground-truth text file."""
    image_path = Path(image_path)
    return labels_root / image_path.parent.name / f"{image_path.stem}.txt"


def load_ground_truth(
    image_path: str | Path, labels_root: Path = LABELS_ROOT
) -> str | None:
    """Return the ground-truth text for a page image, or None if unlabeled."""
    gt_path = ground_truth_path(image_path, labels_root)
    if not gt_path.exists():
        return None
    return gt_path.read_text(encoding="utf-8")


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit distance over reference length, char-level."""
    reference = (reference or "").strip()
    hypothesis = (hypothesis or "").strip()
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.cer(reference, hypothesis)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word Error Rate: edit distance over reference length, word-level."""
    reference = (reference or "").strip()
    hypothesis = (hypothesis or "").strip()
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return jiwer.wer(reference, hypothesis)
