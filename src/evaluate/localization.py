"""Spatial (bounding-box) localization accuracy for OCR benchmarking.

Complements src.evaluate.field_extraction (did the engine read the right
*value*?) with a spatial question: did the engine also correctly *locate*
where that value is on the page? Uses IoU (intersection-over-union)
against ground-truth boxes, the same style of metric real document-AI
benchmarks use (FUNSD, CORD, DocVQA).

Ground truth lives under
``data/labels/<document>/<document>_<page>.boxes.json`` -- a separate file
from field_extraction's ``.fields.json`` (both are generated together for
the shipped sample doc, but load/compare independently, so a field present
in one and missing from the other is tolerated, not an error).

All boxes -- ground truth and predicted -- are fractions (0-1) of their own
image's width/height, never pixels. This is required, not stylistic: the
sample generator draws on a 1700x2200 canvas but every conversion path in
this repo rasterizes at 300 DPI (yielding a 2550x3300 PNG, an exact 1.5x
scale-up) -- comparing absolute pixel boxes across that mismatch would
silently produce near-zero IoU for everything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ocr_engines.utils import group_regions_into_lines

LABELS_ROOT = Path("data/labels")
IOU_THRESHOLD = 0.5  # standard object-detection / document-AI convention


def field_boxes_ground_truth_path(
    image_path: str | Path, labels_root: Path = LABELS_ROOT
) -> Path:
    """Map a processed page image to its expected boxes ground-truth file."""
    image_path = Path(image_path)
    return labels_root / image_path.parent.name / f"{image_path.stem}.boxes.json"


def load_field_boxes_ground_truth(
    image_path: str | Path, labels_root: Path = LABELS_ROOT
) -> dict[str, dict] | None:
    """Return the {label: {left, top, width, height}} ground truth for a
    page image, or None if unlabeled."""
    gt_path = field_boxes_ground_truth_path(image_path, labels_root)
    if not gt_path.exists():
        return None
    return json.loads(gt_path.read_text(encoding="utf-8"))


def _union_bbox(regions: list[dict]) -> dict:
    lefts = [r["bbox"]["left"] for r in regions]
    tops = [r["bbox"]["top"] for r in regions]
    rights = [r["bbox"]["left"] + r["bbox"]["width"] for r in regions]
    bottoms = [r["bbox"]["top"] + r["bbox"]["height"] for r in regions]
    left, top = min(lefts), min(tops)
    return {
        "left": left,
        "top": top,
        "width": max(rights) - left,
        "height": max(bottoms) - top,
    }


def locate_fields(
    regions: list[dict], field_names: list[str]
) -> dict[str, dict | None]:
    """For each field label, find the line containing it and return the
    union bounding box of whatever comes after it on that line -- the
    predicted location of the value. None if a label isn't found on any
    line, or is the last thing on its line.

    Known limitation: if an engine merges an entire "Label: Value" phrase
    into one detection (EasyOCR sometimes does), that single region's
    offset range extends past the label's own end, so it's included whole
    as the "value" box -- an imprecise but defensible fallback that shifts
    the predicted box's left edge left of the true value. This structurally
    suppresses IoU for engines that merge label+value, and is an
    intentional, documented approximation, not a bug to chase further.
    """
    lines = group_regions_into_lines(regions)
    located: dict[str, dict | None] = dict.fromkeys(field_names)

    for line in lines:
        parts: list[str] = []
        offsets: list[tuple[int, int]] = []
        pos = 0
        for region in line:
            text = region["text"]
            offsets.append((pos, pos + len(text)))
            parts.append(text)
            pos += len(text) + 1  # +1 for the joining space
        line_text = " ".join(parts)
        line_text_lower = line_text.lower()

        for label in field_names:
            if located.get(label) is not None:
                continue
            idx = line_text_lower.find(label.lower())
            if idx == -1:
                continue
            label_end = idx + len(label)
            value_regions = [
                region
                for region, (_, end) in zip(line, offsets, strict=True)
                if end > label_end
            ]
            if value_regions:
                located[label] = _union_bbox(value_regions)

    return located


def iou(box_a: dict, box_b: dict) -> float:
    """Intersection-over-union of two axis-aligned {left, top, width, height} boxes."""
    a_left, a_top = box_a["left"], box_a["top"]
    a_right, a_bottom = a_left + box_a["width"], a_top + box_a["height"]
    b_left, b_top = box_b["left"], box_b["top"]
    b_right, b_bottom = b_left + box_b["width"], b_top + box_b["height"]

    inter_left = max(a_left, b_left)
    inter_top = max(a_top, b_top)
    inter_right = min(a_right, b_right)
    inter_bottom = min(a_bottom, b_bottom)

    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    intersection = inter_width * inter_height

    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def localization_accuracy(
    ground_truth_boxes: dict[str, dict],
    predicted_boxes: dict[str, dict | None],
    threshold: float = IOU_THRESHOLD,
) -> dict:
    """Compare predicted field-value boxes to ground truth.

    Tolerates a field present in ground truth but missing from predicted
    (and vice versa) -- the two ground-truth sidecars (.fields.json,
    .boxes.json) are independently loaded and could drift.

    Returns {"fields_total", "fields_located", "fields_correct", "avg_iou",
    "per_field"} where per_field maps each ground-truth label to
    {"iou", "located", "correct", "gt_bbox", "predicted_bbox"}.
    """
    per_field: dict[str, dict[str, Any]] = {}
    for label, gt_box in ground_truth_boxes.items():
        predicted = predicted_boxes.get(label)
        located = predicted is not None
        field_iou = iou(gt_box, predicted) if predicted is not None else 0.0
        per_field[label] = {
            "iou": field_iou,
            "located": located,
            "correct": located and field_iou >= threshold,
            "gt_bbox": gt_box,
            "predicted_bbox": predicted,
        }

    total = len(per_field)
    located_count = sum(1 for f in per_field.values() if f["located"])
    correct_count = sum(1 for f in per_field.values() if f["correct"])
    avg_iou = (sum(f["iou"] for f in per_field.values()) / total) if total else 0.0

    return {
        "fields_total": total,
        "fields_located": located_count,
        "fields_correct": correct_count,
        "avg_iou": avg_iou,
        "per_field": per_field,
    }
