from pathlib import Path

from src.utils.config_parser import load_config


def _load_tesseract_engine():
    from src.ocr_engines.tesseract_engine import TesseractEngine

    return TesseractEngine


def _load_easyocr_engine():
    from src.ocr_engines.easyocr_engine import EasyOCREngine

    return EasyOCREngine


# Values are loaders rather than classes so that an engine's (potentially
# heavy) dependencies are only imported when that engine is actually used.
ENGINE_LOADERS = {
    "tesseract": _load_tesseract_engine,
    "easyocr": _load_easyocr_engine,
}


def load_engine(config_path: str):
    engine_name = Path(config_path).stem
    config = load_config(config_path)

    if engine_name not in ENGINE_LOADERS:
        raise ValueError(f"Unsupported OCR engine: {engine_name}")

    engine_class = ENGINE_LOADERS[engine_name]()
    return engine_class(config)


# Each engine reports "confidence" on its own native scale -- Tesseract's
# image_to_data gives 0-100, EasyOCR's readtext gives a 0-1 probability.
# Comparing them raw (e.g. in a confidence-calibration chart) would make
# Tesseract look wildly less confident than EasyOCR for no real reason.
# Keyed by the same engine identifier stored in ocr_page_metrics.engine
# (the config file stem, e.g. "tesseract"/"easyocr") -- extend this when a
# new engine is added if its scale isn't already 0-1.
CONFIDENCE_SCALE_MAX = {
    "tesseract": 100.0,
    "easyocr": 1.0,
}


def normalize_confidence(engine_name: str, confidence: float | None) -> float | None:
    """Scale a raw engine confidence value onto a common 0-1 range.

    `confidence` may arrive as a Decimal (Postgres NUMERIC rows come back
    that way via asyncpg/SQLAlchemy) rather than a float, and Python won't
    divide a Decimal by a float directly -- coerce it first.
    """
    if confidence is None:
        return None
    scale = CONFIDENCE_SCALE_MAX.get(engine_name, 1.0)
    return float(confidence) / scale


def group_regions_into_lines(regions: list[dict]) -> list[list[dict]]:
    """Cluster regions into reading-order lines using vertical position only
    -- engine-agnostic, since only Tesseract exposes an explicit line
    grouping and EasyOCR has no equivalent.

    Regions are sorted by vertical center and greedily joined into the
    current line while their center stays within a tolerance of that
    line's running average height (not the new region's own height alone --
    a lone ":" glyph's box is much shorter than its neighbors and would
    otherwise fail to cluster). Each resulting line is then sorted
    left-to-right for reading order.

    Shared by src.evaluate.localization (locating field-value boxes) and
    EasyOCREngine.predict() (reconstructing line breaks in "text" -- EasyOCR
    itself returns detections in no particular line order).
    """
    if not regions:
        return []

    def center(r: dict) -> float:
        return r["bbox"]["top"] + r["bbox"]["height"] / 2

    ordered = sorted(regions, key=center)
    lines: list[list[dict]] = [[ordered[0]]]
    line_avg_height = ordered[0]["bbox"]["height"]
    line_avg_center = center(ordered[0])

    for region in ordered[1:]:
        tolerance = 0.75 * line_avg_height
        if abs(center(region) - line_avg_center) <= tolerance:
            lines[-1].append(region)
        else:
            lines.append([region])
            line_avg_height = region["bbox"]["height"]
            line_avg_center = center(region)
            continue

        current_line = lines[-1]
        line_avg_height = sum(r["bbox"]["height"] for r in current_line) / len(
            current_line
        )
        line_avg_center = sum(center(r) for r in current_line) / len(current_line)

    for line in lines:
        line.sort(key=lambda r: r["bbox"]["left"])
    return lines
