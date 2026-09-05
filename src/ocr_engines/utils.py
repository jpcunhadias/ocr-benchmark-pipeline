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
    """Scale a raw engine confidence value onto a common 0-1 range."""
    if confidence is None:
        return None
    scale = CONFIDENCE_SCALE_MAX.get(engine_name, 1.0)
    return confidence / scale
