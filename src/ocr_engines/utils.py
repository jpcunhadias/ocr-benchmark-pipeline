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
