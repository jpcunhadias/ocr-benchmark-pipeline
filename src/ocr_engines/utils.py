from pathlib import Path

from src.ocr_engines.easyocr_engine import EasyOCREngine
from src.ocr_engines.tesseract_engine import TesseractEngine
from src.utils.config_parser import load_config

ENGINE_MAP = {"tesseract": TesseractEngine, "easyocr": EasyOCREngine}


def load_engine(config_path: str):
    engine_name = Path(config_path).stem
    config = load_config(config_path)

    if engine_name not in ENGINE_MAP:
        raise ValueError(f"Unsupported OCR engine: {engine_name}")

    engine_class = ENGINE_MAP[engine_name]
    return engine_class(config)
