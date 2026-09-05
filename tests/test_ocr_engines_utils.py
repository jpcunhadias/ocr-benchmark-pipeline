import pytest

from src.ocr_engines.utils import normalize_confidence


def test_tesseract_confidence_normalized_from_0_100():
    assert normalize_confidence("tesseract", 85.0) == pytest.approx(0.85)


def test_easyocr_confidence_already_0_1():
    assert normalize_confidence("easyocr", 0.85) == pytest.approx(0.85)


def test_none_confidence_stays_none():
    assert normalize_confidence("tesseract", None) is None
    assert normalize_confidence("easyocr", None) is None


def test_unknown_engine_assumed_already_0_1():
    assert normalize_confidence("some_future_engine", 0.42) == pytest.approx(0.42)
