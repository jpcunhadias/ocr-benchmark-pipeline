import numpy as np
from PIL import Image

from src.ocr_engines.easyocr_engine import EasyOCREngine


class FakeReader:
    """Stands in for easyocr.Reader -- readtext() returns detections in
    whatever order the model happened to produce them, not reading order."""

    def __init__(self, results):
        self._results = results

    def readtext(self, img_np):
        return self._results


def _quad(left, top, width, height):
    return [
        [left, top],
        [left + width, top],
        [left + width, top + height],
        [left, top + height],
    ]


def _make_engine(monkeypatch, results):
    monkeypatch.setattr(
        "src.ocr_engines.easyocr_engine.preprocess_image",
        lambda image_path: Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)),
    )
    engine = EasyOCREngine.__new__(EasyOCREngine)
    engine.config = {}
    engine.reader = FakeReader(results)
    return engine


def test_predict_reconstructs_line_breaks_from_out_of_order_detections(monkeypatch):
    # Two lines on the page, but readtext() hands them back interleaved and
    # out of left-to-right order within each line.
    results = [
        (_quad(0, 60, 30, 20), "Date:", 0.9),
        (_quad(0, 10, 40, 20), "Report", 0.95),
        (_quad(35, 62, 50, 18), "2024-01-01", 0.9),
        (_quad(75, 10, 25, 20), "12345", 0.9),
        (_quad(45, 12, 25, 18), "ID:", 0.9),
    ]
    engine = _make_engine(monkeypatch, results)

    result = engine.predict("fake_path.png")

    assert result["text"] == "Report ID: 12345\nDate: 2024-01-01"


def test_predict_field_extraction_works_on_reconstructed_lines(monkeypatch):
    from src.evaluate.field_extraction import extract_fields

    results = [
        (_quad(0, 60, 30, 20), "Date:", 0.9),
        (_quad(0, 10, 40, 20), "Report", 0.95),
        (_quad(35, 62, 50, 18), "2024-01-01", 0.9),
        (_quad(75, 10, 25, 20), "12345", 0.9),
        (_quad(45, 12, 25, 18), "ID:", 0.9),
    ]
    engine = _make_engine(monkeypatch, results)

    text = engine.predict("fake_path.png")["text"]
    extracted = extract_fields(text, ["Report ID:", "Date:"])

    assert extracted["Report ID:"] == "12345"
    assert extracted["Date:"] == "2024-01-01"


def test_predict_single_line_has_no_newline(monkeypatch):
    results = [
        (_quad(0, 10, 40, 20), "Status:", 0.9),
        (_quad(45, 10, 40, 20), "PASSED", 0.9),
    ]
    engine = _make_engine(monkeypatch, results)

    result = engine.predict("fake_path.png")

    assert result["text"] == "Status: PASSED"


def test_predict_no_detections_returns_empty_text(monkeypatch):
    engine = _make_engine(monkeypatch, [])

    result = engine.predict("fake_path.png")

    assert result["text"] == ""
    assert result["confidence"] is None
    assert result["regions"] == []
