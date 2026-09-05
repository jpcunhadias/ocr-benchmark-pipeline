import numpy as np
from PIL import Image

from src.ocr_engines.paddleocr_engine import PaddleOCREngine


class FakePaddleOCR:
    """Stands in for paddleocr.PaddleOCR -- predict() returns detections in
    whatever order the model happened to produce them, not reading order."""

    def __init__(self, result: dict | None):
        self._result = result

    def predict(self, img_np):
        return [self._result] if self._result is not None else []


def _box(left, top, width, height):
    return [left, top, left + width, top + height]


def _make_engine(monkeypatch, result: dict | None):
    monkeypatch.setattr(
        "src.ocr_engines.paddleocr_engine.preprocess_image",
        lambda image_path: Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)),
    )
    engine = PaddleOCREngine.__new__(PaddleOCREngine)
    engine.config = {}
    engine.ocr = FakePaddleOCR(result)
    return engine


def test_predict_reconstructs_line_breaks_from_out_of_order_detections(monkeypatch):
    # Two lines on the page, but predict() hands them back interleaved and
    # out of left-to-right order within each line.
    result = {
        "rec_texts": ["Date:", "Report", "2024-01-01", "12345", "ID:"],
        "rec_scores": [0.9, 0.95, 0.9, 0.9, 0.9],
        "rec_boxes": np.array(
            [
                _box(0, 60, 30, 20),
                _box(0, 10, 40, 20),
                _box(35, 62, 50, 18),
                _box(75, 10, 25, 20),
                _box(45, 12, 25, 18),
            ]
        ),
    }
    engine = _make_engine(monkeypatch, result)

    prediction = engine.predict("fake_path.png")

    assert prediction["text"] == "Report ID: 12345\nDate: 2024-01-01"


def test_predict_field_extraction_works_on_reconstructed_lines(monkeypatch):
    from src.evaluate.field_extraction import extract_fields

    result = {
        "rec_texts": ["Date:", "Report", "2024-01-01", "12345", "ID:"],
        "rec_scores": [0.9, 0.95, 0.9, 0.9, 0.9],
        "rec_boxes": np.array(
            [
                _box(0, 60, 30, 20),
                _box(0, 10, 40, 20),
                _box(35, 62, 50, 18),
                _box(75, 10, 25, 20),
                _box(45, 12, 25, 18),
            ]
        ),
    }
    engine = _make_engine(monkeypatch, result)

    text = engine.predict("fake_path.png")["text"]
    extracted = extract_fields(text, ["Report ID:", "Date:"])

    assert extracted["Report ID:"] == "12345"
    assert extracted["Date:"] == "2024-01-01"


def test_predict_single_line_has_no_newline(monkeypatch):
    result = {
        "rec_texts": ["Status:", "PASSED"],
        "rec_scores": [0.9, 0.9],
        "rec_boxes": np.array([_box(0, 10, 40, 20), _box(45, 10, 40, 20)]),
    }
    engine = _make_engine(monkeypatch, result)

    prediction = engine.predict("fake_path.png")

    assert prediction["text"] == "Status: PASSED"


def test_predict_no_detections_returns_empty_text(monkeypatch):
    engine = _make_engine(monkeypatch, None)

    prediction = engine.predict("fake_path.png")

    assert prediction["text"] == ""
    assert prediction["confidence"] is None
    assert prediction["regions"] == []
