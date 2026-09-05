from pathlib import Path

import pytest

from src.evaluate.benchmark import run_benchmark
from src.evaluate.metrics import (
    char_error_rate,
    ground_truth_path,
    load_ground_truth,
    word_error_rate,
)
from src.ocr_engines.base_engine import BaseOCREngine


class FakeEngine(BaseOCREngine):
    """Returns a fixed OCR result regardless of input, for pipeline-level tests."""

    def __init__(self, text: str, confidence: float = 0.9):
        super().__init__(config={})
        self._text = text
        self._confidence = confidence

    def predict(self, image_path: str) -> dict:
        return {"text": self._text, "confidence": self._confidence, "engine": "fake"}


def test_char_error_rate_perfect_match():
    assert char_error_rate("hello world", "hello world") == pytest.approx(0.0)


def test_char_error_rate_counts_edits():
    # "hello" -> "hallo" is one substitution out of 5 reference chars.
    assert char_error_rate("hello", "hallo") == pytest.approx(0.2)


def test_word_error_rate_counts_edits():
    # One substituted word out of 3.
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_error_rates_empty_reference():
    assert char_error_rate("", "") == pytest.approx(0.0)
    assert char_error_rate("", "junk") == pytest.approx(1.0)
    assert word_error_rate("", "") == pytest.approx(0.0)
    assert word_error_rate("", "junk") == pytest.approx(1.0)


def test_ground_truth_path_mirrors_document_and_page(tmp_path):
    image_path = tmp_path / "processed" / "2025-04" / "doc1" / "doc1" / "doc1_001.png"
    labels_root = tmp_path / "labels"
    expected = labels_root / "doc1" / "doc1_001.txt"
    assert ground_truth_path(image_path, labels_root=labels_root) == expected


def test_load_ground_truth_missing_label_returns_none(tmp_path):
    image_path = tmp_path / "processed" / "doc1" / "doc1_001.png"
    assert load_ground_truth(image_path, labels_root=tmp_path / "labels") is None


def test_load_ground_truth_reads_matching_label(tmp_path):
    labels_root = tmp_path / "labels"
    label_dir = labels_root / "doc1"
    label_dir.mkdir(parents=True)
    (label_dir / "doc1_001.txt").write_text("hello world\n", encoding="utf-8")

    image_path = tmp_path / "processed" / "doc1" / "doc1_001.png"
    assert load_ground_truth(image_path, labels_root=labels_root) == "hello world\n"


def test_run_benchmark_flags_pages_without_ground_truth(tmp_path):
    image_path = tmp_path / "unlabeled_doc" / "unlabeled_doc_001.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"")  # run_benchmark only checks existence

    results = run_benchmark([str(image_path)], FakeEngine(text="anything"))

    assert len(results) == 1
    assert results[0]["has_ground_truth"] is False
    assert results[0]["cer"] is None
    assert results[0]["wer"] is None


def test_run_benchmark_computes_error_rates_when_labeled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    doc_dir = Path("data/processed/labeled_doc")
    doc_dir.mkdir(parents=True)
    image_path = doc_dir / "labeled_doc_001.png"
    image_path.write_bytes(b"")

    label_dir = Path("data/labels/labeled_doc")
    label_dir.mkdir(parents=True)
    (label_dir / "labeled_doc_001.txt").write_text("hello world", encoding="utf-8")

    results = run_benchmark([str(image_path)], FakeEngine(text="hallo world"))

    assert len(results) == 1
    row = results[0]
    assert row["has_ground_truth"] is True
    assert row["cer"] == pytest.approx(char_error_rate("hello world", "hallo world"))
    assert row["wer"] == pytest.approx(word_error_rate("hello world", "hallo world"))
