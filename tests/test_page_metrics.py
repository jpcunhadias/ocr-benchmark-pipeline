import pytest

from scripts.ocr.run_benchmark import _build_page_metrics_df


def _page_result(**overrides) -> dict:
    base = {
        "image_path": "data/processed/doc1/doc1_001.png",
        "elapsed_sec": 1.23,
        "avg_confidence": 87.5,
        "n_chars": 42,
        "has_ground_truth": False,
        "cer": None,
        "wer": None,
        "raw_text": "x" * 42,
    }
    base.update(overrides)
    return base


def test_reads_run_benchmark_output_keys():
    """_build_page_metrics_df must match the dict shape run_benchmark() produces
    (elapsed_sec/avg_confidence/n_chars), not an older/different shape."""
    df = _build_page_metrics_df(
        [_page_result()], doc_name="doc1", engine_name="tesseract"
    )
    row = df.iloc[0]

    assert row["page"] == 1
    assert row["document"] == "doc1"
    assert row["engine"] == "tesseract"
    assert row["elapsed_sec"] == pytest.approx(1.23)
    assert row["avg_confidence"] == pytest.approx(87.5)
    assert row["char_count"] == 42


def test_carries_cer_wer_when_labeled():
    df = _build_page_metrics_df(
        [_page_result(has_ground_truth=True, cer=0.1, wer=0.2)],
        doc_name="doc1",
        engine_name="tesseract",
    )
    row = df.iloc[0]

    assert row["cer"] == pytest.approx(0.1)
    assert row["wer"] == pytest.approx(0.2)


def test_cer_wer_none_when_unlabeled():
    df = _build_page_metrics_df(
        [_page_result()], doc_name="doc1", engine_name="tesseract"
    )
    row = df.iloc[0]

    assert row["cer"] is None
    assert row["wer"] is None
