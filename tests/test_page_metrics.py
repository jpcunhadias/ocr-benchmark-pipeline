import numpy as np
import pytest

from scripts.ocr.run_benchmark import (
    _build_field_results_df,
    _build_localization_results_df,
    _build_page_metrics_df,
)


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


def _field_result(**overrides) -> dict:
    base = {
        "image_path": "data/processed/doc1/doc1_001.png",
        "field_details": {
            "Report ID:": {
                "expected": "RPT-1000",
                "extracted": "RPT-1000",
                "correct": True,
            },
            "Status:": {"expected": "PASSED", "extracted": "PA55ED", "correct": False},
        },
    }
    base.update(overrides)
    return base


def test_build_field_results_df_explodes_per_field_rows():
    df = _build_field_results_df(
        [_field_result()], doc_name="doc1", engine_name="tesseract"
    )

    assert len(df) == 2
    row = df[df["field_name"] == "Status:"].iloc[0]
    assert row["document"] == "doc1"
    assert row["engine"] == "tesseract"
    assert row["page"] == 1
    assert row["expected_value"] == "PASSED"
    assert row["extracted_value"] == "PA55ED"
    assert bool(row["correct"]) is False


def test_build_field_results_df_skips_unlabeled_pages():
    df = _build_field_results_df(
        [_field_result(field_details=None)], doc_name="doc1", engine_name="tesseract"
    )

    assert df.empty


def _localization_result(**overrides) -> dict:
    base = {
        "image_path": "data/processed/doc1/doc1_001.png",
        "localization_details": {
            "Report ID:": {
                "iou": 1.0,
                "located": True,
                "correct": True,
                "gt_bbox": {"left": 0.2, "top": 0.1, "width": 0.08, "height": 0.02},
                "predicted_bbox": {
                    "left": 0.2,
                    "top": 0.1,
                    "width": 0.08,
                    "height": 0.02,
                },
            },
            "Status:": {
                "iou": 0.1,
                "located": True,
                "correct": False,
                "gt_bbox": {"left": 0.3, "top": 0.2, "width": 0.05, "height": 0.02},
                "predicted_bbox": {
                    "left": 0.31,
                    "top": 0.25,
                    "width": 0.05,
                    "height": 0.02,
                },
            },
        },
    }
    base.update(overrides)
    return base


def test_build_localization_results_df_explodes_per_field_rows():
    df = _build_localization_results_df(
        [_localization_result()], doc_name="doc1", engine_name="tesseract"
    )

    assert len(df) == 2
    row = df[df["field_name"] == "Status:"].iloc[0]
    assert row["document"] == "doc1"
    assert row["engine"] == "tesseract"
    assert row["page"] == 1
    assert row["iou"] == pytest.approx(0.1)
    assert bool(row["located"]) is True
    assert bool(row["correct"]) is False
    assert row["gt_bbox"] == pytest.approx(
        {"left": 0.3, "top": 0.2, "width": 0.05, "height": 0.02}
    )


def test_build_localization_results_df_skips_unlabeled_pages():
    df = _build_localization_results_df(
        [_localization_result(localization_details=None)],
        doc_name="doc1",
        engine_name="tesseract",
    )

    assert df.empty


def test_build_localization_results_df_casts_numpy_bbox_values_to_native_float():
    """EasyOCR's regions carry numpy scalars upstream. gt_bbox/predicted_bbox
    are JSONB columns -- json.dumps() (used by psycopg2's JSONB adapter)
    can't serialize numpy floats, so the DataFrame builder must cast them to
    native float or the insert breaks at write time. (Plain NUMERIC columns
    like `iou` don't have this problem -- pandas/psycopg2 already handle
    numpy scalars fine there, same as the pre-existing cer/wer columns.)"""
    result = _localization_result(
        localization_details={
            "Report ID:": {
                "iou": np.float32(0.87),
                "located": True,
                "correct": True,
                "gt_bbox": {
                    "left": np.float32(0.2),
                    "top": np.float32(0.1),
                    "width": np.float32(0.08),
                    "height": np.float32(0.02),
                },
                "predicted_bbox": {
                    "left": np.float32(0.2),
                    "top": np.float32(0.1),
                    "width": np.float32(0.08),
                    "height": np.float32(0.02),
                },
            }
        }
    )

    df = _build_localization_results_df(
        [result], doc_name="doc1", engine_name="easyocr"
    )
    row = df.iloc[0]

    assert all(type(v) is float for v in row["gt_bbox"].values())
    assert all(type(v) is float for v in row["predicted_bbox"].values())
