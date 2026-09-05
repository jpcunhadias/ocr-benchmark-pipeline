import pytest

from src.evaluate.field_extraction import extract_fields, field_accuracy


def test_extract_fields_finds_labels_in_text():
    text = (
        "ACME LOGISTICS\n"
        "Delivery Inspection Report\n"
        "Report ID: RPT-1000\n"
        "Date: 2025-04-14\n"
        "Status: PASSED\n"
    )
    extracted = extract_fields(text, ["Report ID:", "Date:", "Status:"])

    assert extracted["Report ID:"] == "RPT-1000"
    assert extracted["Date:"] == "2025-04-14"
    assert extracted["Status:"] == "PASSED"


def test_extract_fields_missing_label_is_none():
    extracted = extract_fields("Report ID: RPT-1000\n", ["Report ID:", "Inspector:"])

    assert extracted["Report ID:"] == "RPT-1000"
    assert extracted["Inspector:"] is None


def test_extract_fields_is_case_insensitive_on_label():
    extracted = extract_fields("report id: RPT-1000\n", ["Report ID:"])

    assert extracted["Report ID:"] == "RPT-1000"


def test_field_accuracy_counts_correct_and_incorrect():
    ground_truth = {
        "Report ID:": "RPT-1000",
        "Date:": "2025-04-14",
        "Status:": "PASSED",
    }
    # Simulate one OCR error: "PASSED" misread as "PA55ED".
    extracted = {"Report ID:": "RPT-1000", "Date:": "2025-04-14", "Status:": "PA55ED"}

    result = field_accuracy(ground_truth, extracted)

    assert result["fields_total"] == 3
    assert result["fields_correct"] == 2
    assert result["field_accuracy"] == pytest.approx(2 / 3)
    assert result["per_field"]["Report ID:"]["correct"] is True
    assert result["per_field"]["Status:"]["correct"] is False
    assert result["per_field"]["Status:"]["expected"] == "PASSED"
    assert result["per_field"]["Status:"]["extracted"] == "PA55ED"


def test_field_accuracy_normalizes_whitespace_and_case():
    ground_truth = {"Route:": "Warehouse 4 -> Distribution Center B"}
    extracted = {"Route:": "  warehouse 4 ->  distribution center b  "}

    result = field_accuracy(ground_truth, extracted)

    assert result["fields_correct"] == 1
    assert result["field_accuracy"] == 1.0
