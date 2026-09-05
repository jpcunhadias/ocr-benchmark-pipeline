import pytest

from src.evaluate.localization import (
    group_regions_into_lines,
    iou,
    localization_accuracy,
    locate_fields,
)


def _region(text: str, left: float, top: float, width: float, height: float) -> dict:
    return {
        "text": text,
        "confidence": 0.9,
        "bbox": {"left": left, "top": top, "width": width, "height": height},
    }


def test_group_regions_into_lines_separates_distinct_lines():
    regions = [
        _region("Report", 0.10, 0.10, 0.05, 0.02),
        _region("Date:", 0.10, 0.30, 0.04, 0.02),
    ]
    lines = group_regions_into_lines(regions)

    assert len(lines) == 2
    assert [r["text"] for r in lines[0]] == ["Report"]
    assert [r["text"] for r in lines[1]] == ["Date:"]


def test_group_regions_into_lines_clusters_same_line_left_to_right():
    regions = [
        _region("ID:", 0.16, 0.10, 0.03, 0.02),
        _region("Report", 0.10, 0.10, 0.05, 0.02),
    ]
    lines = group_regions_into_lines(regions)

    assert len(lines) == 1
    assert [r["text"] for r in lines[0]] == ["Report", "ID:"]


def test_group_regions_into_lines_uses_running_line_height_not_new_region_height():
    """A short punctuation-only region (much shorter than its line-mates)
    must still cluster into the line, using the line's own running average
    height as the tolerance basis -- not the tiny region's own height,
    which would otherwise fail to join."""
    status = _region("Status", 0.10, 0.100, 0.06, 0.020)  # center 0.110
    colon = _region(":", 0.165, 0.112, 0.005, 0.005)  # center 0.1145, diff 0.0045
    far = _region("Footer", 0.10, 0.500, 0.05, 0.020)  # a clearly separate line

    lines = group_regions_into_lines([status, colon, far])

    assert len(lines) == 2
    assert [r["text"] for r in lines[0]] == ["Status", ":"]
    assert [r["text"] for r in lines[1]] == ["Footer"]


def test_locate_fields_finds_value_region_after_label():
    regions = [
        _region("Report", 0.10, 0.10, 0.05, 0.02),
        _region("ID:", 0.16, 0.10, 0.03, 0.02),
        _region("RPT-1000", 0.20, 0.10, 0.08, 0.02),
    ]
    located = locate_fields(regions, ["Report ID:"])

    assert located["Report ID:"] == pytest.approx(
        {"left": 0.20, "top": 0.10, "width": 0.08, "height": 0.02}
    )


def test_locate_fields_merged_single_region_fallback():
    """When label+value are merged into one detection, that whole region is
    used as the value box -- an imprecise but documented fallback."""
    regions = [_region("Report ID: RPT-1000", 0.10, 0.10, 0.19, 0.02)]
    located = locate_fields(regions, ["Report ID:"])

    assert located["Report ID:"] == pytest.approx(
        {"left": 0.10, "top": 0.10, "width": 0.19, "height": 0.02}
    )


def test_locate_fields_missing_label_returns_none():
    regions = [_region("Date:", 0.10, 0.10, 0.04, 0.02)]
    located = locate_fields(regions, ["Report ID:"])

    assert located["Report ID:"] is None


def test_locate_fields_label_last_on_line_returns_none():
    regions = [_region("Status:", 0.10, 0.10, 0.05, 0.02)]
    located = locate_fields(regions, ["Status:"])

    assert located["Status:"] is None


def test_iou_no_overlap():
    box_a = {"left": 0, "top": 0, "width": 1, "height": 1}
    box_b = {"left": 2, "top": 2, "width": 1, "height": 1}
    assert iou(box_a, box_b) == pytest.approx(0.0)


def test_iou_identical_boxes():
    box = {"left": 0.1, "top": 0.1, "width": 0.2, "height": 0.2}
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_partial_overlap():
    box_a = {"left": 0, "top": 0, "width": 2, "height": 2}
    box_b = {"left": 1, "top": 1, "width": 2, "height": 2}
    # intersection is a 1x1 square; union = 4 + 4 - 1 = 7
    assert iou(box_a, box_b) == pytest.approx(1 / 7)


def test_localization_accuracy_counts_located_and_correct():
    box_a = {"left": 0, "top": 0, "width": 1, "height": 1}
    box_b = {"left": 5, "top": 5, "width": 1, "height": 1}
    ground_truth = {"A": box_a, "B": box_b}
    predicted = {"A": dict(box_a)}  # B not located at all

    result = localization_accuracy(ground_truth, predicted)

    assert result["fields_total"] == 2
    assert result["fields_located"] == 1
    assert result["fields_correct"] == 1
    assert result["avg_iou"] == pytest.approx(0.5)
    assert result["per_field"]["A"]["correct"] is True
    assert result["per_field"]["B"]["located"] is False
    assert result["per_field"]["B"]["iou"] == pytest.approx(0.0)


def test_localization_accuracy_below_threshold_is_located_but_incorrect():
    box_a = {"left": 0, "top": 0, "width": 2, "height": 2}
    box_a_predicted = {"left": 1.9, "top": 1.9, "width": 2, "height": 2}  # tiny overlap
    result = localization_accuracy({"A": box_a}, {"A": box_a_predicted})

    field = result["per_field"]["A"]
    assert field["located"] is True
    assert field["iou"] < 0.5
    assert field["correct"] is False
