import math
from decimal import Decimal

import pytest

from src.utils.text_utils import sanitize_numeric_value


def test_sanitize_none_like_values():
    assert sanitize_numeric_value(None) is None
    assert sanitize_numeric_value("") is None
    assert sanitize_numeric_value("   ") is None


def test_sanitize_invalid_string_returns_none():
    assert sanitize_numeric_value("abc") is None


def test_sanitize_valid_float_string():
    assert sanitize_numeric_value("0.75") == pytest.approx(0.75)


def test_sanitize_nan_inputs():
    assert sanitize_numeric_value("NaN") is None
    assert sanitize_numeric_value(float("nan")) is None
    assert sanitize_numeric_value(math.inf) is None
    assert sanitize_numeric_value(-math.inf) is None


def test_sanitize_numeric_types_preserved():
    assert sanitize_numeric_value(3) == 3
    assert sanitize_numeric_value(3.5) == pytest.approx(3.5)


def test_sanitize_decimal_inputs():
    assert sanitize_numeric_value(Decimal("2.5")) == pytest.approx(2.5)


def test_force_int_conversion():
    assert sanitize_numeric_value("4", as_int=True) == 4
    assert sanitize_numeric_value(4.2, as_int=True) == 4
    assert sanitize_numeric_value(4.7, as_int=True) == 5


def test_force_float_conversion():
    assert sanitize_numeric_value(5, as_int=False) == pytest.approx(5.0)


def test_zero_values_are_retained():
    assert sanitize_numeric_value("0") == pytest.approx(0.0)
    assert sanitize_numeric_value(0) == 0
