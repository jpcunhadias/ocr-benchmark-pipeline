import math
from decimal import Decimal
from numbers import Integral, Real
from typing import Any, Union

Number = Union[int, float]


def sanitize_numeric_value(
    value: Any, *, as_int: bool | None = None
) -> Number | None:
    """
    Coerce inputs into numeric types safe for database insertion.

    Treats empty strings, whitespace-only strings, NaN (any casing), and
    infinite values as ``None``. When ``as_int`` is ``True`` the result is
    coerced to ``int`` (rounded to the nearest integer to account for float
    artifacts). When ``as_int`` is ``False`` the result is returned as
    ``float``. With the default ``None`` the function preserves the incoming
    numeric type (``int``/``float``) whenever possible.
    """

    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            numeric = float(stripped)
        except ValueError:
            return None
    elif isinstance(value, Real | Decimal):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

    if math.isnan(numeric) or math.isinf(numeric):
        return None

    if as_int is True:
        return int(round(numeric))
    if as_int is False:
        return float(numeric)

    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)

    return float(numeric)
