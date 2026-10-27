"""Small checks shared by the configuration readers."""

from collections.abc import Mapping
import math
from numbers import Real


def require_mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def require_integer(value, name: str, minimum: int = 0, maximum=None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value

