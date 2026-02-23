from __future__ import annotations

from typing import Optional, Sequence


def _floor_to_i32_float(t: float) -> int:
    i = int(t)  # trunc toward 0
    if t < 0.0 and float(i) != t:
        i -= 1
    return i


def _approx_eval_float(
    x: float,
    xmin: float,
    inv_dx: float,
    y: Sequence[float],
    k: Optional[Sequence[float]],
) -> float:
    """
    Runtime-equivalent float evaluation (no math.h floor).
    """
    n = len(y)
    t = (x - xmin) * inv_dx

    i = _floor_to_i32_float(t)

    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2

    frac = t - float(i)

    if k is None:
        return y[i] + frac * (y[i + 1] - y[i])
    return y[i] + frac * k[i]
