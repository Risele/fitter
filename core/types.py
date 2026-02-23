from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FitResult:
    name: str
    xmin: float
    xmax: float
    n: int
    dx: float
    inv_dx: float
    y: List[float]
    k: Optional[List[float]]  # slopes (same length as segments: n-1) or None
    max_abs_err: float
    mean_signed_err: float
    verify_points: int
    xmin_eff: Optional[float] = None
    shift: float = 0.0


@dataclass
class FixedPointConfig:
    q_frac: int
    x_shift: int
    y_scale: int


class FitError(RuntimeError):
    def __init__(self, code: str, data: Optional[Dict[str, Any]] = None, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.data = data or {}
