from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class SegmentEstimate:
    epsilon: float
    length: float
    is_uniform_input_grid: bool
    safety_factor: float
    m_raw: float
    m_used: float
    h_max: float
    h_used: float
    n_segments: int
    theoretical_bound: float
    check_passed: bool
    assumptions: List[str]
    warnings: List[str]
    spike_indices: List[int]
    derivation: str


def _is_uniform_grid(xs: Sequence[float], rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    if len(xs) < 3:
        return True
    dx0 = xs[1] - xs[0]
    for i in range(1, len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        if abs(dx - dx0) > max(abs_tol, rel_tol * abs(dx0)):
            return False
    return True


def _estimate_curvature_uniform(xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        curv = (ys[i + 1] - 2.0 * ys[i] + ys[i - 1]) / (dx * dx)
        out.append(curv)
    return out


def _estimate_curvature_nonuniform(xs: Sequence[float], ys: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(xs) - 1):
        dx_left = xs[i] - xs[i - 1]
        dx_right = xs[i + 1] - xs[i]
        sec_right = (ys[i + 1] - ys[i]) / dx_right
        sec_left = (ys[i] - ys[i - 1]) / dx_left
        curv = (2.0 / (xs[i + 1] - xs[i - 1])) * (sec_right - sec_left)
        out.append(curv)
    return out


def estimate_min_uniform_segments(
    xy: Sequence[Tuple[float, float]],
    epsilon: float,
    safety_factor: float = 1.0,
    spike_ratio: float = 5.0,
) -> SegmentEstimate:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be >= 1.0")
    if len(xy) < 2:
        raise ValueError("Need at least 2 points")

    xs = [float(p[0]) for p in xy]
    ys = [float(p[1]) for p in xy]

    for i in range(1, len(xs)):
        if not (xs[i] > xs[i - 1]):
            raise ValueError("x values must be strictly increasing")

    length = xs[-1] - xs[0]
    if length <= 0.0:
        raise ValueError("Invalid interval length")

    assumptions = [
        "f is continuous on [x0, xn]",
        "f is C^2 on [x0, xn] except possibly isolated non-smooth points",
        "tabulated points capture dominant curvature (no hidden high-frequency oscillations)",
    ]
    warnings: List[str] = []

    is_uniform = _is_uniform_grid(xs)
    if len(xs) < 5:
        warnings.append("Input table is sparse; curvature estimate may be unreliable.")

    if len(xs) >= 3:
        dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        dx_min = min(dxs)
        dx_max = max(dxs)
        if dx_max / dx_min > 10.0:
            warnings.append("Input grid is highly non-uniform; local curvature can be under-resolved.")

    if len(xs) < 3:
        curvatures: List[float] = []
        warnings.append("Only two points are provided; second-derivative estimate is unavailable.")
    elif is_uniform:
        curvatures = _estimate_curvature_uniform(xs, ys)
    else:
        curvatures = _estimate_curvature_nonuniform(xs, ys)

    abs_curvatures = [abs(v) for v in curvatures]
    m_raw = max(abs_curvatures) if abs_curvatures else 0.0

    spike_indices: List[int] = []
    if abs_curvatures:
        med = statistics.median(abs_curvatures)
        if med == 0.0:
            spike_indices = [i + 1 for i, v in enumerate(abs_curvatures) if v > 0.0]
        else:
            spike_indices = [i + 1 for i, v in enumerate(abs_curvatures) if v >= spike_ratio * med]
        if spike_indices:
            warnings.append(
                "Potential non-smooth points detected near indices {}.".format(spike_indices)
            )

    m_used = m_raw * safety_factor
    if m_used == 0.0:
        h_max = length
        n_segments = 1
        h_used = length
        theoretical_bound = 0.0
    else:
        h_max = math.sqrt((8.0 * epsilon) / m_used)
        ratio = length / h_max
        n_segments = max(1, int(math.ceil(ratio - 1e-12)))
        h_used = length / float(n_segments)
        theoretical_bound = (h_used * h_used * m_used) / 8.0
        while theoretical_bound > epsilon * (1.0 + 1e-12):
            n_segments += 1
            h_used = length / float(n_segments)
            theoretical_bound = (h_used * h_used * m_used) / 8.0

    check_passed = theoretical_bound <= epsilon * (1.0 + 1e-12)
    if not check_passed:
        warnings.append("Internal inequality check failed after rounding adjustment.")

    derivation = (
        "Interpolation bound:\n"
        "||f-l||_inf <= (h^2/8) * M.\n"
        "Estimated M_raw = {:.12g}, safety_factor = {:.12g}, M = {:.12g}.\n"
        "Required h <= sqrt(8*epsilon/M) = sqrt(8*{:.12g}/{:.12g}) = {:.12g}.\n"
        "Interval length L = x_n - x_0 = {:.12g}.\n"
        "N >= ceil(L/h) = ceil({:.12g}/{:.12g}) = {}.\n"
        "Used h = L/N = {:.12g}/{:d} = {:.12g}.\n"
        "Check: (h^2/8)*M = {:.12g} <= epsilon ({:.12g}) is {}."
    ).format(
        m_raw,
        safety_factor,
        m_used,
        epsilon,
        (m_used if m_used > 0.0 else float("inf")),
        h_max,
        length,
        length,
        h_max,
        n_segments,
        length,
        n_segments,
        h_used,
        theoretical_bound,
        epsilon,
        "true" if check_passed else "false",
    )

    return SegmentEstimate(
        epsilon=epsilon,
        length=length,
        is_uniform_input_grid=is_uniform,
        safety_factor=safety_factor,
        m_raw=m_raw,
        m_used=m_used,
        h_max=h_max,
        h_used=h_used,
        n_segments=n_segments,
        theoretical_bound=theoretical_bound,
        check_passed=check_passed,
        assumptions=assumptions,
        warnings=warnings,
        spike_indices=spike_indices,
        derivation=derivation,
    )
