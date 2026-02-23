from __future__ import annotations

from typing import List, Tuple

from core.evaluation import _approx_eval_float
from core.types import FitResult
from ios.sources import Source


def _linspace(xmin: float, xmax: float, count: int) -> List[float]:
    if count < 2:
        return [xmin]
    step = (xmax - xmin) / float(count - 1)
    return [xmin + i * step for i in range(count)]


def _source_points(source: Source) -> Tuple[List[float], List[float]]:
    xs = []
    ys = []
    if hasattr(source, "x") and hasattr(source, "y"):
        xs = list(getattr(source, "x"))
        ys = list(getattr(source, "y"))
    return xs, ys


def build_fit_figure(source: Source, res: FitResult, sample_points: int = 1000):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    xmin, xmax = source.domain()
    xmin_eff = res.xmin_eff if res.xmin_eff is not None else res.xmin

    xs_line = _linspace(xmin, xmax, max(2, sample_points))
    xs_fit = _linspace(xmin, xmax, max(2, res.n))
    ys_source = [source.eval(x) for x in xs_line]
    ys_fit = [
        _approx_eval_float(x, xmin_eff, res.inv_dx, res.y, res.k)
        for x in xs_fit
    ]
    ys_fit_line = [
        _approx_eval_float(x, xmin_eff, res.inv_dx, res.y, res.k)
        for x in xs_line
    ]
    ys_err = [yf - ys for yf, ys in zip(ys_fit_line, ys_source)]

    xs_pts, ys_pts = _source_points(source)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(7, 6))

    ax1.plot(xs_line, ys_source, label="source", linewidth=1.5)
    if xs_pts:
        ax1.plot(xs_pts, ys_pts, "o", label="table points", markersize=4)
    ax1.plot(xs_fit, ys_fit, "-x", label="fit", linewidth=1.5, markersize=4)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, linestyle=":", linewidth=0.7)

    ax2.plot(xs_line, ys_err, label="error", linewidth=1.2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("error")
    ax2.grid(True, linestyle=":", linewidth=0.7)
    max_abs = max((abs(v) for v in ys_err), default=0.0)
    ax2.text(
        0.98,
        0.95,
        "max abs err = {:.6g}".format(max_abs),
        transform=ax2.transAxes,
        ha="right",
        va="top",
    )

    plt.tight_layout()
    return fig


def plot_fit(source: Source, res: FitResult, sample_points: int = 1000) -> None:
    fig = build_fit_figure(source, res, sample_points=sample_points)
    import matplotlib.pyplot as plt

    plt.show()
