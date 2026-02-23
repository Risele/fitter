from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from ios.parsing import sort_and_collapse_duplicates


class Source:
    def eval(self, x: float) -> float:
        raise NotImplementedError

    def domain(self) -> Tuple[float, float]:
        raise NotImplementedError


class AnalyticSource(Source):
    def __init__(self, fn: Callable[[float], float], xmin: float, xmax: float):
        if not (xmax > xmin):
            raise ValueError("xmax must be > xmin")
        self._fn = fn
        self._xmin = float(xmin)
        self._xmax = float(xmax)

    def eval(self, x: float) -> float:
        return float(self._fn(float(x)))

    def domain(self) -> Tuple[float, float]:
        return self._xmin, self._xmax


class TableSourceLinear(Source):
    def __init__(self, xy: Sequence[Tuple[float, float]]):
        pts = sort_and_collapse_duplicates(xy)
        self.x = [p[0] for p in pts]
        self.y = [p[1] for p in pts]
        self.n = len(pts)

    def domain(self) -> Tuple[float, float]:
        return self.x[0], self.x[-1]

    def eval(self, xq: float) -> float:
        xq = float(xq)
        if xq <= self.x[0]:
            x0, x1 = self.x[0], self.x[1]
            y0, y1 = self.y[0], self.y[1]
            k = (y1 - y0) / (x1 - x0)
            return y0 + (xq - x0) * k
        if xq >= self.x[-1]:
            x0, x1 = self.x[-2], self.x[-1]
            y0, y1 = self.y[-2], self.y[-1]
            k = (y1 - y0) / (x1 - x0)
            return y1 + (xq - x1) * k

        lo, hi = 0, self.n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.x[mid] <= xq:
                lo = mid
            else:
                hi = mid
        x0, x1 = self.x[lo], self.x[lo + 1]
        y0, y1 = self.y[lo], self.y[lo + 1]
        t = (xq - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


class TableSourceSpline(Source):
    """
    Natural cubic spline (second derivative = 0 at ends) + linear extrapolation by end slope.
    Без внешних библиотек.
    """

    def __init__(self, xy: Sequence[Tuple[float, float]]):
        pts = sort_and_collapse_duplicates(xy)
        self.x = [p[0] for p in pts]
        self.y = [p[1] for p in pts]
        self.n = len(pts)
        if self.n < 3:
            self._lin = TableSourceLinear(pts)
            self._m = None
        else:
            self._lin = None
            self._m = self._compute_second_derivatives()

    def domain(self) -> Tuple[float, float]:
        return self.x[0], self.x[-1]

    def _compute_second_derivatives(self) -> List[float]:
        n = self.n
        x = self.x
        y = self.y

        h = [x[i + 1] - x[i] for i in range(n - 1)]
        if any(v == 0.0 for v in h):
            raise ValueError("Zero spacing after duplicate collapse (unexpected)")

        a = [0.0] * n
        b = [0.0] * n
        c = [0.0] * n
        d = [0.0] * n

        b[0] = 1.0
        d[0] = 0.0
        b[n - 1] = 1.0
        d[n - 1] = 0.0

        for i in range(1, n - 1):
            a[i] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        for i in range(1, n):
            w = a[i] / b[i - 1] if b[i - 1] != 0.0 else 0.0
            b[i] -= w * c[i - 1]
            d[i] -= w * d[i - 1]

        m = [0.0] * n
        m[n - 1] = d[n - 1] / b[n - 1]
        for i in range(n - 2, -1, -1):
            m[i] = (d[i] - c[i] * m[i + 1]) / b[i]
        return m

    def eval(self, xq: float) -> float:
        xq = float(xq)

        if self._lin is not None:
            return self._lin.eval(xq)

        x = self.x
        y = self.y
        m = self._m
        assert m is not None

        if xq <= x[0]:
            k0 = self._spline_slope_at_left()
            return y[0] + (xq - x[0]) * k0
        if xq >= x[-1]:
            k1 = self._spline_slope_at_right()
            return y[-1] + (xq - x[-1]) * k1

        lo, hi = 0, self.n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x[mid] <= xq:
                lo = mid
            else:
                hi = mid

        x0, x1 = x[lo], x[lo + 1]
        y0, y1 = y[lo], y[lo + 1]
        m0, m1 = m[lo], m[lo + 1]
        h = x1 - x0
        t = (xq - x0) / h

        a = (1.0 - t)
        b = t
        return (
            a * y0
            + b * y1
            + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * (h * h) / 6.0
        )

    def _spline_slope_at_left(self) -> float:
        x0, x1 = self.x[0], self.x[1]
        y0, y1 = self.y[0], self.y[1]
        m0, m1 = self._m[0], self._m[1]
        h = x1 - x0
        return (y1 - y0) / h - (2.0 * m0 + m1) * h / 6.0

    def _spline_slope_at_right(self) -> float:
        n = self.n
        x0, x1 = self.x[n - 2], self.x[n - 1]
        y0, y1 = self.y[n - 2], self.y[n - 1]
        m0, m1 = self._m[n - 2], self._m[n - 1]
        h = x1 - x0
        return (y1 - y0) / h + (m0 + 2.0 * m1) * h / 6.0
