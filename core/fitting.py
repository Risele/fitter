from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

from core.types import FitError, FitResult
from ios.sources import Source


ADAPT_MAX_ITERS = 4
ADAPT_BATCH_POINTS = 6
VERIFY_CHECK_FACTOR = 2
VERIFY_CHECK_MAX_POINTS = 200000
GRID_CTRL_MAX_POINTS = 250000
MINIMAX_STAGE2_RELAX = 1e-6
ACCEPT_REL_TOL = 1e-6


def _timing_enabled() -> bool:
    raw = os.getenv("FITTER_TIMING_LOG", "0").strip().lower()
    return raw not in ("0", "false", "no", "off")


class _TraceLogger:
    def __init__(self) -> None:
        self.enabled = _timing_enabled()
        self.depth = 0

    def _format_kv(self, values: Dict[str, object]) -> str:
        if not values:
            return ""
        parts: List[str] = []
        for key, value in values.items():
            if isinstance(value, float):
                parts.append("{}={:.6g}".format(key, value))
            else:
                parts.append("{}={}".format(key, value))
        return ", ".join(parts)

    @contextmanager
    def span(self, name: str, **params: object):
        if not self.enabled:
            yield
            return

        enter_prefix = "|  " * self.depth + "+- "
        param_text = self._format_kv(params)
        if param_text:
            print("[fit-timing] {}{}({})".format(enter_prefix, name, param_text))
        else:
            print("[fit-timing] {}{}".format(enter_prefix, name))

        self.depth += 1
        t0 = time.perf_counter()
        ok = True
        try:
            yield
        except Exception:
            ok = False
            raise
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.depth -= 1
            exit_prefix = "|  " * self.depth + "\\- "
            status = "ok" if ok else "error"
            print("[fit-timing] {}{} [{}] {:.3f} ms".format(exit_prefix, name, status, dt_ms))

    def log(self, message: str, **fields: object) -> None:
        if not self.enabled:
            return
        prefix = "|  " * self.depth + "* "
        text = self._format_kv(fields)
        if text:
            print("[fit-timing] {}{}: {}".format(prefix, message, text))
        else:
            print("[fit-timing] {}{}".format(prefix, message))


_TRACE = _TraceLogger()


def _require_scipy():
    with _TRACE.span("_require_scipy"):
        try:
            import numpy as np
            from scipy.optimize import linprog
            from scipy.sparse import csr_matrix, vstack
        except Exception as exc:
            raise RuntimeError("scipy is required for LP fitting") from exc
        return np, linprog, csr_matrix, vstack


def _linspace(np, xmin: float, xmax: float, count: int):
    if count <= 1:
        return np.array([xmin], dtype=np.float64)
    return np.linspace(xmin, xmax, int(count), dtype=np.float64)


def _build_interval_points(np, xmin: float, xmax: float, xmin_eff: float, dx: float, n: int, ppi: int):
    seg = np.arange(n - 1, dtype=np.float64)
    sub = (np.arange(ppi, dtype=np.float64) + 0.5) / float(ppi)
    xs = xmin_eff + seg[:, None] * dx + sub[None, :] * dx
    flat = xs.reshape(-1)
    mask = (flat >= xmin) & (flat <= xmax)
    return flat[mask]


def _build_extra_points(np, xmin: float, xmax: float, extra_points: int):
    if extra_points <= 0:
        return np.empty(0, dtype=np.float64)
    xs = _linspace(np, xmin, xmax, extra_points + 2)
    return xs[1:-1]


def _dedup_sorted(np, xs, ys=None, tol: float = 1e-15):
    if xs.shape[0] <= 1:
        return xs, ys
    keep = np.ones(xs.shape[0], dtype=bool)
    keep[1:] = np.diff(xs) > tol
    if ys is None:
        return xs[keep], None
    return xs[keep], ys[keep]


def _finalize_grid(
    np,
    xmin: float,
    xmax: float,
    points: List,
    max_points: Optional[int] = None,
):
    xs = np.concatenate(points) if points else np.empty(0, dtype=np.float64)
    xs = xs[(xs >= xmin) & (xs <= xmax)]
    xs = np.concatenate([xs, np.array([xmin, xmax], dtype=np.float64)])
    xs = np.unique(xs)
    if max_points is not None and xs.shape[0] > max_points:
        xs = _linspace(np, xmin, xmax, max_points)
    return xs


def _build_ctrl_grid(
    np,
    xmin: float,
    xmax: float,
    xmin_eff: float,
    dx: float,
    n: int,
    verify_ppi: int,
    verify_extra_points: int,
):
    interval_points = _build_interval_points(np, xmin, xmax, xmin_eff, dx, n, max(1, verify_ppi))
    extra_points = _build_extra_points(np, xmin, xmax, verify_extra_points)
    nodes = xmin_eff + np.arange(n, dtype=np.float64) * dx
    nodes = nodes[(nodes >= xmin) & (nodes <= xmax)]
    return _finalize_grid(
        np=np,
        xmin=xmin,
        xmax=xmax,
        points=[interval_points, extra_points, nodes],
        max_points=GRID_CTRL_MAX_POINTS,
    )


def _build_check_grid(
    np,
    xmin: float,
    xmax: float,
    xmin_eff: float,
    dx: float,
    n: int,
    verify_ppi: int,
    verify_extra_points: int,
):
    check_ppi = max(verify_ppi, verify_ppi * VERIFY_CHECK_FACTOR)
    interval_points = _build_interval_points(np, xmin, xmax, xmin_eff, dx, n, max(1, check_ppi))
    extra_points = _build_extra_points(np, xmin, xmax, verify_extra_points)
    nodes = xmin_eff + np.arange(n, dtype=np.float64) * dx
    nodes = nodes[(nodes >= xmin) & (nodes <= xmax)]
    return _finalize_grid(
        np=np,
        xmin=xmin,
        xmax=xmax,
        points=[interval_points, extra_points, nodes],
        max_points=VERIFY_CHECK_MAX_POINTS,
    )


def _eval_piecewise_linear(np, xs, xmin_eff: float, inv_dx: float, y, k):
    n = int(y.shape[0])
    t = (xs - xmin_eff) * inv_dx
    i = np.floor(t).astype(np.int64)
    i = np.clip(i, 0, n - 2)
    frac = t - i.astype(np.float64)

    yi = y[i]
    if k is None:
        yi1 = y[i + 1]
        return yi + frac * (yi1 - yi)

    ki = k[i]
    return yi + frac * ki


def _build_interp_rows(np, xs, xmin_eff: float, inv_dx: float, n: int):
    t = (xs - xmin_eff) * inv_dx
    i = np.floor(t).astype(np.int64)
    i = np.clip(i, 0, n - 2)
    frac = t - i.astype(np.float64)
    a = 1.0 - frac
    b = frac
    return i, a, b


def _build_interp_ub_matrix(np, csr_matrix, i, a, b, n_vars: int, extra_col: Optional[int] = None, extra_val: float = -1.0):
    m = int(i.shape[0])
    extra_nnz = 2 * m if extra_col is not None else 0
    nnz = 4 * m + extra_nnz
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)

    idx = np.arange(m, dtype=np.int64)
    r1 = 2 * idx
    r2 = r1 + 1

    rows[0:m] = r1
    cols[0:m] = i
    data[0:m] = a

    rows[m : 2 * m] = r1
    cols[m : 2 * m] = i + 1
    data[m : 2 * m] = b

    rows[2 * m : 3 * m] = r2
    cols[2 * m : 3 * m] = i
    data[2 * m : 3 * m] = -a

    rows[3 * m : 4 * m] = r2
    cols[3 * m : 4 * m] = i + 1
    data[3 * m : 4 * m] = -b

    if extra_col is not None:
        start = 4 * m
        rows[start : start + m] = r1
        cols[start : start + m] = extra_col
        data[start : start + m] = extra_val

        rows[start + m : start + 2 * m] = r2
        cols[start + m : start + 2 * m] = extra_col
        data[start + m : start + 2 * m] = extra_val

    return csr_matrix((data, (rows, cols)), shape=(2 * m, n_vars))


def _error_constraints(np, csr_matrix, i, a, b, fx, eps: float, n_vars: int):
    A_ub = _build_interp_ub_matrix(np, csr_matrix, i, a, b, n_vars=n_vars, extra_col=None)
    b_ub = np.empty(2 * fx.shape[0], dtype=np.float64)
    b_ub[0::2] = fx + eps
    b_ub[1::2] = eps - fx
    return A_ub, b_ub


def _bias_weights(np, i, a, b, n: int):
    w = np.zeros(n, dtype=np.float64)
    m = float(i.shape[0])
    if m <= 0:
        return w
    inv_m = 1.0 / m
    np.add.at(w, i, a * inv_m)
    np.add.at(w, i + 1, b * inv_m)
    return w


def _solve_stage2_bias(
    np,
    linprog,
    csr_matrix,
    vstack,
    i_err,
    a_err,
    b_err,
    fx_err,
    eps: float,
    i_mean,
    a_mean,
    b_mean,
    fx_mean,
    n: int,
    cancel_check: Optional[Callable[[], bool]],
):
    with _TRACE.span("_solve_stage2_bias", n=n, eps=eps, m_err=int(fx_err.shape[0]), m_mean=int(fx_mean.shape[0])):
        if cancel_check and cancel_check():
            raise FitError(code="cancelled", data={}, message="Fit cancelled")

        n_vars = n + 1
        A_err, b_err_vec = _error_constraints(np, csr_matrix, i_err, a_err, b_err, fx_err, eps, n_vars)

        w = _bias_weights(np, i_mean, a_mean, b_mean, n)
        f_mean = float(np.mean(fx_mean)) if fx_mean.shape[0] else 0.0

        nz = np.nonzero(w)[0].astype(np.int64)
        row_idx = np.empty(2 * nz.size + 2, dtype=np.int64)
        col_idx = np.empty(2 * nz.size + 2, dtype=np.int64)
        dat = np.empty(2 * nz.size + 2, dtype=np.float64)

        row_idx[: nz.size] = 0
        col_idx[: nz.size] = nz
        dat[: nz.size] = -w[nz]

        row_idx[nz.size : 2 * nz.size] = 1
        col_idx[nz.size : 2 * nz.size] = nz
        dat[nz.size : 2 * nz.size] = w[nz]

        row_idx[2 * nz.size :] = np.array([0, 1], dtype=np.int64)
        col_idx[2 * nz.size :] = np.array([n, n], dtype=np.int64)
        dat[2 * nz.size :] = np.array([-1.0, -1.0], dtype=np.float64)

        A_bias = csr_matrix((dat, (row_idx, col_idx)), shape=(2, n_vars))
        b_bias = np.array([-f_mean, f_mean], dtype=np.float64)

        A_ub = vstack([A_err, A_bias], format="csr")
        b_ub = np.concatenate([b_err_vec, b_bias])

        c = np.zeros(n_vars, dtype=np.float64)
        c[n] = 1.0
        bounds = [(None, None)] * n + [(0.0, None)]

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        _TRACE.log("linprog.stage2", success=bool(res.success), status=getattr(res, "status", "na"))
        if not res.success:
            return None
        return np.array(res.x[:n], dtype=np.float64)


def _solve_minimax(
    np,
    linprog,
    csr_matrix,
    i,
    a,
    b,
    fx,
    n: int,
    cancel_check: Optional[Callable[[], bool]],
):
    with _TRACE.span("_solve_minimax", n=n, m=int(fx.shape[0])):
        if cancel_check and cancel_check():
            raise FitError(code="cancelled", data={}, message="Fit cancelled")

        n_vars = n + 1
        t_idx = n

        A_ub = _build_interp_ub_matrix(np, csr_matrix, i, a, b, n_vars=n_vars, extra_col=t_idx, extra_val=-1.0)
        b_ub = np.empty(2 * fx.shape[0], dtype=np.float64)
        b_ub[0::2] = fx
        b_ub[1::2] = -fx

        c = np.zeros(n_vars, dtype=np.float64)
        c[t_idx] = 1.0
        bounds = [(None, None)] * n + [(0.0, None)]

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        _TRACE.log("linprog.minimax", success=bool(res.success), status=getattr(res, "status", "na"))
        if not res.success:
            return None, None
        t_star = float(res.x[t_idx])
        y_minimax = np.array(res.x[:n], dtype=np.float64)
        return y_minimax, t_star


def _eval_metrics(np, xs_check, fx_check, xmin_eff: float, inv_dx: float, y, mode: str):
    with _TRACE.span("_eval_metrics", points=int(xs_check.shape[0])):
        k_np = None
        if mode == "y_and_slope":
            k_np = y[1:] - y[:-1]
        appr = _eval_piecewise_linear(np, xs_check, xmin_eff, inv_dx, y, k_np)
        err = appr - fx_check
        abs_err = np.abs(err)
        max_abs = float(np.max(abs_err))
        mean_signed = float(np.mean(err))
        return fx_check, err, abs_err, max_abs, mean_signed, k_np


def _solve_for_shift(
    source: Source,
    xmin_orig: float,
    xmax: float,
    n: int,
    mode: str,
    xmin_eff: float,
    eps_target: Optional[float],
    cancel_check: Optional[Callable[[], bool]],
    verify_ppi: int,
    verify_extra_points: int,
) -> Tuple[Optional[FitResult], bool]:
    with _TRACE.span(
        "_solve_for_shift",
        n=n,
        xmin_eff=xmin_eff,
        shift=(xmin_eff - xmin_orig),
        eps_target=(-1.0 if eps_target is None else eps_target),
        verify_ppi=verify_ppi,
        verify_extra_points=verify_extra_points,
    ):
        np, linprog, csr_matrix, vstack = _require_scipy()

        dx = (xmax - xmin_eff) / float(n - 1)
        inv_dx = 1.0 / dx

        xs_ctrl = _build_ctrl_grid(
            np=np,
            xmin=xmin_orig,
            xmax=xmax,
            xmin_eff=xmin_eff,
            dx=dx,
            n=n,
            verify_ppi=verify_ppi,
            verify_extra_points=verify_extra_points,
        )
        xs_check = _build_check_grid(
            np=np,
            xmin=xmin_orig,
            xmax=xmax,
            xmin_eff=xmin_eff,
            dx=dx,
            n=n,
            verify_ppi=verify_ppi,
            verify_extra_points=verify_extra_points,
        )
        _TRACE.log("grid_sizes", n=n, xs_ctrl=int(xs_ctrl.shape[0]), xs_check=int(xs_check.shape[0]))

        fx_ctrl = np.array([source.eval(float(x)) for x in xs_ctrl], dtype=np.float64)
        i_mean, a_mean, b_mean = _build_interp_rows(np, xs_check, xmin_eff, inv_dx, n)
        fx_mean = np.array([source.eval(float(x)) for x in xs_check], dtype=np.float64)

        best: Optional[FitResult] = None
        strict_ok = eps_target is None
        eps_rel = 1.0 + 1e-12

        for adapt_iter in range(ADAPT_MAX_ITERS + 1):
            if cancel_check and cancel_check():
                raise FitError(code="cancelled", data={}, message="Fit cancelled")

            _TRACE.log("adapt_iter_begin", idx=adapt_iter, ctrl_points=int(xs_ctrl.shape[0]))
            i_ctrl, a_ctrl, b_ctrl = _build_interp_rows(np, xs_ctrl, xmin_eff, inv_dx, n)

            with _TRACE.span("step_minimax", n=n, adapt_iter=adapt_iter):
                y_minimax, t_star = _solve_minimax(
                    np=np,
                    linprog=linprog,
                    csr_matrix=csr_matrix,
                    i=i_ctrl,
                    a=a_ctrl,
                    b=b_ctrl,
                    fx=fx_ctrl,
                    n=n,
                    cancel_check=cancel_check,
                )

            if y_minimax is None or t_star is None:
                return None, False
            _TRACE.log("minimax_result", t_star=t_star)

            if eps_target is not None and t_star > eps_target * eps_rel:
                _TRACE.log("minimax_over_target", t_star=t_star, eps_target=eps_target, adapt_iter=adapt_iter)
                fx_check, err, abs_err, max_abs, mean_signed, k_np = _eval_metrics(
                    np=np,
                    xs_check=xs_check,
                    fx_check=fx_mean,
                    xmin_eff=xmin_eff,
                    inv_dx=inv_dx,
                    y=y_minimax,
                    mode=mode,
                )
                strict_ok = False
                best = FitResult(
                    name="table",
                    xmin=xmin_orig,
                    xmax=xmax,
                    n=n,
                    dx=dx,
                    inv_dx=inv_dx,
                    y=[float(v) for v in y_minimax],
                    k=[float(v) for v in k_np] if k_np is not None else None,
                    max_abs_err=max_abs,
                    mean_signed_err=mean_signed,
                    verify_points=int(xs_check.shape[0]),
                    xmin_eff=xmin_eff,
                    shift=xmin_eff - xmin_orig,
                )
                return best, strict_ok

            y_eval = y_minimax
            if eps_target is not None:
                eps_delta = max(1e-12, abs(t_star) * MINIMAX_STAGE2_RELAX)
                eps_bias = min(eps_target, max(0.0, t_star) + eps_delta)
                _TRACE.log(
                    "stage2_eps_policy",
                    minimax_first=True,
                    t_star=t_star,
                    eps_target=eps_target,
                    stage2_eps_used=eps_bias,
                )
                with _TRACE.span("step_bias", n=n, adapt_iter=adapt_iter, eps=eps_bias):
                    y_bias = _solve_stage2_bias(
                        np=np,
                        linprog=linprog,
                        csr_matrix=csr_matrix,
                        vstack=vstack,
                        i_err=i_ctrl,
                        a_err=a_ctrl,
                        b_err=b_ctrl,
                        fx_err=fx_ctrl,
                        eps=eps_bias,
                        i_mean=i_mean,
                        a_mean=a_mean,
                        b_mean=b_mean,
                        fx_mean=fx_mean,
                        n=n,
                        cancel_check=cancel_check,
                    )
                if y_bias is not None:
                    y_eval = y_bias
                else:
                    _TRACE.log("stage2_bias_fallback", reason="infeasible_or_solver_fail", fallback_to_minimax=True)
            else:
                eps_bias = max(0.0, t_star) + max(1e-12, abs(t_star) * 1e-9)
                _TRACE.log(
                    "stage2_eps_policy",
                    minimax_first=True,
                    t_star=t_star,
                    eps_target=(-1.0 if eps_target is None else eps_target),
                    stage2_eps_used=eps_bias,
                )
                with _TRACE.span("step_bias", n=n, adapt_iter=adapt_iter, eps=eps_bias):
                    y_bias = _solve_stage2_bias(
                        np=np,
                        linprog=linprog,
                        csr_matrix=csr_matrix,
                        vstack=vstack,
                        i_err=i_ctrl,
                        a_err=a_ctrl,
                        b_err=b_ctrl,
                        fx_err=fx_ctrl,
                        eps=eps_bias,
                        i_mean=i_mean,
                        a_mean=a_mean,
                        b_mean=b_mean,
                        fx_mean=fx_mean,
                        n=n,
                        cancel_check=cancel_check,
                    )
                if y_bias is not None:
                    y_eval = y_bias
                else:
                    _TRACE.log("stage2_bias_fallback", reason="infeasible_or_solver_fail", fallback_to_minimax=True)

            with _TRACE.span("step_check", n=n, adapt_iter=adapt_iter):
                fx_check, err, abs_err, max_abs, mean_signed, k_np = _eval_metrics(
                    np=np,
                    xs_check=xs_check,
                    fx_check=fx_mean,
                    xmin_eff=xmin_eff,
                    inv_dx=inv_dx,
                    y=y_eval,
                    mode=mode,
                )

            _TRACE.log(
                "check_metrics",
                n=n,
                adapt_iter=adapt_iter,
                max_abs=max_abs,
                mean_signed=mean_signed,
                t_star=t_star,
                xs_ctrl=int(xs_ctrl.shape[0]),
                xs_check=int(xs_check.shape[0]),
            )

            best = FitResult(
                name="table",
                xmin=xmin_orig,
                xmax=xmax,
                n=n,
                dx=dx,
                inv_dx=inv_dx,
                y=[float(v) for v in y_eval],
                k=[float(v) for v in k_np] if k_np is not None else None,
                max_abs_err=max_abs,
                mean_signed_err=mean_signed,
                verify_points=int(xs_check.shape[0]),
                xmin_eff=xmin_eff,
                shift=xmin_eff - xmin_orig,
            )

            eps_accept_tol = max(1e-12, eps_target * ACCEPT_REL_TOL) if eps_target is not None else 0.0
            if eps_target is not None and max_abs <= (eps_target + eps_accept_tol):
                strict_ok = True
                _TRACE.log("strict_eps_satisfied", at_iter=adapt_iter, max_abs=max_abs, eps_target=eps_target, eps_tol=eps_accept_tol)
                break

            can_adapt = True
            if eps_target is not None and t_star > eps_target * eps_rel:
                can_adapt = False
            if not can_adapt:
                break
            if adapt_iter >= ADAPT_MAX_ITERS:
                _TRACE.log("adapt_stop", reason="max_iters")
                break

            order = np.argsort(abs_err)[::-1]
            new_x: List[float] = []
            new_fx: List[float] = []
            for idx in order:
                if len(new_x) >= ADAPT_BATCH_POINTS:
                    break
                if eps_target is not None and float(abs_err[idx]) <= (eps_target + eps_accept_tol):
                    break
                x_star = float(xs_check[int(idx)])
                pos = int(np.searchsorted(xs_ctrl, x_star))
                near = False
                if pos < xs_ctrl.shape[0] and abs(float(xs_ctrl[pos]) - x_star) <= 1e-15:
                    near = True
                if pos > 0 and abs(float(xs_ctrl[pos - 1]) - x_star) <= 1e-15:
                    near = True
                if near:
                    continue
                new_x.append(x_star)
                new_fx.append(float(fx_check[int(idx)]))

            if not new_x:
                _TRACE.log("adapt_stop", reason="no_new_points")
                break

            xs_ctrl = np.concatenate([xs_ctrl, np.array(new_x, dtype=np.float64)])
            fx_ctrl = np.concatenate([fx_ctrl, np.array(new_fx, dtype=np.float64)])
            order_ctrl = np.argsort(xs_ctrl)
            xs_ctrl = xs_ctrl[order_ctrl]
            fx_ctrl = fx_ctrl[order_ctrl]
            xs_ctrl, fx_ctrl = _dedup_sorted(np, xs_ctrl, fx_ctrl)
            _TRACE.log("adapt_points_added", count=len(new_x), ctrl_points=int(xs_ctrl.shape[0]))

        if best is None:
            return None, False
        return best, strict_ok


def eval_uniform_n(
    source: Source,
    n: int,
    mode: str,  # "y_only" or "y_and_slope"
    verify_ppi: int = 32,
    verify_extra_points: int = 1024,
    xmin_shift: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
    lp_abs_err: Optional[float] = None,
) -> FitResult:
    with _TRACE.span(
        "eval_uniform_n",
        n=n,
        mode=mode,
        xmin_shift=xmin_shift,
        lp_abs_err=(-1.0 if lp_abs_err is None else lp_abs_err),
        verify_ppi=verify_ppi,
        verify_extra_points=verify_extra_points,
    ):
        if n < 2:
            raise ValueError("n must be >= 2")
        if mode not in ("y_only", "y_and_slope"):
            raise ValueError("mode must be 'y_only' or 'y_and_slope'")
        if verify_ppi < 2:
            raise ValueError("verify_ppi must be >= 2")
        if verify_extra_points < 0:
            raise ValueError("verify_extra_points must be >= 0")

        xmin, xmax = source.domain()

        def pick_best(candidates: List[FitResult], shifts: List[float]) -> FitResult:
            best = candidates[0]
            best_shift = shifts[0]
            for res, sh in zip(candidates[1:], shifts[1:]):
                if res.max_abs_err < best.max_abs_err:
                    best, best_shift = res, sh
                    continue
                if res.max_abs_err > best.max_abs_err:
                    continue
                if abs(res.mean_signed_err) < abs(best.mean_signed_err):
                    best, best_shift = res, sh
                    continue
                if abs(res.mean_signed_err) > abs(best.mean_signed_err):
                    continue
                if abs(sh) < abs(best_shift):
                    best, best_shift = res, sh
            return best

        if not xmin_shift:
            res, _ = _solve_for_shift(
                source=source,
                xmin_orig=xmin,
                xmax=xmax,
                n=n,
                mode=mode,
                xmin_eff=xmin,
                eps_target=lp_abs_err,
                cancel_check=cancel_check,
                verify_ppi=verify_ppi,
                verify_extra_points=verify_extra_points,
            )
            if res is None:
                raise FitError(code="constraints_unsatisfied", data={"n_max": n, "max_abs": float("inf"), "mean_signed": float("inf")})
            return res

        dx0 = (xmax - xmin) / float(n - 1)
        coarse_step = dx0 / 32.0
        _TRACE.log("shift_search_coarse", from_shift=-dx0, to_shift=0.0, step=coarse_step)

        shifts1: List[float] = []
        results1: List[FitResult] = []
        for i in range(33):
            if cancel_check and cancel_check():
                raise FitError(code="cancelled", data={}, message="Fit cancelled")
            shift = -dx0 + i * coarse_step
            xmin_eff = xmin + shift
            res, _ = _solve_for_shift(
                source=source,
                xmin_orig=xmin,
                xmax=xmax,
                n=n,
                mode=mode,
                xmin_eff=xmin_eff,
                eps_target=lp_abs_err,
                cancel_check=cancel_check,
                verify_ppi=verify_ppi,
                verify_extra_points=verify_extra_points,
            )
            if res is None:
                continue
            shifts1.append(shift)
            results1.append(res)

        if not results1:
            raise FitError(code="constraints_unsatisfied", data={"n_max": n, "max_abs": float("inf"), "mean_signed": float("inf")})

        best1 = pick_best(results1, shifts1)
        best_shift = best1.shift

        win_start = max(-dx0, best_shift - 2.0 * coarse_step)
        win_end = min(0.0, best_shift + 2.0 * coarse_step)
        fine_step = (win_end - win_start) / 16.0 if win_end > win_start else 0.0
        _TRACE.log("shift_search_fine", from_shift=win_start, to_shift=win_end, step=fine_step)

        shifts2: List[float] = []
        results2: List[FitResult] = []
        for i in range(17):
            if cancel_check and cancel_check():
                raise FitError(code="cancelled", data={}, message="Fit cancelled")
            shift = win_start + i * fine_step if fine_step > 0 else best_shift
            shift = min(0.0, max(-dx0, shift))
            xmin_eff = xmin + shift
            res, _ = _solve_for_shift(
                source=source,
                xmin_orig=xmin,
                xmax=xmax,
                n=n,
                mode=mode,
                xmin_eff=xmin_eff,
                eps_target=lp_abs_err,
                cancel_check=cancel_check,
                verify_ppi=verify_ppi,
                verify_extra_points=verify_extra_points,
            )
            if res is None:
                continue
            shifts2.append(shift)
            results2.append(res)

        if not results2:
            _TRACE.log("shift_result", stage="coarse", best_shift=best1.shift, max_abs=best1.max_abs_err)
            return best1
        best2 = pick_best(results2, shifts2)
        _TRACE.log("shift_result", stage="fine", best_shift=best2.shift, max_abs=best2.max_abs_err)
        return best2


def fit_uniform_table(
    source: Source,
    abs_err: float,
    sym_err: float,
    mode: str,  # "y_only" or "y_and_slope"
    n_min: int = 2,
    n_max: int = 65536,
    n_hint: Optional[int] = None,
    verify_ppi: int = 32,
    verify_extra_points: int = 1024,
    xmin_shift: bool = False,
    cancel_check: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> FitResult:
    with _TRACE.span(
        "fit_uniform_table",
        abs_err=abs_err,
        sym_err=sym_err,
        n_min=n_min,
        n_max=n_max,
        n_hint=(-1 if n_hint is None else n_hint),
        mode=mode,
        xmin_shift=xmin_shift,
        verify_ppi=verify_ppi,
        verify_extra_points=verify_extra_points,
    ):
        if abs_err <= 0:
            raise ValueError("abs_err must be > 0")
        if sym_err < 0:
            raise ValueError("sym_err must be >= 0")
        if verify_ppi < 2:
            raise ValueError("verify_ppi must be >= 2")
        if verify_extra_points < 0:
            raise ValueError("verify_extra_points must be >= 0")
        if mode not in ("y_only", "y_and_slope"):
            raise ValueError("mode must be 'y_only' or 'y_and_slope'")

        best_ok_so_far: Optional[FitResult] = None

        def build_and_eval(n: int) -> Tuple[bool, FitResult]:
            nonlocal best_ok_so_far
            with _TRACE.span("candidate_n", n=n):
                eps_internal = max(1e-15, abs_err - max(1e-9, abs_err * 1e-5))
                res = eval_uniform_n(
                    source=source,
                    n=n,
                    mode=mode,
                    verify_ppi=verify_ppi,
                    verify_extra_points=verify_extra_points,
                    xmin_shift=xmin_shift,
                    cancel_check=cancel_check,
                    lp_abs_err=eps_internal,
                )
                abs_tol = max(1e-12, abs_err * ACCEPT_REL_TOL)
                sym_tol = max(1e-15, sym_err * ACCEPT_REL_TOL)
                ok_abs = res.max_abs_err <= (abs_err + abs_tol)
                ok_sym = abs(res.mean_signed_err) <= (sym_err + sym_tol)
                ok = ok_abs and ok_sym
                if ok and (best_ok_so_far is None or res.n < best_ok_so_far.n):
                    best_ok_so_far = res
                _TRACE.log(
                    "candidate_metrics",
                    ok=ok,
                    ok_abs=ok_abs,
                    ok_sym=ok_sym,
                    max_abs=res.max_abs_err,
                    mean_signed=res.mean_signed_err,
                    abs_tol=abs_tol,
                    sym_tol=sym_tol,
                )
                if progress_callback is not None:
                    try:
                        payload: Dict[str, object] = {
                            "stage": "candidate",
                            "checked_n": int(n),
                            "ok": bool(ok),
                            "max_abs_err": float(res.max_abs_err),
                            "mean_signed_err": float(res.mean_signed_err),
                        }
                        if best_ok_so_far is not None:
                            payload["best_n"] = int(best_ok_so_far.n)
                            payload["best_max_abs_err"] = float(best_ok_so_far.max_abs_err)
                        progress_callback(payload)
                    except Exception:
                        pass
                return ok, res

        lo_min = max(n_min, 2)

        def refine_down(lo: int, hi: int, best: FitResult) -> FitResult:
            while lo <= hi:
                mid = (lo + hi) // 2
                okm, resm = build_and_eval(mid)
                if okm:
                    best = resm
                    hi = mid - 1
                else:
                    lo = mid + 1
            return best

        if n_hint is not None:
            n0 = min(n_max, max(lo_min, int(n_hint)))
            _TRACE.log("search_seed", n_hint=n_hint, n0=n0, n_min=lo_min, n_max=n_max)
        else:
            n0 = lo_min
            _TRACE.log("search_seed", n_hint=-1, n0=n0, n_min=lo_min, n_max=n_max)

        ok0, res0 = build_and_eval(n0)
        if ok0:
            if n0 == lo_min:
                return res0
            best = refine_down(lo_min, n0 - 1, res0)
            _TRACE.log("search_done", best_n=best.n, max_abs=best.max_abs_err, mean_signed=best.mean_signed_err)
            return best

        n = n0
        res = res0

        while n < n_max:
            if cancel_check and cancel_check():
                raise FitError(code="cancelled", data={}, message="Fit cancelled")
            n2 = min(n_max, n * 2)
            _TRACE.log("search_expand", prev_n=n, next_n=n2)
            ok2, res2 = build_and_eval(n2)
            if ok2:
                lo = n + 1
                hi = n2
                best = res2
                best = refine_down(lo, hi, best)
                _TRACE.log("search_done", best_n=best.n, max_abs=best.max_abs_err, mean_signed=best.mean_signed_err)
                return best
            n = n2
            res = res2

        raise FitError(
            code="constraints_unsatisfied",
            data={
                "n_max": n_max,
                "max_abs": res.max_abs_err,
                "mean_signed": res.mean_signed_err,
            },
            message=(
                "Cannot satisfy error constraints up to n_max={}"
                ". Last max_abs={}, mean={}".format(n_max, res.max_abs_err, res.mean_signed_err)
            ),
        )


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _prev_pow2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while (p << 1) < n:
        p <<= 1
    return p


def summarize_nearby_n(
    source: Source,
    best: FitResult,
    abs_err: float,
    sym_err: float,
    mode: str,
    verify_ppi: int = 32,
    verify_extra_points: int = 1024,
    xmin_shift: bool = False,
) -> List[Dict[str, object]]:
    with _TRACE.span("summarize_nearby_n", best_n=best.n, mode=mode):
        next_pw2 = next_pow2(best.n)
        if next_pw2 == best.n:
            next_pw2 = best.n * 2
        prev_pw2 = _prev_pow2(best.n)
        ordered_candidates = [
            best.n,
            prev_pw2,
            best.n - 2,
            best.n - 1,
            best.n + 1,
            best.n + 2,
            next_pw2,
        ]
        cand: List[int] = []
        seen = set()
        for n in ordered_candidates:
            if n < 2:
                continue
            if n in seen:
                continue
            seen.add(n)
            cand.append(n)

        rows: List[Dict[str, object]] = []
        for n in cand:
            res = eval_uniform_n(
                source=source,
                n=n,
                mode=mode,
                verify_ppi=verify_ppi,
                verify_extra_points=verify_extra_points,
                xmin_shift=xmin_shift,
            )
            ok_abs = res.max_abs_err <= abs_err
            ok_sym = abs(res.mean_signed_err) <= sym_err
            _TRACE.log("summary_row", n=n, max_abs=res.max_abs_err, mean_signed=res.mean_signed_err, ok_abs=ok_abs, ok_sym=ok_sym)
            rows.append(
                {
                    "N": n,
                    "dx": res.dx,
                    "max_abs_err": res.max_abs_err,
                    "mean_signed_err": res.mean_signed_err,
                    "ok_abs": ok_abs,
                    "ok_sym": ok_sym,
                }
            )
        return rows
