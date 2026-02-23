from __future__ import annotations

from typing import List, Optional, Sequence
import math

from codegen.formatting import emit_wrapped_array_items
from codegen.memory import estimate_memory_float
from core.evaluation import _approx_eval_float
from core.types import FitResult


def _c_fp_literal(x: float, precision: str) -> str:
    if not math.isfinite(x):
        raise ValueError("Non-finite float cannot be emitted")

    if precision == "float":
        s = "{:.10g}".format(x)
        if "e" not in s and "." not in s:
            s += ".0"
        return s + "f"

    if precision == "double":
        s = "{:.17g}".format(x)
        if "e" not in s and "." not in s:
            s += ".0"
        return s

    raise ValueError("precision must be 'float' or 'double'")


def generate_c_float(
    res: FitResult,
    func_name: str,
    array_name: str,
    store_slopes: bool,
    test_count: int = 64,
    precision: str = "float",
    max_line_len: int = 80,
    array_placement: str = "file_static",
) -> str:
    n = res.n
    y = res.y
    xmin = res.xmin_eff if res.xmin_eff is not None else res.xmin
    inv_dx = res.inv_dx
    dx = res.dx

    if array_placement not in ("file_static", "function_static"):
        raise ValueError("array_placement must be 'file_static' or 'function_static'")

    k: Optional[Sequence[float]] = None
    if store_slopes:
        k = res.k if res.k is not None else [y[i + 1] - y[i] for i in range(n - 1)]

    test_xs: List[float] = []
    test_apr: List[float] = []
    if test_count > 0:
        x0 = xmin - dx
        x1 = res.xmax + dx
        for i in range(test_count):
            t = i / float(test_count - 1) if test_count > 1 else 0.0
            test_xs.append(x0 + t * (x1 - x0))
        for x in test_xs:
            test_apr.append(_approx_eval_float(x, xmin, inv_dx, y, k))

    c_type = "float" if precision == "float" else "double"
    literal = lambda v: _c_fp_literal(v, precision)
    memory_bytes = estimate_memory_float(n, store_slopes, precision)

    lines: List[str] = []

    def _append_array_block(dst: List[str], c_decl: str, values: Sequence[float]) -> None:
        lead = c_decl[: len(c_decl) - len(c_decl.lstrip(" "))]
        item_indent = lead + "  "
        dst.append(c_decl + " = {")
        items = [literal(v) for v in values]
        dst.extend(emit_wrapped_array_items(items, indent=item_indent, max_line_len=max_line_len))
        dst.append(lead + "};")

    lines.append("/* Auto-generated uniform table approximation */")
    lines.append("/*")
    lines.append("  N            = {}".format(n))
    lines.append("  xmin         = {}".format("{:.10g}".format(xmin)))
    lines.append("  xmax         = {}".format("{:.10g}".format(res.xmax)))
    lines.append("  dx           = {}".format("{:.10g}".format(res.dx)))
    lines.append("  inv_dx       = {}".format("{:.10g}".format(res.inv_dx)))
    lines.append("  max_abs_err  = {}   (verified on {} points)".format("{:.10g}".format(res.max_abs_err), res.verify_points))
    lines.append("  mean_err     = {}".format("{:.10g}".format(res.mean_signed_err)))
    lines.append("  memory_bytes = {}".format(memory_bytes))
    lines.append("*/")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("static const {0} {1}_xmin = {2};".format(c_type, array_name, literal(xmin)))
    lines.append("static const {0} {1}_inv_dx = {2};".format(c_type, array_name, literal(inv_dx)))
    lines.append("static const uint32_t {0}_n = {1}u;".format(array_name, n))
    lines.append("")

    if array_placement == "file_static":
        _append_array_block(lines, "static const {0} {1}_y[{2}]".format(c_type, array_name, n), y)
        lines.append("")
        if store_slopes:
            assert k is not None
            _append_array_block(lines, "static const {0} {1}_k[{2}]".format(c_type, array_name, n - 1), k)
            lines.append("")

    lines.append("static inline int32_t _tbl_floor_to_i32({} t)".format(c_type))
    lines.append("{")
    if precision == "float":
        lines.append("  int32_t i = (int32_t)t; /* trunc toward 0 */")
        lines.append("  if ((t < 0.0f) && ((float)i != t)) { i -= 1; }")
    else:
        lines.append("  int32_t i = (int32_t)t; /* trunc toward 0 */")
        lines.append("  if ((t < 0.0) && ((double)i != t)) { i -= 1; }")
    lines.append("  return i;")
    lines.append("}")
    lines.append("")

    lines.append("static inline {0} {1}({0} x)".format(c_type, func_name))
    lines.append("{")
    if array_placement == "function_static":
        _append_array_block(lines, "  static const {0} {1}_y[{2}]".format(c_type, array_name, n), y)
        if store_slopes:
            assert k is not None
            _append_array_block(lines, "  static const {0} {1}_k[{2}]".format(c_type, array_name, n - 1), k)
    lines.append("  {0} t = (x - {1}_xmin) * {1}_inv_dx;".format(c_type, array_name))
    lines.append("  int32_t i = _tbl_floor_to_i32(t);")
    lines.append("  if (i < 0) { i = 0; }")
    lines.append("  else if ((uint32_t)i > ({0}_n - 2u)) {{ i = (int32_t)({0}_n - 2u); }}".format(array_name))
    lines.append("  {0} frac = t - ({0})i;".format(c_type))
    if store_slopes:
        lines.append("  return {0}_y[i] + frac * {0}_k[i];".format(array_name))
    else:
        lines.append("  return {0}_y[i] + frac * ({0}_y[i + 1] - {0}_y[i]);".format(array_name))
    lines.append("}")
    lines.append("")

    if test_count > 0:
        lines.append("/* Test vectors: verify generated function on target */")
        lines.append("static const uint32_t {0}_test_n = {1}u;".format(array_name, test_count))
        _append_array_block(lines, "static const {0} {1}_test_x[{2}]".format(c_type, array_name, test_count), test_xs)
        _append_array_block(lines, "static const {0} {1}_test_expected[{2}]".format(c_type, array_name, test_count), test_apr)
        lines.append("/* Suggested check (pseudo):")
        lines.append("   for i: y = {0}({1}_test_x[i]);".format(func_name, array_name))
        lines.append("           assert( abs(y - {0}_test_expected[i]) <= tol );".format(array_name))
        lines.append("*/")

    return "\n".join(lines)
