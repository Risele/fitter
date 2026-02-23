from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from codegen.formatting import emit_wrapped_array_items
from codegen.memory import estimate_memory_fixed
from core.types import FitResult, FixedPointConfig


def _int_limits(int_type: str) -> Tuple[int, int, str]:
    if int_type == "int16":
        return -32768, 32767, "int16_t"
    if int_type == "int32":
        return -2147483648, 2147483647, "int32_t"
    raise ValueError("int_type must be 'int16' or 'int32'")


def _assert_fits(values: Sequence[int], int_type: str, label: str) -> None:
    min_v, max_v, _ = _int_limits(int_type)
    for v in values:
        if v < min_v or v > max_v:
            raise AssertionError("{} value {} out of range for {}".format(label, v, int_type))


def _approx_eval_fixed_int(
    x_qS: int,
    xmin_qS: int,
    inv_dx_q: int,
    y: Sequence[int],
    k: Optional[Sequence[int]],
    n: int,
    Q: int,
    S: int,
) -> int:
    dx_qS = int(x_qS) - int(xmin_qS)
    t_q = (int(dx_qS) * int(inv_dx_q)) >> S
    i = int(t_q >> Q)
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    frac_q = int(t_q - (int(i) << Q))
    if k is None:
        dy = int(y[i + 1]) - int(y[i])
        y_out = int(y[i]) + ((frac_q * dy) >> Q)
    else:
        y_out = int(y[i]) + ((frac_q * int(k[i])) >> Q)
    return int(y_out)


def generate_c_fixed(
    res: FitResult,
    func_name: str,
    array_name: str,
    cfg: FixedPointConfig,
    store_slopes: bool,
    int_type: str,
    test_count: int = 64,
    max_line_len: int = 80,
    array_placement: str = "file_static",
) -> str:
    if array_placement not in ("file_static", "function_static"):
        raise ValueError("array_placement must be 'file_static' or 'function_static'")

    Q = int(cfg.q_frac)
    S = int(cfg.x_shift)
    y_scale = int(cfg.y_scale)
    if Q <= 0 or S < 0 or y_scale <= 0:
        raise ValueError("Invalid fixed-point config")

    n = res.n
    xmin = res.xmin_eff if res.xmin_eff is not None else res.xmin
    inv_dx = res.inv_dx

    xmin_int = int(round(xmin * (1 << S)))
    inv_dx_q = int(round(inv_dx * (1 << Q)))

    # Ensure constants fit int32_t
    if xmin_int < -2147483648 or xmin_int > 2147483647:
        raise AssertionError("xmin_int out of int32_t range")
    if inv_dx_q < -2147483648 or inv_dx_q > 2147483647:
        raise AssertionError("inv_dx_q out of int32_t range")

    y_int: List[int] = [int(round(v * y_scale)) for v in res.y]
    _assert_fits(y_int, int_type, "y")

    k_int: List[int] = []
    if store_slopes:
        k_float = res.k if res.k is not None else [res.y[i + 1] - res.y[i] for i in range(len(res.y) - 1)]
        k_int = [int(round(v * y_scale)) for v in k_float]
        _assert_fits(k_int, int_type, "k")

    min_v, max_v, y_type = _int_limits(int_type)
    _ = (min_v, max_v)

    test_x_qS: List[int] = []
    test_expected: List[int] = []
    if test_count > 0:
        x0 = xmin - res.dx
        x1 = res.xmax + res.dx
        for i in range(test_count):
            t = i / float(test_count - 1) if test_count > 1 else 0.0
            xf = x0 + t * (x1 - x0)
            x_qS = int(round(xf * (1 << S)))
            test_x_qS.append(x_qS)
        for x_qS in test_x_qS:
            test_expected.append(
                _approx_eval_fixed_int(
                    x_qS=x_qS,
                    xmin_qS=xmin_int,
                    inv_dx_q=inv_dx_q,
                    y=y_int,
                    k=k_int if store_slopes else None,
                    n=n,
                    Q=Q,
                    S=S,
                )
            )

    memory_bytes = estimate_memory_fixed(n, store_slopes, int_type)

    lines: List[str] = []

    def _append_array_block_int(dst: List[str], c_decl: str, values: Sequence[int], cast_type: Optional[str] = None) -> None:
        lead = c_decl[: len(c_decl) - len(c_decl.lstrip(" "))]
        item_indent = lead + "  "
        dst.append(c_decl + " = {")
        if cast_type:
            items = ["({}){}".format(cast_type, v) for v in values]
        else:
            items = [str(v) for v in values]
        dst.extend(emit_wrapped_array_items(items, indent=item_indent, max_line_len=max_line_len))
        dst.append(lead + "};")

    lines.append("/* Auto-generated fixed-point uniform table approximation */")
    lines.append("/*")
    lines.append("  x input: x_qS = round(x * 2^{} )  (int32)".format(S))
    lines.append("  y output: y_scaled = round(y * {})".format(y_scale))
    lines.append("  inv_dx_q = round(inv_dx * 2^{}) = {}".format(Q, inv_dx_q))
    lines.append("  xmin_qS  = round(xmin * 2^{})   = {}".format(S, xmin_int))
    lines.append("  memory_bytes = {}".format(memory_bytes))
    lines.append("*/")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("static const int32_t  {0}_xmin_qS   = (int32_t){1};".format(array_name, xmin_int))
    lines.append("static const int32_t  {0}_inv_dx_q  = (int32_t){1};".format(array_name, inv_dx_q))
    lines.append("static const uint32_t {0}_n         = {1}u;".format(array_name, n))
    lines.append("static const int32_t  {0}_Q         = {1};".format(array_name, Q))
    lines.append("static const int32_t  {0}_S         = {1};".format(array_name, S))
    lines.append("static const int32_t  {0}_y_scale   = {1};".format(array_name, y_scale))
    lines.append("")

    if array_placement == "file_static":
        _append_array_block_int(lines, "static const {0} {1}_y[{2}]".format(y_type, array_name, n), y_int, cast_type=y_type)
        lines.append("")

        if store_slopes:
            _append_array_block_int(lines, "static const {0} {1}_k[{2}]".format(y_type, array_name, n - 1), k_int, cast_type=y_type)
            lines.append("")

    lines.append("static inline int32_t _tbl_floor_q_to_i32(int64_t t_q, int32_t Q)")
    lines.append("{")
    lines.append("  int32_t i = (int32_t)(t_q >> Q);")
    lines.append("  return i;")
    lines.append("}")
    lines.append("")

    lines.append("static inline {0} {1}(int32_t x_qS)".format(y_type, func_name))
    lines.append("{")
    if array_placement == "function_static":
        _append_array_block_int(lines, "  static const {0} {1}_y[{2}]".format(y_type, array_name, n), y_int, cast_type=y_type)
        if store_slopes:
            _append_array_block_int(lines, "  static const {0} {1}_k[{2}]".format(y_type, array_name, n - 1), k_int, cast_type=y_type)
    lines.append("  int64_t dx_qS = (int64_t)x_qS - (int64_t){0}_xmin_qS;".format(array_name))
    lines.append("  int64_t t_q = (dx_qS * (int64_t){0}_inv_dx_q) >> {0}_S;".format(array_name))
    lines.append("  int32_t i = _tbl_floor_q_to_i32(t_q, {0}_Q);".format(array_name))
    lines.append("  if (i < 0) { i = 0; }")
    lines.append("  else if ((uint32_t)i > ({0}_n - 2u)) {{ i = (int32_t)({0}_n - 2u); }}".format(array_name))
    lines.append("  int64_t frac_q = t_q - ((int64_t)i << {0}_Q);".format(array_name))
    if store_slopes:
        lines.append("  int64_t y = (int64_t){0}_y[i] + ((frac_q * (int64_t){0}_k[i]) >> {0}_Q);".format(array_name))
    else:
        lines.append("  int64_t dy = (int64_t){0}_y[i + 1] - (int64_t){0}_y[i];".format(array_name))
        lines.append("  int64_t y = (int64_t){0}_y[i] + ((frac_q * dy) >> {0}_Q);".format(array_name))
    lines.append("  return ({0})y;".format(y_type))
    lines.append("}")
    lines.append("")

    if test_count > 0:
        lines.append("/* Test vectors: verify generated function on target */")
        lines.append("static const uint32_t {0}_test_n = {1}u;".format(array_name, test_count))
        _append_array_block_int(lines, "static const int32_t {0}_test_x_qS[{1}]".format(array_name, test_count), test_x_qS)
        _append_array_block_int(lines, "static const {0} {1}_test_expected[{2}]".format(y_type, array_name, test_count), test_expected, cast_type=y_type)
        lines.append("/* Suggested check (pseudo):")
        lines.append("   for i: y = {0}({1}_test_x_qS[i]);".format(func_name, array_name))
        lines.append("           assert( y == {0}_test_expected[i] );".format(array_name))
        lines.append("*/")

    return "\n".join(lines)
