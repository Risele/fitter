from __future__ import annotations

from typing import Callable, Dict, List, TextIO

from core.types import FitResult


def _format_summary_table(
    rows: List[Dict[str, object]],
    memory_fn: Callable[[int], int],
) -> str:
    header = ("N", "dx", "max_abs_err", "mean_signed_error", "memory_bytes", "ok(abs)", "ok(signed)")
    rows_str: List[List[str]] = []
    rows_str.append(list(header))

    for r in rows:
        n = int(r["N"])
        mem = memory_fn(n)
        ok_abs = "yes" if r["ok_abs"] else "no"
        ok_sym = "yes" if r["ok_sym"] else "no"
        rows_str.append(
            [
                str(n),
                "{:.6g}".format(r["dx"]),
                "{:.6g}".format(r["max_abs_err"]),
                "{:.6g}".format(r["mean_signed_err"]),
                str(mem),
                ok_abs,
                ok_sym,
            ]
        )

    colw = [0] * len(header)
    for r in rows_str:
        for i, cell in enumerate(r):
            colw[i] = max(colw[i], len(cell))

    def fmt_row(r: List[str]) -> str:
        return "  ".join(cell.rjust(colw[i]) for i, cell in enumerate(r))

    lines: List[str] = []
    lines.append(fmt_row(rows_str[0]))
    lines.append("  ".join("-" * w for w in colw))
    for idx, r in enumerate(rows_str[1:], start=1):
        lines.append(fmt_row(r))
        if idx == 1 and len(rows_str) > 2:
            lines.append("-" * max(4, len(lines[-1])))

    return "\n".join(lines)


def render_summary(res: FitResult, rows: List[Dict[str, object]], memory_fn: Callable[[int], int]) -> str:
    memory_bytes = memory_fn(res.n)
    lines: List[str] = []
    lines.append("Memory usage (bytes): {}".format(memory_bytes))
    lines.append(_format_summary_table(rows, memory_fn))
    return "\n".join(lines)


def format_report(summary: str, c_code: str, comment_style: str = "block") -> str:
    if comment_style != "block":
        raise ValueError("Only block comment style is supported")
    lines: List[str] = []
    lines.append("/*")
    for line in summary.splitlines():
        lines.append(line)
    lines.append("*/")
    lines.append("")
    lines.append(c_code)
    return "\n".join(lines)


def emit_report(summary: str, c_code: str, out: TextIO) -> None:
    out.write(format_report(summary, c_code, comment_style="block") + "\n")


def emit_text(text: str, out: TextIO) -> None:
    out.write(text)
