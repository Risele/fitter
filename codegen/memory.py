from __future__ import annotations


def _element_size_from_precision(precision: str) -> int:
    if precision == "float":
        return 4
    if precision == "double":
        return 8
    raise ValueError("precision must be 'float' or 'double'")


def _element_size_from_int_type(int_type: str) -> int:
    if int_type == "int16":
        return 2
    if int_type == "int32":
        return 4
    raise ValueError("int_type must be 'int16' or 'int32'")


def estimate_memory_float(N: int, use_slopes: bool, precision: str) -> int:
    size = _element_size_from_precision(precision)
    if use_slopes:
        return N * size + (N - 1) * size
    return N * size


def estimate_memory_fixed(N: int, use_slopes: bool, int_type: str) -> int:
    size = _element_size_from_int_type(int_type)
    if use_slopes:
        return N * size + (N - 1) * size
    return N * size
