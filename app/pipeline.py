from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from app.reporting import format_report, render_summary
from app.runtime_config import codegen_max_line_length, default_array_placement
from codegen.c_fixed import generate_c_fixed
from codegen.c_float import generate_c_float
from codegen.memory import estimate_memory_fixed, estimate_memory_float
from core.algorithms import AlgorithmRunConfig, default_algorithm_registry
from core.segment_estimator import SegmentEstimate, estimate_min_uniform_segments
from core.types import FitResult, FixedPointConfig
from ios.expr import compile_expression, load_expression
from ios.parsing import read_xy_csv
from ios.sources import AnalyticSource, Source, TableSourceLinear, TableSourceSpline


@dataclass
class FitConfig:
    path: Optional[str] = None
    func: Optional[str] = None
    func_file: Optional[str] = None
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    a: float = 1.0
    b: float = 0.0
    interp: str = "linear"
    mode: str = "float"
    abs_err: float = 1e-2
    sym_err: float = 1e-2
    n_max: int = 16384
    with_slopes: bool = False
    test_count: int = 64
    xmin_shift: bool = False
    q_frac: int = 20
    x_shift: int = 12
    y_scale: int = 100000
    func_name: str = "f_approx"
    array_name: str = "tbl_f"
    array_placement: Optional[str] = None
    algorithm_ids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class FitExecutionConfig:
    path: Optional[str] = None
    func: Optional[str] = None
    func_file: Optional[str] = None
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    a: float = 1.0
    b: float = 0.0
    interp: str = "linear"
    abs_err: float = 1e-2
    sym_err: float = 1e-2
    n_max: int = 16384
    with_slopes: bool = False
    xmin_shift: bool = False
    algorithm_ids: Optional[List[str]] = None


@dataclass
class CodegenConfig:
    mode: str = "float"
    with_slopes: bool = False
    test_count: int = 64
    q_frac: int = 20
    x_shift: int = 12
    y_scale: int = 100000
    func_name: str = "f_approx"
    array_name: str = "tbl_f"
    array_placement: str = "file_static"
    max_line_len: int = 80


@dataclass
class FitComputationBundle:
    source: Source
    fit: FitResult
    rows: List[Dict[str, object]]
    estimate: Optional[SegmentEstimate] = None


@dataclass
class RenderedOutputBundle:
    summary: str
    c_code: str
    report_text: str


@dataclass
class FitResultBundle:
    source: Source
    fit: FitResult
    summary: str
    c_code: str
    report_text: str
    rows: List[Dict[str, object]]
    estimate: Optional[SegmentEstimate] = None


def fit_result_to_dict(res: FitResult) -> Dict[str, object]:
    return asdict(res)


def fit_result_from_dict(data: Dict[str, object]) -> FitResult:
    return FitResult(
        name=str(data.get("name", "")),
        xmin=float(data.get("xmin", 0.0)),
        xmax=float(data.get("xmax", 0.0)),
        n=int(data.get("n", 0)),
        dx=float(data.get("dx", 0.0)),
        inv_dx=float(data.get("inv_dx", 0.0)),
        y=[float(v) for v in (data.get("y", []) or [])],
        k=([float(v) for v in data.get("k", [])] if data.get("k") is not None else None),
        max_abs_err=float(data.get("max_abs_err", 0.0)),
        mean_signed_err=float(data.get("mean_signed_err", 0.0)),
        verify_points=int(data.get("verify_points", 0)),
        xmin_eff=(float(data["xmin_eff"]) if data.get("xmin_eff") is not None else None),
        shift=float(data.get("shift", 0.0)),
    )


def estimate_to_dict(est: Optional[SegmentEstimate]) -> Optional[Dict[str, object]]:
    if est is None:
        return None
    return asdict(est)


def estimate_from_dict(data: Optional[Dict[str, object]]) -> Optional[SegmentEstimate]:
    if not data:
        return None
    return SegmentEstimate(**data)


def rows_to_list(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [dict(r) for r in rows]


def rows_from_list(rows: object) -> List[Dict[str, object]]:
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, object]] = []
    for item in rows:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _analytic_function(name: str, a: float, b: float):
    name = name.lower()
    if name == "sin":
        import math

        return math.sin
    if name == "cos":
        import math

        return math.cos
    if name == "exp":
        import math

        return math.exp
    if name == "log":
        import math

        return math.log
    if name == "linear":
        return lambda x: a * x + b
    raise ValueError("Unknown analytic function: {}".format(name))


def _is_builtin_func(value: str) -> bool:
    value = value.strip().lower()
    if "(" in value or ")" in value:
        return False
    return value in {"sin", "cos", "exp", "log", "linear"}


def _validate_config(config: FitConfig) -> None:
    if config.func is None and config.func_file is None and config.path is None:
        raise ValueError("Provide either a path or an analytic function/file")
    if config.path is not None and (config.func is not None or config.func_file is not None):
        raise ValueError("Provide either a path or an analytic function/file, not both")
    if config.func is not None and config.func_file is not None:
        raise ValueError("Provide either func or func_file, not both")
    if config.func is not None or config.func_file is not None:
        if config.xmin is None or config.xmax is None:
            raise ValueError("Analytic source requires xmin and xmax")
        if not (config.xmax > config.xmin):
            raise ValueError("xmax must be > xmin")

    if config.interp not in ("linear", "spline"):
        raise ValueError("Unknown interpolation: {}".format(config.interp))

    if config.mode not in ("float", "double", "int16", "int32"):
        raise ValueError("Unknown numeric mode: {}".format(config.mode))

    if config.n_max <= 1:
        raise ValueError("n_max must be > 1")


def _validate_fit_execution_config(config: FitExecutionConfig) -> None:
    cfg = FitConfig(
        path=config.path,
        func=config.func,
        func_file=config.func_file,
        xmin=config.xmin,
        xmax=config.xmax,
        a=config.a,
        b=config.b,
        interp=config.interp,
        mode="float",
        abs_err=config.abs_err,
        sym_err=config.sym_err,
        n_max=config.n_max,
        with_slopes=config.with_slopes,
        xmin_shift=config.xmin_shift,
    )
    _validate_config(cfg)


def _build_source(config: FitConfig) -> Source:
    if config.func is None and config.func_file is None:
        assert config.path is not None
        xy = read_xy_csv(config.path)
        if config.interp == "linear":
            return TableSourceLinear(xy)
        return TableSourceSpline(xy)

    if config.func_file is not None:
        expr = load_expression(config.func_file)
        fn = compile_expression(expr)
    else:
        assert config.func is not None
        value = config.func
        if _is_builtin_func(value):
            fn = _analytic_function(value, config.a, config.b)
        else:
            fn = compile_expression(value)
    return AnalyticSource(fn, xmin=config.xmin, xmax=config.xmax)


def _build_source_from_exec(config: FitExecutionConfig) -> Source:
    return _build_source(
        FitConfig(
            path=config.path,
            func=config.func,
            func_file=config.func_file,
            xmin=config.xmin,
            xmax=config.xmax,
            a=config.a,
            b=config.b,
            interp=config.interp,
            mode="float",
            abs_err=config.abs_err,
            sym_err=config.sym_err,
            n_max=config.n_max,
            with_slopes=config.with_slopes,
            xmin_shift=config.xmin_shift,
        )
    )


def build_source(config: FitConfig) -> Source:
    _validate_config(config)
    return _build_source(config)


def _build_dense_xy(source: Source, n_max: int) -> List[Tuple[float, float]]:
    xmin, xmax = source.domain()
    # Always-on deterministic dense sampling for curvature-based N estimation.
    base = 4 * int(math.sqrt(max(2, n_max))) + 1
    samples = min(4097, max(513, base))
    if samples % 2 == 0:
        samples += 1
    if samples < 3:
        samples = 3
    dx = (xmax - xmin) / float(samples - 1)
    return [(xmin + i * dx, source.eval(xmin + i * dx)) for i in range(samples)]


def _estimate_n_hint(source: Source, abs_err: float, n_max: int) -> SegmentEstimate:
    dense_xy = _build_dense_xy(source, n_max)
    return estimate_min_uniform_segments(
        xy=dense_xy,
        epsilon=abs_err,
        safety_factor=1.2,
    )


def estimate_for_config(config: FitConfig, source: Optional[Source] = None) -> SegmentEstimate:
    _validate_config(config)
    src = source if source is not None else _build_source(config)
    return _estimate_n_hint(source=src, abs_err=config.abs_err, n_max=config.n_max)


def split_fit_config(config: FitConfig) -> Tuple[FitExecutionConfig, CodegenConfig]:
    _validate_config(config)
    exec_cfg = FitExecutionConfig(
        path=config.path,
        func=config.func,
        func_file=config.func_file,
        xmin=config.xmin,
        xmax=config.xmax,
        a=config.a,
        b=config.b,
        interp=config.interp,
        abs_err=config.abs_err,
        sym_err=config.sym_err,
        n_max=config.n_max,
        with_slopes=config.with_slopes,
        xmin_shift=config.xmin_shift,
        algorithm_ids=(list(config.algorithm_ids) if config.algorithm_ids else ["lut_uniform"]),
    )
    code_cfg = CodegenConfig(
        mode=config.mode,
        with_slopes=config.with_slopes,
        test_count=config.test_count,
        q_frac=config.q_frac,
        x_shift=config.x_shift,
        y_scale=config.y_scale,
        func_name=config.func_name,
        array_name=config.array_name,
        array_placement=(config.array_placement or default_array_placement()),
        max_line_len=codegen_max_line_length(),
    )
    return exec_cfg, code_cfg


def run_fit_compute(
    fit_cfg: FitExecutionConfig,
    cancel_check=None,
    source: Optional[Source] = None,
    estimate: Optional[SegmentEstimate] = None,
    progress_callback=None,
) -> FitComputationBundle:
    _validate_fit_execution_config(fit_cfg)
    src = source if source is not None else _build_source_from_exec(fit_cfg)
    est = estimate if estimate is not None else _estimate_n_hint(source=src, abs_err=fit_cfg.abs_err, n_max=fit_cfg.n_max)
    registry = default_algorithm_registry()
    algorithm_ids = list(fit_cfg.algorithm_ids or ["lut_uniform"])
    if not algorithm_ids:
        algorithm_ids = ["lut_uniform"]
    algo_id = algorithm_ids[0]
    algo = registry.get(algo_id)
    if algo is None:
        raise ValueError("Unknown algorithm: {}".format(algo_id))
    algo_result = algo.run(
        source=src,
        cfg=AlgorithmRunConfig(
            abs_err=fit_cfg.abs_err,
            sym_err=fit_cfg.sym_err,
            n_max=fit_cfg.n_max,
            with_slopes=False,
            xmin_shift=fit_cfg.xmin_shift,
        ),
        cancel_check=cancel_check,
        estimate=est,
        progress_callback=progress_callback,
    )
    return FitComputationBundle(
        source=src,
        fit=algo_result.fit,
        rows=algo_result.rows,
        estimate=est,
    )


def render_outputs(comp: FitComputationBundle, codegen_cfg: CodegenConfig) -> RenderedOutputBundle:
    if codegen_cfg.mode in ("float", "double"):
        precision = codegen_cfg.mode
        memory_fn = lambda n: estimate_memory_float(n, codegen_cfg.with_slopes, precision)
        summary = render_summary(comp.fit, comp.rows, memory_fn)
        c_code = generate_c_float(
            res=comp.fit,
            func_name=codegen_cfg.func_name,
            array_name=codegen_cfg.array_name,
            store_slopes=codegen_cfg.with_slopes,
            test_count=codegen_cfg.test_count,
            precision=precision,
            max_line_len=codegen_cfg.max_line_len,
            array_placement=codegen_cfg.array_placement,
        )
    else:
        int_type = codegen_cfg.mode
        cfg = FixedPointConfig(q_frac=codegen_cfg.q_frac, x_shift=codegen_cfg.x_shift, y_scale=codegen_cfg.y_scale)
        memory_fn = lambda n: estimate_memory_fixed(n, codegen_cfg.with_slopes, int_type)
        summary = render_summary(comp.fit, comp.rows, memory_fn)
        c_code = generate_c_fixed(
            res=comp.fit,
            func_name=codegen_cfg.func_name,
            array_name=codegen_cfg.array_name,
            cfg=cfg,
            store_slopes=codegen_cfg.with_slopes,
            int_type=int_type,
            test_count=codegen_cfg.test_count,
            max_line_len=codegen_cfg.max_line_len,
            array_placement=codegen_cfg.array_placement,
        )

    report_text = format_report(summary, c_code)
    return RenderedOutputBundle(summary=summary, c_code=c_code, report_text=report_text)


def run_fit(
    config: FitConfig,
    cancel_check=None,
    source: Optional[Source] = None,
    estimate: Optional[SegmentEstimate] = None,
    progress_callback=None,
) -> FitResultBundle:
    exec_cfg, code_cfg = split_fit_config(config)
    comp = run_fit_compute(
        exec_cfg,
        cancel_check=cancel_check,
        source=source,
        estimate=estimate,
        progress_callback=progress_callback,
    )
    rendered = render_outputs(comp, code_cfg)
    return FitResultBundle(
        source=comp.source,
        fit=comp.fit,
        summary=rendered.summary,
        c_code=rendered.c_code,
        report_text=rendered.report_text,
        rows=comp.rows,
        estimate=comp.estimate,
    )


def resolve_out_path(path: str) -> str:
    import os

    root, ext = os.path.splitext(path)
    if ext:
        return path
    return path + ".txt"
