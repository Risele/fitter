from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol

from core.fitting import fit_uniform_table, summarize_nearby_n
from core.segment_estimator import SegmentEstimate
from core.types import FitResult
from ios.sources import Source


@dataclass
class AlgorithmRunResult:
    fit: FitResult
    rows: List[Dict[str, object]]


@dataclass
class AlgorithmRunConfig:
    abs_err: float
    sym_err: float
    n_max: int
    with_slopes: bool
    xmin_shift: bool = False


class FitAlgorithm(Protocol):
    algorithm_id: str
    display_name: str

    def run(
        self,
        source: Source,
        cfg: AlgorithmRunConfig,
        cancel_check: Optional[Callable[[], bool]] = None,
        estimate: Optional[SegmentEstimate] = None,
        progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> AlgorithmRunResult:
        ...


class LUTUniformAlgorithm:
    algorithm_id = "lut_uniform"
    display_name = "Uniform LUT"

    def run(
        self,
        source: Source,
        cfg: AlgorithmRunConfig,
        cancel_check: Optional[Callable[[], bool]] = None,
        estimate: Optional[SegmentEstimate] = None,
        progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> AlgorithmRunResult:
        mode = "y_only"
        n_hint = None
        if estimate is not None:
            n_hint = min(cfg.n_max, max(2, int(estimate.n_segments)))
        res = fit_uniform_table(
            source=source,
            abs_err=cfg.abs_err,
            sym_err=cfg.sym_err,
            mode=mode,
            n_max=cfg.n_max,
            n_hint=n_hint,
            xmin_shift=cfg.xmin_shift,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        rows = summarize_nearby_n(
            source=source,
            best=res,
            abs_err=cfg.abs_err,
            sym_err=cfg.sym_err,
            mode=mode,
            xmin_shift=cfg.xmin_shift,
        )
        return AlgorithmRunResult(fit=res, rows=rows)


def default_algorithm_registry() -> Dict[str, FitAlgorithm]:
    return {LUTUniformAlgorithm.algorithm_id: LUTUniformAlgorithm()}
