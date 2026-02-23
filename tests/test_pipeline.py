import os
import unittest
from dataclasses import replace

from app.pipeline import (
    FitConfig,
    build_source,
    estimate_for_config,
    fit_result_from_dict,
    fit_result_to_dict,
    render_outputs,
    run_fit,
    run_fit_compute,
    split_fit_config,
)


class PipelineTest(unittest.TestCase):
    def test_run_fit_csv(self):
        path = os.path.join(os.path.dirname(__file__), "test.csv")
        cfg = FitConfig(
            path=path,
            interp="linear",
            mode="float",
            abs_err=1e-2,
            sym_err=1e-2,
            n_max=1024,
            with_slopes=False,
            test_count=0,
        )
        bundle = run_fit(cfg)
        self.assertTrue(len(bundle.summary) > 0)
        self.assertTrue(len(bundle.c_code) > 0)
        self.assertNotIn("Estimator (always-on curvature bound):", bundle.summary)
        self.assertIsNotNone(bundle.estimate)

    def test_run_fit_analytic(self):
        cfg = FitConfig(
            func="sin",
            xmin=-2.0,
            xmax=5.0,
            interp="linear",
            mode="float",
            abs_err=1e-2,
            sym_err=1e-2,
            n_max=1024,
            with_slopes=False,
            test_count=0,
        )
        bundle = run_fit(cfg)
        self.assertTrue(len(bundle.summary) > 0)
        self.assertTrue(len(bundle.c_code) > 0)
        self.assertNotIn("Estimator (always-on curvature bound):", bundle.summary)
        self.assertIsNotNone(bundle.estimate)

    def test_run_fit_uses_precomputed_estimate(self):
        cfg = FitConfig(
            func="sin",
            xmin=-2.0,
            xmax=5.0,
            interp="linear",
            mode="float",
            abs_err=1e-2,
            sym_err=1e-2,
            n_max=1024,
            with_slopes=False,
            test_count=0,
        )
        src = build_source(cfg)
        est = estimate_for_config(cfg, source=src)
        bundle = run_fit(cfg, source=src, estimate=est)
        self.assertIsNotNone(bundle.estimate)
        self.assertEqual(bundle.estimate.n_segments, est.n_segments)

    def test_split_compute_and_render_match_wrapper(self):
        cfg = FitConfig(
            func="sin",
            xmin=-2.0,
            xmax=5.0,
            interp="linear",
            mode="float",
            abs_err=1e-2,
            sym_err=1e-2,
            n_max=1024,
            with_slopes=False,
            test_count=0,
        )
        src = build_source(cfg)
        est = estimate_for_config(cfg, source=src)
        wrapped = run_fit(cfg, source=src, estimate=est)
        fit_cfg, code_cfg = split_fit_config(cfg)
        comp = run_fit_compute(fit_cfg, source=src, estimate=est)
        rendered = render_outputs(comp, code_cfg)
        self.assertEqual(comp.fit.n, wrapped.fit.n)
        self.assertEqual(len(comp.rows), len(wrapped.rows))
        self.assertEqual(rendered.summary, wrapped.summary)
        self.assertEqual(rendered.c_code, wrapped.c_code)

    def test_rerender_codegen_without_refit_changes_output(self):
        cfg = FitConfig(
            func="sin",
            xmin=-2.0,
            xmax=5.0,
            interp="linear",
            mode="float",
            abs_err=1e-2,
            sym_err=1e-2,
            n_max=1024,
            with_slopes=False,
            test_count=0,
        )
        fit_cfg, code_cfg = split_fit_config(cfg)
        comp = run_fit_compute(fit_cfg)
        rendered1 = render_outputs(comp, code_cfg)
        code_cfg2 = replace(code_cfg, func_name="f_alt")
        rendered2 = render_outputs(comp, code_cfg2)
        self.assertNotEqual(rendered1.c_code, rendered2.c_code)
        self.assertEqual(rendered1.summary, rendered2.summary)

    def test_fit_result_serialization_roundtrip(self):
        cfg = FitConfig(
            func="sin",
            xmin=-2.0,
            xmax=5.0,
            test_count=0,
        )
        bundle = run_fit(cfg)
        restored = fit_result_from_dict(fit_result_to_dict(bundle.fit))
        self.assertEqual(restored.n, bundle.fit.n)
        self.assertEqual(len(restored.y), len(bundle.fit.y))


if __name__ == "__main__":
    unittest.main()
