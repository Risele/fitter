import unittest
import math
import time

from core.fitting import eval_uniform_n, fit_uniform_table
from core.types import FitError
from ios.sources import AnalyticSource


try:
    import scipy  # noqa: F401
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class LSFittingTests(unittest.TestCase):
    def _require_scipy(self):
        if not SCIPY_AVAILABLE:
            raise unittest.SkipTest("scipy not available")

    def test_xmin_shift_bounds(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: x * 0.5 + 1.0, xmin=-2.0, xmax=5.0)
        res = eval_uniform_n(
            source=src,
            n=32,
            mode="y_only",
            verify_ppi=16,
            verify_extra_points=128,
            xmin_shift=True,
        )
        dx0 = (res.xmax - res.xmin) / float(res.n - 1)
        self.assertLessEqual(res.shift, 0.0)
        self.assertGreaterEqual(res.shift, -dx0 - 1e-12)
        self.assertIsNotNone(res.xmin_eff)
        self.assertAlmostEqual(res.xmin + res.shift, res.xmin_eff, places=12)

    def test_lp_strict_abs_err_for_linear(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: 2.0 * x - 3.0, xmin=-2.0, xmax=5.0)
        res = eval_uniform_n(
            source=src,
            n=16,
            mode="y_only",
            xmin_shift=False,
            lp_abs_err=1e-9,
        )
        self.assertLessEqual(res.max_abs_err, 1e-9 + 1e-12)
        self.assertAlmostEqual(res.shift, 0.0, places=12)
        self.assertAlmostEqual(res.xmin_eff, res.xmin, places=12)

    def test_bias_is_small_for_odd_symmetric_function(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: x * x * x, xmin=-1.0, xmax=1.0)
        res = eval_uniform_n(
            source=src,
            n=64,
            mode="y_only",
            xmin_shift=False,
            lp_abs_err=5e-4,
        )
        self.assertLessEqual(abs(res.mean_signed_err), 1e-4)

    def test_constraints_unsatisfied_fields(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: x * x, xmin=-2.0, xmax=5.0)
        with self.assertRaises(FitError) as ctx:
            fit_uniform_table(
                source=src,
                abs_err=1e-8,
                sym_err=1e-8,
                mode="y_only",
                n_min=2,
                n_max=2,
                xmin_shift=False,
            )
        exc = ctx.exception
        self.assertEqual(exc.code, "constraints_unsatisfied")
        self.assertIn("n_max", exc.data)
        self.assertIn("max_abs", exc.data)
        self.assertIn("mean_signed", exc.data)

    def test_cancelled_error(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: x * x, xmin=-2.0, xmax=5.0)

        def _cancel():
            return True

        with self.assertRaises(FitError) as ctx:
            fit_uniform_table(
                source=src,
                abs_err=1e-2,
                sym_err=1e-2,
                mode="y_only",
                n_min=2,
                n_max=16,
                xmin_shift=False,
                cancel_check=_cancel,
            )
        self.assertEqual(ctx.exception.code, "cancelled")

    def test_plot_has_two_axes(self):
        self._require_scipy()
        try:
            import matplotlib
        except Exception:
            raise unittest.SkipTest("matplotlib not available")
        matplotlib.use("Agg")

        from app.plotting import build_fit_figure

        src = AnalyticSource(lambda x: x * x, xmin=-2.0, xmax=5.0)
        res = eval_uniform_n(
            source=src,
            n=16,
            mode="y_only",
            verify_ppi=8,
            verify_extra_points=64,
            xmin_shift=False,
        )
        fig = build_fit_figure(src, res, sample_points=200)
        self.assertEqual(len(fig.axes), 2)
        fig.clf()

    def test_sin_fit_reaches_abs_err_under_512(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: math.sin(x), xmin=-2.0, xmax=5.0)
        res = fit_uniform_table(
            source=src,
            abs_err=0.1,
            sym_err=1e-4,
            mode="y_only",
            n_min=2,
            n_max=512,
            verify_ppi=12,
            verify_extra_points=256,
            xmin_shift=False,
        )
        self.assertLess(res.n, 512)
        self.assertLessEqual(res.max_abs_err, 0.1 + 1e-12)

    def test_infeasible_n2_rejected_quickly(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: math.sin(x), xmin=-2.0, xmax=5.0)
        t0 = time.perf_counter()
        res = eval_uniform_n(
            source=src,
            n=2,
            mode="y_only",
            verify_ppi=12,
            verify_extra_points=256,
            xmin_shift=False,
            lp_abs_err=1e-3,
        )
        elapsed = time.perf_counter() - t0
        self.assertGreater(res.max_abs_err, 1e-3)
        self.assertLess(elapsed, 8.0)

    def test_sin_abs_1e2_no_artificial_n_bloat(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: math.sin(x), xmin=-2.0, xmax=5.0)
        res = fit_uniform_table(
            source=src,
            abs_err=1e-2,
            sym_err=1e-2,
            mode="y_only",
            n_min=2,
            n_max=1024,
            verify_ppi=12,
            verify_extra_points=256,
            xmin_shift=False,
        )
        tol = max(1e-12, 1e-2 * 1e-6)
        self.assertLessEqual(res.max_abs_err, 1e-2 + tol)
        self.assertLessEqual(res.n, 32)

    def test_n_hint_does_not_break_min_search(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: math.sin(x), xmin=-2.0, xmax=5.0)
        res = fit_uniform_table(
            source=src,
            abs_err=1e-2,
            sym_err=1e-2,
            mode="y_only",
            n_min=2,
            n_max=1024,
            n_hint=224,
            verify_ppi=12,
            verify_extra_points=256,
            xmin_shift=False,
        )
        tol = max(1e-12, 1e-2 * 1e-6)
        self.assertLessEqual(res.max_abs_err, 1e-2 + tol)
        self.assertLessEqual(res.n, 32)

    def test_sin_abs_1e3_no_infinite_expand(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: math.sin(x), xmin=-2.0, xmax=5.0)
        res = fit_uniform_table(
            source=src,
            abs_err=1e-3,
            sym_err=1e-2,
            mode="y_only",
            n_min=2,
            n_max=1024,
            verify_ppi=12,
            verify_extra_points=256,
            xmin_shift=False,
        )
        tol = max(1e-12, 1e-3 * 1e-6)
        self.assertLessEqual(res.max_abs_err, 1e-3 + tol)
        self.assertLessEqual(res.n, 96)

    def test_stage2_does_not_inflate_far_above_minimax(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: math.sin(x), xmin=-2.0, xmax=5.0)
        constrained = eval_uniform_n(
            source=src,
            n=32,
            mode="y_only",
            verify_ppi=12,
            verify_extra_points=256,
            xmin_shift=False,
            lp_abs_err=1e-2,
        )
        minimax_like = eval_uniform_n(
            source=src,
            n=32,
            mode="y_only",
            verify_ppi=12,
            verify_extra_points=256,
            xmin_shift=False,
            lp_abs_err=None,
        )
        self.assertLessEqual(constrained.max_abs_err, minimax_like.max_abs_err * 1.1 + 1e-8)
        self.assertLess(constrained.max_abs_err, 5e-3)

    def test_verify_ppi_and_extra_points_control_verification_grid(self):
        self._require_scipy()
        src = AnalyticSource(lambda x: math.sin(x), xmin=-2.0, xmax=5.0)
        low = eval_uniform_n(
            source=src,
            n=32,
            mode="y_only",
            verify_ppi=4,
            verify_extra_points=0,
            xmin_shift=False,
            lp_abs_err=0.2,
        )
        high = eval_uniform_n(
            source=src,
            n=32,
            mode="y_only",
            verify_ppi=16,
            verify_extra_points=500,
            xmin_shift=False,
            lp_abs_err=0.2,
        )
        self.assertGreater(high.verify_points, low.verify_points)

    def test_summary_rows_are_ordered_around_solution(self):
        self._require_scipy()
        from core.fitting import summarize_nearby_n

        src = AnalyticSource(lambda x: math.sin(x), xmin=-2.0, xmax=5.0)
        best = fit_uniform_table(
            source=src,
            abs_err=1e-2,
            sym_err=1e-2,
            mode="y_only",
            n_min=2,
            n_max=256,
            verify_ppi=12,
            verify_extra_points=128,
            xmin_shift=False,
        )
        rows = summarize_nearby_n(src, best, 1e-2, 1e-2, mode="y_only", xmin_shift=False)
        ns = [int(r["N"]) for r in rows]
        self.assertGreaterEqual(len(ns), 1)
        self.assertEqual(ns[0], best.n)
        self.assertIn(best.n + 1, ns)
        self.assertIn(best.n + 2, ns)
        self.assertTrue(all(n >= 2 for n in ns))
        for row in rows:
            for key in ("N", "dx", "max_abs_err", "mean_signed_err", "ok_abs", "ok_sym"):
                self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
