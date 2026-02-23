import unittest

from core.segment_estimator import estimate_min_uniform_segments


class SegmentEstimatorTests(unittest.TestCase):
    def test_uniform_grid_quadratic(self):
        xy = [(0.0, 0.0), (0.25, 0.0625), (0.5, 0.25), (0.75, 0.5625), (1.0, 1.0)]
        res = estimate_min_uniform_segments(xy=xy, epsilon=0.01, safety_factor=1.0)
        self.assertTrue(res.is_uniform_input_grid)
        self.assertAlmostEqual(res.m_raw, 2.0, places=12)
        self.assertEqual(res.n_segments, 5)
        self.assertTrue(res.check_passed)
        self.assertLessEqual(res.theoretical_bound, 0.01 * (1.0 + 1e-12))

    def test_nonuniform_grid_quadratic(self):
        xs = [0.0, 0.1, 0.35, 0.7, 1.0]
        xy = [(x, x * x) for x in xs]
        res = estimate_min_uniform_segments(xy=xy, epsilon=0.01, safety_factor=1.0)
        self.assertFalse(res.is_uniform_input_grid)
        self.assertAlmostEqual(res.m_raw, 2.0, places=12)
        self.assertEqual(res.n_segments, 5)
        self.assertTrue(res.check_passed)

    def test_linear_table_has_zero_curvature(self):
        xy = [(0.0, 1.0), (0.3, 1.6), (1.0, 3.0)]
        res = estimate_min_uniform_segments(xy=xy, epsilon=1e-6)
        self.assertAlmostEqual(res.m_raw, 0.0, places=12)
        self.assertEqual(res.n_segments, 1)
        self.assertAlmostEqual(res.theoretical_bound, 0.0, places=12)

    def test_detects_spike_warning(self):
        xy = [(-1.0, 1.0), (-0.5, 0.25), (-0.1, 0.01), (0.0, 0.0), (0.1, 0.3), (0.5, 0.55), (1.0, 0.8)]
        res = estimate_min_uniform_segments(xy=xy, epsilon=0.01, spike_ratio=3.0)
        self.assertGreaterEqual(len(res.spike_indices), 1)
        self.assertTrue(any("non-smooth" in w for w in res.warnings))

    def test_rejects_non_increasing_x(self):
        with self.assertRaises(ValueError):
            estimate_min_uniform_segments([(0.0, 0.0), (0.2, 0.1), (0.2, 0.5)], epsilon=1e-3)


if __name__ == "__main__":
    unittest.main()
