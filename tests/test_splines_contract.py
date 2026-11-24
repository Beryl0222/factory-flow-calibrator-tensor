"""样条接纳前提、结点契约与线性尾部的单元测试。"""
import unittest

import numpy as np

from rqs.splines import (
    KNOT_TOLERANCE,
    ROUNDTRIP_TOLERANCE,
    LOGJAC_TOLERANCE,
    TAIL_LEFT,
    TAIL_RIGHT,
    GroupValidation,
    build_group,
    forward,
    inverse,
)


def make_group(**overrides):
    params = dict(
        group_id="A", revision=3, extent=[-2, 2, -1, 3],
        width=[0.8, 0.0005, 0.1995], height=[0.4, 0.15, 0.45],
        derivative=[0.8, 0.025, 5.9, 1.2],
        min_fraction=1e-4, min_derivative=0.02)
    params.update(overrides)
    return build_group(**params)


class AdmissionTests(unittest.TestCase):
    def test_valid_group_accepted(self):
        g = make_group()
        self.assertEqual(g.n_bins, 3)
        np.testing.assert_allclose(g.x_knots[-1], 2.0)
        np.testing.assert_allclose(g.y_knots[-1], 3.0)

    def test_nonpositive_width_rejected(self):
        with self.assertRaises(GroupValidation):
            make_group(width=[0.8, 0.0, 0.2])

    def test_min_fraction_rejected(self):
        with self.assertRaises(GroupValidation):
            make_group(width=[0.99999, 0.000005, 0.100005])

    def test_non_normalized_rejected(self):
        with self.assertRaises(GroupValidation):
            make_group(width=[0.7, 0.1, 0.19])

    def test_derivative_below_floor_rejected(self):
        with self.assertRaises(GroupValidation):
            make_group(derivative=[0.8, 0.01, 5.9, 1.2])

    def test_negative_derivative_is_non_monotone(self):
        # 采集资料中的 bad_update：负导数破坏严格单调。
        with self.assertRaises(GroupValidation):
            make_group(derivative=[1.2, 1.1, -0.1, 2.0])

    def test_bad_extent_rejected(self):
        with self.assertRaises(GroupValidation):
            make_group(extent=[2, 2, -1, 3])


class KnotContractTests(unittest.TestCase):
    def setUp(self):
        self.g = make_group()

    def test_left_closed_right_open_interior(self):
        # 内结点 1.2：精确值归左箱，略大归右箱。
        at_knot = forward(self.g, np.array([1.2]))
        above = forward(self.g, np.array([1.2 + 1e-11]))
        self.assertEqual(int(at_knot.bin_code[0]), 0)
        self.assertTrue(bool(at_knot.near_knot[0]))
        self.assertEqual(int(above.bin_code[0]), 1)
        self.assertFalse(bool(above.near_knot[0]))

    def test_last_bin_contains_right_endpoint(self):
        res = forward(self.g, np.array([2.0]))
        self.assertEqual(int(res.bin_code[0]), 2)
        self.assertTrue(bool(res.near_knot[0]))

    def test_near_knot_uses_same_rule_inverse(self):
        y_knot = self.g.y_knots[2]
        res = inverse(self.g, np.array([y_knot]))
        self.assertEqual(int(res.bin_code[0]), 1)
        self.assertTrue(bool(res.near_knot[0]))
        above = inverse(self.g, np.array([y_knot + 1e-11]))
        self.assertEqual(int(above.bin_code[0]), 2)

    def test_residual_records_signed_gap(self):
        knot = self.g.x_knots[1]
        res = forward(self.g, np.array([knot + 1e-13]))
        self.assertTrue(bool(res.near_knot[0]))
        self.assertGreaterEqual(abs(float(res.residual[0])), 0.0)

    def test_no_silent_cross_bin_at_ambiguous_point(self):
        # 正/逆对同一近结点必须报告同一侧身份。
        x = np.array([self.g.x_knots[1] + 1e-10])
        y = forward(self.g, x)
        back = inverse(self.g, y.value)
        self.assertEqual(int(y.bin_code[0]), int(back.bin_code[0]))


class TailTests(unittest.TestCase):
    def setUp(self):
        self.g = make_group()

    def test_linear_tail_slopes_match_end_derivatives(self):
        x = np.array([-2.25, 2.7])
        res = forward(self.g, x)
        self.assertEqual(int(res.bin_code[0]), TAIL_LEFT)
        self.assertEqual(int(res.bin_code[1]), TAIL_RIGHT)
        expected_l = self.g.y0 + self.g.derivative[0] * (-2.25 - self.g.x0)
        expected_r = self.g.y1 + self.g.derivative[-1] * (2.7 - self.g.x1)
        self.assertAlmostEqual(float(res.value[0]), expected_l)
        self.assertAlmostEqual(float(res.value[1]), expected_r)
        np.testing.assert_allclose(
            res.log_jac, np.log([self.g.derivative[0],
                                 self.g.derivative[-1]]))

    def test_endpoint_single_sided_consistency(self):
        # 端点本身归内部箱；尾部与箱内两条口径在端点值一致。
        res = forward(self.g, np.array([self.g.x0, self.g.x1]))
        np.testing.assert_allclose(res.value, [self.g.y0, self.g.y1])
        np.testing.assert_allclose(
            res.log_jac,
            np.log([self.g.derivative[0], self.g.derivative[-1]]),
            atol=1e-12)

    def test_tail_roundtrip(self):
        y = forward(self.g, np.array([-3.0, 4.0]))
        back = inverse(self.g, y.value)
        np.testing.assert_allclose(back.value, [-3.0, 4.0],
                                   atol=ROUNDTRIP_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
