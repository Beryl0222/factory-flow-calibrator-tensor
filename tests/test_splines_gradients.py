"""往返误差、对数雅可比一致性与解析梯度的有限差分核对。"""
import unittest

import numpy as np

from rqs.splines import (
    GRADIENT_TOLERANCE,
    LOGJAC_TOLERANCE,
    ROUNDTRIP_TOLERANCE,
    build_group,
    forward,
    forward_vjp,
    inverse,
    inverse_vjp,
)

GROUPS = {
    "A": dict(extent=[-2, 2, -1, 3], width=[0.8, 0.0005, 0.1995],
              height=[0.4, 0.15, 0.45], derivative=[0.8, 0.025, 5.9, 1.2],
              min_fraction=1e-4, min_derivative=0.02),
    "B": dict(extent=[0, 10, 0, 12], width=[0.2, 0.7, 0.1],
              height=[0.2, 0.1, 0.7], derivative=[1.2, 1.1, 0.1, 2.0],
              min_fraction=4e-4, min_derivative=0.06),
}


def build(name, rev=1):
    return build_group(name, rev, **GROUPS[name])


def rebuild_relaxed(group, width=None, height=None, derivative=None):
    return build_group(
        group.group_id, group.revision,
        [group.x0, group.x1, group.y0, group.y1],
        group.width if width is None else width,
        group.height if height is None else height,
        group.derivative if derivative is None else derivative,
        min_fraction=1e-12, min_derivative=1e-12)


class RoundTripTests(unittest.TestCase):
    def test_forward_inverse_roundtrip_all_groups(self):
        rng = np.random.default_rng(7)
        for name in GROUPS:
            g = build(name)
            xs = rng.uniform(g.x0, g.x1, 500)
            # 近乎贴边的有效分箱点。
            xs = np.concatenate([
                xs, g.x_knots, g.x_knots + 1e-11, g.x_knots - 1e-11,
                np.array([g.x0 - 0.3, g.x1 + 0.3])])
            y = forward(g, xs)
            back = inverse(g, y.value)
            np.testing.assert_allclose(back.value, xs,
                                       atol=ROUNDTRIP_TOLERANCE)
            np.testing.assert_allclose(back.log_jac, -y.log_jac,
                                       atol=LOGJAC_TOLERANCE)

    def test_inverse_forward_roundtrip(self):
        rng = np.random.default_rng(11)
        for name in GROUPS:
            g = build(name)
            ys = rng.uniform(g.y0, g.y1, 500)
            ys = np.concatenate([ys, g.y_knots])
            x = inverse(g, ys)
            back = forward(g, x.value)
            np.testing.assert_allclose(back.value, ys,
                                       atol=ROUNDTRIP_TOLERANCE)
            np.testing.assert_allclose(back.log_jac, -x.log_jac,
                                       atol=LOGJAC_TOLERANCE)


class GradientTests(unittest.TestCase):
    EPS = 1e-7

    def _fd_param(self, group, z0, gz, gl, mode):
        scalar = (lambda grp, x: (lambda r: (r.value[0], r.log_jac[0]))(
            forward(grp, np.array([x])))) if mode == "f" else \
            (lambda grp, y: (lambda r: (r.value[0], r.log_jac[0]))(
                inverse(grp, np.array([y]))))
        gw, gh = np.zeros(group.n_bins), np.zeros(group.n_bins)
        gd = np.zeros(group.n_bins + 1)
        e = self.EPS
        for k in range(group.n_bins):
            d = np.zeros(group.n_bins)
            d[k] = 1
            d -= 1 / group.n_bins
            yp, lp = scalar(rebuild_relaxed(
                group, width=group.width + e * d), z0)
            ym, lm = scalar(rebuild_relaxed(
                group, width=group.width - e * d), z0)
            gw[k] = gz * (yp - ym) / (2 * e) + gl * (lp - lm) / (2 * e)
            yp, lp = scalar(rebuild_relaxed(
                group, height=group.height + e * d), z0)
            ym, lm = scalar(rebuild_relaxed(
                group, height=group.height - e * d), z0)
            gh[k] = gz * (yp - ym) / (2 * e) + gl * (lp - lm) / (2 * e)
        for k in range(group.n_bins + 1):
            dp = group.derivative.copy()
            dp[k] += e
            dm = group.derivative.copy()
            dm[k] -= e
            yp, lp = scalar(rebuild_relaxed(group, derivative=dp), z0)
            ym, lm = scalar(rebuild_relaxed(group, derivative=dm), z0)
            gd[k] = gz * (yp - ym) / (2 * e) + gl * (lp - lm) / (2 * e)
        return gw, gh, gd

    def test_forward_vjp_matches_fd(self):
        for name in GROUPS:
            g = build(name)
            x0 = 0.6 * g.x0 + 0.4 * g.x1
            gz, gl = 1.7, -0.6
            g_x, pg, _ = forward_vjp(g, np.array([x0]),
                                     np.array([gz]), np.array([gl]))
            e = self.EPS
            yp = forward(g, np.array([x0 + e]))
            ym = forward(g, np.array([x0 - e]))
            fd_gx = gz * (yp.value[0] - ym.value[0]) / (2 * e) \
                + gl * (yp.log_jac[0] - ym.log_jac[0]) / (2 * e)
            self.assertAlmostEqual(float(g_x[0]), fd_gx,
                                   delta=GRADIENT_TOLERANCE)
            gw, gh, gd = self._fd_param(g, x0, gz, gl, "f")
            np.testing.assert_allclose(pg.width, gw,
                                       atol=GRADIENT_TOLERANCE)
            np.testing.assert_allclose(pg.height, gh,
                                       atol=GRADIENT_TOLERANCE)
            np.testing.assert_allclose(pg.derivative, gd,
                                       atol=GRADIENT_TOLERANCE)
            self.assertAlmostEqual(float(pg.width.sum()), 0.0, places=12)
            self.assertAlmostEqual(float(pg.height.sum()), 0.0, places=12)

    def test_inverse_vjp_matches_fd(self):
        for name in GROUPS:
            g = build(name)
            y0 = 0.55 * g.y0 + 0.45 * g.y1
            gz, gl = 2.3, 0.4
            g_y, pg, _ = inverse_vjp(g, np.array([y0]),
                                     np.array([gz]), np.array([gl]))
            e = self.EPS
            yp = inverse(g, np.array([y0 + e]))
            ym = inverse(g, np.array([y0 - e]))
            fd_gy = gz * (yp.value[0] - ym.value[0]) / (2 * e) \
                + gl * (yp.log_jac[0] - ym.log_jac[0]) / (2 * e)
            self.assertAlmostEqual(float(g_y[0]), fd_gy,
                                   delta=GRADIENT_TOLERANCE)
            gw, gh, gd = self._fd_param(g, y0, gz, gl, "i")
            np.testing.assert_allclose(pg.width, gw,
                                       atol=GRADIENT_TOLERANCE)
            np.testing.assert_allclose(pg.height, gh,
                                       atol=GRADIENT_TOLERANCE)
            np.testing.assert_allclose(pg.derivative, gd,
                                       atol=GRADIENT_TOLERANCE)

    def test_tail_vjp_matches_fd(self):
        g = build("A")
        for x0, d_idx, end in ((-2.25, 0, g.x0), (2.7, -1, g.x1)):
            gz, gl = 1.1, 0.7
            g_x, pg, _ = forward_vjp(g, np.array([x0]),
                                     np.array([gz]), np.array([gl]))
            e = self.EPS
            yp = forward(g, np.array([x0 + e]))
            ym = forward(g, np.array([x0 - e]))
            fd = gz * (yp.value[0] - ym.value[0]) / (2 * e) \
                + gl * (yp.log_jac[0] - ym.log_jac[0]) / (2 * e)
            self.assertAlmostEqual(float(g_x[0]), fd,
                                   delta=GRADIENT_TOLERANCE)
            d = g.derivative.copy()
            dp, dm = d.copy(), d.copy()
            dp[d_idx] += e
            dm[d_idx] -= e
            gp = rebuild_relaxed(g, derivative=dp)
            gm = rebuild_relaxed(g, derivative=dm)
            yp = forward(gp, np.array([x0]))
            ym = forward(gm, np.array([x0]))
            fd_d = gz * (yp.value[0] - ym.value[0]) / (2 * e) \
                + gl * (yp.log_jac[0] - ym.log_jac[0]) / (2 * e)
            self.assertAlmostEqual(float(pg.derivative[d_idx]), fd_d,
                                   delta=GRADIENT_TOLERANCE)

    def test_shared_group_gradients_merge(self):
        g = build("B")
        xs = np.array([1.0, 5.0, 9.0])
        g_x, pg, _ = forward_vjp(g, xs, np.ones(3), np.full(3, 0.5))
        # 同一组身份在多个位置复用：used_bins 总数等于点数。
        self.assertEqual(int(pg.used_bins.sum()), 3)
        self.assertAlmostEqual(float(pg.width.sum()), 0.0, places=12)
        self.assertAlmostEqual(float(pg.height.sum()), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
