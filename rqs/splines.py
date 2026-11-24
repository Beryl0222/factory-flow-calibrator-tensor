"""单调有理二次样条（RQS）的正逆变换、结点判定与解析梯度。

契约（见项目 README）：

* 箱边左闭右开，最后一个箱包含右边界；
* 绝对往返误差 2e-9，对数雅可比误差 5e-8；
* 允许区间外线性尾部，斜率取对应端导数；
* 正宽、正高、导数与连续性保证是接纳前提；
* 宽度/高度已归一化，参数梯度沿和为一的切空间返回（分量和为零），
  端导数独立，量程端点固定。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- 项目判定契约 -------------------------------------------------------
ROUNDTRIP_TOLERANCE = 2e-9      # 绝对往返误差
LOGJAC_TOLERANCE = 5e-8         # 对数雅可比误差
GRADIENT_TOLERANCE = 2e-6       # 输入/参数梯度绝对容差
_EPS = np.finfo(np.float64).eps
# 当 float64 不能可靠区分结点两侧时，用坐标与往返误差复核的半径。
# 精确结点的浮点表示误差在 1e-15 量级；1e-11 这种可区分的近邻不得被吸走。
KNOT_TOLERANCE = ROUNDTRIP_TOLERANCE
# 同一数学结点的浮点表示误差量级（cumsum 累积约 2e-16）。
_REPRESENTATION_EPS = 8 * _EPS
KNOT_SNAP_EPS = 1e-12

# 分箱编码：[0, n_bins) 为内部箱，负数为尾部。
TAIL_LEFT = -1
TAIL_RIGHT = -2


class GroupValidation(ValueError):
    """分组参数不满足接纳前提。"""


@dataclass(frozen=True)
class RqsGroup:
    """一个参数组的一个不可变版本。"""

    group_id: str
    revision: int
    x0: float
    x1: float
    y0: float
    y1: float
    width: np.ndarray          # 归一化箱宽，和为 1
    height: np.ndarray         # 归一化箱高，和为 1
    derivative: np.ndarray     # 结点正导数（物理斜率，n+1 个）
    min_fraction: float
    min_derivative: float
    x_knots: np.ndarray = field(repr=False)
    y_knots: np.ndarray = field(repr=False)
    bin_width: np.ndarray = field(repr=False)
    bin_height: np.ndarray = field(repr=False)

    @property
    def n_bins(self) -> int:
        return int(self.width.shape[0])


def build_group(
    group_id: str,
    revision: int,
    extent,
    width,
    height,
    derivative,
    *,
    min_fraction: float,
    min_derivative: float,
) -> RqsGroup:
    """按接纳前提构造参数组；不满足时抛出 ``GroupValidation``。"""
    width = np.asarray(width, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    derivative = np.asarray(derivative, dtype=np.float64)
    x0, x1, y0, y1 = map(float, extent)

    problems = []
    if width.ndim != 1 or height.ndim != 1 or width.shape != height.shape:
        raise GroupValidation(f"{group_id} r{revision}: 宽高必须为等长一维数组")
    if derivative.shape != (width.shape[0] + 1,):
        raise GroupValidation(f"{group_id} r{revision}: 导数数应为箱数+1")
    if not np.isfinite([x0, x1, y0, y1]).all() or x1 <= x0 or y1 <= y0:
        problems.append("量程必须有限且严格递增")
    for name, vec, floor in (
        ("width", width, min_fraction),
        ("height", height, min_fraction),
    ):
        if not np.isfinite(vec).all():
            problems.append(f"{name} 含非有限值")
        elif np.any(vec <= 0.0):
            problems.append(f"{name} 必须严格为正")
        elif np.any(vec < floor):
            problems.append(f"{name} 存在小于最小占比 {floor:g} 的分量")
        elif not np.isclose(vec.sum(), 1.0, rtol=0.0, atol=1e-12):
            problems.append(f"{name} 必须归一化（和为 1），实际和 {vec.sum()!r}")
    if not np.isfinite(derivative).all() or np.any(derivative <= 0.0):
        problems.append("结点导数必须严格为正")
    elif np.any(derivative < min_derivative):
        problems.append(
            f"结点导数 {derivative.min()!r} 低于下界 {min_derivative:g}"
        )
    if problems:
        raise GroupValidation(
            f"{group_id} r{revision} 未通过接纳: " + "；".join(problems)
        )

    rx, ry = x1 - x0, y1 - y0
    bin_width = width * rx
    bin_height = height * ry
    x_knots = np.concatenate(([x0], x0 + np.cumsum(bin_width)))
    y_knots = np.concatenate(([y0], y0 + np.cumsum(bin_height)))
    # 浮点求和后强制端点精确，保证尾部与端点口径一致。
    x_knots[-1] = x1
    y_knots[-1] = y1

    return RqsGroup(
        group_id=group_id,
        revision=int(revision),
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        width=width.copy(),
        height=height.copy(),
        derivative=derivative.copy(),
        min_fraction=float(min_fraction),
        min_derivative=float(min_derivative),
        x_knots=x_knots,
        y_knots=y_knots,
        bin_width=bin_width,
        bin_height=bin_height,
    )


@dataclass
class PointResult:
    """单点（或批量点）变换结果。"""

    value: np.ndarray
    log_jac: np.ndarray
    bin_code: np.ndarray          # 内部箱下标，或 TAIL_LEFT / TAIL_RIGHT
    near_knot: np.ndarray         # 是否按结点契约贴边处理
    residual: np.ndarray          # 与最近判定结点的坐标差（复核用）
    u: np.ndarray                 # 箱内归一横坐标（尾部为线性外推比例）


def _locate(knots: np.ndarray, x: np.ndarray):
    """按左闭右开规则定位箱，并执行结点契约复核。

    返回 ``(bin_code, near_knot, residual, u)``。
    箱 k 覆盖 ``[knots[k], knots[k+1])``，最后一箱包含右端点。
    """
    x = np.asarray(x, dtype=np.float64)
    n_bins = knots.shape[0] - 1
    near = np.zeros(x.shape, dtype=bool)
    residual = np.full(x.shape, np.nan, dtype=np.float64)
    u = np.zeros(x.shape, dtype=np.float64)
    code = np.zeros(x.shape, dtype=np.int64)

    left_tail = x < knots[0]
    right_tail = x > knots[-1]
    code[left_tail] = TAIL_LEFT
    code[right_tail] = TAIL_RIGHT

    interior = ~(left_tail | right_tail)
    if interior.any():
        xi = x[interior]
        # 左闭右开：side='right' 给出严格小于 x 的结点数，减一即箱下标。
        placed = np.searchsorted(knots, xi, side="right") - 1
        # 仅严格等于结点（含同一数学量的浮点表示误差）时归左箱，
        # 可区分的近邻保留 searchsorted 的真实归属，保证正逆两侧一致。
        left_knot = knots[np.clip(placed, 0, n_bins - 1)]
        exact = np.abs(xi - left_knot) <= _REPRESENTATION_EPS
        placed = np.where(exact, np.maximum(placed - 1, 0), placed)
        kk = np.minimum(placed, n_bins - 1)
        code[interior] = kk

        xl = knots[kk]
        xr = knots[kk + 1]
        gap_l = np.abs(xi - xl)
        gap_r = np.abs(xi - xr)
        near_l = gap_l <= KNOT_SNAP_EPS
        near_r = (~near_l) & (gap_r <= KNOT_SNAP_EPS)
        near[interior] = near_l | near_r

        denom = xr - xl
        safe_denom = np.where(denom == 0.0, 1.0, denom)
        uu = np.clip((xi - xl) / safe_denom, 0.0, 1.0)
        is_last = kk == (n_bins - 1)
        # 左结点吸附 u=0；右结点仅最后一箱取 u=1，其余保留真实比例。
        uu = np.where(near_l, 0.0, uu)
        uu = np.where(near_r & is_last, 1.0, uu)
        u[interior] = uu
        residual[interior] = np.where(
            near_r, xi - xr, xi - xl)
    return code, near, residual, u


def _rqs_ratio(u, s, d0, d1):
    """README 定义的箱内比率 η(u)。"""
    v = 1.0 - u
    num = s * u * u + d0 * u * v
    den = s + (d1 + d0 - 2.0 * s) * u * v
    return num / den


def _bin_partials(u, w, h, d0, d1):
    """箱内全部一阶偏导，供正/逆向与解析 VJP 共用。

    物理量：箱宽 w、箱高 h、端导数 d0/d1、归一横坐标 u。
    箱内 y = yL + h·η(u,s,d0,d1)，s = h/w。
    """
    s = h / w
    v = 1.0 - u
    uv = u * v
    k = d1 + d0 - 2.0 * s
    q = s + k * uv
    eta = _rqs_ratio(u, s, d0, d1)

    # η = N/Q 的一阶商法则。N = s u² + d0 u(1-u)。
    n_u = 2.0 * s * u + d0 * (v - u)
    n_s = u * u
    n_d0 = uv
    n_d1 = 0.0
    q_u = k * (v - u)
    q_s = 1.0 - 2.0 * uv
    q_d0 = uv
    q_d1 = uv
    eta_u = (n_u - eta * q_u) / q
    eta_s = (n_s - eta * q_s) / q
    eta_d0 = (n_d0 - eta * q_d0) / q
    eta_d1 = (n_d1 - eta * q_d1) / q

    # η 对 u 的二阶导：η_u = F/Q，F = n_u - η·q_u。
    n_uu = 2.0 * s - 2.0 * d0
    q_uu = -2.0 * k
    f = n_u - eta * q_u
    f_u = n_uu - eta_u * q_u - eta * q_uu
    eta_uu = (f_u - eta_u * q_u) / q

    # 混合二阶导：∂F/∂θ = n_{uθ} - η_θ·q_u - η·q_{uθ}，
    # 再由 η_{uθ} = (F_θ·Q - F·Q_θ)/Q²。
    n_us = 2.0 * u
    q_us = -2.0 * (v - u)
    f_s = n_us - eta_s * q_u - eta * q_us
    eta_us = (f_s * q - f * q_s) / (q * q)

    n_ud = v - u
    q_ud = v - u
    f_d0 = n_ud - eta_d0 * q_u - eta * q_ud
    eta_ud0 = (f_d0 * q - f * q_d0) / (q * q)

    f_d1 = -eta_d1 * q_u - eta * q_ud
    eta_ud1 = (f_d1 * q - f * q_d1) / (q * q)

    # dy/dx = h·η_u/w = s·η_u；ell = log(dy/dx) = log(s)+log(η_u)。
    jac = s * eta_u
    ell = np.log(jac)
    ell_u = eta_uu / eta_u
    ell_s = 1.0 / s + eta_us / eta_u
    ell_d0 = eta_ud0 / eta_u
    ell_d1 = eta_ud1 / eta_u

    # 物理坐标偏导：u=(x-xL)/w, s=h/w, y = yL + h·η。
    # 对 w（固定 h）：du/dw=-u/w, ds/dw=-s/w；
    # 对 h（固定 w）：du/dh=0, ds/dh=1/w，且 dy/dh 含显式 η 项。
    dy_dx = s * eta_u
    dy_dw = -h * (u * eta_u + s * eta_s) / w
    dy_dh = eta + s * eta_s
    dy_dd0 = h * eta_d0
    dy_dd1 = h * eta_d1

    # ell = log(s·η_u)：dell/dx = ell_u/w；
    # 对 w 同时含 ell_u（经 u）与 ell_s（经 s）两条路径。
    dl_dx = ell_u / w
    dl_dw = -(u * ell_u + s * ell_s) / w
    dl_dh = ell_s / w
    dl_dd0 = ell_d0
    dl_dd1 = ell_d1

    return {
        "s": s,
        "eta": eta,
        "jac": jac,
        "ell": ell,
        "dy_dx": dy_dx,
        "dy_dw": dy_dw,
        "dy_dh": dy_dh,
        "dy_dd0": dy_dd0,
        "dy_dd1": dy_dd1,
        "dl_dx": dl_dx,
        "dl_dw": dl_dw,
        "dl_dh": dl_dh,
        "dl_dd0": dl_dd0,
        "dl_dd1": dl_dd1,
    }


def forward(group: RqsGroup, x) -> PointResult:
    """正向变换 x -> y，附对数雅可比与分箱归属。"""
    x = np.asarray(x, dtype=np.float64)
    code, near, residual, u = _locate(group.x_knots, x)
    y = np.empty_like(x)
    log_jac = np.empty_like(x)

    # 线性尾部：斜率取对应端导数，与端点值相连。
    left = code == TAIL_LEFT
    right = code == TAIL_RIGHT
    y[left] = group.y0 + group.derivative[0] * (x[left] - group.x0)
    log_jac[left] = np.log(group.derivative[0])
    y[right] = group.y1 + group.derivative[-1] * (x[right] - group.x1)
    log_jac[right] = np.log(group.derivative[-1])

    interior = ~(left | right)
    if interior.any():
        k = code[interior]
        uu = u[interior]
        w = group.bin_width[k]
        h = group.bin_height[k]
        d0 = group.derivative[k]
        d1 = group.derivative[k + 1]
        part = _bin_partials(uu, w, h, d0, d1)
        xl = group.x_knots[k]
        yl = group.y_knots[k]
        y[interior] = yl + h * part["eta"]
        log_jac[interior] = part["ell"]
    return PointResult(value=y, log_jac=log_jac, bin_code=code,
                       near_knot=near, residual=residual, u=u)


def _solve_inverse_u(r, s, d0, d1):
    """由箱内比率 r 反解归一横坐标 u（稳定二次公式）。"""
    # r=0/1 对应精确结点，直接给精确值，避免求根噪声。
    out = np.empty_like(r)
    exact0 = r <= 0.0
    exact1 = r >= 1.0
    out[exact0] = 0.0
    out[exact1] = 1.0
    todo = ~(exact0 | exact1)
    if todo.any():
        rr = r[todo]
        ss, a0, a1 = s[todo], d0[todo], d1[todo]
        # eta(u)=r 的二次式（N = s u² + d0 u(1-u)，Q = s + k·u(1-u)，
        # k = d1+d0-2s；展开 N-rQ=0）：
        # ((s-d0)+rk)u² + (d0-rk)u - rs = 0
        k = a1 + a0 - 2.0 * ss
        a = (ss - a0) + rr * k
        b = a0 - rr * k
        c = -rr * ss
        disc = b * b - 4.0 * a * c
        disc = np.maximum(disc, 0.0)
        root = np.sqrt(disc)
        # 稳定求根（Numerical Recipes），再以 [0,1] 归属复核。
        q = -0.5 * (b + np.sign(b) * root)
        u_via_q = np.where(np.abs(a) > np.abs(q), q / a, c / q)
        u_other = np.where(np.abs(a) > np.abs(q), c / q, q / a)
        u = np.where((u_via_q >= 0.0) & (u_via_q <= 1.0), u_via_q,
                     u_other)
        # 退化（a≈0，线性/二次退化）兜底。
        flat = ~np.isfinite(u) | (u < 0.0) | (u > 1.0)
        if flat.any():
            u[flat] = -b[flat] / (2.0 * a[flat])
        out[todo] = np.minimum(np.maximum(u, 0.0), 1.0)
    return out


def inverse(group: RqsGroup, y) -> PointResult:
    """逆向变换 y -> x，附 log(dx/dy) 与同一套分箱归属规则。"""
    y = np.asarray(y, dtype=np.float64)
    code, near, residual, u_y = _locate(group.y_knots, y)
    x = np.empty_like(y)
    log_jac = np.empty_like(y)
    u = np.zeros_like(y)

    left = code == TAIL_LEFT
    right = code == TAIL_RIGHT
    x[left] = group.x0 + (y[left] - group.y0) / group.derivative[0]
    log_jac[left] = -np.log(group.derivative[0])
    x[right] = group.x1 + (y[right] - group.y1) / group.derivative[-1]
    log_jac[right] = -np.log(group.derivative[-1])

    interior = ~(left | right)
    if interior.any():
        k = code[interior]
        yl = group.y_knots[k]
        h = group.bin_height[k]
        w = group.bin_width[k]
        d0 = group.derivative[k]
        d1 = group.derivative[k + 1]
        s = h / w
        r = (y[interior] - yl) / h
        uu = _solve_inverse_u(r, np.broadcast_to(s, r.shape).copy(),
                              d0, d1)
        part = _bin_partials(uu, w, h, d0, d1)
        xl = group.x_knots[k]
        x[interior] = xl + w * uu
        log_jac[interior] = -part["ell"]
        u[interior] = uu
    return PointResult(value=x, log_jac=log_jac, bin_code=code,
                       near_knot=near, residual=residual, u=u)


@dataclass
class ParamGradient:
    """一个参数组在一批点上累积的梯度（切空间口径）。"""

    width: np.ndarray
    height: np.ndarray
    derivative: np.ndarray
    used_bins: np.ndarray
    n_points: int = 0


def _empty_param_gradient(group: RqsGroup) -> ParamGradient:
    return ParamGradient(
        width=np.zeros(group.n_bins),
        height=np.zeros(group.n_bins),
        derivative=np.zeros(group.n_bins + 1),
        used_bins=np.zeros(group.n_bins, dtype=np.int64),
    )


def _project_tangent(grad: np.ndarray) -> np.ndarray:
    """投影到和为一的切空间：返回分量和为零的代表。"""
    return grad - grad.sum() / grad.shape[0]


def forward_vjp(group: RqsGroup, x, g_y, g_logjac):
    """正向反向模式：给定 (g_y, g_logjac) 回传输入与组参数梯度。

    宽度/高度梯度先按物理箱宽/箱高累积，再换算为归一化分量并投影到
    和为零的切空间；端导数独立；量程端点固定（无梯度）。
    """
    x = np.asarray(x, dtype=np.float64)
    g_y = np.broadcast_to(np.asarray(g_y, dtype=np.float64), x.shape).copy()
    g_logjac = np.broadcast_to(
        np.asarray(g_logjac, dtype=np.float64), x.shape
    ).copy()

    g_x = np.zeros_like(x)
    pg = _empty_param_gradient(group)
    result = forward(group, x)

    # 尾部：线性 y = yEnd + D*(x-xEnd)
    for tail_code, d_idx in ((TAIL_LEFT, 0), (TAIL_RIGHT, -1)):
        sel = result.bin_code == tail_code
        if not sel.any():
            continue
        d = group.derivative[d_idx]
        g_x[sel] = g_y[sel] * d
        # d ell / d D = 1/D；端点固定，x0/x1/y0/y1 不回传
        np.add.at(pg.derivative, d_idx,
                  np.sum(g_y[sel] * (x[sel] - (group.x0 if d_idx == 0
                                               else group.x1)))
                  + np.sum(g_logjac[sel]) / d)

    interior = result.bin_code >= 0
    if interior.any():
        k = result.bin_code[interior]
        uu = result.u[interior]
        w = group.bin_width[k]
        h = group.bin_height[k]
        d0 = group.derivative[k]
        d1 = group.derivative[k + 1]
        part = _bin_partials(uu, w, h, d0, d1)
        gy, gl = g_y[interior], g_logjac[interior]

        g_x[interior] = gy * part["dy_dx"] + gl * part["dl_dx"]

        g_w = gy * part["dy_dw"] + gl * part["dl_dw"]
        g_h = gy * part["dy_dh"] + gl * part["dl_dh"]
        g_d0 = gy * part["dy_dd0"] + gl * part["dl_dd0"]
        g_d1 = gy * part["dy_dd1"] + gl * part["dl_dd1"]
        np.add.at(pg.derivative, k, g_d0)
        np.add.at(pg.derivative, k + 1, g_d1)

        # 归一化 W_j 同时移动本箱宽（j=k）与后续所有箱的左结点：
        # xL_k = x0 + Rx·Σ_{j<k} W_j，且 yL_k 同理；端点固定。
        rx, ry = group.x1 - group.x0, group.y1 - group.y0
        g_xL = -(gy * part["dy_dx"] + gl * part["dl_dx"])
        g_yL = gy
        for row, kk in enumerate(k):
            pg.width[kk] += rx * g_w[row]
            pg.width[:kk] += rx * g_xL[row]
            pg.height[kk] += ry * g_h[row]
            pg.height[:kk] += ry * g_yL[row]
        np.add.at(pg.used_bins, k, 1)

    pg.width[:] = _project_tangent(pg.width)
    pg.height[:] = _project_tangent(pg.height)
    pg.n_points = int(x.size)
    return g_x, pg, result


def inverse_vjp(group: RqsGroup, y, g_x, g_logjac):
    """逆向反向模式：给定对 (x, log|dx/dy|) 的梯度回传。

    逆向内部在反解出的 u 点复用同一组偏导；隐函数定理保证
    dx*/dy = 1/(dy/dx)、d ell_inv/dy = -ell_u/(w*J) 等关系。
    """
    y = np.asarray(y, dtype=np.float64)
    g_x = np.broadcast_to(np.asarray(g_x, dtype=np.float64), y.shape).copy()
    g_logjac = np.broadcast_to(
        np.asarray(g_logjac, dtype=np.float64), y.shape
    ).copy()

    g_y = np.zeros_like(y)
    pg = _empty_param_gradient(group)
    result = inverse(group, y)

    for tail_code, d_idx in ((TAIL_LEFT, 0), (TAIL_RIGHT, -1)):
        sel = result.bin_code == tail_code
        if not sel.any():
            continue
        d = group.derivative[d_idx]
        # x = xEnd + (y-yEnd)/D；ell_inv = -log D
        g_y[sel] = g_x[sel] / d
        np.add.at(pg.derivative, d_idx,
                  np.sum(-g_x[sel] * (y[sel] - (group.y0 if d_idx == 0
                                                else group.y1)) / (d * d))
                  + np.sum(-g_logjac[sel]) / d)

    interior = result.bin_code >= 0
    if interior.any():
        k = result.bin_code[interior]
        uu = result.u[interior]
        w = group.bin_width[k]
        h = group.bin_height[k]
        d0 = group.derivative[k]
        d1 = group.derivative[k + 1]
        part = _bin_partials(uu, w, h, d0, d1)
        jac = part["jac"]
        gx, gl = g_x[interior], g_logjac[interior]

        # 值路径：y = f(x*; θ), 0 = gx·f_x + 隐式项
        g_y[interior] = gx / jac
        # 对数雅可比路径 ell_inv = -ell(x*; θ)
        g_y[interior] += gl * (-part["dl_dx"] / jac)

        # 隐函数定理：y = f(x*; θ)，ell_inv(y;θ) = -ell(x*;θ)。
        # dx*/dθ = -f_θ/f_x；d ell_inv/dθ = -(ell_θ - ell_x·f_θ/f_x)。
        def implicit(f_theta, ell_theta, ell_x):
            return (
                gx * (-f_theta / jac)
                + gl * (-(ell_theta - ell_x * f_theta / jac))
            )

        g_w = implicit(part["dy_dw"], part["dl_dw"], part["dl_dx"])
        g_h = implicit(part["dy_dh"], part["dl_dh"], part["dl_dx"])
        g_d0 = implicit(part["dy_dd0"], part["dl_dd0"], part["dl_dx"])
        g_d1 = implicit(part["dy_dd1"], part["dl_dd1"], part["dl_dx"])
        np.add.at(pg.derivative, k, g_d0)
        np.add.at(pg.derivative, k + 1, g_d1)

        # 归一化分量 W_j：本箱宽 w_k=Rx·W_k，且 xL_k=x0+Rx·Σ_{j<k}W_j。
        # xL 平移时 u 不变，逆 logjac 无贡献；yL 经隐函数同时影响 x* 与
        # 逆 logjac（dx*/dyL=-1/J，d ell_inv/dyL=ell_x/J）。
        rx, ry = group.x1 - group.x0, group.y1 - group.y0
        g_xL = gx * 1.0
        g_yL = gx * (-1.0 / jac) + gl * (part["dl_dx"] / jac)
        for row, kk in enumerate(k):
            pg.width[kk] += rx * g_w[row]
            pg.width[:kk] += rx * g_xL[row]
            pg.height[kk] += ry * g_h[row]
            pg.height[:kk] += ry * g_yL[row]
        np.add.at(pg.used_bins, k, 1)

    pg.width[:] = _project_tangent(pg.width)
    pg.height[:] = _project_tangent(pg.height)
    pg.n_points = int(y.size)
    return g_y, pg, result


def merge_param_gradients(gradients) -> ParamGradient | None:
    """把同一组身份在不同位置/行累积的梯度合并。"""
    gradients = [g for g in gradients if g is not None]
    if not gradients:
        return None
    base = ParamGradient(
        width=np.sum([g.width for g in gradients], axis=0),
        height=np.sum([g.height for g in gradients], axis=0),
        derivative=np.sum([g.derivative for g in gradients], axis=0),
        used_bins=np.sum([g.used_bins for g in gradients], axis=0),
        n_points=sum(g.n_points for g in gradients),
    )
    # 各分量本身已是切空间代表，求和后仍在切空间内。
    return base
