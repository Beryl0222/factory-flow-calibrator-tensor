"""多级校准链：固定通道置换 + 分组 RQS，带前向承诺收据与反向回放。

关键约定：

* 每级先按 ``新通道 j 读旧通道 perm[j]`` 的方式置换，再用该级配置的
  参数组逐通道处理；逆向沿反方向恢复；
* 掩码随通道身份一起移动，无观测值原样绕过各级且不贡献密度；
* 总对数雅可比只按有效通道及实际走过的各级合成；
* 反向必须匹配同一组前向承诺（组身份、版本、分箱编码逐点复核），
  不匹配时抛出 ``CommitmentMismatch``，绝不静默改用另一侧密度；
* 无效组只落在绕过位的行仍可评分，命中观测位的行给不可用原因。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .receipts import ReceiptStore
from .registry import (
    GroupRegistry,
    InvalidRevision,
)
from .splines import (
    GroupValidation,
    forward,
    inverse,
    forward_vjp,
    inverse_vjp,
    merge_param_gradients,
)


class CommitmentMismatch(RuntimeError):
    """回放与前向承诺的组版本或分箱归属不一致。"""


@dataclass(frozen=True)
class StageSpec:
    perm: tuple[int, ...]
    groups: tuple[str, ...]


@dataclass(frozen=True)
class ChainSpec:
    chain_id: str
    stages: tuple[StageSpec, ...]

    @property
    def n_channels(self) -> int:
        return len(self.stages[0].perm)


@dataclass
class UnavailableReason:
    request_id: str
    group_id: str
    revision: int
    status: str
    detail: str


@dataclass
class RowResult:
    request_id: str
    chain_id: str
    direction: str
    values: np.ndarray
    log_jac: float
    mask: np.ndarray
    identity: np.ndarray
    available: bool
    reason: str = ""


@dataclass
class BatchResult:
    batch_index: int
    rows: list[RowResult]
    unavailable: list[UnavailableReason]
    versions: dict
    receipts: ReceiptStore
    shard_rows: int

    @property
    def scoreable(self) -> list[RowResult]:
        return [r for r in self.rows if r.available]


@dataclass
class BackpropResult:
    batch_index: int
    input_grads: dict                  # request_id -> 输入通道梯度
    logjac_grads: dict                 # request_id -> 已消费的总 logjac 梯度
    param_grads: dict                  # (group_id, rev) -> ParamGradient
    replayed: bool = False


def estimate_shard_scratch(n_rows: int, n_channels: int,
                           n_stages: int) -> int:
    """估算单分片工作区字节（解包/转换/归约/暂存，只读输入不计）。"""
    floats = n_rows * n_channels
    stage_pts = n_rows * n_channels * n_stages
    # 两块值数组 + 掩码/身份 + 逐阶段痕迹（箱编码/标志/残差/版本/身份）
    return int(
        2 * floats * 8
        + 2 * floats * 1
        + stage_pts * (2 + 1 + 8 + 4 + 2)
        + floats * 8
    )


class CalibrationService:
    def __init__(self, registry: GroupRegistry, chains: dict[str, ChainSpec],
                 *, partition_rows: int = 3500,
                 peak_scratch_bytes: int = 5 * 1024 * 1024,
                 max_receipt_bytes: int = 4 * 1024 * 1024,
                 receipt_resident_bytes: int | None = None,
                 spill_dir=None):
        self.registry = registry
        self.chains = chains
        self.partition_rows = int(partition_rows)
        self.peak_scratch_bytes = int(peak_scratch_bytes)
        self.max_receipt_bytes = int(max_receipt_bytes)
        # 收据驻留上限默认与单分片暂存预算对齐，超出即外溢。
        self.receipt_resident_bytes = (
            int(receipt_resident_bytes) if receipt_resident_bytes is not None
            else int(peak_scratch_bytes))
        self.spill_dir = spill_dir
        self._batch_cache: dict[int, BatchResult] = {}

    # ------------------------------------------------------------------
    # 可用性规划
    # ------------------------------------------------------------------
    def plan_row(self, chain: ChainSpec, mask: np.ndarray,
                 versions: dict) -> tuple[bool, list[UnavailableReason],
                                          dict]:
        """沿实际置换确定该行会触达的 (组, 版本)，判定行级可用性。"""
        identity = np.arange(chain.n_channels)
        problems: list[UnavailableReason] = []
        needed: dict = {}
        for stage in chain.stages:
            observed = mask[np.asarray(stage.perm)]
            for pos, gid in enumerate(stage.groups):
                rev = int(versions[gid])
                key = (gid, rev)
                if not bool(observed[pos]):
                    continue                      # 绕过位不触达
                needed[key] = needed.get(key, 0) + 1
                status = self.registry.status(gid, rev)
                if status != "ok" and not any(
                        p.group_id == gid and p.revision == rev
                        for p in problems):
                    rec = self.registry.rejection(gid, rev)
                    detail = rec.reason if rec is not None else \
                        "注册表中无法确认该版本参数"
                    problems.append(UnavailableReason(
                        request_id="", group_id=gid, revision=rev,
                        status=status, detail=detail))
        return not problems, problems, needed

    # ------------------------------------------------------------------
    # 分批处理
    # ------------------------------------------------------------------
    def run_batch(self, batch_index: int, rows: list,
                  versions: dict) -> BatchResult:
        """rows: chain_rows 记录（id, chain_id, direction, values, mask）。"""
        receipts = ReceiptStore(
            resident_bytes=self.receipt_resident_bytes,
            spill_dir=self.spill_dir)
        row_results: list[RowResult] = []
        unavailable: list[UnavailableReason] = []

        chain_groups: dict[tuple, list] = {}
        for row in rows:
            chain_groups.setdefault((row[1], row[2]), []).append(row)

        shard_size = self._choose_shard_size(rows)
        for (chain_id, direction), grouped in sorted(chain_groups.items()):
            chain = self.chains[chain_id]
            for start in range(0, len(grouped), shard_size):
                block = grouped[start:start + shard_size]
                self._run_block(chain, block, direction, versions, receipts,
                                row_results, unavailable)
        result = BatchResult(
            batch_index=batch_index, rows=row_results,
            unavailable=unavailable, versions=dict(versions),
            receipts=receipts, shard_rows=shard_size)
        self._batch_cache[batch_index] = result
        return result

    def _choose_shard_size(self, rows: list) -> int:
        max_stages = max(len(self.chains[r[1]].stages) for r in rows)
        n_channels = len(rows[0][3])
        shard = self.partition_rows
        while shard > 1 and estimate_shard_scratch(
                shard, n_channels, max_stages) > self.peak_scratch_bytes:
            shard //= 2
        return min(shard, len(rows))

    def _run_block(self, chain: ChainSpec, block, direction, versions,
                   receipts: ReceiptStore, row_results: list,
                   unavailable: list) -> None:
        rids = [r[0] for r in block]
        assert {r[2] for r in block} == {direction}
        R, C, S = len(block), chain.n_channels, len(chain.stages)

        values_in = np.asarray([r[3] for r in block], dtype=np.float64)
        mask_in = np.asarray([r[4] for r in block], dtype=np.bool_)
        availability = []
        for rid, _, _, _, mask in block:
            ok, problems, _ = self.plan_row(chain, np.asarray(mask, bool),
                                            versions)
            availability.append(ok)
            for p in problems:
                unavailable.append(UnavailableReason(
                    request_id=rid, group_id=p.group_id,
                    revision=p.revision, status=p.status, detail=p.detail))

        codes = np.zeros((R, S, C), dtype=np.int64)
        near = np.zeros((R, S, C), dtype=bool)
        residual = np.full((R, S, C), np.nan, dtype=np.float64)
        rev_arr = np.full((R, S, C), -1, dtype=np.int32)
        id_arr = np.zeros((R, S, C), dtype=np.int16)
        observed_arr = np.zeros((R, S, C), dtype=bool)
        stage_groups = [list(st.groups) for st in chain.stages]

        reasons = np.asarray(["" for _ in range(R)], dtype=object)
        unavail_flag = np.asarray([not ok for ok in availability])

        local_sel = np.ones(R, dtype=bool)
        global_idx = np.arange(R)
        if direction == "forward":
            values_out, mask_out, identity_out, total_lj = \
                self._forward_block(
                    chain, values_in, mask_in, versions,
                    codes, near, residual, rev_arr, id_arr,
                    observed_arr, local_sel, local_sel, global_idx)
        else:
            values_out, mask_out, identity_out, total_lj = \
                self._inverse_block(
                    chain, values_in, mask_in, versions,
                    codes, near, residual, rev_arr, id_arr,
                    observed_arr, local_sel, local_sel, global_idx)
        for i, ok in enumerate(availability):
            if not ok:
                hit = [u for u in unavailable if u.request_id == rids[i]]
                reasons[i] = "; ".join(
                    f"组 {u.group_id} r{u.revision} {u.status}: {u.detail}"
                    for u in hit)

        receipts.add_shard(
            request_ids=rids, chain_id=chain.chain_id,
            direction=direction,
            values_in=values_in, values_out=values_out, log_jac=total_lj,
            mask_in=mask_in, mask_out=mask_out, identity_out=identity_out,
            stage_codes=codes, stage_near=near, stage_residual=residual,
            stage_versions=rev_arr, stage_groups=stage_groups,
            stage_identity=id_arr, observed=observed_arr,
            unavailable=unavail_flag, reasons=reasons)

        for i, rid in enumerate(rids):
            row_results.append(RowResult(
                request_id=rid, chain_id=chain.chain_id,
                direction=block[i][2], values=values_out[i],
                log_jac=float(total_lj[i]), mask=mask_out[i],
                identity=identity_out[i], available=availability[i],
                reason=str(reasons[i])))

    def _group_at(self, gid: str, versions: dict):
        rev = int(versions[gid])
        return self.registry.get(gid, rev), rev

    def _forward_block(self, chain, values, mask, versions,
                       codes, near, residual, rev_arr, id_arr,
                       observed_arr, _row_sel, local_sel, global_idx):
        R, C = values.shape
        # 无观测位置统一以 NaN 表示“绕过、无输出”，且全程不参与密度。
        v = values.copy()
        v[~mask] = np.nan
        m = mask.copy()
        ident = np.broadcast_to(np.arange(C), (R, C)).copy()
        total = np.zeros(R)
        local_rows = np.where(local_sel)[0]
        for s, stage in enumerate(chain.stages):
            perm = np.asarray(stage.perm)
            v2, m2, id2 = v[:, perm], m[:, perm], ident[:, perm]
            nv = v2.copy()
            for gid in dict.fromkeys(stage.groups):
                positions = [j for j, g in enumerate(stage.groups)
                             if g == gid]
                # 先判断该组在本批是否只落在绕过位；若是则不实例化参数，
                # 无效/未知版本也不影响评分。
                any_observed = any(m2[:, j].any() for j in positions)
                rev = int(versions[gid])
                # 无效/未知版本：不可用行已在 plan_row 标记，这里不实例化，
                # 观测位保持 NaN，绝不能用较早级中间值冒充实整条链输出。
                group = None
                if any_observed and self.registry.status(gid, rev) == "ok":
                    group = self.registry.get(gid, rev)
                for j in positions:
                    obs = m2[:, j]
                    rev_arr[global_idx[local_rows], s, j] = \
                        np.where(obs, rev, -1)
                    id_arr[global_idx[local_rows], s, j] = id2[:, j]
                    observed_arr[global_idx[local_rows], s, j] = obs
                    if not obs.any():
                        continue
                    if group is None:
                        # 无效/未知组：该观测位无法变换，置 NaN 并留痕，
                        # 不把较早级中间值当作整条链输出。
                        nv[np.where(obs)[0], j] = np.nan
                        continue
                    idx = np.where(obs)[0]
                    res = forward(group, v2[idx, j])
                    nv[idx, j] = res.value
                    total[idx] += res.log_jac
                    gidx = global_idx[local_rows[idx]]
                    codes[gidx, s, j] = res.bin_code
                    near[gidx, s, j] = res.near_knot
                    residual[gidx, s, j] = res.residual
            v, m, ident = nv, m2, id2
        return v, m, ident, total

    def _inverse_block(self, chain, values, mask, versions,
                       codes, near, residual, rev_arr, id_arr,
                       observed_arr, _row_sel, local_sel, global_idx):
        R, C = values.shape
        new_v = values.copy()
        new_v[~mask] = np.nan
        new_m = mask.copy()
        new_id = np.broadcast_to(np.arange(C), (R, C)).copy()
        total = np.zeros(R)
        local_rows = np.where(local_sel)[0]
        for s in range(len(chain.stages) - 1, -1, -1):
            stage = chain.stages[s]
            perm = np.asarray(stage.perm)
            old_v = np.full((R, C), np.nan)
            old_m = np.zeros((R, C), dtype=bool)
            old_id = np.zeros((R, C), dtype=np.int16)
            for j, gid in enumerate(stage.groups):
                obs = new_m[:, j]
                rev = int(versions[gid])
                group = None
                if obs.any() and self.registry.status(gid, rev) == "ok":
                    group = self.registry.get(gid, rev)
                target = perm[j]
                rev_arr[global_idx[local_rows], s, j] = \
                    np.where(obs, rev, -1)
                id_arr[global_idx[local_rows], s, j] = new_id[:, j]
                observed_arr[global_idx[local_rows], s, j] = obs
                old_m[:, target] = obs
                old_id[:, target] = new_id[:, j]
                if obs.any() and group is not None:
                    idx = np.where(obs)[0]
                    res = inverse(group, new_v[idx, j])
                    old_v[idx, target] = res.value
                    total[idx] += res.log_jac
                    gidx = global_idx[local_rows[idx]]
                    codes[gidx, s, j] = res.bin_code
                    near[gidx, s, j] = res.near_knot
                    residual[gidx, s, j] = res.residual
                free = np.where(~obs)[0]
                old_v[free, target] = new_v[free, j]
            new_v, new_m, new_id = old_v, old_m, old_id
        return new_v, new_m, new_id, total

    # ------------------------------------------------------------------
    # 单点请求（requests/upstream 交换记录）
    # ------------------------------------------------------------------
    def run_requests(self, requests) -> dict:
        out = {}
        for rid, gid, direction, value in requests:
            rev = self.registry.latest(gid)
            group = self.registry.get(gid, rev)
            res = forward(group, np.array([value])) if direction == "forward" \
                else inverse(group, np.array([value]))
            out[rid] = (res, gid, rev)
        return out

    def backprop_requests(self, requests, request_results, upstream):
        """upstream: [[rid, g_value, g_logjac], ...]；按组身份合并梯度。"""
        buckets: dict = {}
        input_grads: dict = {}
        for rid, gv, gl in upstream:
            _res, gid, rev = request_results[rid]
            direction = next(r[2] for r in requests if r[0] == rid)
            value = next(r[3] for r in requests if r[0] == rid)
            group = self.registry.get(gid, rev)
            if direction == "forward":
                g_x, pg, _ = forward_vjp(
                    group, np.array([value]), np.array([gv]),
                    np.array([gl]))
            else:
                g_x, pg, _ = inverse_vjp(
                    group, np.array([value]), np.array([gv]),
                    np.array([gl]))
            input_grads[rid] = float(g_x[0])
            buckets.setdefault((gid, rev), []).append(pg)
        return input_grads, {k: merge_param_gradients(v)
                             for k, v in buckets.items()}

    # ------------------------------------------------------------------
    # 链级反向（承诺回放）
    # ------------------------------------------------------------------
    def backprop_batch(self, batch_index: int, upstream) -> BackpropResult:
        """upstream: [(rid, [每交付通道值梯度], 总 logjac 梯度)]。"""
        if batch_index not in self._batch_cache:
            raise KeyError(f"批次 {batch_index} 尚未执行前向")
        batch = self._batch_cache[batch_index]
        up = {rid: (np.asarray(gv, dtype=np.float64), float(gl))
              for rid, gv, gl in upstream}
        input_grads: dict = {}
        logjac_grads: dict = {}
        buckets: dict = {}

        rows_by_id = {r.request_id: r for r in batch.rows}
        for rid in up:
            row = rows_by_id[rid]
            if not row.available:
                raise InvalidRevision(
                    f"{rid} 不可评分：{row.reason}")

        # 严格按前向收据的分片组织反向：每个 chunk 内的行共享同一链、
        # 同一方向与同一承诺布局，避免 values_in 与 committed 错行。
        for chunk_idx, chunk in enumerate(batch.receipts.shard_iter()):
            chunk_ids = [
                str(rid) for rid in chunk["request_ids"]
                if str(rid) in up]
            if not chunk_ids:
                continue
            chain = self.chains[str(chunk["chain_id"])]
            for start in range(0, len(chunk_ids), batch.shard_rows):
                sub = chunk_ids[start:start + batch.shard_rows]
                self._backprop_shard(chain, sub, batch, up, input_grads,
                                     logjac_grads, buckets)
        result = BackpropResult(
            batch_index=batch_index, input_grads=input_grads,
            logjac_grads=logjac_grads,
            param_grads={k: merge_param_gradients(v)
                         for k, v in buckets.items()})
        return result

    def replay_backprop(self, batch_index: int, upstream) -> BackpropResult:
        """重发回传：必须与首次回传逐分量一致（同一组前向承诺）。"""
        first = self.backprop_batch(batch_index, upstream)
        second = self.backprop_batch(batch_index, upstream)
        for rid, g in first.input_grads.items():
            if not np.allclose(g, second.input_grads[rid], rtol=0.0,
                               atol=0.0):
                raise CommitmentMismatch(f"{rid} 重放回传与首次不一致")
        for key, pg in first.param_grads.items():
            other = second.param_grads[key]
            if not (np.array_equal(pg.width, other.width)
                    and np.array_equal(pg.height, other.height)
                    and np.array_equal(pg.derivative, other.derivative)):
                raise CommitmentMismatch(f"组 {key} 重放梯度与首次不一致")
        second.replayed = True
        return second

    def _backprop_shard(self, chain, rids, batch, up, input_grads,
                        logjac_grads, buckets):
        receipts = batch.receipts
        rows_by_id = {r.request_id: r for r in batch.rows}
        R, C, S = len(rids), chain.n_channels, len(chain.stages)
        directions = {rows_by_id[rid].direction for rid in rids}
        assert directions <= {"forward", "inverse"}
        direction = next(iter(directions))
        versions = batch.versions

        values_in = np.stack([receipts.row(rid)["values_in"]
                              for rid in rids])
        mask_in = np.stack([receipts.row(rid)["mask_in"] for rid in rids])
        # 取出收据中的承诺编码，回放后逐点核对。lookup 只回观测位，
        # 这里直接用分片各行的原始痕迹数组，保持与回放相同的张量级布局。
        committed = np.zeros((R, S, C), dtype=np.int64)
        committed_near = np.zeros((R, S, C), dtype=bool)
        for i, rid in enumerate(rids):
            chunk_idx, row_idx = receipts._index[str(rid)]
            chunk = receipts._load(chunk_idx)
            committed[i] = chunk["stage_codes"][row_idx].astype(np.int64)
            committed_near[i] = chunk["stage_near"][row_idx]

        gv = np.stack([up[rid][0] for rid in rids])
        gL = np.asarray([up[rid][1] for rid in rids])
        logjac_grads.update({rid: float(up[rid][1]) for rid in rids})

        if direction == "forward":
            final_g = self._backprop_forward(
                chain, values_in, mask_in, versions, gv, gL,
                committed, committed_near, buckets)
        else:
            final_g = self._backprop_inverse(
                chain, values_in, mask_in, versions, gv, gL,
                committed, committed_near, buckets)
        for i, rid in enumerate(rids):
            input_grads[rid] = final_g[i]

    def _check_commitment(self, replay_codes, replay_near, committed,
                          committed_near, observed, where):
        if not np.array_equal(replay_codes[observed], committed[observed]):
            mismatch = np.argwhere(
                observed & (replay_codes != committed))
            raise CommitmentMismatch(
                f"{where} 分箱归属偏离前向承诺，首处 {mismatch[0].tolist()}")
        if not np.array_equal(replay_near[observed], committed_near[observed]):
            raise CommitmentMismatch(
                f"{where} 结点贴边标志偏离前向承诺")

    def _backprop_forward(self, chain, values_in, mask_in, versions,
                          gv, gL, committed, committed_near, buckets):
        R, C = values_in.shape
        # 回放前向，保存每级置换后的输入与痕迹并核对承诺。
        v = values_in.copy()
        m = mask_in.copy()
        stage_inputs = []
        replay_codes = np.zeros_like(committed)
        replay_near = np.zeros_like(committed_near)
        for s, stage in enumerate(chain.stages):
            perm = np.asarray(stage.perm)
            v2, m2 = v[:, perm], m[:, perm]
            stage_inputs.append((v2.copy(), m2, perm))
            nv = v2.copy()
            for gid in dict.fromkeys(stage.groups):
                group, rev = self._group_at(gid, versions)
                for j in [j for j, g in enumerate(stage.groups) if g == gid]:
                    obs = m2[:, j]
                    if not obs.any():
                        continue
                    idx = np.where(obs)[0]
                    res = forward(group, v2[idx, j])
                    nv[idx, j] = res.value
                    replay_codes[idx, s, j] = res.bin_code
                    replay_near[idx, s, j] = res.near_knot
            v, m = nv, m2
        # 观测位以各级置换后的掩码为准（内部箱 0 是合法箱，不能用编码零判定）。
        observed = np.zeros_like(committed, dtype=bool)
        for s, stage in enumerate(chain.stages):
            observed[:, s, :] = stage_inputs[s][1]
        self._check_commitment(replay_codes, replay_near, committed,
                               committed_near, observed, "正向回放")

        # 反向遍历各级。
        for s in range(len(chain.stages) - 1, -1, -1):
            stage = chain.stages[s]
            perm = np.asarray(stage.perm)
            v2, m2, _ = stage_inputs[s]
            old_g = np.zeros((R, C))
            for gid in dict.fromkeys(stage.groups):
                group, rev = self._group_at(gid, versions)
                for j in [j for j, g in enumerate(stage.groups) if g == gid]:
                    obs = m2[:, j]
                    target = perm[j]
                    if not obs.any():
                        free = np.where(~obs)[0]
                        old_g[free, target] = gv[free, j]
                        continue
                    idx = np.where(obs)[0]
                    g_x, pg, _ = forward_vjp(
                        group, v2[idx, j], gv[idx, j], gL[idx])
                    old_g[idx, target] = g_x
                    buckets.setdefault((gid, rev), []).append(pg)
                    free = np.where(~obs)[0]
                    old_g[free, target] = gv[free, j]
            gv = old_g
        return gv

    def _backprop_inverse(self, chain, values_in, mask_in, versions,
                          gv, gL, committed, committed_near, buckets):
        R, C = values_in.shape
        # 回放逆向链（处理顺序 S-1..0），保存每级 new 侧输入。
        new_v = values_in.copy()
        new_m = mask_in.copy()
        stage_inputs = []
        replay_codes = np.zeros_like(committed)
        replay_near = np.zeros_like(committed_near)
        observed = np.zeros_like(committed, dtype=bool)
        for s in range(len(chain.stages) - 1, -1, -1):
            stage = chain.stages[s]
            stage_inputs.append((s, new_v.copy(), new_m.copy()))
            perm = np.asarray(stage.perm)
            old_v = np.full((R, C), np.nan)
            old_m = np.zeros((R, C), dtype=bool)
            for j, gid in enumerate(stage.groups):
                group, rev = self._group_at(gid, versions)
                obs = new_m[:, j]
                observed[:, s, j] = obs
                target = perm[j]
                old_m[:, target] = obs
                if obs.any():
                    idx = np.where(obs)[0]
                    res = inverse(group, new_v[idx, j])
                    old_v[idx, target] = res.value
                    replay_codes[idx, s, j] = res.bin_code
                    replay_near[idx, s, j] = res.near_knot
                free = np.where(~obs)[0]
                old_v[free, target] = new_v[free, j]
            new_v, new_m = old_v, old_m
        self._check_commitment(replay_codes, replay_near, committed,
                               committed_near, observed, "逆向回放")

        # 反向：逆级处理顺序的反方向，即 s=0..S-1。
        for s, stage in enumerate(chain.stages):
            perm = np.asarray(stage.perm)
            _stored_s, in_v, in_m = stage_inputs[len(chain.stages) - 1 - s]
            new_g = np.zeros((R, C))
            for j, gid in enumerate(stage.groups):
                group, rev = self._group_at(gid, versions)
                target = perm[j]
                obs = in_m[:, j]
                if not obs.any():
                    free = np.where(~obs)[0]
                    new_g[free, j] = gv[free, target]
                    continue
                idx = np.where(obs)[0]
                g_y, pg, _ = inverse_vjp(
                    group, in_v[idx, j], gv[idx, target], gL[idx])
                new_g[idx, j] = g_y
                buckets.setdefault((gid, rev), []).append(pg)
                free = np.where(~obs)[0]
                new_g[free, j] = gv[free, target]
            gv = new_g
        return gv


def chains_from_records(records: dict) -> dict[str, ChainSpec]:
    specs = {}
    for chain_id, stages in records["chains"].items():
        specs[chain_id] = ChainSpec(
            chain_id=chain_id,
            stages=tuple(StageSpec(tuple(st["perm"]),
                                   tuple(st["groups"])) for st in stages))
    return specs


def registry_from_capture(records: dict, tensors: dict) -> GroupRegistry:
    """按采集资料构造注册表，包含有效版本与 bad_update 隔离。"""
    registry = GroupRegistry()
    for gid, meta in records["groups"].items():
        p = meta["params"]
        registry.commit(
            gid, meta["rev"], meta["extent"],
            tensors[f"{p}_width"], tensors[f"{p}_height"],
            tensors[f"{p}_derivative"],
            min_fraction=meta["min_fraction"],
            min_derivative=meta["min_derivative"])
    bad = records["bad_update"]
    p = bad["params"]
    try:
        meta = records["groups"][bad["group"]]
        registry.commit(
            bad["group"], bad["rev"], meta["extent"],
            tensors[f"{p}_width"], tensors[f"{p}_height"],
            tensors[f"{p}_derivative"],
            min_fraction=meta["min_fraction"],
            min_derivative=meta["min_derivative"])
    except GroupValidation:
        pass
    return registry
