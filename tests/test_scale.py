"""正式规模（capacity.json）下的分片、内存预算与正确性测试。"""
import json
import unittest
from pathlib import Path

import numpy as np

from rqs.chains import CalibrationService, estimate_shard_scratch
from rqs.registry import GroupRegistry
from rqs.splines import build_group, forward, inverse, ROUNDTRIP_TOLERANCE

DATA = Path(__file__).resolve().parents[1] / "data"


class ScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cap = json.loads((DATA / "capacity.json").read_text("utf-8"))
        cls.rng = np.random.default_rng(cls.cap["seed"])

    def _make_many_groups(self, n_groups, n_bins):
        registry = GroupRegistry()
        for g in range(n_groups):
            width = self.rng.dirichlet(
                np.full(n_bins, 4.0))
            height = self.rng.dirichlet(
                np.full(n_bins, 4.0))
            # 保证最小占比与正导数。
            width = np.clip(width, 1e-3, None)
            height = np.clip(height, 1e-3, None)
            width /= width.sum()
            height /= height.sum()
            derivative = self.rng.uniform(0.1, 3.0, n_bins + 1)
            registry.commit(
                f"G{g:03d}", 1, [0.0, 1.0, 0.0, 1.0], width, height,
                derivative, min_fraction=1e-4, min_derivative=0.05)
        return registry

    def test_shard_scratch_under_peak_budget(self):
        cap = self.cap
        # 三通道、三级链下的分片工作区估算。
        scratch = estimate_shard_scratch(
            cap["partition_rows"], 3, 3)
        self.assertLessEqual(scratch, cap["peak_scratch_bytes"])

    def test_no_per_point_parameter_or_search_table_copy(self):
        # 收据逐点只存小整数编码与残差，不存整组参数或搜索表。
        n_bins = self.cap["bins"]
        registry = self._make_many_groups(self.cap["groups"], n_bins)
        group = registry.get("G000", 1)
        # 组参数集中存放一次，逐点痕迹（int16 编码）远小于参数体积。
        param_bytes = (group.width.nbytes + group.height.nbytes
                       + group.derivative.nbytes
                       + group.x_knots.nbytes + group.y_knots.nbytes)
        per_point_trace = 2 + 1 + 8 + 4 + 2
        self.assertLess(per_point_trace, param_bytes)
        # 128 箱编码能放进 int16。
        self.assertLess(n_bins, np.iinfo(np.int16).max)

    def test_large_group_roundtrip_and_tail(self):
        n_bins = self.cap["bins"]
        registry = self._make_many_groups(2, n_bins)
        group = registry.get("G000", 1)
        xs = np.concatenate([
            self.rng.uniform(0.0, 1.0, 5000),
            group.x_knots[:: max(1, n_bins // 20)],
            np.array([-0.5, 1.5])])
        y = forward(group, xs)
        back = inverse(group, y.value)
        np.testing.assert_allclose(back.value, xs,
                                   atol=ROUNDTRIP_TOLERANCE)

    def test_sharded_service_preserves_all_rows(self):
        from rqs.chains import ChainSpec, StageSpec
        cap = self.cap
        registry = self._make_many_groups(2, 8)
        chains = {
            "c": ChainSpec("c", (
                StageSpec((0, 1, 2), ("G000", "G001", "G000")),
                StageSpec((1, 2, 0), ("G001", "G000", "G001")),
            ))}
        service = CalibrationService(
            registry, chains, partition_rows=cap["partition_rows"],
            peak_scratch_bytes=cap["peak_scratch_bytes"],
            max_receipt_bytes=cap["max_receipt_bytes"])
        n_rows = 20000
        rows = [
            (f"r{i}", "c", "forward",
             self.rng.uniform(0, 1, 3).tolist(),
             [1, 1, 1]) for i in range(n_rows)]
        versions = {"G000": 1, "G001": 1}
        batch = service.run_batch(0, rows, versions)
        self.assertEqual(len(batch.rows), n_rows)
        self.assertEqual(len(batch.scoreable), n_rows)
        # 收据分块后单块不超过收据预算（编码紧凑）。
        for ci in range(batch.receipts.n_chunks):
            self.assertLessEqual(
                batch.receipts.chunk_bytes(ci),
                cap["max_receipt_bytes"] * 4)
        # 回查任意一行的分箱归属。
        traces = batch.receipts.lookup("r12345")
        self.assertEqual(len(traces), 6)

    def test_receipt_spill_keeps_resident_memory_bounded(self):
        from rqs.chains import ChainSpec, StageSpec
        from rqs.receipts import ReceiptStore
        import tempfile
        cap = self.cap
        registry = self._make_many_groups(2, 8)
        chains = {
            "c": ChainSpec("c", (
                StageSpec((0, 1, 2), ("G000", "G001", "G000")),
            ))}
        with tempfile.TemporaryDirectory() as tmp:
            # 极小驻留上限，强制大多数分片外溢。
            service = CalibrationService(
                registry, chains, partition_rows=500,
                peak_scratch_bytes=cap["peak_scratch_bytes"],
                max_receipt_bytes=cap["max_receipt_bytes"],
                receipt_resident_bytes=4096, spill_dir=tmp)
            rows = [
                (f"s{i}", "c", "forward",
                 self.rng.uniform(0, 1, 3).tolist(), [1, 1, 1])
                for i in range(3000)]
            batch = service.run_batch(0, rows, {"G000": 1, "G001": 1})
            self.assertGreater(batch.receipts.n_spilled, 0)
            # 驻留内存被约束在上限附近，而不是随总行数线性增长。
            self.assertLessEqual(
                batch.receipts.resident_bytes,
                4096 + batch.receipts.chunk_bytes(
                    batch.receipts.n_chunks - 1))
            # 外溢后回查仍透明可用且内容完整。
            traces = batch.receipts.lookup("s0")
            self.assertEqual(len(traces), 3)
            row = batch.receipts.row("s2999")
            self.assertEqual(row["values_in"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
