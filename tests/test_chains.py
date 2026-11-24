"""多级校准链：置换、掩码身份、绕过、总对数雅可比与往返。"""
import json
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
from rqs.chains import (
    CalibrationService,
    chains_from_records,
    registry_from_capture,
)
from rqs.splines import LOGJAC_TOLERANCE, ROUNDTRIP_TOLERANCE

DATA = Path(__file__).resolve().parents[1] / "data"


class ChainMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.tensors = load_case(DATA / "capture.json")
        cls.registry = registry_from_capture(cls.records, cls.tensors)
        cls.chains = chains_from_records(cls.records)
        cls.service = CalibrationService(cls.registry, cls.chains)
        cls.rows = {r[0]: r for r in cls.records["chain_rows"]}

    def test_registry_quarantines_bad_update(self):
        self.assertEqual(self.registry.status("B", 6), "invalid")
        self.assertEqual(self.registry.status("B", 5), "ok")

    def test_forward_inverse_chain_roundtrip(self):
        versions = {"A": 3, "B": 5}
        rows = [self.rows[rid] for rid in ("mix0", "mix2")]
        batch = self.service.run_batch(0, rows, versions)
        for row in batch.scoreable:
            src = self.rows[row.request_id]
            back_rows = [[
                "rt", src[1], "inverse", row.values.tolist(),
                row.mask.tolist()]]
            back = self.service.run_batch(
                99, back_rows, versions).scoreable[0]
            observed = np.asarray(src[4], dtype=bool)
            np.testing.assert_allclose(
                back.values[observed], np.asarray(src[3])[observed],
                atol=ROUNDTRIP_TOLERANCE)
            self.assertAlmostEqual(back.log_jac, -row.log_jac,
                                   delta=LOGJAC_TOLERANCE * 8)

    def test_bypass_channels_are_untouched_and_contribute_no_density(self):
        versions = {"A": 3, "B": 5}
        batch = self.service.run_batch(
            1, [self.rows["mix0"]], versions)
        row = batch.scoreable[0]
        src = self.rows["mix0"]
        # 通道 1 无观测：值原样穿过，掩码仍为 0。
        identities = row.identity
        bypass_pos = int(np.where(identities == 1)[0][0])
        self.assertFalse(bool(row.mask[bypass_pos]))
        self.assertTrue(np.isnan(row.values[bypass_pos]))

    def test_total_logjac_only_over_observed_channels(self):
        versions = {"A": 3, "B": 5}
        full = self.service.run_batch(
            2, [self.rows["mix2"]], versions).scoreable[0]
        bypass = self.service.run_batch(
            3, [self.rows["mix0"]], versions).scoreable[0]
        # 全观测行有 3 通道 × 3 级贡献；绕过行贡献更少。
        self.assertNotAlmostEqual(full.log_jac, bypass.log_jac)
        self.assertEqual(len(self.service.run_batch(
            4, [self.rows["mix0"]], versions).receipts.lookup("mix0")),
            6)   # 2 个观测通道 × 3 级

    def test_identity_follows_permutation(self):
        versions = {"A": 3, "B": 5}
        row = self.service.run_batch(
            5, [self.rows["mix2"]], versions).scoreable[0]
        np.testing.assert_array_equal(np.sort(row.identity), [0, 1, 2])

    def test_invalid_group_on_bypass_only_row_is_scoreable(self):
        # 构造一个 B 全程只落在绕过位的场景：掩码使 B 位置始终无观测。
        versions = {"A": 3, "B": 6}
        chain = self.chains["sensor-bypass"]
        # sensor-bypass 两级的 B 位：stage0 位置2 读旧通道1；
        # stage1 位置1 读旧通道2。要让 B 全程绕过，需通道1、2 均无观测。
        row = ("bx", "sensor-bypass", "forward",
               [1.2, 2.001, 1.1999999998], [1, 0, 0])
        batch = self.service.run_batch(21, [row], versions)
        self.assertEqual(len(batch.scoreable), 1)
        self.assertTrue(batch.rows[0].available)

    def test_invalid_group_reaches_observed_position_after_perm(self):
        # mix3：置换后第二级 B 位有观测，故 B r6 无效时不可评分。
        versions = {"A": 3, "B": 6}
        batch = self.service.run_batch(22, [self.rows["mix3"]], versions)
        self.assertEqual(len(batch.scoreable), 0)
        self.assertIn("B", batch.rows[0].reason)

    def test_invalid_group_on_observed_row_is_unavailable_with_reason(self):
        versions = {"A": 3, "B": 6}
        batch = self.service.run_batch(
            7, [self.rows["mix4"]], versions)
        self.assertEqual(len(batch.scoreable), 0)
        row = batch.rows[0]
        self.assertFalse(row.available)
        self.assertIn("B", row.reason)
        self.assertIn("6", row.reason)
        reasons = {u.group_id: u for u in batch.unavailable}
        self.assertEqual(reasons["B"].status, "invalid")

    def test_late_invalid_stage_does_not_emit_partial_chain_output(self):
        # 不可用行：受坏组影响的观测位置为 NaN，且带明确原因；
        # 有效的较早级中间值不得被包装成整条链的可评分输出。
        versions = {"A": 3, "B": 6}
        batch = self.service.run_batch(
            8, [self.rows["mix4"]], versions)
        row = batch.rows[0]
        self.assertFalse(row.available)
        self.assertTrue(row.reason)
        # 至少一个观测位因坏组而为 NaN。
        self.assertTrue(np.any(np.isnan(row.values[row.mask])))
        # 行级状态明确不可评分，而不是静默给出看似完整的输出。
        self.assertEqual(len(batch.scoreable), 0)

    def test_receipts_record_group_revision_and_bin_per_stage(self):
        versions = {"A": 3, "B": 5}
        batch = self.service.run_batch(
            9, [self.rows["mix2"]], versions)
        traces = batch.receipts.lookup("mix2")
        self.assertEqual(len(traces), 9)   # 3 通道 × 3 级
        for t in traces:
            self.assertIn(t.group_id, ("A", "B"))
            self.assertEqual(t.revision, versions[t.group_id])
            self.assertIsInstance(t.bin_code, int)


if __name__ == "__main__":
    unittest.main()
