"""链级反向：承诺回放、重放一致、有限差分与坏组隔离。"""
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
from rqs.chains import (
    CalibrationService,
    CommitmentMismatch,
    estimate_shard_scratch,
    chains_from_records,
    registry_from_capture,
)
from rqs.registry import InvalidRevision

DATA = Path(__file__).resolve().parents[1] / "data"


def build_service(**kw):
    records, tensors = load_case(DATA / "capture.json")
    registry = registry_from_capture(records, tensors)
    chains = chains_from_records(records)
    return CalibrationService(registry, chains, **kw), records, tensors


class BackpropTests(unittest.TestCase):
    def setUp(self):
        self.service, self.records, self.tensors = build_service()
        self.rows = {r[0]: r for r in self.records["chain_rows"]}
        self.versions = {"A": 3, "B": 5}

    def test_input_gradients_match_finite_differences(self):
        rids = ["mix2"]
        batch = self.service.run_batch(10, [self.rows[r] for r in rids],
                                       self.versions)
        upstream = [("mix2", [0.4, 2.0, 0.3], 0.9)]
        bp = self.service.backprop_batch(10, upstream)
        # 有限差分：直接扰动输入，重跑前向，按交付通道评估损失。
        eps = 1e-7
        gv, gl = upstream[0][1], upstream[0][2]
        fd = np.zeros(3)
        for c in range(3):
            vals = np.array(self.rows["mix2"][3], dtype=float)
            row = self.rows["mix2"]
            for sign in (1, -1):
                v2 = vals.copy()
                v2[c] += sign * eps
                probe = ("fd", row[1], row[2], v2.tolist(), row[4])
                out = self.service.run_batch(
                    100 + c + (1 if sign > 0 else 0), [probe],
                    self.versions).scoreable[0]
                loss = float(np.dot(gv, out.values)) + gl * out.log_jac
                fd[c] += sign * loss / (2 * eps)
        np.testing.assert_allclose(bp.input_grads["mix2"], fd, atol=2e-6)

    def test_replay_matches_first_backprop(self):
        batch = self.service.run_batch(
            11, [self.rows["mix0"], self.rows["mix2"]], self.versions)
        upstream = [
            ("mix2", [0.4, 2.0, 0.3], 0.9),
            ("mix0", [0.8, 0.0, 1.0], 0.2),
        ]
        first = self.service.backprop_batch(11, upstream)
        second = self.service.replay_backprop(11, upstream)
        self.assertTrue(second.replayed)
        for rid in first.input_grads:
            np.testing.assert_array_equal(
                first.input_grads[rid], second.input_grads[rid])
        for key in first.param_grads:
            np.testing.assert_array_equal(
                first.param_grads[key].width,
                second.param_grads[key].width)
            np.testing.assert_array_equal(
                first.param_grads[key].derivative,
                second.param_grads[key].derivative)

    def test_param_gradients_merged_by_group_identity(self):
        batch = self.service.run_batch(
            12, [self.rows["mix0"], self.rows["mix2"]], self.versions)
        upstream = [
            ("mix2", [0.4, 2.0, 0.3], 0.9),
            ("mix0", [0.8, 0.0, 1.0], 0.2),
        ]
        bp = self.service.backprop_batch(12, upstream)
        self.assertIn(("A", 3), bp.param_grads)
        self.assertIn(("B", 5), bp.param_grads)
        for key, pg in bp.param_grads.items():
            self.assertAlmostEqual(float(pg.width.sum()), 0.0, places=12)
            self.assertAlmostEqual(float(pg.height.sum()), 0.0, places=12)
            self.assertGreater(pg.used_bins.sum(), 0)

    def test_backprop_rejects_unavailable_row(self):
        batch = self.service.run_batch(
            13, [self.rows["mix4"]], {"A": 3, "B": 6})
        with self.assertRaises(InvalidRevision):
            self.service.backprop_batch(
                13, [("mix4", [1.0, 1.0, 1.0], 0.5)])

    def test_commitment_mismatch_detected(self):
        self.service.run_batch(
            14, [self.rows["mix2"]], self.versions)
        # 篡改收据的分箱编码，回放必须拒绝而非静默换侧。
        batch = self.service._batch_cache[14]
        chunk = batch.receipts._chunks[0]
        original = chunk["stage_codes"].copy()
        chunk["stage_codes"][0, 0, 0] = \
            int(chunk["stage_codes"][0, 0, 0]) + 1
        try:
            with self.assertRaises(CommitmentMismatch):
                self.service.backprop_batch(
                    14, [("mix2", [0.4, 2.0, 0.3], 0.9)])
        finally:
            chunk["stage_codes"][:] = original

    def test_inverse_chain_backprop_runs(self):
        batch = self.service.run_batch(
            15, [self.rows["mix1"]], self.versions)
        bp = self.service.backprop_batch(
            15, [("mix1", [0.75, -0.3, 1.0], 1.5)])
        self.assertIn("mix1", bp.input_grads)
        self.assertEqual(len(bp.input_grads["mix1"]), 3)


class ScratchBudgetTests(unittest.TestCase):
    def test_shard_estimate_within_peak_budget(self):
        capacity = __import__("json").loads(
            (DATA / "capacity.json").read_text(encoding="utf-8"))
        # 正式规模下分片工作区不得超过峰值暂存预算。
        per_shard = estimate_shard_scratch(
            capacity["partition_rows"], 3, 3)
        self.assertLessEqual(per_shard,
                             capacity["peak_scratch_bytes"])

    def test_service_shards_large_batch(self):
        service, records, _ = build_service(
            partition_rows=4, peak_scratch_bytes=10**9)
        rows = [self._row(i, records) for i in range(10)]
        batch = service.run_batch(
            20, rows, {"A": 3, "B": 5})
        # 分片不改变行完整性。
        self.assertEqual(len(batch.rows), 10)
        self.assertLessEqual(batch.shard_rows, 4)
        self.assertGreaterEqual(batch.receipts.n_chunks,
                                int(np.ceil(10 / 4)))

    @staticmethod
    def _row(i, records):
        base = records["chain_rows"][i % len(records["chain_rows"])]
        return (f"big{i}", base[1], base[2], base[3], base[4])


if __name__ == "__main__":
    unittest.main()
