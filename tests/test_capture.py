"""核对合成采集的交换记录与原始尺寸。"""
import json
from pathlib import Path
import unittest

import numpy as np

from caseio import load_case


class CaptureTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1] / "data"
        self.records, self.tensors = load_case(data / "capture.json")
        self.capacity = json.loads((data / "capacity.json").read_text(encoding="utf-8"))

    def test_capture_contract(self):
        r, a = self.records, self.tensors
        for group in ("a", "b"):
            self.assertAlmostEqual(a[group+"_width"].sum(), 1.0)
            self.assertAlmostEqual(a[group+"_height"].sum(), 1.0)
            self.assertEqual(len(a[group+"_width"]) + 1, len(a[group+"_derivative"]))
        self.assertTrue(np.any(a["br_derivative"] < 0))
        self.assertEqual(len(r["requests"]), sum(map(len, r["upstream"])))
        chain_rows = {row[0]: row for row in r["chain_rows"]}
        for batch in r["chain_batches"]:
            for row_id in batch["rows"]:
                _, chain, _, values, mask = chain_rows[row_id]
                self.assertEqual(len(values), len(mask))
                self.assertTrue(all(len(stage["perm"]) == len(mask) for stage in r["chains"][chain]))
        self.assertIn(r["replay_chain_upstream"]["upstream"], r)
