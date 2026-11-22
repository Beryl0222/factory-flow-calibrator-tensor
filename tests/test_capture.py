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
