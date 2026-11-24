"""多版本注册表：坏提交隔离、旧版保留、未知版本处理。"""
import unittest

import numpy as np

from rqs.registry import (
    GroupRegistry,
    InvalidRevision,
    UnknownRevision,
)
from rqs.splines import GroupValidation


class RegistryTests(unittest.TestCase):
    def _meta(self):
        return dict(extent=[0, 10, 0, 12],
                    width=[0.2, 0.7, 0.1], height=[0.2, 0.1, 0.7],
                    min_fraction=4e-4, min_derivative=0.06)

    def test_keeps_historical_revisions(self):
        reg = GroupRegistry()
        reg.commit("B", 5, derivative=[1.2, 1.1, 0.1, 2.0], **self._meta())
        reg.commit("B", 7, derivative=[1.3, 1.0, 0.2, 2.1], **self._meta())
        self.assertEqual(reg.versions("B"), [5, 7])
        old = reg.get("B", 5)
        self.assertAlmostEqual(old.derivative[0], 1.2)
        self.assertEqual(reg.latest("B"), 7)

    def test_non_monotone_commit_is_rejected_and_quarantined(self):
        reg = GroupRegistry()
        reg.commit("B", 5, derivative=[1.2, 1.1, 0.1, 2.0], **self._meta())
        with self.assertRaises(GroupValidation):
            reg.commit("B", 6, derivative=[1.2, 1.1, -0.1, 2.0],
                       **self._meta())
        self.assertEqual(reg.status("B", 6), "invalid")
        with self.assertRaises(InvalidRevision):
            reg.get("B", 6)
        # 旧版本仍可用于回传。
        self.assertEqual(reg.status("B", 5), "ok")
        self.assertIsNotNone(reg.rejection("B", 6))

    def test_unknown_revision_is_isolated(self):
        reg = GroupRegistry()
        self.assertEqual(reg.status("B", 99), "unknown")
        with self.assertRaises(UnknownRevision):
            reg.get("B", 99)

    def test_explicit_reject_records_reason(self):
        reg = GroupRegistry()
        reg.reject("B", 6, "上游提交破坏单调性",
                   params={"derivative": [1.2, 1.1, -0.1, 2.0]})
        self.assertEqual(reg.status("B", 6), "invalid")
        self.assertIn("单调", reg.rejection("B", 6).reason)


if __name__ == "__main__":
    unittest.main()
