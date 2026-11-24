"""参数组多版本注册表。

规则：

* 每个组身份保留全部已接纳的历史版本，旧训练步继续按旧组参数回传；
* 提交不通过接纳前提（含非单调）的版本时标记为无效版本，不产生可用变换；
* 无法确认旧参数的引用直接按未知版本隔离；
* 无效版本与未知版本分别给出原因，供批处理决定行级可用性。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .splines import RqsGroup, GroupValidation, build_group


class InvalidRevision(Exception):
    """引用的版本已被标记为无效提交。"""


class UnknownRevision(Exception):
    """引用的版本在注册表中不存在，无法确认旧参数。"""


@dataclass
class RejectedRevision:
    group_id: str
    revision: int
    reason: str
    params: dict


@dataclass
class GroupRegistry:
    _groups: dict = field(default_factory=dict)          # id -> {rev: RqsGroup}
    _rejected: dict = field(default_factory=dict)        # (id, rev) -> RejectedRevision
    _latest: dict = field(default_factory=dict)          # id -> rev

    def commit(self, group_id: str, revision: int, extent, width, height,
               derivative, *, min_fraction: float,
               min_derivative: float) -> RqsGroup:
        """提交一个新版本；接纳失败时登记原因并抛出 ``GroupValidation``。"""
        try:
            group = build_group(
                group_id, revision, extent,
                np.asarray(width, dtype=np.float64),
                np.asarray(height, dtype=np.float64),
                np.asarray(derivative, dtype=np.float64),
                min_fraction=min_fraction,
                min_derivative=min_derivative,
            )
        except GroupValidation as exc:
            self._rejected[(group_id, int(revision))] = RejectedRevision(
                group_id=group_id, revision=int(revision),
                reason=str(exc),
                params={"width": np.asarray(width).tolist(),
                        "height": np.asarray(height).tolist(),
                        "derivative": np.asarray(derivative).tolist()},
            )
            raise
        self._groups.setdefault(group_id, {})[int(revision)] = group
        # 版本号单调推进才更新 latest；历史版本始终保留。
        if group_id not in self._latest or revision > self._latest[group_id]:
            self._latest[group_id] = int(revision)
        return group

    def reject(self, group_id: str, revision: int, reason: str,
               params: dict | None = None) -> None:
        """显式登记一个无效提交（如采集资料中的 bad_update）。"""
        self._rejected[(group_id, int(revision))] = RejectedRevision(
            group_id=group_id, revision=int(revision), reason=reason,
            params=params or {},
        )

    def is_available(self, group_id: str, revision: int) -> bool:
        return group_id in self._groups and int(revision) in self._groups[group_id]

    def status(self, group_id: str, revision: int) -> str:
        """返回 ``ok`` / ``invalid`` / ``unknown``。"""
        if self.is_available(group_id, revision):
            return "ok"
        if (group_id, int(revision)) in self._rejected:
            return "invalid"
        return "unknown"

    def get(self, group_id: str, revision: int) -> RqsGroup:
        revision = int(revision)
        if (group_id, revision) in self._rejected:
            rec = self._rejected[(group_id, revision)]
            raise InvalidRevision(
                f"组 {group_id} 版本 {revision} 为无效提交：{rec.reason}"
            )
        if group_id not in self._groups or revision not in self._groups[group_id]:
            raise UnknownRevision(
                f"组 {group_id} 版本 {revision} 不在注册表中，无法确认旧参数"
            )
        return self._groups[group_id][revision]

    def rejection(self, group_id: str, revision: int) -> RejectedRevision | None:
        return self._rejected.get((group_id, int(revision)))

    def latest(self, group_id: str) -> int:
        return self._latest[group_id]

    def versions(self, group_id: str) -> list[int]:
        return sorted(self._groups.get(group_id, {}))
