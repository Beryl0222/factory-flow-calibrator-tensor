"""前向承诺的紧凑收据。

正式输入的内存预算不允许为每个输入点保存整份分组参数或分箱搜索表，
因此收据只保存逐点的小整数分箱编码、结点标志和每段的组版本承诺；
中间坐标可在反向时按同一承诺重算，并与编码逐点核对。

当累计收据超过调用方给定的内存上限时，较早的分片透明外溢到临时目录，
回查时按需读回，使驻留内存只覆盖工作区与最近分片，而不是全部输入。
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 分片落盘时保留的紧凑数组键（不物化整组参数或搜索表）。
_ARRAY_KEYS = (
    "values_in", "values_out", "log_jac", "mask_in", "mask_out",
    "identity_out", "stage_codes", "stage_near", "stage_residual",
    "stage_versions", "stage_identity", "observed", "unavailable",
)
_META_KEYS = ("request_ids", "chain_id", "direction", "stage_groups",
              "reasons")


@dataclass(frozen=True)
class StageTrace:
    """一行在一级上的可回查痕迹。"""

    stage: int
    position: int                 # 置换后处理位置（交付通道序）
    group_id: str
    revision: int
    bin_code: int                 # 内部箱下标或 TAIL_LEFT/TAIL_RIGHT
    near_knot: bool
    residual: float               # 与判定结点的坐标差
    identity: int                 # 原始通道身份
    observed: bool


def _encode_bins(codes: np.ndarray) -> np.ndarray:
    # TAIL_LEFT=-1, TAIL_RIGHT=-2 与内部下标一起放进 int16（128 箱足够）。
    return codes.astype(np.int16)


class ReceiptStore:
    """按分片累积的行级收据；支持按请求 ID 回查组与分箱归属。

    ``resident_bytes`` 限制驻留内存；超出后最旧分片以 ``np.savez_compressed``
    外溢到 ``spill_dir``，回查透明读回。
    """

    def __init__(self, resident_bytes: int | None = None,
                 spill_dir: str | Path | None = None):
        self._chunks: list[dict] = []
        self._spilled: dict[int, Path] = {}
        self._index: dict[str, tuple[int, int]] = {}
        self._resident = 0
        self._resident_limit = resident_bytes
        self._spill_dir = Path(spill_dir) if spill_dir is not None else None
        if self._spill_dir is not None:
            self._spill_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def trace_nbytes(n_rows: int, n_channels: int, n_stages: int) -> int:
        """估算单分片收据物化字节（含编码、标志、残差与版本承诺）。"""
        per_point = 2 + 1 + 8 + 4          # int16 箱 + bool + float64 残差 + int32 版本
        return n_rows * n_channels * n_stages * per_point

    def add_shard(self, *, request_ids, chain_id, direction, values_in,
                  values_out, log_jac, mask_in, mask_out, identity_out,
                  stage_codes, stage_near, stage_residual, stage_versions,
                  stage_groups, stage_identity, observed,
                  unavailable=None, reasons=None):
        n_rows, n_channels, n_stages = stage_codes.shape
        chunk = {
            "request_ids": np.asarray(request_ids, dtype=object),
            "chain_id": chain_id,
            "direction": direction,
            "values_in": np.asarray(values_in, dtype=np.float64),
            "values_out": np.asarray(values_out, dtype=np.float64),
            "log_jac": np.asarray(log_jac, dtype=np.float64),
            "mask_in": np.asarray(mask_in, dtype=np.bool_),
            "mask_out": np.asarray(mask_out, dtype=np.bool_),
            "identity_out": np.asarray(identity_out, dtype=np.int16),
            "stage_codes": _encode_bins(stage_codes),
            "stage_near": np.asarray(stage_near, dtype=np.bool_),
            "stage_residual": np.asarray(stage_residual, dtype=np.float64),
            "stage_versions": np.asarray(stage_versions, dtype=np.int32),
            "stage_identity": np.asarray(stage_identity, dtype=np.int16),
            "observed": np.asarray(observed, dtype=np.bool_),
            "stage_groups": stage_groups,
            "unavailable": np.asarray(
                unavailable if unavailable is not None
                else [False] * n_rows, dtype=bool),
            "reasons": np.asarray(
                reasons if reasons is not None
                else ["" for _ in range(n_rows)], dtype=object),
        }
        chunk_idx = len(self._chunks)
        for i, rid in enumerate(request_ids):
            self._index[str(rid)] = (chunk_idx, i)
        self._chunks.append(chunk)
        self._resident += self._array_bytes(chunk)
        self._enforce_resident_limit()
        return chunk

    def _enforce_resident_limit(self):
        if self._resident_limit is None:
            return
        idx = 0
        while (self._resident > self._resident_limit
               and idx < len(self._chunks)):
            if idx in self._spilled:
                idx += 1
                continue
            self._spill(idx)
            idx += 1

    def _spill(self, chunk_idx: int):
        if self._spill_dir is None:
            self._spill_dir = Path(tempfile.mkdtemp(prefix="rqs_receipts_"))
        chunk = self._chunks[chunk_idx]
        path = self._spill_dir / f"chunk_{chunk_idx:06d}.npz"
        payload = {key: chunk[key] for key in _ARRAY_KEYS}
        payload["request_ids"] = np.asarray(chunk["request_ids"], dtype=object)
        np.savez_compressed(path, **payload)
        # 保留轻量元数据，移除驻留大数组。
        self._chunks[chunk_idx] = {
            "request_ids": chunk["request_ids"],
            "chain_id": chunk["chain_id"],
            "direction": chunk["direction"],
            "stage_groups": chunk["stage_groups"],
            "reasons": chunk["reasons"],
        }
        self._spilled[chunk_idx] = path
        self._resident = sum(
            self._array_bytes(c) for i, c in enumerate(self._chunks)
            if i not in self._spilled)

    def _load(self, chunk_idx: int) -> dict:
        if chunk_idx not in self._spilled:
            return self._chunks[chunk_idx]
        path = self._spilled[chunk_idx]
        with np.load(path, allow_pickle=True) as data:
            chunk = {key: data[key] for key in data.files}
        meta = self._chunks[chunk_idx]
        chunk["chain_id"] = meta["chain_id"]
        chunk["direction"] = meta["direction"]
        chunk["stage_groups"] = meta["stage_groups"]
        chunk["reasons"] = meta["reasons"]
        return chunk

    @staticmethod
    def _array_bytes(chunk: dict) -> int:
        return sum(arr.nbytes for arr in chunk.values()
                   if isinstance(arr, np.ndarray))

    def lookup(self, request_id: str) -> list[StageTrace]:
        """回查一行实际走过的组身份、版本与分箱归属。"""
        chunk_idx, row_idx = self._index[str(request_id)]
        chunk = self._load(chunk_idx)
        traces = []
        n_stages = chunk["stage_codes"].shape[1]
        for stage in range(n_stages):
            for pos in range(chunk["stage_codes"].shape[2]):
                if not chunk["observed"][row_idx, stage, pos]:
                    continue
                traces.append(StageTrace(
                    stage=stage,
                    position=pos,
                    group_id=chunk["stage_groups"][stage][pos],
                    revision=int(chunk["stage_versions"][
                        row_idx, stage, pos]),
                    bin_code=int(chunk["stage_codes"][
                        row_idx, stage, pos]),
                    near_knot=bool(chunk["stage_near"][
                        row_idx, stage, pos]),
                    residual=float(chunk["stage_residual"][
                        row_idx, stage, pos]),
                    identity=int(chunk["stage_identity"][
                        row_idx, stage, pos]),
                    observed=True,
                ))
        return traces

    def row(self, request_id: str) -> dict:
        chunk_idx, row_idx = self._index[str(request_id)]
        chunk = self._load(chunk_idx)
        return {key: chunk[key][row_idx]
                for key in ("values_in", "values_out", "log_jac", "mask_in",
                            "mask_out", "identity_out", "unavailable",
                            "reasons")}

    def chunk_bytes(self, chunk_idx: int) -> int:
        if chunk_idx in self._spilled:
            return self._spilled[chunk_idx].stat().st_size
        return self._array_bytes(self._chunks[chunk_idx])

    @property
    def resident_bytes(self) -> int:
        return self._resident

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    @property
    def n_spilled(self) -> int:
        return len(self._spilled)

    @property
    def request_ids(self) -> list[str]:
        return [str(rid) for chunk in self._chunks
                for rid in chunk["request_ids"].tolist()]

    def shard_iter(self):
        for i in range(len(self._chunks)):
            yield self._load(i)
