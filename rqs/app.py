"""把采集资料接到训练/回放工作流的应用层驱动。

提供：

* ``build_service_from_capture``：构造注册表（含 bad_update 隔离）与链；
* ``process_point_requests``：单组正/逆请求与上游梯度回传；
* ``process_chain_batches``：按批次版本运行链、产出收据与不可用原因；
* ``run_upstream``：链级反向与重放；
* 紧凑收据落盘/回查，满足正式输入的内存预算。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from caseio import load_case
from .chains import (
    CalibrationService,
    chains_from_records,
    registry_from_capture,
)


def build_service_from_capture(capture_path: Path,
                               capacity_path: Path | None = None,
                               **overrides) -> tuple:
    records, tensors = load_case(capture_path)
    registry = registry_from_capture(records, tensors)
    chains = chains_from_records(records)
    kwargs = {}
    if capacity_path is not None:
        cap = json.loads(Path(capacity_path).read_text(encoding="utf-8"))
        kwargs.update(
            partition_rows=cap["partition_rows"],
            peak_scratch_bytes=cap["peak_scratch_bytes"],
            max_receipt_bytes=cap["max_receipt_bytes"])
    kwargs.update(overrides)
    service = CalibrationService(registry, chains, **kwargs)
    return service, records, tensors


def process_point_requests(service: CalibrationService, records: dict):
    """单点 requests：返回每行变换结果与所用组版本。"""
    results = service.run_requests(records["requests"])
    payload = {}
    for rid, (res, gid, rev) in results.items():
        payload[rid] = {
            "group": gid,
            "revision": rev,
            "value": float(res.value[0]),
            "log_jac": float(res.log_jac[0]),
            "bin_code": int(res.bin_code[0]),
            "near_knot": bool(res.near_knot[0]),
            "residual": None if np.isnan(res.residual[0])
            else float(res.residual[0]),
        }
    return payload


def process_point_upstream(service: CalibrationService, records: dict):
    """upstream 的两批回传：按组身份合并参数梯度。"""
    request_results = service.run_requests(records["requests"])
    batches = []
    for sublist in records["upstream"]:
        input_grads, param_grads = service.backprop_requests(
            records["requests"], request_results, sublist)
        batches.append({
            "input_grads": {k: float(v) for k, v in input_grads.items()},
            "param_grads": {
                f"{gid}:r{rev}": {
                    "width": pg.width.tolist(),
                    "height": pg.height.tolist(),
                    "derivative": pg.derivative.tolist(),
                    "used_bins": pg.used_bins.tolist(),
                }
                for (gid, rev), pg in param_grads.items()},
        })
    return batches


def process_chain_batches(service: CalibrationService, records: dict):
    """按 chain_batches 顺序运行，返回每批评分记录与不可用原因。"""
    rows = {r[0]: r for r in records["chain_rows"]}
    outputs = []
    for index, batch in enumerate(records["chain_batches"]):
        batch_rows = [rows[rid] for rid in batch["rows"]]
        result = service.run_batch(
            index, batch_rows, batch["group_versions"])
        outputs.append({
            "batch_index": index,
            "scoreable": [
                {
                    "request_id": r.request_id,
                    "chain_id": r.chain_id,
                    "direction": r.direction,
                    "values": [None if np.isnan(v) else float(v)
                               for v in r.values],
                    "log_jac": r.log_jac,
                    "mask": r.mask.tolist(),
                    "identity": r.identity.tolist(),
                }
                for r in result.scoreable],
            "unavailable": [
                {
                    "request_id": u.request_id,
                    "group": u.group_id,
                    "revision": u.revision,
                    "status": u.status,
                    "detail": u.detail,
                }
                for u in result.unavailable],
            "versions": result.versions,
        })
    return outputs


def run_chain_upstream(service: CalibrationService, records: dict):
    """chain_upstream 回传与 replay_chain_upstream 重放。"""
    replay_ref = records["replay_chain_upstream"]
    batch_index = int(replay_ref["batch_index"])
    first = service.backprop_batch(batch_index, records["chain_upstream"])
    replayed = service.replay_backprop(batch_index,
                                       records["chain_upstream"])
    return _serialize_backprop(first), _serialize_backprop(replayed)


def _serialize_backprop(result):
    return {
        "batch_index": result.batch_index,
        "input_grads": {
            rid: np.asarray(g).tolist()
            for rid, g in result.input_grads.items()},
        "logjac_grads": result.logjac_grads,
        "param_grads": {
            f"{gid}:r{rev}": {
                "width": pg.width.tolist(),
                "height": pg.height.tolist(),
                "derivative": pg.derivative.tolist(),
                "used_bins": pg.used_bins.tolist(),
                "n_points": pg.n_points,
            }
            for (gid, rev), pg in result.param_grads.items()},
        "replayed": result.replayed,
    }


def lookup_attribution(service: CalibrationService, batch_index: int,
                       request_id: str):
    """回查一行实际使用的参数组、版本与分箱归属。"""
    batch = service._batch_cache[batch_index]
    return [
        {**t, "residual": (None if not np.isfinite(t["residual"])
                           else t["residual"])}
        for t in (trace.__dict__ for trace in
                  batch.receipts.lookup(request_id))]
