"""命令行驱动：对采集资料跑正反变换、回传与分箱回查，结果落盘。

用法：
    python -m rqs.cli data/capture.json data/capacity.json -o out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .app import (
    build_service_from_capture,
    lookup_attribution,
    process_chain_batches,
    process_point_requests,
    process_point_upstream,
    run_chain_upstream,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2,
                   allow_nan=False, default=_json_default),
        encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        value = obj.item()
        return None if (isinstance(value, float) and not np.isfinite(value)) \
            else value
    if isinstance(obj, np.ndarray):
        return [None if (isinstance(v, float) and not np.isfinite(v)) else v
                for v in obj.tolist()]
    raise TypeError(f"不可序列化类型 {type(obj)!r}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="分组 RQS 校准链训练/回放驱动")
    parser.add_argument("capture", type=Path)
    parser.add_argument("capacity", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("out"))
    args = parser.parse_args(argv)

    service, records, tensors = build_service_from_capture(
        args.capture, args.capacity)

    point = process_point_requests(service, records)
    _write(args.out / "point_requests.json", point)

    point_up = process_point_upstream(service, records)
    _write(args.out / "point_upstream.json", point_up)

    chain = process_chain_batches(service, records)
    _write(args.out / "chain_batches.json", chain)

    first, replayed = run_chain_upstream(service, records)
    _write(args.out / "chain_backprop.json",
           {"first": first, "replay": replayed})

    # 分箱归属回查（每个链行实际走过的组/版本/箱）。
    attribution = {}
    for index, batch in enumerate(records["chain_batches"]):
        for rid in batch["rows"]:
            attribution[f"{index}:{rid}"] = lookup_attribution(
                service, index, rid)
    _write(args.out / "bin_attribution.json", attribution)

    # 容量自检：分片工作区与收据字节。
    from .chains import estimate_shard_scratch
    cap = json.loads(args.capacity.read_text(encoding="utf-8"))
    max_stages = max(len(spec.stages) for spec in service.chains.values())
    scratch = estimate_shard_scratch(
        cap["partition_rows"],
        len(records["chain_rows"][0][3]), max_stages)
    receipt_bytes = [
        service._batch_cache[i].receipts.chunk_bytes(c)
        for i in service._batch_cache
        for c in range(service._batch_cache[i].receipts.n_chunks)]
    summary = {
        "partition_rows": cap["partition_rows"],
        "estimated_shard_scratch_bytes": int(scratch),
        "peak_scratch_budget": cap["peak_scratch_bytes"],
        "within_scratch_budget": scratch <= cap["peak_scratch_bytes"],
        "max_receipt_chunk_bytes": int(max(receipt_bytes, default=0)),
        "receipt_budget": cap["max_receipt_bytes"],
        "within_receipt_budget":
            max(receipt_bytes, default=0) <= cap["max_receipt_bytes"],
        "groups": {
            gid: {"versions": service.registry.versions(gid),
                  "status_latest": service.registry.status(
                      gid, service.registry.latest(gid))}
            for gid in records["groups"]},
    }
    _write(args.out / "capacity_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
