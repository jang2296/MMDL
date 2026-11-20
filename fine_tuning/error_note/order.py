"""Write the pinned training order (item_order.json). Run once locally; the file is committed and pods only read it.

CLI: python -m fine_tuning.error_note.order build --approval APPROVAL_EXPORT.json --manifest DATASET_MANIFEST.json
         --lineage VAULT/train/lineage.jsonl --output fine_tuning/error_note/item_order.json
     python -m fine_tuning.error_note.order spread --base fine_tuning/error_note/item_order.json
         --output fine_tuning/error_note/item_order_v2.json
     python -m fine_tuning.error_note.order quota --base ORDER.json --quotas QUOTAS.json --head-count 1600
         --output ORDER_V3.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .contract import MASTER_SEED, round_robin_order, spread_order

VERSION = "error-note-order-v1"
VERSION_V2 = "error-note-order-v2"
VERSION_V3 = "error-note-order-v3"
PILOT_COUNT = 300  # = frozen_data.PILOT_COUNT = protocol.single_blocks' pilot boundary
RULE_V3 = ("The base order's items, ids unchanged. Head (first head_count items): for each source the first quotas[source] "
           "items of that source in base order. The head is cut at the SINGLE campaign's update blocks "
           "(protocol.single_blocks: 16-item blocks, the pilot300 closing its own 12-item block) and the head end. "
           "Every segment except the last gets floor(size*q/head_count) items per source, and the remaining slots go to "
           "the sources with the largest cumulative deficit end*q/head_count - (taken + floor share), ties by source name "
           "(no RNG); the last segment takes what is left of each quota. Inside a segment sources follow name order, "
           "each in base order. Tail: every other item in base order. Reads only problem_id/source/cluster, never gold "
           "or results.")


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def build(approval: dict, manifest: dict, lineage_rows: list[dict]) -> dict:
    """Training ids from approval["selected_ids"], subject and source from manifest["items"], structure_hash from lineage."""
    manifest_rows = {row["problem_id"]: row for row in manifest["items"]}
    structure_hashes = {row["problem_id"]: row["structure_hash"] for row in lineage_rows}
    items = []
    for problem_id in approval["selected_ids"]:
        if problem_id not in manifest_rows:
            raise ValueError(f"{problem_id} is missing from the manifest")
        if problem_id not in structure_hashes:
            raise ValueError(f"{problem_id} is missing from the lineage")
        row = manifest_rows[problem_id]
        if row["status"] != "INCLUDED":
            raise ValueError(f"{problem_id} has manifest status {row['status']}, not INCLUDED")
        items.append({"problem_id": problem_id, "subject": row["subject"], "structure_hash": structure_hashes[problem_id]})
    order = round_robin_order(items)
    kept = set(order)
    return {
        "version": VERSION,
        "master_seed": MASTER_SEED,
        "rule": "Selected ids sorted by sha256(master_seed:id), one item kept per structure_hash, then taken round-robin across "
                "subjects in sorted subject order.",
        "dropped_duplicate_structure_ids": [item["problem_id"] for item in items if item["problem_id"] not in kept],
        "items": [{"problem_id": i, "subject": manifest_rows[i]["subject"], "source": manifest_rows[i]["source"]} for i in order],
        "count": len(order),
        "ids_sha256": _ids_sha256(order),
    }


def build_v2(base: dict, base_sha256: str) -> dict:
    """The same ids as the v1 order, sources spread evenly over the run (contract.spread_order). Ids and sources only."""
    by_id = {item["problem_id"]: item for item in base["items"]}
    order = spread_order(base["items"], [item["problem_id"] for item in base["items"]])
    return {
        "version": VERSION_V2,
        "master_seed": MASTER_SEED,
        "base_order": {"version": base["version"], "sha256": base_sha256, "ids_sha256": base["ids_sha256"]},
        "rule": "The v1 order's items; within each source the v1 order is kept, and all items are sorted by "
                "((rank in source + 0.5) / source size, v1 position), so every source is spread over the whole run.",
        "items": [by_id[problem_id] for problem_id in order],
        "count": len(order),
        "ids_sha256": _ids_sha256(order),
    }


def build_v3(base: dict, base_sha256: str, quotas: dict[str, int], head_count: int) -> dict:
    """Same ids as base; a quota-fixed, block-proportional head, then the rest in base order (RULE_V3)."""
    from .protocol import single_blocks  # imports torch; kept off read_order's import path

    items = base["items"]
    by_source: dict[str, list[dict]] = {}
    for item in items:
        by_source.setdefault(item["source"], []).append(item)
    if (type(head_count) is not int or not 0 < head_count <= len(items) or type(quotas) is not dict
            or any(type(q) is not int or not 0 < q <= len(by_source.get(s, ())) for s, q in quotas.items())
            or sum(quotas.values()) != head_count):
        raise ValueError("order v3: quotas must be positive ints, each <= the source's items, summing to head_count")
    names = sorted(quotas)
    segments, start = [], 0  # (size inside the head, campaign block size)
    for block in single_blocks([item["problem_id"] for item in items], pilot_count=PILOT_COUNT):
        if start < head_count:
            segments.append((min(len(block), head_count - start), len(block)))
        start += len(block)
    taken = dict.fromkeys(names, 0)
    head: list[dict] = []
    end = 0
    for index, (size, block_size) in enumerate(segments):
        end += size
        if index == len(segments) - 1:
            counts = {s: quotas[s] - taken[s] for s in names}
            off = any(abs(counts[s] - Fraction(size * quotas[s], head_count)) > 1 for s in names)
            if min(counts.values()) < 0 or (size == block_size and off):
                raise ValueError("order v3: these quotas cannot be interleaved within +-1 per block")
        else:
            counts = {s: size * quotas[s] // head_count for s in names}
            deficit = {s: Fraction(end * quotas[s], head_count) - taken[s] - counts[s] for s in names}
            for s in sorted(names, key=lambda s: (-deficit[s], s))[:size - sum(counts.values())]:
                counts[s] += 1
        for s in names:
            head += by_source[s][taken[s]:taken[s] + counts[s]]
            taken[s] += counts[s]
    chosen = {item["problem_id"] for item in head}
    ordered = head + [item for item in items if item["problem_id"] not in chosen]
    ids = [item["problem_id"] for item in ordered]
    selection = dict(base.get("selection", {})) | {"pilot_ids_sha256": _ids_sha256(ids[:PILOT_COUNT])}
    if "cluster_pilot_counts" in selection:
        selection["cluster_pilot_counts"] = dict(sorted(Counter(item["cluster"] for item in ordered[:PILOT_COUNT]).items()))
    return {
        "version": VERSION_V3,
        "master_seed": MASTER_SEED,
        "base_order": {"version": base["version"], "sha256": base_sha256, "ids_sha256": base["ids_sha256"]},
        "rule": RULE_V3,
        "head_count": head_count,
        "quotas": {s: quotas[s] for s in names},
        "selection": selection,
        "items": ordered,
        "count": len(ordered),
        "ids_sha256": _ids_sha256(ids),
    }


def read_order(path: Path, *, expected_sha256: str | None = None) -> dict:
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("item order file sha256 does not match the expected value")
    data = json.loads(raw)
    ids = [item["problem_id"] for item in data["items"]]
    if data.get("version") not in (VERSION, VERSION_V2, VERSION_V3):
        raise ValueError(f"unexpected order version {data.get('version')!r}")
    if data["count"] != len(ids):
        raise ValueError("count does not match the number of items")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate problem_id in the order")
    if data["ids_sha256"] != _ids_sha256(ids):
        raise ValueError("ids_sha256 does not match the items")
    if data["version"] == VERSION_V3 and (
            dict(Counter(item["source"] for item in data["items"][:data["head_count"]])) != data["quotas"]
            or data["selection"]["pilot_ids_sha256"] != _ids_sha256(ids[:PILOT_COUNT])):
        raise ValueError("order v3: head source counts or pilot digest do not match the items")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    for name in ("approval", "manifest", "lineage", "output"):
        build_parser.add_argument(f"--{name}", type=Path, required=True)
    spread_parser = sub.add_parser("spread")
    for name in ("base", "output"):
        spread_parser.add_argument(f"--{name}", type=Path, required=True)
    quota_parser = sub.add_parser("quota")
    for name in ("base", "quotas", "output"):
        quota_parser.add_argument(f"--{name}", type=Path, required=True)
    quota_parser.add_argument("--head-count", type=int, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.command == "quota":
        order = build_v3(read_order(args.base), hashlib.sha256(args.base.read_bytes()).hexdigest(),
                         json.loads(args.quotas.read_text()), args.head_count)
        args.output.write_text(json.dumps(order, indent=1) + "\n")
        print(f"count={order['count']} ids_sha256={order['ids_sha256']} "
              f"pilot_ids_sha256={order['selection']['pilot_ids_sha256']}")
        return 0
    if args.command == "spread":
        base = read_order(args.base)
        if base["version"] != VERSION:
            raise SystemExit("spread: the base must be a v1 order")
        order = build_v2(base, hashlib.sha256(args.base.read_bytes()).hexdigest())
        args.output.write_text(json.dumps(order, indent=1) + "\n")
        print(f"count={order['count']} ids_sha256={order['ids_sha256']}")
        return 0
    lineage_rows = [json.loads(line) for line in args.lineage.read_text().splitlines() if line.strip()]
    order = build(json.loads(args.approval.read_text()), json.loads(args.manifest.read_text()), lineage_rows)
    args.output.write_text(json.dumps(order, indent=1) + "\n")
    print(f"count={order['count']} ids_sha256={order['ids_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
