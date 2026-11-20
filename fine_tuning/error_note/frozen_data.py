"""Hash-only runtime validation of the locally approved 3,500-item training order."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from fine_tuning.notebook_rl import local_run, pool_freeze
from fine_tuning.notebook_rl.launch import descriptor, read_descriptor

from .order import _ids_sha256, read_order

TRAIN_COUNT = 3500
PILOT_COUNT = 300


def _counts(value: object, actual: Counter, name: str, *, caps: bool = False) -> None:
    if (type(value) is not dict or any(type(k) is not str or type(v) is not int or v < 0
                                      for k, v in value.items())):
        raise ValueError(f"frozen data: invalid {name}")
    if (any(actual[k] > v for k, v in value.items()) if caps else dict(actual) != value):
        raise ValueError(f"frozen data: {name} differs from the materialized selection")


def verify_frozen_data(build_root: Path, frozen_approval: dict, order_path: Path, *, order_sha256: str,
                       expected_pilot_sha256: str | None = None) -> dict:
    """Require externally pinned approval/order bytes; never open the benchmark index or re-run a scan.

    The pilot digest must come from the caller's pinned job or selection.pilot_ids_sha256 in the pinned order.
    Optional selection quotas are checked against manifest metadata, not counters declared by the order.
    This validates data only; it grants neither paid execution nor parameter-update permission.
    """
    order_path = Path(order_path).absolute()
    descriptor({"path": str(order_path), "file_sha256": order_sha256})
    order = read_order(order_path, expected_sha256=order_sha256)
    checked, _, verification = local_run.frozen_approval_output(Path(build_root), frozen_approval)
    union = read_descriptor({"path": str(Path(build_root).absolute() / "contamination" / pool_freeze.UNION_FILE),
                             "file_sha256": verification["vault"]["union_receipt_file_sha256"]})
    manifest = read_descriptor(union["dataset_manifest"])
    selected_categories = pool_freeze.selection_categories(manifest)
    rows = manifest["items"]
    ids = [row["problem_id"] for row in rows]
    if any(type(pid) is not str or not pid for pid in ids) or len(ids) != len(set(ids)):
        raise ValueError("frozen data: duplicate or invalid manifest problem_id")
    included = {row["problem_id"]: row for row in rows if row["status"] == pool_freeze.INCLUDED}
    if len(included) != TRAIN_COUNT or any(row["split"] != "train" for row in included.values()):
        raise ValueError("frozen data: expected exactly 3500 INCLUDED train items")
    ordered_ids = [item["problem_id"] for item in order["items"]]
    selected = checked["selected_ids"]
    if (len(selected) != TRAIN_COUNT or len(set(selected)) != TRAIN_COUNT
            or set(selected) != set(included)):
        raise ValueError("frozen data: approval must select exactly the 3500 INCLUDED train items")
    if len(ordered_ids) != TRAIN_COUNT or set(ordered_ids) != set(included):
        raise ValueError("frozen data: order must contain exactly the 3500 INCLUDED train items, no held-out/excluded IDs")
    if order.get("dropped_duplicate_structure_ids"):
        raise ValueError("frozen data: order dropped duplicate structures")
    for item in order["items"]:
        if item.get("source") != included[item["problem_id"]].get("source"):
            raise ValueError("frozen data: order source differs from the manifest")
    selection = order.get("selection", {})
    if type(selection) is not dict:
        raise ValueError("frozen data: selection metadata must be an object")
    pilot_sha = _ids_sha256(ordered_ids[:PILOT_COUNT])
    pins = [pin for pin in (expected_pilot_sha256, selection.get("pilot_ids_sha256")) if pin is not None]
    if not pins or any(pin != pilot_sha for pin in pins):
        raise ValueError("frozen data: missing or different pinned pilot300 prefix digest")
    sources = Counter(included[pid]["source"] for pid in ordered_ids)
    for name, caps in (("source_train_counts", False), ("source_caps", True)):
        if name in selection:
            _counts(selection[name], sources, name, caps=caps)
    clusters: Counter = Counter()
    pilot_clusters: Counter = Counter()
    if selected_categories is not None or "cluster_train_counts" in selection or "cluster_pilot_counts" in selection:
        categories = manifest.get("category_map", {})
        for index, item in enumerate(order["items"]):
            row = included[item["problem_id"]]
            cluster = (selected_categories[row["problem_id"]] if selected_categories is not None
                       else categories.get(row.get("subject")))
            if (type(cluster) is not str or not cluster or row.get("category") != cluster
                    or item.get("cluster", cluster if selected_categories is None else None) != cluster):
                origin = "bound per-ID selection" if selected_categories is not None else "manifest category_map"
                raise ValueError(f"frozen data: cluster does not match the {origin}")
            clusters[cluster] += 1
            if index < PILOT_COUNT:
                pilot_clusters[cluster] += 1
        for name, actual in (("cluster_train_counts", clusters), ("cluster_pilot_counts", pilot_clusters)):
            if name in selection:
                _counts(selection[name], actual, name)
    return verification | {"order_sha256": order_sha256, "ordered_ids_sha256": order["ids_sha256"],
                           "pilot_ids_sha256": pilot_sha, "train_count": TRAIN_COUNT, "pilot_count": PILOT_COUNT,
                           "main_count": TRAIN_COUNT - PILOT_COUNT, "source_train_counts": dict(sources),
                           "cluster_train_counts": dict(clusters), "cluster_pilot_counts": dict(pilot_clusters)}
