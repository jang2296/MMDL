"""Decide the proof dataset locally, item by item; the pod only re-hashes.

  freeze       reviewed shards (an hf_pool build whose contamination/receipt_v2.json is an adjudicated
               exclusion receipt) -> dataset_manifest.json with one status per shard item + summary.{json,md};
               optional --quality-exclusions (quality-exclusions-v1) marks listed items EXCLUDED_QUALITY_FILTER
               optional --selection/--selection-sha256 pins explicit IDs and per-ID C1..C8 categories;
               selection bytes must be dataset._bytes(body) + b"\n": sorted compact UTF-8 JSON and exactly one LF.
  materialize  manifest + the same shards (+ each shard's source registry and scan evidence) -> one
               candidate-build vault of INCLUDED/HELD_OUT items, byte-identical records and images, bound by
               freeze_binding.json, with contamination/union_receipt.json binding every shard receipt
  verify       manifest + vault -> re-hash every record line and image; exit 1 on any mismatch/extra/missing

Exclusion decisions are re-derived from the shard evidence (adjudication.validate re-opens the raw
receipt and review decisions); nothing is re-scanned. The union receipt is no new exclusion decision:
consumers (approval, pool_data_evidence) re-verify each shard receipt against its own shard and accept
only items allowlisted there; materialize refuses anything else and shards that mix index, calibration
or detector identities. The train quota is label-blind: per subject the
order is sha256(f"{seed}:{source}:{native_id}") (hf_common.candidate_order's key). The manifest
holds ids, hashes and metadata only: no question text, choices or gold. MMMU/MMMU-Pro appear only
through the receipts' exclusion index hashes. AI review stays human_verified=false; nothing here
grants training permission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import adjudication, approval, vault
from . import contamination as c
from .dataset import SPLITS, _bytes, _write, build_splits
from .generators.common import GeneratedProblem
from .hf_pool import ITEM_INDEX_VERSION
from .schema import ActorItem, Lineage, VerifierItem, from_json, load_protocol

VERSION = "pool-freeze-manifest-v1"
SELECTION_VERSION = "pool-freeze-selection-v1"
SELECTION_ORDER = "explicit-selected-ids-v1"
_SELECTION_SHARD_FIELDS = ("shard", "build_manifest_sha256", "exclusion_receipt_file_sha256", "item_index_sha256")
BINDING_VERSION = "pool-freeze-vault-binding-v1"
UNION_VERSION = "pool-freeze-union-receipt-v1"
UNION_STATUS = "EXCLUSION_CHECKED_WITH_LIMITS_UNION"
UNION_FILE = approval.UNION_RECEIPT
_IDENTITY = ("index_sha256", "index_file_sha256", "protocol_sha256", "calibration_sha256", "scanner_policy_sha256",
             "detectors", "scanner_code_sha256")
INCLUDED, DEV, SEALED = "INCLUDED", "HELD_OUT_DEV", "HELD_OUT_SEALED"
QUOTA, BLOCKED = "EXCLUDED_QUOTA", "EXCLUDED_SOURCE_BLOCKED"
DUPLICATE, UNRESOLVED = "EXCLUDED_CONTAMINATION_DUPLICATE", "EXCLUDED_CONTAMINATION_UNRESOLVED"
NONRELEASABLE = "EXCLUDED_NONRELEASABLE"
STATUSES = (INCLUDED, DEV, SEALED, QUOTA, BLOCKED, DUPLICATE, UNRESOLVED, NONRELEASABLE)
KEPT = (INCLUDED, DEV, SEALED)
# Opt-in program quality filter: after the contamination statuses, before source blocks, held-out splits and quota.
# A manifest frozen without --quality-exclusions keeps the statuses list and bytes above.
QUALITY_FILTER = "EXCLUDED_QUALITY_FILTER"
QUALITY_VERSION = "quality-exclusions-v1"
_CONTAMINATION = (DUPLICATE, UNRESOLVED, NONRELEASABLE)
_RECORDS = (("actor", "actor_inputs.jsonl"), ("verifier", "verifier_private.jsonl"), ("lineage", "lineage.jsonl"))
_SPLIT_FILES = {name for _, name in _RECORDS} | {"manifest.yaml"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path) -> tuple[dict, str]:
    raw = vault._read(Path(path))
    value = json.loads(raw, object_pairs_hook=vault._unique)
    if type(value) is not dict:
        raise ValueError(f"freeze: {path} must hold a JSON object")
    return value, _sha(raw)


def _lines(path: Path) -> dict[str, bytes]:
    """problem_id -> exact JSONL line bytes (no trailing newline); the record hash is sha256 of these bytes."""
    raw = vault._read(path)
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"freeze: {path} lacks a trailing newline")
    out: dict[str, bytes] = {}
    for line in raw.split(b"\n")[:-1]:
        pid = json.loads(line)["problem_id"]
        if type(pid) is not str or pid in out:
            raise ValueError(f"freeze: {path} has a duplicate or invalid problem_id")
        out[pid] = line
    return out


def _rel(rel: str) -> str:
    if type(rel) is not str or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError(f"freeze: unsafe image path {rel!r}")
    return rel


def _item_index(root: Path) -> tuple[dict, str]:
    """<shard>/item_index.json, else beside the source build named by hf_pool_exclusion's assemble receipt."""
    path = root / "item_index.json"
    if not path.exists():
        try:
            prior, _ = _load(root.parent / "receipt.json")
            path = Path(prior["inputs"]["build_manifest"]["path"]).parent / "item_index.json"
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"freeze: {root} has no item_index.json (nor an assemble receipt naming its build)") from None
        if prior.get("stage") != "assemble":
            raise ValueError(f"freeze: {root.parent}/receipt.json is not an assemble receipt")
    return _load(path)


def _quality_blocked(path: Path, root: Path, items: dict, receipt: dict) -> tuple[dict[str, list[str]], str]:
    """approval's own ai_reviewed_local rule: failing verdict, unresolved leak or short sample blocks the source.
    A v2 (acceptance-sampling) review that leaves a source eligible but makes items ineligible is refused."""
    review, sha = _load(path)
    warnings = {split: _load(root / split / "manifest.yaml")[0]["leak_review"] for split in SPLITS}
    seed = load_protocol(Path(__file__).with_name("protocol_v1.yaml"))["seeds"][0]
    try:
        blocked, _, quality = approval._ai_local_review(review, review.get("reviewer_id"), items, warnings,
                                                        receipt["input_files_sha256"], seed, set(receipt["allowlist"]))
    except approval._Blocked as exc:
        raise ValueError(f"freeze: quality review {path} rejected ({exc})") from None
    # No per-item status for an acceptance failure here: those items would otherwise stay in the pool.
    failed = sorted(pid for source, row in (quality or {}).get("sources", {}).items() if source not in blocked
                    for pid in row["failed_ids"])
    if failed:
        raise ValueError(f"freeze: quality review {path} makes items {failed[:5]} ineligible; review the materialized "
                         "vault instead, or list them in --quality-exclusions")
    return blocked, sha


def _approval_blocked(path: Path, build_sha: str, receipt_sha: str, review_sha: str | None,
                      sources: dict[str, str], allowlist: list[str]) -> tuple[dict[str, list[str]], str]:
    value, sha = _load(path)
    binding, eligibility = value.get("input_binding"), value.get("eligibility")
    if (value.get("status") != "VALIDATED_FOR_MEASUREMENT" or value.get("stage") != approval.AI_LOCAL_STAGE
            or value.get("human_verified") is not False or type(binding) is not dict or type(eligibility) is not dict
            or binding.get("build_manifest_sha256") != build_sha
            or binding.get("exclusion_receipt_sha256") != receipt_sha
            or (review_sha is not None and binding.get("ai_review_file_sha256") != review_sha)
            or type(eligibility.get("blocked_sources")) is not dict or type(eligibility.get("eligible_ids")) is not list):
        raise ValueError(f"freeze: approval {path} is not a validated ai_reviewed_local result for this shard")
    blocked = eligibility["blocked_sources"]
    if any(type(reasons) is not list for reasons in blocked.values()):
        raise ValueError(f"freeze: approval {path} blocked_sources must map source -> [reason]")
    if sorted(eligibility["eligible_ids"]) != sorted(pid for pid in allowlist if sources[pid] not in blocked):
        raise ValueError(f"freeze: approval {path} eligibility disagrees with the reviewed receipt")
    return blocked, sha


def _quality_exclusions(path: Path, known: set[str]) -> dict:
    """quality-exclusions-v1 {version, rules: [{id, description, code_sha256 | script_sha256}],
    items: [{problem_id, rule_id}]} -> manifest section {version, file_sha256, rules, items: {problem_id: [rule ids]}}.

    Refuses unknown problem ids (not a shard item) and unknown rule ids; the rule code hash is recorded, not run."""
    spec, sha = _load(path)
    rules, rows = spec.get("rules"), spec.get("items")
    if (set(spec) != {"version", "rules", "items"} or spec["version"] != QUALITY_VERSION
            or type(rules) is not list or not rules or type(rows) is not list
            or any(type(rule) is not dict or set(rule) not in ({"id", "description", "code_sha256"},
                                                                {"id", "description", "script_sha256"})
                   or type(rule["id"]) is not str or not rule["id"] or type(rule["description"]) is not str
                   or re.fullmatch(r"[0-9a-f]{64}", str(rule.get("code_sha256", rule.get("script_sha256")))) is None
                   for rule in rules)
            or len({rule["id"] for rule in rules}) != len(rules)
            or any(type(row) is not dict or set(row) != {"problem_id", "rule_id"}
                   or type(row["problem_id"]) is not str or type(row["rule_id"]) is not str for row in rows)):
        raise ValueError(f"freeze: {path} is not a {QUALITY_VERSION} file "
                         "{version, rules: [{id, description, code_sha256}], items: [{problem_id, rule_id}]}")
    unknown_rules = sorted({row["rule_id"] for row in rows} - {rule["id"] for rule in rules})
    unknown_items = sorted({row["problem_id"] for row in rows} - known)
    if unknown_rules or unknown_items:
        raise ValueError(f"freeze: {path} names unknown rule ids {unknown_rules[:5]} / problem ids {unknown_items[:5]}")
    pairs = sorted((row["problem_id"], row["rule_id"]) for row in rows)
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"freeze: {path} repeats a (problem_id, rule_id) pair")
    items: dict[str, list[str]] = defaultdict(list)
    for pid, rule_id in pairs:
        items[pid].append(rule_id)
    return {"version": QUALITY_VERSION, "file_sha256": sha, "rules": rules, "items": dict(items)}


def _reviewed(root: Path) -> tuple[dict, str, dict, str, dict, list[dict]]:
    """(manifest, its sha, adjudicated receipt, its file sha, items, effective candidates) of a shard's current bytes."""
    manifest, build_sha = _load(root / "manifest.yaml")
    receipt, receipt_sha = _load(root / "contamination" / "receipt_v2.json")
    if manifest.get("exclusion_receipt_sha256") != receipt_sha:
        raise ValueError(f"freeze: {root} manifest is not bound to its exclusion receipt")
    if type(receipt.get("adjudication")) is not dict:
        raise ValueError(f"freeze: {root} has no reviewed (adjudicated) exclusion receipt")
    items, files, _ = c._snapshot({split: root / split for split in SPLITS}, root / "images")
    if (receipt.get("version") != c.VERSION or receipt.get("status") != "EXCLUSION_CHECKED_WITH_LIMITS"
            or receipt.get("blockers") != [] or receipt.get("input_files") != files
            or receipt.get("input_files_sha256") != c._digest(files)
            or receipt.get("receipt_sha256") != c._digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or any(manifest.get("split_manifests", {}).get(s) != files[str(root / s / "manifest.yaml")] for s in SPLITS)):
        raise ValueError(f"freeze: {root} receipt is not a complete binding of the shard's current bytes")
    effective = adjudication.validate(receipt, items)    # re-opens raw receipt + review decisions, re-derives
    return manifest, build_sha, receipt, receipt_sha, items, effective


def _shard(number: int, root: Path, review_path: Path | None, approval_path: Path | None, *,
           reviewed: tuple | None = None) -> tuple[dict, list[dict]]:
    root = Path(root).absolute()
    manifest, build_sha, receipt, receipt_sha, items, effective = _reviewed(root) if reviewed is None else reviewed
    adj = receipt["adjudication"]
    by_id = {adjudication.candidate_id(candidate): candidate for candidate in effective}

    def family(cids: list[str]) -> set[str]:
        return set(c.family_quarantine(items, [by_id[cid] for cid in cids])["quarantined_ids"])

    duplicate, unresolved = family(adj["duplicate_candidate_ids"]), family(adj["unresolved_candidate_ids"])
    nonreleasable = family(adj["nonreleasable_candidate_ids"])
    if set(items) - set(receipt["allowlist"]) != duplicate | unresolved | nonreleasable:
        raise ValueError(f"freeze: {root} quarantine is not explained by the adjudication")

    index, index_sha = _item_index(root)
    rows = index.get("items")
    if (index.get("version") != ITEM_INDEX_VERSION or index.get("split_manifests") != manifest["split_manifests"]
            or type(rows) is not dict or set(rows) != set(items)):
        raise ValueError(f"freeze: {root} item_index does not cover exactly this shard")
    sources = {pid: item["actor"].source for pid, item in items.items()}
    blocked: dict[str, dict[str, list[str]]] = defaultdict(dict)
    review_sha = approval_sha = None
    if review_path is not None:
        found, review_sha = _quality_blocked(review_path, root, items, receipt)
        for source, reasons in found.items():
            blocked[source]["quality_review"] = reasons
    if approval_path is not None:
        found, approval_sha = _approval_blocked(approval_path, build_sha, receipt_sha, review_sha, sources,
                                                receipt["allowlist"])
        for source, reasons in found.items():
            blocked[source]["approval"] = reasons
    evidence = {"exclusion_receipt_sha256": receipt_sha, "adjudication_review_sha256": adj["review"]["file_sha256"],
                "quality_review_sha256": review_sha, "approval_sha256": approval_sha}
    lines = {kind: {pid: _sha(line) for split in SPLITS for pid, line in _lines(root / split / name).items()}
             for kind, name in _RECORDS}
    records = []
    for pid, item in sorted(items.items()):
        row, actor, lineage = rows[pid], item["actor"], item["lineage"]
        if (type(row) is not dict or row.get("source") != actor.source or row.get("split") != item["split"]
                or row.get("family") != actor.family or row.get("image_sha256") != list(actor.image_sha256)
                or row.get("source_revision") != lineage.source_revision
                or any(type(row.get(key)) is not str or not row[key] for key in ("native_id", "subject"))):
            raise ValueError(f"freeze: {root} item_index row for {pid} disagrees with the vault")
        status = (DUPLICATE if pid in duplicate else NONRELEASABLE if pid in nonreleasable
                  else UNRESOLVED if pid in unresolved else BLOCKED if actor.source in blocked
                  else DEV if item["split"] == "dev" else SEALED if item["split"] != "train" else None)
        records.append({"problem_id": pid, "shard": number, "status": status, "source": actor.source,
                        "source_revision": lineage.source_revision, "native_id": row["native_id"],
                        "subject": row["subject"], "split": item["split"], "family": actor.family,
                        "plan_version": row.get("plan_version"), "adapter_code_sha256": row.get("adapter_code_sha256"),
                        "image_paths": list(actor.image_paths), "image_sha256": list(actor.image_sha256),
                        "pixel_sha256": list(item["pixel_sha256"]),
                        "record_sha256": {kind: lines[kind][pid] for kind, _ in _RECORDS}, "evidence": evidence})
    meta = {"shard": number, "root": str(root), "build_manifest_sha256": build_sha,
            "split_manifests": manifest["split_manifests"], "item_index_sha256": index_sha,
            "exclusion_receipt_file_sha256": receipt_sha, "receipt_sha256": receipt["receipt_sha256"],
            "index_sha256": receipt.get("index_sha256"), "index_file_sha256": receipt.get("index_file_sha256"),
            "detectors": sorted(receipt.get("steps", {})), "limitations": receipt.get("limitations", []),
            "adjudication_counts": adj["counts"], "adjudication_review_sha256": evidence["adjudication_review_sha256"],
            "adjudication_human_verified": adj.get("human_verified"),
            "quality_review_sha256": review_sha, "approval_sha256": approval_sha,
            "blocked_sources": {source: dict(sorted(v.items())) for source, v in sorted(blocked.items())}}
    return meta, records


def _order(seed: int, record: dict) -> tuple[str, str]:
    return _sha(f"{seed}:{record['source']}:{record['native_id']}".encode()), record["problem_id"]


def _selection_items(document: dict, shards: list[dict]) -> dict[str, str]:
    """Validate the selector's externally pinned decision against the reviewed shard identities."""
    if (set(document) != {"version", "shards", "items"} or document.get("version") != SELECTION_VERSION
            or type(document.get("shards")) is not list or type(document.get("items")) is not list
            or not document["items"]):
        raise ValueError(f"freeze: invalid {SELECTION_VERSION} receipt")
    bindings = document["shards"]
    if (len(bindings) != len(shards)
            or any(type(binding) is not dict or set(binding) != set(_SELECTION_SHARD_FIELDS)
                   or type(binding.get("shard")) is not int
                   or any(re.fullmatch(r"[0-9a-f]{64}", str(binding.get(key))) is None
                          for key in _SELECTION_SHARD_FIELDS[1:]) for binding in bindings)
            or any(binding != {key: meta.get(key) for key in _SELECTION_SHARD_FIELDS}
                   for binding, meta in zip(bindings, shards, strict=True))):
        raise ValueError("freeze: selection reviewed shard bindings differ")
    rows = document["items"]
    if any(type(row) is not dict or set(row) != {"problem_id", "category"}
           or type(row.get("problem_id")) is not str or not row["problem_id"]
           or type(row.get("category")) is not str or re.fullmatch(r"C[1-8]", row["category"]) is None for row in rows):
        raise ValueError("freeze: selection items require problem_id and category C1..C8")
    selected = {row["problem_id"]: row["category"] for row in rows}
    if len(selected) != len(rows):
        raise ValueError("freeze: duplicate selected problem_id")
    return selected


def _exact_selection_quota(rows: list[dict], quotas: dict) -> None:
    if (type(quotas) is not dict or any(type(subject) is not str or type(count) is not int or count < 0
                                      for subject, count in quotas.items())
            or dict(Counter(row["subject"] for row in rows)) != {subject: count for subject, count in quotas.items() if count}):
        raise ValueError("freeze: explicit selection does not match exact subject quota (shortfall or excess)")


def selection_categories(manifest: dict) -> dict[str, str] | None:
    """Read only embedded, manifest-bound selection metadata; no source shard, receipt or benchmark access."""
    if "selection" not in manifest:
        if manifest.get("quota_order") == SELECTION_ORDER:
            raise ValueError("freeze: explicit selection marker requires its embedded receipt")
        return None
    selection = manifest["selection"]
    if (type(selection) is not dict or set(selection) != {"version", "shards", "items", "file_sha256"}
            or re.fullmatch(r"[0-9a-f]{64}", str(selection.get("file_sha256"))) is None
            or manifest.get("quota_order") != SELECTION_ORDER):
        raise ValueError("freeze: invalid embedded selection binding")
    document = {key: value for key, value in selection.items() if key != "file_sha256"}
    if _sha(_bytes(document) + b"\n") != selection["file_sha256"]:
        raise ValueError("freeze: embedded selection SHA256 differs from the originally pinned canonical receipt")
    selected = _selection_items(document, manifest["shards"])
    rows = manifest["items"]
    if any(type(row) is not dict or type(row.get("problem_id")) is not str for row in rows):
        raise ValueError("freeze: invalid selected manifest item")
    by_id = {row["problem_id"]: row for row in rows}
    included = [row for row in rows if row["status"] == INCLUDED]
    if len(by_id) != len(rows) or set(selected) != {row["problem_id"] for row in included}:
        raise ValueError("freeze: selected IDs differ from INCLUDED manifest items")
    if any(by_id[pid]["split"] != "train" or by_id[pid].get("category") != category for pid, category in selected.items()):
        raise ValueError("freeze: selected manifest split/category differs from the bound receipt")
    _exact_selection_quota(included, manifest["quota"])
    return selected


def freeze(shards: list[Path], reviews: list[Path | None], approvals: list[Path | None], category_map: Path,
           quota: Path, seed: int, output: Path, quality_exclusions: Path | None = None, *,
           selection: Path | None = None, selection_sha256: str | None = None) -> dict:
    if not shards or len(reviews) != len(shards) or (approvals and len(approvals) != len(shards)):
        raise ValueError("freeze: one --quality-review (path or none) per --shard, and none or one --approval per shard")
    if type(seed) is not int or seed < 0:
        raise ValueError("freeze: seed must be a nonnegative integer")
    if (selection is None) != (selection_sha256 is None):
        raise ValueError("freeze: selection path and expected selection SHA256 must be supplied together")
    selected_document = None
    if selection is not None:
        raw = vault._read(selection)
        selected_document = json.loads(raw, object_pairs_hook=vault._unique)
        if type(selected_document) is not dict or raw != _bytes(selected_document) + b"\n":
            raise ValueError("freeze: selection requires dataset._bytes(body) plus exactly one final LF (canonical JSON)")
        actual_sha = _sha(raw)
        if (re.fullmatch(r"[0-9a-f]{64}", str(selection_sha256)) is None or actual_sha != selection_sha256):
            raise ValueError("freeze: selection file SHA256 differs from its expected value")
    categories, categories_sha = _load(category_map)
    quotas, quota_sha = _load(quota)
    if any(type(k) is not str or type(v) is not str or not v for k, v in categories.items()):
        raise ValueError("freeze: category map must be {subject: category}")
    if any(k not in categories or type(v) is not int or v < 0 for k, v in quotas.items()):
        raise ValueError("freeze: quota must be {mapped subject: nonnegative int}")
    metas, records = [], []
    reviewed_shards = [_reviewed(Path(root).absolute()) for root in shards] if selected_document is not None else None
    for number, root in enumerate(shards):
        meta, rows = _shard(number, root, reviews[number], approvals[number] if approvals else None,
                            reviewed=reviewed_shards[number] if reviewed_shards is not None else None)
        metas.append(meta)
        records += rows
    owners: dict[str, int] = {}
    for record in records:
        if record["subject"] not in categories:
            raise ValueError(f"freeze: subject {record['subject']!r} has no category")
        record["category"] = categories[record["subject"]]
        keys = [f"id:{record['problem_id']}", f"native:{record['source']}:{record['native_id']}",
                *(f"image:{x}" for x in record["image_sha256"]), *(f"pixel:{x}" for x in record["pixel_sha256"])]
        for key in keys:
            if owners.setdefault(key, record["shard"]) != record["shard"]:
                raise ValueError(f"freeze: shards overlap on {key}; rebuild the shards disjoint")
    quality = None
    if quality_exclusions is not None:
        quality = _quality_exclusions(quality_exclusions, {record["problem_id"] for record in records})
        for record in records:
            if record["problem_id"] in quality["items"] and record["status"] not in _CONTAMINATION:
                record["status"] = QUALITY_FILTER
    pending: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["status"] is None:
            pending[record["subject"]].append(record)
    if selected_document is None:
        for subject, rows in pending.items():
            for rank, record in enumerate(sorted(rows, key=lambda r: _order(seed, r))):
                record["status"] = INCLUDED if rank < quotas.get(subject, 0) else QUOTA
    else:
        selected = _selection_items(selected_document, metas)
        by_id = {record["problem_id"]: record for record in records}
        assert reviewed_shards is not None
        allowed = cross_shard_allowlist([(reviewed[4], _allowed(reviewed[2])) for reviewed in reviewed_shards])
        if any(pid not in by_id or by_id[pid]["status"] is not None or by_id[pid]["split"] != "train"
               or pid not in allowed for pid in selected):
            raise ValueError("freeze: selected IDs must be known eligible train items outside every quarantined family")
        _exact_selection_quota([by_id[pid] for pid in selected], quotas)
        for rows in pending.values():
            for record in rows:
                pid = record["problem_id"]
                record["status"] = INCLUDED if pid in selected else QUOTA
                if pid in selected:
                    record["category"] = selected[pid]
    records.sort(key=lambda r: r["problem_id"])
    manifest = {"version": VERSION, "status": "FROZEN_DATA_DECISION_NOT_TRAINING_APPROVAL", "training_allowed": False,
                "human_verified": False, "seed": seed, "statuses": list(STATUSES),
                "category_map": categories, "category_map_sha256": categories_sha,
                "quota": quotas, "quota_sha256": quota_sha, "quota_order": "sha256(f'{seed}:{source}:{native_id}')",
                "shards": metas, "counts": dict(Counter(r["status"] for r in records)), "items": records}
    if quality is not None:
        manifest.update(statuses=[*STATUSES, QUALITY_FILTER], quality_exclusions=quality)
    if selected_document is not None:
        manifest.update(selection={**selected_document, "file_sha256": selection_sha256}, quota_order=SELECTION_ORDER)
        selection_categories(manifest)
    output = Path(output)
    vault._safe_root(output.parent, create=True)
    output.mkdir(mode=0o700)
    _write(output / "dataset_manifest.json", _bytes(manifest) + b"\n")
    summary = _summary(manifest, _sha((output / "dataset_manifest.json").read_bytes()))
    _write(output / "summary.json", json.dumps(summary, sort_keys=True, indent=1).encode() + b"\n")
    _write(output / "summary.md", _markdown(summary).encode())
    return summary


def _summary(manifest: dict, manifest_sha: str) -> dict:
    items = manifest["items"]

    def table(key: str) -> dict:
        out: dict[str, Counter] = defaultdict(Counter)
        for r in items:
            out[r[key]][r["status"]] += 1
        return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}

    included = Counter(r["subject"] for r in items if r["status"] == INCLUDED)
    shards = [{k: m[k] for k in ("shard", "index_sha256", "index_file_sha256", "detectors", "adjudication_counts",
                                 "adjudication_human_verified", "limitations", "blocked_sources")} for m in manifest["shards"]]
    flagged = {s: sum(r["status"] == s for r in items) for s in (DUPLICATE, UNRESOLVED, NONRELEASABLE)}
    statement = (f"All {len(items)} shard items were scanned against the pinned MMMU/MMMU-Pro exclusion index "
                 f"({', '.join(sorted({str(m['index_sha256']) for m in shards}))}) with detectors "
                 f"{', '.join(sorted({d for m in shards for d in m['detectors']}))}. Flagged candidates were "
                 f"AI-adjudicated; duplicate, unresolved (incl. unreviewed) and non-releasable families are "
                 f"excluded ({flagged[DUPLICATE]} / {flagged[UNRESOLVED]} / {flagged[NONRELEASABLE]} items). "
                 "Benchmark data served only as an exclusion signal. human_verified=false.")
    summary = {"version": VERSION, "dataset_manifest_sha256": manifest_sha, "seed": manifest["seed"],
               "training_allowed": False, "human_verified": False, "totals": dict(sorted(manifest["counts"].items())),
               "items": len(items), "by_category": table("category"), "by_subject": table("subject"),
               "by_source": table("source"), "by_split": table("split"),
               "quota": {s: {"quota": q, "included": included[s], "shortfall": max(0, q - included[s])}
                         for s, q in sorted(manifest["quota"].items())},
               "contamination": {"statement": statement, "human_verified": False, "shards": shards}}
    quality = manifest.get("quality_exclusions")
    if quality is not None:
        status = {r["problem_id"]: r["status"] for r in items}
        summary["quality_exclusions"] = {"version": quality["version"], "file_sha256": quality["file_sha256"],
            "excluded_items": sum(s == QUALITY_FILTER for s in status.values()),
            "rules": [{**rule, "listed": sum(rule["id"] in ids for ids in quality["items"].values()),
                       "excluded": sum(rule["id"] in ids and status[pid] == QUALITY_FILTER
                                       for pid, ids in quality["items"].items())} for rule in quality["rules"]]}
    return summary


def _markdown(summary: dict) -> str:
    quality = summary.get("quality_exclusions")
    statuses = STATUSES if quality is None else (*STATUSES, QUALITY_FILTER)

    def table(title: str, rows: dict) -> list[str]:
        head = [title, *statuses, "total"]
        out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for name, counts in rows.items():
            out.append("| " + " | ".join([name, *(str(counts.get(s, 0)) for s in statuses), str(sum(counts.values()))]) + " |")
        return [*out, ""]

    lines = ["# Frozen proof dataset", "", f"dataset_manifest.json sha256 `{summary['dataset_manifest_sha256']}`, "
             f"seed {summary['seed']}, {summary['items']} items. training_allowed=false, human_verified=false.", "",
             "## Totals", "", *table("all", {"all": summary["totals"]}),
             "## By category", "", *table("category", summary["by_category"]),
             "## By subject", "", *table("subject", summary["by_subject"]),
             "## By source", "", *table("source", summary["by_source"]),
             "## Train quota", "", "| subject | quota | included | shortfall |", "|---|---|---|---|",
             *(f"| {s} | {q['quota']} | {q['included']} | {q['shortfall']} |" for s, q in summary["quota"].items()),
             "", "## MMMU/MMMU-Pro contamination", "", summary["contamination"]["statement"], ""]
    for shard in summary["contamination"]["shards"]:
        lines.append(f"- shard {shard['shard']}: index `{shard['index_sha256']}`, adjudication "
                     f"{json.dumps(shard['adjudication_counts'], sort_keys=True)}, blocked sources "
                     f"{sorted(shard['blocked_sources']) or 'none'}")
    if quality is not None:
        lines += ["", "## Quality exclusions", "", f"{QUALITY_VERSION} file sha256 `{quality['file_sha256']}`, "
                  f"{quality['excluded_items']} items {QUALITY_FILTER}.", "",
                  "| rule | code sha256 | listed | excluded | description |", "|---|---|---|---|---|",
                  *(f"| {r['id']} | `{r.get('code_sha256', r.get('script_sha256'))}` | {r['listed']} | {r['excluded']} | "
                    f"{r['description']} |" for r in quality["rules"])]
    return "\n".join(lines) + "\n"


def _manifest(path: Path) -> tuple[dict, str]:
    manifest, sha = _load(path)
    if manifest.get("version") != VERSION or type(manifest.get("items")) is not list:
        raise ValueError(f"freeze: {path} is not a {VERSION} manifest")
    selection_categories(manifest)
    return manifest, sha


def _allowed(receipt: dict) -> set[str]:
    """The only materializable ids of a re-derived shard receipt: allowlisted and not quarantined."""
    return set(receipt["allowlist"]) - set(receipt["quarantined_ids"])


def cross_shard_allowlist(shards: list[tuple[dict, set[str]]]) -> set[str]:
    """[(shard items, shard allowlist)] -> ids outside every family (structure/image/pixel) that holds an id
    quarantined in any shard: a family spanning shards stays quarantined in all of them. Never releases."""
    merged = {pid: item for items, _ in shards for pid, item in items.items()}
    if len(merged) != sum(len(items) for items, _ in shards):
        raise ValueError("union: shards share a problem id")
    seeds = [{"problem_id": pid} for items, allowed in shards for pid in sorted(set(items) - allowed)]
    return set(c.family_quarantine(merged, seeds)["allowlist"])


def _union_entry(meta: dict, root: Path, registry_path: Path, evidence_path: Path) -> tuple[dict, dict, set[str]]:
    """A shard's re-derived adjudicated receipt + the registry/evidence it was scanned with -> (union entry, items,
    allowed)."""
    _, build_sha, receipt, receipt_sha, items, _ = _reviewed(root)
    if build_sha != meta["build_manifest_sha256"] or receipt_sha != meta["exclusion_receipt_file_sha256"]:
        raise ValueError(f"materialize: {root} is not shard {meta['shard']} of the manifest")
    registry, registry_sha = _load(registry_path)
    evidence, evidence_sha = _load(evidence_path)
    if (approval._hash(registry) != receipt.get("source_registry_sha256")
            or approval._hash(evidence) != receipt.get("evidence_sha256")):
        raise ValueError(f"materialize: shard {meta['shard']} receipt was not scanned with this registry/evidence")
    c._calibration(evidence, registry, receipt.get("index_sha256"), receipt.get("protocol_sha256"))
    adj = receipt["adjudication"]
    entry = {"shard": meta["shard"], "root": str(root), "build_manifest_sha256": build_sha,
             "receipt": {"path": str(root / "contamination" / "receipt_v2.json"), "file_sha256": receipt_sha},
             "receipt_sha256": receipt["receipt_sha256"],
             "source_registry": {"path": str(Path(registry_path).absolute()), "file_sha256": registry_sha},
             "evidence": {"path": str(Path(evidence_path).absolute()), "file_sha256": evidence_sha},
             "index_sha256": receipt.get("index_sha256"), "index_file_sha256": receipt.get("index_file_sha256"),
             "protocol_sha256": receipt.get("protocol_sha256"),
             "calibration_sha256": approval._hash(evidence["calibration"]),
             "scanner_policy_sha256": approval._hash(registry["scanner_policy"]),
             "detectors": sorted(receipt.get("steps", {})),
             "scanner_code_sha256": {Path(p).name: d for p, d in sorted(receipt.get("code_sha256", {}).items())},
             "source_registry_sha256": receipt["source_registry_sha256"], "evidence_sha256": receipt["evidence_sha256"],
             "input_files_sha256": receipt["input_files_sha256"], "adjudication_counts": adj["counts"],
             "adjudication_review_sha256": adj["review"]["file_sha256"], "adjudication_human_verified": False,
             "allowlisted": len(receipt["allowlist"]), "quarantined": len(receipt["quarantined_ids"])}
    return entry, items, _allowed(receipt)


def materialize(manifest_path: Path, shards: list[Path], output: Path, registries: list[Path],
                evidences: list[Path], *, allow_empty_heldout: bool = False) -> dict:
    manifest, manifest_sha = _manifest(manifest_path)
    if not len(shards) == len(registries) == len(evidences) == len(manifest["shards"]):
        raise ValueError("materialize: give exactly the manifest's shards, in order, each with its registry and evidence")
    roots = [Path(root).absolute() for root in shards]
    bound = [_union_entry(meta, root, registry, evidence)
             for meta, root, registry, evidence in zip(manifest["shards"], roots, registries, evidences, strict=True)]
    entries = [entry for entry, _, _ in bound]
    mixed = [key for key in _IDENTITY if any(entry[key] != entries[0][key] for entry in entries)]
    if mixed:
        raise ValueError(f"materialize: shards mix {', '.join(mixed)}; one index/calibration/detector identity required")
    cross = cross_shard_allowlist([(items, allowed) for _, items, allowed in bound])
    pools: dict[str, list[GeneratedProblem]] = {split: [] for split in SPLITS}
    owners = {}
    for meta, root, (_, _, allowed) in zip(manifest["shards"], roots, bound, strict=True):
        lines = {kind: {pid: line for split in SPLITS for pid, line in _lines(root / split / name).items()}
                 for kind, name in _RECORDS}
        for r in manifest["items"]:
            if r["shard"] != meta["shard"] or r["status"] not in KEPT:
                continue
            pid = r["problem_id"]
            if pid not in allowed:
                raise ValueError(f"materialize: {pid} is not allowlisted in shard {meta['shard']}'s reviewed receipt")
            if pid not in cross:
                raise ValueError(f"materialize: {pid} shares a family with an item quarantined in another shard")
            owners[pid] = {"shard": meta["shard"], "shard_allowlisted": True}
            raw = {kind: lines[kind].get(pid, b"") for kind, _ in _RECORDS}
            if any(_sha(raw[kind]) != r["record_sha256"][kind] for kind in raw):
                raise ValueError(f"materialize: record hash mismatch for {pid}")
            actor = from_json(ActorItem, raw["actor"].decode())
            verifier = from_json(VerifierItem, raw["verifier"].decode())
            lineage = from_json(Lineage, raw["lineage"].decode())
            images = tuple(vault._read(root / "images" / _rel(rel)) for rel in r["image_paths"])
            if ([_sha(x) for x in images] != r["image_sha256"] or list(actor.image_paths) != r["image_paths"]
                    or lineage.split != r["split"]):
                raise ValueError(f"materialize: image hash or identity mismatch for {pid}")
            pools[r["split"]].append(GeneratedProblem(actor, verifier, lineage, images))
    build_splits(output, pools, allow_empty_heldout=allow_empty_heldout)
    output = Path(output).absolute()
    binding = {"version": BINDING_VERSION, "dataset_manifest_sha256": manifest_sha,
               "vault_manifest_sha256": _sha(vault._read(output / "manifest.yaml")),
               "counts": {split: len(rows) for split, rows in pools.items()}, "training_allowed": False}
    _write(output / "freeze_binding.json", _bytes(binding) + b"\n")
    union = {"version": UNION_VERSION, "status": UNION_STATUS, "training_allowed": False, "approval_granted": False,
             "human_verified": False,
             "dataset_manifest": {"path": str(Path(manifest_path).absolute()), "file_sha256": manifest_sha},
             "vault_manifest_sha256": binding["vault_manifest_sha256"],
             "freeze_binding_sha256": _sha(vault._read(output / "freeze_binding.json")),
             **{key: entries[0][key] for key in _IDENTITY},
             "source_registry_sha256": approval._hash([e["source_registry_sha256"] for e in entries]),
             "evidence_sha256": approval._hash([e["evidence_sha256"] for e in entries]),
             "shards": entries, "items": dict(sorted(owners.items())),
             "limitations": ["Union of per-shard exclusion receipts: each is re-verified against its own shard, "
                             "nothing is re-scanned; only items allowlisted in their own shard are present.",
                             "Adjudication is AI review; human_verified=false. Not a training approval."]}
    union["union_receipt_sha256"] = approval._hash(union)
    _write(output / "contamination" / UNION_FILE, _bytes(union) + b"\n")
    read_union(output)
    errors = verify(manifest_path, output)
    if errors:
        raise ValueError(f"materialize: written vault fails verification: {errors[:5]}")
    return binding | {"union_receipt_sha256": _sha(vault._read(output / "contamination" / UNION_FILE))}


def read_union(vault_root: Path) -> tuple[dict, dict[str, str]]:
    """A materialized vault's union receipt after its structural checks -> (union, {path: file sha256 read}).

    Grants nothing by itself: callers re-verify each shard receipt against its own shard (the single-receipt
    functions) and the vault bytes with verify().
    """
    root = Path(vault_root).absolute()
    paths = {"union": root / "contamination" / UNION_FILE, "placeholder": root / "contamination" / "receipt_v2.json",
             "vault": root / "manifest.yaml", "binding": root / "freeze_binding.json"}
    loaded = {name: _load(path) for name, path in paths.items()}
    union, placeholder, vault_manifest, binding = (loaded[name][0] for name in paths)
    descriptor = union.get("dataset_manifest")
    if type(descriptor) is not dict or set(descriptor) != {"path", "file_sha256"} or type(descriptor["path"]) is not str:
        raise ValueError(f"union: {root} names no dataset manifest")
    manifest, manifest_sha = _manifest(Path(descriptor["path"]))
    kept = {r["problem_id"]: {"shard": r["shard"], "shard_allowlisted": True} for r in manifest["items"] if r["status"] in KEPT}
    shards = union.get("shards")
    if (union.get("version") != UNION_VERSION or union.get("status") != UNION_STATUS
            or union.get("training_allowed") is not False or union.get("approval_granted") is not False
            or union.get("human_verified") is not False
            or union.get("union_receipt_sha256") != approval._hash({k: v for k, v in union.items()
                                                                   if k != "union_receipt_sha256"})
            or manifest_sha != descriptor["file_sha256"]
            or union.get("vault_manifest_sha256") != loaded["vault"][1]
            or binding.get("vault_manifest_sha256") != loaded["vault"][1]
            or union.get("freeze_binding_sha256") != loaded["binding"][1]
            or binding.get("dataset_manifest_sha256") != manifest_sha
            or vault_manifest.get("exclusion_receipt_sha256") != loaded["placeholder"][1]
            or placeholder.get("status") != "BLOCKED_NOT_CHECKED" or placeholder.get("allowlist") != []
            or union.get("items") != kept or type(shards) is not list or len(shards) != len(manifest["shards"])
            or any(type(s) is not dict for s in shards)
            or union.get("source_registry_sha256") != approval._hash([s.get("source_registry_sha256") for s in shards])
            or union.get("evidence_sha256") != approval._hash([s.get("evidence_sha256") for s in shards])):
        raise ValueError(f"union: {root} union receipt does not bind this vault and its frozen manifest")
    for number, (entry, meta) in enumerate(zip(shards, manifest["shards"], strict=True)):
        descriptors = [entry.get(key) for key in ("receipt", "source_registry", "evidence")]
        if (any(type(d) is not dict or set(d) != {"path", "file_sha256"} or type(d["path"]) is not str
                or not Path(d["path"]).is_absolute() for d in descriptors)
                or not entry.get("shard") == meta["shard"] == number or type(entry.get("root")) is not str
                or entry.get("build_manifest_sha256") != meta["build_manifest_sha256"]
                or entry["receipt"] != {"path": str(Path(entry["root"]) / "contamination" / "receipt_v2.json"),
                                        "file_sha256": meta["exclusion_receipt_file_sha256"]}
                or any(entry.get(key) != union.get(key) for key in _IDENTITY)):
            raise ValueError(f"union: shard {meta['shard']} entry does not match the frozen manifest or mixes identities")
    read = {str(paths[name]): loaded[name][1] for name in paths}
    return union, read | {str(Path(descriptor["path"]).absolute()): manifest_sha}


def verify(manifest_path: Path, vault_root: Path) -> list[str]:
    """Every mismatch, extra or missing record/image; empty means the vault is exactly the manifest's kept items."""
    manifest, manifest_sha = _manifest(manifest_path)
    root = Path(vault_root)
    errors = []
    try:
        binding, _ = _load(root / "freeze_binding.json")
        if (binding.get("dataset_manifest_sha256") != manifest_sha
                or binding.get("vault_manifest_sha256") != _sha(vault._read(root / "manifest.yaml"))):
            errors.append("binding: freeze_binding.json does not bind this manifest and vault manifest.yaml")
    except ValueError as exc:
        errors.append(f"binding: {exc}")
    kept = {r["problem_id"]: r for r in manifest["items"] if r["status"] in KEPT}
    for split in SPLITS:
        directory = root / split
        present = {p.name for p in directory.iterdir()} if directory.is_dir() else set()
        if present != _SPLIT_FILES:
            errors.append(f"{split}: files {sorted(present)} != {sorted(_SPLIT_FILES)}")
        expected = {pid for pid, r in kept.items() if r["split"] == split}
        for kind, name in _RECORDS:
            try:
                found = {pid: _sha(line) for pid, line in _lines(directory / name).items()}
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{split}/{name}: unreadable ({exc})")
                continue
            errors += [f"{split}/{name}: extra record {pid}" for pid in sorted(found.keys() - expected)]
            errors += [f"{split}/{name}: missing record {pid}" for pid in sorted(expected - found.keys())]
            errors += [f"{split}/{name}: hash mismatch {pid}" for pid in sorted(expected & found.keys())
                       if found[pid] != kept[pid]["record_sha256"][kind]]
    images = {_rel(rel): sha for r in kept.values() for rel, sha in zip(r["image_paths"], r["image_sha256"], strict=True)}
    image_root = root / "images"
    present = {p.relative_to(image_root).as_posix() for p in image_root.rglob("*") if not p.is_dir()} \
        if image_root.is_dir() else set()
    errors += [f"images: extra file {rel}" for rel in sorted(present - images.keys())]
    errors += [f"images: missing file {rel}" for rel in sorted(images.keys() - present)]
    for rel in sorted(images.keys() & present):
        try:
            if _sha(vault._read(image_root / rel)) != images[rel]:
                errors.append(f"images: hash mismatch {rel}")
        except ValueError as exc:
            errors.append(f"images: {rel} unreadable ({exc})")
    return errors


def _optional(value: str) -> Path | None:
    return None if value == "none" else Path(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fine_tuning.notebook_rl.pool_freeze")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--shard", type=Path, action="append", required=True,
                   help="build-reviewed root with an adjudicated contamination/receipt_v2.json (repeat)")
    f.add_argument("--quality-review", type=_optional, action="append", required=True,
                   help="ai-reviewed-local-review-v1 JSON or 'none', one per --shard in order")
    f.add_argument("--approval", type=_optional, action="append", default=[],
                   help="validated ai_reviewed_local approval output JSON or 'none', one per --shard (optional)")
    f.add_argument("--category-map", type=Path, required=True, help="JSON {subject: category}")
    f.add_argument("--quota", type=Path, required=True, help="JSON {subject: train target}")
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--quality-exclusions", type=Path, default=None,
                   help=f"{QUALITY_VERSION} JSON: listed items become {QUALITY_FILTER} (optional)")
    f.add_argument("--selection", type=Path, help=f"opt-in {SELECTION_VERSION} explicit eligible train IDs/categories")
    f.add_argument("--selection-sha256", help="externally pinned SHA256 of --selection (both flags required)")
    m = sub.add_parser("materialize")
    m.add_argument("--manifest", type=Path, required=True)
    m.add_argument("--shard", type=Path, action="append", required=True)
    m.add_argument("--source-registry", type=Path, action="append", required=True,
                   help="the source_registry.json each shard was scanned with, one per --shard in order")
    m.add_argument("--evidence", type=Path, action="append", required=True,
                   help="the evidence.json each shard was scanned with, one per --shard in order")
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--allow-empty-heldout", action="store_true",
                   help="explicit opt-in: dev/sealed/style_sealed may be empty (recorded in the vault manifest)")
    v = sub.add_parser("verify")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--vault", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        summary = freeze(args.shard, args.quality_review, args.approval, args.category_map, args.quota,
                         args.seed, args.output, args.quality_exclusions,
                         selection=args.selection, selection_sha256=args.selection_sha256)
        print(json.dumps({"status": "FROZEN", "totals": summary["totals"],
                          "dataset_manifest_sha256": summary["dataset_manifest_sha256"]}, sort_keys=True))
        return 0
    if args.command == "materialize":
        print(json.dumps({"status": "MATERIALIZED", **materialize(args.manifest, args.shard, args.output,
                                                                    args.source_registry, args.evidence,
                                                                    allow_empty_heldout=args.allow_empty_heldout)},
                         sort_keys=True))
        return 0
    errors = verify(args.manifest, args.vault)
    print(json.dumps({"status": "MISMATCH" if errors else "VERIFIED", "errors": len(errors),
                      "first_errors": errors[:50], "manifest_sha256": _manifest(args.manifest)[1]}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
