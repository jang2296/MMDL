"""Bind direct AI overlap reviews to complete scans; never repair missing scan gates.

Callers supply their already input-verified item inventory. Identical warning
objects share a canonical candidate ID and one decision, while the full original
warning list stays intact. This module creates no verdicts or training authority.
"""
from __future__ import annotations

import copy
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path

from . import contamination as c
from .vault import _read, _safe_root, _unique

FORMAT = "notebook-rl-exclusion-adjudication-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RELEASABLE = {"question_exact", "text_ngrams", "ocr_ngrams", "perceptual", "sscd", "siglip"}


def candidate_id(candidate: dict) -> str:
    """Stable content identity, including the exact similarity score/reference."""
    if type(candidate) is not dict:
        raise ValueError("adjudication: candidate object required")
    return c._digest(candidate)


def _binding(files: dict[str, str] | None, path: Path, digest: str) -> None:
    if files is not None:
        key = str(path.absolute())
        if key in files and files[key] != digest:
            raise ValueError("adjudication: inconsistent repeated file binding")
        files[key] = digest


def _artifact(descriptor: dict, files: dict[str, str] | None = None) -> dict:
    import json

    if (type(descriptor) is not dict or set(descriptor) != {"path", "file_sha256"}
            or not isinstance(descriptor["path"], str) or not Path(descriptor["path"]).is_absolute()
            or not isinstance(descriptor["file_sha256"], str) or not _SHA.fullmatch(descriptor["file_sha256"])):
        raise ValueError("adjudication: absolute path/SHA descriptor required")
    path = Path(descriptor["path"])
    _safe_root(path.parent)
    raw = _read(path)
    if hashlib.sha256(raw).hexdigest() != descriptor["file_sha256"]:
        raise ValueError("adjudication: referenced artifact changed")
    _binding(files, path, descriptor["file_sha256"])
    value = json.loads(raw, object_pairs_hook=_unique)
    if type(value) is not dict:
        raise ValueError("adjudication: object artifact required")
    return value


def _evidence(descriptor: dict, files: dict[str, str] | None = None) -> None:
    if (type(descriptor) is not dict or set(descriptor) != {"path", "file_sha256"}
            or not isinstance(descriptor["path"], str) or not Path(descriptor["path"]).is_absolute()
            or not isinstance(descriptor["file_sha256"], str) or not _SHA.fullmatch(descriptor["file_sha256"])):
        raise ValueError("adjudication: evidence path/SHA required")
    path = Path(descriptor["path"])
    _safe_root(path.parent)
    if hashlib.sha256(_read(path)).hexdigest() != descriptor["file_sha256"]:
        raise ValueError("adjudication: review evidence changed")
    _binding(files, path, descriptor["file_sha256"])


def _items_sha(items: dict) -> str:
    identity = {}
    for pid, item in items.items():
        if (not isinstance(pid, str) or not pid or not isinstance(item.get("structure_hash"), str)
                or not item["structure_hash"] or not isinstance(item.get("image_sha256"), (list, tuple))
                or not item["image_sha256"] or not isinstance(item.get("pixel_sha256", []), (list, tuple))):
            raise ValueError("adjudication: verified item family/image inventory required")
        images, pixels = list(item["image_sha256"]), list(item.get("pixel_sha256", []))
        if (any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in images + pixels)
                or (pixels and len(pixels) != len(images))):
            raise ValueError("adjudication: invalid item image identities")
        identity[pid] = {"structure_hash": item["structure_hash"], "image_sha256": images, "pixel_sha256": pixels}
    if not identity:
        raise ValueError("adjudication: nonempty item inventory required")
    return c._digest(identity)


def _complete(raw: dict, items: dict) -> None:
    _items_sha(items)
    if ("adjudication" in raw or raw.get("version") != c.VERSION
            or raw.get("status") != "EXCLUSION_CHECKED_WITH_LIMITS" or raw.get("blockers") != []
            or raw.get("partial_name") is not None or raw.get("training_allowed") is not False
            or raw.get("approval_granted") is not False or set(raw.get("steps", {})) != set(c._STEPS)
            or any(step.get("status") != "COMPLETE" for step in raw["steps"].values())
            or type(raw.get("input_count")) is not int or raw["input_count"] != len(items)
            or raw.get("receipt_sha256") != c._digest({key: value for key, value in raw.items() if key != "receipt_sha256"})
            or any(not isinstance(raw.get(key), str) or not _SHA.fullmatch(raw[key])
                   for key in ("index_sha256", "input_files_sha256"))
            or type(raw.get("candidates")) is not list):
        raise ValueError("adjudication: complete original exclusion receipt required")
    for candidate in raw["candidates"]:
        if (type(candidate) is not dict or candidate.get("problem_id") not in items
                or candidate.get("stage") not in c._STEPS
                or candidate.get("action") not in ("REVIEW", "QUARANTINE")
                or not isinstance(candidate.get("reason"), str) or not candidate["reason"].strip()):
            raise ValueError("adjudication: invalid original candidate")
    family = c.family_quarantine(items, raw["candidates"])
    if (raw.get("allowlist") != family["allowlist"] or raw.get("quarantined_ids") != family["quarantined_ids"]
            or raw.get("quarantined_families") != family["families"]):
        raise ValueError("adjudication: original family inventory mismatch")


def _derive(raw: dict, review: dict, raw_descriptor: dict, review_descriptor: dict, items: dict,
            files: dict[str, str] | None = None) -> tuple[dict, list[dict]]:
    _complete(raw, items)
    if (review.get("format") != FORMAT or review.get("raw_receipt_sha256") != raw_descriptor["file_sha256"]
            or any(review.get(key) != raw[key] for key in ("index_sha256", "input_files_sha256"))
            or type(review.get("decisions")) is not list):
        raise ValueError("adjudication: exact raw/index/input review binding required")
    provenance = review.get("provenance")
    if (type(provenance) is not dict or provenance.get("kind") != "AI_DIRECT_SIMILARITY_REVIEW"
            or provenance.get("human_verified") is not False or provenance.get("labels_modified", False) is not False
            or not isinstance(provenance.get("reviewer_id"), str) or not provenance["reviewer_id"].strip()
            or not isinstance(provenance.get("reviewed_at_utc"), str)):
        raise ValueError("adjudication: explicit nonhuman AI review provenance required")
    stamp = datetime.fromisoformat(provenance["reviewed_at_utc"].replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError("adjudication: UTC review timestamp required")
    candidates = {candidate_id(candidate): candidate for candidate in raw["candidates"]}
    decisions: dict[str, dict] = {}
    for decision in review["decisions"]:
        if (type(decision) is not dict or decision.get("candidate_id") not in candidates
                or decision["candidate_id"] in decisions or decision.get("verdict") not in ("DISTINCT", "DUPLICATE", "UNRESOLVED")
                or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
                or type(decision.get("evidence")) is not list or not decision["evidence"]):
            raise ValueError("adjudication: unique covered decision/reason/evidence required")
        for evidence in decision["evidence"]:
            _evidence(evidence, files)
        decisions[decision["candidate_id"]] = decision
    if set(decisions) != set(candidates):
        raise ValueError("adjudication: every unique original candidate must be reviewed")
    permanent = {cid for cid, candidate in candidates.items()
                 if candidate["stage"] not in _RELEASABLE or candidate["reason"] == "EXACT_RAW_OR_PIXEL_IMAGE"}
    released = {cid for cid, decision in decisions.items() if decision["verdict"] == "DISTINCT"} - permanent
    duplicates = {cid for cid, decision in decisions.items() if decision["verdict"] == "DUPLICATE"}
    unresolved = {cid for cid, decision in decisions.items() if decision["verdict"] == "UNRESOLVED"}
    effective = [candidate for candidate in raw["candidates"] if candidate_id(candidate) not in released]
    quarantine = c.family_quarantine(items, effective)
    result = copy.deepcopy(raw)
    result.update(allowlist=quarantine["allowlist"], quarantined_ids=quarantine["quarantined_ids"],
                  quarantined_families=quarantine["families"])
    module_sha = hashlib.sha256(_read(Path(__file__))).hexdigest()
    _binding(files, Path(__file__), module_sha)
    result["adjudication"] = {"format": FORMAT, "raw_receipt": copy.deepcopy(raw_descriptor),
        "review": copy.deepcopy(review_descriptor), "module_sha256": module_sha,
        "items_sha256": _items_sha(items), "unique_candidate_count": len(candidates),
        "released_candidate_ids": sorted(released), "duplicate_candidate_ids": sorted(duplicates),
        "unresolved_candidate_ids": sorted(unresolved), "nonreleasable_candidate_ids": sorted(permanent),
        "blacklisted_problem_ids": c.family_quarantine(items, [candidates[cid] for cid in duplicates | permanent])["quarantined_ids"],
        "unresolved_problem_ids": c.family_quarantine(items, [candidates[cid] for cid in unresolved])["quarantined_ids"],
        "counts": {verdict: sum(decision["verdict"] == verdict for decision in decisions.values())
                   for verdict in ("DISTINCT", "DUPLICATE", "UNRESOLVED")}, "human_verified": False}
    result["receipt_sha256"] = c._digest({key: value for key, value in result.items() if key != "receipt_sha256"})
    return result, effective


def apply(raw_receipt: dict, review: dict, items: dict) -> dict:
    """Apply SHA-bound explicit decisions; originals are read, never modified."""
    raw, decisions = _artifact(raw_receipt), _artifact(review)
    result, _ = _derive(raw, decisions, raw_receipt, review, items)
    return result


def validate(receipt: dict, items: dict, *, files: dict[str, str] | None = None) -> list[dict]:
    """Reopen/rederive every binding; raw receipts retain all unreviewed warnings."""
    if "adjudication" not in receipt:
        # The caller validates the original scanner contract; no review releases
        # anything here or imposes new fields on historical raw candidates.
        if type(receipt.get("candidates")) is not list:
            raise ValueError("adjudication: original candidate list required")
        return copy.deepcopy(receipt["candidates"])
    binding = receipt["adjudication"]
    if type(binding) is not dict:
        raise ValueError("adjudication: malformed binding")
    raw, review = _artifact(binding["raw_receipt"], files), _artifact(binding["review"], files)
    expected, effective = _derive(raw, review, binding["raw_receipt"], binding["review"], items, files)
    if receipt != expected:
        raise ValueError("adjudication: receipt or current source differs from rederived evidence")
    return copy.deepcopy(effective)
