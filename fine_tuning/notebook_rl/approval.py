"""Read-only launch validation, not an approval/report generator or GPU executor.

The public facade opens no vault/index/review payloads: a fresh trusted preflight
process does so, returning only selected actors, structure IDs and hash bindings.
Index/evidence descriptors are {path,file_sha256}; inline public registry/reports
are also accepted. Human/measurement provenance remains an upstream attestation,
not cryptographic proof that a human or GPU performed the declared work. No
current source registry/protocol/report is implicitly approved by this module.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .schema import ActorItem, from_json, to_json
from .resources import resource_path
from .vault import _COMPATIBLE_SCHEMA, _LEGACY_CODE, _read, _safe_root, _unique

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
_SPLITS = ("train", "dev", "sealed", "style_sealed")
_MAX_BYTES = 16 * 1024 * 1024
SMOKE_PURPOSE = "AI_PROGRAM_REVIEWED_SMOKE_ONLY"
SMOKE_REVIEW_POLICY = "AI_PROGRAM_NOT_HUMAN_v1"
# Opt-in stage for a mixed HUMAN/PROGRAM local pool: measurement checks, with an AI review
# record per source in place of the human D3/D7 review. Never human_verified, never training approval.
AI_LOCAL_STAGE = "ai_reviewed_local"
AI_LOCAL_PURPOSE = "AI_REVIEWED_LOCAL_POOL_ONLY"
AI_LOCAL_REVIEW_FORMAT = "ai-reviewed-local-review-v1"
AI_LOCAL_REVIEW_KIND = "AI_REVIEWED_LOCAL"
AI_LOCAL_REVIEW_METHOD = "DIRECT_ORIGINAL_INPUT_AI_REVIEW"
AI_LOCAL_SAMPLE = 20
AI_LOCAL_VERDICTS = ("render_ok", "leak_free", "gold_supported")
# User decision 2026-10-30: a v2 review applies acceptance sampling instead of blocking a source on any failure.
# F failures (reviewed items with a false verdict + leak-warning rows not leak_free): F=0 passes; F>=1 passes only
# with >= min(60, n) reviewed and F <= 2, every failed item ineligible; otherwise the source is BLOCKED.
AI_LOCAL_REVIEW_FORMAT_V2 = "ai-reviewed-local-review-v2"
ACCEPTANCE_RULE = "acceptance-sampling-v1"
ACCEPTANCE_SAMPLE = 60
ACCEPTANCE_MAX_FAILED = 2
ACCEPTANCE_ALPHA = .05
# User decision 2026-11-05: an explicit, recorded no-sample-review policy. No verdicts exist or are written;
# it is accepted only when no allowlisted item carries a leak warning (those are excluded before selection).
# Contamination/exclusion, allowlist, quarantine and every other D gate are unchanged.
AI_LOCAL_REVIEW_FORMAT_SKIPPED = "ai-reviewed-local-review-skipped-v1"
SOURCE_REVIEW_SKIPPED = "SKIPPED_BY_USER_DECISION_2026-11-05"
SOURCE_REVIEW_SKIPPED_KIND = "SOURCE_REVIEW_SKIPPED"
UNION_RECEIPT ="union_receipt.json"  # pool_freeze materialize: <vault>/contamination/union_receipt.json
_SCANNER_FILES = frozenset({"audit.py", "notebook_rl/contamination.py",
    "notebook_rl/benchmark_inventory_v1.json", "notebook_rl/vault.py",
    "notebook_rl/schema.py", "notebook_rl/protocol_v1.yaml"})
_LEGACY_SCANNER_CODE = {
    "audit.py": "5fb3765042def4f2dce5c2d31da2ee739f25bb83063df2aee379bee55b9b8c13",
    "notebook_rl/benchmark_inventory_v1.json": "743e03e10836e035c3cdb245d2bbc255e8c19f5fd70e00735d872435e583ab21",
    "notebook_rl/contamination.py": "caf0d31785ae9e88948b7b1ef6ecf39e72e6fef023165c4d98a7e7eef51c91dd",
    "notebook_rl/protocol_v1.yaml": "653fd442b9184384f1e028ad7269c8d8022cbe59b06c26f90afacd8c82d3daea",
    "notebook_rl/schema.py": _LEGACY_CODE["schema.py"],
    "notebook_rl/vault.py": _LEGACY_CODE["vault.py"],
}
_COMPATIBLE_SCANNER_CODE = _LEGACY_SCANNER_CODE | {
    "notebook_rl/schema.py": _COMPATIBLE_SCHEMA,
    "notebook_rl/vault.py": "538ac47c3465f9269c21b75a28746d49656477274bf2d363d623a56f5cc1fa44",
}


class _Blocked(Exception):
    pass


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value) -> str:
    digest = hashlib.sha256()
    active: set[int] = set()

    class DeepJSON(Exception):
        pass

    def update(item) -> None:
        if len(active) == 64:
            raise DeepJSON
        mapping = type(item) is dict and all(type(key) is str for key in item)
        sequence = type(item) is list and any(type(child) in (dict, list) for child in item)
        if not mapping and not sequence:
            # Keep numeric vectors on the C serializer without joining the full evidence.
            digest.update(_json(item))
            return
        identity = id(item)
        if identity in active:
            raise ValueError("Circular reference detected")
        active.add(identity)
        try:
            digest.update(b"{" if mapping else b"[")
            for index, key in enumerate(sorted(item) if mapping else range(len(item))):
                if index:
                    digest.update(b",")
                if mapping:
                    digest.update(_json(key))
                    digest.update(b":")
                update(item[key])
            digest.update(b"}" if mapping else b"]")
        finally:
            active.remove(identity)

    try:
        update(value)
    except (DeepJSON, UnicodeEncodeError):
        # Preserve the C encoder's depth and mixed-invalid exception ordering.
        return _sha(_json(value))
    return digest.hexdigest()


def _zero(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value == 0


def _require(condition, code: str) -> None:
    if not condition:
        raise _Blocked(code)


def _read_json(path: Path, files: dict) -> dict:
    path = Path(path).absolute()
    _safe_root(path.parent)
    raw = _read(path)
    def invalid(_):
        raise ValueError("nonfinite JSON")
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid)
    _require(type(value) is dict, "MALFORMED_ARTIFACT")
    files[str(path)] = _sha(raw)
    return value


def _artifact(value: dict, files: dict) -> tuple[dict, str]:
    if type(value) is dict and set(value) == {"path", "file_sha256"}:
        _require(isinstance(value["path"], str), "ARTIFACT_PATH")
        result = _read_json(Path(value["path"]), files)
        digest = files[str(Path(value["path"]).absolute())]
        _require(digest == value["file_sha256"], "ARTIFACT_SHA_MISMATCH")
        return result, digest
    _require(type(value) is dict, "MISSING_ARTIFACT")
    return value, _hash(value)


def code_hashes() -> dict[str, str]:
    """Public production-code snapshot for the caller's live expected GPU scope."""
    return {str(path.relative_to(_HERE)): _sha(_read(path)) for path in sorted(_HERE.rglob("*.py"))
            if "data_tests" not in path.relative_to(_HERE).parts}


def _scanner_code(code: dict, files: dict) -> None:
    """Keep historical receipt paths, checking exact identities in this package."""
    _require(type(code) is dict and len(code) == len(_SCANNER_FILES), "SCANNER_CODE_COVERAGE")
    roots, identities = set(), {}
    for name, digest in code.items():
        _require(type(name) is str and type(digest) is str
                 and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "SCANNER_CODE_PATH")
        path = Path(name)
        _require(path.is_absolute() and str(path) == name and not name.startswith("//")
                 and ".." not in path.parts and path.parts.count("fine_tuning") == 1,
                 "SCANNER_CODE_PATH")
        index = path.parts.index("fine_tuning")
        root = Path(*path.parts[:index + 1])
        relative = path.relative_to(root).as_posix()
        _require(relative in _SCANNER_FILES and relative not in identities, "SCANNER_CODE_COVERAGE")
        roots.add(root)
        identities[relative] = digest
    _require(len(roots) == 1 and set(identities) == _SCANNER_FILES, "SCANNER_CODE_COVERAGE")
    current_code = {}
    for relative in identities:
        current = _HERE.parent / relative
        _safe_root(current.parent)
        files[str(current)] = _sha(_read(current))
        current_code[relative] = files[str(current)]
    if current_code == identities:
        return
    # The reviewed SlotReport/data-reader change leaves these historical scanner inputs intact.
    _require(identities == _LEGACY_SCANNER_CODE and current_code == _COMPATIBLE_SCANNER_CODE,
             "SCANNER_CODE_CHANGED")
    guards = _HERE / "guards.py"
    files[str(guards)] = _sha(_read(guards))
    _require(files[str(guards)] == _LEGACY_CODE["guards.py"], "SCANNER_CODE_CHANGED")


def _time(value) -> None:
    _require(isinstance(value, str), "INVALID_REVIEW_TIME")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(stamp.tzinfo is not None and stamp.utcoffset() == timezone.utc.utcoffset(stamp), "INVALID_REVIEW_TIME")


def _selection(selection: dict, phase: dict, phase_sha: str, build_sha: str,
               train: dict[str, ActorItem], structures: dict, protocol: dict) -> list[str]:
    _require(selection.get("selection_id") == "notebook-rl-selection-v1"
             and selection.get("status") == "SELECTION_FROZEN_NOT_TRAINING_APPROVAL"
             and selection.get("training_allowed") is False
             and selection.get("phase_a_report_sha256") == phase_sha
             and selection.get("build_id") == phase.get("build_id") == build_sha, "SELECTION_BINDING")
    _time(selection.get("frozen_at_utc"))
    _require(phase.get("status") == "PHASE_A_COMPLETED" and type(phase.get("policy_version")) is int
             and phase["policy_version"] == 0 and phase.get("split") == "train"
             and type(phase.get("rows")) is list and phase["rows"], "PHASE_A_NOT_COMPLETE_F0_TRAIN")
    rows = phase["rows"]
    _require(all(type(row) is dict and isinstance(row.get("problem_id"), str) for row in rows), "PHASE_A_ROWS")
    ids = [row["problem_id"] for row in rows]
    _require(len(set(ids)) == len(ids) and set(ids) == set(train)
             and phase.get("candidate_ids_sha256") == _hash(sorted(ids)), "PHASE_A_CANDIDATE_COVERAGE")
    caps = selection.get("max_new_tokens")
    _require(caps == phase.get("max_new_tokens") and type(caps) is dict and set(caps) == {"1", "2", "3"}
             and all(type(v) is int and 2048 <= v <= 4096 for v in caps.values())
             and caps == {str(k): v for k, v in protocol["rollout"]["max_new_tokens"].items()}, "TURN_BUDGET_BINDING")
    rules = selection["rules"]
    keys = {"max_pass_rate", "max_zero_pass_share", "blind_floor", "truncation_family_max", "max_selected", "min_open_answer_fraction"}
    _require(type(rules) is dict and set(rules) == keys and type(rules["max_selected"]) is int
             and rules["max_selected"] > 0 and all(type(rules[k]) in (int, float) and math.isfinite(rules[k])
             and 0 <= rules[k] <= 1 for k in keys - {"max_selected"}), "SELECTION_RULES")
    declared = protocol["data_selection"]
    _require(rules["max_pass_rate"] == declared["phase_a"]["max_pass_rate"]
             and rules["max_zero_pass_share"] == declared["phase_a"]["max_zero_pass_fraction"]
             and rules["blind_floor"] == declared["blind"]["accuracy_floor"]
             and rules["min_open_answer_fraction"] >= declared["min_open_answer_fraction"]
             and rules["truncation_family_max"] <= .1, "SELECTION_RULES_NOT_PROTOCOL_BOUND")
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        pid = row["problem_id"]
        _require(all(type(row.get(k)) is int for k in ("correct", "n", "blind_correct", "blind_n", "truncated", "n_choices"))
                 and row["n"] == 8 and row["blind_n"] == 4 and 0 <= row["correct"] <= 8
                 and 0 <= row["blind_correct"] <= 4 and 0 <= row["truncated"] <= 8
                 and row["n_choices"] in (0, 10) and row["n_choices"] == len(train[pid].choices)
                 and row.get("family") == train[pid].family and type(row.get("context_ok")) is bool, "PHASE_A_MEASUREMENTS")
        counts[row["family"]][0] += row["truncated"]
        counts[row["family"]][1] += 8
    zero: list[str] = []
    nonzero: list[str] = []
    excluded: list[dict] = []
    for row in sorted(rows, key=lambda row: row["problem_id"]):
        pid, p, blind = row["problem_id"], row["correct"] / 8, row["blind_correct"] / 4
        a, b = counts[row["family"]]
        reason = ("CONTEXT" if not row["context_ok"] else "FAMILY_TRUNCATION" if a / b > rules["truncation_family_max"]
                  else "TOO_EASY" if p > rules["max_pass_rate"] else "BLIND_SOLVABLE"
                  if blind > max(1 / row["n_choices"] if row["n_choices"] else 0, rules["blind_floor"]) else None)
        if reason:
            excluded.append({"problem_id": pid, "reason": reason})
        else:
            (zero if p == 0 else nonzero).append(pid)
    selected = nonzero[:rules["max_selected"]]
    share = rules["max_zero_pass_share"]
    zero_cap = len(zero) if share == 1 else math.floor(len(selected) * share / (1 - share))
    selected.extend(zero[:min(zero_cap, rules["max_selected"] - len(selected))])
    excluded.extend({"problem_id": pid, "reason": "ZERO_PASS_CAP" if pid in zero else "POOL_CAP"}
                    for pid in set(zero + nonzero) - set(selected))
    _require(selected and selection.get("selected_train") == sorted(selected)
             and selection.get("excluded") == sorted(excluded, key=lambda row: row["problem_id"])
             and sum(not train[pid].choices for pid in selected) / len(selected) >= rules["min_open_answer_fraction"]
             and all(pid in structures for pid in selected), "SELECTION_RECOMPUTATION_MISMATCH")
    return sorted(selected)


def _proofs(report: dict, prefix: str, count: int, binding: dict, scope: dict,
            producer: dict, files: dict) -> dict:
    _require(report.get("format") == f"notebook-rl-{'data' if prefix == 'D' else 'gpu'}-gates-v1"
             and report.get("status") == "COMPLETE" and report.get("binding") == binding
             and report.get("scope") == scope, f"{prefix}_REPORT_BINDING")
    gates = report["gates"]
    _require(type(gates) is dict and set(gates) == {f"{prefix}{i}" for i in range(1 if prefix == "D" else 0, count)}, f"{prefix}_GATE_COVERAGE")
    result = {}
    for gate, entry in gates.items():
        _require(type(entry) is dict and set(entry) == {"status", "artifact"}
                 and entry["status"] == "PASS", f"{gate}_NOT_PASS")
        proof, _ = _artifact(entry["artifact"], files)
        _require(set(entry["artifact"]) == {"path", "file_sha256"}, f"{gate}_EVIDENCE_FILE_REQUIRED")
        _require(proof.get("gate") == gate and proof.get("status") == "PASS"
                 and proof.get("binding_sha256") == _hash(binding) and proof.get("scope_sha256") == _hash(scope)
                 and proof.get("producer_sha256") == producer["file_sha256"]
                 and type(proof.get("measurements")) is dict and proof["measurements"], f"{gate}_PROOF_BINDING")
        result[gate] = proof
    return result


def _measurement_scope(scope: dict, binding: dict, protocol: dict, files: dict,
                       *, limited_smoke: bool = False, ai_local: bool = False) -> None:
    """Data-only scope, fixed BEFORE Phase A; never a GPU/training certificate."""
    keys = {"purpose", "protocol_sha256", "code_sha256", "input_files_sha256", "producer"}
    if limited_smoke:
        keys.add("authorization")
    purpose = SMOKE_PURPOSE if limited_smoke else AI_LOCAL_PURPOSE if ai_local else "FROZEN_WEIGHT_MEASUREMENT_ONLY"
    _require(type(scope) is dict and set(scope) == keys and scope["purpose"] == purpose,
             "MEASUREMENT_SCOPE_REQUIRED")
    if limited_smoke:
        authorization = scope["authorization"]
        _require(type(authorization) is dict and set(authorization) == {
            "approval_reference", "approved_at_utc", "review_policy", "max_pool_items",
            "max_train_items", "max_optimizer_updates", "training_campaign_allowed"}
            and isinstance(authorization["approval_reference"], str) and bool(authorization["approval_reference"].strip())
            and authorization["review_policy"] == SMOKE_REVIEW_POLICY
            and type(authorization["max_pool_items"]) is int and 1 <= authorization["max_pool_items"] <= 64
            and type(authorization["max_train_items"]) is int and 1 <= authorization["max_train_items"] <= 24
            and type(authorization["max_optimizer_updates"]) is int and authorization["max_optimizer_updates"] == 2
            and authorization["training_campaign_allowed"] is False, "SMOKE_AUTHORIZATION_REQUIRED")
        _time(authorization["approved_at_utc"])
    _require(scope["protocol_sha256"] == protocol["sha256"] and scope["code_sha256"] == code_hashes()
             and scope["input_files_sha256"] == binding["input_files_sha256"], "MEASUREMENT_CURRENT_SCOPE_MISMATCH")
    producer = scope["producer"]
    _require(type(producer) is dict and set(producer) == {"path", "file_sha256"}, "PRODUCER_BINDING")
    path = Path(producer["path"]).absolute()
    _safe_root(path.parent)
    files[str(path)] = _sha(_read(path))
    _require(files[str(path)] == producer["file_sha256"], "PRODUCER_CODE_CHANGED")


def _human(proof: dict, registry: dict) -> None:
    provenance = proof["provenance"]
    _require(type(provenance) is dict and set(provenance) == {"reviewer_id", "kind", "method", "reviewed_at_utc"}
             and provenance["kind"] == "HUMAN" and provenance["method"] == "DIRECT_ORIGINAL_INPUT_REVIEW",
             "HUMAN_REVIEW_PROVENANCE")
    reviewers = registry.get("approved_human_reviewers", {})
    reviewer = reviewers.get(provenance["reviewer_id"]) if type(reviewers) is dict else None
    _require(type(reviewer) is dict and reviewer.get("kind") == "HUMAN"
             and reviewer.get("approved") is True, "HUMAN_REVIEWER_NOT_REGISTERED")
    _time(provenance["reviewed_at_utc"])


def _ai_program_review(proof: dict, registry: dict) -> None:
    """Declared AI review, never a human-review certificate or label producer."""
    provenance = proof.get("provenance")
    if type(provenance) is not dict:
        raise _Blocked("AI_PROGRAM_REVIEW_PROVENANCE")
    _require(set(provenance) == {
        "reviewer_id", "kind", "method", "reviewed_at_utc", "human_verified", "labels_modified"}
        and provenance["kind"] == "AI_REVIEWED_PROGRAM_VERIFIED"
        and provenance["method"] == "DIRECT_ORIGINAL_INPUT_AND_PROGRAM_REVIEW"
        and provenance["human_verified"] is False and provenance["labels_modified"] is False,
        "AI_PROGRAM_REVIEW_PROVENANCE")
    reviewers = registry.get("limited_smoke_reviewers", {})
    reviewer = reviewers.get(provenance["reviewer_id"]) if type(reviewers) is dict else None
    _require(type(reviewer) is dict and reviewer.get("kind") == "AI"
             and reviewer.get("approved_for") == SMOKE_REVIEW_POLICY
             and reviewer.get("human_verified") is False, "AI_SMOKE_REVIEWER_NOT_REGISTERED")
    _time(provenance["reviewed_at_utc"])


def label_class(lineage) -> str:
    """GENERATOR (project program, regenerable), EXTERNAL_PROGRAM (upstream program label) or HUMAN.

    Only GENERATOR items have a solver to rerun for D1/D2/D4; anything ambiguous blocks.
    """
    generated = lineage.generator_version is not None or lineage.generator_seed is not None
    if lineage.label_provenance == "PROGRAM" and generated:
        _require(lineage.generator_version is not None and type(lineage.generator_seed) is int
                 and lineage.upstream_origin == "procedural" and lineage.license == "project-authored"
                 and tuple(lineage.parents) == (), "AI_LOCAL_LABEL_CLASS")
        return "GENERATOR"
    _require(not generated and lineage.label_provenance in ("PROGRAM", "HUMAN"), "AI_LOCAL_LABEL_CLASS")
    return "EXTERNAL_PROGRAM" if lineage.label_provenance == "PROGRAM" else "HUMAN"


def ai_review_order(problem_ids, seed: int) -> list[str]:
    """Preregistered, label-blind review order: seed-keyed hash of the problem ID only."""
    return sorted(problem_ids, key=lambda pid: _sha(f"{seed}:{pid}".encode()))


def is_skipped_review(review: object) -> bool:
    return type(review) is dict and review.get("format") == AI_LOCAL_REVIEW_FORMAT_SKIPPED


def _ai_local_provenance(proof: dict, skipped: bool = False) -> str:
    provenance: dict = proof.get("provenance")  # type: ignore[assignment]
    kind, method = ((SOURCE_REVIEW_SKIPPED_KIND, SOURCE_REVIEW_SKIPPED) if skipped
                    else (AI_LOCAL_REVIEW_KIND, AI_LOCAL_REVIEW_METHOD))
    _require(type(provenance) is dict and set(provenance) == {
        "reviewer_id", "kind", "method", "reviewed_at_utc", "human_verified", "labels_modified"}
        and isinstance(provenance["reviewer_id"], str) and bool(provenance["reviewer_id"].strip())
        and provenance["kind"] == kind and provenance["method"] == method
        and provenance["human_verified"] is False and provenance["labels_modified"] is False,
        "AI_LOCAL_REVIEW_PROVENANCE")
    _time(provenance["reviewed_at_utc"])
    return provenance["reviewer_id"]


def clopper_pearson_upper(failed: int, n: int, alpha: float = ACCEPTANCE_ALPHA) -> float:
    """Exact one-sided (1 - alpha) Clopper-Pearson upper bound: the p with P(X <= failed | n, p) = alpha.

    Equals the Beta(1 - alpha; failed + 1, n - failed) quantile; bisection on the log-space binomial CDF,
    stdlib only. 1.0 when failed >= n (including n = 0). Rounded to 10 digits for a stable JSON record.
    """
    if failed >= n:
        return 1.0
    log_comb = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in range(failed + 1)]

    def cdf(p: float) -> float:
        return math.fsum(math.exp(c + k * math.log(p) + (n - k) * math.log1p(-p)) for k, c in enumerate(log_comb))
    low, high = 0.0, 1.0
    for _ in range(100):  # the CDF falls in p; 100 halvings reach adjacent doubles
        middle = (low + high) / 2
        low, high = (middle, high) if cdf(middle) > alpha else (low, middle)
    return round(high, 10)


def _ai_local_review(review: dict, reviewer: str, items: dict, warnings: dict, input_files_sha256: str,
                     seed: int, allowlist: set[str]) -> tuple[dict[str, list[str]], int, dict | None]:
    """Per-source AI review of exclusion-allowlisted items only; returns (blocked sources, unresolved leaks,
    acceptance quality section or None for a v1 review).

    v1: a failing verdict, an unresolved leak warning or a sample shorter than min(20, allowlisted) blocks
    that source (every item ineligible). v2 (acceptance-sampling-v1): the short sample still blocks; F >= 1
    failures block unless >= min(60, allowlisted) were reviewed and F <= 2, in which case only the failed
    items are ineligible. A malformed, reordered or quarantine-touching record blocks all.
    Skipped (user decision 2026-11-05): no sources/verdicts at all, and no leak warning on any allowlisted item.
    """
    if is_skipped_review(review):
        _require(set(review) == {"format", "reviewer_id", "human_verified", "labels_modified", "reviewed_at_utc",
                                 "sampling_seed", "input_files_sha256", "source_review", "sources", "leak_warnings"}
                 and review["source_review"] == SOURCE_REVIEW_SKIPPED and review["reviewer_id"] == reviewer
                 and review["human_verified"] is False and review["labels_modified"] is False
                 and type(review["sampling_seed"]) is int and review["sampling_seed"] == seed
                 and review["input_files_sha256"] == input_files_sha256
                 and review["sources"] == {} and review["leak_warnings"] == [], "AI_LOCAL_SKIPPED_REVIEW_BINDING")
        _time(review["reviewed_at_utc"])
        _require(not any(row["problem_id"] in allowlist for rows in warnings.values() for row in rows),
                 "SKIPPED_REVIEW_ALLOWLISTED_LEAK_WARNING")
        return {}, 0, None
    acceptance = type(review) is dict and review.get("format") == AI_LOCAL_REVIEW_FORMAT_V2
    _require(type(review) is dict and set(review) == {
        "format", "reviewer_id", "human_verified", "labels_modified", "reviewed_at_utc", "sampling_seed",
        "input_files_sha256", "sources", "leak_warnings", *(("acceptance_rule",) if acceptance else ())}
        and review["format"] in (AI_LOCAL_REVIEW_FORMAT, AI_LOCAL_REVIEW_FORMAT_V2)
        and (not acceptance or review["acceptance_rule"] == ACCEPTANCE_RULE) and review["reviewer_id"] == reviewer
        and review["human_verified"] is False and review["labels_modified"] is False
        and type(review["sampling_seed"]) is int and review["sampling_seed"] == seed
        and review["input_files_sha256"] == input_files_sha256, "AI_LOCAL_REVIEW_BINDING")
    _time(review["reviewed_at_utc"])
    groups: dict[str, list[str]] = defaultdict(list)
    for pid in sorted(allowlist):
        groups[items[pid]["actor"].source].append(pid)
    _require(type(review["sources"]) is dict and set(review["sources"]) == set(groups), "AI_LOCAL_REVIEW_SOURCE_COVERAGE")
    blocked: dict[str, set[str]] = defaultdict(set)
    details: dict[str, set[str]] = defaultdict(set)    # per-item failure reasons
    failures: dict[str, list[str]] = defaultdict(list)  # one entry per failed reviewed item / failed leak row
    reviewed: dict[str, int] = {}
    for source, pids in groups.items():
        rows, ordered = review["sources"][source], ai_review_order(pids, seed)
        ids = [row.get("problem_id") if type(row) is dict else None for row in rows] if type(rows) is list else None
        # Hash order over allowlisted items only: a quarantined or out-of-order ID is never a valid prefix.
        if ids is None or ids != ordered[:len(ids)]:
            raise _Blocked("AI_LOCAL_REVIEW_PREREGISTERED_SAMPLE")
        reviewed[source] = len(ids)
        if len(ids) < min(AI_LOCAL_SAMPLE, len(pids)):
            blocked[source].add(f"SHORT_SAMPLE reviewed={len(ids)} required={min(AI_LOCAL_SAMPLE, len(pids))}")
        for row in rows:
            _require(set(row) == {"problem_id", "image_sha256", "notes", *AI_LOCAL_VERDICTS}
                     and row["image_sha256"] == list(items[row["problem_id"]]["actor"].image_sha256)
                     and all(type(row[key]) is bool for key in AI_LOCAL_VERDICTS)
                     and isinstance(row["notes"], str), "AI_LOCAL_REVIEW_ITEM")
            failed = [f"FAILED_VERDICT {key} {row['problem_id']}" for key in AI_LOCAL_VERDICTS if not row[key]]
            details[source].update(failed)
            if failed:
                failures[source].append(row["problem_id"])
    expected = sorted((row["problem_id"], row["field"]) for rows in warnings.values() for row in rows
                      if row["problem_id"] in allowlist)
    leaks = review["leak_warnings"]
    _require(type(leaks) is list and all(
        type(row) is dict and set(row) == {"problem_id", "field", "leak_free", "notes"}
        and type(row["leak_free"]) is bool and isinstance(row["notes"], str) for row in leaks)
        and sorted((row["problem_id"], row["field"]) for row in leaks) == expected, "AI_LOCAL_LEAK_REVIEW_COVERAGE")
    for row in leaks:
        if not row["leak_free"]:
            source = items[row["problem_id"]]["actor"].source
            details[source].add(f"LEAK_WARNING_NOT_LEAK_FREE {row['field']} {row['problem_id']}")
            failures[source].append(row["problem_id"])
    unresolved = sum(not row["leak_free"] for row in leaks)
    quality = None
    if acceptance:
        records: dict[str, dict] = {}
        quality = {"rule": ACCEPTANCE_RULE, "bound": "clopper-pearson-one-sided-upper", "confidence": 1 - ACCEPTANCE_ALPHA,
                   "sources": records}
        for source, pids in sorted(groups.items()):
            count, n, required = len(failures[source]), reviewed[source], min(ACCEPTANCE_SAMPLE, len(pids))
            if count and n < required:
                blocked[source].add(f"ACCEPTANCE_SAMPLE_TOO_SMALL failed={count} reviewed={n} required={required}")
            if count > ACCEPTANCE_MAX_FAILED:
                blocked[source].add(f"ACCEPTANCE_TOO_MANY_FAILED failed={count} max={ACCEPTANCE_MAX_FAILED} reviewed={n}")
            if blocked[source]:
                blocked[source] |= details[source]
            records[source] = {
                "reviewed": n, "failed": count, "observed_rate": round(count / n, 10) if n else None,
                "cp_upper": clopper_pearson_upper(count, n), "failed_ids": sorted(set(failures[source])),
                "status": "BLOCKED" if blocked[source] else "ELIGIBLE"}
    else:
        for source, reasons in details.items():
            blocked[source] |= reasons
    return {source: sorted(reasons) for source, reasons in sorted(blocked.items()) if reasons}, unresolved, quality


def ai_local_eligibility(sources: dict[str, str], allowlist, blocked: dict[str, list[str]],
                         quality: dict | None = None, source_review: str | None = None) -> dict:
    """Allowlisted items outside blocked sources and outside acceptance-failed items; quarantined IDs are
    never eligible. The quality section is recorded only for an acceptance-sampling (v2) review."""
    failed = {pid for row in (quality or {}).get("sources", {}).values() for pid in row["failed_ids"]}
    eligible = sorted(pid for pid in allowlist if sources[pid] not in blocked and pid not in failed)
    _require(bool(eligible), "NO_ELIGIBLE_SOURCE")
    return {"allowlisted_by_source": dict(sorted(Counter(sources[pid] for pid in allowlist).items())),
            "quarantined_count": len(set(sources) - set(allowlist)), "blocked_sources": blocked,
            "eligible_ids": eligible, **({"quality": quality} if quality is not None else {}),
            **({"source_review": source_review} if source_review is not None else {})}


def _data_gates(proofs: dict, items: dict, manifests: dict, input_files: dict,
                registry: dict, receipt_sha: str, review_seed: int, *, limited_smoke: bool = False,
                ai_review: dict | None = None, allowlist: set[str] | None = None
                ) -> tuple[dict[str, list[str]] | None, dict | None]:
    """Returns (AI-review blocked sources, acceptance quality section), both None without ai_review (the quality
    section also None for a v1 review); every other failure raises."""
    blocked = quality = None
    if ai_review is not None and allowlist is None:
        raise _Blocked("AI_LOCAL_ALLOWLIST_REQUIRED")
    allowed = allowlist or set()
    if ai_review is None:
        program = sorted(pid for pid, item in items.items() if item["lineage"].label_provenance == "PROGRAM")
    else:
        classes = {pid: label_class(item["lineage"]) for pid, item in items.items()}
        # External PROGRAM labels have no generator solver; the AI review covers them instead.
        # Quarantined items are never eligible, so D1/D2/D4 cover the allowlisted generator items.
        program = sorted(pid for pid in allowed if classes[pid] == "GENERATOR")
    if limited_smoke:
        _require(len(program) == len(items), "SMOKE_PROGRAM_LABELS_ONLY")
    for gate, mismatch in (("D1", "repeat_mismatches"), ("D2", "solver_mismatches")):
        m = proofs[gate]["measurements"]
        _require(m.get("checked_ids_sha256") == _hash(program) and type(m.get("items_checked")) is int
                 and m["items_checked"] == len(program) and type(m.get(mismatch)) is int and m[mismatch] == 0,
                 f"{gate}_COVERAGE_OR_MISMATCH")
    _require(proofs["D2"]["measurements"].get("independent_solvers") is True, "D2_INDEPENDENT_SOLVERS")
    unresolved_leaks = None
    skipped = is_skipped_review(ai_review)
    if ai_review is not None:
        reviewer = _ai_local_provenance(proofs["D3"], skipped)
        m = proofs["D3"]["measurements"]
        _require(set(m) == {"sampling_seed", "review"} and type(m["sampling_seed"]) is int
                 and m["sampling_seed"] == review_seed, "D3_REVIEW_SEED_BINDING")
        blocked, unresolved_leaks, quality = _ai_local_review(ai_review, reviewer, items,
            {split: manifest["leak_review"] for split, manifest in manifests.items()}, _hash(input_files), review_seed,
            allowed)
    else:
        (_ai_program_review if limited_smoke else _human)(proofs["D3"], registry)
        samples = proofs["D3"]["measurements"].get("samples")
        _require(type(proofs["D3"]["measurements"].get("sampling_seed")) is int
                 and proofs["D3"]["measurements"]["sampling_seed"] == review_seed, "D3_REVIEW_SEED_BINDING")
        _require(type(samples) is list and len(samples) == len({s.get("problem_id") for s in samples if type(s) is dict}), "D3_SAMPLE_IDENTITIES")
        counts: Counter = Counter()
        for sample in samples:
            _require(type(sample) is dict and sample.get("problem_id") in items, "D3_UNKNOWN_SAMPLE")
            item = items[sample["problem_id"]]
            _require(sample.get("image_sha256") == list(item["actor"].image_sha256)
                     and sample.get("printed_text_slots_consistent") is True
                     and type(sample.get("mismatches")) is int and sample["mismatches"] == 0, "D3_INPUT_BINDING_OR_MISMATCH")
            counts[item["actor"].source] += 1
        _require(set(counts) == {item["actor"].source for item in items.values()}
                 and (limited_smoke or all(n >= 50 for n in counts.values())), "D3_FIFTY_PER_SOURCE_REQUIRED")
        groups: dict[str, list[str]] = defaultdict(list)
        for pid, item in items.items():
            groups[item["actor"].source].append(pid)
        expected_sample = set(items) if limited_smoke else {pid for pids in groups.values()
                           for pid in sorted(pids, key=lambda pid: _sha(f"{review_seed}:{pid}".encode()))[:50]}
        _require({sample["problem_id"] for sample in samples} == expected_sample, "D3_PREREGISTERED_RANDOM_SAMPLE")
    templates = {items[pid]["lineage"].source + ":" + items[pid]["lineage"].template_id for pid in program}
    pairs = proofs["D4"]["measurements"].get("templates")
    _require(type(pairs) is dict and set(pairs) == templates, "D4_TEMPLATE_COVERAGE")
    for measured in pairs.values():
        _require(type(measured) is dict and type(measured.get("pairs")) is int and measured["pairs"] >= 100
                 and type(measured.get("answer_changed")) is int and .95 <= measured["answer_changed"] / measured["pairs"] <= 1
                 and measured.get("question_unchanged") is True and measured.get("nonimage_inputs_unchanged") is True,
                 "D4_IMAGE_DEPENDENCE")
    m = proofs["D5"]["measurements"]
    _require(set(m) == {"structure_cross_split", "image_cross_split"}
             and all(type(value) is int and value == 0 for value in m.values()), "D5_SPLIT_LEAK")
    _require(proofs["D6"]["measurements"] == {"exclusion_receipt_sha256": receipt_sha}, "D6_RECEIPT_BINDING")
    if ai_review is not None:
        _require(_ai_local_provenance(proofs["D7"], skipped) == reviewer, "AI_LOCAL_REVIEW_PROVENANCE")
    else:
        (_ai_program_review if limited_smoke else _human)(proofs["D7"], registry)
    actor_files = {path: digest for path, digest in input_files.items() if Path(path).name == "actor_inputs.jsonl"}
    warnings = {split: manifest["leak_review"] for split, manifest in manifests.items()}
    m = proofs["D7"]["measurements"]
    # Under ai_review an unresolved leak warning has already blocked its source; its count must still match.
    _require(m.get("reviewed_actor_files") == actor_files and m.get("warnings_sha256") == _hash(warnings)
             and type(m.get("unresolved")) is int
             and m["unresolved"] == (0 if unresolved_leaks is None else unresolved_leaks), "D7_LEAK_REVIEW_BINDING")
    m = proofs["D8"]["measurements"]
    _require(set(m) == {"separate_process", "private_open_denied", "benchmark_open_denied",
                       "teacher_import_denied", "python_guard_not_os_sandbox"}
             and all(value is True for value in m.values()), "D8_ISOLATION")
    return blocked, quality


def _gpu_scope(scope: dict, binding: dict, protocol: dict, files: dict) -> None:
    _require(type(scope) is dict and set(scope) == {"protocol_sha256", "model", "code_sha256", "environment_lock_sha256",
             "device", "probes", "runtime_id", "arm", "input_files_sha256", "producers"}, "GPU_EXPECTED_SCOPE_REQUIRED")
    _require(scope["protocol_sha256"] == protocol["sha256"] and scope["code_sha256"] == code_hashes()
             and scope["input_files_sha256"] == binding["input_files_sha256"]
             and scope["arm"] in protocol["arms"] and isinstance(scope["runtime_id"], str) and scope["runtime_id"], "GPU_CURRENT_SCOPE_MISMATCH")
    path = resource_path(protocol["initial_checkpoint"]["manifest"], project_root=_PROJECT)
    baseline = _read_json(path, files)
    model = {"model_id": baseline["repo_id"], "revision": baseline["revision"], "manifest_sha256": files[str(path)],
             "files_sha256": {entry["path"]: entry["sha256"] for entry in baseline["files"]}}
    _require(scope["model"] == model and model["model_id"] == protocol["initial_checkpoint"]["model_id"]
             and model["revision"] == protocol["initial_checkpoint"]["revision"], "GPU_MODEL_BINDING")
    lock = resource_path(protocol["environment"]["lock"], project_root=_PROJECT)
    files[str(lock)] = _sha(_read(lock))
    _require(scope["environment_lock_sha256"] == files[str(lock)], "GPU_LOCK_BINDING")
    device = scope["device"]
    _require(type(device) is dict and set(device) == {"type", "uuid", "name", "index", "total_memory_bytes", "driver_version"}
             and device["type"] == "cuda" and all(isinstance(device[k], str) and device[k] for k in ("uuid", "name", "driver_version"))
             and type(device["index"]) is int and device["index"] >= 0
             and type(device["total_memory_bytes"]) is int and device["total_memory_bytes"] > 0, "GPU_DEVICE_BINDING")
    probes = scope["probes"]
    _require(type(probes) is list and len(probes) == 2 and {p.get("image_count") for p in probes if type(p) is dict} == {1, 2}, "GPU_TWO_PROBES_REQUIRED")
    for probe in probes:
        _require(type(probe) is dict and set(probe) == {"id", "image_count", "input_sha256"}
                 and type(probe["image_count"]) is int and probe["image_count"] in (1, 2)
                 and isinstance(probe["id"], str) and probe["id"] and isinstance(probe["input_sha256"], str)
                 and len(probe["input_sha256"]) == 64 and not set(probe["input_sha256"]) - set("0123456789abcdef"), "GPU_PROBE_BINDING")
    _require(len({p["id"] for p in probes}) == 2 and len({p["input_sha256"] for p in probes}) == 2, "GPU_DISTINCT_PROBES_REQUIRED")
    producers = scope["producers"]
    _require(type(producers) is dict and set(producers) == {"data", "gpu"}, "PRODUCER_COVERAGE")
    for producer in producers.values():
        _require(type(producer) is dict and set(producer) == {"path", "file_sha256"}, "PRODUCER_BINDING")
        path = Path(producer["path"]).absolute()
        _safe_root(path.parent)
        files[str(path)] = _sha(_read(path))
        _require(files[str(path)] == producer["file_sha256"], "PRODUCER_CODE_CHANGED")


def _gpu_gates(proofs: dict, scope: dict, protocol: dict) -> None:
    m = {gate: proof["measurements"] for gate, proof in proofs.items()}
    _require(all(m["G0"].get(k) is True for k in ("lock_match", "cuda", "bf16_matmul", "model_executor"))
             and m["G0"].get("vllm_multiprocessing") is False, "G0_ENVIRONMENT")
    expected = dict(protocol["expected_trainable"])
    if scope["arm"] == "MOD-FML":
        expected["V"] = 0
    _require(m["G1"].get("trainable_counts") == expected
             and all(type(v) is int for v in m["G1"]["trainable_counts"].values())
             and m["G1"].get("mergers") == 4
             and m["G1"].get("master_dtype") == "float32" and m["G1"].get("base_dtype") == "bfloat16", "G1_INVENTORY")
    _require(type(m["G2"].get("images_checked")) is int and m["G2"]["images_checked"] >= 3
             and _zero(m["G2"].get("grid_mismatches")) and _zero(m["G2"].get("token_feature_mismatches")), "G2_IMAGE_PARITY")
    _require(_zero(m["G3"].get("initial_max_abs")) and _zero(m["G3"].get("post_sync_max_abs"))
             and all(m["G3"].get(k) is True for k in ("frozen_sha_unchanged", "requires_grad_restored")), "G3_REFERENCE")
    checks = {"G4": ("merger_changed", "frozen_unchanged"), "G5": ("all_tensors_match", "all_four_mergers", "hf_base_unchanged"),
              "G6": ("fresh_engine_match", "default_cache_comparison_measured"), "G8": ("tokens_exact", "generated_only_mask", "eos_checked"),
              "G9": ("vision_off_grad_none", "nonvision_grad_finite_nonzero", "hard_freeze_gradient_match"),
              "G10": ("numeric_tests_match", "trl_formula_match"), "G11": ("longest_sample_step_complete", "no_oom"),
              "G12": ("fresh_process_reload", "policy_logits_equal", "reference_logits_equal", "next_loss_equal"),
              "G13": ("exclusion_allowlist_only", "ce_patch_raises", "teacher_import_denied")}
    for gate, keys in checks.items():
        _require(all(m[gate].get(key) is True for key in keys), f"{gate}_MEASUREMENTS")
    gradients = m["G9"]
    _require(type(gradients.get("vision_parameter_count")) is int
             and gradients["vision_parameter_count"] == expected["V"], "G9_VISION_INVENTORY")
    status_keys = ("vision_applicability", "vision_on_status", "mixed_status")
    result_keys = ("vision_on_grad_nonzero", "mixed_on_only")
    if scope["arm"] == "MOD-FML":
        # No installed Vision adapter means no ON/mixed experiment, not a passed one.
        _require(all(gradients.get(k) == "NOT_APPLICABLE_MOD_FML" for k in status_keys)
                 and all(k in gradients and gradients[k] is None for k in result_keys)
                 and gradients.get("vision_na_reason") == "NO_INSTALLED_VISION_ADAPTERS", "G9_VISION_NA")
    else:
        _require(all(gradients.get(k) == "MEASURED" for k in status_keys)
                 and all(gradients.get(k) is True for k in result_keys)
                 and "vision_na_reason" in gradients and gradients["vision_na_reason"] is None, "G9_VISION_ON")
    _require(type(m["G7"].get("trajectories")) is int and m["G7"]["trajectories"] >= 32
             and m["G7"].get("synthetic_0_1_masked") is True
             and m["G7"].get("delta_seq") == protocol["loss"]["mismatch"]["delta_seq"], "G7_TRAJECTORIES")
    for key in ("step0_mean_abs", "post_sync_mean_abs"):
        value = m["G7"].get(key)
        _require(type(value) in (int, float) and math.isfinite(value)
                 and 0 <= value <= protocol["engine"]["probe_max_mean_abs_logprob_diff"], "G7_LOGPROB_LIMIT")
    _require(type(m["G11"].get("peak_memory_bytes")) is int and 0 < m["G11"]["peak_memory_bytes"] <= scope["device"]["total_memory_bytes"]
             and type(m["G11"].get("step_seconds")) in (int, float)
             and math.isfinite(m["G11"]["step_seconds"]) and m["G11"]["step_seconds"] > 0, "G11_ACTUAL_MEASUREMENTS")


def _build_items(root: Path, manifest: dict, registry_of, files: dict,
                 allow_empty_heldout: bool = False) -> tuple[dict, dict, dict, dict]:
    """(items, split manifests, input files, images) of the build at root, bound to its manifest.
    allow_empty_heldout: only a union vault whose bound manifest records the opt-in; train stays nonempty."""
    from . import contamination as c
    from .vault import read_actors, read_lineage, read_verifier
    from PIL import Image
    items: dict[str, dict] = {}
    manifests, expected_files, images = {}, {}, {}
    for split in _SPLITS:
        split_root = _safe_root(root / split)
        manifests[split] = _read_json(split_root / "manifest.yaml", files)
        _require(manifest.get("split_manifests", {}).get(split) == files[str(split_root / "manifest.yaml")], "SPLIT_MANIFEST_CHANGED")
        actors, lineages, verifiers = read_actors(split_root), read_lineage(split_root), read_verifier(split_root)
        _require((actors or (allow_empty_heldout and split != "train")) and len(actors) == manifest["counts"][split] and {a.problem_id for a in actors} == set(lineages) == set(verifiers), "SPLIT_COUNT_OR_IDS")
        for name in ("manifest.yaml", "actor_inputs.jsonl", "lineage.jsonl", "verifier_private.jsonl"):
            path = split_root / name
            expected_files[str(path)] = _sha(_read(path))
        for actor in actors:
            pid, lineage = actor.problem_id, lineages[actor.problem_id]
            _require(pid not in items and lineage.split == split and actor.answer_format == verifiers[pid].answer_type
                     and actor.choices == verifiers[pid].choices
                     and not c._source_checks(actor, lineage, registry_of(pid)), "SOURCE_OR_RECORD_IDENTITY")
            items[pid] = {"actor": actor, "lineage": lineage, "verifier": verifiers[pid], "split": split,
                          "structure_hash": lineage.structure_hash, "image_sha256": list(actor.image_sha256),
                          "pixel_sha256": []}
            for rel, sha in zip(actor.image_paths, actor.image_sha256, strict=True):
                path = root / "images" / rel
                _safe_root(path.parent)
                raw = _read(path)
                _require(_sha(raw) == sha, "IMAGE_CHANGED")
                with Image.open(io.BytesIO(raw)) as image:
                    items[pid]["pixel_sha256"].append(c.audit.image_fingerprint(image)["pixels_sha256"])
                expected_files[str(path)], images[rel] = sha, sha
    _require(manifest.get("image_sha256") == images
             and manifest.get("sources") == sorted({i["actor"].source for i in items.values()}), "ORIGINAL_INPUT_BINDING")
    return items, manifests, expected_files, images


def _receipt_inputs(root: Path, manifest: dict, receipt: dict, receipt_sha: str, index: dict, registry: dict,
                    evidence: dict, protocol: dict, files: dict) -> tuple[dict, dict, dict, dict, str]:
    """A build and its own complete exclusion receipt: every receipt, scan-input and evidence check.

    Returns (items, split manifests, input files, re-derived quarantine, evidence hash). The single-receipt
    stages call it on build_root; a union vault calls it once per shard root.
    """
    from . import contamination as c
    from .adjudication import validate as reviewed_candidates
    _require(receipt_sha == manifest.get("exclusion_receipt_sha256")
             and receipt.get("receipt_sha256") == _hash({k: v for k, v in receipt.items() if k != "receipt_sha256"})
             and receipt.get("version") == c.VERSION and receipt.get("status") == "EXCLUSION_CHECKED_WITH_LIMITS"
             and receipt.get("partial_name") is None and receipt.get("blockers") == []
             and receipt.get("training_allowed") is False and receipt.get("approval_granted") is False
             and type(receipt.get("limitations")) is list and len(receipt["limitations"]) >= 2
             and set(receipt.get("steps", {})) == set(c._STEPS)
             and all(v.get("status") == "COMPLETE" for v in receipt["steps"].values()), "EXCLUSION_NOT_COMPLETE")
    _require(receipt.get("index_sha256") == index["sha256"]
             and receipt.get("index_file_sha256") in (None, index["file_sha256"]), "FULL_INDEX_BINDING")
    _require(receipt.get("source_registry_sha256") == _hash(registry)
             and receipt.get("protocol_sha256") == protocol["sha256"], "EXCLUSION_EVIDENCE_BINDING")
    items, manifests, expected_files, images = _build_items(root, manifest, lambda _: registry, files)
    _require(receipt.get("input_files") == expected_files and receipt.get("input_files_sha256") == _hash(expected_files)
             and receipt.get("input_count") == len(items)
             and receipt.get("split_counts") == dict(Counter(i["split"] for i in items.values())), "ORIGINAL_INPUT_BINDING")
    files.update(expected_files)
    _scanner_code(receipt.get("code_sha256", {}), files)
    quarantine = c.family_quarantine(items, reviewed_candidates(receipt, items, files=files))
    _require(receipt.get("quarantined_ids") == quarantine["quarantined_ids"]
             and receipt.get("allowlist") == quarantine["allowlist"], "SELECTED_ID_NOT_EXCLUSION_ALLOWLISTED")
    thresholds = c._calibration(evidence, registry, index["sha256"], protocol["sha256"])
    for name in ("ocr", "sscd", "siglip"):
        c._engine(evidence.get(name), name, evidence, registry, index["sha256"])
    c.ocr_scan(evidence["ocr"], index, items, images)
    siglip_reference = []
    for name in ("sscd", "siglip"):
        stage = evidence[name]
        _, reference = c._vectors(stage.get("benchmark_images"), set(index["images"]), {sha: sha for sha in index["images"]})
        _, input_vectors = c._vectors(stage.get("input_images"), set(images.values()), {sha: sha for sha in images.values()})
        _require(reference and input_vectors, "EMBEDDING_COVERAGE_MISSING")
        # Linear-size norm/dimension validation, not another quadratic search.
        c.cosine_similarities(reference, reference[:1])
        c.cosine_similarities(input_vectors, reference[:1])
        if name == "siglip":
            siglip_reference = reference[:1]
    text_refs = {entry["key"]: _sha(c.normalize_text(entry["text"]).encode()) for entry in index["text_cache"]["entries"] if c.normalize_text(entry["text"])}
    own_text = {pid: _sha(c.normalize_text("\n".join([item["actor"].question, *item["actor"].choices,
                *item["verifier"].printed_text])).encode()) for pid, item in items.items()}
    _, text_vectors = c._vectors(evidence["siglip"].get("benchmark_texts"), set(text_refs), text_refs)
    _, own_vectors = c._vectors(evidence["siglip"].get("input_texts"), set(own_text), own_text)
    _require(text_vectors and own_vectors, "EMBEDDING_TEXT_COVERAGE_MISSING")
    c.cosine_similarities(text_vectors, siglip_reference)
    c.cosine_similarities(own_vectors, siglip_reference)
    _require(bool(thresholds), "CALIBRATION_MISSING")
    evidence_sha = _hash(evidence)
    _require(receipt.get("evidence_sha256") == evidence_sha, "EXCLUSION_EVIDENCE_BINDING")
    return items, manifests, expected_files, quarantine, evidence_sha


def _union_inputs(root: Path, manifest: dict, payload: dict, index: dict, protocol: dict,
                  files: dict) -> tuple[dict, dict, dict, dict, dict]:
    """A pool_freeze vault: each shard receipt checked by _receipt_inputs against its own shard (registry and
    evidence named by the union receipt); the vault bytes equal the frozen manifest's kept items, and each
    item equals its shard record, is allowlisted in that shard's re-derived receipt and shares no family with
    an item quarantined in any shard. No new decision; nothing quarantined is released.
    """
    from . import pool_freeze
    try:
        union, read = pool_freeze.read_union(root)
    except ValueError:
        raise _Blocked("UNION_RECEIPT_BINDING") from None
    files.update(read)
    _require(pool_freeze.verify(Path(union["dataset_manifest"]["path"]), root) == [], "UNION_VAULT_NOT_FROZEN_MANIFEST")
    shards = union["shards"]
    _require(payload["source_registry"] == {"shards": [entry["source_registry"] for entry in shards]}
             and payload["evidence"] == {"shards": [entry["evidence"] for entry in shards]}, "UNION_EVIDENCE_REQUEST")
    registries, shard_items, allowed = [], [], []
    for entry in shards:
        shard_root = _safe_root(Path(entry["root"]))
        shard_manifest = _read_json(shard_root / "manifest.yaml", files)
        receipt, receipt_sha = _artifact(entry["receipt"], files)
        registry, _ = _artifact(entry["source_registry"], files)
        evidence, _ = _artifact(entry["evidence"], files)
        _require(files[str(shard_root / "manifest.yaml")] == entry["build_manifest_sha256"]
                 and type(receipt.get("adjudication")) is dict and type(receipt["adjudication"].get("counts")) is dict
                 and receipt.get("receipt_sha256") == entry["receipt_sha256"]
                 and receipt.get("input_files_sha256") == entry["input_files_sha256"]
                 and receipt["adjudication"]["counts"] == entry["adjudication_counts"]
                 and _hash(evidence.get("calibration")) == entry["calibration_sha256"]
                 and _hash(registry.get("scanner_policy")) == entry["scanner_policy_sha256"], "UNION_SHARD_RECEIPT_BINDING")
        items, _, _, quarantine, evidence_sha = _receipt_inputs(shard_root, shard_manifest, receipt, receipt_sha,
                                                                index, registry, evidence, protocol, files)
        _require(_hash(registry) == entry["source_registry_sha256"] and evidence_sha == entry["evidence_sha256"],
                 "UNION_SHARD_RECEIPT_BINDING")
        registries.append(registry)
        shard_items.append(items)
        allowed.append(set(quarantine["allowlist"]))
    try:
        cross = pool_freeze.cross_shard_allowlist(list(zip(shard_items, allowed, strict=True)))
    except ValueError:
        raise _Blocked("UNION_SHARD_OVERLAP") from None
    owner = {pid: row["shard"] for pid, row in union["items"].items()}
    items, manifests, expected_files, _ = _build_items(root, manifest, lambda pid: registries[owner[pid]], files,
                                                       manifest.get("allow_empty_heldout") is True)
    files.update(expected_files)
    for pid, item in items.items():
        record = shard_items[owner[pid]].get(pid)
        _require(record is not None and all(item[key] == record[key] for key in
                 ("actor", "lineage", "verifier", "split", "structure_hash", "image_sha256", "pixel_sha256")),
                 "UNION_ITEM_NOT_SHARD_RECORD")
        _require(pid in allowed[owner[pid]], "UNION_ITEM_NOT_ALLOWLISTED")
        _require(pid in cross, "UNION_CROSS_SHARD_FAMILY_QUARANTINED")
    return items, manifests, expected_files, {"allowlist": sorted(items), "quarantined_ids": []}, union


def _validate(payload: dict) -> dict:
    """Trusted worker implementation; deliberately not called by the public actor facade."""
    result: dict = {"status": "BLOCKED_NOT_CHECKED", "training_allowed": False, "approval_created": False,
                    "actors": [], "structures": {}, "selected_ids": [], "blockers": [],
                    "limitations": ["Human/GPU provenance declarations are checked, not cryptographically authenticated or rerun",
                                    "This validation grants no cloud/spending authority and is not an OS sandbox"]}
    files: dict[str, str] = {}
    try:
        from . import contamination as c
        from .schema import load_protocol
        root = _safe_root(Path(payload["build_root"]))
        stage = payload.get("stage", "training")
        _require(stage in ("training", "measurement", "limited_smoke", AI_LOCAL_STAGE), "UNKNOWN_VALIDATION_STAGE")
        limited_smoke = stage == "limited_smoke"
        ai_local = stage == AI_LOCAL_STAGE
        measuring = stage != "training"
        manifest = _read_json(root / "manifest.yaml", files)
        build_sha = files[str(root / "manifest.yaml")]
        protocol = load_protocol(_HERE / "protocol_v1.yaml")
        _require(protocol.get("status") in ("DESIGNED", "FROZEN") if measuring
                 else protocol.get("status") == "FROZEN", "PROTOCOL_NOT_FROZEN")
        initial_code = code_hashes()
        _require(manifest.get("format") == "notebook-rl-candidate-build-v1"
                 and manifest.get("status") == "PREPARED_PENDING_GATES" and manifest.get("training_allowed") is False
                 and manifest.get("dataset_code_sha256") == initial_code["dataset.py"], "BUILD_MANIFEST_BINDING")
        if not measuring:
            selection_path, phase_path = Path(payload["selection_path"]), Path(payload["phase_a_report"])
            selection, phase = _read_json(selection_path, files), _read_json(phase_path, files)
        descriptor = payload["index"]
        _require(type(descriptor) is dict and set(descriptor) == {"path", "file_sha256"}, "INDEX_FILE_BINDING_REQUIRED")
        index = c.load_index(Path(descriptor["path"]))
        files[str(Path(descriptor["path"]).absolute())] = index["file_sha256"]
        _require(index["file_sha256"] == descriptor["file_sha256"] and index["full_coverage"]
                 and not index["limitations"], "FULL_INDEX_BINDING")
        registry: dict
        evidence_file_sha: str | None = None
        union_path = root / "contamination" / UNION_RECEIPT
        if union_path.exists() or union_path.is_symlink():
            # A pool_freeze vault: no receipt of its own, the union of its shards' re-verified receipts.
            _require(ai_local, "UNION_RECEIPT_AI_LOCAL_ONLY")
            items, manifests, expected_files, quarantine, union = _union_inputs(root, manifest, payload, index,
                                                                                protocol, files)
            receipt_sha, registry = files[str(union_path)], {}
            registry_sha, evidence_sha = union["source_registry_sha256"], union["evidence_sha256"]
            registry_file_sha, evidence_file_sha = _hash(payload["source_registry"]), _hash(payload["evidence"])
            result["limitations"].append(f"Union of {len(union['shards'])} shard exclusion receipts, each re-verified "
                                         "against its own shard; no new exclusion scan")
        else:
            receipt_path = root / "contamination" / "receipt_v2.json"
            receipt = _read_json(receipt_path, files)
            receipt_sha = files[str(receipt_path)]
            registry, registry_file_sha = _artifact(payload["source_registry"], files)
            evidence = payload["evidence"]
            _require(type(evidence) is dict, "MISSING_ARTIFACT")
            if set(evidence) == {"path", "file_sha256"}:
                evidence, evidence_file_sha = _artifact(evidence, files)
            items, manifests, expected_files, quarantine, evidence_sha = _receipt_inputs(
                root, manifest, receipt, receipt_sha, index, registry, evidence, protocol, files)
            registry_sha = _hash(registry)
        structures = {pid: i["structure_hash"] for pid, i in items.items()}
        for key in ("structure_hash", "image_sha256"):
            owners: dict[str, str] = {}
            for item in items.values():
                for value in [item[key]] if key == "structure_hash" else item[key]:
                    _require(value not in owners or owners[value] == item["split"], "INPUT_SPLIT_LEAK")
                    owners[value] = item["split"]
        train = {pid: i["actor"] for pid, i in items.items() if i["split"] == "train"}
        if limited_smoke:
            authorization = payload["expected_measurement_scope"]["authorization"]
            _require(len(items) <= authorization["max_pool_items"] <= 64
                     and len(train) <= authorization["max_train_items"] <= 24, "SMOKE_POOL_LIMIT")
        selected = sorted(train) if measuring else _selection(
            selection, phase, files[str(phase_path.absolute())], build_sha, train, structures, protocol)
        if limited_smoke or ai_local:
            # Quarantined families may stay in the pool; they are never eligible.
            selected = [pid for pid in selected if pid in quarantine["allowlist"]]
        if limited_smoke:
            _require(bool(selected), "NO_ELIGIBLE_TRAIN_ITEMS")
        _require(set(selected) <= set(quarantine["allowlist"]), "SELECTED_ID_NOT_EXCLUSION_ALLOWLISTED")
        common_binding = {"build_manifest_sha256": build_sha, "exclusion_receipt_sha256": receipt_sha,
                   "input_files_sha256": _hash(expected_files), "protocol_sha256": protocol["sha256"],
                   "index_sha256": index["sha256"], "index_file_sha256": index["file_sha256"],
                   "source_registry_sha256": registry_sha, "evidence_sha256": evidence_sha}
        binding = dict(common_binding)
        if not measuring:
            binding.update(selection_sha256=files[str(selection_path.absolute())],
                           phase_a_report_sha256=files[str(phase_path.absolute())])
            scope = payload["expected_gpu_scope"]
            _gpu_scope(scope, binding, protocol, files)
        data_report, data_sha = _artifact(payload["data_report"], files)
        # A pre-learning data report can be reused unchanged after Phase A. It
        # binds all data and source checks but cannot assert a future selection,
        # device, or GPU outcome. Legacy final-stage reports remain supported.
        if measuring or data_report.get("scope", {}).get("purpose") == "FROZEN_WEIGHT_MEASUREMENT_ONLY":
            data_scope = payload["expected_measurement_scope"] if measuring else data_report["scope"]
            _measurement_scope(data_scope, common_binding, protocol, files, limited_smoke=limited_smoke,
                               ai_local=ai_local)
            data = _proofs(data_report, "D", 9, common_binding, data_scope, data_scope["producer"], files)
        else:
            data = _proofs(data_report, "D", 9, binding, scope, scope["producers"]["data"], files)
        ai_review, ai_review_sha = None, None
        if ai_local:
            descriptor = data["D3"]["measurements"].get("review")
            _require(type(descriptor) is dict and set(descriptor) == {"path", "file_sha256"}, "AI_LOCAL_REVIEW_FILE_REQUIRED")
            ai_review, ai_review_sha = _artifact(descriptor, files)
        blocked, quality = _data_gates(data, items, manifests, expected_files, registry, receipt_sha,
                                       protocol["seeds"][0], limited_smoke=limited_smoke, ai_review=ai_review,
                                       allowlist=set(quarantine["allowlist"]))
        if ai_local:
            skipped = is_skipped_review(ai_review)
            eligibility = ai_local_eligibility({pid: item["actor"].source for pid, item in items.items()},
                                               quarantine["allowlist"], blocked or {}, quality,
                                               SOURCE_REVIEW_SKIPPED if skipped else None)
            if skipped:
                result["limitations"].append(f"source_review={SOURCE_REVIEW_SKIPPED}: no per-source sample review was "
                                             "made; leak-warned items were excluded before selection; contamination/"
                                             "exclusion checks unchanged")
            eligible = set(eligibility["eligible_ids"])
            selected = [pid for pid in selected if pid in eligible]
            _require(bool(selected), "NO_ELIGIBLE_TRAIN_ITEMS")
            binding["ai_review_file_sha256"] = ai_review_sha
            result["limitations"].append("AI-reviewed local pool only; human_verified=false; "
                                         "not human review, Phase A, campaign or training approval")
        if limited_smoke:
            binding["limited_smoke_authorization_sha256"] = _hash(data_scope["authorization"])
            result["limitations"].append("AI/program-reviewed bounded smoke only; human_verified=false; not Phase A or campaign approval")
        if not measuring:
            gpu_report, gpu_sha = _artifact(payload["gpu_report"], files)
            gpu = _proofs(gpu_report, "G", 14, binding, scope, scope["producers"]["gpu"], files)
            _gpu_gates(gpu, scope, protocol)
        _require(code_hashes() == initial_code and all(_sha(_read(Path(path))) == digest for path, digest in files.items()), "INPUT_OR_CODE_CHANGED_DURING_VALIDATION")
        binding.update(data_report_sha256=data_sha,
                       source_registry_artifact_sha256=registry_file_sha,
                       evidence_artifact_sha256=evidence_file_sha or evidence_sha)
        result.update(status="VALIDATED_FOR_MEASUREMENT" if measuring else "VALIDATED_FOR_TRAINING",
                      training_allowed=not measuring, selected_ids=selected,
                      actors=[json.loads(to_json(train[pid])) for pid in selected],
                      structures={pid: structures[pid] for pid in selected}, input_binding=binding,
                      code_sha256=initial_code)
        if measuring:
            result.update(measurement_allowed=True)
        else:
            binding.update(gpu_report_sha256=gpu_sha)
            result.update(selection_sha256=binding["selection_sha256"])
        if ai_local:
            result.update(stage=AI_LOCAL_STAGE, ai_review_file_sha256=ai_review_sha, human_verified=False,
                          eligibility=eligibility)
    except _Blocked as exc:
        result["blockers"] = [str(exc)]
    except Exception:
        # Never serialize a private parser error, annotation or traceback to the actor.
        result["blockers"] = ["MISSING_MALFORMED_OR_CHANGED_PREREQUISITE"]
    return result


def _public_validate(payload: dict, timeout: float) -> dict:
    """One trusted worker for either stage; never read private files in the actor."""
    blocked: dict = {"status": "BLOCKED_NOT_CHECKED", "training_allowed": False, "approval_created": False,
                     "actors": [], "structures": {}, "selected_ids": [], "blockers": []}
    try:
        max_timeout = 600 if payload.get("stage") in ("limited_smoke", AI_LOCAL_STAGE) else 300
        if type(timeout) not in (int, float) or not 0 < timeout <= max_timeout or not math.isfinite(timeout):
            return blocked | {"blockers": ["INVALID_TIMEOUT"]}
        measuring = payload.get("stage", "training") in ("measurement", "limited_smoke", AI_LOCAL_STAGE)
        ai_local = payload.get("stage") == AI_LOCAL_STAGE
        complete = "VALIDATED_FOR_MEASUREMENT" if measuring else "VALIDATED_FOR_TRAINING"
        raw = _json(payload)
        _require(len(raw) <= _MAX_BYTES, "REQUEST_TOO_LARGE_USE_ARTIFACT_DESCRIPTORS")
        process = subprocess.run([sys.executable, "-m", "fine_tuning.notebook_rl.approval", "--worker"],
                                 input=raw, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 timeout=timeout, check=False, close_fds=True)
        _require(process.returncode == 0 and len(process.stdout) <= _MAX_BYTES, "PREFLIGHT_WORKER_FAILED")
        result = json.loads(process.stdout, object_pairs_hook=_unique)
        _require(type(result) is dict and result.get("status") in ("BLOCKED_NOT_CHECKED", complete)
                 and result.get("approval_created") is False, "PREFLIGHT_RESPONSE_INVALID")
        keys = {"status", "training_allowed", "approval_created", "actors", "structures", "selected_ids", "blockers", "limitations"}
        success_keys = {"input_binding", "code_sha256", "measurement_allowed" if measuring else "selection_sha256"}
        if ai_local:
            success_keys |= {"stage", "ai_review_file_sha256", "human_verified", "eligibility"}
        _require(set(result) == (keys | success_keys if result["status"] == complete else keys), "PREFLIGHT_RESPONSE_INVALID")
        if result["status"] == complete:
            actors = [from_json(ActorItem, _json(value).decode()) for value in result["actors"]]
            _require(result.get("training_allowed") is (not measuring)
                     and (not measuring or result.get("measurement_allowed") is True)
                     and actors and [a.problem_id for a in actors] == result["selected_ids"]
                     and len(set(result["selected_ids"])) == len(actors)
                     and set(result["structures"]) == set(result["selected_ids"]), "PREFLIGHT_RESPONSE_INVALID")
            _require(not ai_local or (result["stage"] == AI_LOCAL_STAGE and result["human_verified"] is False
                     and type(result["ai_review_file_sha256"]) is str
                     and re.fullmatch(r"[0-9a-f]{64}", result["ai_review_file_sha256"]) is not None
                     and result["input_binding"].get("ai_review_file_sha256") == result["ai_review_file_sha256"]),
                     "PREFLIGHT_RESPONSE_INVALID")
            eligibility = result.get("eligibility")
            _require(not ai_local or (type(eligibility) is dict and set(eligibility) - {"quality", "source_review"} == {
                "allowlisted_by_source", "quarantined_count", "blocked_sources", "eligible_ids"}
                and eligibility.get("source_review", SOURCE_REVIEW_SKIPPED) == SOURCE_REVIEW_SKIPPED
                and type(eligibility["eligible_ids"]) is list and type(eligibility["quarantined_count"]) is int
                and type(eligibility["blocked_sources"]) is dict and type(eligibility["allowlisted_by_source"]) is dict
                and set(result["selected_ids"]) <= set(eligibility["eligible_ids"])
                and not {a.source for a in actors} & set(eligibility["blocked_sources"])), "PREFLIGHT_RESPONSE_INVALID")
            quality = eligibility.get("quality") if ai_local else None
            # An acceptance-failed item is never eligible or selected; v1 results carry no quality key at all.
            _require(not ai_local or "quality" not in eligibility or (type(quality) is dict
                     and quality.get("rule") == ACCEPTANCE_RULE
                     and type(quality.get("sources")) is dict and all(
                         type(row) is dict and type(row.get("failed_ids")) is list for row in quality["sources"].values())
                     and not {pid for row in quality["sources"].values() for pid in row["failed_ids"]}
                     & set(eligibility["eligible_ids"])), "PREFLIGHT_RESPONSE_INVALID")
            result["actors"] = actors
        else:
            _require(result.get("training_allowed") is False and result.get("actors") == []
                     and result.get("structures") == {} and result.get("selected_ids") == [], "PREFLIGHT_RESPONSE_INVALID")
        return result
    except _Blocked as exc:
        return blocked | {"blockers": [str(exc)]}
    except subprocess.TimeoutExpired:
        return blocked | {"blockers": ["PREFLIGHT_TIMEOUT"]}
    except Exception:
        return blocked | {"blockers": ["PREFLIGHT_WORKER_FAILED"]}


def validate_for_training(build_root: Path, selection_path: Path, *, phase_a_report: Path,
                          index: dict, source_registry: dict, evidence: dict,
                          data_report: dict | None, gpu_report: dict | None,
                          expected_gpu_scope: dict | None, timeout: float = 120) -> dict:
    """No writes, approval creation, private actor-side reads or in-process fallback."""
    return _public_validate({"build_root": str(Path(build_root).absolute()),
        "selection_path": str(Path(selection_path).absolute()), "phase_a_report": str(Path(phase_a_report).absolute()),
        "index": index, "source_registry": source_registry, "evidence": evidence, "data_report": data_report,
        "gpu_report": gpu_report, "expected_gpu_scope": expected_gpu_scope}, timeout)


def validate_for_measurement(build_root: Path, *, index: dict, source_registry: dict, evidence: dict,
                             data_report: dict | None, expected_measurement_scope: dict | None,
                             timeout: float = 120) -> dict:
    """Validate D1–D8 for the complete candidate pool before Phase A exists.

    Success grants only data eligibility for frozen-weight measurements. It is
    not G0–G13, a certified F0 engine, financial authority, or training approval.
    A DESIGNED snapshot is accepted only for measurement; changing any protocol
    bytes (including freezing or caps) invalidates this snapshot and its proofs.
    """
    return _public_validate({"stage": "measurement", "build_root": str(Path(build_root).absolute()),
        "index": index, "source_registry": source_registry, "evidence": evidence, "data_report": data_report,
        "expected_measurement_scope": expected_measurement_scope}, timeout)


def validate_for_limited_smoke(build_root: Path, *, index: dict, source_registry: dict, evidence: dict,
                               data_report: dict | None, expected_measurement_scope: dict | None,
                               timeout: float = 120) -> dict:
    """Explicit opt-in: <=64 program-only records, every record AI reviewed.

    All full-index/OCR/embedding/calibration/exclusion checks remain mandatory.
    No Phase A, training-campaign, financial or human-review approval is issued.
    The bounded gate runner alone may consume this eligibility for its two
    optimizer transitions; ordinary measurement/training APIs remain strict.
    """
    return _public_validate({"stage": "limited_smoke", "build_root": str(Path(build_root).absolute()),
        "index": index, "source_registry": source_registry, "evidence": evidence, "data_report": data_report,
        "expected_measurement_scope": expected_measurement_scope}, timeout)


def validate_for_ai_reviewed_local(build_root: Path, *, index: dict, source_registry: dict, evidence: dict,
                                   data_report: dict | None, expected_measurement_scope: dict | None,
                                   timeout: float = 300) -> dict:
    """Explicit opt-in local pool stage: every measurement check, HUMAN labels allowed.

    The human D3/D7 review is replaced by an AI review record per source over exclusion-allowlisted
    items; the result says human_verified=false and grants measurement eligibility only, never
    training approval. Quarantined families may remain in the pool but are never eligible; a source
    with a failing AI verdict or a short sample is BLOCKED (listed in `eligibility`), and the stage
    fails only when no source remains eligible. A v2 review (acceptance-sampling-v1) blocks a source
    with failures only outside its acceptance bound, else makes just the failed items ineligible, and
    records the per-source counts and bounds in `eligibility.quality`.
    """
    return _public_validate({"stage": AI_LOCAL_STAGE, "build_root": str(Path(build_root).absolute()),
        "index": index, "source_registry": source_registry, "evidence": evidence, "data_report": data_report,
        "expected_measurement_scope": expected_measurement_scope}, timeout)


def validate_for_gate_data(build_root: Path, **request) -> dict:
    """Only gate_session/G13 dispatch here; full measurement never opts in."""
    scope = request.get("expected_measurement_scope")
    limited = type(scope) is dict and scope.get("purpose") == SMOKE_PURPOSE
    if limited:
        request.setdefault("timeout", 300)
    return (validate_for_limited_smoke if limited else validate_for_measurement)(build_root, **request)


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit(2)
    raw = sys.stdin.buffer.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise SystemExit(2)
    try:
        # Trusted imports first; private inputs and actor-supplied data are opened
        # only after installing the one-shot teacher guard in this fresh process.
        from . import contamination, guards
        guards.install_guards(forbidden_roots=[], vault_files=[], allow_vault=True, allow_benchmark=True)
        del contamination
        result = _validate(json.loads(raw, object_pairs_hook=_unique))
        output = _json(result)
        if len(output) > _MAX_BYTES:
            raise SystemExit(2)
        sys.stdout.buffer.write(output)
    except Exception:
        raise SystemExit(2) from None
