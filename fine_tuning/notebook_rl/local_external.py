"""Frozen circuit evaluation inputs and a separate trusted, program-only scorer.

The actor loader opens no gold or canonical solution ledger. This local NF4
diagnostic uses the explicitly fixed temperature-one sampler, not the official
MMMU evaluation recipe. Historical approvals do not approve new training.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fine_tuning.circuit_scoring import VERSION as CIRCUIT_VERSION, score_circuit

from .cli import artifact_path
from .dataset import _bytes, _sha, _write, load_images
from .prompts import user_message
from .schema import ActorItem, validate_actor
from .vault import _read, _safe_root, _unique

SOURCE = "independent-dc-circuit-bridge-v2"
FROZEN_MANIFEST_SHA256 = "74d942d5a75c0117092bf4110e53c9dd20964487b9ce9128cddb591ba6c0fdd6"
EVALUATION_PROTOCOL = {
    "protocol_id": "local-nf4-circuit-a0-qa-eval-v1", "profile": "local-nf4-notebook-rl-v1",
    "sampler": {"temperature": 1., "top_p": 1., "top_k": 0, "presence_penalty": 0.,
                "repetition_penalty": 1.},
    "max_new_tokens": 1024, "context_limit": 8192,
    "image": {"min_pixels": 262144, "max_pixels": 1003520},
    "seed": 3407, "condition": "A0", "view": "qa", "split": "development",
    "prompt_policy": "frozen user_text; bare-letter QA; no added final-answer instruction",
    "scoring": CIRCUIT_VERSION, "official_evaluation_recipe_claimed": False,
}


def _json(path: Path, expected_sha256: str | None = None):
    raw = _read(path)
    if expected_sha256 is not None and _sha(raw) != expected_sha256:
        raise ValueError("local external: frozen artifact changed")
    return json.loads(raw, object_pairs_hook=_unique)


def _rows(path: Path, expected_sha256: str) -> list[dict]:
    raw = _read(path)
    if _sha(raw) != expected_sha256:
        raise ValueError("local external: frozen record ledger changed")
    rows = [json.loads(line, object_pairs_hook=_unique) for line in raw.splitlines()]
    if not rows or any(type(row) is not dict for row in rows):
        raise ValueError("local external: nonempty record objects required")
    return rows


def _descriptor(path: Path) -> dict:
    return {"path": str(path.absolute()), "file_sha256": _sha(_read(path))}


def _actor_inputs(cohort: Path, expected_manifest_sha256: str) -> tuple[dict, list[ActorItem], list[dict]]:
    """Only public inputs, split metadata, manifest and image bytes are opened."""
    root = _safe_root(Path(cohort).absolute())
    manifest = _json(root / "manifest.json", expected_manifest_sha256)
    if (manifest.get("status") != "COHORT_FROZEN" or manifest.get("count") != 120
            or manifest.get("root_count") != 10 or manifest.get("split") != "development"
            or manifest.get("circuit_contract_version") != CIRCUIT_VERSION
            or manifest.get("benchmark_derived") is not False or manifest.get("training_allowed") is not False):
        raise ValueError("local external: registered evaluation-only circuit cohort required")
    inputs = _rows(root / "inputs.jsonl", manifest["inputs_sha256"])
    pairing = _rows(root / "pairing.jsonl", manifest["pairing_sha256"])
    indexed = {r["id"]: r for r in inputs}
    if (len(inputs) != 120 or len(indexed) != 120 or len(pairing) != 120
            or len({r["id"] for r in pairing}) != 120
            or set(indexed) != {r["id"] for r in pairing}
            or any(r.get("benchmark_derived") is not False or r.get("training_allowed") is not False
                   for r in inputs + pairing)
            or any(r.get("split") != "development" for r in pairing)):
        raise ValueError("local external: frozen identities or evaluation permissions changed")
    selected = [r for r in pairing if r.get("condition") == "A0" and r.get("view") == "qa"]
    if (len(selected) != 10 or len({r["root_id"] for r in selected}) != 10
            or len({r["family_id"] for r in selected}) != 10
            or {r["root_id"] for r in selected} != set(manifest.get("root_ids", []))):
        raise ValueError("local external: fixed ten A0 QA roots required")
    actors = []
    for entry in selected:
        row = indexed[entry["id"]]
        paths = row["image_paths"]
        if (type(paths) is not list or len(paths) != 1 or row.get("options") != []
                or row.get("question") != "" or row.get("question_type") != "open"
                or row.get("user_text") != "Choose the source-current option. Give only its letter."
                or any(name in row for name in ("target", "target_text", "gold", "solution", "answer"))):
            raise ValueError("local external: original one-image bare-letter prompt required")
        actor = ActorItem(row["id"], SOURCE, "dc_circuit", "plain", row["user_text"], (),
            "short_text", tuple(paths), tuple(manifest["images"][p] for p in paths))
        validate_actor(actor)
        if any(_sha(_read(root / path)) != digest
               for path, digest in zip(actor.image_paths, actor.image_sha256, strict=True)):
            raise ValueError("local external: selected image changed")
        actors.append(actor)
    return manifest, actors, selected


def load_external_actors(cohort: Path, *, expected_manifest_sha256: str = FROZEN_MANIFEST_SHA256) -> list[ActorItem]:
    """Gold-free actor boundary; options remain in the original circuit image."""
    return _actor_inputs(cohort, expected_manifest_sha256)[1]


def external_messages(actor: ActorItem) -> list[dict]:
    validate_actor(actor)
    if actor.source != SOURCE or actor.choices or len(actor.image_paths) != 1:
        raise ValueError("local external: registered one-image circuit actor required")
    return [user_message(actor.question, len(actor.image_paths))]


def prepare_external_data(cohort: Path, train_actors: list[ActorItem], train_structures: dict[str, str],
                          out: Path, *, source_registry: Path, historical_approval: Path,
                          expected_manifest_sha256: str = FROZEN_MANIFEST_SHA256) -> dict:
    """Prepare private CPU metadata; no new scan, training, model load or evaluation."""
    cohort = Path(cohort).absolute()
    manifest, actors, selected = _actor_inputs(cohort, expected_manifest_sha256)
    if (len(train_actors) != 10 or len({a.problem_id for a in train_actors}) != 10
            or set(train_structures) != {a.problem_id for a in train_actors}
            or len(set(train_structures.values())) != 10):
        raise ValueError("local external: exact selected ten training structures required")
    for actor in train_actors:
        validate_actor(actor)
    registry, historical = _json(Path(source_registry)), _json(Path(historical_approval))
    source = registry.get(SOURCE, {})
    audit = historical.get("contamination_audit", {})
    coverage = audit.get("coverage", {})
    if (source.get("source_known") is not True or source.get("rights_approved") is not True
            or source.get("benchmark_derived") is not False or source.get("parents") != []
            or historical.get("status") != "APPROVED" or historical.get("benchmark_derived") is not False
            or historical.get("sources", {}).get(SOURCE) != source
            or audit.get("status") != "PASS" or audit.get("quarantined") != 0
            or coverage.get("record_count") != 16740 or len(coverage.get("files", [])) != 100):
        raise ValueError("local external: original source ancestry and historical exclusion evidence required")
    train_sources = {a.source for a in train_actors}
    train_hashes = {h for a in train_actors for h in a.image_sha256}
    external_hashes = {h for a in actors for h in a.image_sha256}
    families = {r["family_id"] for r in selected}
    separation = {"source_disjoint": SOURCE not in train_sources,
        "family_id_disjoint": not families & set(train_structures.values()),
        "image_sha256_disjoint": not train_hashes & external_hashes,
        "external_sources": [SOURCE], "training_sources": sorted(train_sources),
        "external_families": sorted(families), "external_root_count": 10,
        "structure_comparison_limit": "Registered circuit family IDs and graph structure IDs use different schemas; domain/source and exact image disjointness are checked, not universal semantic non-overlap."}
    if not all(separation[k] for k in ("source_disjoint", "family_id_disjoint", "image_sha256_disjoint")):
        raise ValueError("local external: selected evaluation overlaps training inputs")
    for actor in actors:
        for image in load_images(actor, cohort):
            image.close()
    out = artifact_path(Path(out).absolute())
    if out.resolve().is_relative_to(cohort.resolve()):
        raise ValueError("local external: output must be outside immutable cohort")
    if out.exists():
        raise FileExistsError("local external: preserve previous preparation")
    out.mkdir(mode=0o700)
    _write(out / "actors.json", _bytes([asdict(a) for a in actors]) + b"\n")
    _write(out / "protocol.json", _bytes(EVALUATION_PROTOCOL) + b"\n")
    result = {"status": "CPU_EXTERNAL_INPUTS_PREPARED_NOT_EVALUATED", "training_allowed": False,
        "GPU_evaluation_completed": False, "official_evaluation_recipe_claimed": False,
        "cohort": str(cohort), "image_root": str(cohort), "n_problems": 10,
        "problem_ids": [a.problem_id for a in actors], "separation": separation,
        "actors": _descriptor(out / "actors.json"), "evaluation_protocol": _descriptor(out / "protocol.json"),
        "source_binding": {"manifest_sha256": expected_manifest_sha256,
            "inputs_sha256": manifest["inputs_sha256"], "pairing_sha256": manifest["pairing_sha256"],
            "gold_sha256": manifest["gold_sha256"]},
        "source_registry": _descriptor(Path(source_registry)), "historical_approval": _descriptor(Path(historical_approval)),
        "historical_exclusion": {"files": 100, "rows": 16740, "index_sha256": audit["index_sha256"],
            "new_scan_performed": False, "used_as_current_training_approval": False},
        "limitations": ["Existing synthetic DC-circuit development cohort, not a fresh sealed or professional transfer benchmark.",
            "Historical fingerprint/source evidence reused; no new OCR/embedding/full-index scan or human render certification.",
            "New local NF4 temperature-one diagnosis; no comparison with MMMU V8 558/900."]}
    _write(out / "preparation.json", _bytes(result) + b"\n")
    return result


def score_external_responses(cohort: Path, responses: list[dict], *,
                             expected_manifest_sha256: str = FROZEN_MANIFEST_SHA256) -> dict:
    """Trusted scorer: open gold only after complete actual-response coverage checks."""
    manifest, actors, selected = _actor_inputs(cohort, expected_manifest_sha256)
    ids = [a.problem_id for a in actors]
    if (type(responses) is not list or any(type(r) is not dict
            or set(r) != {"problem_id", "text", "finish_reason"}
            or type(r["problem_id"]) is not str or type(r["text"]) is not str
            or r["finish_reason"] not in ("stop", "eos", "length") for r in responses)):
        raise ValueError("local external: actual response identity/text/finish fields required")
    by_id = {r["problem_id"]: r for r in responses}
    if len(by_id) != len(responses) or not set(by_id) <= set(ids):
        raise ValueError("local external: duplicate or unexpected response ID")
    base = {"status": "INCOMPLETE", "accuracy": None, "n_expected": 10, "n_received": len(responses),
        "problem_ids": ids, "cohort_manifest_sha256": expected_manifest_sha256,
        "evaluation_protocol_sha256": _sha(_bytes(EVALUATION_PROTOCOL)), "gold_opened": False,
        "execution_conditions_verified": False, "improvement_claimed": False}
    if set(by_id) != set(ids):
        return base | {"missing_ids": sorted(set(ids) - set(by_id))}
    root = Path(cohort).absolute()
    gold = _rows(root / "gold.jsonl", manifest["gold_sha256"])
    keyed = {r["id"]: r for r in gold}
    if len(keyed) != 120 or len(gold) != 120 or any(
        row["id"] not in keyed or any(keyed[row["id"]].get(k) != row[k]
            for k in ("root_id", "family_id", "condition", "view", "split")) for row in selected):
        raise ValueError("local external: gold identity or pairing changed")
    scored = []
    for pid in ids:
        target = keyed[pid]
        if (target.get("metric_kind") != "circuit_qa" or target.get("circuit_contract_version") != CIRCUIT_VERSION
                or target.get("training_allowed") is not False or target.get("benchmark_derived") is not False):
            raise ValueError("local external: trusted QA target contract changed")
        response = by_id[pid]
        score = score_circuit(response["text"], target["target"], "qa", response["finish_reason"],
                              contract_version=CIRCUIT_VERSION)
        scored.append({"problem_id": pid, "finish_reason": response["finish_reason"], **score})
    correct = sum(r["correct"] for r in scored)
    return base | {"status": "COMPLETE", "gold_opened": True, "n_correct": correct,
        "accuracy": correct/10, "rows": scored, "responses_sha256": _sha(_bytes(responses))}


def compare_external_scores(before: dict, after: dict) -> dict:
    """Paired arithmetic only; the caller must attest actual identical run conditions."""
    if (before.get("status") != "COMPLETE" or after.get("status") != "COMPLETE"
            or any(before.get(k) != after.get(k) for k in
                ("problem_ids", "cohort_manifest_sha256", "evaluation_protocol_sha256"))):
        raise ValueError("local external: complete identical fixed inputs/protocol required")
    old = {r["problem_id"]: r["correct"] for r in before["rows"]}
    new = {r["problem_id"]: r["correct"] for r in after["rows"]}
    if (set(old) != set(new) or set(old) != set(before["problem_ids"]) or len(old) != 10
            or len(before["rows"]) != 10 or len(after["rows"]) != 10
            or any(type(value) is not bool for value in (*old.values(), *new.values()))):
        raise ValueError("local external: paired score coverage changed")
    return {"n_paired": 10, "before_correct": sum(old.values()), "after_correct": sum(new.values()),
        "wrong_to_correct": sum(not old[k] and new[k] for k in old),
        "correct_to_wrong": sum(old[k] and not new[k] for k in old),
        "net_correct_change": sum(new.values()) - sum(old.values()),
        "execution_conditions_verified": False, "improvement_claimed": False}
