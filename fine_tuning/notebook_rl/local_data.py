"""Rebind unchanged reviewed graph inputs for explicitly authorized local NF4 work.

Run this trusted CPU preparation before actor guards. Existing assembly and its
isolated validator check private evidence; only actors and metadata are emitted.
No new scan, human review, GPU success or cloud authority is asserted here.
"""
from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
from collections.abc import Collection, Iterable
from dataclasses import replace
from functools import partial
from datetime import datetime, timezone
from pathlib import Path

from . import approval, pool_freeze, smoke_data_evidence as evidence
from .cli import artifact_path
from .dataset import SPLITS, _bytes, _write
from .schema import ActorItem, from_json, load_protocol, to_json
from .vault import _read, read_actors, read_lineage

PROFILE = "local-nf4-notebook-rl-v1"
ACTOR_MODE_POLICY = "all-required-observations-v1"
LOCAL_PREFLIGHT_SECONDS = 600
AUTHORIZATION_QUOTE = "로컬에서 진행하는 나머지 검증과 테스트는 전부 승인할테니 계속 실험 및 검증과 테스트를 하면서 파인튜닝 파이프라인 완성과 오답노트형 자기개선 학습 구조를 완전히 구현해놔줘."
STAGED_AUTHORIZATION_QUOTE = "먼저 1~2문항으로 실제 RL 업데이트 1회→가중치 변경→저장→새 프로세스 재로딩을 검증하고, 성공 후 사전 고정한 학습 10문항과 구조·출처가 겹치지 않는 별도 평가 문항으로 확장하라."
_HERE = Path(__file__).resolve().parent


def _write_json(path: Path, value: dict | list) -> dict:
    _write(path, _bytes(value) + b"\n")
    return evidence._descriptor(path)


def project_observation_actors(actors: list[ActorItem], protocol: dict) -> list[ActorItem]:
    """Change elicitation only; private answers and reference slots are not inputs."""
    if protocol.get("actor_mode_policy") != ACTOR_MODE_POLICY:
        raise ValueError("local data: explicit observation-elicitation policy required")
    if not actors or len({a.problem_id for a in actors}) != len(actors):
        raise ValueError("local data: nonempty unique actors required")
    for actor in actors:
        to_json(actor)
    return [replace(actor, mode="obs") for actor in actors]


def _split_summary(root: Path, selected: list[str]) -> dict:
    rows = {}
    for split in SPLITS:
        actors, lineage = read_actors(root / split), read_lineage(root / split)
        if set(lineage) != {a.problem_id for a in actors} or any(
                lineage[a.problem_id].split != split for a in actors):
            raise ValueError("local data: split identity mismatch")
        rows[split] = (actors, lineage)
    train = {a.problem_id: a for a in rows["train"][0]}
    if not set(selected) <= train.keys():
        raise ValueError("local data: selected IDs must remain training records")
    chosen = [train[pid] for pid in selected]
    sources, families = {a.source for a in chosen}, {a.family for a in chosen}
    structures = {rows["train"][1][pid].structure_hash for pid in selected}
    result = {}
    for split, (actors, lineage) in rows.items():
        own_sources, own_families = {a.source for a in actors}, {a.family for a in actors}
        own_structures = {v.structure_hash for v in lineage.values()}
        result[split] = {"count": len(actors), "structure_count": len(own_structures),
            "sources": sorted(own_sources), "families": sorted(own_families),
            "selected_structure_overlap": len(structures & own_structures),
            "selected_source_overlap": sorted(sources & own_sources),
            "selected_family_overlap": sorted(families & own_families),
            "promoted_to_training": False,
            "qualifies_source_and_structure_disjoint_evaluation": split != "train"
                and not sources & own_sources and not structures & own_structures}
    return result


def prepare_local_data(previous_request: Path, selection: Path, protocol: dict, out: Path) -> dict:
    """Write new proof bindings, then run the existing isolated D1–D8 validator.

    Data proof bindings retain the unchanged protocol_v1 data-contract identity;
    the new local execution protocol is bound separately. Blocked validation is
    preserved on disk and never represented as training/GPU permission.
    """
    authority = protocol.get("local_authorization", {})
    if (protocol.get("profile", protocol.get("protocol_id")) != PROFILE
            or type(authority) is not dict or authority.get("quote") != AUTHORIZATION_QUOTE
            or type(authority.get("reference")) is not str or not authority["reference"].strip()
            or authority.get("cloud_allowed") is not False):
        raise ValueError("local data: explicit quoted local NF4 authorization required")
    protocol_sha = evidence._hash(json.loads(json.dumps({k: v for k, v in protocol.items() if k != "sha256"})))
    if "sha256" in protocol and protocol["sha256"] != protocol_sha:
        raise ValueError("local data: execution protocol hash mismatch")
    if protocol.get("actor_mode_policy") != ACTOR_MODE_POLICY:
        raise ValueError("local data: explicit observation-elicitation policy required")
    previous_request, selection = Path(previous_request).absolute(), Path(selection).absolute()
    original_request = evidence._descriptor(previous_request)
    original_selection = evidence._descriptor(selection)
    request, old_selection = evidence._bound_json(original_request), evidence._bound_json(original_selection)
    if set(request) != {"build_root", "index", "source_registry", "evidence", "data_report",
                        "expected_measurement_scope"}:
        raise ValueError("local data: exact previous data request required")
    report = evidence._bound_json(request["data_report"])
    if (report.get("status") != "COMPLETE" or report.get("scope") != request["expected_measurement_scope"]
            or set(report.get("gates", {})) != {f"D{i}" for i in range(1, 9)}):
        raise ValueError("local data: complete bound previous data report required")
    try:
        old_proofs = approval._proofs(report, "D", 9, report["binding"], report["scope"],
                                     report["scope"]["producer"], {})
    except approval._Blocked as exc:
        raise ValueError(f"local data: prior data proof invalid: {exc}") from None
    proofs = [old_proofs[f"D{i}"] for i in range(1, 9)]
    if any(p.get("components") != proofs[0].get("components")
           or p.get("metadata_rebind") != proofs[0].get("metadata_rebind") for p in proofs):
        raise ValueError("local data: previous components or metadata mappings differ")
    root = Path(request["build_root"])
    chosen = old_selection.get("problem_ids")
    if (type(chosen) is not list or len(chosen) != 10
            or any(type(pid) is not str for pid in chosen) or len(set(chosen)) != 10):
        raise ValueError("local data: preserve the existing ten selected problem IDs")
    splits = _split_summary(root, chosen)
    train = {a.problem_id: a for a in read_actors(root / "train")}
    lineage = read_lineage(root / "train")
    structures = {pid: lineage[pid].structure_hash for pid in chosen}
    image_counts = {pid: len(train[pid].image_paths) for pid in chosen}
    if (splits["train"]["count"] != 16 or len(set(structures.values())) != 10
            or old_selection.get("structures") != structures or old_selection.get("image_counts") != image_counts
            or sorted(image_counts.values()) != [1]*5 + [2]*5):
        raise ValueError("local data: original train16/distinct10/image selection changed")
    out = artifact_path(Path(out).absolute())
    if out.exists():
        raise FileExistsError("local data: preserve previous preparation outputs")
    scope = copy.deepcopy(request["expected_measurement_scope"])
    scope["code_sha256"] = approval.code_hashes()
    scope["protocol_sha256"] = load_protocol(_HERE / "protocol_v1.yaml")["sha256"]
    scope["authorization"] = {
        "approval_reference": authority["reference"] + "\nUser: " + AUTHORIZATION_QUOTE
            + "\nUser: " + STAGED_AUTHORIZATION_QUOTE
            + "\nCPU input revalidation only; local execution requires staged hardware checks; no cloud authority.",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_policy": approval.SMOKE_REVIEW_POLICY, "max_pool_items": 64,
        "max_train_items": 16, "max_optimizer_updates": 2, "training_campaign_allowed": False}
    binding = copy.deepcopy(report["binding"])
    if binding.get("protocol_sha256") != scope["protocol_sha256"]:
        raise ValueError("local data: original data protocol changed")
    # Existing assembly checks every reused source, input byte, review and exclusion binding.
    evidence.assemble(proofs[0]["components"], root,
        evidence._descriptor(root / "contamination/receipt_v2.json"), binding, scope, out / "data-gates",
        source_registry=request["source_registry"], metadata_rebind=proofs[0]["metadata_rebind"])
    fresh_request = request | {"data_report": evidence._descriptor(out / "data-gates/data_report.json"),
                               "expected_measurement_scope": scope}
    request_descriptor = _write_json(out / "data_request.json", fresh_request)
    checked = approval.validate_for_gate_data(root, timeout=LOCAL_PREFLIGHT_SECONDS,
        **{k: v for k, v in fresh_request.items() if k != "build_root"})
    serial = checked | {"actors": [json.loads(to_json(a)) for a in checked["actors"]]}
    validated = _write_json(out / "validated_data.json", serial)
    ready = (checked.get("status") == "VALIDATED_FOR_MEASUREMENT"
             and checked.get("measurement_allowed") is True and checked.get("blockers") == []
             and checked.get("training_allowed") is False and checked.get("code_sha256") == scope["code_sha256"]
             and set(checked.get("selected_ids", [])) == set(train)
             and checked.get("structures") == {pid: lineage[pid].structure_hash for pid in train})
    if evidence._descriptor(previous_request) != original_request or evidence._descriptor(selection) != original_selection:
        raise ValueError("local data: original request or selection changed during revalidation")
    result = {"format": "notebook-rl-local-data-v1", "status": "LOCAL_DATA_REVALIDATED" if ready else "BLOCKED",
        "profile": PROFILE, "local_protocol_sha256": protocol_sha,
        "training_allowed": False, "gpu_verified": False, "cloud_allowed": False,
        "new_exclusion_scan_performed": False, "new_human_review_performed": False,
        "previous_request": original_request, "previous_selection": original_selection,
        "build_root": str(root), "image_root": str(root / "images"), "data_request": request_descriptor,
        "validated_data": validated, "split_summary": splits,
        "limitations": ["AI/program-reviewed bounded graph pool; human_verified=false.",
            "Existing heldout splits share graph sources/family; source-disjoint transfer requires separate data.",
            "The retained legacy data-only scope has a two-update smoke bound; it grants no new execution authority.",
            "Data-contract validation is separate from the new local NF4 execution protocol and hardware gates."]}
    if ready:
        public_actors = {a.problem_id: a for a in checked["actors"]}
        originals = [public_actors[pid] for pid in chosen]
        projected = project_observation_actors(originals, protocol)
        result["actors"] = _write_json(out / "actors.json", [json.loads(to_json(a)) for a in originals])
        result["execution_actors"] = _write_json(out / "execution_actors.json",
                                                [json.loads(to_json(a)) for a in projected])
        result["actor_projection"] = {"policy": ACTOR_MODE_POLICY,
                                      "changed_ids": [a.problem_id for a in originals if a.mode != "obs"]}
        result["selection"] = _write_json(out / "selection.json", {
            "status": "LOCAL_FIXED_TEN_NOT_TRAINING_APPROVAL", "profile": PROFILE,
            "local_protocol_sha256": result["local_protocol_sha256"], "problem_ids": chosen,
            "structures": structures, "image_counts": image_counts, "previous_selection": original_selection})
    else:
        result["blockers"] = checked.get("blockers") or ["LOCAL_CURRENT_SCOPE_OR_TRAIN_SELECTION_MISMATCH"]
    _write_json(out / "preparation.json", result)
    return result


# ---------------------------------------------------------------- AI-reviewed mixed pool (opt-in)

POOL_FORMAT = "notebook-rl-local-pool-data-v1"
POOL_STATUS = "LOCAL_POOL_DATA_VALIDATED"
POOL_RULE_VERSION = "pool-local-selection-v1"
POOL_MODE_POLICY = "preserve-built-mode-v1"
POOL_PROPOSAL_METHOD = "seed-keyed-sha256-round-robin-v1"
HELD_OUT_SPLITS = ("dev", "sealed", "style_sealed")
_RULE_KEYS = {"version", "build_manifest_sha256", "seed", "minimal", "fixed", "evaluation"}


def public_pool(root: Path) -> dict:
    """Public actors and lineage only; verifier gold is never opened here."""
    rows = {}
    for split in SPLITS:
        actors, lineage = read_actors(root / split), read_lineage(root / split)
        if set(lineage) != {a.problem_id for a in actors} or any(lineage[a.problem_id].split != split for a in actors):
            raise ValueError("local pool: split identity mismatch")
        rows.update({a.problem_id: (a, lineage[a.problem_id]) for a in actors})
    return rows


def _order(problem_ids, seed: int) -> list[str]:
    return sorted(problem_ids, key=lambda pid: evidence._sha(f"{seed}:{pid}".encode()))


def pool_allowlist(root: Path, public: dict) -> set[str]:
    """Allowlist of the build's bound complete exclusion receipt (public metadata, no gold).

    Only a proposal filter: the ai_reviewed_local approval re-derives it and is what prepare enforces.
    A frozen union vault keeps only items allowlisted in their shard receipt (pool_freeze.read_union).
    """
    if (root / "contamination" / pool_freeze.UNION_FILE).exists():
        kept = set(pool_freeze.read_union(root)[0]["items"])
        if not kept or not kept <= public.keys():
            raise ValueError("local pool: union vault items must be public pool items")
        return kept
    raw = _read(root / "contamination/receipt_v2.json")
    receipt = json.loads(raw)
    allowlist = receipt.get("allowlist")
    if (json.loads(_read(root / "manifest.yaml")).get("exclusion_receipt_sha256") != evidence._sha(raw)
            or receipt.get("status") != "EXCLUSION_CHECKED_WITH_LIMITS" or type(allowlist) is not list
            or not set(allowlist) <= public.keys()):
        raise ValueError("local pool: bound complete exclusion receipt with an allowlist required")
    return set(allowlist)


def _renders(actor: ActorItem, prompts: dict) -> bool:
    from .prompts import turn1_text
    try:
        turn1_text(actor, prompts)
    except ValueError:
        return False
    return True


def pool_subjects(root: Path, public: dict) -> dict[str, str | None]:
    """Subjects reconstructed from hf_pool's deterministic order and verified counts; None when not derivable."""
    subjects: dict[str, str | None] = dict.fromkeys(public)
    path = root / "pool_manifest.json"
    if not path.is_file():
        return subjects
    manifest = json.loads(path.read_bytes())
    counts, native = manifest.get("counts"), manifest.get("native_ids")
    if type(counts) is not dict or type(native) is not dict or set(native) != set(public):
        return subjects
    staged = {pid for pid, value in native.items() if type(value) is list and len(value) == 2 and value[1] == pid}
    by_source: dict[str, list[str]] = {}
    for pid, (actor, _) in public.items():
        if pid not in staged:
            by_source.setdefault(actor.source, []).append(pid)
    for source, pids in by_source.items():
        try:
            # hf_pool.select accepts sources, subjects and splits in sorted order under one running index.
            pids.sort(key=lambda pid: int(pid.split(":")[3]))
            names = sorted({subject for split in SPLITS for subject in counts.get(split, {}).get(source, {})})
            blocks = [(subject, split, counts[split][source][subject]) for subject in names for split in sorted(SPLITS)
                      if counts.get(split, {}).get(source, {}).get(subject)]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        expanded = [(subject, split) for subject, split, n in blocks for _ in range(n)]
        if len(expanded) == len(pids) and all(public[pid][1].split == split for pid, (_, split) in zip(pids, expanded)):
            subjects.update({pid: subject for pid, (subject, _) in zip(pids, expanded)})
    for pid in staged:
        split, source = public[pid][1].split, public[pid][0].source
        names = list(counts.get(split, {}).get(source, {}))
        subjects[pid] = names[0] if len(names) == 1 else None
    return subjects


def check_pool_rule(rule: dict, public: dict, prompts: dict, build_sha256: str, *,
                    eligible: Collection[str] | None = None) -> dict:
    """Validate a pre-fixed selection rule against the build; returns the bound selection summary.

    `eligible` (allowlisted IDs outside blocked sources, from the approval) rejects every other ID.
    """
    from .approval import label_class
    from .local_run import minimal_actors

    if type(rule) is not dict or set(rule) != _RULE_KEYS or rule["version"] != POOL_RULE_VERSION:
        raise ValueError("local pool: exact pool-local-selection-v1 rule required")
    lists = [rule[key] for key in ("minimal", "fixed", "evaluation")]
    if (rule["build_manifest_sha256"] != build_sha256 or type(rule["seed"]) is not int
            or any(type(ids) is not list or any(type(pid) is not str for pid in ids) or len(set(ids)) != len(ids)
                   for ids in lists)
            or len(rule["minimal"]) != 2 or len(rule["fixed"]) != 10 or not rule["evaluation"]
            or not set(rule["fixed"] + rule["evaluation"]) <= public.keys()
            or set(rule["fixed"]) & set(rule["evaluation"])):
        raise ValueError("local pool: rule build, ID lists or sizes are invalid")
    if eligible is not None and not set(rule["minimal"] + rule["fixed"] + rule["evaluation"]) <= set(eligible):
        raise ValueError("local pool: every rule ID must be exclusion-allowlisted and outside blocked sources")
    actor = {pid: public[pid][0] for pid in rule["fixed"] + rule["evaluation"]}
    lineage = {pid: public[pid][1] for pid in actor}
    fixed = [actor[pid] for pid in rule["fixed"]]
    if (any(lineage[pid].split != "train" for pid in rule["fixed"])
            or any(lineage[pid].split not in HELD_OUT_SPLITS for pid in rule["evaluation"])):
        raise ValueError("local pool: fixed IDs must be train records and evaluation IDs held-out records")
    if len({lineage[pid].structure_hash for pid in rule["fixed"]}) != 10:
        raise ValueError("local pool: ten distinct fixed structures required")
    # The minimal pair is the existing single/multi-image selection over the listed fixed order.
    if rule["minimal"] != [a.problem_id for a in minimal_actors(fixed)]:
        raise ValueError("local pool: minimal IDs must be the first single- and two-image fixed actors")
    train_keys = {(key, value) for pid in rule["fixed"] for key, value in (
        ("source", actor[pid].source), ("structure", lineage[pid].structure_hash),
        *(("image", sha) for sha in actor[pid].image_sha256))}
    if any((key, value) in train_keys for pid in rule["evaluation"] for key, value in (
            ("source", actor[pid].source), ("structure", lineage[pid].structure_hash),
            *(("image", sha) for sha in actor[pid].image_sha256))):
        raise ValueError("local pool: evaluation sources, structure families and images must be disjoint from training")
    if not all(_renders(a, prompts) for a in actor.values()):
        raise ValueError("local pool: every selected actor must render under the bound protocol prompts")
    return {"minimal_ids": rule["minimal"], "problem_ids": rule["fixed"],
            "pair_ids": [rule["fixed"][i:i + 2] for i in range(0, 10, 2)], "evaluation_ids": rule["evaluation"],
            "structures": {pid: lineage[pid].structure_hash for pid in actor},
            "image_counts": {pid: len(actor[pid].image_paths) for pid in actor},
            "sources": {pid: actor[pid].source for pid in actor}, "modes": {pid: actor[pid].mode for pid in actor},
            "splits": {pid: lineage[pid].split for pid in actor},
            "label_classes": {pid: label_class(lineage[pid]) for pid in actor}}


def _round_robin(public: dict, candidates: list[str], k: int, taken: set[str]) -> list[str]:
    """Round-robin over sources in hash order; one structure family at most once."""
    groups: dict[str, list[str]] = {}
    for pid in candidates:
        groups.setdefault(public[pid][0].source, []).append(pid)
    chosen: list[str] = []
    while len(chosen) < k and any(groups.values()):
        for queue in groups.values():
            while queue and public[queue[0]][1].structure_hash in taken:
                queue.pop(0)
            if queue and len(chosen) < k:
                pid = queue.pop(0)
                chosen.append(pid)
                taken.add(public[pid][1].structure_hash)
    return chosen


def propose_pool_rule(build_root: Path, prompts: dict, *, seed: int = 3407, evaluation_count: int = 12,
                      eligible: Collection[str] | None = None, version: str = "v1",
                      excluded_sources: Collection[str] | None = None, sizes: tuple[int, int] | None = None) -> dict:
    """Label-blind proposal: seed-keyed hash order and public actor/lineage metadata only (no gold, no outputs).

    Only exclusion-allowlisted IDs are proposed; `eligible` (the approval's eligible IDs) also skips
    blocked sources. `version="v2"` proposes a pool-local-selection-v2 rule (existing datasets only, N/K
    `sizes`, extra `excluded_sources` beside the defaults); v1 takes neither option.
    """
    from .approval import label_class

    root = Path(build_root).absolute()
    if version == "v2":
        return _propose_v2(root, prompts, seed=seed, evaluation_count=evaluation_count, eligible=eligible,
                           excluded_sources=excluded_sources or (), sizes=sizes or (10, 2))
    if version != "v1" or excluded_sources is not None or sizes is not None:
        raise ValueError("local pool: v1 proposals take no v2 options; use version='v2'")
    public = public_pool(root)
    allowed = pool_allowlist(root, public)
    if eligible is not None:
        allowed &= set(eligible)
    usable = [pid for pid in _order(allowed, seed) if _renders(public[pid][0], prompts)]

    def pick(candidates: list[str], k: int, taken: set[str]) -> list[str]:
        return _round_robin(public, candidates, k, taken)

    train = [pid for pid in usable if public[pid][1].split == "train"]
    generator = {pid for pid in train if label_class(public[pid][1]) == "GENERATOR"}
    taken: set[str] = set()
    hf = pick([pid for pid in train if pid not in generator], 6, taken)
    obs = [pid for pid in train if pid in generator and public[pid][0].mode == "obs"]
    multi = pick([pid for pid in obs if len(public[pid][0].image_paths) == 2], 1, taken)
    obs_rest = pick([pid for pid in obs if pid not in multi], 1, taken)
    plain = pick([pid for pid in train if pid in generator and public[pid][0].mode == "plain"], 2, taken)
    single = [pid for pid in hf if len(public[pid][0].image_paths) == 1]
    if (len(hf) != 6 or len({public[pid][0].source for pid in hf}) < 4 or not single or len(multi) != 1
            or len(obs_rest) != 1 or len(plain) != 2):
        raise ValueError("local pool: need 6 non-generator items from >=4 sources (one single-image), "
                         "2 obs generator items (one with two images) and 2 plain generator items")
    # Minimal first (one single-image slotless item, then the two-image obs item), so the existing
    # first-single/first-multi selection over the fixed order reproduces it exactly.
    fixed = [single[0], multi[0], *_order([pid for pid in hf if pid != single[0]] + obs_rest + plain, seed)]
    train_sources = {public[pid][0].source for pid in fixed}
    held = [pid for pid in usable if public[pid][1].split in HELD_OUT_SPLITS
            and public[pid][0].source not in train_sources]
    evaluation = pick(held, evaluation_count, {public[pid][1].structure_hash for pid in fixed})
    if len(evaluation) != evaluation_count:
        raise ValueError("local pool: not enough held-out items from sources outside the fixed set")
    rule = {"version": POOL_RULE_VERSION, "build_manifest_sha256": evidence._sha(_read(root / "manifest.yaml")),
            "seed": seed, "minimal": [single[0], multi[0]], "fixed": fixed, "evaluation": evaluation}
    check_pool_rule(rule, public, prompts, rule["build_manifest_sha256"], eligible=allowed)
    return rule


def _pool_protocol(protocol: dict) -> str:
    from .local_profile import REPAIR_PROTOCOL_ID

    authority = protocol.get("local_authorization", {})
    if (protocol.get("profile", protocol.get("protocol_id")) != PROFILE
            or type(authority) is not dict or authority.get("quote") != AUTHORIZATION_QUOTE
            or type(authority.get("reference")) is not str or not authority["reference"].strip()
            or authority.get("cloud_allowed") is not False):
        raise ValueError("local pool: explicit quoted local NF4 authorization required")
    protocol_sha = evidence._hash(json.loads(json.dumps({k: v for k, v in protocol.items() if k != "sha256"})))
    if "sha256" in protocol and protocol["sha256"] != protocol_sha:
        raise ValueError("local pool: execution protocol hash mismatch")
    if protocol.get("protocol_id") != REPAIR_PROTOCOL_ID:
        raise ValueError("local pool: the pool path is wired for the repair-notebook protocol only")
    return protocol_sha


def prepare_pool_data(build_root: Path, approval_request: Path, rule_path: Path, protocol: dict, out: Path, *,
                      frozen_approval: dict | None = None) -> dict:
    """Bind a pre-fixed pool rule to an ai_reviewed_local validation; built actor modes are kept.

    A pool-local-selection-v2 rule writes notebook-rl-local-pool-data-v2 (N/K from the bound scale group and a
    separate probe_actors.json); a pool-proof-selection-v1 rule writes notebook-rl-local-pool-data-proof-v1 (v2
    layout, its sentinels as the evaluation actors) once the rule equals the label-blind derivation of its selection
    source; a v1 rule keeps the v1 format unchanged. `frozen_approval` ({path, file_sha256}; explicit opt-in,
    frozen-approval-verify-v1) stands in for the fresh approval: the pinned saved approval output of a union vault is
    verified by hashes (local_run.frozen_approval_output) and copied byte for byte as validated_data.json.
    """
    from .local_profile import local_prompts

    protocol_sha = _pool_protocol(protocol)
    root = Path(build_root).absolute()
    request_descriptor = evidence._descriptor(Path(approval_request).absolute())
    rule_descriptor = evidence._descriptor(Path(rule_path).absolute())
    request, rule = evidence._bound_json(request_descriptor), evidence._bound_json(rule_descriptor)
    if (set(request) != {"build_root", "index", "source_registry", "evidence", "data_report",
                         "expected_measurement_scope"} or request["build_root"] != str(root)
            or request["expected_measurement_scope"].get("purpose") != approval.AI_LOCAL_PURPOSE):
        raise ValueError("local pool: exact ai_reviewed_local data request for this build required")
    public = public_pool(root)
    build_sha = evidence._sha(_read(root / "manifest.yaml"))
    version = rule.get("version") if type(rule) is dict else None
    proof = version == POOL_RULE_VERSION_PROOF
    v2 = proof or version == POOL_RULE_VERSION_V2
    check = (partial(check_proof_rule, sizes=pool_sizes(protocol), subjects=proof_subjects(root, public)) if proof
             else partial(check_pool_rule_v2, sizes=pool_sizes(protocol)) if v2 else check_pool_rule)
    summary = check(rule, public, local_prompts(protocol), build_sha)
    out = artifact_path(Path(out).absolute())
    if out.exists():
        raise FileExistsError("local pool: preserve previous preparation outputs")
    verification = None
    if frozen_approval is None:
        checked = approval.validate_for_ai_reviewed_local(root, timeout=LOCAL_PREFLIGHT_SECONDS,
            **{k: v for k, v in request.items() if k != "build_root"})
    else:
        from .local_run import frozen_approval_output

        saved, _, verification = frozen_approval_output(root, frozen_approval)
        checked = saved | {"actors": [from_json(ActorItem, json.dumps(row)) for row in saved["actors"]]}
    eligibility = checked.get("eligibility") or {}  # shape checked by the facade; absent when blocked
    eligible = set(eligibility.get("eligible_ids", ()))
    train = {pid: row for pid, row in public.items() if row[1].split == "train" and pid in eligible}
    ready = (checked.get("status") == "VALIDATED_FOR_MEASUREMENT" and checked.get("measurement_allowed") is True
             and checked.get("blockers") == [] and checked.get("training_allowed") is False
             and checked.get("stage") == approval.AI_LOCAL_STAGE and checked.get("human_verified") is False
             and checked.get("code_sha256") == approval.code_hashes()
             and checked.get("input_binding", {}).get("build_manifest_sha256") == build_sha
             and bool(train) and set(checked.get("selected_ids", [])) == set(train)
             and checked.get("structures") == {pid: row[1].structure_hash for pid, row in train.items()})
    if ready:
        # Quarantined IDs and blocked sources are never eligible; reject before writing any output.
        check(rule, public, local_prompts(protocol), build_sha, eligible=eligible)
        if proof:
            _rederive_proof(rule, root, local_prompts(protocol), eligible, pool_sizes(protocol))
    if frozen_approval is None:
        validated = _write_json(out / "validated_data.json",
                                checked | {"actors": [json.loads(to_json(a)) for a in checked["actors"]]})
    else:
        # The exact pinned bytes, so later frozen reads pin this preparation's validated_data.
        raw = _read(Path(frozen_approval["path"]))
        if evidence._sha(raw) != frozen_approval["file_sha256"]:
            raise ValueError("local pool: the frozen approval changed during preparation")
        _write(out / "validated_data.json", raw)
        validated = evidence._descriptor(out / "validated_data.json")
    request_copy = _write_json(out / "data_request.json", request)
    if (evidence._descriptor(request_descriptor["path"]) != request_descriptor
            or evidence._descriptor(rule_descriptor["path"]) != rule_descriptor
            or evidence._sha(_read(root / "manifest.yaml")) != build_sha):
        raise ValueError("local pool: request, rule or build changed during validation")
    result = {"format": POOL_FORMAT_PROOF if proof else POOL_FORMAT_V2 if v2 else POOL_FORMAT,
        "status": POOL_STATUS if ready else "BLOCKED", "profile": PROFILE,
        "local_protocol_sha256": protocol_sha, "approval_stage": approval.AI_LOCAL_STAGE, "human_verified": False,
        "training_allowed": False, "gpu_verified": False, "cloud_allowed": False,
        "new_exclusion_scan_performed": False, "new_human_review_performed": False,
        "build_root": str(root), "image_root": str(root / "images"), "data_request": request_copy,
        "validated_data": validated, "rule": rule_descriptor,
        **({"data_verification": verification} if verification is not None else {}),
        "limitations": PROOF_LIMITATIONS if proof else V2_LIMITATIONS if v2 else [
            "AI-reviewed local pool (ai_reviewed_local); human_verified=false.",
            "Evaluation items are held-out records from sources and structure families outside the fixed ten.",
            "Quarantined families and AI-review-blocked sources stay in the build but are never selected.",
            "Measurement eligibility only; no training campaign, cloud or GPU authority is granted."]}
    if v2:
        result["data_policy"] = POOL_DATA_POLICY
        result["probe_coverage"] = summary["probe_coverage"]
    if ready:
        result["eligibility"] = {key: eligibility[key]
                                 for key in ("allowlisted_by_source", "quarantined_count", "blocked_sources")}
        approved = {a.problem_id: a for a in checked["actors"]}
        fixed = [approved.get(pid) for pid in rule["fixed"]]
        if fixed != [public[pid][0] for pid in rule["fixed"]]:
            raise ValueError("local pool: validated actors differ from the build")
        rows = [json.loads(to_json(a)) for a in fixed]
        result["actors"] = _write_json(out / "actors.json", rows)
        result["execution_actors"] = _write_json(out / "execution_actors.json", rows)
        result["evaluation_actors"] = _write_json(out / "evaluation_actors.json",
            [json.loads(to_json(public[pid][0])) for pid in summary["evaluation_ids"]])
        if v2:
            # Sync/reload probes (v2: minimal ∪ probe_only, probe-only ones never trained; proof: trained fixed actors).
            probes = [approved.get(pid) for pid in summary["probe_ids"]]
            if probes != [public[pid][0] for pid in summary["probe_ids"]]:
                raise ValueError("local pool: validated probe actors differ from the build")
            result["probe_actors"] = _write_json(out / "probe_actors.json", [json.loads(to_json(a)) for a in probes])
        result["actor_projection"] = {"policy": POOL_MODE_POLICY, "changed_ids": []}
        # A proof summary binds its subjects (frozen manifest); v1/v2 reconstruct them from hf_pool when derivable.
        subjects = summary.get("subjects") or pool_subjects(root, public)
        result["selection"] = _write_json(out / "selection.json", {
            "status": "LOCAL_POOL_SELECTION_NOT_TRAINING_APPROVAL", "version": rule["version"],
            "profile": PROFILE, "local_protocol_sha256": protocol_sha, "rule": rule,
            "rule_sha256": rule_descriptor["file_sha256"], **summary,
            "subjects": {pid: subjects[pid] for pid in summary["structures"]}})
    else:
        result["blockers"] = checked.get("blockers") or ["LOCAL_POOL_SCOPE_OR_SELECTION_MISMATCH"]
    _write_json(out / "preparation.json", result)
    return result


# ---------------------------------------------------------------- existing-datasets-only pool (v2, opt-in)

POOL_FORMAT_V2 = "notebook-rl-local-pool-data-v2"
POOL_FORMAT_PROOF = "notebook-rl-local-pool-data-proof-v1"
POOL_FORMATS = (POOL_FORMAT, POOL_FORMAT_V2, POOL_FORMAT_PROOF)
POOL_RULE_VERSION_V2 = "pool-local-selection-v2"
POOL_DATA_POLICY = "existing-datasets-only-v1"
# Program-generator items, RDKit-computed labels (ChEBI) and our own ECG renderings (PTB-XL).
DEFAULT_EXCLUDED_SOURCES = ("gen-*", "hf-chebi", "hf-ptbxl")
PROBE_ONLY_POLICY = ("probe_only IDs feed only the synchronization/reload forward probes; they are never trained, "
                     "sampled for rollouts or evaluated, so they may come from any approved source, excluded ones included.")
PROBE_COVERAGE_RULE = "probe-coverage-trained-image-counts-v1"
PROBE_COVERAGE_LIMITATION = (
    f"{PROBE_COVERAGE_RULE}: the synchronization/reload probes are one actor per image count of the rule's minimal, "
    "fixed and evaluation actors, plus the single-image probe floor (single-image only, or single- and two-image). "
    "A pool whose trained/evaluated actors are all single-image therefore has no two-image sync probe; multi-image "
    "capability of the base stays covered by the unchanged NF4 cache readiness probes (quantized_cache).")
_RULE_KEYS_V2 = _RULE_KEYS | {"probe_only", "excluded_sources", "policy", "probe_coverage"}
V2_LIMITATIONS = ["AI-reviewed local pool (ai_reviewed_local); human_verified=false.",
    "existing-datasets-only-v1: training and evaluation items come only from existing datasets; program-generator "
    "items and the rule's excluded sources are never trained or evaluated.",
    PROBE_ONLY_POLICY, PROBE_COVERAGE_LIMITATION,
    "Evaluation items are held-out records from sources and structure families outside the fixed training items.",
    "Quarantined families and AI-review-blocked sources stay in the build but are never selected.",
    "Measurement eligibility only; no training campaign, cloud or GPU authority is granted."]


def excluded_source(source: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(source, pattern) for pattern in patterns)


def probe_coverage(image_counts: Iterable[int]) -> dict:
    """The probe-coverage-trained-image-counts-v1 record for the image counts of a rule's trained/evaluated actors."""
    counts = set(image_counts)
    if not counts <= {1, 2}:
        raise ValueError("local pool: probe coverage supports one- and two-image actors only")
    return {"rule": PROBE_COVERAGE_RULE, "image_counts": sorted(counts | {1})}


def pool_sizes(protocol: dict) -> tuple[int, int]:
    """(N training items, K per update) of the bound scale group, else the historical (10, 2)."""
    from .scale import options

    opts = options(protocol)
    return opts["train_problems"], opts["problems_per_update"]


def check_pool_rule_v2(rule: dict, public: dict, prompts: dict, build_sha256: str, *,
                       sizes: tuple[int, int] = (10, 2), eligible: Collection[str] | None = None) -> dict:
    """Validate a pre-fixed v2 rule; probe-only actors only complete the bound probe coverage
    (probe-coverage-trained-image-counts-v1: one probe per image count of the trained/evaluated actors)."""
    from .approval import label_class
    from .local_run import covering_actors

    count, per_update = sizes
    if type(rule) is not dict or set(rule) != _RULE_KEYS_V2 or rule["version"] != POOL_RULE_VERSION_V2:
        raise ValueError("local pool: exact pool-local-selection-v2 rule required")
    lists = [rule[key] for key in ("minimal", "fixed", "evaluation", "probe_only", "excluded_sources")]
    every = [pid for key in ("fixed", "evaluation", "probe_only") if type(rule[key]) is list for pid in rule[key]]
    if (rule["build_manifest_sha256"] != build_sha256 or type(rule["seed"]) is not int
            or rule["policy"] != POOL_DATA_POLICY
            or any(type(ids) is not list or any(type(v) is not str or not v for v in ids) or len(set(ids)) != len(ids)
                   for ids in lists)
            or not set(DEFAULT_EXCLUDED_SOURCES) <= set(rule["excluded_sources"])
            or type(count) is not int or type(per_update) is not int or not 1 <= per_update <= count
            or count % per_update or len(rule["fixed"]) != count or len(rule["minimal"]) != 2
            or not set(rule["minimal"]) <= set(rule["fixed"]) or not rule["evaluation"]
            or len(set(every)) != len(every) or not set(every) <= public.keys()):
        raise ValueError("local pool: v2 rule build, policy, ID lists or sizes are invalid")
    if eligible is not None and not set(every) <= set(eligible):
        raise ValueError("local pool: every rule ID must be exclusion-allowlisted and outside blocked sources")
    actor = {pid: public[pid][0] for pid in every}
    lineage = {pid: public[pid][1] for pid in every}
    if any(excluded_source(actor[pid].source, rule["excluded_sources"]) or label_class(lineage[pid]) == "GENERATOR"
           for pid in rule["fixed"] + rule["evaluation"]):
        raise ValueError("local pool: training and evaluation items must come from existing, non-excluded datasets")
    if (any(lineage[pid].split != "train" for pid in rule["fixed"] + rule["probe_only"])
            or any(lineage[pid].split not in HELD_OUT_SPLITS for pid in rule["evaluation"])):
        raise ValueError("local pool: fixed/probe-only IDs must be train records and evaluation IDs held-out records")
    if len({lineage[pid].structure_hash for pid in rule["fixed"]}) != count:
        raise ValueError("local pool: distinct fixed structures required")
    # minimal ⊆ fixed, so the fixed and evaluation actors carry every trained/evaluated image count.
    coverage = probe_coverage(len(actor[pid].image_paths) for pid in rule["fixed"] + rule["evaluation"])
    if rule["probe_coverage"] != coverage:
        raise ValueError("local pool: the rule's probe coverage differs from its trained/evaluated image counts")
    try:
        probes = covering_actors([actor[pid] for pid in rule["minimal"] + rule["probe_only"]], coverage["image_counts"])
    except ValueError:
        raise ValueError("local pool: minimal and probe-only IDs must hold one actor per covered image count") from None
    if not set(rule["probe_only"]) <= {a.problem_id for a in probes}:
        raise ValueError("local pool: every probe-only ID must complete the covered probe image counts")
    train_keys = {(key, value) for pid in rule["fixed"] for key, value in (
        ("source", actor[pid].source), ("structure", lineage[pid].structure_hash),
        *(("image", sha) for sha in actor[pid].image_sha256))}
    if any((key, value) in train_keys for pid in rule["evaluation"] for key, value in (
            ("source", actor[pid].source), ("structure", lineage[pid].structure_hash),
            *(("image", sha) for sha in actor[pid].image_sha256))):
        raise ValueError("local pool: evaluation sources, structure families and images must be disjoint from training")
    if not all(_renders(a, prompts) for a in actor.values()):
        raise ValueError("local pool: every selected actor must render under the bound protocol prompts")
    return {"minimal_ids": rule["minimal"], "problem_ids": rule["fixed"],
            "batch_ids": [rule["fixed"][i:i + per_update] for i in range(0, count, per_update)],
            "evaluation_ids": rule["evaluation"], "probe_only_ids": rule["probe_only"],
            "probe_ids": [a.problem_id for a in probes], "probe_coverage": coverage, "policy": POOL_DATA_POLICY,
            "excluded_sources": rule["excluded_sources"], "probe_only_policy": PROBE_ONLY_POLICY,
            "structures": {pid: lineage[pid].structure_hash for pid in actor},
            "image_counts": {pid: len(actor[pid].image_paths) for pid in actor},
            "sources": {pid: actor[pid].source for pid in actor}, "modes": {pid: actor[pid].mode for pid in actor},
            "splits": {pid: lineage[pid].split for pid in actor},
            "label_classes": {pid: label_class(lineage[pid]) for pid in actor}}


def _propose_v2(root: Path, prompts: dict, *, seed: int, evaluation_count: int, eligible: Collection[str] | None,
                excluded_sources: Collection[str], sizes: tuple[int, int]) -> dict:
    from .approval import label_class

    public = public_pool(root)
    allowed = pool_allowlist(root, public)
    if eligible is not None:
        allowed &= set(eligible)
    usable = [pid for pid in _order(allowed, seed) if _renders(public[pid][0], prompts)]
    patterns = sorted(set(DEFAULT_EXCLUDED_SOURCES) | set(excluded_sources))
    count, _ = sizes

    def existing(pid: str) -> bool:
        return (not excluded_source(public[pid][0].source, patterns)
                and label_class(public[pid][1]) != "GENERATOR")

    train = [pid for pid in usable if public[pid][1].split == "train"]
    candidates = [pid for pid in train if existing(pid)]
    # Label-blind source split: the first half of the sources in seed-keyed hash order feed the fixed set, so
    # the rest stay available as source-disjoint evaluation sources.
    pool_sources = sorted({public[pid][0].source for pid in candidates},
                          key=lambda s: hashlib.sha256(f"{seed}:source:{s}".encode()).hexdigest())
    fixed_sources = set(pool_sources[:max(4, (len(pool_sources) + 1) // 2)])
    candidates = [pid for pid in candidates if public[pid][0].source in fixed_sources]
    fixed = _round_robin(public, candidates, count, set())
    sources = {public[pid][0].source for pid in fixed}
    if len(fixed) != count or len(sources) < min(4, len({public[pid][0].source for pid in candidates})):
        raise ValueError(f"local pool: need {count} existing-dataset training items with distinct structures")
    held = [pid for pid in usable if public[pid][1].split in HELD_OUT_SPLITS and existing(pid)
            and public[pid][0].source not in sources]
    evaluation = _round_robin(public, held, evaluation_count, {public[pid][1].structure_hash for pid in fixed})
    if evaluation_count < 1 or len(evaluation) != evaluation_count:
        raise ValueError("local pool: not enough held-out existing-dataset items from sources outside the fixed set")
    minimal = fixed[:2]
    coverage = probe_coverage(len(public[pid][0].image_paths) for pid in fixed + evaluation)
    # The first approved train actor in hash order (any source; never trained) completes a covered count minimal lacks.
    have = {len(public[pid][0].image_paths) for pid in minimal}
    probe_only = [next((pid for pid in train if pid not in fixed and len(public[pid][0].image_paths) == n), "")
                  for n in coverage["image_counts"] if n not in have]
    if "" in probe_only:
        raise ValueError("local pool: no approved actor completes the covered probe image counts")
    rule = {"version": POOL_RULE_VERSION_V2, "build_manifest_sha256": evidence._sha(_read(root / "manifest.yaml")),
            "seed": seed, "policy": POOL_DATA_POLICY, "excluded_sources": patterns, "minimal": minimal,
            "fixed": fixed, "evaluation": evaluation, "probe_only": probe_only, "probe_coverage": coverage}
    check_pool_rule_v2(rule, public, prompts, rule["build_manifest_sha256"], sizes=sizes, eligible=allowed)
    return rule


def _ai_local_request(root: Path, approval_request: Path) -> dict:
    request = evidence._bound_json(evidence._descriptor(Path(approval_request).absolute()))
    if (set(request) != {"build_root", "index", "source_registry", "evidence", "data_report",
                         "expected_measurement_scope"} or request["build_root"] != str(root)
            or request["expected_measurement_scope"].get("purpose") != approval.AI_LOCAL_PURPOSE):
        raise ValueError("local pool: exact ai_reviewed_local data request for this build required")
    return request


def approved_eligible_ids(build_root: Path, approval_request: Path) -> list[str]:
    """`eligibility.eligible_ids` of a current ai_reviewed_local validation of this build."""
    root = Path(build_root).absolute()
    request = _ai_local_request(root, approval_request)
    checked = approval.validate_for_ai_reviewed_local(root, timeout=LOCAL_PREFLIGHT_SECONDS,
        **{k: v for k, v in request.items() if k != "build_root"})
    eligible = (checked.get("eligibility") or {}).get("eligible_ids")
    if (checked.get("status") != "VALIDATED_FOR_MEASUREMENT" or checked.get("blockers") != []
            or checked.get("stage") != approval.AI_LOCAL_STAGE or checked.get("human_verified") is not False
            or checked.get("input_binding", {}).get("build_manifest_sha256") != evidence._sha(_read(root / "manifest.yaml"))
            or type(eligible) is not list):
        raise ValueError("local pool: a current validated ai_reviewed_local approval is required for eligibility")
    return eligible


# ---------------------------------------------------------------- proof selection (pool-proof-selection-v1, opt-in)

POOL_RULE_VERSION_PROOF = "pool-proof-selection-v1"
SENTINEL_SPLITS = ("dev", "sealed")
_RULE_KEYS_PROOF = {"version", "build_manifest_sha256", "seed", "policy", "excluded_sources", "minimal", "fixed",
                    "sentinels", "sentinels_per_subject", "probe_coverage", "selection_source"}
_SOURCE_KEYS = {"all-eligible": {"kind", "dropped_ids"},
                "passrate-selection": {"kind", "path", "file_sha256", "report", "report_sha256", "excluded_ids",
                                       "dropped_ids"}}
_HEX = frozenset("0123456789abcdef")
PROOF_LIMITATIONS = ["AI-reviewed local pool (ai_reviewed_local); human_verified=false.",
    "existing-datasets-only-v1: training and sentinel items come only from existing datasets; program-generator "
    "items and the rule's excluded sources are never trained or evaluated.",
    f"{POOL_RULE_VERSION_PROOF}: the training items are the pass-rate-selected (or all) approval-eligible train items "
    "in seed-keyed hash order, one per structure family, capped at the bound train_problems; the remainder is "
    "recorded as dropped_ids, selected items that are not eligible training candidates as excluded_ids.",
    "Sentinels are held-out dev/sealed items per subject (frozen-manifest subjects) from the same sources as "
    "training, structure- and image-disjoint from it: in-distribution held-out checks, not source transfer. "
    "MMMU is evaluated separately.",
    PROBE_COVERAGE_LIMITATION,
    "Quarantined families and AI-review-blocked sources stay in the build but are never selected.",
    "Measurement eligibility only; no training campaign, cloud or GPU authority is granted."]


def proof_subjects(root: Path, public: dict) -> dict[str, str]:
    """Each vault item's subject from the union vault's bound frozen dataset manifest (pool_freeze carries one)."""
    root = Path(root).absolute()
    if not (root / "contamination" / pool_freeze.UNION_FILE).exists():
        raise ValueError(f"local pool: {POOL_RULE_VERSION_PROOF} requires a union vault (its frozen manifest names subjects)")
    union, _ = pool_freeze.read_union(root)
    manifest, sha = pool_freeze._manifest(Path(union["dataset_manifest"]["path"]))
    subjects = {row["problem_id"]: row.get("subject") for row in manifest["items"] if row["problem_id"] in public}
    if (sha != union["dataset_manifest"]["file_sha256"] or subjects.keys() != public.keys()
            or any(type(subject) is not str or not subject for subject in subjects.values())):
        raise ValueError("local pool: every vault item needs a subject in the union vault's bound frozen manifest")
    return subjects


def check_proof_rule(rule: dict, public: dict, prompts: dict, build_sha256: str, *, sizes: tuple[int, int],
                     subjects: dict, eligible: Collection[str] | None = None) -> dict:
    """Validate a pool-proof-selection-v1 rule (N/K = `sizes` of the bound scale group); returns the selection summary.

    Sentinels may share sources with the training items, never a structure family or an image. The full label-blind
    re-derivation from the selection source is prepare-pool's (`_rederive_proof`); this re-check runs on every read.
    """
    from .approval import label_class
    from .local_run import covering_actors

    count, per_update = sizes
    if type(rule) is not dict or set(rule) != _RULE_KEYS_PROOF or rule["version"] != POOL_RULE_VERSION_PROOF:
        raise ValueError(f"local pool: exact {POOL_RULE_VERSION_PROOF} rule required")
    source, sentinels, per_subject = rule["selection_source"], rule["sentinels"], rule["sentinels_per_subject"]
    if (type(source) is not dict or source.get("kind") not in _SOURCE_KEYS or set(source) != _SOURCE_KEYS[source["kind"]]
            or source["kind"] == "passrate-selection" and (
                any(type(source[key]) is not str or not Path(source[key]).is_absolute() for key in ("path", "report"))
                or any(type(source[key]) is not str or len(source[key]) != 64 or not set(source[key]) <= _HEX
                       for key in ("file_sha256", "report_sha256")))
            or type(sentinels) is not dict or not sentinels or type(per_subject) is not int or per_subject < 1
            or any(type(subject) is not str or not subject or type(ids) is not list or not 1 <= len(ids) <= per_subject
                   for subject, ids in sentinels.items())):
        raise ValueError("local pool: proof selection source or per-subject sentinels are malformed")
    held = [pid for subject in sorted(sentinels) for pid in sentinels[subject]]
    dropped, excluded = source["dropped_ids"], source.get("excluded_ids", [])
    every = [pid for pid in (rule["fixed"] if type(rule["fixed"]) is list else ()) if type(pid) is str] + held
    lists = [rule["minimal"], rule["fixed"], rule["excluded_sources"], held, dropped, excluded]
    if (rule["build_manifest_sha256"] != build_sha256 or type(rule["seed"]) is not int
            or rule["policy"] != POOL_DATA_POLICY
            or any(type(ids) is not list or any(type(v) is not str or not v for v in ids) or len(set(ids)) != len(ids)
                   for ids in lists)
            or not set(DEFAULT_EXCLUDED_SOURCES) <= set(rule["excluded_sources"])
            or type(count) is not int or type(per_update) is not int or not 1 <= per_update <= count
            or count % per_update or count < 2 or excluded != sorted(excluded)
            or len(set(every)) != len(every) or set(dropped) & set(every) or set(excluded) & set(every + dropped)
            or not set(every + dropped) <= public.keys()):
        raise ValueError("local pool: proof rule build, policy, ID lists or sizes are invalid")
    if len(rule["fixed"]) != count:
        raise ValueError(f"local pool: the proof rule fixes {len(rule['fixed'])} training items, "
                         f"the bound scale train_problems is {count}")
    if rule["minimal"] != rule["fixed"][:2]:
        raise ValueError("local pool: proof minimal IDs must be the first two fixed IDs")
    if eligible is not None and not set(every + dropped) <= set(eligible):
        raise ValueError("local pool: every rule ID must be exclusion-allowlisted and outside blocked sources")
    actor = {pid: public[pid][0] for pid in every + dropped}
    lineage = {pid: public[pid][1] for pid in actor}
    if any(excluded_source(actor[pid].source, rule["excluded_sources"]) or label_class(lineage[pid]) == "GENERATOR"
           for pid in actor):
        raise ValueError("local pool: training and evaluation items must come from existing, non-excluded datasets")
    if (any(lineage[pid].split != "train" for pid in rule["fixed"] + dropped)
            or any(lineage[pid].split not in SENTINEL_SPLITS for pid in held)):
        raise ValueError("local pool: fixed/dropped IDs must be train records and sentinels dev/sealed records")
    if any(subjects.get(pid) != subject for subject, ids in sentinels.items() for pid in ids):
        raise ValueError("local pool: every sentinel must be listed under its own subject")
    if (rule["fixed"] + dropped != _order(rule["fixed"] + dropped, rule["seed"])
            or any(ids != _order(ids, rule["seed"]) for ids in sentinels.values())):
        raise ValueError("local pool: fixed then dropped IDs, and each subject's sentinels, must be in seed-keyed hash order")
    trained = [lineage[pid].structure_hash for pid in rule["fixed"] + dropped]
    checks = [lineage[pid].structure_hash for pid in held]
    if len(set(trained)) != len(trained) or len(set(checks)) != len(checks):
        raise ValueError("local pool: distinct fixed and sentinel structures required")
    images = {sha for pid in rule["fixed"] for sha in actor[pid].image_sha256}
    if set(checks) & set(trained[:count]) or any(images & set(actor[pid].image_sha256) for pid in held):
        raise ValueError("local pool: sentinel structure families and images must be disjoint from the training items")
    coverage = probe_coverage(len(actor[pid].image_paths) for pid in every)
    if rule["probe_coverage"] != coverage:
        raise ValueError("local pool: the rule's probe coverage differs from its trained/evaluated image counts")
    try:
        probes = covering_actors([actor[pid] for pid in rule["fixed"]], coverage["image_counts"])
    except ValueError:
        raise ValueError("local pool: the training items must hold one probe actor per covered image count") from None
    if not all(_renders(actor[pid], prompts) for pid in every):
        raise ValueError("local pool: every selected actor must render under the bound protocol prompts")
    return {"minimal_ids": rule["minimal"], "problem_ids": rule["fixed"],
            "batch_ids": [rule["fixed"][i:i + per_update] for i in range(0, count, per_update)],
            "evaluation_ids": held, "sentinel_ids": {subject: sentinels[subject] for subject in sorted(sentinels)},
            "probe_ids": [a.problem_id for a in probes], "probe_coverage": coverage, "policy": POOL_DATA_POLICY,
            "excluded_sources": rule["excluded_sources"],
            "structures": {pid: lineage[pid].structure_hash for pid in every},
            "image_counts": {pid: len(actor[pid].image_paths) for pid in every},
            "sources": {pid: actor[pid].source for pid in every}, "modes": {pid: actor[pid].mode for pid in every},
            "splits": {pid: lineage[pid].split for pid in every},
            "label_classes": {pid: label_class(lineage[pid]) for pid in every},
            "subjects": {pid: subjects[pid] for pid in every}}


def propose_proof_rule(build_root: Path, prompts: dict, *, eligible: Collection[str], sizes: tuple[int, int],
                       sentinels_per_subject: int, seed: int = 3407, excluded_sources: Collection[str] = (),
                       passrate: tuple[Path, Path] | None = None) -> dict:
    """Label-blind pool-proof-selection-v1 from public metadata, the approval's eligible IDs and (optionally) a
    pass-rate selection re-derived from its measurement report (passrate.verify_selection).

    fixed: the selected (or all) eligible existing-dataset train items in seed-keyed hash order, the first per
    structure family, capped at N = sizes[0]; the rest are dropped_ids (fewer than N is refused). sentinels: per
    frozen-manifest subject, the first `sentinels_per_subject` eligible dev/sealed items in hash order whose structure
    family and images are disjoint from fixed (same sources allowed).
    """
    from . import passrate as passrate_module
    from .approval import label_class

    root = Path(build_root).absolute()
    public = public_pool(root)
    subjects = proof_subjects(root, public)
    build_sha = evidence._sha(_read(root / "manifest.yaml"))
    patterns = sorted(set(DEFAULT_EXCLUDED_SOURCES) | set(excluded_sources))
    allowed = pool_allowlist(root, public) & set(eligible)
    usable = [pid for pid in _order(allowed, seed) if not excluded_source(public[pid][0].source, patterns)
              and label_class(public[pid][1]) != "GENERATOR" and _renders(public[pid][0], prompts)]
    train = [pid for pid in usable if public[pid][1].split == "train"]
    source: dict = {"kind": "all-eligible"}
    if passrate is not None:
        selection_path, report = (Path(path).absolute() for path in passrate)
        selection, measured, selection_sha = passrate_module.verify_selection(selection_path, report)
        if (selection["kind"] != "PASSRATE_SELECTION" or selection["split"] != "train"
                or selection["policy_version"] != 0 or measured.get("build_manifest_sha256") != build_sha):
            raise ValueError("local pool: a policy-version-0 PASSRATE_SELECTION of this vault's train split required")
        if measured["weights"]["checkpoint"] is not None:
            raise ValueError("local pool: a training selection must be measured at F0; a checkpoint-measured report "
                             "can only feed hard_negatives")
        chosen = set(selection["selected_ids"])
        train = [pid for pid in train if pid in chosen]
        source = {"kind": "passrate-selection", "path": str(selection_path), "file_sha256": selection_sha,
                  "report": str(report), "report_sha256": selection["report_sha256"]}
    distinct: list[str] = []
    families: set[str] = set()
    for pid in train:
        if public[pid][1].structure_hash not in families:
            families.add(public[pid][1].structure_hash)
            distinct.append(pid)
    count = sizes[0]
    if len(distinct) < count:
        raise ValueError(f"local pool: {len(distinct)} eligible distinct-structure training items, "
                         f"fewer than train_problems {count}")
    fixed = distinct[:count]
    source["dropped_ids"] = distinct[count:]
    if passrate is not None:
        source["excluded_ids"] = sorted(chosen - set(distinct))
    taken = {public[pid][1].structure_hash for pid in fixed}
    images = {sha for pid in fixed for sha in public[pid][0].image_sha256}
    sentinels: dict[str, list[str]] = {}
    for pid in usable:
        actor, lineage = public[pid]
        if (lineage.split in SENTINEL_SPLITS and lineage.structure_hash not in taken
                and not images & set(actor.image_sha256) and len(sentinels.get(subjects[pid], ())) < sentinels_per_subject):
            sentinels.setdefault(subjects[pid], []).append(pid)
            taken.add(lineage.structure_hash)
    rule = {"version": POOL_RULE_VERSION_PROOF, "build_manifest_sha256": build_sha, "seed": seed,
            "policy": POOL_DATA_POLICY, "excluded_sources": patterns, "minimal": fixed[:2], "fixed": fixed,
            "sentinels": dict(sorted(sentinels.items())), "sentinels_per_subject": sentinels_per_subject,
            "probe_coverage": probe_coverage(len(public[pid][0].image_paths)
                                             for pid in fixed + [p for ids in sentinels.values() for p in ids]),
            "selection_source": source}
    check_proof_rule(rule, public, prompts, build_sha, sizes=sizes, subjects=subjects, eligible=allowed)
    return rule


def _rederive_proof(rule: dict, root: Path, prompts: dict, eligible: Collection[str], sizes: tuple[int, int]) -> None:
    """prepare-pool: the rule must be exactly the label-blind derivation of its (unchanged) selection source."""
    source = rule["selection_source"]
    passrate = None
    if source["kind"] == "passrate-selection":
        if evidence._sha(_read(Path(source["path"]))) != source["file_sha256"]:
            raise ValueError("local pool: the pass-rate selection changed since the rule was proposed")
        passrate = (Path(source["path"]), Path(source["report"]))
    derived = propose_proof_rule(root, prompts, eligible=eligible, sizes=sizes,
                                 sentinels_per_subject=rule["sentinels_per_subject"], seed=rule["seed"],
                                 excluded_sources=rule["excluded_sources"], passrate=passrate)
    if derived != rule:
        raise ValueError("local pool: the proof rule is not the label-blind derivation of its selection source")


def export_approval(build_root: Path, approval_request: Path, output: Path) -> dict:
    """Save a fresh ai_reviewed_local approval output of a union vault: the file frozen-approval-verify-v1 pins.

    The bytes are what prepare-pool writes as validated_data.json; the file is kept only when the frozen round trip
    a pod runs (pinned bytes, approval checks, union vault re-hash) passes on it. Returns its {path, file_sha256}."""
    from .local_run import frozen_approval_output

    root = Path(build_root).absolute()
    request = _ai_local_request(root, approval_request)
    output = artifact_path(Path(output).absolute())
    if output.exists():
        raise FileExistsError("local pool: preserve an existing approval export")
    checked = approval.validate_for_ai_reviewed_local(root, timeout=LOCAL_PREFLIGHT_SECONDS,
        **{k: v for k, v in request.items() if k != "build_root"})
    descriptor = _write_json(output, checked | {"actors": [json.loads(to_json(a)) for a in checked.get("actors", [])]})
    try:
        frozen_approval_output(root, descriptor)
    except ValueError:
        output.unlink()
        raise
    return descriptor
