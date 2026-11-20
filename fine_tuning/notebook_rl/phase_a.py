"""Fixed-F0 train measurements, never Phase-A or training approval.

The caller owns model identity/authorization and an outer watchdog for in-flight
GPU calls. This core binds actual requests/results and uses only the private
verifier worker for scoring. No oracle, held-out reference or update runs here.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .approval import code_hashes
from .dataset import load_images
from .engines import Completion, RolloutRequest, RolloutResult, SamplingSettings, _completion
from .monitor import _Incomplete, _append, _frozen, _hash, _image_binding, _require, _sha, _write
from .prompts import retry_text, turn1_text, user_message
from .rollout import RETRYABLE, continuation_ids, seeds, select_failures
from .schema import Sample, to_json
from .vault import _read, _safe_root, _unique, read_actors
from .verifier_service import VerifierService

VERSION = "notebook-rl-phase-a-measurements-v1"
_SCORED = {"CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"}
_CAP = 8192


def _snapshot(value):
    """Keep malformed returned payloads durable, including nonfinite numbers."""
    if is_dataclass(value) and not isinstance(value, type):
        return _snapshot(asdict(value))
    if type(value) is dict:
        return {str(k): _snapshot(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [_snapshot(v) for v in value]
    if type(value) is float and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    return {"unsupported_type": type(value).__module__ + "." + type(value).__qualname__}


def _cap(lengths: list[int]) -> tuple[int | None, int | None]:
    if not lengths:
        return None, None
    p95 = sorted(lengths)[math.ceil(.95 * len(lengths)) - 1]
    return min(4096, max(2048, math.ceil(p95 / 256) * 256)), p95


def _turn_cap(observations: list[dict], actors, kind: str, blocked: list[dict], upstream=()) -> dict:
    rows = [row["sample"] for row in observations if row["kind"] == kind]
    complete = [row for row in rows if row["result"]["status"] in ("CORRECT", "INCORRECT")
                and row["finish_reason"] == "stop"]
    cap, p95 = _cap([len(row["completion_token_ids"]) for row in complete])

    def status(n, finals, blocks, unknown):
        return ("CONTEXT_BLOCKED" if blocks else "UPSTREAM_CONTEXT_BLOCKED" if unknown
                else "NOT_APPLICABLE_NO_ELIGIBLE_PARENT" if not n
                else "NO_COMPLETED_FINAL_ANSWERS" if not finals else "COMPLETED")

    families = {}
    for family in sorted({actor.family for actor in actors}):
        pids = {actor.problem_id for actor in actors if actor.family == family}
        group = [row for row in rows if row["problem_id"] in pids]
        final = [row for row in complete if row["problem_id"] in pids]
        blocks = [row for row in blocked if row["problem_id"] in pids]
        unknown = [row for row in upstream if row["problem_id"] in pids]
        family_cap, family_p95 = _cap([len(row["completion_token_ids"]) for row in final])
        predicted = sum(row["finish_reason"] == "length" or len(row["completion_token_ids"]) > cap
                        for row in group) if cap is not None else None
        observed_rate = predicted / len(group) if predicted is not None and group else None
        families[family] = {"status": status(len(group), len(final), len(blocks), len(unknown)), "n": len(group),
            "completed_final_n": len(final), "complete_final_p95": family_p95,
            "family_cap_recommendation": family_cap, "context_blocked_samples": 4 * len(blocks),
            "upstream_unmeasured_parent_slots": len(unknown),
            "truncated_at_8192": sum(row["finish_reason"] == "length" for row in group),
            "truncated_at_recommended_cap": predicted,
            "truncation_rate_at_recommended_cap": observed_rate if not blocks and not unknown else None,
            "truncation_rate_conditional_on_executed": observed_rate,
            "family_action_required": bool(blocks) or bool(unknown) or family_cap is None or
                (observed_rate is not None and observed_rate > .1)
                if group or blocks or unknown else None}
    return {"status": status(len(rows), len(complete), len(blocked), len(upstream)), "n": len(rows),
        "completed_final_n": len(complete), "cap": cap, "complete_final_p95": p95,
        "method": "nearest-rank-p95-ceil256-clamp2048-4096", "families": families,
        "measurement_cap": _CAP, "cap_role": kind + "_MEASUREMENT_ONLY_NOT_TRAINING_CAP",
        "context_blocked_samples": 4 * len(blocked), "upstream_unmeasured_parent_slots": len(upstream)}


def _attribution(observations: list[dict], actors, kind: str) -> dict:
    eligible = {actor.problem_id for actor in actors if actor.mode == "obs"}
    rows = [row for row in observations if row["kind"] == kind and row["sample"]["problem_id"] in eligible]
    failed = [row for row in rows if row["sample"]["result"]["status"] != "CORRECT"]
    known = [row for row in failed if row["cause"] not in (None, "UNKNOWN")]
    slots = [row["sample"]["result"]["slots"] for row in rows if row["sample"]["result"]["slots"] is not None]
    required = sum(slot["required"] for slot in slots)
    parsed = sum(slot["required_correct"] + len(slot["wrong_required"]) for slot in slots)
    observed_ids = {row["sample"]["problem_id"] for row in rows}
    known_ids = {row["sample"]["problem_id"] for row in known}
    visual_ids = {row["sample"]["problem_id"] for row in known if row["cause"] in ("VISUAL_MISREAD", "VISUAL_RELATION")}
    return {"n_obs_candidate_items": len(eligible), "n_obs_observed_items": len(observed_ids),
        "n_obs_observations": len(rows), "failed_observations": len(failed),
        "cause_counts": {cause: sum(row["cause"] == cause for row in failed) for cause in
            ("VISUAL_MISREAD", "VISUAL_RELATION", "NON_VISUAL", "FINALIZATION", "UNKNOWN")},
        "unattributed_failures": sum(row["cause"] is None for row in failed),
        "attributable_failed_observations": len(known),
        "attributable_failure_rate": len(known) / len(failed) if failed else None,
        "attributable_observed_items": len(known_ids),
        "attributable_item_rate_conditional_on_observed": len(known_ids) / len(observed_ids) if observed_ids else None,
        "visually_attributable_items": len(visual_ids),
        "vision_evidence_item_rate_conditional_on_observed": len(visual_ids) / len(observed_ids) if observed_ids else None,
        "required_slots": required, "parsed_required_slots": parsed,
        "required_slot_parse_rate": parsed / required if required else None,
        "complete_required_slot_observations": sum(slot["required"] > 0 and
            slot["required_correct"] + len(slot["wrong_required"]) == slot["required"] for slot in slots),
        "module_causality_certified": False}


def measure_candidates(build_root: Path, output: Path, *, engine: Any, protocol: dict,
                       input_binding: dict, deadline_timestamp: float, seed: int = 3407,
                       forbidden_roots=(), clock: Callable[[], float] = time.time) -> dict:
    """Measure all 576 train candidates and a preselected 128-item retry subset.

    The engine must be a caller-verified fresh F0 (policy_version=0), expose
    prepare/tokenizer/sample_with_settings and explicitly support temperature_1.
    Four SHA bindings identify caller provenance; separate hashes bind this run's
    actual build, actor inputs, images and producer sources. CPU evidence never
    certifies GPU execution. Outputs are write-once; errors return INCOMPLETE.
    """
    return _measure_candidates(build_root, output, engine=engine, protocol=protocol,
        input_binding=input_binding, deadline_timestamp=deadline_timestamp, seed=seed,
        forbidden_roots=forbidden_roots, clock=clock, expected_candidates=576, retry_subset_size=128)


def _measure_candidates(build_root: Path, output: Path, *, engine: Any, protocol: dict,
                        input_binding: dict, deadline_timestamp: float, seed: int = 3407,
                        forbidden_roots=(), clock: Callable[[], float] = time.time,
                        expected_candidates: int, retry_subset_size: int) -> dict:
    """Bounded authored-fixture harness; production entry always supplies 576/128."""
    import json

    prompts = _frozen(protocol, input_binding, seed)
    if (type(deadline_timestamp) not in (int, float) or not math.isfinite(deadline_timestamp)
            or any(type(v) is not int or v < 1 for v in (expected_candidates, retry_subset_size))
            or retry_subset_size > expected_candidates):
        raise ValueError("phase_a: finite deadline and explicit candidate/retry counts required")
    if (type(getattr(engine, "policy_version", None)) is not int or engine.policy_version != 0
            or "temperature_1" not in getattr(engine, "supported_sampling", ())
            or not callable(getattr(engine, "sample_with_settings", None))
            or getattr(engine, "_phase_a_measurement_started", False)):
        raise ValueError("phase_a: unused fixed-F0 explicit T1 sampler required")
    build_root = _safe_root(Path(build_root))
    split_root, image_root = build_root / "train", _safe_root(build_root / "images")
    output = Path(output)
    protected = Path(__file__).resolve().parents[2]
    protected_roots = [build_root, protected]
    if getattr(engine, "model_dir", None) is not None:
        protected_roots.append(Path(engine.model_dir))
    if any(output.resolve().is_relative_to(root.resolve()) for root in protected_roots):
        raise ValueError("phase_a: output must be outside source/build/model roots")
    _safe_root(output.parent, create=True)
    actors = sorted(read_actors(split_root), key=lambda actor: actor.problem_id)
    if len(actors) != expected_candidates:
        raise ValueError("phase_a: full train candidate coverage required; no subsampling")
    build_raw = _read(build_root / "manifest.yaml")
    build = json.loads(build_raw, object_pairs_hook=_unique)
    actor_sha, vault_sha = (_sha(_read(split_root / name)) for name in ("actor_inputs.jsonl", "manifest.yaml"))
    if (build.get("format") != "notebook-rl-candidate-build-v1"
            or build.get("training_allowed") is not False
            or build.get("counts", {}).get("train") != expected_candidates
            or build.get("split_manifests", {}).get("train") != vault_sha
            or any(build.get("image_sha256", {}).get(path) != digest for actor in actors
                   for path, digest in zip(actor.image_paths, actor.image_sha256, strict=True))):
        raise ValueError("phase_a: actual candidate build/split/image binding mismatch")
    subset = sorted(actors, key=lambda actor: hashlib.sha256(
        f"phase-a-retry-v1|{seed}|{actor.problem_id}".encode()).digest())[:retry_subset_size]
    subset_ids = [actor.problem_id for actor in subset]
    source = code_hashes()
    configuration = {"format": VERSION, "build_id": _sha(build_raw), "split": "train",
        "actor_inputs_sha256": actor_sha, "vault_manifest_sha256": vault_sha,
        "candidate_ids_sha256": _hash([a.problem_id for a in actors]), "n_candidates": len(actors),
        "retry_subset_ids": subset_ids, "retry_subset_size": retry_subset_size,
        "retry_subset_sha256": _hash(subset_ids), "seed": seed, "policy_version": 0,
        "input_binding": dict(input_binding), "protocol_sha256": protocol["sha256"],
        "resolved_protocol_sha256": _hash(protocol), "prompts_sha256": prompts["sha256"],
        "producer_sources": source, "producer_sources_sha256": _hash(source),
        "sampling": asdict(SamplingSettings()), "measurement_cap": _CAP,
        "n": {"T1": 8, "blind": 4, "T2_per_parent": 4, "T3_per_parent": 4}, "failed_parents_per_item": 2,
        "T3_branch_scope": "one_per_measured_T2_group_measurement_only",
        "T3_failed_parents_per_T2_group": 1,
        "engine_class": type(engine).__module__ + "." + type(engine).__qualname__,
        "model_dir": str(getattr(engine, "model_dir", "CALLER_OWNED_F0")),
        "controlled_fixture_scope": (expected_candidates, retry_subset_size) != (576, 128),
        "training_allowed": False, "gpu_execution_certified": False}
    output.mkdir(mode=0o700, exist_ok=False)
    engine._phase_a_measurement_started = True
    _write(output / "manifest.json", configuration | {"deadline_timestamp": deadline_timestamp,
        "deadline_reserve_seconds": 120, "model_identity_verified_by_producer": False})
    summary: dict[str, Any] = configuration | {"status": "INCOMPLETE", "errors": [], "rows": [],
        "max_new_tokens": {}, "turn_cap_recommendations": {}, "t1_cap_recommendation": None,
        "t2_cap_recommendation": None, "t3_cap_recommendation": None, "retry": None, "third_retry": None,
        "attribution": {},
        "unmeasured": ["oracle_counterfactual", "held_out_baselines", "full_phase_a_assessment"],
        "sampling_cost_counts": {"attempted_calls": 0, "returned_calls": 0,
            "returned_completions": 0, "returned_tokens": 0, "verifier_calls": 0}}
    costs = summary["sampling_cost_counts"]
    observations: list[dict] = []
    items: dict[str, list[Sample]] = {}
    image_snapshots: dict[str, list[dict]] = {}
    retry_inapplicable: list[dict] = []
    retry_blocked: list[dict] = []
    third_parent_coverage: list[dict] = []
    interrupted: BaseException | None = None
    began = time.monotonic()

    def deadline():
        now = clock()
        _require(type(now) in (int, float) and math.isfinite(now), "CLOCK_INVALID")
        _require(now < deadline_timestamp - 120, "BUDGET_STOP")
        _require(type(engine.policy_version) is int and engine.policy_version == 0, "POLICY_VERSION_CHANGED")

    try:
        deadline()
        with VerifierService(split_root, forbidden_roots=list(forbidden_roots)) as verifier:
            _require(verifier.metadata["vault_manifest_sha256"] == vault_sha
                     and verifier.metadata["item_count"] == len(actors), "VERIFIER_INPUT_MISMATCH")
            _write(output / "verifier_manifest.json", verifier.metadata)
            with (output / "records.jsonl").open("xb") as records, (output / "sampler_calls.jsonl").open("xb") as raw:
                for stream in (records, raw):
                    os.fchmod(stream.fileno(), 0o600)

                def query(actor, images, kind, parent=None):
                    deadline()
                    turn, n = (parent.turn + 1, 4) if parent is not None else (1, 8 if kind == "T1" else 4)
                    branch = kind + (":" + parent.sample_id if parent is not None else "")
                    rid = f"phase-a:{actor.problem_id}:{branch}"
                    ids = continuation_ids(parent, retry_text(parent.result.status, parent.result.extracted,
                        actor.mode, actor.family, prompts), engine.tokenizer) if parent is not None else None
                    messages = None if ids is not None else [user_message(turn1_text(actor, prompts), len(images))]
                    request = RolloutRequest(rid, ids, messages, images, n, _CAP,
                        seeds(actor.problem_id, 0, turn, "phase-a:" + branch, n, master_seed=seed))
                    wire = {"request_id": rid, "prompt_token_ids": ids, "messages": messages,
                        "n": n, "max_new_tokens": _CAP, "seeds": request.seeds,
                        "images": _image_binding(actor, images), "kind": kind,
                        "parent_sample_id": parent.sample_id if parent is not None else None}
                    wire_sha = _hash(wire)
                    # Request records survive prepare errors as well as sampler errors.
                    _append(raw, {"event": "request", "request": wire, "sampling": configuration["sampling"]})
                    try:
                        if ids is not None:
                            _require(len(ids) + _CAP <= protocol["context_limit"], "CONTEXT_BLOCKED")
                        prepared = list(engine.prepare(request).token_ids)
                        _require(prepared and all(type(t) is int and t >= 0 for t in prepared), "PROMPT_IDS_INVALID")
                        _require(ids is None or prepared == ids, "LITERAL_HISTORY_CHANGED")
                        _require(_hash(wire) == wire_sha, "REQUEST_MUTATED")
                        _require(len(prepared) + _CAP <= protocol["context_limit"], "CONTEXT_BLOCKED")
                        deadline()
                        costs["attempted_calls"] += 1
                        started = time.monotonic()
                        results = engine.sample_with_settings([request], SamplingSettings())
                        elapsed = time.monotonic() - started
                        _append(raw, {"event": "return", "request_id": rid, "prepared_prompt_token_ids": prepared,
                            "results": _snapshot(results), "sample_seconds": elapsed})
                        costs["returned_calls"] += 1
                        _require(type(results) is list and all(type(r) is RolloutResult for r in results), "RESULT_TYPE")
                        costs["returned_completions"] += sum(len(r.completions) for r in results)
                        costs["returned_tokens"] += sum(len(c.token_ids) for r in results for c in r.completions)
                        deadline()
                        _require(_hash(wire) == wire_sha and _image_binding(actor, images) == wire["images"], "REQUEST_MUTATED")
                        _require(len(results) == 1 and results[0].request_id == rid
                                 and type(results[0].prompt_token_ids) is list
                                 and all(type(t) is int and t >= 0 for t in results[0].prompt_token_ids)
                                 and results[0].prompt_token_ids == prepared
                                 and len(results[0].completions) == n, "RESULT_IDENTITY_OR_COUNT")
                        samples = []
                        # Validate the entire group before issuing any scores.
                        for completion in results[0].completions:
                            _require(type(completion) is Completion, "COMPLETION_TYPE")
                            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
                            _require(all(type(p) in (int, float) for p in completion.logprobs)
                                and type(completion.text) is str and len(completion.token_ids) <= _CAP
                                and engine.tokenizer.decode(completion.token_ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False) == completion.text, "COMPLETION_INVALID")
                        for index, completion in enumerate(results[0].completions):
                            deadline()
                            costs["verifier_calls"] += 1
                            result = verifier.grade(actor.problem_id, completion.text, completion.finish_reason, actor.mode)
                            sample = Sample(rid + f":{index}", actor.problem_id, rid,
                                ("T3" if turn == 3 else "T2") if parent is not None else "T1", turn, wire["parent_sample_id"], prepared,
                                list(completion.token_ids), list(completion.logprobs), completion.finish_reason,
                                completion.text, request.seeds[index], 0, result)
                            row = {"kind": kind, "request_id": rid, "sample": json.loads(to_json(sample)),
                                "images": wire["images"], "cause": verifier.cause(actor.problem_id, result)}
                            _append(records, row)
                            observations.append(row)
                            _require(result.status in _SCORED, "VERIFIER_UNSCORABLE")
                            samples.append(sample)
                        deadline()
                        return samples
                    except BaseException as exc:
                        _append(raw, {"event": "error", "request_id": rid,
                            "error": str(exc) if isinstance(exc, _Incomplete) else type(exc).__name__})
                        if parent is not None and (isinstance(exc, _Incomplete) and str(exc) == "CONTEXT_BLOCKED"
                                or isinstance(exc, ValueError) and str(exc).startswith("CONTEXT_BLOCKED:")):
                            blocked = {"kind": kind, "request_id": rid, "problem_id": actor.problem_id,
                                "status": "CONTEXT_BLOCKED", "parent_sample_id": parent.sample_id,
                                "prompt_token_ids": list(ids or []), "prompt_tokens": len(ids or []),
                                "reserved_output": _CAP, "n": n, "seeds": list(request.seeds)}
                            _append(records, blocked)
                            retry_blocked.append(blocked)
                            return []
                        raise

                for actor in actors:
                    deadline()
                    images = load_images(actor, image_root)
                    gray = [Image.new("RGB", image.size, (128, 128, 128)) for image in images]
                    try:
                        image_snapshots[actor.problem_id] = _image_binding(actor, images)
                        items[actor.problem_id] = query(actor, images, "T1")
                        query(actor, gray, "blind")
                    finally:
                        for image in images + gray:
                            image.close()
                for actor in subset:
                    parents = select_failures(items[actor.problem_id], 2, f"phase-a:{seed}:{actor.problem_id}")
                    retry_inapplicable.append({"problem_id": actor.problem_id, "failed_parents": len(parents),
                        "missing_parent_slots": 2 - len(parents), "reason": "NO_ADDITIONAL_FAILED_T1_PARENT"})
                    images = load_images(actor, image_root)
                    try:
                        _require(_image_binding(actor, images) == image_snapshots[actor.problem_id], "IMAGE_CHANGED")
                        for parent in parents:
                            second = query(actor, images, "T2", parent)
                            third = select_failures(second, 1, f"phase-a:T3:{seed}:{parent.sample_id}")
                            failed_ids = [sample.sample_id for sample in second if sample.result is not None
                                          and sample.result.status in RETRYABLE]
                            selection = {"kind": "T3_parent_selection", "problem_id": actor.problem_id,
                                "T2_parent_sample_id": parent.sample_id,
                                "T2_group_id": second[0].group_id if second else f"phase-a:{actor.problem_id}:T2:{parent.sample_id}",
                                "status": "UPSTREAM_CONTEXT_BLOCKED" if not second else "SELECTED" if third else "NO_FAILED_T2_PARENT",
                                "failed_sample_ids": failed_ids, "selected_sample_ids": [sample.sample_id for sample in third],
                                "unselected_failures": len(failed_ids) - len(third) if second else None}
                            _append(records, selection)
                            third_parent_coverage.append(selection)
                            for second_parent in third:
                                query(actor, images, "T3", second_parent)
                    finally:
                        for image in images:
                            image.close()
                deadline()
                _require(_sha(_read(build_root / "manifest.yaml")) == configuration["build_id"]
                    and _sha(_read(split_root / "actor_inputs.jsonl")) == actor_sha
                    and _sha(_read(split_root / "manifest.yaml")) == vault_sha
                    and code_hashes() == source and _hash(protocol) == configuration["resolved_protocol_sha256"]
                    and input_binding == configuration["input_binding"], "INPUT_OR_SOURCE_CHANGED")
                # Reopen every source image after sampling; metadata alone is not a byte check.
                for actor in actors:
                    deadline()
                    images = load_images(actor, image_root)
                    try:
                        _require(_image_binding(actor, images) == image_snapshots[actor.problem_id], "IMAGE_CHANGED")
                    finally:
                        for image in images:
                            image.close()
                t2_blocked = [row for row in retry_blocked if row["kind"] == "T2"]
                t3_blocked = [row for row in retry_blocked if row["kind"] == "T3"]
                upstream = [row for row in third_parent_coverage if row["status"] == "UPSTREAM_CONTEXT_BLOCKED"]
                turn_caps = {str(turn): _turn_cap(observations, actors, kind,
                    [row for row in retry_blocked if row["kind"] == kind], upstream if turn == 3 else ())
                    for turn, kind in ((1, "T1"), (2, "T2"), (3, "T3"))}
                summary.update(turn_cap_recommendations=turn_caps,
                    max_new_tokens={turn: result["cap"] for turn, result in turn_caps.items() if result["cap"] is not None},
                    t1_cap_recommendation=turn_caps["1"], t2_cap_recommendation=turn_caps["2"],
                    t3_cap_recommendation=turn_caps["3"],
                    attribution={kind: _attribution(observations, actors, kind) for kind in ("T1", "T2", "T3")})
                cap = turn_caps["1"]["cap"]
                retry = [r["sample"] for r in observations if r["kind"] == "T2"]
                retry_correct = sum(s["result"]["status"] == "CORRECT" for s in retry)
                summary.update(retry={"subset_size": retry_subset_size, "parent_coverage": retry_inapplicable,
                        "requested_parent_slots": 2 * retry_subset_size,
                        "eligible_parent_slots": sum(r["failed_parents"] for r in retry_inapplicable),
                        "inapplicable_parent_slots": sum(r["missing_parent_slots"] for r in retry_inapplicable),
                        "requested_samples": 8 * retry_subset_size,
                        "eligible_samples": 4 * sum(r["failed_parents"] for r in retry_inapplicable),
                        "context_blocked_parent_slots": len(t2_blocked), "context_blocked_samples": 4 * len(t2_blocked),
                        "blocked_branches": t2_blocked, "n": len(retry), "correct": retry_correct,
                        "success_rate": retry_correct / len(retry) if retry and not t2_blocked else None,
                        "success_rate_conditional_on_executed": retry_correct / len(retry) if retry else None,
                        "sparse_signal": retry_correct / len(retry) < .05 if retry and not t2_blocked else None,
                        "measurement_cap": _CAP, "cap_role": "T2_MEASUREMENT_ONLY_NOT_TRAINING_CAP"})
                third_retry = [row["sample"] for row in observations if row["kind"] == "T3"]
                third_correct = sum(sample["result"]["status"] == "CORRECT" for sample in third_retry)
                third_eligible = sum(row["status"] == "SELECTED" for row in third_parent_coverage)
                summary["third_retry"] = {"parent_coverage": third_parent_coverage,
                    "potential_parent_slots": 2 * retry_subset_size,
                    "requested_parent_slots": len(third_parent_coverage), "eligible_parent_slots": third_eligible,
                    "inapplicable_parent_slots": sum(row["status"] == "NO_FAILED_T2_PARENT" for row in third_parent_coverage),
                    "upstream_unmeasured_parent_slots": len(upstream),
                    "requested_samples": 4 * len(third_parent_coverage), "eligible_samples": 4 * third_eligible,
                    "context_blocked_parent_slots": len(t3_blocked), "context_blocked_samples": 4 * len(t3_blocked),
                    "blocked_branches": t3_blocked, "n": len(third_retry), "correct": third_correct,
                    "success_rate": third_correct / len(third_retry) if third_retry and not t3_blocked and not upstream else None,
                    "success_rate_conditional_on_executed": third_correct / len(third_retry) if third_retry else None,
                    "measurement_cap": _CAP, "cap_role": "T3_MEASUREMENT_ONLY_NOT_TRAINING_CAP"}
                for actor in actors:
                    roots = items[actor.problem_id]
                    blind = [r["sample"] for r in observations if r["kind"] == "blind" and r["sample"]["problem_id"] == actor.problem_id]
                    summary["rows"].append({"problem_id": actor.problem_id, "family": actor.family,
                        "n": 8, "correct": sum(s.result is not None and s.result.status == "CORRECT" for s in roots), "blind_n": 4,
                        "blind_correct": sum(s["result"]["status"] == "CORRECT" for s in blind),
                        "truncated_at_8192": sum(s.finish_reason == "length" for s in roots),
                        "truncated": sum(s.finish_reason == "length" or len(s.completion_token_ids) > cap for s in roots)
                            if cap is not None else None,
                        "n_choices": len(actor.choices), "context_ok": True,
                        "context_ok_scope": "T1_AND_BLIND_ONLY"})
                summary["measurement_status"] = {"T1": turn_caps["1"]["status"], "blind": "COMPLETED",
                    "T2": turn_caps["2"]["status"], "T3": turn_caps["3"]["status"]}
                summary["errors"] = [kind + "_CONTEXT_BLOCKED" for kind in ("T2", "T3")
                    if any(row["kind"] == kind for row in retry_blocked)]
                summary["errors"].extend("T" + turn + "_NO_COMPLETED_FINAL_ANSWERS_FOR_CAP"
                    for turn, result in turn_caps.items() if result["status"] == "NO_COMPLETED_FINAL_ANSWERS")
                summary["status"] = "INCOMPLETE" if summary["errors"] else "MEASUREMENTS_COMPLETED_NOT_PHASE_A_APPROVAL"
    except BaseException as exc:
        summary.update(status="INCOMPLETE", max_new_tokens={}, turn_cap_recommendations={},
            t1_cap_recommendation=None, t2_cap_recommendation=None, t3_cap_recommendation=None,
            attribution={}, retry=None, third_retry=None, rows=[])
        summary["errors"] = [str(exc) if isinstance(exc, _Incomplete) else type(exc).__name__]
        if not isinstance(exc, Exception):
            interrupted = exc
    for name in ("records.jsonl", "sampler_calls.jsonl"):
        if not (output / name).exists():
            with (output / name).open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
    if not (output / "verifier_manifest.json").exists():
        _write(output / "verifier_manifest.json", {"status": "NOT_STARTED"})
    summary.update(images=image_snapshots, observed_records=len(observations),
        elapsed_seconds=time.monotonic() - began,
        files={name: _sha(_read(output / name)) for name in
            ("manifest.json", "verifier_manifest.json", "records.jsonl", "sampler_calls.jsonl")})
    summary["summary_sha256"] = _hash(summary)
    _write(output / "summary.json", summary)
    if interrupted is not None:
        raise interrupted
    return summary
