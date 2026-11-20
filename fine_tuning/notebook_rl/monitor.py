"""Matched dev QA and independent observation health measurements, never training.

Public runs require all 96 dev IDs and 192 independent n=1 official-sampler calls.
No gold is opened here: grading uses the separate VerifierService. Incomplete
measurements always have null health metrics. Deadlines are checked before calls;
an outer process watchdog is still needed to interrupt an in-flight engine/kernel.
Dev QA token lengths are observations, not the Phase-A train-length reference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from .dataset import load_images
from .engines import QWEN_OFFICIAL, RolloutRequest, RolloutResult, _completion
from .prompts import load_prompts, turn1_text, user_message
from .rollout import seeds
from .schema import VerifierResult, from_json, load_protocol, to_json
from .vault import _read, _safe_root, _unique, read_actors
from .verifier_service import VerifierService

VERSION = "notebook-rl-dev-monitor-v1"
_HERE = Path(__file__).resolve().parent
_SCORED = {"CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"}
_BINDING = {"protocol_sha256", "prompts_sha256", "data_sha256", "code_sha256"}
_FILES = {"manifest.json", "verifier_manifest.json", "records.jsonl", "sampler_calls.jsonl"}
_SOURCE = ("monitor.py", "engines.py", "dataset.py", "prompts.py", "rollout.py", "schema.py",
           "verifier_service.py", "verifier.py", "answers.py", "observations.py", "attribution.py",
           "guards.py", "vault.py", "protocol_v1.yaml", "prompts_v1.yaml")


class _Incomplete(Exception):
    pass


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value) -> str:
    return _sha(_json(value))


def _source_hashes() -> dict[str, str]:
    return {name: _sha(_read(_HERE / name)) for name in _SOURCE}


def _configuration(actor_sha, vault_sha, protocol, prompts, input_binding, seed, n) -> dict:
    return {"format": VERSION, "n": n, "samples_per_problem": 1,
        "protocol_sha256": protocol["sha256"], "resolved_protocol_sha256": _hash(protocol),
        "prompts_sha256": prompts["sha256"], "actor_inputs_sha256": actor_sha, "vault_manifest_sha256": vault_sha,
        "input_binding": dict(input_binding), "seed": seed, "sampling": asdict(QWEN_OFFICIAL),
        "decode_cap": protocol["rollout"]["max_new_tokens"][1], "qa_prompt_mode": "plain",
        "grounding_prompt_mode": "obs", "seed_policy_version": 0, "code_sha256": _source_hashes()}


def _image_binding(actor, images) -> list[dict]:
    return [{"sha256": sha, "size": list(image.size), "mode": image.mode, "pixels_sha256": _sha(image.tobytes())}
            for sha, image in zip(actor.image_sha256, images, strict=True)]


def _frozen(protocol: dict, input_binding: dict, seed: int) -> dict:
    if (type(seed) is not int or seed not in protocol["seeds"] or set(input_binding) != _BINDING
            or any(not isinstance(v, str) or len(v) != 64 or set(v) - set("0123456789abcdef") for v in input_binding.values())
            or input_binding["protocol_sha256"] != protocol["sha256"]
            or input_binding["prompts_sha256"] != protocol["prompts_sha256"]):
        raise ValueError("monitor: seed and four checkpoint SHA bindings required")
    if protocol != load_protocol(_HERE / "protocol_v1.yaml"):
        raise ValueError("monitor: actual protocol bytes/content must match")
    prompts = load_prompts(_HERE / "prompts_v1.yaml")
    if prompts["sha256"] != protocol["prompts_sha256"]:
        raise ValueError("monitor: prompts/protocol differ")
    return prompts


def _require(condition, code: str) -> None:
    if not condition:
        raise _Incomplete(code)


def _write(path: Path, value: dict) -> None:
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append(stream, value: dict) -> None:
    stream.write(_json(value) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def _aggregate(rows: list[dict], n: int) -> dict:
    qa, obs = ([r for r in rows if r["kind"] == kind] for kind in ("qa", "obs"))
    _require(len(qa) == len(obs) == n and len({r["problem_id"] for r in qa}) == n
             and {r["problem_id"] for r in qa} == {r["problem_id"] for r in obs}, "MEASUREMENT_COVERAGE")
    results = [from_json(VerifierResult, _json(r["result"]).decode()) for r in rows]
    _require(all(r.status in _SCORED for r in results), "VERIFIER_UNSCORABLE")
    slots = [from_json(VerifierResult, _json(r["result"]).decode()).slots for r in obs]
    required = sum(s.required for s in slots if s is not None)
    correct = sum(s.required_correct for s in slots if s is not None)
    _require(required > 0, "NO_REQUIRED_GROUNDING_SLOTS")
    return {"dev_first_attempt_acc": sum(r["result"]["status"] == "CORRECT" for r in qa) / n,
            "grounding_probe": correct / required,
            "grounding_counts": {"required_correct": correct, "required_total": required,
                                 "items_with_required_slots": sum(s is not None and s.required > 0 for s in slots),
                                 "independent_obs_requests": n},
            "len_median": statistics.median(len(r["completion_token_ids"]) for r in qa)}


def _reference(path: Path, n: int) -> tuple[dict, str]:
    _require(path.name == "summary.json", "F0_REFERENCE_PATH")
    _safe_root(path.parent)
    raw = _read(path)
    reference = json.loads(raw, object_pairs_hook=_unique)
    _require(reference.get("format") == VERSION and reference.get("status") == "COMPLETED"
             and type(reference.get("step")) is int and reference["step"] == 0
             and reference.get("n") == n and reference.get("samples_per_problem") == 1
             and reference.get("summary_sha256") == _hash({k: v for k, v in reference.items() if k != "summary_sha256"})
             and reference.get("measurement_sha256") == _hash(reference["measurement"]), "F0_REFERENCE_NOT_COMPLETE")
    _require(set(reference.get("files", {})) == _FILES, "F0_FILE_COVERAGE")
    for name, digest in reference["files"].items():
        _require(_sha(_read(path.parent / name)) == digest, "F0_RAW_FILE_CHANGED")
    rows = [json.loads(line, object_pairs_hook=_unique) for line in _read(path.parent / "records.jsonl").splitlines()]
    values = _aggregate(rows, n)
    _require(all(reference.get(key) == value for key, value in values.items()), "F0_METRIC_RAW_MISMATCH")
    _require(len(reference["measurement"].get("items", [])) == n, "F0_INPUT_COVERAGE")
    _reference_ledgers(reference, rows, path.parent, n)
    return reference, _sha(raw)


def _reference_ledgers(reference: dict, rows: list[dict], root: Path, n: int) -> None:
    configuration = reference["measurement"]["configuration"]
    calls = [json.loads(line, object_pairs_hook=_unique) for line in _read(root / "sampler_calls.jsonl").splitlines()]
    manifest = json.loads(_read(root / "manifest.json"), object_pairs_hook=_unique)
    verifier = json.loads(_read(root / "verifier_manifest.json"), object_pairs_hook=_unique)
    _require(len(calls) == len(rows) == 2 * n and reference.get("errors") == []
             and reference.get("completed_qa") == reference.get("completed_obs") == n
             and all(reference.get(key) == configuration[key] for key in
                     ("protocol_sha256", "actor_inputs_sha256", "input_binding", "sampling", "decode_cap"))
             and all(manifest.get(key) == value for key, value in configuration.items())
             and manifest.get("step") == manifest.get("policy_version") == 0
             and manifest.get("independent_requests_planned") == 2 * n
             and verifier.get("vault_manifest_sha256") == configuration["vault_manifest_sha256"], "F0_LEDGER_IDENTITY")
    token_count = 0
    for i, (row, call) in enumerate(zip(rows, calls, strict=True)):
        kind = "qa" if i % 2 == 0 else "obs"
        item = reference["measurement"]["items"][i // 2]
        request = row["request"]
        mode = "plain" if kind == "qa" else "obs"
        planned = {"seed": row["seed"], "mode": mode, "messages_sha256": _hash(request["messages"]),
                   "prompt_token_ids_sha256": _hash(row["prompt_token_ids"])}
        result = call["results"][0] if len(call.get("results", [])) == 1 else None
        completion = {"token_ids": row["completion_token_ids"], "logprobs": row["logprobs"],
                      "finish_reason": row["finish_reason"], "text": row["raw_text"]}
        _require(row["kind"] == kind and row["problem_id"] == item["problem_id"] and row["error"] is None
                 and request["request_id"] == f"health:0:{item['problem_id']}:{kind}"
                 and request["n"] == 1 and request["max_new_tokens"] == configuration["decode_cap"]
                 and request["seeds"] == [row["seed"]] and request["images"] == item["images"]
                 and planned == item["requests"][kind] and call["request"] == request
                 and call["sampling"] == configuration["sampling"]
                 and call["sample_seconds"] == row["sample_seconds"] and result is not None
                 and result["request_id"] == request["request_id"]
                 and result["prompt_token_ids"] == row["prompt_token_ids"]
                 and result["completions"] == [completion]
                 and row["prompt_token_ids"] and all(type(t) is int and t >= 0 for t in row["prompt_token_ids"])
                 and all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                         for k in ("prepare_seconds", "sample_seconds", "grade_seconds")), "F0_RAW_LEDGER_MISMATCH")
        _completion(completion["token_ids"], completion["logprobs"], completion["finish_reason"], completion["text"])
        _require(len(completion["token_ids"]) <= configuration["decode_cap"], "F0_DECODE_CAP")
        token_count += len(completion["token_ids"])
    _require(reference.get("sampling_cost_counts") == {"planned_sampler_calls": 2 * n, "attempted_sampler_calls": 2 * n,
             "returned_sampler_calls": 2 * n, "returned_completions": 2 * n, "returned_tokens": token_count,
             "verifier_calls": 2 * n}, "F0_COST_LEDGER_MISMATCH")


def validate_f0_reference(path: Path, split_root: Path, image_root: Path, *, protocol: dict,
                          input_binding: dict, seed: int = 3407) -> dict:
    """Read-only dev96 preallocation check; raises ValueError, never creates approval.

    Verifies file/input/source consistency, not authenticity of a fabricated bundle
    whose hashes and producer records were all replaced. Protect the original F0.
    Actual processor-token parity is additionally checked by each live measurement.
    """
    try:
        return _validate_reference(path, split_root, image_root, protocol=protocol,
                                   input_binding=input_binding, seed=seed, _expected_n=96)
    except Exception as exc:
        raise ValueError("monitor: invalid F0 reference: " + (str(exc) if isinstance(exc, _Incomplete) else type(exc).__name__)) from None


def _validate_reference(path, split_root, image_root, *, protocol, input_binding, seed, _expected_n):
    prompts = _frozen(protocol, input_binding, seed)
    split_root, image_root = _safe_root(Path(split_root)), _safe_root(Path(image_root))
    actors = sorted(read_actors(split_root), key=lambda actor: actor.problem_id)
    _require(split_root.name == "dev" and len(actors) == _expected_n, "F0_DEV_COVERAGE")
    actor_sha, vault_sha = (_sha(_read(split_root / name)) for name in ("actor_inputs.jsonl", "manifest.yaml"))
    configuration = _configuration(actor_sha, vault_sha, protocol, prompts, input_binding, seed, _expected_n)
    reference, digest = _reference(Path(path), _expected_n)
    _require(reference["measurement"]["configuration"] == configuration, "F0_CONFIGURATION_MISMATCH")
    for actor, item in zip(actors, reference["measurement"]["items"], strict=True):
        images = load_images(actor, image_root)
        try:
            _require(item["problem_id"] == actor.problem_id and item["images"] == _image_binding(actor, images), "F0_IMAGE_MISMATCH")
            for kind, mode in (("qa", "plain"), ("obs", "obs")):
                messages = [user_message(turn1_text(replace(actor, mode=mode), prompts), len(images))]
                expected_seed = seeds(actor.problem_id, 0, 1, "dev-health:" + kind, 1, master_seed=seed)[0]
                _require(item["requests"][kind]["seed"] == expected_seed
                         and item["requests"][kind]["mode"] == mode
                         and item["requests"][kind]["messages_sha256"] == _hash(messages), "F0_QUESTION_OR_SEED_MISMATCH")
        finally:
            for image in images:
                image.close()
    _require(_source_hashes() == configuration["code_sha256"]
             and _sha(_read(split_root / "actor_inputs.jsonl")) == actor_sha
             and _sha(_read(split_root / "manifest.yaml")) == vault_sha, "INPUT_OR_CODE_CHANGED")
    return {"status": "VALIDATED_F0_REFERENCE", "n": _expected_n, "samples_per_problem": 1,
            "protocol_sha256": protocol["sha256"], "actor_inputs_sha256": actor_sha,
            "measurement_sha256": reference["measurement_sha256"], "f0_report_sha256": digest,
            "input_binding": dict(input_binding), "dev_first_attempt_acc": reference["dev_first_attempt_acc"],
            "grounding_probe": reference["grounding_probe"], "len_median": reference["len_median"],
            "len_median_source": "dev_qa_n1_tokens_not_phase_a_train", "approval_created": False}


def measure_dev(split_root: Path, image_root: Path, output: Path, *, engine: Any, protocol: dict,
                step: int, deadline_timestamp: float, input_binding: dict, f0_report: Path | None = None,
                seed: int = 3407, forbidden_roots=(), clock: Callable[[], float] = time.time) -> dict:
    """Production entry point: fixed dev96, no exposed small-count/test verifier override."""
    return _measure(split_root, image_root, output, engine=engine, protocol=protocol, step=step,
        deadline_timestamp=deadline_timestamp, input_binding=input_binding, f0_report=f0_report,
        seed=seed, forbidden_roots=forbidden_roots, clock=clock, _expected_n=96)


def _measure(split_root: Path, image_root: Path, output: Path, *, engine: Any, protocol: dict,
             step: int, deadline_timestamp: float, input_binding: dict, f0_report: Path | None,
             seed: int, forbidden_roots, clock: Callable[[], float], _expected_n: int) -> dict:
    """Private small-fixture harness; public measure_dev always supplies 96."""
    split_root, image_root, output = Path(split_root), Path(image_root), Path(output)
    if (type(step) is not int or step < 0 or step % 12
            or type(deadline_timestamp) not in (int, float) or not math.isfinite(deadline_timestamp)):
        raise ValueError("monitor: frozen schedule, seed, deadline and four checkpoint SHA bindings required")
    prompts = _frozen(protocol, input_binding, seed)
    if split_root.name != "dev":
        raise ValueError("monitor: dev split only")
    actors = sorted(read_actors(split_root), key=lambda actor: actor.problem_id)
    if len(actors) != _expected_n:
        raise ValueError("monitor: exactly 96 dev actors required in production")
    if (getattr(engine, "policy_version", None) != step or type(engine.policy_version) is not int
            or "qwen_official" not in getattr(engine, "supported_sampling", ())
            or not callable(getattr(engine, "sample_with_settings", None))):
        raise ValueError("monitor: synchronized live-policy official-sampler engine required")
    _safe_root(split_root)
    _safe_root(image_root)
    project = _HERE.parents[1]
    model_dir = getattr(engine, "model_dir", None)
    if (output.resolve().is_relative_to(project) or output.resolve().is_relative_to(split_root.resolve())
            or output.resolve().is_relative_to(image_root.resolve()) or output.is_symlink()
            or (model_dir is not None and output.resolve().is_relative_to(Path(model_dir).resolve()))):
        raise ValueError("monitor: external non-source output directory required")
    _safe_root(output.parent, create=True)
    output.mkdir(mode=0o700, exist_ok=False)
    actor_sha = _sha(_read(split_root / "actor_inputs.jsonl"))
    vault_sha = _sha(_read(split_root / "manifest.yaml"))
    cap = protocol["rollout"]["max_new_tokens"][1]
    configuration = _configuration(actor_sha, vault_sha, protocol, prompts, input_binding, seed, _expected_n)
    code = configuration["code_sha256"]
    measurement: dict[str, Any] = {"configuration": configuration, "items": []}
    _write(output / "manifest.json", configuration | {"step": step, "policy_version": step,
        "independent_requests_planned": 2 * _expected_n, "deadline_timestamp": deadline_timestamp,
        "deadline_reserve_seconds": 120, "gpu_execution_certified": False})
    summary: dict = {"format": VERSION, "status": "INCOMPLETE", "step": step, "n": _expected_n,
        "samples_per_problem": 1, "dev_first_attempt_acc": None, "grounding_probe": None,
        "grounding_counts": None, "len_median": None, "len_median_source": "dev_qa_n1_tokens_not_phase_a_train",
        "protocol_sha256": protocol["sha256"], "actor_inputs_sha256": actor_sha,
        "input_binding": dict(input_binding), "sampling": asdict(QWEN_OFFICIAL), "decode_cap": cap,
        "reference_role": "F0" if step == 0 else "CURRENT", "f0_report_sha256": None,
        "matched_f0": None, "gpu_execution_certified": False, "errors": [],
        "sampling_cost_counts": {"planned_sampler_calls": 2 * _expected_n, "attempted_sampler_calls": 0,
            "returned_sampler_calls": 0, "returned_completions": 0, "returned_tokens": 0, "verifier_calls": 0}}
    costs, rows, reference = summary["sampling_cost_counts"], [], None
    began = time.monotonic()
    def check_deadline():
        _require(clock() < deadline_timestamp - 120, "DEADLINE_REACHED")
    try:
        if step:
            _require(f0_report is not None, "F0_REFERENCE_REQUIRED")
            assert f0_report is not None
            reference, summary["f0_report_sha256"] = _reference(Path(f0_report), _expected_n)
            _require(reference["measurement"]["configuration"] == configuration, "F0_CONFIGURATION_MISMATCH")
        else:
            _require(f0_report is None, "F0_CANNOT_BORROW_REFERENCE")
        check_deadline()
        with VerifierService(split_root, forbidden_roots=list(forbidden_roots)) as verifier:
            _require(verifier.metadata["vault_manifest_sha256"] == vault_sha, "VERIFIER_VAULT_MISMATCH")
            _write(output / "verifier_manifest.json", verifier.metadata)
            with (output / "records.jsonl").open("xb") as stream, (output / "sampler_calls.jsonl").open("xb") as raw_stream:
                os.fchmod(stream.fileno(), 0o600)
                os.fchmod(raw_stream.fileno(), 0o600)
                for actor in actors:
                    check_deadline()
                    images = load_images(actor, image_root)
                    try:
                        image_binding = _image_binding(actor, images)
                        item = {"problem_id": actor.problem_id, "images": image_binding, "requests": {}}
                        for kind, mode in (("qa", "plain"), ("obs", "obs")):
                            request_seed = seeds(actor.problem_id, 0, 1, "dev-health:" + kind, 1, master_seed=seed)[0]
                            messages = [user_message(turn1_text(replace(actor, mode=mode), prompts), len(images))]
                            request = RolloutRequest(f"health:{step}:{actor.problem_id}:{kind}", None, messages, images, 1, cap, [request_seed])
                            request_wire = {"request_id": request.request_id, "messages": messages, "images": image_binding,
                                            "n": 1, "max_new_tokens": cap, "seeds": [request_seed]}
                            row = {"problem_id": actor.problem_id, "kind": kind, "seed": request_seed,
                                "request": request_wire, "prompt_token_ids": [], "completion_token_ids": [],
                                "logprobs": [], "raw_text": None, "finish_reason": None, "result": None,
                                "error": None, "prepare_seconds": None, "sample_seconds": None, "grade_seconds": None}
                            try:
                                check_deadline()
                                _require(engine.policy_version == step, "POLICY_VERSION_CHANGED")
                                started = time.monotonic()
                                prepared = engine.prepare(request)
                                row["prepare_seconds"] = time.monotonic() - started
                                prompt_ids = list(prepared.token_ids)
                                _require(prompt_ids and all(type(t) is int and t >= 0 for t in prompt_ids)
                                         and len(prompt_ids) + cap <= protocol["context_limit"], "PROMPT_OR_CONTEXT_INVALID")
                                row["prompt_token_ids"] = prompt_ids
                                planned = {"seed": request_seed, "mode": mode, "messages_sha256": _hash(messages),
                                           "prompt_token_ids_sha256": _hash(prompt_ids)}
                                item["requests"][kind] = planned
                                if reference is not None:
                                    expected = reference["measurement"]["items"][len(measurement["items"])]
                                    _require(expected["problem_id"] == actor.problem_id and expected["images"] == image_binding
                                             and expected["requests"][kind] == planned, "F0_ACTUAL_INPUT_MISMATCH")
                                check_deadline()
                                started = time.monotonic()
                                costs["attempted_sampler_calls"] += 1
                                results = engine.sample_with_settings([request], QWEN_OFFICIAL)
                                row["sample_seconds"] = time.monotonic() - started
                                costs["returned_sampler_calls"] += 1
                                _require(type(results) is list and all(type(r) is RolloutResult for r in results), "RESULT_TYPE")
                                costs["returned_completions"] += sum(len(r.completions) for r in results)
                                costs["returned_tokens"] += sum(len(c.token_ids) for r in results for c in r.completions)
                                # Preserve all actual responses before identity/count/score validation.
                                _append(raw_stream, {"request": request_wire, "results": [asdict(r) for r in results],
                                                     "sample_seconds": row["sample_seconds"], "sampling": asdict(QWEN_OFFICIAL)})
                                _require(engine.policy_version == step and len(results) == 1
                                         and results[0].request_id == request.request_id
                                         and results[0].prompt_token_ids == prompt_ids and len(results[0].completions) == 1,
                                         "POLICY_OR_RESULT_IDENTITY")
                                completion = results[0].completions[0]
                                row.update(completion_token_ids=list(completion.token_ids), logprobs=list(completion.logprobs),
                                           raw_text=completion.text, finish_reason=completion.finish_reason)
                                _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
                                _require(len(completion.token_ids) <= cap and engine.tokenizer.decode(completion.token_ids,
                                         skip_special_tokens=True, clean_up_tokenization_spaces=False) == completion.text, "COMPLETION_TOKEN_TEXT_MISMATCH")
                                check_deadline()
                                started = time.monotonic()
                                costs["verifier_calls"] += 1
                                result = verifier.grade(actor.problem_id, completion.text, completion.finish_reason, mode)
                                row["grade_seconds"] = time.monotonic() - started
                                row["result"] = json.loads(to_json(result))
                                _require(result.status in _SCORED, "VERIFIER_UNSCORABLE")
                                check_deadline()
                            except Exception as exc:
                                row["error"] = str(exc) if isinstance(exc, _Incomplete) else type(exc).__name__
                                raise
                            finally:
                                _append(stream, row)
                                rows.append(row)
                        measurement["items"].append(item)
                    finally:
                        for image in images:
                            image.close()
                _require(engine.policy_version == step, "POLICY_VERSION_CHANGED")
                _require(_sha(_read(split_root / "actor_inputs.jsonl")) == actor_sha
                         and _sha(_read(split_root / "manifest.yaml")) == vault_sha
                         and all(_sha(_read(_HERE / name)) == sha for name, sha in code.items()), "INPUT_OR_CODE_CHANGED")
                values = _aggregate(rows, _expected_n)
                if reference is not None:
                    _require(_hash(measurement) == reference["measurement_sha256"], "F0_MEASUREMENT_MISMATCH")
                summary.update(values, status="COMPLETED", matched_f0=True if reference is not None else None)
    except Exception as exc:
        summary["status"] = "INCOMPLETE"
        summary["errors"] = [str(exc) if isinstance(exc, _Incomplete) else type(exc).__name__]
    # Empty files record unexecuted work without pretending to have a verifier/model result.
    if not (output / "verifier_manifest.json").exists():
        _write(output / "verifier_manifest.json", {"status": "NOT_STARTED"})
    for name in ("records.jsonl", "sampler_calls.jsonl"):
        if not (output / name).exists():
            with (output / name).open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
    summary.update(measurement=measurement, measurement_sha256=_hash(measurement),
        elapsed_seconds=time.monotonic() - began, completed_qa=sum(r["kind"] == "qa" and r["error"] is None for r in rows),
        completed_obs=sum(r["kind"] == "obs" and r["error"] is None for r in rows),
        files={name: _sha(_read(output / name)) for name in sorted(_FILES)})
    if summary["status"] != "COMPLETED":
        summary.update(dev_first_attempt_acc=None, grounding_probe=None, grounding_counts=None, len_median=None, matched_f0=False)
    summary["summary_sha256"] = _hash(summary)
    _write(output / "summary.json", summary)
    return summary
