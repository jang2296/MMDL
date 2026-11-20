"""Native first-attempt ladder inference core; callers own all execution/data gates.

No model loading, CLI, training, grading, or provenance certification happens here.
The supplied engine must already represent the separately verified F0 or FT policy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time
from typing import Any

from . import prompts
from .approval import code_hashes
from .cli import artifact_path
from .dataset import _bytes, _sha, _write, load_images
from .engines import RolloutRequest
from .input_preparation import tensor_identity
from .local_ladder import _descriptor, _json, load_ladder_actors
from .rollout import _sample_request, seeds
from .schema import ActorItem, from_json, to_json
from .vault import _safe_root

VERSION = "local-native-ladder-first-attempt-v1"
CONTEXT_LIMIT = 16384
OUTPUT_TOKENS = 1024
CIRCUIT_PROMPT_POLICY = "generic-circuit-observations-v1"
CIRCUIT_REFERENCE_PROMPT_POLICY = "circuit-ladder-reference-notebook-v1"
CIRCUIT_REPAIR_PROMPT_POLICY = "circuit-ladder-verified-repair-notebook-v1"


def circuit_evaluation_prompts(policy: str = CIRCUIT_PROMPT_POLICY) -> dict:
    """Explicit evaluation-only templates; graph training trace prompts stay separate."""
    templates = prompts.load_prompts(Path(__file__).with_name("prompts_v1.yaml"))
    if policy == CIRCUIT_PROMPT_POLICY:
        return templates | {"evaluation_prompt_policy": policy}
    if policy in (CIRCUIT_REFERENCE_PROMPT_POLICY, CIRCUIT_REPAIR_PROMPT_POLICY):
        from . import circuit_trace
        from .generators import circuit
        return templates | {"evaluation_prompt_policy": policy,
            "solution_trace_contract": circuit_trace.VERSION,
            "source_observation_instruction": {circuit.GENERATOR_VERSION: circuit.OBSERVATION_INSTRUCTION}}
    raise ValueError("ladder generation: canonical circuit evaluation prompt policy required")


def _digest(value: Any) -> str:
    return _sha(_bytes(value))


def _sha256(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def protocol(prompt_templates: dict, local_protocol_sha256: str, *, context_limit: int = CONTEXT_LIMIT) -> dict:
    if type(prompt_templates) is not dict or not _sha256(local_protocol_sha256):
        raise ValueError("ladder generation: protocol-bound prompt templates required")
    if type(context_limit) is not int or context_limit not in (16384, 24576):
        raise ValueError("ladder generation: canonical native context limit required")
    policy = {}
    if "evaluation_prompt_policy" in prompt_templates:
        selected = prompt_templates["evaluation_prompt_policy"]
        if prompt_templates != circuit_evaluation_prompts(selected):
            raise ValueError("ladder generation: canonical circuit evaluation prompt policy required")
        policy = {"evaluation_prompt_policy": selected,
                  "evaluation_prompt_policy_sha256": _digest(prompt_templates)}
        if selected in (CIRCUIT_REFERENCE_PROMPT_POLICY, CIRCUIT_REPAIR_PROMPT_POLICY):
            if context_limit != 24576:
                raise ValueError("ladder generation: reference notebook evaluation requires context 24576")
            public = prompts.circuit_solution_prompt_contract()
            policy.update(solution_trace_contract=public["trace_contract"],
                          solution_prompt_contract=public,
                          solution_prompt_contract_sha256=_digest(public))
    return {"protocol_id": VERSION, "local_protocol_sha256": local_protocol_sha256,
        "prompt_templates_sha256": _digest(prompt_templates), "n": 1, "max_attempts": 1,
        "max_new_tokens": OUTPUT_TOKENS, "context_limit": context_limit,
        "master_seed": 3407, "seed_rule": "rollout.seeds(policy_version=0,turn=1,branch=protocol_id)",
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                     "presence_penalty": 0.0, "repetition_penalty": 1.0},
        "purpose": "bounded native diagnostic; not the official MMMU evaluation protocol", **policy}


def _seed(problem_id: str) -> int:
    return seeds(problem_id, 0, 1, VERSION, 1, master_seed=3407)[0]


def _input(actor, request: RolloutRequest, engine: Any, *, context_limit: int) -> dict:
    prepared = engine.prepare(request)
    grid = prepared.pixels.get("image_grid_thw")
    if (type(engine.context_limit) is not int or engine.context_limit != context_limit
            or set(prepared.pixels) != {"pixel_values", "image_grid_thw"} or grid is None
            or list(grid.shape) != [len(request.images), 3] or grid.is_floating_point()
            or len(prepared.images) != len(request.images)
            or not prepared.token_ids or any(type(token) is not int or token < 0 for token in prepared.token_ids)
            or not bool((grid > 0).all())
            or len(prepared.token_ids) + request.max_new_tokens > context_limit):
        raise ValueError("CONTEXT_BLOCKED: context or actual image tensor coverage differs")
    return {"actor": json.loads(to_json(actor)), "messages": request.messages,
        "prompt_token_ids": list(prepared.token_ids), "image_key": prepared.image_key,
        "input_tokens": len(prepared.token_ids), "image_token_id": engine.model_config.image_token_id,
        "image_tokens": prepared.token_ids.count(engine.model_config.image_token_id),
        "image_spans": [list(span) for span in prepared.image_spans],
        "pixel_tensors": {key: tensor_identity(value) for key, value in prepared.pixels.items()},
        "image_grid_thw": grid.tolist(),
        "original_image_sizes": [list(image.size) for image in request.images],
        "processed_image_sizes": [list(image.size) for image in prepared.images]}


def generate_ladder(cohort: Path, output: Path, engine: Any, prompt_templates: dict, *,
                    expected_preparation_sha256: str, local_protocol_sha256: str,
                    deadline: float | None = None, context_limit: int = CONTEXT_LIMIT) -> dict:
    """Record exactly ten native first attempts from an already gated live engine.

    The caller owns the execution deadline. Partial per-call files survive a later
    failure; only a complete run gets result.json.
    Full text, literal IDs and log probabilities are retained without answer extraction.
    """
    started = time.monotonic()
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("ladder generation: finite monotonic deadline required")

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("ladder generation: evaluation deadline reached")

    check_deadline()
    root, output = _safe_root(cohort), Path(output).absolute()
    if (".." in output.parts or any(path.is_symlink() for path in (output, *output.parents))
            or output.exists() or output.is_relative_to(root)):
        raise ValueError("ladder generation: preserve output and input directories")
    preparation = _descriptor(root / "preparation.json")
    if (not _sha256(expected_preparation_sha256)
            or preparation["file_sha256"] != expected_preparation_sha256):
        raise ValueError("ladder generation: pinned preparation SHA required")
    manifest, actors, sources = _json(root / "preparation.json"), load_ladder_actors(root), code_hashes()
    if (manifest.get("status") != "CPU_COHORT_PREPARED_PENDING_GATES" or manifest.get("count") != 10
            or manifest.get("gpu_evaluation_completed") is not False
            or len({actor.problem_id for actor in actors}) != 10
            or any(actor.mode != "obs" or actor.answer_format != "unit_number" or actor.choices for actor in actors)):
        raise ValueError("ladder generation: fixed ten native unit-answer inputs required")
    contract = protocol(prompt_templates, local_protocol_sha256, context_limit=context_limit)
    version = engine.policy_version
    if (type(version) is not int or version < 0 or type(engine.context_limit) is not int
            or engine.context_limit != context_limit):
        raise ValueError("ladder generation: explicit policy and context required")

    def request(actor, images):
        return RolloutRequest(actor.problem_id + ":T1", None,
            [prompts.user_message(prompts.turn1_text(actor, prompt_templates), len(images))],
            images, 1, OUTPUT_TOKENS, [_seed(actor.problem_id)])

    # Check all ten reservations before the first call; never truncate or replace an item.
    inputs = []
    for actor in actors:
        check_deadline()
        images = load_images(actor, root / "images")
        try:
            inputs.append(_input(actor, request(actor, images), engine, context_limit=context_limit))
        finally:
            for image in images:
                image.close()
    output = artifact_path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    rows: list[dict] = []
    for actor, expected in zip(actors, inputs, strict=True):
        check_deadline()
        images = load_images(actor, root / "images")
        try:
            call = request(actor, images)
            if _input(actor, call, engine, context_limit=context_limit) != expected:
                raise ValueError("ladder generation: actual inputs changed before sampling")
            check_deadline()
            begin = time.monotonic()
            sampled = _sample_request(call, engine, version, context_limit)
            elapsed = time.monotonic() - begin
            completion = sampled.completions[0]
            row = {"problem_id": actor.problem_id, "request_id": call.request_id,
                "policy_version": version, "seed": call.seeds[0], "input": expected,
                "n": call.n, "max_new_tokens": call.max_new_tokens, "context_limit": engine.context_limit,
                "input_sha256": _digest(expected), "text": completion.text,
                "completion_token_ids": list(completion.token_ids), "logprobs": list(completion.logprobs),
                "completion_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason, "elapsed_seconds": elapsed}
            try:
                row["record_sha256"] = _digest(row)
            except (TypeError, ValueError):
                _write(output / f"invalid-response-{len(rows):04d}.json", _bytes({
                    "request_id": call.request_id, "text": str(completion.text),
                    "completion_token_ids_repr": repr(completion.token_ids),
                    "logprobs_repr": repr(completion.logprobs), "finish_reason": str(completion.finish_reason)}) + b"\n")
                raise
            _write(output / f"response-{len(rows):04d}.json", _bytes(row) + b"\n")
            check_deadline()
            _validate_row(row, context_limit=context_limit)
            if (_input(actor, call, engine, context_limit=context_limit) != expected
                    or sampled.prompt_token_ids != expected["prompt_token_ids"]):
                raise ValueError("ladder generation: actual inputs changed during sampling")
            if engine.tokenizer.decode(completion.token_ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False) != completion.text:
                raise ValueError("ladder generation: tokenizer decode differs from actual text")
            rows.append(row)
        finally:
            for image in images:
                image.close()
    if (_descriptor(root / "preparation.json") != preparation or load_ladder_actors(root) != actors
            or code_hashes() != sources
            or protocol(prompt_templates, local_protocol_sha256, context_limit=context_limit) != contract):
        raise ValueError("ladder generation: source/input/protocol changed during generation")
    check_deadline()
    result = {"format": VERSION, "status": "TEN_FIRST_ATTEMPTS_RECORDED", "protocol": contract,
        "protocol_sha256": _digest(contract), "preparation": preparation, "source_sha256": sources,
        "problem_ids": [actor.problem_id for actor in actors], "policy_version": version,
        "rows": rows, "elapsed_seconds": time.monotonic() - started,
        "usage": {"input_tokens": sum(row["input"]["input_tokens"] for row in rows),
                  "image_tokens": sum(row["input"]["image_tokens"] for row in rows),
                  "completion_tokens": sum(row["completion_tokens"] for row in rows)},
        "training_allowed": False, "evaluation_execution_authorized": False,
        "model_generation_provenance_verified": False, "accuracy": None}
    check_deadline()
    _write(output / "result.json", _bytes(result) + b"\n")
    return result


def _validate_row(row: dict, *, context_limit: int) -> None:
    if type(row) is not dict:
        raise ValueError("ladder generation: response object required")
    ids, logs = row.get("completion_token_ids"), row.get("logprobs")
    inp = row.get("input")
    if (type(inp) is not dict or row.get("input_sha256") != _digest(inp)
            or type(row.get("problem_id")) is not str or row.get("seed") != _seed(row["problem_id"])
            or type(row.get("seed")) is not int or row.get("request_id") != row["problem_id"] + ":T1"
            or type(row.get("policy_version")) is not int or row["policy_version"] < 0
            or any(type(row.get(key)) is not int or row[key] != value for key, value in
                   (("n", 1), ("max_new_tokens", OUTPUT_TOKENS), ("context_limit", context_limit)))
            or type(row.get("text")) is not str or row.get("finish_reason") not in ("stop", "length")
            or type(ids) is not list or not 1 <= len(ids) <= OUTPUT_TOKENS
            or type(row.get("completion_tokens")) is not int or row["completion_tokens"] != len(ids)
            or any(type(token) is not int or token < 0 for token in ids)
            or type(logs) is not list or len(logs) != len(ids)
            or any(type(log) not in (int, float) or not math.isfinite(log) or log > 1e-5 for log in logs)
            or type(row.get("elapsed_seconds")) not in (int, float)
            or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0):
        raise ValueError("ladder generation: invalid actual response contract")
    tokens = inp.get("prompt_token_ids")
    if (type(tokens) is not list or not tokens or len(tokens) + OUTPUT_TOKENS > context_limit
            or any(type(token) is not int or token < 0 for token in tokens)
            or type(inp.get("input_tokens")) is not int or inp["input_tokens"] != len(tokens)
            or type(inp.get("image_token_id")) is not int or inp["image_token_id"] < 0
            or type(inp.get("image_tokens")) is not int or inp["image_tokens"] != tokens.count(inp["image_token_id"])
            or type(inp.get("actor")) is not dict
            or inp.get("actor", {}).get("problem_id") != row["problem_id"]):
        raise ValueError("ladder generation: invalid actual input contract")


def validate_record(result: dict) -> None:
    """Validate one complete literal first-attempt record without inventing a pair."""
    if type(result) is not dict or type(result.get("protocol")) is not dict:
        raise ValueError("ladder generation: bound generation result required")
    contract = result.get("protocol", {})
    rows, ids = result.get("rows"), result.get("problem_ids")
    templates = (circuit_evaluation_prompts(contract["evaluation_prompt_policy"])
                 if "evaluation_prompt_policy" in contract else {})
    expected = protocol(templates, contract.get("local_protocol_sha256"), context_limit=contract.get("context_limit"))
    if not templates:
        expected["prompt_templates_sha256"] = contract.get("prompt_templates_sha256")
    if (result.get("format") != VERSION or result.get("status") != "TEN_FIRST_ATTEMPTS_RECORDED"
            or _bytes(contract) != _bytes(expected) or not _sha256(contract.get("prompt_templates_sha256"))
            or result.get("protocol_sha256") != _digest(contract)
            or type(rows) is not list or len(rows) != 10 or type(ids) is not list
            or any(type(row) is not dict for row in rows)
            or len(ids) != 10 or any(type(pid) is not str for pid in ids) or len(set(ids)) != 10
            or [row.get("problem_id") for row in rows] != ids
            or type(result.get("policy_version")) is not int
            or not result.get("source_sha256") or not result.get("preparation")):
        raise ValueError("ladder generation: complete bound first-attempt records required")
    for row in rows:
        _validate_row(row, context_limit=contract["context_limit"])
        if (row.get("record_sha256") != _digest({k: v for k, v in row.items() if k != "record_sha256"})
                or row["policy_version"] != result.get("policy_version")):
            raise ValueError("ladder generation: recorded response binding changed")
        if templates:
            actor = from_json(ActorItem, json.dumps(row["input"]["actor"]))
            messages = [prompts.user_message(prompts.turn1_text(actor, templates), len(actor.image_paths))]
            if row["input"].get("messages") != messages:
                raise ValueError("ladder generation: literal canonical circuit prompt messages required")
    if result.get("usage") != {"input_tokens": sum(row["input"]["input_tokens"] for row in rows),
        "image_tokens": sum(row["input"]["image_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows)}:
        raise ValueError("ladder generation: recorded token usage changed")


def validate_paired_records(left: dict, right: dict) -> dict:
    """Check recorded F0/FT conditions only; records cannot certify model provenance."""
    for result in (left, right):
        validate_record(result)
    if any(left[key] != right[key] for key in
           ("protocol", "protocol_sha256", "preparation", "source_sha256", "problem_ids")):
        raise ValueError("ladder generation: paired protocol/preparation/source mismatch")
    for a, b in zip(left["rows"], right["rows"], strict=True):
        if any(a[key] != b[key] for key in ("problem_id", "request_id", "seed", "input", "input_sha256")):
            raise ValueError("ladder generation: paired actual input/pixel/question/image/seed mismatch")
    if left["policy_version"] != 0 or right["policy_version"] <= 0:
        raise ValueError("ladder generation: recorded F0 version zero and updated FT version required")
    return {"status": "RECORDED_PAIRED_CONDITIONS_MATCH", "n_pairs": 10,
            "model_generation_provenance_verified": False, "accuracy": None}
