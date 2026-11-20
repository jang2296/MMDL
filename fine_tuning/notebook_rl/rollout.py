"""Same-state retry trees. Observed failures are data, not labels for SFT.

Only the verifier interface handles private answers. The actor/engine receives
images, question and its own actual prior tokens plus bounded status feedback.
No model-token history is re-tokenized or silently shortened.
"""
from __future__ import annotations

import hashlib
import json
import copy
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

from . import prompts as prompt_builders
from .engines import Completion, RolloutRequest, RolloutResult
from .schema import ActorItem, Cause, DiagTarget, Group, GroupKind, ProblemTree, Sample, from_json, to_json
from .verifier import answer_reward

RETRYABLE = frozenset(("INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"))
BLOCKING = frozenset(("UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR"))
REPAIR_COLLECTOR_VERSION = "repair-retry-collector-v3"
REPAIR_SELECTION_POLICY = "native-first-completed-grammar-fallback-v1"


@dataclass(frozen=True)
class RolloutConfig:
    n: dict[int, int]
    branch: dict[int, int]
    max_attempts: int
    max_new_tokens: dict[int, int]
    context_limit: int
    diag: bool
    n_diag: int
    oracle_counterfactual: bool = False
    self_perception_probe: bool = False
    diag_max_new_tokens: int = 256
    visual_lambda: float = 0.0
    master_seed: int = 3407
    final_notebook: bool = False
    retry_contract: str | None = None

    @classmethod
    def from_protocol(cls, protocol: dict, algorithm: str, *, master_seed: int = 3407):
        if algorithm not in protocol["algorithms"]:
            raise ValueError("rollout: unknown algorithm")
        r = protocol["rollout"]
        cfg = cls(r["n"], r["branch"], 1 if algorithm == "ALG-T1" else 3 if algorithm == "ALG-NB3" else 2,
            r["max_new_tokens"], protocol["context_limit"], algorithm in r["diag"]["enabled_in"],
            r["diag"]["n"], r["oracle_counterfactual"], r["self_perception_probe"],
            r["diag"]["max_new_tokens"], protocol["reward"]["visual_lambda"] if algorithm == "ALG-NB+D+V" else 0,
            master_seed, r.get("final_notebook", False), r.get("retry_contract"))
        if cfg.retry_contract is not None:
            cfg.validate()
        return cfg

    def validate(self) -> None:
        if (self.max_attempts not in (1, 2, 3) or type(self.context_limit) is not int or self.context_limit < 1
                or any(type(self.n.get(t)) is not int or self.n[t] < 2 for t in range(1, self.max_attempts+1))
                or any(type(self.max_new_tokens.get(t)) is not int or self.max_new_tokens[t] < 1
                       for t in range(1, self.max_attempts+1))
                or any(type(self.branch.get(t)) is not int or self.branch[t] < 0 for t in (1, 2))
                or self.n_diag < 2 or self.diag_max_new_tokens < 1 or not 0 <= self.visual_lambda <= .1
                or type(self.master_seed) is not int or not 0 <= self.master_seed < 2**31
                or type(self.final_notebook) is not bool):
            raise ValueError("rollout: invalid explicit budgets")
        if self.oracle_counterfactual or self.self_perception_probe:
            raise ValueError("rollout: optional diagnostic probes require a separate probe runner; never use loss groups")
        if self.retry_contract is not None:
            if (type(self.retry_contract) is not str or self.retry_contract != "repair-check-v1"
                    or self.n != {1: 4, 2: 2, 3: 2} or self.branch != {1: 1, 2: 1}
                    or self.max_new_tokens != {1: 1024, 2: 1024, 3: 1024}
                    or any(type(key) is not int or type(value) is not int
                           for mapping in (self.n, self.branch, self.max_new_tokens) for key, value in mapping.items())
                    or type(self.max_attempts) is not int or self.max_attempts != 3 or self.context_limit != 24576
                    or self.diag is not False or self.final_notebook is not False
                    or type(self.n_diag) is not int or self.n_diag != 4
                    or type(self.diag_max_new_tokens) is not int or self.diag_max_new_tokens != 8192
                    or type(self.visual_lambda) not in (int, float) or self.visual_lambda != 0):
                raise ValueError("rollout: invalid explicit repair collector contract or fixed budgets")


def seeds(problem_id: str, policy_version: int, turn: int, branch: str, n: int,
          *, master_seed: int = 3407) -> list[int]:
    if (not problem_id or policy_version < 0 or turn not in (1, 2, 3) or n < 1
            or type(master_seed) is not int or not 0 <= master_seed < 2**31):
        raise ValueError("rollout: invalid seed identity")
    return [int.from_bytes(hashlib.sha256(f"{master_seed}|{problem_id}|{policy_version}|{turn}|{branch}|{k}".encode()).digest()[:8],
                           "big") % 2**31 for k in range(n)]


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select_failures(samples: list[Sample], k: int, key: str, *,
                    process_contradictions: set[str] | None = None) -> list[Sample]:
    if type(k) is not int or k < 0:
        raise ValueError("rollout: nonnegative branch count required")
    if process_contradictions is not None:
        ids = [sample.sample_id for sample in samples]
        if (type(process_contradictions) is not set or any(type(sid) is not str or not sid for sid in ids)
                or len(set(ids)) != len(ids) or any(type(sid) is not str for sid in process_contradictions)
                or not process_contradictions <= set(ids)
                or any(sample.sample_id in process_contradictions and (
                    sample.result is None or sample.result.status != "CORRECT") for sample in samples)):
            raise ValueError("rollout: invalid process contradiction candidate IDs or statuses")
    candidates = sorted((s for s in samples if s.result is not None and (
                        s.result.status in RETRYABLE or process_contradictions is not None
                        and s.sample_id in process_contradictions)),
                        key=lambda s: hashlib.sha256(f"{key}:{s.sample_id}".encode()).digest())
    distinct, repeated, seen = [], [], set()
    for sample in candidates:
        assert sample.result is not None
        answer = sample.result.extracted
        if answer not in seen:
            distinct.append(sample)
            seen.add(answer)
        else:
            repeated.append(sample)
    return (distinct + repeated)[:k]


def continuation_ids(sample: Sample, feedback: str, tokenizer: Any) -> list[int]:
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if type(end) is not int or end < 0 or end == getattr(tokenizer, "unk_token_id", None):
        raise ValueError("rollout: tokenizer lacks the Qwen conversation boundary")
    closure = [] if sample.completion_token_ids and sample.completion_token_ids[-1] == end else [end]
    suffix = tokenizer.encode("\n<|im_start|>user\n" + feedback + "<|im_end|>\n<|im_start|>assistant\n",
                              add_special_tokens=False)
    return sample.prompt_token_ids + sample.completion_token_ids + closure + suffix


def malformed_retry_receipt(sample: Sample, response_contract: str | None) -> dict:
    """Recheck saved raw syntax; an absent native grade alone never permits a retry."""
    from .engines import _completion
    from .repair_contract import GRAMMAR_ERRORS, VERSION, RepairContractError, parse_retry_sections

    if response_contract != VERSION or sample.result is not None or sample.reward != 0:
        raise ValueError("rollout: malformed retry requires its original ungraded repair response")
    _completion(sample.completion_token_ids, sample.sampler_logprobs, sample.finish_reason, sample.text)
    try:
        parse_retry_sections(sample.text, contract_version=response_contract)
    except RepairContractError as exc:
        if exc.code not in GRAMMAR_ERRORS:
            raise
        return {"sample_id": sample.sample_id, "code": exc.code,
                "raw_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest()}
    raise ValueError("rollout: valid repair response is missing its actual native grade")


def _bound_malformed_retry(sample: Sample, response_contracts: dict, invalid_responses: list[dict]) -> dict:
    receipt = malformed_retry_receipt(sample, response_contracts.get(sample.sample_id))
    matches = [row for row in invalid_responses if row.get("sample_id") == sample.sample_id]
    if matches != [receipt]:
        raise ValueError("rollout: malformed retry raw hash or grammar error binding differs")
    return receipt


def select_repair_parents(samples: list[Sample], k: int, key: str, *, native_audits: dict,
                          invalid_responses: list[dict], response_contracts: dict) -> tuple[list[Sample], dict]:
    """Keep native selection unchanged; retry completed malformed raw only if that pool is empty."""
    if type(k) is not int or k < 0:
        raise ValueError("rollout: nonnegative branch count required")
    malformed = [_bound_malformed_retry(sample, response_contracts, invalid_responses)
                 for sample in samples if sample.result is None]
    contradicted = {sample.sample_id for sample in samples if sample.result is not None
        and sample.result.status == "CORRECT" and any(row["kind"] in ("OBSERVATION", "TRACE")
            and row["identity"] is not None and row["refs"] for row in native_audits[sample.sample_id]["issues"])}
    ordered = select_failures(samples, len(samples), key, process_contradictions=contradicted)
    eligible = [sample for sample in samples if sample.result is not None
                and (sample.result.status in RETRYABLE or sample.sample_id in contradicted)]
    if not ordered:
        eligible = [sample for sample in samples if sample.result is None]
        ordered = sorted(eligible, key=lambda sample: hashlib.sha256(f"{key}:{sample.sample_id}".encode()).digest())
        k = min(k, 1)
    chosen = ordered[:k]
    return chosen, {"key": key, "frontier_ids": [sample.sample_id for sample in samples],
        "selection_policy": REPAIR_SELECTION_POLICY, "malformed_candidates": malformed,
        "candidate_original_order": [sample.sample_id for sample in eligible],
        "candidate_hash_order": [sample.sample_id for sample in sorted(eligible,
            key=lambda sample: hashlib.sha256(f"{key}:{sample.sample_id}".encode()).digest())],
        "candidate_final_order": [sample.sample_id for sample in ordered],
        "process_contradictions": [sample.sample_id for sample in samples if sample.sample_id in contradicted],
        "selected_ids": [sample.sample_id for sample in chosen]}


def repair_parent_feedback(parent: Sample, actor: ActorItem, prompts: dict, inventory: Any, *,
                           native_audits: dict, invalid_responses: list[dict], response_contracts: dict) -> str:
    if parent.result is None:
        invalid = _bound_malformed_retry(parent, response_contracts, invalid_responses)
        return prompt_builders.repair_retry_text(None, None, actor, prompts, inventory,
            malformed_code=invalid["code"], malformed_finish_reason=parent.finish_reason)
    return prompt_builders.repair_retry_text(parent.result.status, parent.result.extracted, actor, prompts,
        inventory, process_contradicted=parent.result.status == "CORRECT"
        and bool(native_audits[parent.sample_id]["issues"]))


def _sample_request(request: RolloutRequest, engine: Any, policy_version: int,
                    context_limit: int, *,
                    on_completion: Callable[[str, int, int, Completion], None] | None = None,
                    validate_request: Callable[[], None] | None = None,
                    validate_prepared: Callable[[Any], None] | None = None) -> RolloutResult:
    prepared = engine.prepare(request)
    if validate_request is not None:
        validate_request()
    if validate_prepared is not None:
        validate_prepared(prepared)
    if len(prepared.token_ids)+request.max_new_tokens > context_limit:
        raise ValueError("CONTEXT_BLOCKED: prompt plus completion reservation")
    if engine.policy_version != policy_version:
        raise RuntimeError("rollout: policy changed during a tree")
    outputs = (engine.sample([request]) if on_completion is None else
               engine.sample([request], on_completion=on_completion))
    if validate_request is not None:
        validate_request()
    if (len(outputs) != 1 or outputs[0].request_id != request.request_id
            or len(outputs[0].completions) != request.n
            or outputs[0].prompt_token_ids != prepared.token_ids
            or request.prompt_token_ids is not None and outputs[0].prompt_token_ids != request.prompt_token_ids
            or engine.policy_version != policy_version):
        raise RuntimeError("rollout: sampler altered the actual state, version or group size")
    return outputs[0]


def _build_repair_tree(actor: ActorItem, images: list, engine: Any, verifier: Any, prompts: dict,
                       cfg: RolloutConfig, policy_version: int, route_rule: str, *,
                       on_group, should_stop, on_raw) -> ProblemTree:
    """Collect bounded raw retries; full-history admission/loss wiring is deliberately pending."""
    from . import fresh_observation as fresh, notebook_synthesis as synthesis
    from .engines import _completion
    from .notebook_layout_probe import _image_identity
    from .repair_contract import (GRAMMAR_ERRORS, ORIGIN_VERSION, VERSION, RepairContractError,
                                  parse_retry_sections, public_origin_inventory)
    from PIL.Image import Image

    cfg = replace(cfg, n=dict(cfg.n), branch=dict(cfg.branch), max_new_tokens=dict(cfg.max_new_tokens))
    cfg.validate()
    actor = from_json(ActorItem, to_json(actor))
    prompts = copy.deepcopy(prompt_builders.for_actor(actor, prompts))
    images = list(images)
    # A bound answer-verified (slotless) source may also be a plain-mode actor; every other source stays obs.
    if (not callable(on_raw) or actor.mode != "obs" and "_slotless_source" not in prompts
            or "solution_trace_contract" not in prompts
            or len(images) != len(actor.image_paths) or engine.policy_version != policy_version):
        raise ValueError("rollout: native repair collector requires raw persistence and matching inputs")
    # Opt-in eval-aligned T1 (answer-verified actors only): exact P0 prompt, its own T1 cap, V8 grading.
    turn1 = prompts.get("_turn1_contract")
    # Opt-in retry rule: bound only with the eval-aligned T1 and applied to the same actors' retries.
    retry_rule = prompts.get("repair_retry_out_of_choice") if turn1 is not None else None
    groups: list[Group] = []
    causes: dict[str, Cause] = {}
    audits: dict[str, dict] = {}
    contracts: dict[str, str | None] = {}
    skipped: list[dict] = []
    record: dict[str, Any] = {"version": REPAIR_COLLECTOR_VERSION, "retry_contract": VERSION,
        "retry_prompt_contract_sha256": _hash(prompt_builders.repair_retry_prompt_contract()),
        "calls": [], "fresh_observation": None, "fresh_observation_images": None, "fresh_observation_calls": 0,
        "fresh_observation_events": [], "selections": [], "response_contracts": contracts, "native_audits": audits,
        "invalid_responses": [], "full_history_admission": "UNRESOLVED",
        "loss_integration_ready": False, "binary_route_provisional": True}

    def image_values(values: list) -> list:
        return [(id(image), image.mode, image.size, hashlib.sha256(image.tobytes()).hexdigest())
                if isinstance(image, Image) else (id(image),) for image in values]

    original_images = image_values(images)

    def select(frontier: list[Sample], turn: int) -> list[Sample]:
        key = f"{actor.problem_id}:{policy_version}" + (f":{turn}" if turn > 1 else "")
        chosen, selection = select_repair_parents(frontier, cfg.branch.get(turn, 0), key,
            native_audits=audits, invalid_responses=record["invalid_responses"], response_contracts=contracts)
        record["selections"].append({"turn": turn, **selection})
        return chosen

    def generate(turn: int, parent: Sample | None) -> Group | None:
        if image_values(images) != original_images:
            raise ValueError("rollout: repair input image values changed")
        kinds: dict[int, GroupKind] = {1: "T1", 2: "T2", 3: "T3"}
        kind, branch = kinds[turn], "root" if parent is None else parent.sample_id
        if should_stop is not None and should_stop():
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "BUDGET_STOP"})
            return None
        if parent is not None and record["fresh_observation"] is None:
            schema = prompts["observation_instruction"][actor.family]

            try:
                observation_input = engine.prepare(fresh.request_for(actor, schema, images, policy_version, cfg.master_seed))
                if len(observation_input.token_ids) + fresh.MAX_NEW_TOKENS > cfg.context_limit:
                    raise ValueError("CONTEXT_BLOCKED: fresh observation reservation")
            except ValueError as exc:
                if not str(exc).startswith("CONTEXT_BLOCKED:"):
                    raise
                skipped.append({"kind": kind, "parent_sample_id": branch, "status": "CONTEXT_BLOCKED"})
                return None
            observation_images = json.loads(json.dumps(_image_identity(actor, observation_input)))

            def observation_event(event: dict) -> None:
                before = _hash(event)
                payload = copy.deepcopy({"version": record["version"], "purpose": "RETRY_FRESH_OBSERVATION",
                    "problem_id": actor.problem_id, "policy_version": policy_version, "event": event})
                payload_sha = _hash(payload)
                on_raw(payload)
                if _hash(payload) != payload_sha:
                    raise ValueError("rollout: fresh observation persistence payload changed")
                if _hash(event) != before or image_values(images) != original_images:
                    raise ValueError("rollout: fresh observation callback changed original input")
                record["fresh_observation_events"].append(copy.deepcopy(event))

            record["fresh_observation"] = fresh.generate(actor, schema, images, engine,
                master_seed=cfg.master_seed, context_limit=cfg.context_limit, on_event=observation_event)
            record["fresh_observation_calls"] = 1
            record["fresh_observation_images"] = observation_images
            if should_stop is not None and should_stop():
                skipped.append({"kind": kind, "parent_sample_id": branch, "status": "BUDGET_STOP"})
                return None
        inventory, feedback, ids = None, None, None
        contract = None if turn == 1 else VERSION
        if parent is not None:
            history = [sample for group in groups for sample in group.samples]
            inventory = public_origin_inventory(history, contract_version=ORIGIN_VERSION,
                response_contracts=contracts, parent_sample_id=parent.sample_id)
            feedback = repair_parent_feedback(parent, actor, prompts, inventory, native_audits=audits,
                invalid_responses=record["invalid_responses"], response_contracts=contracts)
            feedback += "\n" + fresh.quoted_retry_context(record["fresh_observation"])
            ids = continuation_ids(parent, feedback, engine.tokenizer)
        cap = turn1["max_new_tokens"] if turn == 1 and turn1 is not None else cfg.max_new_tokens[turn]
        n = cfg.n[turn]
        if ids is not None and len(ids) + cap > cfg.context_limit:
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "CONTEXT_BLOCKED",
                            "input_tokens": len(ids), "reserved_output": cap})
            return None
        group_id = f"{actor.problem_id}:v{policy_version}:{kind}:{_hash(branch)[:16]}"
        request = RolloutRequest(group_id, ids, None if ids is not None else [prompt_builders.user_message(
            prompt_builders.turn1_text(actor, prompts), len(images))], images, n, cap,
            seeds(actor.problem_id, policy_version, turn, kind + ":" + branch, n, master_seed=cfg.master_seed))
        bound_seeds = tuple(request.seeds)

        def request_values() -> str:
            try:
                return json.dumps({"request_id": request.request_id, "prompt_token_ids": request.prompt_token_ids,
                    "messages": request.messages, "images": image_values(request.images), "n": request.n,
                    "max_new_tokens": request.max_new_tokens, "seeds": request.seeds,
                    "policy_version": engine.policy_version},
                    sort_keys=True, separators=(",", ":"), allow_nan=False)
            except (AttributeError, TypeError, ValueError):
                raise ValueError("rollout: repair request value contract changed") from None

        bound_request = request_values()

        def check_request() -> None:
            if (request_values() != bound_request or engine.policy_version != policy_version
                    or image_values(images) != original_images):
                raise ValueError("rollout: repair request state or policy changed")

        try:
            prepared = engine.prepare(request)
            check_request()
            if len(prepared.token_ids) + cap > cfg.context_limit:
                raise ValueError("CONTEXT_BLOCKED: full repair reservation")
        except ValueError as exc:
            if not str(exc).startswith("CONTEXT_BLOCKED:"):
                raise
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "CONTEXT_BLOCKED"})
            return None
        call = {"group_id": group_id, "turn": turn, "parent_sample_id": parent.sample_id if parent else None,
            "response_contract": contract, "prompt_token_ids": list(prepared.token_ids),
            "retry_prompt_contract_sha256": record["retry_prompt_contract_sha256"],
            "prompt_sha256": _hash(prepared.token_ids), "messages": copy.deepcopy(request.messages),
            "images": json.loads(json.dumps(_image_identity(actor, prepared))),
            "n": n, "max_new_tokens": cap, "seeds": list(bound_seeds),
            "feedback": feedback, "feedback_sha256": _hash(feedback),
            "ancestor_inventory": json.loads(json.dumps(asdict(inventory), allow_nan=False)) if inventory is not None else None,
            "ancestor_inventory_sha256": inventory.sha256 if inventory is not None else None,
            "completions": []}
        bound_prepared = (call["prompt_sha256"], _hash(call["images"]))

        def check_prepared(current: Any) -> None:
            check_request()
            if (_hash(current.token_ids), _hash(_image_identity(actor, current))) != bound_prepared:
                raise ValueError("rollout: prepared repair token or image values changed")

        captured: list[dict] = []

        def completed(request_id: str, index: int, seed: int, completion: Completion) -> None:
            if (request_id != group_id or type(index) is not int or index != len(captured)
                    or index >= n or type(seed) is not int or not 0 <= seed < 2**31
                    or type(completion) is not Completion or type(completion.text) is not str):
                raise ValueError("rollout: completed repair response identity changed")
            _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
            if (len(completion.token_ids) > cap or engine.tokenizer.decode(completion.token_ids,
                    skip_special_tokens=True, clean_up_tokenization_spaces=False) != completion.text):
                raise ValueError("rollout: completed repair budget or decode changed")
            captured.append(copy.deepcopy(asdict(completion)))
            on_raw(copy.deepcopy({"version": record["version"], "purpose": "NATIVE_SOLUTION_COMPLETION",
                "problem_id": actor.problem_id, "policy_version": policy_version,
                **call, "index": index, "seed": seed, "completion": captured[-1]}))
            check_request()
            if seed != bound_seeds[index]:
                raise ValueError("rollout: completed repair response seed changed")

        output = _sample_request(request, engine, policy_version, cfg.context_limit,
                                 on_completion=completed, validate_request=check_request,
                                 validate_prepared=check_prepared)
        if (captured != [asdict(completion) for completion in output.completions]
                or output.prompt_token_ids != call["prompt_token_ids"]):
            raise ValueError("rollout: streamed and final repair state or raw responses differ")
        call["completions"] = copy.deepcopy(captured)
        record["calls"].append(call)
        samples: list[Sample] = []
        for index, payload in enumerate(captured):
            sample = Sample(f"{group_id}:{index}", actor.problem_id, group_id, kind, turn,
                parent.sample_id if parent else None, list(output.prompt_token_ids),
                payload["token_ids"], payload["logprobs"], payload["finish_reason"], payload["text"],
                bound_seeds[index], policy_version)
            contracts[sample.sample_id] = contract
            invalid = None
            if contract is not None:
                try:
                    parse_retry_sections(sample.text, contract_version=VERSION)
                except RepairContractError as exc:
                    if exc.code not in GRAMMAR_ERRORS:
                        raise
                    invalid = exc.code
            if invalid is not None:
                record["invalid_responses"].append({"sample_id": sample.sample_id, "code": invalid,
                    "raw_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest()})
            else:
                audit = synthesis.audit_native_sample(sample, actor, verifier, response_contract=contract,
                    **synthesis.grade_options(turn, None if turn1 is None else turn1["version"], retry_rule))
                sample.result = from_json(type_result, json.dumps(audit["result"], allow_nan=False))
                if sample.result.status not in BLOCKING:
                    sample.reward = answer_reward(sample.result)
                cause = verifier.cause(actor.problem_id, sample.result)
                if cause is not None:
                    causes[sample.sample_id] = cause
                audits[sample.sample_id] = audit
                record.setdefault("sample_bindings", []).append({"sample_id": sample.sample_id,
                    "raw_sha256": audit["raw_sha256"], "native_audit_sha256": _hash(audit),
                    "process_assessment": "ANSWER_CORRECT_PROCESS_CONTRADICTED" if sample.result.status == "CORRECT"
                    and any(row["identity"] is not None and row["refs"] for row in audit["issues"]) else None})
            to_json(sample)
            samples.append(sample)
        bad = sorted({sample.result.status for sample in samples if sample.result is not None
                      and sample.result.status in BLOCKING})
        group = Group(group_id, kind, actor.problem_id, _hash(output.prompt_token_ids), samples,
                      not bad, ",".join(bad) if bad else None)
        check_request()
        to_json(group)
        groups.append(group)
        if on_group is not None:
            on_group(from_json(Group, to_json(group)))
            check_request()
        return group

    from .schema import VerifierResult as type_result

    first = generate(1, None)
    frontier = select(first.samples, 1) if first is not None and first.usable else []
    for turn in (2, 3):
        next_samples: list[Sample] = []
        for parent in frontier:
            group = generate(turn, parent)
            if group is not None and group.usable:
                next_samples.extend(group.samples)
        frontier = select(next_samples, turn) if turn == 2 else []
    results = [sample.result for group in groups for sample in group.samples if sample.result is not None]
    route, evidence = verifier.route(actor.problem_id, results, route_rule)
    record["missing_result_ids"] = [sample.sample_id for group in groups for sample in group.samples if sample.result is None]
    record["quarantined_groups"] = [group.group_id for group in groups if group.reason == "INVALID_RESPONSE_CONTRACT"]
    record["source_contracts_sha256"] = _hash(contracts)
    notebook = {"repair_collector": record, "routing": evidence, "skipped_branches": skipped,
                "model_notes": [], "verified_internal_module_responsibility": False}
    tree = ProblemTree(actor.problem_id, policy_version, groups, route, causes, notebook)
    to_json(tree)
    return tree


def build_tree(actor: ActorItem, images: list, engine: Any, verifier: Any, prompts: dict,
               cfg: RolloutConfig, policy_version: int, route_rule: str, *,
               on_group: Callable[[Group], None] | None = None,
               should_stop: Callable[[], bool] | None = None,
               on_raw: Callable[[dict], None] | None = None) -> ProblemTree:
    """Solution tree first; DIAG is deferred to a class-balanced batch operation."""
    cfg.validate()
    if cfg.retry_contract is not None:
        return _build_repair_tree(actor, images, engine, verifier, prompts, cfg, policy_version,
                                  route_rule, on_group=on_group, should_stop=should_stop, on_raw=on_raw)
    to_json(actor)
    prompts = prompt_builders.for_actor(actor, prompts)
    if len(images) != len(actor.image_paths) or engine.policy_version != policy_version:
        raise ValueError("rollout: image coverage or policy version mismatch")
    groups: list[Group] = []
    causes: dict[str, Cause] = {}
    skipped: list[dict] = []
    by_id: dict[str, Sample] = {}
    candidate = None

    def generate(turn: int, parent: Sample | None, ids: list[int] | None) -> Group | None:
        kinds: dict[int, GroupKind] = {1: "T1", 2: "T2", 3: "T3"}
        kind = kinds[turn]
        branch = "root" if parent is None else parent.sample_id
        if should_stop is not None and should_stop():
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "BUDGET_STOP"})
            return None
        group_id = f"{actor.problem_id}:v{policy_version}:{kind}:{_hash(branch)[:16]}"
        cap = cfg.max_new_tokens[turn]
        n = cfg.n[turn]
        if ids is not None and len(ids)+cap > cfg.context_limit:
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "CONTEXT_BLOCKED",
                            "input_tokens": len(ids), "reserved_output": cap})
            return None
        request = RolloutRequest(group_id, ids, None if ids is not None else [prompt_builders.user_message(
            prompt_builders.turn1_text(actor, prompts), len(images))], images, n, cap,
            seeds(actor.problem_id, policy_version, turn, kind+":"+branch, n, master_seed=cfg.master_seed))
        try:
            output = _sample_request(request, engine, policy_version, cfg.context_limit)
        except ValueError as exc:
            if not str(exc).startswith("CONTEXT_BLOCKED:"):
                raise
            skipped.append({"kind": kind, "parent_sample_id": branch, "status": "CONTEXT_BLOCKED"})
            return None
        samples = []
        for i, completion in enumerate(output.completions):
            sample = Sample(group_id+f":{i}", actor.problem_id, group_id, kind, turn,
                parent.sample_id if parent is not None else None, list(output.prompt_token_ids),
                completion.token_ids, completion.logprobs, completion.finish_reason,
                completion.text, request.seeds[i], policy_version)
            sample.result = verifier.grade(actor.problem_id, sample.text, sample.finish_reason, actor.mode)
            if sample.result.status not in BLOCKING:
                sample.reward = answer_reward(sample.result, visual_lambda=cfg.visual_lambda)
            cause = verifier.cause(actor.problem_id, sample.result)
            if cause is not None:
                causes[sample.sample_id] = cause
            to_json(sample)
            samples.append(sample)
            by_id[sample.sample_id] = sample
        bad = sorted({s.result.status for s in samples if s.result is not None and s.result.status in BLOCKING})
        group = Group(group_id, kind, actor.problem_id, _hash(output.prompt_token_ids), samples, not bad,
                      ",".join(bad) if bad else None)
        to_json(group)
        groups.append(group)
        if on_group is not None:
            on_group(group)
        return group

    first = generate(1, None, None)
    frontier = select_failures(first.samples, cfg.branch[1], f"{actor.problem_id}:{policy_version}") if (
        first is not None and first.usable) else []
    for turn in range(2, cfg.max_attempts+1):
        next_failures = []
        for parent in frontier:
            assert parent.result is not None
            feedback = prompt_builders.retry_text(parent.result.status, parent.result.extracted,
                                                  actor.mode, actor.family, prompts)
            ids = continuation_ids(parent, feedback, engine.tokenizer)
            group = generate(turn, parent, ids)
            if group is not None and group.usable:
                next_failures.extend(group.samples)
        frontier = select_failures(next_failures, cfg.branch.get(turn, 0),
                                   f"{actor.problem_id}:{policy_version}:{turn}")
    if cfg.diag:
        candidates = sorted((s for g in groups if g.usable for s in g.samples
                             if s.turn >= 2 and s.result is not None and s.result.status == "CORRECT"),
                            key=lambda s: _hash(s.sample_id))
        for success in candidates:
            earliest = success
            while earliest.parent_sample_id is not None:
                earliest = by_id[earliest.parent_sample_id]
            target = verifier.diagnosis_target(actor.problem_id, earliest.result)
            if target is not None:
                candidate = {"parent_sample_id": success.sample_id, "target": json.loads(to_json(target))}
                break
    results = [s.result for g in groups for s in g.samples if s.result is not None]
    route, evidence = verifier.route(actor.problem_id, results, route_rule)
    notebook = {"routing": evidence, "skipped_branches": skipped, "model_notes": [],
        "diagnosis_candidate": candidate, "diagnosis_target": None, "diagnosis_batch_closed": False,
        "attempts": [{"sample_id": s.sample_id, "status": s.result.status,
                      "cause": causes.get(s.sample_id), "slots": json.loads(to_json(s.result.slots))
                      if s.result.slots else None} for g in groups for s in g.samples if s.result is not None],
        "verified_internal_module_responsibility": False}
    tree = ProblemTree(actor.problem_id, policy_version, groups, route, causes, notebook)
    if cfg.final_notebook:
        append_final_notebook(tree, images, engine, prompts, cfg, should_stop=should_stop)
    to_json(tree)
    return tree


def append_final_notebook(tree: ProblemTree, images: list, engine: Any, prompts: dict,
                          cfg: RolloutConfig, *, should_stop: Callable[[], bool] | None = None) -> dict:
    """Preserve one post-recovery model note, independently of balanced DIAG training.

    This audit action is never a loss group or a source of routing authority.
    The original protocol stays opt-in: enabling it changes the measured call budget.
    """
    cfg.validate()
    to_json(tree)
    if engine.policy_version != tree.policy_version or "final_notebook" in tree.notebook:
        raise ValueError("rollout: one fresh same-policy final notebook required")
    record = {"status": "NO_RECOVERED_PATH", "used_for_loss": False, "used_for_routing": False,
              "lesson_semantics_verified": False, "module_causality_certified": False}
    tree.notebook["final_notebook"] = record
    candidates = sorted((s for g in tree.groups if g.usable and g.kind != "DIAG" for s in g.samples
                         if s.turn >= 2 and s.result is not None and s.result.status == "CORRECT"),
                        key=lambda sample: (sample.turn, _hash(sample.sample_id)))
    if not candidates:
        return record
    parent = candidates[0]
    record["parent_sample_id"] = parent.sample_id
    if should_stop is not None and should_stop():
        record["status"] = "BUDGET_STOP"
        return record
    ids = continuation_ids(parent, prompt_builders.diagnose_text(parent.turn, prompts), engine.tokenizer)
    if len(ids)+cfg.diag_max_new_tokens > cfg.context_limit:
        record["status"] = "CONTEXT_BLOCKED"
        return record
    request = RolloutRequest(f"{tree.problem_id}:v{tree.policy_version}:FINAL_NOTE:{_hash(parent.sample_id)[:16]}",
        ids, None, images, 1, cfg.diag_max_new_tokens,
        seeds(tree.problem_id, tree.policy_version, parent.turn, "FINAL_NOTE:"+parent.sample_id, 1,
              master_seed=cfg.master_seed))
    output = _sample_request(request, engine, tree.policy_version, cfg.context_limit)
    completion = output.completions[0]
    status = ("TRUNCATED" if completion.finish_reason != "stop" else
              "COMPLETED" if completion.text.strip() else "EMPTY")
    record.update(status=status,
                  request_id=request.request_id, prompt_token_ids=output.prompt_token_ids,
                  completion_token_ids=completion.token_ids, sampler_logprobs=completion.logprobs,
                  text=completion.text, finish_reason=completion.finish_reason, seed=request.seeds[0],
                  policy_version=tree.policy_version)
    to_json(tree)
    return record


def append_diagnoses(trees: list[ProblemTree], images: dict[str, list], engine: Any,
                     verifier: Any, prompts: dict, cfg: RolloutConfig, *,
                     on_group: Callable[[Group], None] | None = None,
                     should_stop: Callable[[], bool] | None = None) -> dict:
    """Choose <=50% per verified class BEFORE generating; never drop after seeing rewards."""
    cfg.validate()
    if (not cfg.diag or not trees or len({t.problem_id for t in trees}) != len(trees)
            or set(images) != {t.problem_id for t in trees}
            or any(t.policy_version != engine.policy_version for t in trees)
            or any(t.notebook.get("diagnosis_batch_closed") for t in trees)):
        raise ValueError("rollout: one fresh same-policy diagnosis batch required")
    eligible: dict[str, list] = {c: [] for c in ("VISUAL", "NON_VISUAL", "FINALIZATION")}
    for tree in trees:
        to_json(tree)
        candidate = tree.notebook.get("diagnosis_candidate")
        if candidate is None:
            continue
        target = from_json(DiagTarget, json.dumps(candidate["target"]))
        parents = [s for g in tree.groups if g.usable and g.kind != "DIAG" for s in g.samples
                   if s.sample_id == candidate["parent_sample_id"] and s.turn >= 2
                   and s.result is not None and s.result.status == "CORRECT"]
        if len(parents) != 1:
            raise ValueError("rollout: diagnosis needs an actual recovered path")
        parent = parents[0]
        ids = continuation_ids(parent, prompt_builders.diagnose_text(parent.turn, prompts), engine.tokenizer)
        gid = f"{tree.problem_id}:v{tree.policy_version}:DIAG:{_hash(parent.sample_id)[:16]}"
        request = RolloutRequest(gid, ids, None, images[tree.problem_id], cfg.n_diag,
            cfg.diag_max_new_tokens, seeds(tree.problem_id, tree.policy_version, parent.turn,
                                          "DIAG:"+parent.sample_id, cfg.n_diag, master_seed=cfg.master_seed))
        if len(ids)+request.max_new_tokens > cfg.context_limit:
            tree.notebook["skipped_branches"].append({"kind": "DIAG", "status": "CONTEXT_BLOCKED"})
            continue
        eligible[target.cause3].append((tree, target, parent, request))
    for entries in eligible.values():
        entries.sort(key=lambda entry: _hash(entry[0].problem_id))
    excluded = []
    while sum(map(len, eligible.values())):
        largest = max(eligible, key=lambda key: (len(eligible[key]), key))
        if len(eligible[largest])*2 <= sum(map(len, eligible.values())):
            break
        excluded.append(eligible[largest].pop())
    for tree, _, _, _ in excluded:
        tree.notebook["skipped_branches"].append({"kind": "DIAG", "status": "CLASS_CAP"})
    admitted = [entry for entries in eligible.values() for entry in entries]
    report: dict[str, Any] = {"class_counts": {key: len(values) for key, values in eligible.items()},
              "excluded_by_cap": len(excluded), "generated_groups": 0}
    # Mark before generation so an interrupted batch cannot silently generate duplicates.
    for tree in trees:
        tree.notebook["diagnosis_batch_closed"] = True
    for tree, target, parent, request in sorted(admitted, key=lambda entry: _hash(entry[0].problem_id)):
        if should_stop is not None and should_stop():
            report["budget_stopped"] = True
            break
        output = _sample_request(request, engine, tree.policy_version, cfg.context_limit)
        samples = []
        for i, completion in enumerate(output.completions):
            reward = verifier.grade_diagnosis(tree.problem_id, completion.text, target) if (
                completion.finish_reason == "stop") else 0.0
            sample = Sample(request.request_id+f":{i}", tree.problem_id, request.request_id, "DIAG",
                parent.turn, parent.sample_id, output.prompt_token_ids, completion.token_ids,
                completion.logprobs, completion.finish_reason, completion.text, request.seeds[i],
                tree.policy_version, None, reward, reward)
            samples.append(sample)
            tree.notebook["model_notes"].append({"sample_id": sample.sample_id, "text": sample.text,
                "reward": reward, "lesson_semantics_verified": False})
        tree.groups.append(Group(request.request_id, "DIAG", tree.problem_id,
                                 _hash(output.prompt_token_ids), samples, True, None))
        if on_group is not None:
            on_group(tree.groups[-1])
        tree.notebook["diagnosis_target"] = json.loads(to_json(target))
        report["generated_groups"] += 1
        to_json(tree)
    return report
