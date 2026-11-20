"""Explicit 8GB local experiment; no change to the historical BF16 protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .answers import RETRY_OUT_OF_CHOICE, TURN1_EVAL_CONTRACT
from .checkpoint import digest
from .episode_notebook import (ANSWER_ROUTE_RULE, FEEDBACK_POLICY, INPUT_POLICY, MAX_NOTEBOOK_ROUNDS, REINSPECTION_INSTRUCTION,
                              PROMPT as NOTEBOOK_PROMPT, VERSION as NOTEBOOK_REWARD_VERSION, STRICT_ROUTE_RULE)
from .prompts import SLOTLESS_REREAD_INSTRUCTION, load_prompts
from .local_data import ACTOR_MODE_POLICY
from .quantized_policy import quantization_contract
from .compact_note import VERSION as COMPACT_NOTE_CONTRACT
from .repair_contract import FINAL_VERSION as DEFAULT_NOTE_CONTRACT, FINALIZATION_VERSION, FINALIZATION_WEIGHT
from .schema import load_protocol
from .verifier import VERSION as VERIFIER_VERSION
from . import fresh_observation

PROFILE = "local-nf4-notebook-rl-v1"
PROTOCOL_ID = "local-nf4-full-history-notebook-v13"
ATOMIC_PROTOCOL_ID = "local-nf4-verified-note-atoms-v14"
COMPONENT_PROTOCOL_ID = "local-nf4-verified-note-components-v15"
FACT_CREDIT_PROTOCOL_ID = "local-nf4-verified-note-fact-credit-v16"
FACT_CREDIT_24K_PROTOCOL_ID = "local-nf4-verified-note-fact-credit-24k-v19"
SYNTHESIS_PROTOCOL_ID = "local-nf4-whole-history-bounded-synthesis-v31"
SYNTHESIS_24K_PROTOCOL_ID = "local-nf4-whole-history-bounded-synthesis-24k-v31"
SYNTHESIS_CREDIT_PROTOCOL_ID = "local-nf4-whole-history-bounded-synthesis-credit-24k-v31"
PARTIAL_NOTEBOOK_PROTOCOL_ID = "local-nf4-partial-notebook-synthesis-credit-24k-v1"
REFERENCED_NOTEBOOK_PROTOCOL_ID = "local-nf4-referenced-notebook-synthesis-credit-24k-v1"
PAIRED_NOTEBOOK_PROTOCOL_ID = "local-nf4-paired-notebook-synthesis-credit-24k-v1"
PAIR_ACTION_NOTEBOOK_PROTOCOL_ID = "local-nf4-pair-action-notebook-synthesis-credit-24k-v2"
PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID = "local-nf4-pair-draft-synthesis-credit-24k-v2"
WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID = "local-nf4-whole-reference-synthesis-credit-24k-v1"
WHOLE_REFERENCE_PAGES_PROTOCOL_ID = "local-nf4-whole-reference-pages-synthesis-credit-24k-v2"
ASSESSMENT_PROTOCOL_ID = "local-nf4-whole-reference-assessment-credit-24k-v3"
REPAIR_PROTOCOL_ID = "local-nf4-verified-repair-notebook-v1"
REPAIR_OBJECTIVE_VERSION = "verified-native-repair-token-credit-v1"
NOTE_CONTRACT_CHOICES = (DEFAULT_NOTE_CONTRACT, COMPACT_NOTE_CONTRACT)
TURN1_CONTRACT_CHOICES = (TURN1_EVAL_CONTRACT,)
RETRY_OUT_OF_CHOICE_CHOICES = (RETRY_OUT_OF_CHOICE,)
SOURCE_OBSERVATION_INSTRUCTION = {
    "gen-graph-v1": (
        "Use the slot form matching this question. Put one assignment on each line. "
        "For a directed graph, list EVERY arrow as edge[<source>,<target>] = <weight or 1>; "
        "use 1 for an unweighted arrow. Do not replace assignments with a bullet list of arrows. "
        "For a DFA, list every trans[<state>,<symbol>] = <target state>, plus "
        "start = <state> and accepting = <comma-separated states>. "
        "For a binary tree, list every child[<parent>,<L or R>] = <child>, plus root = <node>. "
        "Use only the applicable form, copying labels and direction from the image."
    ),
    "gen-graph-pair-v1": (
        "Image 1 gives directed topology; image 2 gives weights matched by printed edge IDs. "
        "For every edge ID, write dir[<edge ID>] = <source>-><target> and "
        "weight[<edge ID>] = <integer>. Put each assignment on its own line, using exactly "
        "these dir and weight slot names. Preserve arrow direction."
    ),
}
HERE = Path(__file__).resolve().parent
AUTHORIZATION = {
    "reference": "active_thread_goal_20261021",
    "quote": "로컬에서 진행하는 나머지 검증과 테스트는 전부 승인할테니 계속 실험 및 검증과 테스트를 하면서 파인튜닝 파이프라인 완성과 오답노트형 자기개선 학습 구조를 완전히 구현해놔줘.",
    "cloud_allowed": False,
}


def repair_objective_contract(note_contract: str = DEFAULT_NOTE_CONTRACT, *, answer_verified_route: bool = False,
                              finalization_reward: bool = False) -> dict:
    """Versioned objective metadata; execution and gradient evidence remain separate."""
    from . import compact_note
    from .repair_notebook import NOTE_CONTRACTS, S_DEFINITIONS, TEXT_S_DEFINITION

    if note_contract not in NOTE_CONTRACTS:
        raise ValueError("local profile: explicit known final-note contract required")
    if (type(answer_verified_route) is not bool or type(finalization_reward) is not bool
            or (answer_verified_route or finalization_reward) and note_contract != compact_note.VERSION):
        raise ValueError("local profile: opt-in route or finalization objective requires the compact final note")
    weights = {"answer": 1.0, "online_repair": .0625, "final_note": .0625,
               "legacy_strict": 0.0, "legacy_fact": 0.0, "legacy_assessment": 0.0}
    if finalization_reward:
        weights["finalization"] = FINALIZATION_WEIGHT
    objective = {"objective_version": REPAIR_OBJECTIVE_VERSION,
        "component_weights": weights,
        "answer_mask": {"T1": "full_original_completion", "T2_T3": "revised_solution_body_only", "DIAG": "zero"},
        "group_mean": "original_n_before_mis_no_recenter", "lesson_pg": False,
        "credit_version": "repair-group-credit-v1", "preparation_version": "repair-tree-credit-v1",
        "loss_version": "separate-answer-online-final-kl-v1", "vjp_version": "native-corrected-signed-and-positive-v1",
        "F_requires": "retained_policy_gradient_not_accuracy_or_complete_note",
        "S_requires": "positive_verified_corrected_tokens_and_nonzero_signed_and_positive_aggregate_vjp"}
    if note_contract == compact_note.VERSION:
        objective.update(final_note_contract=note_contract, final_credit_version=compact_note.CREDIT_VERSION,
            s_definition=S_DEFINITIONS[note_contract],
            S_requires="positive_verified_note_tokens_and_nonzero_signed_and_positive_aggregate_vjp")
    if answer_verified_route:
        objective.update(route_policy=ANSWER_ROUTE_RULE, answer_verified_modules="merger_and_llm_adapter_vision_frozen",
            answer_verified_online_pg=False, answer_verified_note_scope="NO_FINAL_ANSWER_and_WRONG_STEP_only",
            text_s_definition=TEXT_S_DEFINITION,
            S_text_requires="positive_verified_text_note_tokens_and_nonzero_vjp_reported_separately_never_S")
    if finalization_reward:
        objective.update(finalization_contract=FINALIZATION_VERSION,
            finalization_mask={"T1": "full_original_completion", "T2_T3": "full_original_completion", "DIAG": "zero"},
            finalization_reward="one_parseable_final_answer_stop_within_cap_gold_free",
            finalization_arms="notebook_and_answer_only_control",
            F_requires="retained_policy_gradient_with_per_channel_breakdown_not_accuracy_or_complete_note")
    return objective


def repair_flag_keywords(answer_verified_route: bool, finalization_reward: bool,
                         slotless_sources: tuple[str, ...]) -> dict:
    """build_protocol keywords for the opt-in flags only when set, so default calls stay byte-identical."""
    return {**({"answer_verified_route": True} if answer_verified_route else {}),
            **({"slotless_sources": tuple(slotless_sources)} if slotless_sources else {}),
            **({"finalization_reward": True} if finalization_reward else {})}


def add_repair_cli_arguments(parser) -> None:
    """Every opt-in repair-protocol flag a run can bind. local_run, passrate and quantized_cache share
    this and `repair_cli_keywords`, so a cache check and a run cannot bind different protocols."""
    from .scale import add_arguments

    parser.add_argument("--answer-verified-route", action="store_true")
    parser.add_argument("--finalization-reward", action="store_true")
    parser.add_argument("--slotless-source", action="append", default=[])
    parser.add_argument("--turn1-contract", choices=TURN1_CONTRACT_CHOICES)
    parser.add_argument("--turn1-max-new-tokens", type=int)
    parser.add_argument("--retry-out-of-choice", choices=RETRY_OUT_OF_CHOICE_CHOICES)
    add_arguments(parser)


def repair_cli_keywords(args) -> dict[str, Any]:
    """build_protocol keywords (beside repair_notebook/context_limit) from parsed shared flags; only given ones."""
    from .scale import from_args

    return {**({"note_contract": args.note_contract} if args.note_contract != DEFAULT_NOTE_CONTRACT else {}),
            **repair_flag_keywords(args.answer_verified_route, args.finalization_reward, tuple(args.slotless_source)),
            **({"turn1_contract": args.turn1_contract} if args.turn1_contract is not None else {}),
            **({"turn1_max_new_tokens": args.turn1_max_new_tokens} if args.turn1_max_new_tokens is not None else {}),
            **({"retry_out_of_choice": args.retry_out_of_choice} if args.retry_out_of_choice is not None else {}),
            **from_args(args)}


def repair_cli_error(args, *, repair_notebook: bool) -> str | None:
    from .scale import from_args

    if ((args.answer_verified_route or args.finalization_reward)
            and (not repair_notebook or args.note_contract == DEFAULT_NOTE_CONTRACT)
            or args.answer_verified_route is not bool(args.slotless_source)):
        return ("--answer-verified-route/--finalization-reward require --repair-notebook with the compact "
                "--note-contract; --slotless-source is required exactly with --answer-verified-route")
    if from_args(args) and not repair_notebook:
        return "scale options (--rollout-batch-rows, --policy-precision, ...) require --repair-notebook"
    if (args.turn1_contract is not None and not args.answer_verified_route
            or args.turn1_max_new_tokens is not None and args.turn1_contract is None):
        return "--turn1-contract requires --answer-verified-route; --turn1-max-new-tokens requires --turn1-contract"
    if args.retry_out_of_choice is not None and args.turn1_contract is None:
        return "--retry-out-of-choice requires --turn1-contract (with --answer-verified-route)"
    return None


def repair_options(bound: dict) -> dict:
    """Opt-in repair flags bound in a protocol or prompts. Unknown values switch a flag off,
    so every recompute-and-compare against the bound record fails closed."""
    if type(bound) is not dict:
        return {}
    options: dict = {}
    if bound.get("repair_route_policy") == ANSWER_ROUTE_RULE:
        sources = bound.get("repair_slotless_sources")
        options.update(answer_verified_route=True, slotless_sources=tuple(sources) if type(sources) is list
                       and all(type(source) is str for source in sources) else ())
    if bound.get("repair_finalization_contract") == FINALIZATION_VERSION:
        options["finalization_reward"] = True
    return options


def turn1_options(bound: dict) -> dict:
    """build_protocol keywords of a bound eval-aligned T1 contract and its retry rule; kept apart from repair_options, whose
    keys are objective flags. An unknown contract yields none, so a rebuild-and-compare fails closed."""
    contract = bound.get("repair_turn1_contract") if type(bound) is dict else None
    if type(contract) is not dict or contract.get("version") != TURN1_EVAL_CONTRACT:
        return {}
    return {"turn1_contract": TURN1_EVAL_CONTRACT, "turn1_max_new_tokens": contract.get("max_new_tokens"),
            **({"retry_out_of_choice": bound["repair_retry_out_of_choice"]} if "repair_retry_out_of_choice" in bound
               else {})}


def build_protocol(*, verified_note_atoms: bool = False, verified_note_components: bool = False,
                   verified_note_fact_credit: bool = False, verified_note_synthesis: bool = False,
                   verified_note_synthesis_credit: bool = False, verified_note_assessment_credit: bool = False,
                   partial_notebook: bool = False,
                   referenced_notebook: bool = False, paired_notebook: bool = False,
                   pair_action_notebook: bool = False, pair_draft_synthesis: bool = False,
                   whole_reference_notebook: bool = False,
                   whole_reference_pages: bool = False,
                   repair_notebook: bool = False,
                   context_limit: int = 16384, note_contract: str = DEFAULT_NOTE_CONTRACT,
                   answer_verified_route: bool = False, finalization_reward: bool = False,
                   slotless_sources: tuple[str, ...] = (), scale: Any = None,
                   turn1_contract: str | None = None, turn1_max_new_tokens: int | None = None,
                   retry_out_of_choice: str | None = None) -> dict:
    if scale is not None and repair_notebook is not True:
        raise ValueError("local profile: scale options require the repair notebook")
    choices = (verified_note_atoms, verified_note_components, verified_note_fact_credit,
               verified_note_synthesis, verified_note_synthesis_credit)
    legacy = (*choices, verified_note_assessment_credit, partial_notebook, referenced_notebook,
              paired_notebook, pair_action_notebook, pair_draft_synthesis, whole_reference_notebook,
              whole_reference_pages)
    if (type(repair_notebook) is not bool or repair_notebook and (context_limit != 24576
            or any(type(choice) is not bool or choice for choice in legacy))):
        raise ValueError("local profile: repair notebook requires exclusive explicit 24576 context")
    if note_contract != DEFAULT_NOTE_CONTRACT and not repair_notebook:
        raise ValueError("local profile: a non-default final-note contract requires the repair notebook")
    if (type(answer_verified_route) is not bool or type(finalization_reward) is not bool
            or (answer_verified_route or finalization_reward)
            and (repair_notebook is not True or note_contract != COMPACT_NOTE_CONTRACT)):
        raise ValueError("local profile: answer-verified route and finalization reward require the compact repair notebook")
    if (type(slotless_sources) is not tuple or answer_verified_route is not bool(slotless_sources)
            or any(type(source) is not str or not source.strip() or source != source.strip()
                   or source in SOURCE_OBSERVATION_INSTRUCTION or source == "gen-circuit-v1" for source in slotless_sources)
            or len(set(slotless_sources)) != len(slotless_sources)):
        raise ValueError("local profile: the answer-verified route requires explicit unique slotless sources")
    if (turn1_contract is not None and (turn1_contract not in TURN1_CONTRACT_CHOICES or answer_verified_route is not True)
            or turn1_max_new_tokens is not None and turn1_contract is None):
        raise ValueError("local profile: the eval-aligned T1 contract requires the answer-verified route; "
                         "its T1 cap requires the contract")
    if retry_out_of_choice is not None and (retry_out_of_choice not in RETRY_OUT_OF_CHOICE_CHOICES or turn1_contract is None):
        raise ValueError("local profile: the retry out-of-choice rule requires the eval-aligned T1 contract")
    if any(type(choice) is not bool for choice in choices) or sum(choices) > 1:
        raise ValueError("local profile: explicit atomic objective selection required")
    if type(partial_notebook) is not bool or partial_notebook and not verified_note_synthesis_credit:
        raise ValueError("local profile: partial notebook requires explicit synthesis credit")
    if (type(referenced_notebook) is not bool
            or referenced_notebook and (partial_notebook or not verified_note_synthesis_credit)):
        raise ValueError("local profile: referenced notebook requires exclusive explicit synthesis credit")
    if (type(paired_notebook) is not bool
            or paired_notebook and (partial_notebook or referenced_notebook or not verified_note_synthesis_credit)):
        raise ValueError("local profile: paired notebook requires exclusive explicit synthesis credit")
    if (type(pair_action_notebook) is not bool or pair_action_notebook
            and (partial_notebook or referenced_notebook or paired_notebook or not verified_note_synthesis_credit)):
        raise ValueError("local profile: pair-action notebook requires exclusive explicit synthesis credit")
    if (type(pair_draft_synthesis) is not bool or pair_draft_synthesis
            and (partial_notebook or referenced_notebook or paired_notebook or pair_action_notebook
                 or not verified_note_synthesis_credit)):
        raise ValueError("local profile: pair-draft synthesis requires exclusive explicit synthesis credit")
    if (type(whole_reference_notebook) is not bool
            or whole_reference_notebook and not pair_draft_synthesis):
        raise ValueError("local profile: complete reference notebook requires explicit pair-draft synthesis credit")
    if (type(whole_reference_pages) is not bool or whole_reference_pages
            and (whole_reference_notebook or not pair_draft_synthesis)):
        raise ValueError("local profile: reference pages require exclusive explicit pair-draft synthesis credit")
    if (type(verified_note_assessment_credit) is not bool or verified_note_assessment_credit
            and (not verified_note_synthesis_credit or not pair_draft_synthesis or not whole_reference_notebook
                 or whole_reference_pages or partial_notebook or referenced_notebook or paired_notebook
                 or pair_action_notebook or context_limit != 24576)):
        raise ValueError("local profile: assessment credit requires exclusive monolithic whole-reference synthesis credit at 24576")
    if type(context_limit) is not int or context_limit not in (16384, 24576):
        raise ValueError("local profile: context limit must be exactly 16384 or 24576")
    if verified_note_synthesis_credit and context_limit != 24576:
        raise ValueError("local profile: synthesis credit requires explicit 24576 context")
    if context_limit == 24576 and not (verified_note_fact_credit or verified_note_synthesis or verified_note_synthesis_credit or repair_notebook):
        raise ValueError("local profile: 24576 context requires verified note fact credit or synthesis")
    protocol = load_protocol(HERE / "protocol_v1.yaml")
    protocol["parent_protocol_sha256"] = protocol.pop("sha256")
    protocol.update(
        protocol_id=PROTOCOL_ID, profile=PROFILE, status="FROZEN_FOR_LOCAL_MEASUREMENT",
        quantization=quantization_contract(), local_authorization=AUTHORIZATION.copy(),
        context_limit=context_limit,
        verifier_version=VERIFIER_VERSION,
        image={**protocol["image"], "min_pixels": 262144, "max_pixels": 1003520},
        notebook_reward=NOTEBOOK_REWARD_VERSION,
        notebook_prompt_sha256=digest(NOTEBOOK_PROMPT),
        notebook_input_policy=INPUT_POLICY,
        notebook_max_rounds=MAX_NOTEBOOK_ROUNDS,
        notebook_retry_instruction_sha256=digest(REINSPECTION_INSTRUCTION),
        notebook_scope="all_actual_attempts_including_siblings_and_success",
        notebook_route_rule=STRICT_ROUTE_RULE,
        notebook_feedback_policy=FEEDBACK_POLICY,
        notebook_observation_policy=fresh_observation.POLICY,
        notebook_observation_prompt_sha256=digest(fresh_observation.PROMPT),
        notebook_observation_max_new_tokens=fresh_observation.MAX_NEW_TOKENS,
        notebook_observation_n=1,
        notebook_observation_seed_rule="per-item-policy-stage-sha256-fixed-first-no-selection",
        actor_mode_policy=ACTOR_MODE_POLICY,
        source_observation_instruction=SOURCE_OBSERVATION_INSTRUCTION.copy(),
        batch={"problems_per_step": 2, "max_attempted_problems_per_step": 2, "epochs": 1},
        local_budget={"minimal_problems": 2, "minimal_updates": 1, "train_problems": 10,
                      "train_updates": 5, "seed": 3407, "max_seconds_per_run": 14400},
    )
    protocol["rollout"].update(
        n={1: 4, 2: 2, 3: 2}, branch={1: 1, 2: 1}, max_attempts=3,
        max_new_tokens={1: 1024, 2: 1024, 3: 1024}, final_notebook=False,
        diag={"enabled_in": [], "n": 4, "max_new_tokens": 2048, "class_cap": None},
    )
    protocol["model"]["logprob_chunk"] = 64
    protocol["optimizer"]["warmup_steps"] = 0
    protocol["engine"] = {
        "backend": "shared_hf_policy", "concurrency": 1,
        "probe_max_mean_abs_logprob_diff": .02,
    }
    if repair_notebook:
        from . import notebook_synthesis, reasoning_trace, repair_contract, repair_credit, repair_notebook as repair
        from .prompts import native_solution_prompt_contract, repair_retry_prompt_contract
        from .rollout import _hash

        protocol = {key: value for key, value in protocol.items() if not key.startswith("notebook_")}
        objective = repair_objective_contract(note_contract, answer_verified_route=answer_verified_route,
                                              finalization_reward=finalization_reward)
        generation = repair.contract(note_contract)
        protocol.update(protocol_id=REPAIR_PROTOCOL_ID,
            repair_objective=objective, repair_objective_sha256=digest(objective),
            repair_notebook_version=generation["version"], repair_notebook_contract=generation,
            repair_notebook_contract_sha256=_hash(generation),
            repair_retry_contract=repair_contract.VERSION, repair_origin_contract=repair_contract.ORIGIN_VERSION,
            repair_final_contract=note_contract,
            repair_retry_prompt_sha256=_hash(repair_retry_prompt_contract()),
            repair_native_audit_version=notebook_synthesis.NATIVE_SAMPLE_AUDIT_VERSION,
            repair_online_audit_version=repair_credit.AUDIT_VERSION,
            repair_final_audit_version=(repair_credit.FINAL_AUDIT_VERSION if note_contract == repair_contract.FINAL_VERSION
                                        else note_contract),
            repair_route_rule=STRICT_ROUTE_RULE, solution_trace_contract=reasoning_trace.VERSION,
            solution_prompt_sha256=digest(native_solution_prompt_contract()))
        protocol["rollout"]["retry_contract"] = repair_contract.VERSION
        protocol["rollout"]["diag"]["max_new_tokens"] = 8192
        if answer_verified_route:
            from .prompts import SLOTLESS_VERSION, slotless_solution_prompt_contract

            protocol.update(repair_route_policy=ANSWER_ROUTE_RULE, repair_slotless_solution_contract=SLOTLESS_VERSION,
                repair_slotless_prompt_sha256=digest(slotless_solution_prompt_contract()),
                repair_slotless_sources=list(slotless_sources))
        if finalization_reward:
            protocol.update(repair_finalization_contract=FINALIZATION_VERSION,
                            repair_finalization_weight=FINALIZATION_WEIGHT)
        if turn1_contract is not None:
            from .prompts import turn1_eval_contract

            # Bound only when chosen, so every default and historical protocol hash is unchanged.
            turn1 = turn1_eval_contract(*(() if turn1_max_new_tokens is None else (turn1_max_new_tokens,)))
            protocol.update(repair_turn1_contract=turn1, repair_turn1_contract_sha256=digest(turn1))
            if retry_out_of_choice is not None:
                protocol["repair_retry_out_of_choice"] = retry_out_of_choice
        if scale is not None:
            from . import scale as scale_options

            # Opt-in speed/scale group; absent unless explicitly given, so default hashes are unchanged.
            scale_options.bind(protocol, scale)
    elif verified_note_atoms:
        from .notebook_objective import contract

        protocol.update(protocol_id=ATOMIC_PROTOCOL_ID, notebook_objective=contract())
    elif verified_note_components:
        from .notebook_objective import COMPONENT_VERSION, contract

        protocol.update(protocol_id=COMPONENT_PROTOCOL_ID, notebook_objective=contract(COMPONENT_VERSION))
    elif verified_note_fact_credit:
        from .notebook_objective import FACT_CREDIT_VERSION, contract

        protocol.update(protocol_id=(FACT_CREDIT_24K_PROTOCOL_ID if context_limit == 24576 else FACT_CREDIT_PROTOCOL_ID),
                        notebook_objective=contract(FACT_CREDIT_VERSION))
    elif verified_note_synthesis or verified_note_synthesis_credit:
        from . import notebook_synthesis, reasoning_trace
        from .prompts import native_solution_prompt_contract

        protocol.update(
            protocol_id=SYNTHESIS_24K_PROTOCOL_ID if context_limit == 24576 else SYNTHESIS_PROTOCOL_ID,
            notebook_reward=notebook_synthesis.BOUNDED_VERSION,
            notebook_synthesis_contract=notebook_synthesis.BOUNDED_VERSION,
            notebook_repair_parent_policy=notebook_synthesis.REPAIR_PARENT_POLICY,
            notebook_prompt_sha256=digest(notebook_synthesis.prompt_contract(notebook_synthesis.BOUNDED_VERSION)),
            notebook_input_policy=notebook_synthesis.INPUT_POLICY,
            notebook_feedback_policy=notebook_synthesis.FEEDBACK_POLICY,
            notebook_retry_instruction_sha256=digest(notebook_synthesis.RETRY_INSTRUCTION),
            notebook_format_feedback_policy=notebook_synthesis.FORMAT_DIAGNOSTICS_VERSION,
            notebook_format_feedback_sha256=digest(notebook_synthesis.format_feedback_contract()),
            solution_trace_contract=reasoning_trace.VERSION,
            solution_prompt_sha256=digest(native_solution_prompt_contract()),
            synthesis_scope="verified_claims_and_complete_evidence_bound_unknown_inventory_not_full_diagnosis",
        )
        # Whole-history inventories can exceed 2K even with compact JSON. This
        # is an explicit new protocol budget, not truncation or an adaptive retry.
        protocol["rollout"]["diag"]["max_new_tokens"] = 8192
        if verified_note_synthesis_credit:
            from .synthesis_credit import contract as synthesis_credit_contract

            protocol.update(protocol_id=SYNTHESIS_CREDIT_PROTOCOL_ID, notebook_objective=synthesis_credit_contract())
    if partial_notebook or referenced_notebook or paired_notebook:
        from . import partial_notebook as partial, referenced_notebook as referenced, paired_notebook as paired

        generation = paired if paired_notebook else referenced if referenced_notebook else partial
        protocol.update(protocol_id=(PAIRED_NOTEBOOK_PROTOCOL_ID if paired_notebook else
            REFERENCED_NOTEBOOK_PROTOCOL_ID if referenced_notebook else PARTIAL_NOTEBOOK_PROTOCOL_ID),
            notebook_generation_policy=generation.VERSION,
            notebook_prompt_sha256=digest(generation.prompt_contract()), notebook_max_rounds=1,
            notebook_strict_feedback_repairs=False)
    if pair_action_notebook:
        from . import pair_action_notebook as action
        from .updates import NATIVE_RELATIONSHIP_REQUIREMENT

        protocol.update(protocol_id=PAIR_ACTION_NOTEBOOK_PROTOCOL_ID,
            notebook_generation_policy=action.VERSION, notebook_generation_contract=action.contract(),
            notebook_prompt_sha256=digest(action.prompt_contract()), notebook_max_rounds=1,
            notebook_strict_feedback_repairs=False,
            notebook_update_requirement=NATIVE_RELATIONSHIP_REQUIREMENT)
        protocol["rollout"]["diag"]["max_new_tokens"] = action.MAX_NEW_TOKENS
    if pair_draft_synthesis:
        from . import notebook_pair_drafts as drafts

        protocol.update(protocol_id=PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID,
            notebook_draft_policy=drafts.VERSION, notebook_draft_contract=drafts.contract(),
            notebook_draft_contract_sha256=digest(drafts.contract()), notebook_max_rounds=1,
            notebook_strict_feedback_repairs=False)
    if whole_reference_notebook or whole_reference_pages:
        from . import whole_reference_notebook as reference
        from .updates import NATIVE_RELATIONSHIP_REQUIREMENT

        protocol.update(protocol_id=WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID,
            notebook_generation_policy=reference.VERSION,
            notebook_reference_contract=reference.contract(),
            notebook_reference_contract_sha256=digest(reference.contract()),
            notebook_prompt_sha256=digest(reference.prompt_contract()),
            notebook_update_requirement=NATIVE_RELATIONSHIP_REQUIREMENT)
    if whole_reference_pages:
        from . import whole_reference_pages as pages

        protocol.update(protocol_id=WHOLE_REFERENCE_PAGES_PROTOCOL_ID,
            notebook_generation_policy=pages.VERSION,
            notebook_page_contract=pages.contract(),
            notebook_page_contract_sha256=digest(pages.contract()),
            notebook_prompt_sha256=digest(pages.prompt_contract()))
    if verified_note_assessment_credit:
        from .synthesis_credit import ASSESSMENT_VERSION, contract as synthesis_credit_contract
        from .updates import ASSESSMENT_REQUIREMENT

        protocol.update(protocol_id=ASSESSMENT_PROTOCOL_ID,
            notebook_objective=synthesis_credit_contract(ASSESSMENT_VERSION),
            notebook_update_requirement=ASSESSMENT_REQUIREMENT)
    protocol["sha256"] = digest(protocol)
    return protocol


def _bound_scale(record: dict) -> dict:
    from .scale import explicit

    group = explicit(record)
    assert group is not None
    return group


def restore_protocol(record: dict) -> dict:
    """Only explicitly versioned complete contracts may resume across JSON boundaries."""
    import json

    from .repair_notebook import note_contract

    if type(record) is not dict or record.get("protocol_id") not in (
            PROTOCOL_ID, ATOMIC_PROTOCOL_ID, COMPONENT_PROTOCOL_ID, FACT_CREDIT_PROTOCOL_ID, FACT_CREDIT_24K_PROTOCOL_ID,
            SYNTHESIS_PROTOCOL_ID, SYNTHESIS_24K_PROTOCOL_ID, SYNTHESIS_CREDIT_PROTOCOL_ID,
            PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID, PAIRED_NOTEBOOK_PROTOCOL_ID,
            PAIR_ACTION_NOTEBOOK_PROTOCOL_ID, PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID, WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID, REPAIR_PROTOCOL_ID):
        raise ValueError("local profile: unknown saved protocol")
    protocol = build_protocol(verified_note_atoms=record["protocol_id"] == ATOMIC_PROTOCOL_ID,
                              verified_note_components=record["protocol_id"] == COMPONENT_PROTOCOL_ID,
                              verified_note_fact_credit=record["protocol_id"] in (
                                  FACT_CREDIT_PROTOCOL_ID, FACT_CREDIT_24K_PROTOCOL_ID),
                              verified_note_synthesis=record["protocol_id"] in (
                                  SYNTHESIS_PROTOCOL_ID, SYNTHESIS_24K_PROTOCOL_ID),
                              verified_note_synthesis_credit=record["protocol_id"] in (
                                  SYNTHESIS_CREDIT_PROTOCOL_ID, PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID,
                                  PAIRED_NOTEBOOK_PROTOCOL_ID, PAIR_ACTION_NOTEBOOK_PROTOCOL_ID, PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID, WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID),
                              partial_notebook=record["protocol_id"] == PARTIAL_NOTEBOOK_PROTOCOL_ID,
                              referenced_notebook=record["protocol_id"] == REFERENCED_NOTEBOOK_PROTOCOL_ID,
                              paired_notebook=record["protocol_id"] == PAIRED_NOTEBOOK_PROTOCOL_ID,
                              pair_action_notebook=record["protocol_id"] == PAIR_ACTION_NOTEBOOK_PROTOCOL_ID,
                              pair_draft_synthesis=record["protocol_id"] in (PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID, WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID),
                              whole_reference_notebook=record["protocol_id"] in (WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID),
                              verified_note_assessment_credit=record["protocol_id"] == ASSESSMENT_PROTOCOL_ID,
                              whole_reference_pages=record["protocol_id"] == WHOLE_REFERENCE_PAGES_PROTOCOL_ID,
                              repair_notebook=record["protocol_id"] == REPAIR_PROTOCOL_ID,
                              **({"note_contract": note_contract(record), **repair_options(record), **turn1_options(record),
                                  **({"scale": _bound_scale(record)} if "scale" in record else {})}
                                 if record["protocol_id"] == REPAIR_PROTOCOL_ID else {}),
                              context_limit=24576 if record["protocol_id"] in (
                                  FACT_CREDIT_24K_PROTOCOL_ID, SYNTHESIS_24K_PROTOCOL_ID,
                                  SYNTHESIS_CREDIT_PROTOCOL_ID, PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID,
                                  PAIRED_NOTEBOOK_PROTOCOL_ID, PAIR_ACTION_NOTEBOOK_PROTOCOL_ID,
                                  PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID, WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID, REPAIR_PROTOCOL_ID) else 16384)
    if (type(record.get("context_limit")) is not int
            or json.loads(json.dumps(record)) != json.loads(json.dumps(protocol))
            or record["protocol_id"] in (PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID,
                                        PAIRED_NOTEBOOK_PROTOCOL_ID, PAIR_ACTION_NOTEBOOK_PROTOCOL_ID,
                                        PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID, WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID, REPAIR_PROTOCOL_ID)
            and digest(record) != digest(protocol)):
        raise ValueError("local profile: saved protocol changed")
    return protocol


def local_prompts(protocol: dict) -> dict:
    if protocol.get("verifier_version") != VERIFIER_VERSION:
        raise ValueError("local profile: verifier version differs from the current contract")
    if protocol.get("protocol_id") == REPAIR_PROTOCOL_ID:
        restore_protocol(protocol)
        return load_prompts(HERE / "prompts_v1.yaml") | {
            "solution_trace_contract": protocol["solution_trace_contract"],
            "source_observation_instruction": SOURCE_OBSERVATION_INSTRUCTION.copy(),
            **{key: protocol[key] for key in ("repair_notebook_version", "repair_notebook_contract",
                                            "repair_notebook_contract_sha256")},
            # Opt-in keys pass through only when bound, so default prompt bytes and hashes are unchanged.
            **{key: protocol[key] for key in ("repair_route_policy", "repair_slotless_sources",
                                            "repair_finalization_contract", "repair_finalization_weight",
                                            "repair_turn1_contract", "repair_retry_out_of_choice")
               if key in protocol},
            **({"repair_slotless_observation_instruction": SLOTLESS_REREAD_INSTRUCTION}
               if "repair_route_policy" in protocol else {})}
    if any(key.startswith("repair_") for key in protocol):
        raise ValueError("local profile: repair metadata requires its own explicit protocol")
    if protocol.get("protocol_id") in (
            SYNTHESIS_PROTOCOL_ID, SYNTHESIS_24K_PROTOCOL_ID, SYNTHESIS_CREDIT_PROTOCOL_ID,
            PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID, PAIRED_NOTEBOOK_PROTOCOL_ID,
            PAIR_ACTION_NOTEBOOK_PROTOCOL_ID, PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID, WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID):
        from . import notebook_synthesis

        # A new reward is a complete new contract, never a rehashed historical protocol.
        restore_protocol(protocol)
        prompts = load_prompts(HERE / "prompts_v1.yaml")
        result = prompts | {
            "rewarded_notebook": notebook_synthesis.prompt_contract(notebook_synthesis.BOUNDED_VERSION),
            "notebook_synthesis_contract": notebook_synthesis.BOUNDED_VERSION,
            "notebook_repair_parent_policy": notebook_synthesis.REPAIR_PARENT_POLICY,
            "notebook_input_policy": notebook_synthesis.INPUT_POLICY,
            "notebook_route_rule": STRICT_ROUTE_RULE,
            "notebook_feedback_policy": notebook_synthesis.FEEDBACK_POLICY,
            "notebook_observation_policy": fresh_observation.POLICY,
            "solution_trace_contract": protocol["solution_trace_contract"],
            "source_observation_instruction": SOURCE_OBSERVATION_INSTRUCTION.copy(),
        }
        if protocol["protocol_id"] in (PARTIAL_NOTEBOOK_PROTOCOL_ID, REFERENCED_NOTEBOOK_PROTOCOL_ID, PAIRED_NOTEBOOK_PROTOCOL_ID):
            from . import partial_notebook as partial, referenced_notebook as referenced, paired_notebook as paired

            generation = (paired if protocol["protocol_id"] == PAIRED_NOTEBOOK_PROTOCOL_ID else
                          referenced if protocol["protocol_id"] == REFERENCED_NOTEBOOK_PROTOCOL_ID else partial)
            result.update(rewarded_notebook=generation.prompt_contract(),
                notebook_generation_policy=generation.VERSION, notebook_max_rounds=1,
                notebook_strict_feedback_repairs=False)
        if protocol["protocol_id"] == PAIR_ACTION_NOTEBOOK_PROTOCOL_ID:
            from . import pair_action_notebook as action

            result.update(rewarded_notebook=action.prompt_contract(),
                notebook_generation_policy=action.VERSION, notebook_generation_contract=action.contract(),
                notebook_max_rounds=1, notebook_strict_feedback_repairs=False)
        if protocol["protocol_id"] in (PAIR_DRAFT_SYNTHESIS_PROTOCOL_ID, WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID):
            result.update(notebook_draft_policy=protocol["notebook_draft_policy"],
                notebook_draft_contract=protocol["notebook_draft_contract"],
                notebook_draft_contract_sha256=protocol["notebook_draft_contract_sha256"],
                notebook_max_rounds=1, notebook_strict_feedback_repairs=False)
        if protocol["protocol_id"] in (WHOLE_REFERENCE_NOTEBOOK_PROTOCOL_ID, WHOLE_REFERENCE_PAGES_PROTOCOL_ID, ASSESSMENT_PROTOCOL_ID):
            from . import whole_reference_notebook as reference

            result.update(rewarded_notebook=reference.prompt_contract(),
                notebook_generation_policy=reference.VERSION,
                notebook_reference_contract=protocol["notebook_reference_contract"],
                notebook_reference_contract_sha256=protocol["notebook_reference_contract_sha256"])
        if protocol["protocol_id"] == WHOLE_REFERENCE_PAGES_PROTOCOL_ID:
            from . import whole_reference_pages as pages

            result.update(rewarded_notebook=pages.prompt_contract(), notebook_generation_policy=pages.VERSION,
                notebook_page_contract=protocol["notebook_page_contract"],
                notebook_page_contract_sha256=protocol["notebook_page_contract_sha256"])
        return result
    if any(key in protocol for key in ("notebook_synthesis_contract", "notebook_repair_parent_policy", "notebook_generation_policy",
                                        "notebook_generation_contract", "notebook_draft_policy", "notebook_draft_contract",
                                        "notebook_draft_contract_sha256", "notebook_reference_contract", "notebook_reference_contract_sha256",
                                        "notebook_page_contract", "notebook_page_contract_sha256",
                                        "notebook_update_requirement", "solution_trace_contract", "solution_prompt_sha256")):
        raise ValueError("local profile: synthesis requires its own protocol")
    if protocol.get("protocol_id") in (
            ATOMIC_PROTOCOL_ID, COMPONENT_PROTOCOL_ID, FACT_CREDIT_PROTOCOL_ID, FACT_CREDIT_24K_PROTOCOL_ID):
        from .notebook_objective import COMPONENT_VERSION, FACT_CREDIT_VERSION, VERSION, contract

        version = {ATOMIC_PROTOCOL_ID: VERSION, COMPONENT_PROTOCOL_ID: COMPONENT_VERSION,
                   FACT_CREDIT_PROTOCOL_ID: FACT_CREDIT_VERSION,
                   FACT_CREDIT_24K_PROTOCOL_ID: FACT_CREDIT_VERSION}[protocol["protocol_id"]]
        if protocol.get("notebook_objective") != contract(version):
            raise ValueError("local profile: atomic objective contract changed")
    elif protocol.get("protocol_id") != PROTOCOL_ID or "notebook_objective" in protocol:
        raise ValueError("local profile: notebook objective requires its own protocol")
    expected_context = 24576 if protocol["protocol_id"] == FACT_CREDIT_24K_PROTOCOL_ID else 16384
    if type(protocol.get("context_limit")) is not int or protocol["context_limit"] != expected_context:
        raise ValueError("local profile: context limit differs from its canonical protocol")
    prompts = load_prompts(HERE / "prompts_v1.yaml")
    if (prompts["sha256"] != protocol["prompts_sha256"]
            or digest(NOTEBOOK_PROMPT) != protocol["notebook_prompt_sha256"]
            or protocol["notebook_reward"] != NOTEBOOK_REWARD_VERSION):
        raise ValueError("local profile: frozen prompt/reward mismatch")
    if protocol.get("notebook_input_policy") != INPUT_POLICY:
        raise ValueError("local profile: notebook input policy changed")
    if (protocol.get("notebook_max_rounds") != MAX_NOTEBOOK_ROUNDS
            or protocol.get("notebook_retry_instruction_sha256") != digest(REINSPECTION_INSTRUCTION)):
        raise ValueError("local profile: frozen notebook retry budget/instruction mismatch")
    if protocol.get("source_observation_instruction") != SOURCE_OBSERVATION_INSTRUCTION:
        raise ValueError("local profile: source observation instructions changed")
    if protocol.get("notebook_route_rule") != STRICT_ROUTE_RULE:
        raise ValueError("local profile: strict routing admission required")
    if protocol.get("notebook_feedback_policy") != FEEDBACK_POLICY:
        raise ValueError("local profile: submitted-fact feedback policy required")
    if any(protocol.get(key) != value for key, value in {
        "notebook_observation_policy": fresh_observation.POLICY,
        "notebook_observation_prompt_sha256": digest(fresh_observation.PROMPT),
        "notebook_observation_max_new_tokens": fresh_observation.MAX_NEW_TOKENS,
        "notebook_observation_n": 1,
        "notebook_observation_seed_rule": "per-item-policy-stage-sha256-fixed-first-no-selection",
    }.items()):
        raise ValueError("local profile: fresh observation contract changed")
    if protocol.get("actor_mode_policy") != ACTOR_MODE_POLICY:
        raise ValueError("local profile: explicit observation-elicitation policy required")
    return prompts | {"rewarded_notebook": NOTEBOOK_PROMPT,
        "notebook_input_policy": INPUT_POLICY,
        "notebook_route_rule": STRICT_ROUTE_RULE,
        "notebook_feedback_policy": FEEDBACK_POLICY,
        "notebook_observation_policy": fresh_observation.POLICY,
        "source_observation_instruction": SOURCE_OBSERVATION_INSTRUCTION.copy()}
