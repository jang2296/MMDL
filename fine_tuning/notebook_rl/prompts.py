"""Gold-free prompt builders; retry histories are joined by token ID elsewhere."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from .schema import ActorItem, Mode, Status, validate_actor

_TOKENS = {
    "turn1_plain": {"P0_BODY"}, "turn1_obs": {"P0_BODY", "OBSERVATION_INSTRUCTION"},
    "retry_incorrect": {"PREV_ANSWER", "OBS_SECTION"},
    "retry_invalid": {"REASON", "OBS_SECTION"}, "diagnose": {"K", "K_MINUS_1"},
    "oracle_facts": {"FACT_LINES", "P0_BODY"},
    "self_perception_probe": {"OBSERVED_TEXT", "P0_BODY"},
}
_PLACEHOLDER = re.compile(r"\{([A-Z_][A-Z_0-9]*)\}")
NATIVE_TURN1 = """{P0_BODY}
Reinspect the original images. Record your observations and your executed operations separately.
{TRACE_INSTRUCTION}
Observed facts assignment syntax (no reference values):
{OBSERVATION_INSTRUCTION}
Use only Observed facts:, Solution:, the single Trace JSON line, and Final answer:.
"""
NATIVE_RETRY = """{FEEDBACK}
Reinspect the original images and all your earlier attempts. Check observations and operations again;
neither an earlier answer nor an earlier observation is certified by its presence in the history.
{TRACE_INSTRUCTION}
Observed facts assignment syntax (no reference values):
{OBSERVATION_INSTRUCTION}
Write a new complete response, with no separate Error check or explanatory prose outside the trace.
"""
REPAIR_OPERATION_PATCH_INSTRUCTION = """Operation patch convention (public syntax, no reference values):
field is a top-level Trace key; value is the exact selected piece below, not always the whole field.
Use the matching operation as actually emitted in the selected native solution; its index may differ from the
ancestor locus. An array index alone does not establish the same operation. Slices are half-open.
PATH_ENDPOINTS and MINIMALITY: field nodes, value the whole nodes list.
DIRECTED_EDGE at nodes[i:i+2]: field nodes, value only the two-node nodes[i:i+2] slice.
EDGE_WEIGHT at weights[i]: field weights, value only weights[i], with its original JSON integer type.
COST_SUM: field cost, value the cost scalar.
BFS_FIFO and BFS_NEW_NEIGHBORS at steps[i].node or steps[i].new: field steps, value the whole steps[i]
object including both node and new; do not submit the whole steps list or just node/new.
BFS_COMPLETENESS: field steps, value the whole steps list.
TREE_START, TREE_CHILD_EDGE and TREE_EVENT_ORDER at events[i]: field events, value the whole events[i] object.
TREE_COMPLETENESS: field events, value the whole events list.
DFA_START: field states, value only states[0]. DFA_TRANSITION at states[i:i+2]: field states,
value only that two-state slice. DFA_COMPLETE_WORD: field states, value the whole states list.
DFA_ACCEPTANCE: field decision, value the decision string.
CIRCUIT_NODE_VOLTAGE at nodes[i]: field nodes, value the whole selected node row.
CIRCUIT_INCIDENCE, CIRCUIT_RESISTANCE, CIRCUIT_DROP and CIRCUIT_OHM at ohm[i]: field ohm,
value the whole selected resistor row. CIRCUIT_KCL_INCIDENCE, CIRCUIT_KCL_SUM and CIRCUIT_KCL_ZERO
at kcl[i]: field kcl, value the whole selected KCL row.
CIRCUIT_NODE_COVERAGE: field nodes, value the whole nodes list. CIRCUIT_RESISTOR_COVERAGE:
field ohm, value the whole ohm list. CIRCUIT_KCL_COVERAGE: field kcl, value the whole kcl list.
Every CIRCUIT_INPUT_* rule: field input, value the whole input list, even when the locus has an index.
CIRCUIT_QUERY_SPEC and CIRCUIT_QUERY_VALUE: field query, value the whole query list including the unit.
Preserve JSON types, rational strings and units exactly; do not convert or normalize the emitted piece.
TRACE_FORMAT, TRACE_SCHEMA, RESPONSE_SCOPE and NATIVE_CONTRACT have no operation_patch mapping.
"""
PUBLIC_ORIGIN_DICTIONARY_VERSION = "public-origin-dictionary-v2"
PUBLIC_ORIGIN_DICTIONARY_INSTRUCTION = """Public origins use public-origin-dictionary-v2. schemas is a zero-based list of
[kind,rule,locus]. Each source gives its full sample_id once and rows [schema,start,end,start,end,...].
A row's ordinal is its one-based position within that source; zero or more span pairs follow its schema.
Reconstruct each canonical ref as sample_id + ':r' + the decimal ordinal (for example sample_id:r1).
Submit these full canonical refs, never a source index or local ordinal alone. Spans [start,end] are
half-open character offsets in that source's original wrapped response, not bytes or token offsets.
All source and row orders are original; normal, unknown and malformed origins remain public work items.

"""
REPAIR_RETRY = """{FEEDBACK}
Reinspect the original images and all actual earlier attempts. Write one completion, within the
same total output limit: exactly one leading ErrorCheck: JSON object, then a newline, RevisedSolution:
on its own line, then your new complete native response. The ErrorCheck object has EXACTLY version
(the string repair-check-v1), claims (at most 3), and uncertain_refs (at most 3 unique public ref strings).
Each claim has EXACTLY origin_ref, intent, rule, proposed_value. Copy an actual ancestor origin_ref and
its rule from the public inventory. Use each origin once; uncertain_refs must not repeat claim origins.
intent is RECHECK_OBSERVATION, RECHECK_OPERATION, or UNKNOWN. Observation proposed_value is your own
single-line scalar string. Operation proposed_value is an operation_patch object with EXACTLY field
and value, using this selected-piece convention:
""" + REPAIR_OPERATION_PATCH_INSTRUCTION + """
UNKNOWN uses proposed_value null. Rechecking is an intention, not a claim that an origin was wrong.
Do not merely propose a patch: emit the matching observation/operation in RevisedSolution itself.
The public inventory includes normal and unknown claims; membership certifies no error or correction.
No expected values or private issue list is provided. Quoted earlier answers/facts in ErrorCheck are
not new observations, executed operations, or your new final answer.
Within RevisedSolution use the following existing native contract:
{TRACE_INSTRUCTION}
Observed facts assignment syntax (no reference values):
{OBSERVATION_INSTRUCTION}
Use only Observed facts:, Solution:, the single Trace JSON line, and Final answer: within that body.
""" + PUBLIC_ORIGIN_DICTIONARY_INSTRUCTION + """<public_origins>
{ORIGINS}
</public_origins>
"""
SLOTLESS_VERSION = "slotless-answer-solution-v1"
# The verifier grades a unit/percent/prose-bearing numeric answer UNVERIFIABLE (blocking the group),
# so the slotless final line names the bare answer shape for each public answer format.
_NUMBER_ONLY = "Final answer: <a number only, with no units, percent sign or other words>"
SLOTLESS_FINAL_LINES = {"mcq": "Final answer: <the letter of the correct option>", "int": _NUMBER_ONLY,
                        "decimal": _NUMBER_ONLY,
                        "fraction": "Final answer: <the exact value as a fraction or whole number, with no units or words>",
                        "unit_number": "Final answer: <answer>",
                        "short_text": "Final answer: <a short word or phrase only>",
                        "list": "Final answer: <the items separated by commas>"}
SLOTLESS_TURN1 = """{P0_BODY}
Write your reasoning under Solution:, then end your response with exactly one line: {FINAL_LINE}
"""
SLOTLESS_RETRY = """{FEEDBACK}
Reinspect the original images and all your earlier attempts; neither an earlier answer nor an earlier
step is certified by its presence in the history.
Write a new complete response: your reasoning under Solution:, then exactly one line: {FINAL_LINE}
"""
SLOTLESS_REPAIR_RETRY = """{FEEDBACK}
Reinspect the original images and all actual earlier attempts. Write one completion, within the
same total output limit: exactly one leading ErrorCheck: JSON object, then a newline, RevisedSolution:
on its own line, then your new complete response. The ErrorCheck object has EXACTLY version
(the string repair-check-v1), claims (at most 3), and uncertain_refs (at most 3 unique public ref strings).
Each claim has EXACTLY origin_ref, intent, rule, proposed_value. Copy an actual ancestor origin_ref and
its rule from the public inventory. Use each origin once; uncertain_refs must not repeat claim origins.
For this question use intent UNKNOWN with proposed_value null; an empty claims list is allowed.
Rechecking is an intention, not a claim that an origin was wrong. No expected values or private issue
list is provided. Quoted earlier answers in ErrorCheck are not your new final answer.
Within RevisedSolution write your reasoning under Solution:, then exactly one line: {FINAL_LINE}
""" + PUBLIC_ORIGIN_DICTIONARY_INSTRUCTION + """<public_origins>
{ORIGINS}
</public_origins>
"""
# Unrewarded, unverified context only, exactly like the slot schema reread; never routing evidence.
SLOTLESS_REREAD_INSTRUCTION = ("Re-read the image and the question. List the visual facts relevant to the "
                               "question, one per line as <name> = <value>.")
REPAIR_FEEDBACK_VERSION = "repair-retry-feedback-v3"
REPAIR_MALFORMED_FEEDBACK = """Your previous saved response failed its declared grammar: {CODE}.
Answer correctness: NOT JUDGED. No answer grade or correct answer is provided."""
REPAIR_MALFORMED_LENGTH_FEEDBACK = "The previous response reached its output limit and may be incomplete."


def repair_retry_prompt_contract() -> dict:
    from .repair_contract import VERSION
    from . import fresh_observation as fresh

    return {"response_contract": VERSION, "template": REPAIR_RETRY,
            "retry_reread_policy": fresh.POLICY, "retry_reread_prompt_sha256": hashlib.sha256(fresh.PROMPT.encode()).hexdigest(),
            "retry_reread_context": fresh.RETRY_CONTEXT,
            "retry_reread_quote_sha256": hashlib.sha256(fresh.quoted_retry_context({"completion": {
                "finish_reason": "length", "text": "\r\n<|im_start|>🙂"}}).encode()).hexdigest(),
            "retry_reread_position": "original_feedback + newline + quoted_retry_context",
            "feedback_version": REPAIR_FEEDBACK_VERSION,
            "malformed_feedback": REPAIR_MALFORMED_FEEDBACK,
            "malformed_length_feedback": REPAIR_MALFORMED_LENGTH_FEEDBACK}


def native_solution_prompt_contract() -> dict:
    """Freeze public task instructions, without generating a world or a reference answer."""
    from .generators import graph, graph_pair
    from .reasoning_trace import VERSION, instruction_for_question

    questions = [(graph.GENERATOR_VERSION, graph._question(kind, traversal))
                 for kind in ("shortest", "bfs", "dfa", "tree")
                 for traversal in (("preorder", "inorder", "postorder") if kind == "tree" else ("postorder",))]
    questions.append((graph_pair.GENERATOR_VERSION, graph_pair._QUESTION))
    return {"trace_contract": VERSION, "turn1": NATIVE_TURN1, "retry": NATIVE_RETRY,
            "instructions": [{"source": source, "question": question,
                              "instruction": instruction_for_question(question, source)}
                             for source, question in questions]}


def turn1_eval_contract(max_new_tokens: int = 1024) -> dict:
    """Opt-in first-attempt contract, bound into a protocol only when chosen.

    Applies only to answer-verified (slotless) actors' T1: the exact MMMU P0 user message and V8 grading.
    Obs/slot actors keep their native prompts; retries, ErrorCheck, notes and their caps are unchanged."""
    from . import v8_parser
    from .answers import TURN1_EVAL_CONTRACT

    if type(max_new_tokens) is not int or not 1 <= max_new_tokens < 24576:
        raise ValueError("prompts: T1 cap must be a positive int below the 24576 context")
    return {"version": TURN1_EVAL_CONTRACT, "scope": "answer_verified_slotless_actors_T1_only",
            "message": "single user message, images first then the P0 text, chat template add_generation_prompt",
            "p0_examples": {"mcq": p0_eval_text("Q", ("x", "y")), "open": p0_eval_text("Q", ())},
            "parser": v8_parser.VERSION, "parser_source_sha256": dict(v8_parser.SOURCE_SHA256),
            "vendored_parser_sha256": hashlib.sha256(Path(v8_parser.__file__).read_bytes()).hexdigest(),
            "comparison": "answers.compare with the private verifier gold",
            "status": {"extracted_match": "CORRECT", "extracted_mismatch": "INCORRECT",
                       "no_extraction": "INVALID_NO_FINAL", "length_stop": "TRUNCATED",
                       "uncomparable_gold": "UNVERIFIABLE"},
            "finalization": "T1: V8 extracts an answer and finish_reason is stop; T2/T3 unchanged",
            "max_new_tokens": max_new_tokens}


def turn1_contract(actor: ActorItem, prompts: dict) -> dict | None:
    """The bound eval-aligned T1 contract for this actor, or None (obs/slot actors, unflagged prompts)."""
    return for_actor(actor, prompts).get("_turn1_contract")


def slotless_solution_prompt_contract() -> dict:
    """Freeze the answer-only native texts; hashed only by the opt-in answer-verified route."""
    return {"version": SLOTLESS_VERSION, "turn1": SLOTLESS_TURN1, "retry": SLOTLESS_RETRY,
            "repair_retry": SLOTLESS_REPAIR_RETRY, "reread_instruction": SLOTLESS_REREAD_INSTRUCTION,
            "final_lines": dict(SLOTLESS_FINAL_LINES), "actor_modes": ["obs", "plain"],
            "trace": None, "observation_slots": None}


def circuit_solution_prompt_contract() -> dict:
    """Bind every heldout ladder query's public grammar without generating a solution."""
    from . import circuit_trace
    from .generators import circuit
    from .reasoning_trace import instruction_for_question

    queries = [("current", "R4"), *(("voltage", f"N{node}") for node in (3, 4, 5)),
               ("resistance", "input")]
    return {"trace_contract": circuit_trace.VERSION, "turn1": NATIVE_TURN1, "retry": NATIVE_RETRY,
            "scope": circuit_trace.SCOPE, "source": circuit.GENERATOR_VERSION,
            "observation_instruction": circuit.OBSERVATION_INSTRUCTION,
            "instructions": [{"question": circuit._question(query),
                              "instruction": instruction_for_question(circuit._question(query), circuit.GENERATOR_VERSION)}
                             for query in queries]}


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError("prompts: duplicate or non-string key")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_prompts(path: Path) -> dict:
    raw = path.read_bytes()
    data = yaml.load(raw, Loader=_UniqueLoader)
    if type(data) is not dict or set(data) != {*_TOKENS, "observation_instruction"}:
        raise ValueError("prompts: invalid template keys")
    for key, tokens in _TOKENS.items():
        if (not isinstance(data[key], str) or not data[key].strip()
                or set(_PLACEHOLDER.findall(data[key])) != tokens):
            raise ValueError("prompts: invalid template placeholders")
    obs = data["observation_instruction"]
    if (type(obs) is not dict or not obs or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in obs.items()
    )):
        raise ValueError("prompts: invalid observation instructions")
    return data | {"sha256": hashlib.sha256(raw).hexdigest()}


def _render(template: str, **values: str) -> str:
    # Single template pass: literal placeholders inside user text cannot be substituted.
    return _PLACEHOLDER.sub(lambda m: values[m[1]], template)


def p0_body(question: str, choices: tuple[str, ...]) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("prompts: nonempty question required")
    if len(choices) > 10 or any(not isinstance(v, str) or not v.strip() for v in choices):
        raise ValueError("prompts: invalid choices")
    body = "Question: " + question + "\n"
    if choices:
        body += "Options:\n" + "".join(f"{chr(65 + i)}. {v}\n" for i, v in enumerate(choices))
        body += "Please select the correct answer from the options above.\n"
    return body


def p0_eval_text(question: str, choices: tuple[str, ...]) -> str:
    """The MMMU evaluation P0 text bytes (src/mmdl/evaluation/prompt.build_prompt, no hint): it renders the
    same template and strips the trailing whitespace."""
    return p0_body(question, choices).rstrip()


def _observation(mode: Mode, family: str, prompts: dict) -> str:
    if mode not in ("obs", "plain"):
        raise ValueError("prompts: unsupported mode")
    if mode == "plain":
        return ""
    if family not in prompts["observation_instruction"]:
        raise ValueError("prompts: missing family observation contract")
    return prompts["observation_instruction"][family]


def for_actor(actor: ActorItem, prompts: dict) -> dict:
    """Select a protocol-bound public source schema without reading verifier slots."""
    if "solution_trace_contract" in prompts:
        from . import circuit_trace
        from .generators import circuit, graph, graph_pair
        from .reasoning_trace import VERSION, instruction_for_question

        contract = prompts["solution_trace_contract"]
        circuit_contract = contract == circuit_trace.VERSION
        slotless = prompts.get("repair_slotless_sources", ())
        if type(slotless) not in (list, tuple):
            raise ValueError("prompts: invalid answer-verified source list")
        if actor.source in slotless:
            reread = prompts.get("repair_slotless_observation_instruction")
            if (actor.mode not in ("obs", "plain") or contract != VERSION or type(reread) is not str
                    or not reread.strip() or actor.answer_format not in SLOTLESS_FINAL_LINES
                    or actor.source in (graph.GENERATOR_VERSION, graph_pair.GENERATOR_VERSION, circuit.GENERATOR_VERSION)):
                raise ValueError("prompts: explicit answer-verified source contract required")
            turn1 = prompts.get("repair_turn1_contract")
            if turn1 is not None and (type(turn1) is not dict
                                      or turn1 != turn1_eval_contract(turn1.get("max_new_tokens"))):
                raise ValueError("prompts: unknown or changed eval-aligned T1 contract")
            # No trace or slot schema: the family instruction is only the unverified reread schema.
            return prompts | {"_slotless_source": actor.source,
                              "_slotless_final_line": SLOTLESS_FINAL_LINES[actor.answer_format],
                              "observation_instruction": prompts["observation_instruction"] | {actor.family: reread},
                              **({"_turn1_contract": turn1} if turn1 is not None else {})}
        if (actor.mode != "obs" or contract not in (VERSION, circuit_trace.VERSION)
                or circuit_contract and (actor.source != circuit.GENERATOR_VERSION or actor.family != circuit.FAMILY
                                         or actor.answer_format != "unit_number" or actor.choices)
                or not circuit_contract and actor.source not in (graph.GENERATOR_VERSION, graph_pair.GENERATOR_VERSION)):
            raise ValueError("prompts: explicit native trace contract and observation mode required")
        prompts = prompts | {"_native_trace_instruction": instruction_for_question(actor.question, actor.source)}
        if circuit_contract:
            prompts = prompts | {"_native_trace_source": actor.source}
    overrides = prompts.get("source_observation_instruction", {})
    if type(overrides) is not dict:
        raise ValueError("prompts: invalid source observation instruction mapping")
    instruction = overrides.get(actor.source)
    if instruction is None:
        return prompts
    if type(instruction) is not str or not instruction.strip():
        raise ValueError("prompts: nonempty source observation instruction required")
    return prompts | {"observation_instruction": prompts["observation_instruction"] | {actor.family: instruction}}


def turn1_text(actor: ActorItem, prompts: dict) -> str:
    validate_actor(actor)
    prompts = for_actor(actor, prompts)
    if "_turn1_contract" in prompts:
        return p0_eval_text(actor.question, actor.choices)
    if "_slotless_source" in prompts:
        return _render(SLOTLESS_TURN1, P0_BODY=p0_body(actor.question, actor.choices),
                       FINAL_LINE=prompts["_slotless_final_line"])
    if "solution_trace_contract" in prompts:
        return _render(NATIVE_TURN1, P0_BODY=p0_body(actor.question, actor.choices),
                       TRACE_INSTRUCTION=prompts["_native_trace_instruction"],
                       OBSERVATION_INSTRUCTION=_observation(actor.mode, actor.family, prompts))
    return _render(prompts["turn1_" + actor.mode],
                   P0_BODY=p0_body(actor.question, actor.choices),
                   OBSERVATION_INSTRUCTION=_observation(actor.mode, actor.family, prompts))


def retry_text(status: Status, prev_answer: str | None, mode: Mode, family: str, prompts: dict) -> str:
    if status not in ("INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED"):
        raise ValueError("prompts: this status cannot receive correctness retry feedback")
    obs = _observation(mode, family, prompts)
    if "solution_trace_contract" in prompts:
        from . import circuit_trace
        from .generators import circuit
        from .reasoning_trace import VERSION

        instruction = prompts.get("_native_trace_instruction")
        contract = prompts["solution_trace_contract"]
        slotless = "_slotless_source" in prompts
        if slotless and (contract != VERSION or mode not in ("obs", "plain") or instruction is not None
                         or prompts.get("_slotless_final_line") not in SLOTLESS_FINAL_LINES.values()):
            raise ValueError("prompts: actor-bound answer-verified retry contract required")
        if not slotless and (contract not in (VERSION, circuit_trace.VERSION) or mode != "obs"
                or contract == circuit_trace.VERSION and (
                    family != circuit.FAMILY or prompts.get("_native_trace_source") != circuit.GENERATOR_VERSION)
                or not isinstance(instruction, str) or not instruction.strip()):
            raise ValueError("prompts: actor-bound native trace retry contract required")
        if status == "INCORRECT":
            if not isinstance(prev_answer, str) or not prev_answer.strip() or "\n" in prev_answer or "\r" in prev_answer:
                raise ValueError("prompts: incorrect status requires one submitted answer")
            feedback = (f'Verifier result for your previous answer "{prev_answer}": INCORRECT.\n'
                        "The correct answer is not provided.")
        else:
            reasons = {"INVALID_NO_FINAL": "no Final answer line was given",
                       "INVALID_MULTIPLE": "more than one or conflicting final answers were given",
                       "TRUNCATED": "the response was cut off at its output limit"}
            feedback = f"Your previous response was not graded because {reasons[status]}. Its correctness is unknown."
        if slotless:
            return _render(SLOTLESS_RETRY, FEEDBACK=feedback, FINAL_LINE=prompts["_slotless_final_line"])
        assert isinstance(instruction, str)
        return _render(NATIVE_RETRY, FEEDBACK=feedback, TRACE_INSTRUCTION=instruction,
                       OBSERVATION_INSTRUCTION=obs)
    section = "Observed facts:\n" + obs if obs else ""
    if status == "INCORRECT":
        if not isinstance(prev_answer, str) or not prev_answer.strip() or "\n" in prev_answer or "\r" in prev_answer:
            raise ValueError("prompts: incorrect status requires one submitted answer")
        return _render(prompts["retry_incorrect"], PREV_ANSWER=prev_answer, OBS_SECTION=section)
    reasons = {"INVALID_NO_FINAL": "no Final answer line was given",
               "INVALID_MULTIPLE": "more than one or conflicting final answers were given",
               "TRUNCATED": "the response was cut off at its output limit"}
    return _render(prompts["retry_invalid"], REASON=reasons[status], OBS_SECTION=section)


def repair_retry_text(status: Status | None, prev_answer: str | None, actor: ActorItem, prompts: dict,
                      inventory, *, process_contradicted: bool = False,
                      malformed_code: str | None = None, malformed_finish_reason: str | None = None) -> str:
    """Request a typed check and native body using public ancestor references only."""
    from .repair_contract import GRAMMAR_ERRORS, PublicOriginInventory

    validate_actor(actor)
    prompts = for_actor(actor, prompts)
    if (actor.mode != "obs" and "_slotless_source" not in prompts or "solution_trace_contract" not in prompts
            or type(inventory) is not PublicOriginInventory or type(process_contradicted) is not bool
            or status == "CORRECT" and not process_contradicted
            or status is not None and status not in ("CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED")
            or status is None and (malformed_code not in GRAMMAR_ERRORS or prev_answer is not None
                or process_contradicted or malformed_finish_reason not in ("stop", "length"))
            or status is not None and (malformed_code is not None or malformed_finish_reason is not None)):
        raise ValueError("prompts: explicit native repair retry contract required")
    if status is None:
        assert malformed_code is not None
        feedback = _render(REPAIR_MALFORMED_FEEDBACK, CODE=malformed_code)
        if malformed_finish_reason == "length":
            feedback += "\n" + REPAIR_MALFORMED_LENGTH_FEEDBACK
    elif status == "CORRECT":
        feedback = ("Your previous answer was verified CORRECT. An independently detected process "
                    "contradiction requires review; answer correctness does not certify observations or operations.")
    else:
        # Preserve the existing public answer/termination feedback without its legacy response instruction.
        native = retry_text(status, prev_answer, actor.mode, actor.family, prompts)
        feedback = native.split("\nReinspect the original images", 1)[0]
    if "_slotless_source" in prompts:
        return _render(SLOTLESS_REPAIR_RETRY, FEEDBACK=feedback, FINAL_LINE=prompts["_slotless_final_line"],
                       ORIGINS=render_public_origin_dictionary(asdict(inventory)))
    return _render(REPAIR_RETRY, FEEDBACK=feedback, TRACE_INSTRUCTION=prompts["_native_trace_instruction"],
                   OBSERVATION_INSTRUCTION=_observation(actor.mode, actor.family, prompts),
                   ORIGINS=render_public_origin_dictionary(asdict(inventory)))


def public_origin_dictionary(inventory: dict) -> dict:
    """Losslessly factor public refs and repeated schemas without inspecting error truth."""
    sources: dict[str, list] = {}
    for source in inventory["sources"]:
        sample_id = source["sample_id"]
        if type(sample_id) is not str or not sample_id or sample_id in sources:
            raise ValueError("prompts: invalid public origin source")
        sources[sample_id] = []
    positions = {sample_id: index for index, sample_id in enumerate(sources)}
    schemas: dict[tuple[str, str, str], int] = {}
    seen: set[str] = set()
    previous = -1
    for origin in inventory["origins"]:
        sample_id, ref_id = origin["sample_id"], origin["ref_id"]
        if type(sample_id) is not str or sample_id not in sources or type(ref_id) is not str:
            raise ValueError("prompts: invalid public origin reference")
        match = re.fullmatch(re.escape(sample_id) + r":r([1-9][0-9]*)", ref_id)
        if (match is None or ref_id != sample_id + ":r" + str(len(sources[sample_id]) + 1)
                or ref_id in seen or positions[sample_id] < previous):
            raise ValueError("prompts: invalid public origin reference or order")
        previous = positions[sample_id]
        seen.add(ref_id)
        schema = (origin["kind"], origin["rule"], origin["locus"])
        if any(type(value) is not str for value in schema):
            raise ValueError("prompts: invalid public origin schema")
        if schema not in schemas:
            schemas[schema] = len(schemas)
        spans: list[int] = []
        for span in origin["occurrence_spans"]:
            start, end = span["start"], span["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end:
                raise ValueError("prompts: invalid public origin character span")
            spans.extend((start, end))
        sources[sample_id].append([schemas[schema], *spans])
    return {"version": PUBLIC_ORIGIN_DICTIONARY_VERSION, "columns": ["schema", "start,end,..."],
            "schemas": [list(schema) for schema in schemas],
            "sources": [{"sample_id": sample_id, "rows": rows} for sample_id, rows in sources.items()]}


def render_public_origin_dictionary(inventory: dict) -> str:
    """Use the same pinned table bytes in retry feedback and its final-history reconstruction."""
    return json.dumps(public_origin_dictionary(inventory), ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def diagnose_text(k: int, prompts: dict) -> str:
    if type(k) is not int or k not in (2, 3):
        raise ValueError("prompts: diagnosis requires a successful retry at turn 2 or 3")
    return _render(prompts["diagnose"], K=str(k), K_MINUS_1=str(k - 1))


def oracle_text(actor: ActorItem, fact_lines: tuple[str, ...], prompts: dict) -> str:
    validate_actor(actor)
    if not fact_lines or any(not isinstance(line, str) or not line.strip() for line in fact_lines):
        raise ValueError("prompts: oracle requires verified fact lines")
    return _render(prompts["oracle_facts"], FACT_LINES="\n".join(fact_lines),
                   P0_BODY=p0_body(actor.question, actor.choices))


def probe_text(observed_text: str, actor: ActorItem, prompts: dict) -> str:
    validate_actor(actor)
    if actor.choices or not isinstance(observed_text, str) or not observed_text.strip():
        raise ValueError("prompts: probe requires open question and nonempty observed text")
    return _render(prompts["self_perception_probe"], OBSERVED_TEXT=observed_text,
                   P0_BODY=p0_body(actor.question, actor.choices))


def user_message(text: str, n_images: int) -> dict:
    if not isinstance(text, str) or type(n_images) is not int or not 0 <= n_images <= 2:
        raise ValueError("prompts: text and 0..2 image placeholders required")
    return {"role": "user", "content": [*({"type": "image"} for _ in range(n_images)),
                                        {"type": "text", "text": text}]}
