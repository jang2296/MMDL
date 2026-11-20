"""Private deterministic rewards. This object must stay in the verifier process."""
from __future__ import annotations

import re
from pathlib import Path

from .answers import (RETRY_OUT_OF_CHOICE, TRAINING_ANSWER_CONTRACT, TURN1_EVAL_CONTRACT,
                      VERSION as ANSWER_VERSION, compare, compare_training, extract_eval_aligned,
                      extract_final, extract_training, is_ambiguous, mcq_out_of_choice)
from .observations import compare_slots, native_absent_relations, parse_observations, slot_id
from .repair_contract import parse_retry_sections, project_solution
from .schema import DiagTarget, Mode, VerifierResult, to_json
from .vault import read_actors, read_verifier

VERSION = ANSWER_VERSION + "+native-graph-relations-v3"
RETRY_OUT_OF_CHOICE_TAG = "+retry-out-of-choice-" + RETRY_OUT_OF_CHOICE


def diagnosis_fields(text: str) -> tuple[str, str] | None:
    """A single three-line note, not a search for whichever label earns reward."""
    match = re.fullmatch(
        r"\s*Cause:[ \t]*(VISUAL_MISREAD|VISUAL_RELATION|NON_VISUAL|FINALIZATION)[ \t]*\n"
        r"[ \t]*Wrong item:[ \t]*([^\n]+)\n[ \t]*Lesson:[ \t]*([^\n]+)[ \t]*", text)
    if match is None or not match[2].strip() or not match[3].strip():
        return None
    return ("VISUAL" if match[1].startswith("VISUAL_") else match[1], slot_id(match[2]))


class Verifier:
    def __init__(self, vault_root: Path, *, training_rules_path: Path | None = None,
                 training_rules_sha256: str | None = None):
        if (training_rules_path is None) != (training_rules_sha256 is None):
            raise ValueError("verifier: training rules path and SHA256 must be provided together")
        self.items = read_verifier(vault_root)
        self.actors = {actor.problem_id: actor for actor in read_actors(vault_root)}
        self.training_rules: dict[str, dict] = {}
        if training_rules_path is not None:
            assert training_rules_sha256 is not None
            from fine_tuning.error_note.answer_contract import load_training_rules
            from .vault import read_lineage

            self.training_rules = load_training_rules(training_rules_path, training_rules_sha256,
                vault_root=vault_root, actors=self.actors, items=self.items, lineages=read_lineage(vault_root))

    def trace(self, problem_id: str, text: str, finish_reason: str, *,
              response_contract: str | None = None) -> dict:
        from .reasoning_trace import check_trace
        if response_contract is None:
            return check_trace(actor=self.actors[problem_id], item=self.items[problem_id],
                               text=text, finish_reason=finish_reason)
        return check_trace(actor=self.actors[problem_id], item=self.items[problem_id],
                           text=text, finish_reason=finish_reason, response_contract=response_contract)

    def cause(self, problem_id: str, result: VerifierResult):
        from .attribution import attempt_cause
        return attempt_cause(result, self.items[problem_id])

    def diagnosis_target(self, problem_id: str, result: VerifierResult):
        from .attribution import diagnosis_target
        return diagnosis_target(result, self.items[problem_id])

    def route(self, problem_id: str, results: list[VerifierResult], rule: str):
        from .attribution import problem_route
        return problem_route(results, self.items[problem_id], rule)

    def grade(self, problem_id: str, text: str, finish_reason: str, mode: Mode, *,
              response_contract: str | None = None, answer_contract: str | None = None,
              retry_out_of_choice: str | None = None) -> VerifierResult:
        if mode not in ("obs", "plain"):
            raise ValueError("verifier: unsupported observation mode")
        if answer_contract is not None and (answer_contract not in (TURN1_EVAL_CONTRACT, TRAINING_ANSWER_CONTRACT)
                                            or response_contract is not None):
            raise ValueError("verifier: answer contracts grade only an unwrapped attempt")
        if retry_out_of_choice is not None and (retry_out_of_choice != RETRY_OUT_OF_CHOICE or response_contract is None):
            raise ValueError("verifier: the out-of-choice rule grades only a typed retry")
        version = VERSION
        if response_contract is not None:
            sections = parse_retry_sections(text, contract_version=response_contract)
            text = project_solution(text, sections)
            version += "+" + response_contract
        if retry_out_of_choice is not None:
            version += RETRY_OUT_OF_CHOICE_TAG
        item = self.items[problem_id]
        if answer_contract is None:
            extracted = extract_final(text, finish_reason)
        else:
            version += "+" + answer_contract
            extractor = extract_training if answer_contract == TRAINING_ANSWER_CONTRACT else extract_eval_aligned
            extracted = extractor(text, finish_reason, self.actors[problem_id].question, item.choices, item.answer_type)
        slots = compare_slots(parse_observations(text), item) if mode == "obs" and item.slots else None
        if extracted.status != "OK":
            result = VerifierResult(extracted.status, extracted.answer, 0.0, slots, version)
        elif extracted.answer is None:
            raise ValueError("verifier: extractor returned OK without answer")
        elif answer_contract is None and is_ambiguous(extracted.answer, item):
            result = VerifierResult("INVALID_MULTIPLE", extracted.answer, 0.0, slots, version)
        else:
            matched = (compare_training(extracted.answer, item, self.training_rules.get(problem_id))
                       if answer_contract == TRAINING_ANSWER_CONTRACT else compare(extracted.answer, item))
            if matched is None and answer_contract == TURN1_EVAL_CONTRACT and compare(item.gold, item) is not None:
                # V8 committed exactly one answer; one the typed comparison cannot match is a mismatch,
                # as evaluation scores it. An item whose own gold is uncomparable stays UNVERIFIABLE.
                matched = False
            if (matched is None and retry_out_of_choice is not None and item.answer_type == "mcq"
                    and compare(item.gold, item) is not None and mcq_out_of_choice(extracted.answer, item.choices)):
                # One committed retry answer naming no option cannot be the gold option; V8 scores it wrong.
                matched = False
            result = VerifierResult("UNVERIFIABLE" if matched is None else "CORRECT" if matched else "INCORRECT",
                                    extracted.answer, float(matched is True), slots, version)
        to_json(result)
        return result

    def grade_diagnosis(self, problem_id: str, text: str, target: DiagTarget) -> float:
        item = self.items[problem_id]
        to_json(target)
        extra = {slot_id(k) for k in target.wrong_slots} - {slot_id(k) for k in item.required_slots}
        if extra != set(native_absent_relations(extra, item)):
            raise ValueError("verifier: diagnosis target contains unverified observation slots")
        fields = diagnosis_fields(text)
        if fields is None or fields[0] != target.cause3:
            return 0.0
        if target.cause3 != "VISUAL" and fields[1] != "none":
            return 0.0
        return 1.0 + 0.25 * (target.cause3 == "VISUAL" and fields[1] in {
            slot_id(k) for k in target.wrong_slots})


def answer_reward(result: VerifierResult, *, visual_lambda: float = 0.0) -> float:
    """Optional S4 bonus is bounded and cannot reward facts-only or invalid answers."""
    to_json(result)
    if type(visual_lambda) not in (float, int) or not 0 <= visual_lambda <= 0.1:
        raise ValueError("verifier: visual coefficient must be in [0,0.1]")
    if result.status in ("UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR"):
        raise ValueError("verifier: this result must be excluded, not assigned a training reward")
    if result.status not in ("CORRECT", "INCORRECT") or result.slots is None or not result.slots.required:
        return result.reward_answer
    return result.reward_answer + visual_lambda * result.slots.required_correct / result.slots.required
