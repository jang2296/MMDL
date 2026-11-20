"""Strict wire records for the v2 notebook-RL design; no model dependencies."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import UnionType
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

AnswerType = Literal["mcq", "int", "decimal", "fraction", "unit_number", "short_text", "list"]
Mode = Literal["obs", "plain"]
Split = Literal["train", "dev", "sealed", "style_sealed"]
Status = Literal["CORRECT", "INCORRECT", "INVALID_NO_FINAL", "INVALID_MULTIPLE", "TRUNCATED",
                 "UNVERIFIABLE", "CONTEXT_BLOCKED", "SYSTEM_ERROR"]
Cause = Literal["VISUAL_MISREAD", "VISUAL_RELATION", "NON_VISUAL", "FINALIZATION", "UNKNOWN"]
GroupKind = Literal["T1", "T2", "T3", "DIAG"]
Route = Literal["VISION_ON", "VISION_OFF"]


@dataclass(frozen=True)
class ActorItem:
    problem_id: str
    source: str
    family: str
    mode: Mode
    question: str
    choices: tuple[str, ...]
    answer_format: AnswerType
    image_paths: tuple[str, ...]
    image_sha256: tuple[str, ...]


@dataclass(frozen=True)
class VerifierItem:
    problem_id: str
    answer_type: AnswerType
    gold: str
    accept: dict
    choices: tuple[str, ...]
    slots: dict[str, str]
    slot_types: dict[str, str]
    required_slots: tuple[str, ...]
    oracle_fact_lines: tuple[str, ...]
    printed_text: tuple[str, ...]


@dataclass(frozen=True)
class Lineage:
    problem_id: str
    source: str
    source_revision: str
    license: str
    label_provenance: Literal["PROGRAM", "HUMAN"]
    image_provenance: str
    upstream_origin: str
    generator_version: str | None
    generator_seed: int | None
    template_id: str
    structure_hash: str
    split: Split
    parents: tuple[str, ...]
    benchmark_derived: bool = False
    model_touched: bool = False


@dataclass(frozen=True)
class SlotReport:
    parsed: int
    required: int
    required_correct: int
    wrong_required: tuple[str, ...]
    all_correct: int
    all_total: int
    conflicting: tuple[str, ...]
    false_relations: tuple[str, ...] = ()
    unverified_relations: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifierResult:
    status: Status
    extracted: str | None
    reward_answer: float
    slots: SlotReport | None
    verifier_version: str


@dataclass
class Sample:
    sample_id: str
    problem_id: str
    group_id: str
    kind: GroupKind
    turn: int
    parent_sample_id: str | None
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    sampler_logprobs: list[float]
    finish_reason: str
    text: str
    seed: int
    policy_version: int
    result: VerifierResult | None = None
    diag_reward: float | None = None
    reward: float = 0.0


@dataclass
class Group:
    group_id: str
    kind: GroupKind
    problem_id: str
    state_sha256: str
    samples: list[Sample]
    usable: bool
    reason: str | None


@dataclass
class ProblemTree:
    problem_id: str
    policy_version: int
    groups: list[Group]
    route: Route
    causes: dict[str, Cause]
    notebook: dict


@dataclass(frozen=True)
class DiagTarget:
    cause3: Literal["VISUAL", "NON_VISUAL", "FINALIZATION"]
    wrong_slots: tuple[str, ...]


@dataclass(frozen=True)
class OracleProbe:
    probe_id: str
    problem_id: str
    kind: Literal["oracle", "self_perception"]
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    status: Status
    policy_version: int


T = TypeVar("T")
_RECORDS = (ActorItem, VerifierItem, Lineage, SlotReport, VerifierResult, Sample, Group,
            ProblemTree, DiagTarget, OracleProbe)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _decode(hint: Any, value: Any, at: str) -> Any:
    origin, args = get_origin(hint), get_args(hint)
    if hint is Any:
        json.dumps(value, allow_nan=False)
        return value
    if origin in (UnionType, Union):
        for candidate in args:
            try:
                return _decode(candidate, value, at)
            except ValueError:
                pass
        raise ValueError(f"{at}: invalid union value")
    if origin is Literal:
        if not any(type(value) is type(x) and value == x for x in args):
            raise ValueError(f"{at}: invalid literal {value!r}")
        return value
    if hint in _RECORDS:
        if type(value) is not dict:
            raise ValueError(f"{at}: expected record object")
        names = {f.name for f in fields(hint)}
        if value.keys() - names:
            raise ValueError(f"{at}: unknown keys {sorted(value.keys() - names)}")
        if hint is SlotReport and names - value.keys() not in (
                set(), {"false_relations", "unverified_relations"}):
            raise ValueError(f"{at}: missing record fields")
        types = get_type_hints(hint)
        try:
            obj = hint(**{k: _decode(types[k], v, f"{at}.{k}") for k, v in value.items()})
        except TypeError as exc:
            raise ValueError(f"{at}: missing or invalid record fields") from exc
        _validate_record(obj)
        return obj
    if origin in (list, tuple):
        if type(value) not in (list, tuple):
            raise ValueError(f"{at}: expected array")
        result = [_decode(args[0], v, f"{at}[{i}]") for i, v in enumerate(value)]
        return tuple(result) if origin is tuple else result
    if origin is dict or hint is dict:
        if type(value) is not dict or any(type(k) is not str for k in value):
            raise ValueError(f"{at}: expected string-key object")
        return {k: _decode(args[1] if args else Any, v, f"{at}.{k}")
                for k, v in value.items()}
    if hint is float and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if hint in (str, int, bool, type(None)) and type(value) is hint:
        return value
    raise ValueError(f"{at}: invalid type or non-finite value")


def validate_actor(a: ActorItem) -> None:
    if type(a) is not ActorItem:
        raise ValueError("actor: expected ActorItem")
    if not a.question.strip() or not all(x.strip() for x in (a.problem_id, a.source, a.family)):
        raise ValueError("actor: empty question or identity")
    if not a.image_paths or len(a.image_paths) != len(a.image_sha256):
        raise ValueError("actor: image path/hash count mismatch")
    for path, digest in zip(a.image_paths, a.image_sha256, strict=True):
        p = PurePosixPath(path)
        if (not path or "\\" in path or ":" in path or p.is_absolute()
                or PureWindowsPath(path).drive or any(x in ("", ".", "..") for x in path.split("/"))):
            raise ValueError("actor: unsafe relative image path")
        if not _SHA256.fullmatch(digest):
            raise ValueError("actor: invalid image SHA256")
    if a.answer_format == "mcq":
        if not 2 <= len(a.choices) <= 10 or any(not x.strip() for x in a.choices):
            raise ValueError("actor: mcq requires 2..10 nonempty choices")
        if len(set(a.choices)) != len(a.choices):
            raise ValueError("actor: duplicate mcq choices")
    elif a.choices:
        raise ValueError("actor: choices require mcq answer_format")


def _validate_record(obj: Any) -> None:
    if isinstance(obj, ActorItem):
        validate_actor(obj)
    elif isinstance(obj, VerifierItem):
        if not obj.problem_id or not obj.gold.strip():
            raise ValueError("verifier: empty identity or gold")
        if set(obj.slots) != set(obj.slot_types) or not set(obj.required_slots) <= obj.slots.keys():
            raise ValueError("verifier: slot key mismatch")
        if len(set(obj.required_slots)) != len(obj.required_slots):
            raise ValueError("verifier: duplicate required slots")
        if not set(obj.slot_types.values()) <= {"number", "pitch", "text", "relation"}:
            raise ValueError("verifier: invalid slot type")
    elif isinstance(obj, SlotReport):
        from .observations import slot_id

        if not (0 <= obj.required_correct <= obj.required <= obj.all_total
                and 0 <= obj.all_correct <= obj.parsed <= obj.all_total
                and obj.required_correct <= obj.all_correct
                and len(obj.wrong_required) <= obj.required - obj.required_correct):
            raise ValueError("slots: inconsistent counts")
        groups = (obj.wrong_required, obj.conflicting, obj.false_relations, obj.unverified_relations)
        if any(len(set(group)) != len(group)
               or any(not key or slot_id(key) != key for key in group) for group in groups):
            raise ValueError("slots: IDs must be unique and normalized")
        if (set(obj.false_relations) & set(obj.unverified_relations)
                or set(obj.wrong_required) & (set(obj.false_relations) | set(obj.unverified_relations))):
            raise ValueError("slots: relation categories must be disjoint")
    elif isinstance(obj, VerifierResult):
        expected = 1.0 if obj.status == "CORRECT" else 0.0
        if obj.reward_answer != expected:
            raise ValueError("verifier result: status/reward mismatch")
        if obj.status in ("CORRECT", "INCORRECT") and not obj.extracted:
            raise ValueError("verifier result: scored answer missing")
    elif isinstance(obj, (Sample, OracleProbe)):
        if obj.policy_version < 0 or any(t < 0 for t in obj.prompt_token_ids + obj.completion_token_ids):
            raise ValueError("sample: negative version or token ID")
        if isinstance(obj, Sample):
            if len(obj.completion_token_ids) != len(obj.sampler_logprobs):
                raise ValueError("sample: token/logprob count mismatch")
            if obj.turn not in (1, 2, 3) or obj.seed < 0 or obj.finish_reason not in ("stop", "length"):
                raise ValueError("sample: invalid turn, seed or finish reason")
    elif isinstance(obj, Group):
        if not _SHA256.fullmatch(obj.state_sha256):
            raise ValueError("group: invalid state SHA256")
        if obj.usable and (len(obj.samples) < 2 or obj.reason is not None):
            raise ValueError("group: usable group requires >=2 samples and no failure reason")
        if not obj.usable and not obj.reason:
            raise ValueError("group: unusable group requires failure reason")
        if len({s.sample_id for s in obj.samples}) != len(obj.samples):
            raise ValueError("group: duplicate sample ID")
        for s in obj.samples:
            if (s.group_id, s.problem_id, s.kind) != (obj.group_id, obj.problem_id, obj.kind):
                raise ValueError("group: sample identity mismatch")
            if (s.prompt_token_ids, s.policy_version, s.parent_sample_id, s.turn) != (
                obj.samples[0].prompt_token_ids, obj.samples[0].policy_version,
                obj.samples[0].parent_sample_id, obj.samples[0].turn
            ):
                raise ValueError("group: mixed input state or policy")
    elif isinstance(obj, ProblemTree):
        for g in obj.groups:
            if g.problem_id != obj.problem_id or any(s.policy_version != obj.policy_version for s in g.samples):
                raise ValueError("tree: mixed problem or policy")
        if len({g.group_id for g in obj.groups}) != len(obj.groups):
            raise ValueError("tree: duplicate group ID")
        sample_ids = [s.sample_id for g in obj.groups for s in g.samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("tree: duplicate sample ID across groups")
    elif isinstance(obj, DiagTarget):
        if obj.cause3 != "VISUAL" and obj.wrong_slots:
            raise ValueError("diagnosis: nonvisual target cannot have wrong slots")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def from_json(cls: type[T], s: str) -> T:
    if cls not in _RECORDS:
        raise ValueError("unknown record class")
    try:
        value = json.loads(s, object_pairs_hook=_unique_pairs, parse_constant=_invalid_constant)
        return _decode(cls, value, cls.__name__)
    except (TypeError, OverflowError) as exc:
        raise ValueError("invalid JSON record") from exc


def to_json(obj: Any) -> str:
    if type(obj) not in _RECORDS:
        raise ValueError("unknown record instance")
    wire = json.dumps(asdict(obj), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
    normalized = from_json(type(obj), wire)
    return json.dumps(asdict(normalized), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def load_protocol(path: Path) -> dict:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or data.get("protocol_id") != "notebook-rl-v1":
        raise ValueError("invalid notebook RL protocol")
    if data.get("forbidden") != {"additional_sft": True, "teacher_or_upper_model_labels": True,
                                  "benchmark_training": True}:
        raise ValueError("protocol: forbidden training paths must remain forbidden")
    if data.get("initial_checkpoint", {}).get("revision") != "ebb281ec70b05090aa6165b016eac8ec08e71b17":
        raise ValueError("protocol: initial checkpoint must be pinned fresh F0")
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return data
