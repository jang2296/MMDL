"""Host-only credit for generated JSON facts; no correct values or teacher targets."""
from __future__ import annotations

import json

from .local_eval import digest
from .notebook_learning import _unique
from .notebook_objective import FACT_CREDIT_VERSION
from .observations import slot_id


def _bad_constant(value):
    raise ValueError("fact credit: nonfinite JSON")


def value_spans(text: str) -> dict[str, tuple[int, int]]:
    """Map canonical IDs to original JSON value literals, rejecting alias collisions."""
    decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_bad_constant)
    parsed = decoder.decode(text)
    if type(parsed) is not dict or type(parsed.get("corrected_facts")) is not dict:
        raise ValueError("fact credit: corrected_facts object required")

    def whitespace(index):
        while index < len(text) and text[index] in " \r\n\t":
            index += 1
        return index

    def members(start):
        index = whitespace(start)
        if text[index] != "{":
            raise ValueError("fact credit: object required")
        index = whitespace(index + 1)
        while text[index] != "}":
            key, end = decoder.raw_decode(text, index)
            colon = whitespace(end)
            if type(key) is not str or text[colon] != ":":
                raise ValueError("fact credit: invalid member")
            begin = whitespace(colon + 1)
            value, end = decoder.raw_decode(text, begin)
            yield key, value, begin, end
            index = whitespace(end)
            if text[index] == "}":
                break
            if text[index] != ",":
                raise ValueError("fact credit: invalid separator")
            index = whitespace(index + 1)

    fact_start = next(begin for key, _, begin, _ in members(0) if key == "corrected_facts")
    spans = {}
    for key, value, begin, end in members(fact_start):
        if type(value) is not str:
            raise ValueError("fact credit: fact values must be strings")
        key = slot_id(key)
        if key in spans:
            raise ValueError("fact credit: colliding normalized fact IDs")
        spans[key] = (begin, end)
    return spans


def token_spans(tokenizer, ids: list[int], text: str) -> list[tuple[int, int]] | None:
    """Abstain on unstable byte-token prefixes; full text disagreement is an error."""
    kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    if tokenizer.decode(ids, **kwargs) != text:
        raise ValueError("fact credit: completion tokens and text differ")
    spans, previous = [], ""
    for index in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:index], **kwargs)
        if not prefix.startswith(previous) or not text.startswith(prefix):
            return None
        spans.append((len(previous), len(prefix)))
        previous = prefix
    return spans


def group_credit(samples, scores: dict[str, dict], tokenizer) -> dict[str, dict]:
    """Full-group means precede MIS. Missing/invalid notes contribute zeros, not labels."""
    if len(samples) < 2 or len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("fact credit: complete unique group required")
    rows = [scores[s.sample_id] for s in samples]
    counts = {row.get("required_slot_count") for row in rows}
    if len(counts) != 1 or any(type(k) is not int or k < 0 for k in counts):
        raise ValueError("fact credit: fixed required-slot count required")
    required = counts.pop()
    assert isinstance(required, int)  # The strict type/count check above also excludes bool.
    for sample, row in zip(samples, rows, strict=True):
        facts = row.get("fact_rewards")
        if (row.get("objective_version") != FACT_CREDIT_VERSION
                or row.get("text_sha256") != digest(sample.text)
                or row.get("strict_reward") != sample.reward
                or type(facts) is not dict or len(facts) > required
                or any(type(k) is not str or k != slot_id(k) or type(v) is not int or v not in (0, 1)
                       for k, v in facts.items())):
            raise ValueError("fact credit: bound independently regraded scores required")
    keys = sorted({key for row in rows for key in row["fact_rewards"]})
    means = {key: sum(row["fact_rewards"].get(key, 0) for row in rows) / len(rows) for key in keys}
    strict_mean = sum(s.reward for s in samples) / len(samples)
    output = {}
    for sample, row in zip(samples, rows, strict=True):
        spans = token_spans(tokenizer, sample.completion_token_ids, sample.text)
        values = value_spans(sample.text) if row["fact_rewards"] else {}
        partial = [0.0] * len(sample.completion_token_ids)
        entries = []
        for key, reward in row["fact_rewards"].items():
            if key not in values:
                raise ValueError("fact credit: submitted fact span missing")
            start, end = values[key]
            indices = ([] if spans is None else [i for i, (a, b) in enumerate(spans)
                                                if start <= a < b <= end])
            advantage = reward - means[key]
            coefficient = .0625 * advantage / required
            for index in indices:
                partial[index] += coefficient
            entries.append({"slot": key, "reward": reward, "group_mean": means[key],
                "advantage": advantage, "coefficient": coefficient,
                "literal_span": [start, end], "token_indices": indices})
        output[sample.sample_id] = {
            "strict_advantage": sample.reward - strict_mean,
            "partial_token_advantages": partial, "facts": entries,
            "mapping_status": "UNSTABLE_TOKEN_PREFIX" if spans is None else
                              "MAPPED" if entries else "NO_ADMISSIBLE_FACTS",
            "token_ids_sha256": digest(sample.completion_token_ids), "text_sha256": digest(sample.text),
            "score_sha256": digest(row), "full_group_size": len(samples),
            "required_slot_count": required}
    return output
