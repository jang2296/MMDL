"""Deterministic ranking and ordering of upstream-verified v16b candidates.

Scores and subject tags are selection heuristics, not verified task difficulty or
evidence of transfer to an MMMU subject. Eligibility and receipts belong upstream.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import re
from typing import Any

VERSION = "v16b-curation-policy-v2"
POLICY_CHANGES = (
    "v2: recognize plural/below visual references and simple ScienceQA region maps",
    "v2: remove lecture-length reasoning credit; detect nested CLEVR relations",
    "v2: support strict source-bound F0_PREUPDATE records and calibrated source caps",
)
AI2D = "hf-ai2d"
# Subject tags describe a coarse skill overlap; ChartQA does not certify finance knowledge.
SOURCE_PROFILE = {
    "hf-ai2d": ("diagram", [], 12),
    "hf-chartqa": ("chart", ["Math"], 12),
    "hf-scienceqa": ("diagram_or_photo", [], 4),
    "hf-geometry3k": ("geometry", ["Math"], 14),
    "hf-gvlqa": ("graph", ["Computer_Science", "Math"], 10),
    "hf-ssmr": ("sheet_music", ["Music"], 10),
    "hf-mapqa": ("map", ["Geography"], 0),
    "hf-nctcrc": ("histology", ["Diagnostics_and_Laboratory_Medicine"], -10),
    "hf-ptbxl": ("electrocardiogram", ["Clinical_Medicine"], 8),
    "hf-slake": ("medical_image", ["Clinical_Medicine"], 6),
    "hf-chebi-role": ("chemical_structure", ["Chemistry"], 8),
    "hf-plantvillage": ("plant_photo", ["Agriculture"], -14),
    "hf-deepweeds": ("plant_photo", ["Agriculture"], -14),
    "hf-clevr": ("scene", [], -8),
    "hf-tqa": ("diagram", [], 10),
}
VISUAL_SOURCES = set(SOURCE_PROFILE) - {"hf-ai2d", "hf-scienceqa", "hf-tqa"}
SIMPLE_PHOTO_SOURCES = {"hf-nctcrc", "hf-plantvillage", "hf-deepweeds"}
SUBJECT_WORDS = {
    "biology": "Biology", "chemistry": "Chemistry", "physics": "Physics",
    "earth science": "Geography", "geography": "Geography", "geology": "Geography",
    "mathematics": "Math", "math": "Math", "music": "Music",
    "computer science": "Computer_Science", "agriculture": "Agriculture",
}
VISUAL_REFERENCE = re.compile(
    r"\b(?:in (?:the |this )?(?:figures?|images?|pictures?|graphs?|tables?|diagrams?|maps?)|"
    r"(?:figures?|diagrams?|tables?|graphs?|charts?) (?:below|above)|"
    r"(?:figures?|diagrams?) (?:below )?show|"
    r"shown|according to (?:the |this )?(?:chart|graph|table|map)|"
    r"labeled|labelled|label [a-z]|shaded|highlighted)\b"
)
OPERATIONS = re.compile(
    r"\b(?:sum|difference|ratio|percentage|percent|average|mean of|probability|"
    r"slope|area|perimeter|volume|distance|shortest path|compare|increase|decrease)\b"
)
ORDER_RULE = {
    "version": "v16b-joint-stratum-deficit-order-v2",
    "strata": ["source", "f0_label", "annotation_label", "long_risk_label"],
    "within_stratum": "ascending sha256(\'v16b-order:\'+problem_id), then problem_id",
    "choice": "minimum maximum absolute marginal deficit, then squared deficits, then within-stratum hash",
    "prefix_constraint": "each marginal label within 2 items at every batch boundary including final partial batch",
    "annotation": "explicit has_annotation bool denotes supplied native reference presence; otherwise unknown",
    "long_risk": "SSMR or ChEBI source, or strict F0 length/8192 hit; no only with a measured nonlong F0; otherwise unknown",
    "seed": "recorded provenance; this deterministic rule uses no RNG or seed search",
    "unknown_prior": "unknown stays a separate label; no invented F0 correctness or length",
    "last_batch": "keep remaining items without padding or dropping",
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _identity(candidate: dict) -> tuple[str, str, str]:
    values = tuple(candidate.get(key) for key in ("problem_id", "source", "family_id"))
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("candidate requires nonempty problem_id, source and family_id")
    return values  # type: ignore[return-value]


def _prior(candidate: dict) -> dict | None:
    prior = _mapping(candidate.get("f0_prior"))
    if prior.get("policy_kind") == "F0" and type(prior.get("correct")) is bool:
        correct = prior["correct"]
    elif (prior.get("model_class") == "F0_PREUPDATE"
          and prior.get("binding") == "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD"
          and prior.get("status") in {"CORRECT", "INCORRECT", "TRUNCATED",
                                      "INVALID_NO_FINAL", "INVALID_MULTIPLE"}):
        correct = prior["status"] == "CORRECT"
    else:
        return None
    if (type(prior.get("completion_tokens")) is not int or prior["completion_tokens"] < 0
            or not isinstance(prior.get("finish_reason"), str) or not prior["finish_reason"]):
        return None
    return prior | {"correct": correct}


def _level(metadata: dict) -> str:
    grade = str(metadata.get("grade", "")).lower().replace("grade", "").strip()
    if grade.isdigit():
        number = int(grade)
        return "elementary_middle" if number <= 8 else "high_school" if number <= 12 else "college"
    if grade in {"college", "university", "undergraduate", "graduate"}:
        return "college"
    if grade in {"high school", "high_school"}:
        return "high_school"
    return "unknown"


def feature_candidate(candidate: dict) -> dict:
    """Use native candidate fields only; missing facts receive no invented credit."""
    source = str(candidate.get("source", ""))
    actor, metadata = _mapping(candidate.get("actor")), _mapping(candidate.get("native_metadata"))
    overrides = _mapping(candidate.get("review_overrides"))
    disabled = candidate.get("disabled_rules") or []
    if not isinstance(disabled, (list, tuple)) or any(not isinstance(rule, str) for rule in disabled):
        raise ValueError("disabled_rules must be a list of rule names")
    question = str(actor.get("question", "")).lower()
    image_type, subjects, relevance = SOURCE_PROFILE.get(source, ("unknown", [], 0))
    subjects = list(subjects)
    native_text = " ".join(str(metadata.get(key, "")) for key in
                           ("subject", "topic", "category", "skill")).lower().replace("_", " ")
    for word, subject in SUBJECT_WORDS.items():
        if re.search(r"\b" + re.escape(word) + r"\b", native_text):
            subjects.append(subject)
    reasons = ["SUBJECT_MAPPING_HEURISTIC"]
    region_map = source == "hf-scienceqa" and (
        re.search(r"\bmap\b", question) or (re.search(r"\b(?:colony|colonies|continents?)\b", question)
                                     and re.search(r"\b(?:shown|highlighted|shaded|named|labeled|labelled)\b", question))
    )
    if region_map:
        image_type = "map"
        subjects.append("Geography")
        reasons.append("SCIENCEQA_REGION_MAP_HEURISTIC")
    image_needed: bool | None = None
    if VISUAL_REFERENCE.search(question):
        image_needed = True
        reasons.append("VISUAL_REFERENCE_HEURISTIC")
    elif source in VISUAL_SOURCES:
        image_needed = True
        reasons.append("VISUAL_TASK_SOURCE_HEURISTIC")
    else:
        reasons.append("VISUAL_NECESSITY_UNKNOWN")
    if "image_needed" in disabled:
        image_needed = None
        reasons.append("IMAGE_NEEDED_RULE_DISABLED_BY_CALIBRATION")
    if "image_needed" in overrides:
        image_needed = overrides["image_needed"]
        if image_needed is not None and type(image_needed) is not bool:
            raise ValueError("review image_needed must be bool or null")
        reasons.append("REVIEW_IMAGE_NEEDED")

    human = _mapping(candidate.get("human_solution"))
    solution = human.get("text")
    if not isinstance(solution, str) or not solution.strip():
        solution = "\n".join(human[key] for key in ("lecture", "solution")
                             if isinstance(human.get(key), str))
    solution_words = len(solution.split())
    operations = set(OPERATIONS.findall(question))
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", question)
    reasoning = 1
    if (operations or source in {"hf-geometry3k", "hf-ssmr"}
            or ("calculate" in question and numbers)):
        reasoning = 2
    if (source == "hf-clevr"
            and len(re.findall(r"\b(?:left|right|behind|in front of)\b", question)) >= 2
            and re.search(r"\b(?:compare|same|count|how many|number|more|fewer|less|larger|smaller)\b", question)):
        reasoning = 2
        reasons.append("NESTED_SCENE_RELATIONS_HEURISTIC")
    if len(operations) >= 2 and len(numbers) >= 2 and re.search(r"\b(?:then|after|if|using)\b", question):
        reasoning = 4
    reasons.append("REASONING_DEPTH_HEURISTIC")
    if "reasoning_depth" in disabled or "reasoning_steps" in disabled:
        reasoning = 1
        reasons.append("REASONING_DEPTH_RULE_DISABLED_BY_CALIBRATION")
    if "reasoning_steps" in overrides:
        reasoning = overrides["reasoning_steps"]
        if type(reasoning) is not int or reasoning not in {1, 2, 3, 4}:
            raise ValueError("review reasoning_steps must be 1, 2, 3 or 4 (4 means 4+)")
        reasons.append("REVIEW_REASONING_STEPS")
    inferred_level = _level(metadata)
    level_kind = "native_metadata" if inferred_level != "unknown" else "unknown"
    if source == "hf-clevr" and inferred_level == "unknown":
        inferred_level, level_kind = "elementary_middle", "source_heuristic"
    level = overrides.get("level", inferred_level)
    if "level" in overrides:
        level_kind = "review_override"
    if level not in {"unknown", "elementary_middle", "high_school", "college"}:
        raise ValueError("unsupported review level")
    references = candidate.get("annotation_refs")
    has_annotation = isinstance(references, list) and any(isinstance(ref, dict) and ref for ref in references)
    annotation_support = candidate.get("annotation_verifier_supported")
    if type(annotation_support) is not bool:
        explicit = [ref["verifier_supported"] for ref in references or []
                    if isinstance(ref, dict) and type(ref.get("verifier_supported")) is bool]
        annotation_support = any(explicit) if explicit else None
    choices = actor.get("choices")
    n_options = len(choices) if isinstance(choices, (list, dict)) else 0
    prior = _prior(candidate)
    prior_score, prior_label = 0, "unknown"
    if prior is None:
        reasons.append("NO_VERIFIED_F0_PRIOR")
    else:
        prior_label = "correct" if prior["correct"] else "wrong"
        tokens = prior.get("completion_tokens")
        if prior.get("finish_reason") in {"length", "max_tokens", "LOOP", "FORMAT_DRIFT"}:
            prior_score = 5
            reasons.append("F0_TERMINATION_TARGET")
        elif not prior["correct"]:
            prior_score = 14
            reasons.append("F0_WRONG_CORRECTION_OPPORTUNITY")
        elif type(tokens) is int and tokens >= 256:
            prior_score = 4
            reasons.append("F0_CORRECT_LONG")
        elif type(tokens) is int and 0 <= tokens <= 64 and reasoning == 1:
            prior_score = -8
            reasons.append("F0_CORRECT_SHORT_SINGLE_STEP")
    low_priority = -8 if source in SIMPLE_PHOTO_SOURCES else 0
    if source == "hf-mapqa" and re.search(r"\b(?:legend|symbol|color|colour)\b", question) and reasoning == 1:
        low_priority -= 12
        reasons.append("SINGLE_STEP_MAP_LEGEND")
    if region_map and reasoning == 1 and re.search(
            r"\b(?:name|named|identify|which colony|which continent|highlighted|shown)\b", question):
        low_priority -= 12
        reasons.append("SINGLE_STEP_NAMED_REGION")
    if source in SIMPLE_PHOTO_SOURCES:
        reasons.append("SINGLE_STEP_PHOTO_SOURCE")
    if source == "hf-scienceqa" and image_needed is False:
        reasons.append("REVIEWED_DECORATIVE_SCIENCEQA")
    components = {
        "visual_necessity": 30 if image_needed is True else -20 if image_needed is False else 0,
        "reasoning_depth": 20 if reasoning >= 4 else 12 if reasoning >= 2 else 0,
        "metadata_level": {"college": 10, "high_school": 6}.get(level, 0),
        "annotation_reference": 12 if has_annotation else 0,
        "human_solution": 10 if solution.strip() else 0,
        "f0_prior": prior_score,
        "source_skill_priority": relevance,
        "single_step_priority": low_priority,
        "original_option_format": 3 if n_options >= 8 else 1 if n_options >= 4 else 0,
    }
    return {
        "policy_version": VERSION, "mmmu_subjects": sorted(set(subjects)),
        "subject_mapping_kind": "heuristic_skill_overlap_not_domain_coverage",
        "image_type": image_type, "image_needed": image_needed,
        "reasoning_steps": reasoning,
        "reasoning_kind": ("review_override" if "reasoning_steps" in overrides
                           else "unknown_disabled_heuristic" if {"reasoning_depth", "reasoning_steps"} & set(disabled)
                           else "heuristic"),
        "level": level, "level_kind": level_kind, "n_options": n_options,
        "answer_type": _mapping(candidate.get("verifier")).get("answer_type"),
        "has_annotation": has_annotation, "annotation_exists": has_annotation,
        "annotation_verifier_supported": annotation_support,
        "has_human_solution": bool(solution.strip()),
        "human_solution_word_count": solution_words,
        "f0_label": prior_label, "score": sum(components.values()),
        "score_components": components, "score_reason_codes": reasons,
        "disabled_rules": list(disabled),
    }


def choose(candidates: list[dict], total: int = 3000, source_cap: int = 450,
           source_caps: dict[str, int] | None = None) -> tuple[list[dict], list[dict]]:
    """Reserve eligible AI2D, then take the highest scores within source/family caps.

    Upstream eligibility is never relaxed to fill the requested count. Soft
    annotation/domain targets do not force low-value source quotas.
    """
    if type(total) is not int or total <= 0 or type(source_cap) is not int or source_cap <= 0:
        raise ValueError("total and source_cap must be positive integers")
    caps = {} if source_caps is None else source_caps
    if (not isinstance(caps, dict) or any(not isinstance(source, str) or not source
            or type(cap) is not int or not 0 < cap <= source_cap for source, cap in caps.items())):
        raise ValueError("source_caps must map source names to positive integers <= source_cap")
    enriched, seen = [], set()
    for candidate in candidates:
        pid, _, _ = _identity(candidate)
        if pid in seen:
            raise ValueError(f"duplicate input problem_id: {pid}")
        seen.add(pid)
        enriched.append(candidate | feature_candidate(candidate))
    eligible = [item for item in enriched if item.get("eligible") is True and not item.get("exclusion_reasons")]
    ranked = sorted(eligible, key=lambda item: (
        item["source"] != AI2D, -item["score"], _sha("v16b-select:" + item["problem_id"]), item["problem_id"],
    ))
    selected: list[dict] = []
    used_families: set[str] = set()
    counts: Counter[str] = Counter()
    for item in ranked:
        if item["family_id"] in used_families:
            continue
        if item["source"] != AI2D and (counts[item["source"]] >= caps.get(item["source"], source_cap) or len(selected) >= total):
            continue
        selected.append(item)
        used_families.add(item["family_id"])
        counts[item["source"]] += 1
    if counts[AI2D] > total:
        raise ValueError(f"mandatory eligible AI2D exceeds total: {counts[AI2D]} > {total}")
    if len(selected) != total:
        raise ValueError(f"insufficient eligible distinct families within source caps: {len(selected)} < {total}")
    selected_ids = {item["problem_id"] for item in selected}
    decisions = []
    for item in sorted(enriched, key=lambda item: item["problem_id"]):
        included = item["problem_id"] in selected_ids
        if included:
            codes = ["MANDATORY_ELIGIBLE_AI2D" if item["source"] == AI2D else "SELECTED_BY_SCORE"]
        elif item.get("eligible") is not True or item.get("exclusion_reasons"):
            codes = list(item.get("exclusion_reasons") or ["UPSTREAM_INELIGIBLE"])
        elif item["family_id"] in used_families:
            codes = ["DUPLICATE_FAMILY"]
        elif counts[item["source"]] >= caps.get(item["source"], source_cap):
            codes = ["SOURCE_CAP"]
        else:
            codes = ["BELOW_SELECTION_CUTOFF"]
        decisions.append({key: item[key] for key in (
            "problem_id", "source", "family_id", "score", "score_components", "score_reason_codes",
        )} | {"selected": included, "reason_codes": codes})
    return selected, decisions


def interleave(selected: list[dict], seed: int = 3407, batch_size: int = 16) -> dict:
    """Balance source/F0/reference/long-risk marginals without changing the selected population."""
    if type(seed) is not int or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("seed must be an integer and batch_size a positive integer")
    ids, families = set(), set()
    groups: dict[tuple[str, ...], list[dict]] = {}
    hashes: dict[str, str] = {}
    fields = ("source", "f0_label", "annotation_label", "long_risk_label")
    for item in selected:
        pid, source, family = _identity(item)
        if pid in ids or family in families:
            raise ValueError("order requires unique problem IDs and families")
        ids.add(pid)
        families.add(family)
        prior = _prior(item)
        f0 = "unknown" if prior is None else "correct" if prior["correct"] else "wrong"
        annotation = item.get("has_annotation")
        annotation_label = "present" if annotation is True else "absent" if annotation is False else "unknown"
        long = source in {"hf-ssmr", "hf-chebi-role"} or (
            prior is not None and (prior["finish_reason"] == "length" or prior["completion_tokens"] >= 8192))
        long_label = "yes" if long else "no" if prior is not None else "unknown"
        key: tuple[str, ...] = (source, f0, annotation_label, long_label)
        groups.setdefault(key, []).append(dict(zip(fields, key)) | {"problem_id": pid})
        hashes[pid] = _sha("v16b-order:" + pid)
    total = len(selected)
    if not total:
        raise ValueError("cannot order an empty selection")
    queues = {key: deque(sorted(items, key=lambda item: (hashes[item["problem_id"]], item["problem_id"])))
              for key, items in sorted(groups.items())}
    cells = sorted({(dimension, label) for key in queues for dimension, label in enumerate(key)})
    cell_index = {cell: index for index, cell in enumerate(cells)}
    indexes = {key: tuple(cell_index[dimension, label] for dimension, label in enumerate(key)) for key in queues}
    populations = [0] * len(cells)
    for key, queue in queues.items():
        for index in indexes[key]:
            populations[index] += len(queue)
    taken = [0] * len(cells)
    ordered: list[dict] = []
    for position in range(1, total + 1):
        errors = [taken[index] * total - position * population for index, population in enumerate(populations)]

        def cost(key: tuple[str, ...]) -> tuple:
            after = errors.copy()
            for index in indexes[key]:
                after[index] += total
            pid = queues[key][0]["problem_id"]
            return max(abs(error) for error in after), sum(error * error for error in after), hashes[pid], pid

        chosen = min((key for key, queue in queues.items() if queue), key=cost)
        ordered.append(queues[chosen].popleft())
        for index in indexes[chosen]:
            taken[index] += 1

    dimensions = ("source", "f0_label", "annotation", "long_risk")
    stats: dict[str, dict[str, Any]] = {
        dimension: {
            "population_counts": {label: populations[index] for index, (dim, label) in enumerate(cells) if dim == number},
            "max_abs_deviation": 0.0, "worst_prefix_items": 0, "violations": [],
        } for number, dimension in enumerate(dimensions)
    }
    batches: list[dict] = []
    prefix_counts: Counter[tuple[int, str]] = Counter()
    for start in range(0, total, batch_size):
        batch = ordered[start:start + batch_size]
        end = start + len(batch)
        for item in batch:
            for dimension, field in enumerate(fields):
                prefix_counts[dimension, item[field]] += 1
        for number, dimension_name in enumerate(dimensions):
            for label, population in stats[dimension_name]["population_counts"].items():
                delta = abs(prefix_counts[number, label] * total - end * population)
                deviation = delta / total
                if deviation > stats[dimension_name]["max_abs_deviation"]:
                    stats[dimension_name]["max_abs_deviation"] = deviation
                    stats[dimension_name]["worst_prefix_items"] = end
                if delta > 2 * total:
                    stats[dimension_name]["violations"].append({
                        "prefix_items": end, "label": label, "abs_deviation": deviation})
        labels = Counter(item["f0_label"] for item in batch)
        batches.append({
            "batch_index": len(batches), "size": len(batch), "items": batch,
            "problem_ids": [item["problem_id"] for item in batch],
            "source_counts": dict(sorted(Counter(item["source"] for item in batch).items())),
            "prior_counts": {label: labels[label] for label in ("correct", "wrong", "unknown")},
            "prior_balance_status": "MIXED_KNOWN_F0" if labels["correct"] and labels["wrong"] else "NO_BOTH_KNOWN_F0_LABELS_IN_BATCH",
        })
    return {
        "version": ORDER_RULE["version"], "seed": seed, "batch_size": batch_size,
        "rule": dict(ORDER_RULE), "rule_sha256": _json_sha(ORDER_RULE | {"seed": seed, "batch_size": batch_size}),
        "input_ids_sha256": _json_sha(sorted(ids)),
        "ids_sha256": _json_sha([item["problem_id"] for item in ordered]),
        "count": total, "items": ordered, "batches": batches,
        "f0_prior_counts": dict(sorted(Counter(item["f0_label"] for item in ordered).items())),
        "prefix_balance": {
            "status": "FAIL" if any(stat["violations"] for stat in stats.values()) else "PASS",
            "tolerance_items": 2, "checked_prefixes": len(batches), "dimensions": stats,
            "infeasibility_proven": False,
        },
    }
