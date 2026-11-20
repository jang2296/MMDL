"""Private, CPU-only native-data curation; prepare a draft, then freeze reviewed data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import curation_policy_v16b as policy

SCHEMA = "v16b-native-curation-v1"
INDEX_PATH = Path.home() / "mmdl-artifacts/notebook-rl-smoke-20261017/exclusion-v2/index.json"
INDEX_SHA256 = "82b1084d24bfd9bde5c287a15d5b5c2a1b493776597b6dd25cfb0b4e8526ed09"
AI2D_RIGHTS = "AI2D-noncommercial-academic; no public release; no third-party distribution"
SHORT_TEXT_SOURCES = {"hf-chartqa", "hf-mapqa", "hf-clevr", "hf-slake"}


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path) -> dict:
    return {"path": str(path.absolute()), "sha256": _sha(path)}


def _check_ref(ref: dict, checked: dict[str, str]) -> Path:
    path, expected = Path(ref["path"]), ref.get("sha256")
    if not path.is_absolute() or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid absolute path/SHA256 reference")
    actual = checked.setdefault(str(path), "")
    if not actual:
        actual = checked[str(path)] = _sha(path)
    if actual != expected:
        raise ValueError(f"reference hash mismatch: {path}")
    return path


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        result = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in result):
        raise ValueError(f"JSONL object required: {path}")
    return result


def _by_id(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        pid = row.get("problem_id")
        if not isinstance(pid, str) or not pid or pid in result:
            raise ValueError("missing or duplicate problem_id")
        result[pid] = row
    return result


def _write(path: Path, value: object, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        if not isinstance(value, list):
            raise ValueError("JSONL payload must be a list")
        payload = b"".join(_bytes(row) + b"\n" for row in value)
    else:
        payload = _bytes(value) + b"\n"
    with path.open("xb") as handle:
        path.chmod(0o600)
        handle.write(payload)


def gradable(candidate: dict) -> tuple[bool, str]:
    """Admit existing verifier targets, without inventing or changing accepted answers."""
    verifier = candidate.get("verifier", {})
    kind, gold = verifier.get("answer_type"), verifier.get("gold")
    if not isinstance(gold, str) or not gold.strip() or not isinstance(verifier.get("accept"), dict):
        return False, "missing_native_gold_or_accept"
    if kind == "mcq":
        choices = verifier.get("choices")
        valid = isinstance(choices, list) and 2 <= len(choices) <= 26
        valid = valid and all(isinstance(choice, str) and choice.strip() for choice in choices)
        return bool(valid and gold in [chr(65 + i) for i in range(len(choices or []))]), "native_mcq_key"
    if kind in {"int", "decimal"}:
        try:
            number = Decimal(gold.replace(",", ""))
            valid = number.is_finite() and (kind != "int" or number == number.to_integral_value())
        except InvalidOperation:
            valid = False
        return valid, "native_finite_numeric"
    if kind == "short_text" and candidate.get("source") == "hf-slake":
        question = " ".join(candidate.get("actor", {}).get("question", "").casefold().split())
        if "how to prevent" in question:
            return False, "slake_unbounded_prevention_advice"
        if question.rstrip("?. ") == "how much damage will be caused to the body taking such an image":
            return False, "slake_unbounded_damage_question"
        if "organ system" in question and gold.casefold().strip() in {"head", "pelvic cavity"}:
            return False, "slake_anatomical_category_mismatch"
    if kind == "short_text" and candidate.get("source") in SHORT_TEXT_SOURCES:
        # shortcut: finite-label source and bounded answer, replace with native vocabularies when available.
        valid = len(gold) <= 160 and len(gold.split()) <= 12 and "\n" not in gold
        return valid, "heuristic_supported_existing_verifier"
    return False, "unsupported_free_form_target"


def _hard_reasons(candidate: dict) -> list[str]:
    reasons = list(candidate.get("exclusion_reasons", []))
    lineage, manifest = candidate.get("lineage", {}), candidate.get("manifest_item", {})
    if candidate.get("historical_bindings_status") != "PASS_RECORDS_AND_IMAGES_RECHECKED" or candidate.get("historical_allowlisted") is not True:
        reasons.append("NO_RECEIPT")
    if lineage.get("split") != "train" or manifest.get("split") != "train":
        reasons.append("NONTRAIN_SPLIT")
    if lineage.get("benchmark_derived") is not False or lineage.get("model_touched") is not False:
        reasons.append("TEACHER_OR_BENCHMARK_FIELD")
    if lineage.get("label_provenance") not in {"HUMAN", "PROGRAM"} or not lineage.get("upstream_origin"):
        reasons.append("UNKNOWN_PROVENANCE")
    if not lineage.get("license") or not lineage.get("image_provenance"):
        reasons.append("RIGHTS_OR_IMAGE_PROVENANCE_MISSING")
    if not candidate.get("image_paths") or not (len(candidate["image_paths"]) == len(candidate.get("image_sha256", [])) == len(candidate.get("pixel_sha256", []))):
        reasons.append("NO_IMAGE")
    for key in ("problem_id", "source", "source_revision", "structure_hash"):
        if not candidate.get(key) or candidate.get(key) != lineage.get(key):
            reasons.append("NATIVE_IDENTITY_MISMATCH")
    for key in ("actor", "verifier", "lineage"):
        if candidate.get("record_sha256", {}).get(key) != _digest(candidate.get(key)):
            reasons.append("RECORD_HASH_MISMATCH")
        if not candidate.get("record_paths", {}).get(key):
            reasons.append("NO_NATIVE_RECORD_REFERENCE")
    valid, _ = gradable(candidate)
    if not valid:
        reasons.append("UNGRADABLE")
    return sorted(set(reasons))


def _strings(value: object, prefix: str):
    if isinstance(value, str):
        if value.strip():
            yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _strings(value[key], f"{prefix}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _strings(child, f"{prefix}[{i}]")


def scan_new_texts(candidates: list[dict], cache: dict, matcher) -> dict:
    """All new textual leaves are checked; every match quarantines the full family."""
    fields, matches, quarantined = [], [], set()
    memo: dict[str, list[dict]] = {}
    for candidate in candidates:
        payload = {key: candidate.get(key) for key in ("native_metadata", "raw_annotation")}
        solution = candidate.get("human_solution")
        if solution:
            payload["human_solution"] = {key: solution.get(key) for key in ("lecture", "solution", "text")}
        for field, text in _strings(payload, "new_native"):
            text_sha = hashlib.sha256(text.encode()).hexdigest()
            if text_sha not in memo:
                memo[text_sha] = matcher(text, "explanation", cache)
            evidence = {"problem_id": candidate["problem_id"], "family_id": candidate["family_id"],
                        "field": field, "text_sha256": text_sha, "matches": memo[text_sha]}
            fields.append(evidence)
            if evidence["matches"]:
                matches.append(evidence)
                quarantined.add(candidate["family_id"])
    return {"schema": SCHEMA, "status": "COMPLETE_FRESH_TEXT_SCAN_WITH_LIMITS",
            "index_sha256": INDEX_SHA256, "field_count": len(fields),
            "unique_text_count": len(memo), "fields": fields, "matches": matches,
            "quarantined_families": sorted(quarantined), "thresholds_changed": False,
            "adjudication": "all_matches_unresolved_and_excluded_without_autoapproval",
            "human_verified": False}


def prepare(*, audit_summary_path: Path, enrichment_path: Path, output: Path) -> dict:
    if (output / "prepared.json").exists() or (output / "candidates.jsonl").exists():
        raise ValueError("prepare output already exists")
    checked: dict[str, str] = {}
    summary = json.loads(audit_summary_path.read_text())
    if summary.get("status") != "PASS_HISTORICAL_BINDINGS_RECHECKED":
        raise ValueError("historical audit did not pass")
    candidate_path = _check_ref(summary, checked)
    audit_ref = {"path": summary["audit_path"], "sha256": summary["audit_sha256"]}
    audit_path = _check_ref(audit_ref, checked)
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != summary["status"] or audit.get("benchmark_used_only_for_exclusion") is not True:
        raise ValueError("invalid historical audit scope")
    holdout_ref = audit["artifacts"]["holdouts"]
    _check_ref(holdout_ref, checked)
    raw_candidates = _by_id(_rows(candidate_path))
    enrichment = _by_id(_rows(enrichment_path))
    if len(raw_candidates) != summary["count"] or set(raw_candidates) != set(enrichment):
        raise ValueError("audit/enrichment ID coverage mismatch")
    candidates = []
    for pid, original in raw_candidates.items():
        extra = enrichment[pid]
        if extra.get("source") != original["source"] or any(key in original and original[key] != value for key, value in extra.items() if key not in {"human_verified"}):
            raise ValueError(f"enrichment changed audited identity/raw field: {pid}")
        if original.get("audit_binding") != audit_ref or original.get("human_verified") is not False or extra.get("human_verified") is not False:
            raise ValueError("missing exact AI-only audit/enrichment binding")
        candidate = original | extra
        for key in ("historical_manifest", "historical_receipt"):
            _check_ref(candidate[key], checked)
        for key in ("native_source_refs", "annotation_refs"):
            for reference in candidate.get(key, []):
                _check_ref(reference, checked)
        solution = candidate.get("human_solution")
        if solution:
            if solution.get("authorship") != "dataset_native_expert_curated" or solution.get("human_verified") is not False:
                raise ValueError(f"unsupported solution provenance: {pid}")
            _check_ref(solution["source_ref"], checked)
        if candidate.get("f0_prior"):
            _check_ref(candidate["f0_prior"]["record_ref"], checked)
        candidate["exclusion_reasons"] = _hard_reasons(candidate)
        candidate["gradable"], candidate["gradability_basis"] = gradable(candidate)
        candidate["rights"] = candidate["lineage"]["license"]
        if candidate["source"] == policy.AI2D:
            candidate["rights"] = AI2D_RIGHTS
            candidate["rights_conflict_note"] = "Archive license is stricter than registry CC-BY-SA-4.0; archive restrictions govern."
        candidates.append(candidate)
    if _sha(INDEX_PATH) != INDEX_SHA256:
        raise ValueError("pinned benchmark exclusion index changed")
    from fine_tuning.notebook_rl import contamination
    index = contamination.load_index(INDEX_PATH)
    if index.get("full_coverage") is not True:
        raise ValueError("incomplete benchmark exclusion index")
    scan = scan_new_texts(candidates, index["text_cache"], contamination._text_matches)
    scan["detector"] = _ref(Path(contamination.__file__))
    quarantined = set(scan["quarantined_families"])
    for candidate in candidates:
        if candidate["family_id"] in quarantined:
            candidate["exclusion_reasons"].append("NEW_TEXT_MATCH_FAMILY")
        candidate["new_fields_checked"] = True
        candidate["eligible"] = not candidate["exclusion_reasons"]
        candidate.update(policy.feature_candidate(candidate))
    _write(output / "candidates.jsonl", candidates, jsonl=True)
    _write(output / "contamination" / "new-text-checks.json", scan)
    inputs = {"audit_summary": _ref(audit_summary_path), "audit": audit_ref,
              "audit_candidates": _ref(candidate_path), "enrichment": _ref(enrichment_path),
              "holdouts": holdout_ref, "index": _ref(INDEX_PATH),
              "source_files": [{"path": key, "sha256": value} for key, value in sorted(checked.items())]}
    prepared = {"schema": SCHEMA, "status": "DRAFT_AWAITING_REVIEW", "inputs": inputs,
                "candidate_count": len(candidates), "eligible_count": sum(x["eligible"] for x in candidates),
                "candidates": _ref(output / "candidates.jsonl"),
                "new_text_checks": _ref(output / "contamination" / "new-text-checks.json"),
                "builder": _ref(Path(__file__)), "human_verified": False}
    _write(output / "prepared.json", prepared)
    return prepared


def apply_reviews(candidates: list[dict], reviews: list[dict], calibration: dict) -> list[dict]:
    by_id = _by_id(reviews)
    candidate_ids = {item["problem_id"] for item in candidates}
    if not set(by_id) <= candidate_ids:
        raise ValueError("review includes unknown candidate")
    sources = calibration.get("sources", {})
    counts = Counter(item["source"] for item in candidates)
    reviewed_counts = Counter(item["source"] for item in reviews)
    for source, count in counts.items():
        stats = sources.get(source, {})
        minimum = min(count, 40 if count > 400 else 20)
        errors = sum(row.get("label_status") == "incorrect" for row in reviews if row.get("source") == source)
        if reviewed_counts[source] < minimum or stats.get("sample_count") != reviewed_counts[source] or stats.get("label_error_count") != errors:
            raise ValueError(f"incomplete source review/calibration: {source}")
        if errors / reviewed_counts[source] > .05 and not stats.get("mitigation"):
            raise ValueError(f"label error rate needs source mitigation: {source}")
    result = []
    for original in candidates:
        candidate = dict(original)
        candidate["exclusion_reasons"] = list(original["exclusion_reasons"])
        rules = sources[candidate["source"]].get("reviewed_rule_precision", {})
        disabled = []
        for rule in ("image_needed", "reasoning_depth"):
            stat = rules.get(rule)
            if not isinstance(stat, dict) or type(stat.get("correct")) is not int or type(stat.get("total")) is not int or not 0 <= stat["correct"] <= stat["total"]:
                raise ValueError(f"missing or invalid source rule calibration: {rule}")
            if not stat["total"] or stat["correct"] / stat["total"] < .8:
                disabled.append(rule)
        candidate["disabled_rules"] = disabled
        review = by_id.get(candidate["problem_id"])
        if review:
            if review.get("source") != candidate["source"] or review.get("reviewed_by") != "codex-ai" or review.get("human_verified") is not False:
                raise ValueError("invalid review identity/provenance")
            if review.get("label_status") not in {"correct", "incorrect", "unknown"}:
                raise ValueError("missing review label status")
            if review.get("label_status") == "incorrect":
                candidate["exclusion_reasons"].append("LABEL_ERROR")
            if review.get("legible") is False:
                candidate["exclusion_reasons"].append("ILLEGIBLE")
            if review.get("question_error") is True:
                candidate["exclusion_reasons"].append("INVALID_QUESTION_PREMISE")
            extra_reasons = review.get("exclusion_reasons", [])
            if not isinstance(extra_reasons, list) or not set(extra_reasons) <= {"LABEL_ERROR", "UNGRADABLE", "INVALID_QUESTION_PREMISE", "ILLEGIBLE"}:
                raise ValueError("unsupported review exclusion reason")
            candidate["exclusion_reasons"].extend(extra_reasons)
            candidate["review_overrides"] = {key: review[key] for key in ("image_needed", "reasoning_steps", "level") if key in review}
            candidate["direct_review"] = review
        candidate["exclusion_reasons"] = sorted(set(candidate["exclusion_reasons"]))
        candidate["eligible"] = not candidate["exclusion_reasons"]
        candidate.update(policy.feature_candidate(candidate))
        result.append(candidate)
    return result


def apply_family_supplement(candidates: list[dict], holdouts: list[dict],
                            supplement: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    """Close native image links over old families before propagating exclusions."""
    parent: dict[str, str] = {}
    identities: dict[str, tuple[str, str]] = {}
    candidate_index = _by_id(candidates)
    derived_holdouts = set()
    for item in candidates + holdouts:
        family = item.get("family_id", item.get("connected_family_sha256"))
        source = item.get("source")
        if not family or not source:
            original = candidate_index.get(item["problem_id"])
            if original is None:
                raise ValueError("missing audited family identity with no candidate binding")
            if any(item.get(key) != original.get(key) for key in
                   ("image_sha256", "pixel_sha256") if key in item):
                raise ValueError("reserved holdout image binding mismatch")
            family = family or original["family_id"]
            source = source or original["source"]
            derived_holdouts.add(item["problem_id"])
        if not isinstance(family, str) or not family or not isinstance(source, str) or not source:
            raise ValueError("missing audited family identity")
        identity = (source, family)
        previous = identities.setdefault(item["problem_id"], identity)
        if previous != identity:
            raise ValueError("inconsistent audited problem/family identity")
        parent.setdefault(family, family)

    def find(family: str) -> str:
        while parent[family] != family:
            parent[family] = parent[parent[family]]
            family = parent[family]
        return family

    links = [] if supplement is None else supplement.get("links")
    if supplement is not None and (
        supplement.get("schema") != "v16b-scienceqa-visual-family-supplement-v1"
        or supplement.get("status") != "COMPLETE"
        or supplement.get("source_scope") != ["hf-scienceqa"]
        or not isinstance(links, list)
    ):
        raise ValueError("invalid ScienceQA family supplement scope/status")
    for link in links or []:
        left, right = link.get("left_problem_id"), link.get("right_problem_id")
        if left not in identities or right not in identities:
            raise ValueError("supplement includes unknown problem identity")
        left_source, left_family = identities[left]
        right_source, right_family = identities[right]
        if (left_source != "hf-scienceqa" or right_source != "hf-scienceqa"
            or link.get("left_family_id") != left_family
            or link.get("right_family_id") != right_family
            or link.get("decision") != "SAME_VISUAL_FAMILY"):
            raise ValueError("supplement changed source or audited family identity")
        first, second = find(left_family), find(right_family)
        parent[max(first, second)] = min(first, second)
    components: dict[str, set[str]] = defaultdict(set)
    for family in parent:
        components[find(family)].add(family)
    renamed = {root: next(iter(families)) if len(families) == 1 else
               _digest(["v16b-supplement-family-v1", sorted(families)])
               for root, families in components.items()}
    held_roots = {find(identities[item["problem_id"]][1]) for item in holdouts}
    propagated: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        root = find(item["family_id"])
        propagated[root].update(set(item.get("exclusion_reasons", [])) &
                                {"HOLDOUT_FAMILY", "NEW_TEXT_MATCH_FAMILY", "LABEL_ERROR"})
    for root in held_roots:
        propagated[root].add("HOLDOUT_FAMILY")
    result = []
    for original in candidates:
        item = dict(original)
        old_family = item["family_id"]
        root = find(old_family)
        item["historical_family_id"] = old_family
        item["family_id"] = renamed[root]
        item["exclusion_reasons"] = sorted(set(item["exclusion_reasons"]) | propagated[root])
        item["eligible"] = not item["exclusion_reasons"]
        result.append(item)
    remapped_holdouts = []
    for original in holdouts:
        item = dict(original)
        family = identities[item["problem_id"]][1]
        item["historical_family_id"] = family
        item["source"] = identities[item["problem_id"]][0]
        item["family_binding_kind"] = "EXACT_CANDIDATE_ID_AND_IMAGE_HASHES" if item["problem_id"] in derived_holdouts else "EXPLICIT_AUDITED_FAMILY"
        item["connected_family_sha256"] = renamed[find(family)]
        remapped_holdouts.append(item)
    summary = {
        "rule": "global_union_of_audited_families_and_source_bound_ScienceQA_visual_links",
        "family_id_rule": "unchanged_single_family_else_sha256_of_version_and_sorted_old_families",
        "accepted_links": len(links or []),
        "derived_reserved_holdout_bindings": len(derived_holdouts),
        "merged_components": sum(len(families) > 1 for families in components.values()),
        "historical_family_count": len(parent), "effective_family_count": len(components),
        "newly_holdout_excluded": sum("HOLDOUT_FAMILY" in item["exclusion_reasons"] and
                                      "HOLDOUT_FAMILY" not in original["exclusion_reasons"]
                                      for item, original in zip(result, candidates, strict=True)),
        "newly_text_excluded": sum("NEW_TEXT_MATCH_FAMILY" in item["exclusion_reasons"] and
                                   "NEW_TEXT_MATCH_FAMILY" not in original["exclusion_reasons"]
                                   for item, original in zip(result, candidates, strict=True)),
        "newly_label_excluded": sum("LABEL_ERROR" in item["exclusion_reasons"] and
                                    "LABEL_ERROR" not in original["exclusion_reasons"]
                                    for item, original in zip(result, candidates, strict=True)),
        "bindings_sha256": _digest(sorted((item["problem_id"], item["historical_family_id"],
                                           item["family_id"]) for item in result)),
        "human_verified": False,
    }
    return result, remapped_holdouts, summary


def monitor_items(candidates: list[dict], selected: list[dict], count: int = 200) -> list[dict]:
    used_ids = {item["problem_id"] for item in selected}
    families = {item["family_id"] for item in selected}
    images = {sha for item in selected for sha in item["image_sha256"]}
    pixels = {sha for item in selected for sha in item["pixel_sha256"]}
    by_source: dict[str, list[dict]] = defaultdict(list)
    for item in sorted(candidates, key=lambda row: (_digest("v16b-monitor:" + row["problem_id"]), row["problem_id"])):
        if not item["eligible"] or item["problem_id"] in used_ids or item["family_id"] in families or images.intersection(item["image_sha256"]) or pixels.intersection(item["pixel_sha256"]):
            continue
        by_source[item["source"]].append(item)
        families.add(item["family_id"])
        images.update(item["image_sha256"])
        pixels.update(item["pixel_sha256"])
    total = sum(map(len, by_source.values()))
    if total < count:
        raise ValueError("insufficient disjoint clean monitor families")
    quotas = {source: count * len(items) // total for source, items in by_source.items()}
    remainder = sorted(by_source, key=lambda source: (-(count * len(by_source[source]) % total), source))
    for source in remainder[:count - sum(quotas.values())]:
        quotas[source] += 1
    return [item for source in sorted(by_source) for item in by_source[source][:quotas[source]]]


def validate_prefix_balance(selected: list[dict], order: dict) -> dict:
    """Recount every closed prefix from native prior evidence, independently of the scheduler."""
    total = len(selected)
    labels = {}
    for item in selected:
        prior = policy._prior(item)
        risk = (item["source"] in {"hf-ssmr", "hf-chebi-role"} or
                prior is not None and (prior["finish_reason"] == "length" or
                                       prior["completion_tokens"] >= 8192))
        annotation = item.get("has_annotation")
        labels[item["problem_id"]] = {
            "source": item["source"],
            "f0_label": "unknown" if prior is None else "correct" if prior["correct"] else "wrong",
            "annotation": "present" if annotation is True else "absent" if annotation is False else "unknown",
            "long_risk": "yes" if risk else "no" if prior is not None else "unknown",
        }
    dimensions = ("source", "f0_label", "annotation", "long_risk")
    populations = {dimension: Counter(value[dimension] for value in labels.values())
                   for dimension in dimensions}
    observed: dict[str, Counter[str]] = {dimension: Counter() for dimension in dimensions}
    maximum = {dimension: 0 for dimension in dimensions}
    worst = {dimension: 0 for dimension in dimensions}
    ordered = order["items"]
    flat_ids = [pid for batch in order["batches"] for pid in batch["problem_ids"]]
    if flat_ids != [item["problem_id"] for item in ordered]:
        raise ValueError("order batches differ from flat item order")
    item_keys = {"source": "source", "f0_label": "f0_label",
                 "annotation": "annotation_label", "long_risk": "long_risk_label"}
    for position, item in enumerate(ordered, 1):
        expected = labels[item["problem_id"]]
        if any(item.get(item_keys[dimension]) != expected[dimension] for dimension in dimensions):
            raise ValueError("order stratum label differs from native evidence")
        for dimension in dimensions:
            observed[dimension][expected[dimension]] += 1
        if position % 16 and position != total:
            continue
        for dimension in dimensions:
            deviation = max(abs(observed[dimension][label] * total - position * count)
                            for label, count in populations[dimension].items())
            if deviation > 2 * total:
                raise ValueError(f"order prefix balance exceeds 2 items: {dimension}/{position}")
            if deviation > maximum[dimension]:
                maximum[dimension], worst[dimension] = deviation, position
    report = order.get("prefix_balance", {})
    prefix_count = (total + 15) // 16
    if report.get("status") != "PASS" or report.get("tolerance_items") != 2 or report.get("checked_prefixes") != prefix_count:
        raise ValueError("missing or invalid required prefix balance report")
    summary = {}
    for dimension in dimensions:
        expected_max = maximum[dimension] / total
        measured = report.get("dimensions", {}).get(dimension, {})
        if measured.get("population_counts") != dict(populations[dimension]) or measured.get("violations") != []:
            raise ValueError("order prefix population/violation report mismatch")
        value = measured.get("max_abs_deviation")
        if type(value) not in {int, float} or abs(value - expected_max) > 1e-9:
            raise ValueError("order prefix deviation report mismatch")
        summary[dimension] = {"population_counts": dict(sorted(populations[dimension].items())),
                              "max_abs_deviation": expected_max,
                              "worst_prefix_items": worst[dimension]}
    return {"status": "PASS", "tolerance_items": 2,
            "checked_prefixes": prefix_count, "dimensions": summary}


def validate_selection(selected: list[dict], monitor: list[dict], candidates: list[dict], holdouts: list[dict], order: dict, *, total: int = 3000, monitor_count: int = 200, source_cap: int = 450, source_caps: dict | None = None) -> dict:
    if len(selected) != total or len(monitor) != monitor_count:
        raise ValueError("incorrect train/monitor count")
    combined = selected + monitor
    for key in ("problem_id", "family_id"):
        if len({item[key] for item in combined}) != len(combined):
            raise ValueError(f"duplicate train/monitor {key}")
    for key in ("image_sha256", "pixel_sha256"):
        owners: dict[str, str] = {}
        for item in combined:
            for sha in item[key]:
                if sha in owners and owners[sha] != item["problem_id"]:
                    raise ValueError(f"train/monitor overlap: {key}")
                owners[sha] = item["problem_id"]
        if set(owners).intersection(sha for row in holdouts for sha in row.get(key, [])):
            raise ValueError(f"historical holdout overlap: {key}")
    held_families = {row.get("connected_family_sha256", row.get("family_id")) for row in holdouts}
    held_ids = {row["problem_id"] for row in holdouts}
    if held_families.intersection(item["family_id"] for item in combined) or held_ids.intersection(item["problem_id"] for item in combined):
        raise ValueError("historical holdout identity/family overlap")
    for item in combined:
        if item.get("eligible") is not True or item.get("new_fields_checked") is not True or _hard_reasons(item):
            raise ValueError("ineligible or incompletely checked selected item")
    mandatory = {item["problem_id"] for item in candidates if item["source"] == policy.AI2D and item["eligible"]}
    if not mandatory <= {item["problem_id"] for item in selected}:
        raise ValueError("eligible mandatory AI2D omitted")
    counts = Counter(item["source"] for item in selected)
    caps = source_caps or {}
    if any(value > min(source_cap, caps.get(source, source_cap)) for source, value in counts.items() if source != policy.AI2D):
        raise ValueError("source cap exceeded")
    expected_sizes = [16] * (total // 16) + ([total % 16] if total % 16 else [])
    if [batch["size"] for batch in order["batches"]] != expected_sizes or sorted(item["problem_id"] for item in order["items"]) != sorted(item["problem_id"] for item in selected):
        raise ValueError("invalid order or final partial batch")
    return {"unique_train_ids": total, "unique_train_families": total, "monitor_count": monitor_count,
            "mandatory_ai2d": len(mandatory), "source_counts": dict(sorted(counts.items())),
            "historical_holdout_overlap": 0, "train_monitor_overlap": 0,
            "batch_count": len(expected_sizes), "last_batch_size": expected_sizes[-1],
            "order_prefix_balance": validate_prefix_balance(selected, order)}


def _manifest_item(item: dict) -> dict:
    keys = ("problem_id", "source", "native_id", "source_revision", "family_id", "historical_family_id", "structure_hash",
            "record_paths", "record_sha256", "image_paths", "image_sha256", "pixel_sha256",
            "historical_manifest", "historical_receipt", "audit_binding", "native_source_refs",
            "native_metadata", "annotation_refs", "annotation_exists", "annotation_kind",
            "annotation_verifier_supported", "annotation_support_status", "raw_annotation",
            "human_solution", "f0_prior", "rights", "rights_conflict_note", "gradable",
            "gradability_basis", "score", "score_components", "score_reason_codes", "image_type",
            "image_needed", "reasoning_steps", "reasoning_kind", "level", "level_kind",
            "mmmu_subjects", "subject_mapping_kind", "disabled_rules")
    return {key: item[key] for key in keys if key in item} | {
        "original_manifest_item": item["manifest_item"], "human_verified": False}


def _verify_selected_files(items: list[dict], checked: dict[str, str]) -> list[dict]:
    records: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in items:
        for key, path in item["record_paths"].items():
            records[path].append((item["problem_id"], item["record_sha256"][key]))
        for path, sha in zip(item["image_paths"], item["image_sha256"], strict=True):
            _check_ref({"path": path, "sha256": sha}, checked)
    for path, expected in records.items():
        indexed = _by_id(_rows(Path(path)))
        for pid, sha in expected:
            if pid not in indexed or _digest(indexed[pid]) != sha:
                raise ValueError(f"native record changed: {path}: {pid}")
        checked[path] = _sha(Path(path))
    return [{"path": key, "sha256": value} for key, value in sorted(checked.items())]


def freeze(*, prepared_path: Path, reviews_path: Path, calibration_path: Path, output: Path,
           family_supplement_path: Path | None = None) -> dict:
    names = ("candidates.jsonl", "selection.json", "train-3000-manifest.json", "monitor-manifest.json", "order.json", "validation.json", "contamination/new-text-checks.json")
    if any((output / name).exists() for name in names):
        raise ValueError("freeze output already exists; never overwrite a frozen artifact")
    prepared = json.loads(prepared_path.read_text())
    if prepared.get("schema") != SCHEMA or prepared.get("status") != "DRAFT_AWAITING_REVIEW":
        raise ValueError("invalid preparation receipt")
    checked: dict[str, str] = {}
    for reference in [prepared["candidates"], prepared["new_text_checks"], prepared["builder"], *prepared["inputs"]["source_files"]]:
        _check_ref(reference, checked)
    for key in ("audit_summary", "audit", "audit_candidates", "enrichment", "holdouts", "index"):
        _check_ref(prepared["inputs"][key], checked)
    scan = json.loads(Path(prepared["new_text_checks"]["path"]).read_text())
    if scan.get("status") != "COMPLETE_FRESH_TEXT_SCAN_WITH_LIMITS" or scan.get("index_sha256") != INDEX_SHA256 or scan.get("thresholds_changed") is not False:
        raise ValueError("fresh text scan incomplete or changed")
    _check_ref(scan["detector"], checked)
    reviews, calibration = _rows(reviews_path), json.loads(calibration_path.read_text())
    if calibration.get("policy_version") != policy.VERSION or calibration.get("policy_sha256") != _sha(Path(policy.__file__)):
        raise ValueError("calibration policy mismatch; rebuild calibration for current policy bytes")
    candidates = apply_reviews(_rows(Path(prepared["candidates"]["path"])), reviews, calibration)
    quarantined = set(scan["quarantined_families"])
    if any(item["eligible"] and item["family_id"] in quarantined for item in candidates):
        raise ValueError("fresh text quarantine lost during review")
    holdouts = _rows(Path(prepared["inputs"]["holdouts"]["path"]))
    supplement = None
    if family_supplement_path is not None:
        supplement = json.loads(family_supplement_path.read_text())
        references = supplement.get("input_refs")
        if not isinstance(references, dict) or not {"audit_candidates", "holdouts", "pairs", "script"} <= set(references):
            raise ValueError("family supplement lacks complete input/evidence bindings")
        for reference in references.values():
            _check_ref(reference, checked)
        for key in ("audit_candidates", "holdouts"):
            if any(references[key].get(field) != prepared["inputs"][key].get(field)
                   for field in ("path", "sha256")):
                raise ValueError("family supplement bound to different audited inputs")
    candidates, holdouts, family_summary = apply_family_supplement(candidates, holdouts, supplement)
    scan["effective_quarantined_families"] = sorted({
        item["family_id"] for item in candidates
        if "NEW_TEXT_MATCH_FAMILY" in item["exclusion_reasons"]})
    scan["family_propagation"] = family_summary
    source_caps = {source: stats["source_cap"] for source, stats in calibration["sources"].items() if "source_cap" in stats}
    selected, decisions = policy.choose(candidates, source_caps=source_caps)
    monitor = monitor_items(candidates, selected)
    order = policy.interleave(selected, seed=3407, batch_size=16)
    checks = validate_selection(selected, monitor, candidates, holdouts, order, source_caps=source_caps)
    file_inventory = _verify_selected_files(selected + monitor, checked)
    bindings = {"prepared": _ref(prepared_path), "reviews": _ref(reviews_path),
                "calibration": _ref(calibration_path), "policy": _ref(Path(policy.__file__)),
                "builder": _ref(Path(__file__)), "inputs": prepared["inputs"],
                "family_propagation": family_summary,
                "family_supplement": _ref(family_supplement_path) if family_supplement_path else None}
    scan_ref = {"path": str((output / "contamination/new-text-checks.json").absolute()), "sha256": hashlib.sha256(_bytes(scan) + b"\n").hexdigest()}
    train = {"schema": SCHEMA, "split": "train", "training_allowed": True, "count": 3000,
             "items": [_manifest_item(item) for item in selected], "bindings": bindings,
             "new_text_checks": scan_ref, "file_inventory": file_inventory, "human_verified": False}
    monitor_manifest = {"schema": SCHEMA, "split": "monitor", "training_allowed": False,
                        "count": 200, "seed": 3407, "selection_rule": "proportional_unselected_source_largest_remainder_hash_order",
                        "reused_v16_sentinel": False, "sentinel_reason": "Existing reserved sets remain held out; 200 clean disjoint native candidates suffice.",
                        "items": [_manifest_item(item) for item in monitor], "bindings": bindings, "human_verified": False}
    selection = {"schema": SCHEMA, "decisions": decisions, "bindings": bindings,
                 "counts": checks, "composition": {key: dict(sorted(Counter(str(item.get(key)) for item in selected).items())) for key in ("source", "image_type", "level", "reasoning_steps", "reasoning_kind", "has_annotation", "has_human_solution", "f0_label")},
                 "human_verified": False}
    for name, value in (("selection.json", selection), ("train-3000-manifest.json", train),
                        ("monitor-manifest.json", monitor_manifest), ("order.json", order),
                        ("contamination/new-text-checks.json", scan)):
        _write(output / name, value)
    _write(output / "candidates.jsonl", candidates, jsonl=True)
    validation = {"schema": SCHEMA, "status": "PASS_CURATED_NATIVE_3000_WITH_LIMITS", "checks": checks,
                  "bindings": bindings, "artifacts": {name: _ref(output / name) for name in names if name != "validation.json"},
                  "scope": "CPU curation only; no generation, training, GPU, download, cloud or publication",
                  "limitations": ["Historical image/OCR/embedding checks reused through exact byte bindings, not rerun.",
                                  "AI review and native expert-curated provenance are not human verification.",
                                  "Bounded short targets use existing verifier semantics; exact V8 compatibility is not established.",
                                  "No proof of absence of pretraining exposure or every semantic paraphrase."],
                  "human_verified": False}
    _write(output / "validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    draft = subparsers.add_parser("prepare")
    draft.add_argument("--audit-summary", required=True, type=Path)
    draft.add_argument("--enrichment", required=True, type=Path)
    draft.add_argument("--output", required=True, type=Path)
    final = subparsers.add_parser("freeze")
    final.add_argument("--prepared", required=True, type=Path)
    final.add_argument("--reviews", required=True, type=Path)
    final.add_argument("--calibration", required=True, type=Path)
    final.add_argument("--family-supplement", type=Path)
    final.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(audit_summary_path=args.audit_summary, enrichment_path=args.enrichment, output=args.output)
        print(json.dumps({key: result[key] for key in ("status", "candidate_count", "eligible_count")}))
    else:
        result = freeze(prepared_path=args.prepared, reviews_path=args.reviews,
                        calibration_path=args.calibration, output=args.output,
                        family_supplement_path=args.family_supplement)
        print(json.dumps({"status": result["status"], "checks": result["checks"]}))


if __name__ == "__main__":
    main()
