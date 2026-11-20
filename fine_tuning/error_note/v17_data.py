"""Private v17 splits and bounded, source-bound observation capabilities.

Canonical annotation JSON is used only for verification and size checks, never as
a generated response or a supervised target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, from_json
from fine_tuning.notebook_rl.vault import write_split

from . import content_v16 as content
from . import curation_policy_v16b as policy
from . import curation_v16b as curation
from . import data_v16b as legacy

VERSION = "v17-source-bound-data-v1"
SEED = 3407
CLAIM_KINDS = {
    "geometry": ["SEGMENT_LABEL", "ANGLE_LABEL", "PARALLEL", "PERPENDICULAR"],
    "graph": ["VISIBLE_NODE", "EDGE_PRESENT", "EDGE_WEIGHT"],
    "scene": ["OBJECT_ATTRIBUTE", "OBJECT_COUNT"],
    "chart": ["CHART_VALUE"],
}
ENTITY_SCHEMA = {
    "geometry": {
        "SEGMENT_LABEL": {"entity_type": "[point-name string, point-name string]", "value_type": "printed numeric or expression string", "unit_type": "optional string; canonical length"},
        "ANGLE_LABEL": {"entity_type": "[point-name string, vertex-name string, point-name string]", "value_type": "printed numeric or expression string", "unit_type": "optional string; canonical degree"},
        "PARALLEL": {"entity_type": "[[point-name string, point-name string], [point-name string, point-name string]]", "value_type": "boolean"},
        "PERPENDICULAR": {"entity_type": "[[point-name string, point-name string], [point-name string, point-name string]]", "value_type": "boolean"}},
    "graph": {
        "VISIBLE_NODE": {"entity_type": "integer", "value_type": "boolean"},
        "EDGE_PRESENT": {"entity_type": "[integer, integer]", "value_type": "boolean"},
        "EDGE_WEIGHT": {"entity_type": "[integer, integer]", "value_type": "numeric string"}},
    "scene": {
        "OBJECT_ATTRIBUTE": {"entity_type": "[one-based left rank integer, attribute-name string]", "value_type": "string",
                             "attribute_names": ["size", "color", "material", "shape"]},
        "OBJECT_COUNT": {"entity_type": "literal all", "value_type": "integer string", "unit_type": "string"}},
    "chart": {"CHART_VALUE": {"entity_type": "[series-name string, category-name string]", "value_type": "bare numeric string", "unit_type": "required string; printed unit separately, empty string for unitless"}},
}
SCOPES = {
    "geometry": "Directly printed facts, kind order then entity name order, first eight; unannotated facts remain unknown.",
    "graph": "First four visible numeric node IDs in ascending order and all edges between them; put each weight in EDGE_WEIGHT.",
    "scene": "First three visible objects from left to right: size, color, material, shape; also the total visible object count.",
    "chart": "Legend series order, then horizontal category order, first eight displayed values.",
}


def digest(value: object) -> str:
    return curation._digest(value)


def _sealed(body: dict) -> dict:
    return body | {"sha256": digest(body)}


def _layer(item: dict) -> str:
    return legacy.LAYERS.get(item.get("source", ""), item.get("layer") or "unsupported")


def public_schema(item_or_layer: dict | str) -> dict:
    """Only generic kinds and instructions cross the actor boundary."""
    layer = _layer(item_or_layer) if isinstance(item_or_layer, dict) else item_or_layer
    return {"schema": "v17-public-observation-schema-v1", "layer": layer,
            "supported_claim_kinds": CLAIM_KINDS.get(layer, []), "entity_schema": ENTITY_SCHEMA.get(layer, {}),
            "scope_instruction": SCOPES.get(layer, "Unsupported observation source."), "max_claims": 16}


def identities(item: dict) -> set[tuple[str, str]]:
    result = {("id", item["problem_id"])}
    aliases = {"family": ("family_id", "family_ids", "historical_family_id", "connected_family_sha256", "group_id"),
               "byte": ("image_sha256", "image_byte_sha256", "source_image_sha256"),
               "pixel": ("pixel_sha256", "image_pixel_sha256")}
    for kind, keys in aliases.items():
        for key in keys:
            values = item.get(key) or []
            result.update((kind, str(x)) for x in ([values] if isinstance(values, str) else values))
    source = item.get("source") or item.get("source_id") or item.get("source_repo")
    if source and item.get("native_id"):
        result.add(("native", f"{source}:{item['native_id']}"))
    return result


def identity_components(items: list[dict]) -> dict[str, str]:
    """Close identity transitively, including bridges in excluded candidates."""
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    for item in items:
        keys = sorted(identities(item))
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            parent[max(a, b)] = min(a, b)
    return {x["problem_id"]: digest(find(("id", x["problem_id"]))) for x in items}


def historical_reservations(manifests: list[Path], release_receipts: Sequence[Path] = ()) -> tuple[list[dict], dict]:
    reserved: dict[str, list[dict]] = {}
    refs: list[dict] = []
    for path in sorted(manifests):
        ref = curation._ref(path)
        refs.append(ref)
        for row in json.loads(path.read_text())["items"]:
            if row.get("training_allowed") is False or row.get("split") in {"sentinel", "dev", "sealed", "monitor", "development"}:
                reserved.setdefault(row["problem_id"], []).append(dict(row, reservation_ref=ref))
    released, approvals = set(), []
    for path in sorted(release_receipts):
        receipt = json.loads(path.read_text())
        if receipt.get("schema") != "v17-reservation-release-v1" or receipt.get("authority") != "USER_EXPLICIT":
            raise ValueError("RESERVATION_RELEASE_NOT_AUTHORIZED")
        checked: dict = {}
        curation._check_ref(receipt["authorization_ref"], checked)
        for ref in receipt["reservation_refs"]:
            curation._check_ref(ref, checked)
            if ref not in refs:
                raise ValueError("RESERVATION_RELEASE_WRONG_MANIFEST")
        for pid in receipt["problem_ids"]:
            if pid not in reserved or any(row["reservation_ref"] not in receipt["reservation_refs"] for row in reserved[pid]):
                raise ValueError("RESERVATION_RELEASE_INCOMPLETE")
            released.add(pid)
        approvals.append(curation._ref(path))
    rows = [row for pid, versions in sorted(reserved.items()) if pid not in released for row in versions]
    return rows, {"union_ids": len(reserved), "retained_ids": len(reserved.keys() - released),
                  "released_ids": sorted(released), "manifests": refs, "release_receipts": approvals}


def _annotation(item: dict) -> tuple[dict, dict]:
    refs = item.get("references", {})
    raw = item.get("raw_annotation") or refs.get("raw_annotation") or item.get("annotation") or {}
    metadata = item.get("native_metadata") or refs.get("native_metadata") or {}
    return raw, metadata


def chart_facts(item: dict) -> tuple[list[dict], str | None, dict]:
    """Require positional visibility/order evidence in addition to the old OCR join."""
    raw, metadata = _annotation(item)
    chart = raw.get("chart", {})
    name = metadata.get("image_filename") or item.get("image_filename", "")
    stages = {"annotation_present": bool(chart and raw.get("table_csv")), "image_join": False,
              "visibility_and_order": False, "scope": False}
    if not stages["annotation_present"]:
        return [], "PEW_NO_SOURCE_TABLE" if metadata.get("chart_source") == "PEW" else "NO_SOURCE_TABLE_OR_ANNOTATION", stages
    visibility = raw.get("visibility_receipt") or {}
    images = item.get("image_sha256") or []
    if isinstance(images, str):
        images = [images]
    if visibility.get("image_sha256") not in images or visibility.get("annotation_sha256") != digest(chart):
        return [], "CHART_IMAGE_ANNOTATION_BINDING_UNVERIFIED", stages
    stages["image_join"] = True
    _, reason = content.chart_target(raw["table_csv"], chart, imgname=name, visibility_receipt=visibility)
    if reason:
        return [], reason, stages
    certificate = raw.get("position_visibility_order_receipt") or {}
    if (certificate.get("schema") != "v17-chart-position-visibility-order-v1"
            or certificate.get("status") != "VERIFIED" or certificate.get("image_sha256") not in images
            or certificate.get("annotation_sha256") != digest(chart)
            or certificate.get("table_sha256") != hashlib.sha256(raw["table_csv"].encode()).hexdigest()
            or certificate.get("position_bound") is not True or certificate.get("units_verified") is not True
            or not certificate.get("evidence_refs")):
        return [], "POSITION_VISIBILITY_ORDER_UNVERIFIED", stages
    checked: dict = {}
    for ref in certificate["evidence_refs"]:
        curation._check_ref(ref, checked)
    table = list(csv.reader(io.StringIO(raw["table_csv"])))
    header = [x.strip() for x in table[0]]
    categories = [row[0].strip() for row in table[1:]]
    series_order, category_order = certificate.get("series_order"), certificate.get("category_order")
    if (not isinstance(series_order, list) or set(series_order) != set(header[1:])
            or len(series_order) != len(header) - 1 or not isinstance(category_order, list)
            or set(category_order) != set(categories) or len(category_order) != len(categories)):
        return [], "CHART_ORDER_CERTIFICATE_MISMATCH", stages
    values = {(row[0].strip(), header[col]): row[col].strip() for row in table[1:] for col in range(1, len(header))}
    visible = certificate.get("visible_values")
    expected = [{"series": series, "category": category, "value": values[category, series]}
                for series in series_order for category in category_order]
    if visible != expected:
        return [], "CHART_POSITION_VALUES_MISMATCH", stages
    units = certificate.get("units")
    if (not isinstance(units, dict) or set(units) != set(series_order)
            or any(not isinstance(unit, str) for unit in units.values())):
        return [], "CHART_UNITS_UNVERIFIED", stages
    stages["visibility_and_order"] = stages["scope"] = True
    facts = []
    for value in expected[:8]:
        text, unit = value["value"].strip(), units[value["series"]]
        if text.endswith("%"):
            if unit != "%":
                return [], "CHART_UNITS_UNVERIFIED", stages
            text = text[:-1]
        facts.append({"kind": "CHART_VALUE", "entity": [value["series"], value["category"]],
                      "value": text.replace(",", ""), "unit": unit})
    return facts, None, stages


def _facts(item: dict) -> tuple[list[dict], str | None, list[str], dict]:
    raw, metadata = _annotation(item)
    layer = _layer(item)
    facts: list[dict] = []
    closed: list[str] = []
    scope: dict = {}
    reason = None
    if not raw:
        return [], "NO_OBSERVATION_ANNOTATION", [], scope
    if layer == "chart":
        facts, reason, scope = chart_facts(item)
    elif layer == "geometry":
        forms = raw.get("diagram_logic_forms", [])
        if "candidates" in raw:
            forms = raw["candidates"][raw["matched_id"]]["diagram_logic_forms"]
        for form in forms:
            length, angle, relation = content._LEN.fullmatch(form), content._ANG.fullmatch(form), content._REL.fullmatch(form)
            if length or angle:
                match = length or angle
                assert match is not None
                width = 2 if length else 3
                entity = list(match.groups()[:width])
                entity = min(entity, entity[::-1])
                facts.append({"kind": "SEGMENT_LABEL" if length else "ANGLE_LABEL", "entity": entity,
                              "value": match[width + 1], "unit": "length" if length else "degree"})
            elif relation:
                entity = sorted([sorted([relation[2], relation[3]]), sorted([relation[4], relation[5]])])
                facts.append({"kind": "PARALLEL" if relation[1] == "Parallel" else "PERPENDICULAR", "entity": entity, "value": True})
        facts.sort(key=lambda f: (CLAIM_KINDS[layer].index(f["kind"]), json.dumps(f["entity"])))
        facts = facts[:8]
        # Partial native annotations can verify a correction, but cannot certify the first eight visible facts.
        reason = "SCOPE_COMPLETENESS_UNVERIFIED"
        scope = {"partial_annotation": True, "unlisted_facts": "UNKNOWN"}
        cert = raw.get("scope_completeness_receipt") or {}
        if (cert.get("schema") == "v17-geometry-visible-scope-v1" and cert.get("status") == "VERIFIED"
                and cert.get("annotation_sha256") == digest(forms)
                and cert.get("image_sha256") in item.get("image_sha256", []) and cert.get("evidence_refs")):
            checked: dict = {}
            for ref in cert["evidence_refs"]:
                curation._check_ref(ref, checked)
            reason = None
    elif layer == "graph":
        if "structure_text" in raw:
            text = raw["structure_text"]
        else:
            text = f"{raw['n']} {raw['m']}\n" + "\n".join(" ".join(map(str, x)) for x in raw["edges"])
        graph = content.parse_gvlqa_structure(text)
        directed = metadata.get("directed", raw.get("directed"))
        if type(directed) is not bool:
            return [], "GRAPH_DIRECTION_UNVERIFIED", [], scope
        nodes = graph["visible_nodes"][:4]
        scope = {"visible_nodes": nodes, "directed": directed, "isolated_nodes": "UNVERIFIED_EXCLUDED",
                 "rendering_review_scope": "PER_TASK_CONVENTION_SPOTCHECK_NOT_EXHAUSTIVE_PIXEL_CERTIFICATION"}
        facts = [{"kind": "VISIBLE_NODE", "entity": n, "value": True} for n in nodes]
        seen = set()
        for edge in graph["edges"]:
            if edge[0] not in nodes or edge[1] not in nodes:
                continue
            entity = list(edge[:2]) if directed else sorted(edge[:2])
            if tuple(entity) in seen:
                return [], "GRAPH_DUPLICATE_EDGE", [], scope
            seen.add(tuple(entity))
            facts.append({"kind": "EDGE_WEIGHT" if len(edge) == 3 else "EDGE_PRESENT", "entity": entity,
                          "value": str(edge[2]) if len(edge) == 3 else True})
        closed = ["VISIBLE_NODE", "EDGE_PRESENT", "EDGE_WEIGHT"]
    elif layer == "scene":
        scene = raw.get("scene", raw)
        if (metadata.get("image_filename") and scene.get("image_filename") != metadata["image_filename"]
                or metadata.get("image_index") is not None and scene.get("image_index") != metadata["image_index"]):
            return [], "SCENE_IDENTITY_MISMATCH", [], scope
        objects = sorted(scene.get("objects", []), key=lambda obj: obj["pixel_coords"][0])
        if not objects or len({obj["pixel_coords"][0] for obj in objects}) != len(objects):
            return [], "SCENE_X_IDENTITY_AMBIGUOUS", [], scope
        facts = [{"kind": "OBJECT_COUNT", "entity": "all", "value": str(len(objects)), "unit": "count"}]
        for rank, obj in enumerate(objects[:3], 1):
            for attribute in ("size", "color", "material", "shape"):
                facts.append({"kind": "OBJECT_ATTRIBUTE", "entity": [rank, attribute], "value": obj[attribute]})
        scope = {"object_identity": "native_scene_pixel_x", "partial_occlusion": "native_scene_identity_only"}
        closed = ["OBJECT_COUNT", "OBJECT_ATTRIBUTE"]
    else:
        return [], "UNSUPPORTED_OBSERVATION_LAYER", [], scope
    if not facts and reason is None:
        reason = "NO_SUPPORTED_FACTS"
    if len(facts) > 16:
        reason = "SCOPE_CLAIM_LIMIT"
    return facts, reason, closed, scope


def private_verifier(item: dict, *, tokenizer=None) -> dict:
    facts, reason, closed, scope = _facts(item)
    raw, _ = _annotation(item)
    token_count = None
    if tokenizer is not None:
        text = json.dumps({"claims": facts}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > 768:
            reason = "SCOPE_TOKEN_LIMIT"
    else:
        saved = item.get("v17_private_verifier") or {}
        if (saved.get("facts") == facts and saved.get("annotation_sha256") == digest(raw)
                and saved.get("verifier_source_sha256") == curation._sha(Path(__file__))
                and type(saved.get("canonical_tokens")) is int):
            token_count = saved["canonical_tokens"]
            if token_count > 768:
                reason = "SCOPE_TOKEN_LIMIT"
        elif reason is None:
            reason = "TOKENIZER_UNVERIFIED"
    kinds = [kind for kind in CLAIM_KINDS.get(_layer(item), []) if any(x["kind"] == kind for x in facts)]
    if _layer(item) == "graph" and reason is None:
        kinds = CLAIM_KINDS["graph"].copy()
    return _sealed({"schema": "v17-private-verifier-v1", "problem_id": item["problem_id"],
        "layer": _layer(item), "image_sha256": item.get("image_sha256", []),
        "annotation_sha256": digest(raw), "facts": facts, "supported_claim_kinds": kinds,
        "closed_world_kinds": closed if reason is None else [], "closed_world_scopes": scope,
        "directed": scope.get("directed", False), "scope": SCOPES.get(_layer(item), "unsupported"),
        "obs_eligible": bool(facts) and reason is None, "obs_reason": reason,
        "repair_catalog_eligible": bool(facts) and reason in {None, "SCOPE_COMPLETENESS_UNVERIFIED"},
        "canonical_tokens": token_count, "public_schema_sha256": digest(public_schema(item)),
        "verifier_source_sha256": curation._sha(Path(__file__))})


def capability_manifest(item: dict, *, tokenizer=None) -> dict:
    verifier = private_verifier(item, tokenizer=tokenizer)
    raw, _ = _annotation(item)
    return {"schema": "v17-capability-v1", "problem_id": item["problem_id"],
        "assembled_role": item.get("split"), "source_id": item.get("source"),
        "source_revision": item.get("source_revision"), "source_split": (item.get("lineage") or {}).get("split", "train"),
        "native_id": item.get("native_id"), "family_ids": sorted(x[1] for x in identities(item) if x[0] == "family"),
        "image_byte_sha256": item.get("image_sha256", []), "image_pixel_sha256": item.get("pixel_sha256", []),
        "training_allowed": item.get("training_allowed", False), "answer_verifier_id": "team-final-answer-v8",
        "annotation_present": bool(raw), "annotation_source_refs": item.get("annotation_refs", []),
        "annotation_binding_receipt_sha256": digest({"annotation": raw, "refs": item.get("annotation_refs", [])}),
        "obs_eligible": verifier["obs_eligible"], "obs_reason": verifier["obs_reason"],
        "repair_catalog_eligible": verifier["repair_catalog_eligible"],
        "repair_reason": None if verifier["repair_catalog_eligible"] else verifier["obs_reason"],
        "supported_claim_kinds": verifier["supported_claim_kinds"], "closed_world_scopes": verifier["closed_world_scopes"],
        "native_human_reference_refs": [item["human_solution"]["source_ref"]] if item.get("human_solution") else [],
        "public_schema_sha256": verifier["public_schema_sha256"], "private_verifier_sha256": verifier["sha256"],
        "verifier_source_sha256": verifier["verifier_source_sha256"], "inherited_reservations": item.get("inherited_reservations", [])}


def _choose_proportional(candidates: list[dict], count: int, prefix: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        buckets[item["source"]].append(item)
    if sum(map(len, buckets.values())) < count:
        raise ValueError("DATA_INSUFFICIENT:" + prefix)
    total = len(candidates)
    quota = {s: count * len(xs) // total for s, xs in buckets.items()}
    for source in sorted(buckets, key=lambda s: (-(count * len(buckets[s]) % total), s))[:count - sum(quota.values())]:
        quota[source] += 1
    return [x for source in sorted(buckets) for x in sorted(buckets[source], key=lambda x: (
        hashlib.sha256(f"{prefix}:{SEED}:{x['problem_id']}".encode()).hexdigest(), x["problem_id"]))[:quota[source]]]


def select_roles(candidates: list[dict], current: dict[str, list[dict]], reservations: list[dict],
                 *, capabilities: dict[str, dict], source_caps: dict[str, int] | None = None,
                 exposures: Sequence[dict] = (), counts: tuple[int, int, int] = (3000, 512, 200),
                 effect_min: int = 64, effect_max: int = 128, performance_count: int = 16) -> tuple[dict, dict]:
    by_id = curation._by_id(candidates)
    components = identity_components(candidates + reservations + list(exposures))
    reserved = {components[x["problem_id"]] for x in reservations}
    roles, used, quarantine = {}, set(reserved), []
    caps = source_caps or {}
    for role, count in zip(("prep", "monitor", "train"), (counts[1], counts[2], counts[0]), strict=True):
        retained = []
        for original in sorted(current[role], key=lambda x: x["problem_id"]):
            item = by_id[original["problem_id"]]
            component = components[item["problem_id"]]
            reason = "HISTORICAL_RESERVATION_CLOSURE" if component in reserved else "ROLE_IDENTITY_OVERLAP" if component in used else None
            if item.get("eligible") is not True or item.get("exclusion_reasons"):
                reason = "HARD_ELIGIBILITY_FAILED"
            if reason:
                quarantine.append({"problem_id": item["problem_id"], "role": role, "reason": reason})
                if role != "train":
                    raise ValueError("FIXED_ROLE_INVALID:" + role + ":" + reason)
            else:
                retained.append(item)
                used.add(component)
        if role == "train":
            source_counts = Counter(x["source"] for x in retained)
            old_ids = {x["problem_id"] for items in current.values() for x in items}
            ranked = sorted((x for x in candidates if x["problem_id"] not in old_ids), key=lambda x: (
                -int(capabilities[x["problem_id"]]["repair_catalog_eligible"]), -int(bool(x.get("human_solution"))),
                hashlib.sha256(f"v17-replace:{SEED}:{x['problem_id']}".encode()).hexdigest(), x["problem_id"]))
            for item in ranked:
                if len(retained) == count:
                    break
                component, source = components[item["problem_id"]], item["source"]
                if (item.get("eligible") is not True or item.get("exclusion_reasons") or component in used
                        or source_counts[source] >= caps.get(source, 450)):
                    continue
                retained.append(item)
                used.add(component)
                source_counts[source] += 1
        if len(retained) != count:
            raise ValueError("DATA_INSUFFICIENT:" + role)
        roles[role] = retained
    exposed = {components[x["problem_id"]] for x in exposures}
    remaining = []
    for item in sorted(candidates, key=lambda x: x["problem_id"]):
        component = components[item["problem_id"]]
        if (item.get("eligible") is True and not item.get("exclusion_reasons") and not item.get("f0_prior")
                and component not in used | exposed):
            remaining.append(item)
            used.add(component)
    effect_count = min(effect_max, len(remaining) - performance_count)
    if effect_count < effect_min:
        raise ValueError("EFFECT_DATA_INSUFFICIENT")
    roles["effect"] = _choose_proportional(remaining, effect_count, "v17-effect")
    effect_ids = {x["problem_id"] for x in roles["effect"]}
    roles["performance"] = _choose_proportional([x for x in remaining if x["problem_id"] not in effect_ids],
                                                performance_count, "v17-performance")
    return roles, {"quarantine": quarantine, "identity_closure": "ID_NATIVE_FAMILY_BYTE_PIXEL_TRANSITIVE",
        "replacement_ids": sorted(x["problem_id"] for x in roles["train"] if x["problem_id"] not in {y["problem_id"] for y in current["train"]}),
        "counts": {s: len(xs) for s, xs in roles.items()}, "effect_fresh_candidate_count": len(remaining),
        "effect_status": "EXPOSURE_HISTORY_INCOMPLETE", "effect_eligible": False,
        "effect_use": "EXTERNAL_DEVELOPMENT_COMPARISON",
        "selection_uses_model_outcomes": False}


def _entry(candidate: dict, split: str) -> dict:
    actor, verifier = candidate["actor"], candidate["verifier"]
    fields = ("problem_id", "source", "source_revision", "native_id", "family_id", "image_sha256", "pixel_sha256",
              "image_paths", "annotation_refs", "record_sha256", "rights", "human_solution", "f0_prior")
    result: dict = {key: candidate.get(key) for key in fields}
    result.update(split=split, training_allowed=split in {"train", "prep"}, benchmark_derived=False,
        human_verified=False, group_id=candidate["family_id"], layer=_layer(candidate),
        question=actor["question"], choices=actor["choices"],
        references={"raw_annotation": candidate.get("raw_annotation"), "native_metadata": candidate.get("native_metadata"),
                    "human_solution": candidate.get("human_solution")},
        evaluation={key: verifier[key] for key in ("gold", "answer_type", "accept")},
        lineage=candidate["lineage"], actor=actor, verifier=verifier)
    result["evaluation"].update(question=actor["question"], choices=actor["choices"])
    result["annotation"] = legacy.native_annotation(result)
    return result


def _summary(items: list[dict], capabilities: dict[str, dict]) -> dict:
    result = {}
    for source in sorted({x["source"] for x in items}):
        values = [capabilities[x["problem_id"]] for x in items if x["source"] == source]
        result[source] = {"candidates": len(values), "annotation_present": sum(x["annotation_present"] for x in values),
            "obs_eligible": sum(x["obs_eligible"] for x in values),
            "repair_eligible": sum(x["repair_catalog_eligible"] for x in values),
            "obs_reasons": dict(Counter(x["obs_reason"] for x in values if x["obs_reason"]))}
    return result


def training_order(candidates: list[dict], capabilities: dict[str, dict]) -> dict:
    """Reuse the three-way reference stratum for the three actual joint capabilities."""
    ordered = []
    for candidate in candidates:
        cap = capabilities[candidate["problem_id"]]
        if cap["obs_eligible"] and not cap["repair_catalog_eligible"]:
            raise ValueError("OBS_WITHOUT_REPAIR_CAPABILITY")
        label = True if cap["obs_eligible"] else None if cap["repair_catalog_eligible"] else False
        ordered.append(dict(candidate, has_annotation=label))
    order = policy.interleave(ordered, seed=SEED, batch_size=16)
    order["version"] = "v17-source-capability-prior-interleave-v1"
    order["v17_capability_label_mapping"] = {"present": "OBS_AND_REPAIR", "unknown": "REPAIR_ONLY", "absent": "NEITHER"}
    population = Counter((bool(capabilities[x["problem_id"]]["obs_eligible"]),
                          bool(capabilities[x["problem_id"]]["repair_catalog_eligible"])) for x in candidates)
    prefix: Counter = Counter()
    maximum = 0.0
    for position, row in enumerate(order["items"], 1):
        cap = capabilities[row["problem_id"]]
        key = (bool(cap["obs_eligible"]), bool(cap["repair_catalog_eligible"]))
        prefix[key] += 1
        maximum = max(maximum, *(abs(prefix[k] - position * count / len(candidates)) for k, count in population.items()))
    order["v17_capability_prefix_balance"] = {"checked_prefixes": len(candidates),
        "population_counts": {f"obs={int(k[0])},repair={int(k[1])}": n for k, n in population.items()},
        "max_abs_deviation": maximum, "stratification": "source/F0/joint_OBS_repair_capability/long_risk"}
    order["rule"] = {"version": order["version"], "base_rule": order["rule"],
                     "capability_label_mapping": order["v17_capability_label_mapping"]}
    order["rule_sha256"] = digest(order["rule"] | {"seed": SEED, "batch_size": 16})
    return order


def _copy_image(source: Path, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
        target.chmod(0o600)
    if target.is_symlink() or target.stat().st_nlink != 1 or curation._sha(target) != expected_sha256:
        raise ValueError("IMAGE_COPY_CHANGED_OR_ALIASED")


def performance_requests(roles: dict[str, list[dict]], *, benchmark_exclusion_sha256: str) -> dict:
    from fine_tuning.notebook_rl.prompts import p0_eval_text, user_message

    if len(benchmark_exclusion_sha256) != 64 or any(x not in "0123456789abcdef" for x in benchmark_exclusion_sha256):
        raise ValueError("PERFORMANCE_BENCHMARK_EXCLUSION_BINDING")
    all_items = [item for rows in roles.values() for item in rows]
    components = identity_components(all_items)
    performance = roles["performance"]
    if not 1 <= len(performance) <= 16:
        raise ValueError("PERFORMANCE_PARENT_COUNT")
    blocked = {components[x["problem_id"]] for role, rows in roles.items() if role != "performance" for x in rows}
    if any(components[x["problem_id"]] in blocked for x in performance):
        raise ValueError("PERFORMANCE_IDENTITY_OVERLAP")
    items = []
    for item in performance:
        pid = item["problem_id"]
        if item.get("training_allowed") is not False or "mmmu" in pid.lower() or pid.startswith("validation_"):
            raise ValueError("PERFORMANCE_PERMISSION_OR_BENCHMARK")
        images = [{"path": path, "sha256": sha} for path, sha in zip(item["image_paths"], item["image_sha256"], strict=True)]
        prompt = p0_eval_text(item["question"], tuple(item["choices"]))
        seed = int.from_bytes(hashlib.sha256(f"{SEED}:{pid}".encode()).digest()[:8], "big") % (2**31)
        items.append({"sample_id": pid, "messages": [user_message(prompt, len(images))], "images": images, "seed": seed})
    disjoint = {"status": "VERIFIED", "overlap_count": 0, "identity_scope": "ID_NATIVE_FAMILY_BYTE_PIXEL_TRANSITIVE",
        "benchmark_exclusion_sha256": benchmark_exclusion_sha256}
    for role, label in (("train", "main"), ("prep", "prep"), ("monitor", "development"), ("effect", "effect")):
        disjoint[label + "_ids_sha256"] = digest(sorted(x["problem_id"] for x in roles[role]))
    return _sealed({"version": "v17-performance-only-v1", "items": items, "disjoint_receipt": disjoint,
                    "diagnostic_only": True, "used_for_training_or_answer_selection": False})


def freeze(curation_root: Path, assembled_root: Path, output: Path, *, reservation_manifests: list[Path],
           tokenizer, exposure_manifests: Sequence[Path] = (), release_receipts: Sequence[Path] = (),
           validation_sha256: str = legacy.FROZEN_VALIDATION_SHA256) -> dict:
    """Reconstruct new lists from fixed, validated local inputs, without changing them."""
    curation_root, assembled_root, output = curation_root.absolute(), assembled_root.absolute(), output.absolute()
    if output.exists():
        raise FileExistsError("refusing to overwrite v17 freeze")
    if len(reservation_manifests) < 4:
        raise ValueError("HISTORICAL_RESERVATION_UNION_INCOMPLETE")
    train, monitor, _, candidates, inherited, scan, checked = legacy._checked_inputs(curation_root, validation_sha256)
    tokenizer_root = Path(tokenizer.name_or_path)
    tokenizer_files = [tokenizer_root / name for name in ("tokenizer.json", "tokenizer_config.json")]
    if not all(path.is_file() for path in tokenizer_files):
        raise ValueError("TOKENIZER_FILES_UNVERIFIED")
    for path in tokenizer_files:
        checked[str(path)] = curation._sha(path)
    rendering_refs = {}
    for manifest in reservation_manifests:
        for row in json.loads(manifest.read_text())["items"]:
            for ref in row.get("provenance", {}).get("annotation_sources", []):
                if ref.get("role") == "graph_rendering_review":
                    curation._check_ref(ref, checked)
                    rendering_refs[ref["sha256"]] = ref
    if any(x.get("raw_annotation") and x["source"] == "hf-gvlqa" for x in candidates) and not rendering_refs:
        raise ValueError("GRAPH_RENDERING_REVIEW_MISSING")
    for candidate in candidates:
        if candidate["source"] == "hf-gvlqa" and candidate.get("raw_annotation"):
            candidate["annotation_refs"] = candidate["annotation_refs"] + list(rendering_refs.values())
    prep_path = assembled_root / "prep-manifest.json"
    prep = json.loads(prep_path.read_text())["items"]
    curation._check_ref({"path": str(prep_path), "sha256": json.loads((assembled_root / "payload-receipt.json").read_text())["files"]["prep-manifest.json"]}, checked)
    reservations, reservation_report = historical_reservations(reservation_manifests, list(release_receipts))
    exposures = []
    for path in exposure_manifests:
        data = json.loads(path.read_text())
        exposures.extend(data["items"])
        checked[str(path)] = curation._sha(path)
    inventory = {x["problem_id"]: capability_manifest(_entry(x, "candidate"), tokenizer=tokenizer) for x in candidates}
    calibration = json.loads(Path(train["bindings"]["calibration"]["path"]).read_text())
    caps = {s: x["source_cap"] for s, x in calibration["sources"].items() if "source_cap" in x}
    current = {"train": train["items"], "prep": prep, "monitor": monitor["items"]}
    roles, selection = select_roles(candidates, current, reservations + inherited,
                                    capabilities=inventory, source_caps=caps, exposures=exposures)
    entries = {split: [_entry(x, split) for x in rows] for split, rows in roles.items()}
    records: dict = {}
    receipts: dict = {}
    fields = {(x["problem_id"], x["field"]): x for x in scan["fields"]}
    for rows in roles.values():
        for candidate in rows:
            legacy._checked_candidate(candidate, checked, records, receipts, fields)
    output.mkdir(mode=0o700, parents=True)
    all_entries, capability_rows, private_rows = [], [], []
    for split, rows in entries.items():
        for entry in rows:
            pid = entry["problem_id"]
            text_receipt = {"version": VERSION, "problem_id": pid, "fresh_text_scan_sha256": train["new_text_checks"]["sha256"],
                "fields": [{"field": name, "text_sha256": evidence["text_sha256"], "matches": []}
                           for (key, name), evidence in fields.items() if key == pid],
                "status": "EXCLUSION_CHECKED_WITH_LIMITS", "human_verified": False}
            text_path = output / "receipts" / f"{hashlib.sha256(pid.encode()).hexdigest()}.json"
            legacy._write(text_path, text_receipt)
            text_ref = curation._ref(text_path)
            entry["contamination_receipt"] = text_ref
            human = entry["human_solution"]
            if human:
                entry["references"]["human_solution"] = dict(human,
                    contamination_status="PASS_FRESH_TEXT_SCAN_WITH_LIMITS", text_check_receipt=text_ref,
                    text_sha256={key: hashlib.sha256(human.get(key, "").encode()).hexdigest() for key in ("lecture", "solution")})
            for relative, original, sha in zip(entry["actor"]["image_paths"], entry["image_paths"], entry["image_sha256"], strict=True):
                target = output / "vault/images" / relative
                if not target.resolve().is_relative_to(output / "vault/images"):
                    raise ValueError("IMAGE_PATH_ESCAPE")
                _copy_image(Path(original), target, sha)
            entry["image_paths"] = [str(output / "vault/images" / path) for path in entry["actor"]["image_paths"]]
            capability = capability_manifest(entry, tokenizer=tokenizer)
            verifier = private_verifier(entry, tokenizer=tokenizer)
            entry["v17_capability"] = capability
            entry["v17_private_verifier"] = verifier
            capability_rows.append(capability)
            private_rows.append(verifier)
            all_entries.append(entry)
        write_split(output / "vault" / split,
            [from_json(ActorItem, json.dumps(x["actor"])) for x in rows],
            [from_json(VerifierItem, json.dumps(x["verifier"])) for x in rows],
            [from_json(Lineage, json.dumps(x["lineage"])) for x in rows])
        legacy._write(output / ("prep-manifest.json" if split == "prep" else f"{split}-manifest.json"),
            {"version": VERSION, "split": split, "training_allowed": split in {"train", "prep"}, "items": rows})
    cap_by_id = {x["problem_id"]: x for x in capability_rows}
    order = training_order(roles["train"], cap_by_id)
    legacy._write(output / "order.json", order)
    legacy._write(output / "dataset-manifest.json", {"version": VERSION, "items": all_entries,
        "counts": selection["counts"], "public_upload_allowed": False, "human_verified": False})
    legacy._write(output / "capabilities.json", {"version": VERSION, "items": capability_rows})
    legacy._write(output / "private-verifiers.json", {"version": VERSION, "items": private_rows})
    provenance = {x["problem_id"]: {"source": x["source"], "source_revision": x["source_revision"],
        "native_id": x["native_id"], "references": x["references"], "record_sha256": x["record_sha256"],
        "image_sha256": x["image_sha256"], "reference_use": "PRIVATE_REVIEW_INPUT_ONLY_NOT_TRAINING_TARGET",
        "human_verified": False} for x in all_entries}
    legacy._write(output / "source-provenance.json", {"version": "note-source-provenance-v1", "items": provenance})
    chart_candidates = [x for x in candidates if x.get("eligible") and x["source"] == "hf-chartqa"]
    chart_audit = []
    old_ids = {x["problem_id"] for rows in current.values() for x in rows}
    for x in chart_candidates:
        _, reason, stages = chart_facts(_entry(x, "candidate"))
        chart_audit.append({"problem_id": x["problem_id"], "old_role": next((s for s, rows in current.items() if any(y["problem_id"] == x["problem_id"] for y in rows)), "unused"),
                            "previously_unused": x["problem_id"] not in old_ids, "stages": stages, "reason": reason})
    legacy._write(output / "chartqa-audit.json", {"items": chart_audit, "counts": dict(Counter(x["reason"] or "SUPPORTED" for x in chart_audit))})
    chart_stages = {}
    for role in sorted({x["old_role"] for x in chart_audit}):
        rows = [x for x in chart_audit if x["old_role"] == role]
        chart_stages[role] = {"candidates": len(rows), **{
            stage: sum(x["stages"][stage] for x in rows)
            for stage in ("annotation_present", "image_join", "visibility_and_order", "scope")},
            "obs_eligible": sum(inventory[x["problem_id"]]["obs_eligible"] for x in rows),
            "repair_eligible": sum(inventory[x["problem_id"]]["repair_catalog_eligible"] for x in rows)}
    report = {"version": VERSION, "status": "FROZEN_WITH_CAPABILITY_LIMITS", "selection": selection,
        "historical_reservations": reservation_report,
        "before": _summary([curation._by_id(candidates)[x["problem_id"]] for x in train["items"]], inventory),
        "after": _summary(roles["train"], cap_by_id),
        "chartqa_stages_by_original_role": chart_stages,
        "chartqa_audit": {"eligible_pool": len(chart_candidates), "initial_main": sum(x["old_role"] == "train" for x in chart_audit),
            "unused": sum(x["previously_unused"] for x in chart_audit)},
        "limits": ["No new contamination scan; byte-bound historical receipts are reused.",
                   "Chart OCR number occurrence does not certify position, visibility, or series/category order.",
                   "Effect sets remain external development comparisons without complete historical exposure certification."]}
    legacy._write(output / "data-report.json", report)
    legacy_files = {str(path): curation._sha(path) for path in [*reservation_manifests, prep_path,
        assembled_root / "dataset-manifest.json", curation_root / "validation.json", curation_root / "candidates.jsonl",
        curation_root / "train-3000-manifest.json", curation_root / "monitor-manifest.json", curation_root / "order.json"]}
    benchmark_binding = digest({"index": train["bindings"]["inputs"]["index"]["sha256"],
                                "fresh_text_scan": train["new_text_checks"]["sha256"]})
    legacy._write(output / "performance-requests.json", performance_requests(entries,
        benchmark_exclusion_sha256=benchmark_binding))
    files = sorted(path for path in output.rglob("*") if path.is_file())
    legacy._write(output / "payload-receipt.json", {"version": VERSION,
        "files": {str(path.relative_to(output)): curation._sha(path) for path in files},
        "legacy_files": legacy_files, "original_bindings": checked,
        "source_sha256": curation._sha(Path(__file__)), "selection": selection})
    legacy._write(output / "bundle-spec.json", {"version": legacy.SPEC_VERSION,
        "paths": [str(path) for path in files] + [str(output / "payload-receipt.json"), str(output / "bundle-spec.json")]})
    result = validate(output, tokenizer=tokenizer, verify_legacy_inputs=True)
    legacy._write(output / "validation.json", result)
    return result


def load_items(root: Path, split: str) -> list[dict]:
    split = {"main": "train", "development": "monitor"}.get(split, split)
    if split not in {"train", "prep", "monitor", "effect", "performance"}:
        raise ValueError("UNKNOWN_V17_ROLE")
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("NOT_V17_DATA")
    items = [dict(x, **{key: x["verifier"][key] for key in ("gold", "answer_type", "accept")})
             for x in manifest["items"] if x["split"] == split]
    if split == "train":
        order = json.loads((root / "order.json").read_text())
        ids = [x["problem_id"] for x in order["items"]]
        by_id = curation._by_id(items)
        if len(ids) != len(set(ids)) or set(ids) != by_id.keys():
            raise ValueError("V17_ORDER_IDS")
        if ids != [pid for batch in order["batches"] for pid in batch["problem_ids"]]:
            raise ValueError("V17_ORDER_BATCHES")
        items = [by_id[pid] for pid in ids]
    for item in items:
        if (item["v17_capability"] != capability_manifest(item)
                or item["v17_private_verifier"] != private_verifier(item)):
            raise ValueError("V17_CAPABILITY_RECOMPUTATION_MISMATCH")
    return items


def validate(root: Path, *, tokenizer=None, verify_legacy_inputs: bool = False) -> dict:
    root = root.absolute()
    receipt = json.loads((root / "payload-receipt.json").read_text())
    if receipt.get("source_sha256") != curation._sha(Path(__file__)):
        raise ValueError("V17_VERIFIER_SOURCE_CHANGED")
    checked: dict = {}
    spec_files = legacy.expand_spec(json.loads((root / "bundle-spec.json").read_text()))
    if set(spec_files) != {str(root / name) for name in receipt["files"]} | {
            str(root / "payload-receipt.json"), str(root / "bundle-spec.json")}:
        raise ValueError("V17_BUNDLE_UNBOUND_FILES")
    for relative, sha in receipt["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("PAYLOAD_PATH_ESCAPE")
        curation._check_ref({"path": str(path), "sha256": sha}, checked)
    if verify_legacy_inputs:
        for path, sha in (receipt["legacy_files"] | receipt["original_bindings"]).items():
            curation._check_ref({"path": path, "sha256": sha}, checked)
    roles = {s: load_items(root, s) for s in ("train", "prep", "monitor", "effect", "performance")}
    if [len(roles[x]) for x in ("train", "prep", "monitor")] != [3000, 512, 200]:
        raise ValueError("V17_ROLE_COUNTS")
    if not 64 <= len(roles["effect"]) <= 128 or len(roles["performance"]) != 16:
        raise ValueError("V17_HOLDOUT_COUNTS")
    all_items = [x for rows in roles.values() for x in rows]
    components = identity_components(all_items)
    if len(set(components.values())) != len(all_items):
        raise ValueError("V17_IDENTITY_OVERLAP")
    for split, rows in roles.items():
        for item in rows:
            if item["training_allowed"] is not (split in {"train", "prep"}):
                raise ValueError("V17_TRAINING_PERMISSION")
            if item["v17_capability"] != capability_manifest(item, tokenizer=tokenizer) or item["v17_private_verifier"] != private_verifier(item, tokenizer=tokenizer):
                raise ValueError("V17_CAPABILITY_RECOMPUTATION_MISMATCH")
    order = json.loads((root / "order.json").read_text())
    if set(x["problem_id"] for x in order["items"]) != {x["problem_id"] for x in roles["train"]}:
        raise ValueError("V17_ORDER_IDS")
    return {"version": VERSION, "status": "PASS_CPU_V17_DATA_WITH_LIMITS", "counts": {s: len(x) for s, x in roles.items()},
        "capability_recomputed": tokenizer is not None, "effect_status": receipt["selection"]["effect_status"], "effect_eligible": receipt["selection"]["effect_eligible"],
        "legacy_files_unchanged": True if verify_legacy_inputs else None, "human_verified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("freeze")
    for name in ("curation-root", "assembled-root", "output", "tokenizer-path"):
        make.add_argument("--" + name, type=Path, required=True)
    make.add_argument("--reservation-manifest", type=Path, action="append", required=True)
    make.add_argument("--exposure-manifest", type=Path, action="append", default=[])
    make.add_argument("--release-receipt", type=Path, action="append", default=[])
    check = sub.add_parser("validate")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--tokenizer-path", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True)
    if args.command == "freeze":
        result = freeze(args.curation_root, args.assembled_root, args.output,
            reservation_manifests=args.reservation_manifest, exposure_manifests=args.exposure_manifest,
            release_receipts=args.release_receipt, tokenizer=tokenizer)
    else:
        result = validate(args.root, tokenizer=tokenizer, verify_legacy_inputs=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
