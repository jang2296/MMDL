"""Authored v16 observation rows; source annotations, never model text, supply targets."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from difflib import SequenceMatcher
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import NoReturn

from .audit import digest, text_sha256

ROW_SCHEMA = "v16-content-row-v1"
MANIFEST_SCHEMA = "v16-content-manifest-v1"
RECEIPT_SCHEMA = "v16-content-receipt-v1"
MAX_LIST_EDGES = 30
O_INSTRUCTION = {
    "geometry": "List every label printed in the diagram (segment lengths and angle measures) "
                "and every right-angle or parallel mark. Do not solve anything.",
    "graph": "List the nodes drawn in the figure and every edge between them.",
    "chart": "State the chart type and printed axis labels, then the value shown for every series and category.",
    "scene": "List every object in the image from left to right with its size, color, material "
             "and shape.",
}
GRAPH_NEIGHBOR_INSTRUCTION = (
    "List every neighbor of the requested drawn node. Include edge direction and weight when shown."
)
PROMPTS_SHA256 = digest({"observation": O_INSTRUCTION, "neighbors": GRAPH_NEIGHBOR_INSTRUCTION})
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_GRAPH_TASKS = {"connectivity": (False, False), "cycle": (False, False), "flow": (True, True)}
_SOURCE_LAYERS = {"hf-gvlqa": "graph", "hf-geometry3k": "geometry",
                  "hf-clevr": "scene", "hf-chartqa": "chart"}


def _fail(reason: str) -> NoReturn:
    raise ValueError(f"v16 content: {reason}")


def parse_gvlqa_structure(text: str) -> dict:
    """Read exactly m edges; trailing task queries must never become edges."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    try:
        n, m = map(int, lines[0][:2])
        if n < 1 or m < 0 or len(lines) < m + 1:
            _fail("MALFORMED_GRAPH")
        body = lines[1:m + 1]
        if len({len(edge) for edge in body}) > 1:
            _fail("MALFORMED_GRAPH")
        edges = []
        for edge in body:
            if len(edge) not in (2, 3):
                _fail("MALFORMED_GRAPH")
            u, v = map(int, edge[:2])
            if not (0 <= u < n and 0 <= v < n) or u == v:
                _fail("MALFORMED_GRAPH")
            if len(edge) == 3 and not re.fullmatch(r"\d+(?:\.\d+)?", edge[2]):
                _fail("MALFORMED_GRAPH")
            edges.append((u, v, edge[2]) if len(edge) == 3 else (u, v))
        if len(set(edges)) != m:
            _fail("MALFORMED_GRAPH")
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("v16 content: MALFORMED_GRAPH") from exc
    return {"n": n, "m": m, "edges": edges,
            "visible_nodes": sorted({node for edge in edges for node in edge[:2]})}


def _query_node(struct: dict, parent_id: str) -> int:
    if not parent_id:
        _fail("MISSING_GRAPH_PARENT")
    return min(struct["visible_nodes"], key=lambda node: text_sha256(f"v16-neighbor:{parent_id}:{node}"))


def graph_target(struct: dict, *, directed: bool, parent_id: str = "") -> tuple[str | None, str | None]:
    if struct["m"] > 60:
        return None, "TOO_MANY_EDGES"
    if not struct["edges"]:
        return None, "NO_FACTS"
    canon = sorted(struct["edges"] if directed else
                   [tuple(sorted(e[:2])) + e[2:] for e in struct["edges"]])
    if len(set(canon)) != struct["m"]:
        return None, "MALFORMED_GRAPH"
    sep = "->" if directed else "-"
    if struct["m"] > MAX_LIST_EDGES:
        node = _query_node(struct, parent_id)
        canon = [edge for edge in canon if node in edge[:2]]
        heading = f"Requested node: {node}.\nIncident edges"
    else:
        heading = "Drawn nodes: " + ", ".join(map(str, struct["visible_nodes"])) + ".\nEdges"
    edges = ", ".join(f"{e[0]}{sep}{e[1]}" + (f" (weight {e[2]})" if len(e) == 3 else "")
                      for e in canon)
    return f"{'Directed' if directed else 'Undirected'} graph.\n{heading}: {edges}.", None


def link_geometry3k(rows: list[dict], intergps: dict) -> dict:
    """Image hash, answer, then unique text match with a 0.05 margin; never row number."""
    def normal(text: str) -> str:
        return re.sub(r"[\s$]+", " ", text.replace("<image>", "")).strip().lower()

    by_image: dict[str, list] = defaultdict(list)
    for pid, item in intergps.items():
        by_image[item["image_sha256"]].append(pid)
    out = {}
    for row in rows:
        candidates = [pid for pid in by_image.get(row["image_sha256"], ())
                      if str(intergps[pid]["answer_value"]).strip() == str(row["answer"]).strip()]
        scored = sorted(((max((SequenceMatcher(None, normal(row["problem"]), normal(text)).ratio()
                               for text in intergps[pid]["texts"]), default=0), str(pid), pid)
                         for pid in candidates), reverse=True)
        unique = scored and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05)
        out[row["index"]] = scored[0][2] if unique else "LINK_AMBIGUOUS"
    taken: dict[object, list] = defaultdict(list)
    for index, pid in out.items():
        taken[pid].append(index)
    for pid, indices in taken.items():
        if pid != "LINK_AMBIGUOUS" and len(indices) > 1:
            for index in indices:
                out[index] = "LINK_AMBIGUOUS"
    return out


_POINT = r"[A-Z]"
_LEN = re.compile(rf"Equals\(LengthOf\(Line\(({_POINT}),\s*({_POINT})\)\),\s*([^()]+)\)\Z")
_ANG = re.compile(rf"Equals\(MeasureOf\(Angle\(({_POINT}),\s*({_POINT}),\s*({_POINT})\)\),\s*([^()]+)\)\Z")
_REL = re.compile(rf"(Perpendicular|Parallel)\(Line\(({_POINT}),\s*({_POINT})\),\s*Line\(({_POINT}),\s*({_POINT})\)\)\Z")
_STRUCTURE = re.compile(
    rf"PointLiesOnLine\({_POINT},\s*Line\({_POINT},\s*{_POINT}\)\)\Z|"
    rf"PointLiesOnCircle\({_POINT},\s*Circle\({_POINT}(?:,\s*[A-Za-z0-9_]+)?\)\)\Z"
)
_EXPRESSION = re.compile(r"[A-Za-z0-9 .+*/^−-]+\Z")
_NUMBER = re.compile(r"\d+(?:\.\d+)?\Z")


def geometry_target(diagram_forms: list[str]) -> tuple[str | None, str | None]:
    """List all supported diagram facts or reject the diagram; text forms are never read."""
    labels, marks = [], []
    for form in diagram_forms:
        if not isinstance(form, str):
            return None, "UNSUPPORTED_FORM"
        if _STRUCTURE.fullmatch(form):
            continue
        if match := _LEN.fullmatch(form):
            if not _EXPRESSION.fullmatch(match[3]):
                return None, "UNSUPPORTED_FORM"
            labels.append(f"segment {match[1]}{match[2]} = {match[3]}")
        elif match := _ANG.fullmatch(form):
            if not _EXPRESSION.fullmatch(match[4]):
                return None, "UNSUPPORTED_FORM"
            labels.append(f"angle {match[1]}{match[2]}{match[3]} = {match[4]}"
                          + ("°" if _NUMBER.fullmatch(match[4]) else ""))
        elif match := _REL.fullmatch(form):
            marks.append(f"{match[2]}{match[3]} {'⊥' if match[1] == 'Perpendicular' else '∥'} "
                         f"{match[4]}{match[5]}")
        else:
            return None, "UNSUPPORTED_FORM"
    if not labels and not marks:
        return None, "NO_FACTS"
    return ("Printed labels: " + ("; ".join(labels) or "none") + ".\nMarks: "
            + ("; ".join(marks) or "none") + "."), None


def scene_target(scene: dict) -> tuple[str | None, str | None]:
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        return None, "NO_FACTS"
    allowed = {"size": {"small", "large"}, "color": {"gray", "red", "blue", "green", "brown",
               "purple", "cyan", "yellow"}, "material": {"rubber", "metal"},
               "shape": {"cube", "sphere", "cylinder"}}
    for obj in objects:
        coords = obj.get("pixel_coords", [])
        if (len(coords) != 3 or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                                   or not math.isfinite(x) for x in coords)
                or any(obj.get(key) not in values for key, values in allowed.items())):
            return None, "UNSUPPORTED_SCENE"
    ordered = sorted(enumerate(objects), key=lambda pair: (pair[1]["pixel_coords"][0], pair[0]))
    return "Objects (left to right):\n" + "\n".join(
        f"{i}. " + " ".join(obj[key] for key in ("size", "color", "material", "shape")) + "."
        for i, (_, obj) in enumerate(ordered, 1)), None


def chart_source(imgname: str) -> str:
    if imgname.startswith("OECD"):
        return "OECD"
    if imgname.startswith(("two_col", "multi_col")):
        return "STATISTA"
    if re.fullmatch(r"\d+\.[A-Za-z]+", imgname):
        return "PEW"
    return "OTHER"


def chart_target(table_csv: str, annotation: dict, *, imgname: str,
                 visibility_receipt: dict | None = None) -> tuple[str | None, str | None]:
    """Join every creator-annotated series/category to its exact original table value."""
    if chart_source(imgname) == "PEW":
        return None, "PEW_NO_SOURCE_TABLE"
    if chart_source(imgname) not in {"OECD", "STATISTA"}:
        return None, "UNSUPPORTED_CHART_SOURCE"
    chart_types = {"v_bar": "Bar", "h_bar": "Bar", "line": "Line"}
    if annotation.get("type") not in chart_types:
        return None, "UNSUPPORTED_CHART"
    models = annotation.get("models")
    table = list(csv.reader(io.StringIO(table_csv)))
    if (not isinstance(models, list) or not models or len(table) < 2
            or len(table[0]) != len(models) + 1 or any(len(row) != len(table[0]) for row in table)):
        return None, "TABLE_ANNOTATION_MISMATCH"

    def number(text: str):
        match = re.fullmatch(r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(%?)", text.strip())
        if not match:
            return None
        try:
            return Decimal(match[1].replace(",", "")), match[2]
        except InvalidOperation:
            return None

    header = [cell.strip() for cell in table[0]]
    categories = [row[0].strip() for row in table[1:]]
    if len(set(header[1:])) != len(models) or len(set(categories)) != len(categories):
        return None, "TABLE_ANNOTATION_MISMATCH"
    table_values = {(row[0].strip(), header[col]): row[col].strip()
                    for row in table[1:] for col in range(1, len(header))}
    series, used_names = [], set()
    for model in models:
        name = header[1] if len(models) == 1 else model.get("name")
        xs, ys = model.get("x"), model.get("y")
        if (not isinstance(name, str) or name not in header[1:] or name in used_names
                or not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys)
                or len(xs) != len(categories) or len(set(map(str, xs))) != len(xs)
                or set(map(str, xs)) != set(categories)):
            return None, "TABLE_ANNOTATION_MISMATCH"
        pairs = [(str(x), str(y).strip()) for x, y in zip(xs, ys, strict=True)]
        for category, value in pairs:
            numeric = number(value)
            if numeric is None or numeric != number(table_values[(category, name)]):
                return None, "TABLE_ANNOTATION_MISMATCH"
        used_names.add(name)
        series.append((name, pairs))
    if (not visibility_receipt or visibility_receipt.get("engine_type") != "NON_LLM_OCR"
            or visibility_receipt.get("status") != "COMPLETE"
            or not isinstance(visibility_receipt.get("text"), str)):
        return None, "VISIBILITY_UNVERIFIED"
    visible = {number(token) for token in re.findall(
        r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w.])", visibility_receipt["text"])}
    if any(number(value) not in visible for _, pairs in series for _, value in pairs):
        return None, "VISIBILITY_UNVERIFIED"
    info = annotation.get("general_figure_info", {})
    axis_labels = [info.get(axis, {}).get("label", {}).get("text") for axis in ("x_axis", "y_axis")]
    if any(label is not None and not isinstance(label, str) for label in axis_labels):
        return None, "UNSUPPORTED_CHART"
    labels = [label for label in axis_labels if label and label.strip()]
    if not labels:
        return None, "UNSUPPORTED_CHART"
    lines = [f"{chart_types[annotation['type']]} chart.", "Axis labels: " + "; ".join(labels) + "."]
    for name, pairs in series:
        lines.append(f"Series: {name}.")
        lines.extend(f"{category}: {value}." for category, value in pairs)
    return "\n".join(lines), None


def content_prompt(item: dict, layer: str) -> str:
    if layer == "graph":
        struct = parse_gvlqa_structure(item["annotation"]["structure_text"])
        if MAX_LIST_EDGES < struct["m"] <= 60:
            return GRAPH_NEIGHBOR_INSTRUCTION + f"\nRequested node: {_query_node(struct, item['parent_id'])}."
    return O_INSTRUCTION[layer]


def _canonical_target(item: dict, layer: str) -> str:
    annotation = item["annotation"]
    if layer == "graph":
        task = annotation["task"]
        if (type(annotation["directed"]) is not bool or type(annotation["weighted"]) is not bool
                or task not in _GRAPH_TASKS or (annotation["directed"], annotation["weighted"])
                != _GRAPH_TASKS[task] or item["provenance"].get("rendering_verified") is not True):
            _fail("UNVERIFIED_GRAPH_RENDERING")
        struct = parse_gvlqa_structure(annotation["structure_text"])
        if any((len(edge) == 3) != annotation["weighted"] for edge in struct["edges"]):
            _fail("MALFORMED_GRAPH")
        identity = f"{task}-{annotation['level']}-{annotation['graph_id']}"
        if item["group_id"] != identity or item.get("visible_nodes") != struct["visible_nodes"]:
            _fail("GRAPH_BINDING_MISMATCH")
        target, reason = graph_target(struct, directed=annotation["directed"], parent_id=item["parent_id"])
    elif layer == "geometry":
        local, candidates = annotation["local"], annotation["candidates"]
        joined = link_geometry3k([local], candidates)[local["index"]]
        if joined == "LINK_AMBIGUOUS" or str(joined) != str(annotation["matched_id"]):
            _fail("LINK_AMBIGUOUS")
        if (local["image_sha256"] != item.get("source_image_sha256", item["image_sha256"])
                or item["group_id"] != local["image_sha256"]):
            _fail("GEOMETRY_IMAGE_MISMATCH")
        if item.get("annotation_process_documented") is not False:
            _fail("ANNOTATION_PROCESS_UNDOCUMENTED")
        join = {"local": local, "candidates": candidates, "matched_id": joined}
        if (item.get("join_method") != "image-answer-text-v1"
                or item.get("join_receipt_sha256") != digest(join)):
            _fail("JOIN_RECEIPT_MISMATCH")
        target, reason = geometry_target(candidates[joined]["diagram_logic_forms"])
    elif layer == "scene":
        if (annotation.get("split") != "train" or not annotation.get("image_filename")
                or annotation["image_filename"] != item.get("image_filename")
                or item["group_id"] != annotation["image_filename"]):
            _fail("SCENE_IMAGE_MISMATCH")
        target, reason = scene_target(annotation)
    elif layer == "chart":
        name = item["image_filename"]
        visibility = annotation.get("visibility_receipt", {})
        if (item["group_id"] != name or item.get("chart_source") != chart_source(name)
                or visibility.get("image_sha256") != item["image_sha256"]
                or visibility.get("annotation_sha256") != digest(annotation["chart"])):
            _fail("CHART_BINDING_MISMATCH")
        target, reason = chart_target(annotation["table_csv"], annotation["chart"], imgname=name,
                                      visibility_receipt=visibility)
    else:
        _fail("UNSUPPORTED_LAYER")
    if reason or not target:
        _fail(reason or "NO_FACTS")
    return target


def _item_bindings(item: dict) -> dict:
    fields = ("problem_id", "parent_id", "group_id", "source_repo", "source_revision", "split",
              "original_split", "training_allowed", "label_source", "image_sha256")
    bindings = {key: item[key] for key in fields}
    provenance = item["provenance"]
    if (item["split"] not in {"train", "dev", "sentinel", "sealed"} or item["original_split"] != "train"
            or item["training_allowed"] is not True
            or item["label_source"] not in {"PROGRAM_EXACT", "HUMAN_ANNOTATION"}
            or not all(isinstance(item[key], str) and item[key] for key in fields[:5])
            or not _SHA.fullmatch(item["image_sha256"])
            or provenance.get("parent_status") not in {"INCLUDED", "EXCLUDED_QUOTA"}
            or not all(_SHA.fullmatch(str(provenance.get(key, ""))) for key in
                       ("parent_manifest_sha256", "parent_record_sha256"))
            or not provenance.get("annotation_sources")
            or item.get("source_split", "train") != "train"
            or item.get("benchmark_derived", False) is not False):
        _fail("UNADMITTED_SOURCE")
    if re.search(r"mmmu|mmstar|mathvista|mathverse", item["source_repo"], re.I):
        _fail("BENCHMARK_DERIVED")
    bindings.update(annotation_sha256=digest(item["annotation"]), provenance_sha256=digest(provenance),
                    evaluation_sha256=digest(item.get("evaluation")),
                    evaluation_mirror_sha256=digest({key: item.get(key) for key in
                        ("question", "choices", "gold", "answer_type", "accept")}))
    return bindings


def _receipt(item: dict, layer: str, target: str) -> dict:
    return {"schema": RECEIPT_SCHEMA, "verdict": "VERIFIED", "validator": "content_v16",
            "layer": layer, **_item_bindings(item), "training_prompt_sha256": text_sha256(content_prompt(item, layer)),
            "target_sha256": text_sha256(target), "generation_sha256": None,
            "generation_prompt_sha256": None, "human_verified": False}


def observation_row(item: dict, target: str | None = None, *, layer: str,
                    receipt: dict | None = None, purpose: str = "training") -> dict:
    if purpose not in {"training", "diagnostic"}:
        _fail("UNSUPPORTED_PURPOSE")
    canonical = _canonical_target(item, layer)
    if target is not None and target != canonical:
        _fail("TARGET_MISMATCH")
    expected_receipt = _receipt(item, layer, canonical)
    if receipt is not None and receipt != expected_receipt:
        _fail("RECEIPT_MISMATCH")
    row = {"schema": ROW_SCHEMA, "kind": "O", "route": "FULL", "row_id": f"O:{item['problem_id']}",
           "problem_id": item["problem_id"], "parent_id": item["parent_id"], "group_id": item["group_id"],
           "source": item["source_repo"], "source_revision": item["source_revision"], "layer": layer,
           "image_sha256": item["image_sha256"], "prompt": content_prompt(item, layer), "target": canonical,
           "target_origin": "program", "verification_receipt": expected_receipt,
           "split": item["split"], "purpose": purpose}
    if purpose == "training" and item["split"] != "train":
        _fail("NON_TRAIN_ROW")
    return row


def validate_content_row(row: dict, item: dict, *, purpose: str = "training") -> None:
    """Regenerate target and receipt; a caller-provided VERIFIED string conveys no trust."""
    try:
        expected = observation_row(item, layer=row["layer"], purpose=purpose)
        if row != expected:
            _fail("ROW_BINDING_MISMATCH")
    except (KeyError, TypeError) as exc:
        raise ValueError("v16 content: MALFORMED_ROW") from exc


def validate_content_manifest(manifest: dict) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("prompts_sha256") != PROMPTS_SHA256:
        _fail("MANIFEST_SCHEMA_MISMATCH")
    items = manifest.get("items", [])
    rows = manifest.get("rows", [])
    by_id = {item["problem_id"]: item for item in items}
    if not items or not rows or len(by_id) != len(items):
        _fail("DUPLICATE_OR_EMPTY_MANIFEST")
    groups: dict[tuple[str, str], str] = {}
    for item in items:
        _item_bindings(item)
        for key in ("parent_id", "group_id", "image_sha256"):
            group = (key, item[key])
            if group in groups and groups[group] != item["split"]:
                _fail("SPLIT_OVERLAP")
            groups[group] = item["split"]
    seen = set()
    seen_groups = set()
    for row in rows:
        if (row["problem_id"] not in by_id or row["parent_id"] in seen
                or (row["layer"], row["group_id"]) in seen_groups):
            _fail("DUPLICATE_OR_UNKNOWN_PARENT")
        validate_content_row(row, by_id[row["problem_id"]])
        seen.add(row["parent_id"])
        seen_groups.add((row["layer"], row["group_id"]))


def validate_human_j_capability(manifest: dict, *, requested_weight: float = 0.0) -> dict:
    """Current O source decoders authenticate observations, not dataset-authored human solutions."""
    if (isinstance(requested_weight, bool) or not isinstance(requested_weight, (int, float))
            or not math.isfinite(requested_weight) or requested_weight < 0):
        _fail("INVALID_J_WEIGHT")
    capability = {"status": "HOLD", "reason": "NO_AUTHENTICATED_HUMAN_J_SOURCE",
                  "eligible_parent_ids": [],
                  "source_repos": sorted({item["source_repo"] for item in manifest.get("items", [])})}
    if requested_weight > 0:
        _fail("J_CAPABILITY_HOLD:NO_AUTHENTICATED_HUMAN_J_SOURCE")
    return capability


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_content_manifest(path: str | Path, expected_sha256: str) -> dict:
    """Verify the pinned file, approved parent records, source annotations and image bytes."""
    path = Path(path)
    if (path.is_symlink() or not _SHA.fullmatch(expected_sha256)
            or _file_sha256(path) != expected_sha256):
        _fail("MANIFEST_HASH_MISMATCH")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_content_manifest(manifest)
    checked: dict[str, str] = {}
    parent_documents: dict[str, dict] = {}
    source_documents: dict[str, object] = {}
    parent_rows: dict[str, dict] = {}
    geometry_catalogs: dict[tuple[str, str], dict] = {}

    def checked_path(source: dict) -> Path:
        file = Path(source["path"])
        if not file.is_absolute():
            file = path.parent / file
        key = str(file)
        if file.is_symlink() or not file.is_file():
            _fail("SOURCE_FILE_HASH_MISMATCH")
        if key not in checked:
            checked[key] = _file_sha256(file)
        if checked[key] != source["sha256"]:
            _fail("SOURCE_FILE_HASH_MISMATCH")
        return file

    def source_json(file: Path):
        key = str(file)
        if key not in source_documents:
            source_documents[key] = json.loads(file.read_text(encoding="utf-8"))
        return source_documents[key]

    for item in manifest["items"]:
        if "human_solution" in item:
            _fail("HUMAN_SOLUTION_SOURCE_UNSUPPORTED")
        checked_path({"path": item["image_path"], "sha256": item["image_sha256"]})
        provenance = item["provenance"]
        parent_path = checked_path({"path": provenance["parent_manifest_path"],
                                    "sha256": provenance["parent_manifest_sha256"]})
        if str(parent_path) not in parent_documents:
            parent_documents[str(parent_path)] = json.loads(parent_path.read_text(encoding="utf-8"))
        parents = parent_documents[str(parent_path)]["items"]
        matches = [parent for parent in parents if parent["problem_id"] == item["parent_id"]]
        if len(matches) != 1:
            _fail("PARENT_BINDING_MISMATCH")
        parent = matches[0]
        if (digest(parent) != provenance["parent_record_sha256"]
                or parent["status"] != provenance["parent_status"]
                or parent["split"] != item["original_split"]
                or parent["source_revision"] != item["source_revision"]
                or item["image_sha256"] not in parent["image_sha256"]):
            _fail("PARENT_BINDING_MISMATCH")
        roles = {}
        for source in provenance["annotation_sources"]:
            file = checked_path(source)
            role = source.get("role")
            if role in roles:
                _fail("DUPLICATE_SOURCE_ROLE")
            roles[role] = file
        annotation = item["annotation"]
        records_by_role = {}
        for role in ("actor", "verifier"):
            record_path = Path(provenance[f"{role}_path"])
            if not record_path.is_absolute():
                record_path = path.parent / record_path
            if str(record_path) not in parent_rows:
                records = {}
                for line in record_path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    if record["problem_id"] in records:
                        _fail("DUPLICATE_PARENT_RECORD")
                    records[record["problem_id"]] = (record, text_sha256(line))
                parent_rows[str(record_path)] = records
            record, record_sha = parent_rows[str(record_path)][item["parent_id"]]
            if record_sha != parent["record_sha256"][role]:
                _fail("PARENT_RECORD_HASH_MISMATCH")
            records_by_role[role] = record
        actor, verifier = records_by_role["actor"], records_by_role["verifier"]
        evaluation = {"question": actor["question"], "choices": actor["choices"],
                      "gold": verifier["gold"], "answer_type": verifier["answer_type"],
                      "accept": verifier["accept"]}
        if (any(item.get(key) != value for key, value in evaluation.items())
                or item.get("evaluation") != evaluation):
            _fail("EVALUATION_BINDING_MISMATCH")
        if "local" in annotation:
            def normalize(text: str) -> str:
                return " ".join(text.replace("<image>", "").split())
            if normalize(actor["question"]) != normalize(annotation["local"]["problem"]):
                _fail("LOCAL_QUESTION_MISMATCH")
            if str(verifier["gold"]).strip() != str(annotation["local"]["answer"]).strip():
                _fail("LOCAL_ANSWER_MISMATCH")
        if "graph_structure" in roles:
            if _SOURCE_LAYERS.get(parent.get("source")) != "graph":
                _fail("PARENT_SOURCE_MISMATCH")
            identity = item["group_id"]
            if not (parent["native_id"] == identity or parent["native_id"].startswith(identity + "-")):
                _fail("GRAPH_BINDING_MISMATCH")
            if "graph_rendering_review" not in roles:
                _fail("UNVERIFIED_GRAPH_RENDERING")
            review = source_json(roles["graph_rendering_review"])
            samples = [sample for sample in review.get("samples", []) if sample["task"] == annotation["task"]]
            if (review.get("version") != "v16-graph-rendering-task-spotcheck-v1"
                    or review.get("human_verified") is not False or not samples
                    or any((sample.get("directed"), sample.get("weighted")) !=
                           (annotation["directed"], annotation["weighted"]) for sample in samples)):
                _fail("UNVERIFIED_GRAPH_RENDERING")
            if (roles["graph_structure"].stem != identity
                    or roles["graph_structure"].read_text(encoding="utf-8") != annotation["structure_text"]):
                _fail("ANNOTATION_SOURCE_MISMATCH")
        elif "scene_json" in roles:
            if _SOURCE_LAYERS.get(parent.get("source")) != "scene":
                _fail("PARENT_SOURCE_MISMATCH")
            document = source_json(roles["scene_json"])
            scenes = document.get("scenes", []) if isinstance(document, dict) else []
            matches = [scene for scene in scenes if scene["image_filename"] == item["image_filename"]]
            if len(matches) != 1 or matches[0] != annotation:
                _fail("ANNOTATION_SOURCE_MISMATCH")
            if "scene_questions_json" not in roles:
                _fail("MISSING_SCENE_JOIN_SOURCE")
            questions = source_json(roles["scene_questions_json"])["questions"]
            question_index = int(parent["native_id"].split(":")[1])
            questions = [question for question in questions if question["question_index"] == question_index]
            if (len(questions) != 1 or questions[0]["image_filename"] != item["image_filename"]
                    or questions[0]["image_index"] != annotation["image_index"]):
                _fail("SCENE_IMAGE_MISMATCH")
            actor = parent_rows[str(Path(provenance["actor_path"]) if Path(provenance["actor_path"]).is_absolute()
                                    else path.parent / provenance["actor_path"])][item["parent_id"]][0]
            if " ".join(questions[0]["question"].split()) != " ".join(actor["question"].split()):
                _fail("LOCAL_QUESTION_MISMATCH")
        elif {"geometry_problem_zip", "geometry_logic_zip"} <= roles.keys():
            if _SOURCE_LAYERS.get(parent.get("source")) != "geometry":
                _fail("PARENT_SOURCE_MISMATCH")
            key = (str(roles["geometry_problem_zip"]), str(roles["geometry_logic_zip"]))
            if key not in geometry_catalogs:
                catalog = {}
                with zipfile.ZipFile(key[1]) as logic_zip, zipfile.ZipFile(key[0]) as problem_zip:
                    forms = json.loads(logic_zip.read("logic_forms/diagram_logic_forms_annot.json"))
                    for member in problem_zip.namelist():
                        if not re.fullmatch(r"train/\d+/data.json", member):
                            continue
                        data = json.loads(problem_zip.read(member))
                        pid = member.split("/")[1]
                        image = problem_zip.read(f"train/{pid}/img_diagram.png")
                        catalog[pid] = {
                            "image_sha256": hashlib.sha256(image).hexdigest(),
                            "answer_value": data["choices"][ord(data["answer"]) - ord("A")],
                            "texts": [data["annotat_text"], data["problem_text"]],
                            "diagram_logic_forms": forms[pid]["diagram_logic_forms"],
                        }
                geometry_catalogs[key] = catalog
            local = annotation["local"]
            candidates = {pid: data for pid, data in geometry_catalogs[key].items()
                          if data["image_sha256"] == local["image_sha256"]}
            if candidates != annotation["candidates"]:
                _fail("ANNOTATION_SOURCE_MISMATCH")
        elif {"chart_table_csv", "chart_annotation_json", "chart_human_questions_json",
              "chart_visibility_ocr"} <= roles.keys():
            if _SOURCE_LAYERS.get(parent.get("source")) != "chart":
                _fail("PARENT_SOURCE_MISMATCH")
            if (roles["chart_table_csv"].read_text(encoding="utf-8") != annotation["table_csv"]
                    or source_json(roles["chart_annotation_json"]) != annotation["chart"]
                    or source_json(roles["chart_visibility_ocr"]) !=
                       annotation["visibility_receipt"]):
                _fail("ANNOTATION_SOURCE_MISMATCH")
            raw_questions = source_json(roles["chart_human_questions_json"])
            actor_path = Path(provenance["actor_path"])
            verifier_path = Path(provenance["verifier_path"])
            actor = parent_rows[str(actor_path if actor_path.is_absolute() else path.parent / actor_path)][item["parent_id"]][0]
            verifier = parent_rows[str(verifier_path if verifier_path.is_absolute() else path.parent / verifier_path)][item["parent_id"]][0]
            matches = [question for question in raw_questions
                       if " ".join(question["query"].split()) == " ".join(actor["question"].split())
                       and str(question["label"]).strip() == str(verifier["gold"]).strip()]
            if not matches or {question["imgname"] for question in matches} != {item["image_filename"]}:
                _fail("CHART_BINDING_MISMATCH")
        else:
            _fail("UNSUPPORTED_ANNOTATION_SOURCE")
    if "admission" in manifest:
        from .data_v16 import validate_admission_binding
        validate_admission_binding(manifest)
    return manifest
