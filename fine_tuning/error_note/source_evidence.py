"""Prepare private, source-bound review inputs and answer rules; never training targets.

Only the supplied frozen train order is used. Cached annotations remain raw references:
their availability does not certify a generated note, and no review is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from fine_tuning.notebook_rl.answers import compare_training
from fine_tuning.notebook_rl.schema import to_json
from fine_tuning.notebook_rl.vault import _read, _safe_root, read_actors, read_lineage, read_verifier

from .answer_contract import VERSION as RULES_VERSION
from .audit import digest, text_sha256
from .collect_v3 import file_sha
from .order import read_order

VERSION = "note-source-provenance-v1"
_REPOS = {"hf-scienceqa": "derek-thomas/ScienceQA", "hf-geometry3k": "hiyouga/geometry3k",
          "hf-chebi-role": "liupf/ChEBI-20-MM", "hf-slake": "BoKelvin/SLAKE",
          "hf-circuitvqa": "MirandaAbhilash/circuitvqa-dataset",
          "hf-clevr": "HuggingFaceM4/clevr", "hf-tatdqa-pdf": "next-tat/TAT-DQA",
          "hf-tatqa-table": "HuggingFaceM4/the_cauldron"}
_COLUMNS = {"hf-scienceqa": ["question", "choices", "answer", "hint", "lecture", "solution", "skill"],
            "hf-geometry3k": ["problem", "answer"]}


def _json(path: Path) -> dict:
    return json.loads(_read(path))


def _checked_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("source evidence: unsafe relative path")
    path = root / rel
    _safe_root(path.parent)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source evidence: regular source/image file required")
    return path


def _new_native_rows(root: Path, source: str, revision: str, wanted: set[str], entries: list[dict],
                     files: dict[str, str], prefix: str) -> dict:
    """Pin native programs/scenes or native PDF pages, including exact actor-image reconstruction."""
    from fine_tuning.notebook_rl.adapters.hf_common import normalize_image
    from fine_tuning.notebook_rl.adapters.hf_registry import sources

    spec, _ = sources()[source]
    if revision != spec.revision:
        raise ValueError("source evidence: new source revision differs from pinned source")
    repo = spec.repo

    def checked(relative: str) -> tuple[Path, str, str]:
        path = _checked_file(root, f"{repo.replace('/', '__')}/{relative}")
        expected = {row["sha256"] for row in entries if row["repo"] == repo and row["revision"] == revision
                    and row.get("path") == relative}
        sha = file_sha(path)
        if expected != {sha}:
            raise ValueError("source evidence: cached source hash/revision mismatch")
        location = prefix + path.relative_to(root).as_posix()
        files[location] = sha
        return path, location, sha

    native = {}
    if source == "hf-clevr":
        path, relative, sha = checked("questions/CLEVR_train_questions.json")
        scene_path, scene_relative, scene_sha = checked("scenes/CLEVR_train_scenes.json")
        scenes = {row["image_index"]: row for row in json.loads(_read(scene_path))["scenes"]}
        for question in json.loads(_read(path))["questions"]:
            key = f"train:{question['question_index']}"
            if key not in wanted:
                continue
            filename = question["image_filename"]
            if re.fullmatch(r"CLEVR_train_\d{6}\.png", filename) is None:
                raise ValueError("source evidence: unsafe CLEVR train image name")
            image, image_relative, image_sha = checked(f"images/train/{filename}")
            row = dict(question, _scene=scenes[question["image_index"]], _scene_relative=scene_relative,
                       _scene_sha256=scene_sha, _image_relative=image_relative, _image_sha256=image_sha,
                       _normalized_image_sha256=hashlib.sha256(normalize_image(_read(image))).hexdigest())
            if (source, key) in native:
                raise ValueError("source evidence: duplicate native id")
            native[source, key] = row, relative, sha
    elif source == "hf-tatqa-table":
        from fine_tuning.notebook_rl.adapters.hf_tatqa_table import (
            AUTHOR_REPO, AUTHOR_REVISION, AUTHOR_SHA256, MIRROR_SHARD, iter_table_records,
        )
        from fine_tuning.notebook_rl.adapters.hf_common import split_for

        _, mirror_relative, mirror_sha = checked(MIRROR_SHARD)
        author_relative = "NExTplusplus__TAT-QA/dataset_raw/tatqa_dataset_train.json"
        author = _checked_file(root, author_relative)
        author_sha = file_sha(author)
        expected = {r["sha256"] for r in entries if r["repo"] == AUTHOR_REPO and r["revision"] == AUTHOR_REVISION
                    and r.get("path") == "dataset_raw/tatqa_dataset_train.json"}
        if expected != {author_sha} or author_sha != AUTHOR_SHA256:
            raise ValueError("source evidence: TAT-QA primary annotation hash/revision mismatch")
        author_relative = prefix + author_relative
        files[author_relative] = author_sha
        original = json.loads(_read(author))
        for candidate, origin in iter_table_records(root / repo.replace("/", "__")):
            key = candidate.native_id
            if key not in wanted:
                continue
            if split_for(candidate) != origin["canonical_split"]:
                raise ValueError("source evidence: TAT-QA canonical ancestor split mismatch")
            entry = original[origin["original_entry_index"]]
            question = entry["questions"][origin["mirror_turn_index"]]
            row = dict(question, _native_table=entry["table"], _origin=origin,
                       _candidate_question=candidate.question, _candidate_gold=candidate.gold,
                       _candidate_answer_type=candidate.answer_type, _candidate_accept=candidate.accept,
                       _normalized_image_sha256=hashlib.sha256(normalize_image(candidate.images[0])).hexdigest(),
                       _mirror_relative=mirror_relative, _mirror_sha256=mirror_sha)
            if (source, key) in native:
                raise ValueError("source evidence: duplicate native table id")
            native[source, key] = row, author_relative, author_sha
    else:
        from fine_tuning.notebook_rl.adapters.hf_docs import render_pdf_pages

        path, relative, sha = checked("tatdqa_dataset_train.json")
        archive, archive_relative, archive_sha = checked("tatdqa_docs_train.zip")
        with zipfile.ZipFile(archive) as documents:
            for entry in json.loads(_read(path)):
                doc = entry["doc"]
                for question in entry["questions"]:
                    key = f"{doc['uid']}:{question['uid']}"
                    if key not in wanted:
                        continue
                    member = f"train/{doc['uid']}.pdf"
                    if re.fullmatch(r"train/[A-Za-z0-9_-]+\.pdf", member) is None:
                        raise ValueError("source evidence: unsafe native PDF member")
                    pdf = documents.read(member)
                    layout_member = f"train/{doc['uid']}.json"
                    layout = documents.read(layout_member)
                    page_count = len(json.loads(layout)["pages"])
                    row = dict(question, _doc=doc, _pdf_member=member, _archive_relative=archive_relative,
                               _archive_sha256=archive_sha, _pdf_sha256=hashlib.sha256(pdf).hexdigest(),
                               _native_pdf_page_count=page_count, _layout_member=layout_member,
                               _layout_sha256=hashlib.sha256(layout).hexdigest(),
                               _normalized_image_sha256=[hashlib.sha256(normalize_image(raw)).hexdigest()
                                                         for raw in render_pdf_pages(pdf, page_count)])
                    if (source, key) in native:
                        raise ValueError("source evidence: duplicate native id")
                    native[source, key] = row, relative, sha
    if not all((source, key) in native for key in wanted):
        raise ValueError("source evidence: native id absent from pinned train source")
    return native


def _native_rows(raw_root: Path, selected: dict[str, dict], *,
                 supplementary_raw_root: Path | None = None) -> tuple[dict, dict]:
    """Read pinned native annotations; an explicit second root may supply a new source."""
    download = raw_root / "download_manifest.jsonl"
    entries = [json.loads(line) for line in _read(download).splitlines() if line.strip()]
    native = {}
    files: dict[str, str] = {}
    cache = {"download_manifest_sha256": file_sha(download), "files": files}
    for source in sorted({row["source"] for row in selected.values()} & _REPOS.keys()):
        wanted = {row["native_id"] for row in selected.values() if row["source"] == source}
        revisions = {row["source_revision"] for row in selected.values() if row["source"] == source}
        if len(revisions) != 1:
            raise ValueError("source evidence: mixed source revisions")
        revision = next(iter(revisions))
        repo = _REPOS[source]
        source_root, source_entries = raw_root, entries
        if source in ("hf-clevr", "hf-tatdqa-pdf", "hf-tatqa-table"):
            if source in ("hf-clevr", "hf-tatqa-table") and supplementary_raw_root is not None:
                source_root = _safe_root(supplementary_raw_root)
                journal = _checked_file(source_root, "download_manifest.jsonl")
                source_entries = [json.loads(line) for line in _read(journal).splitlines() if line.strip()]
                cache["supplementary_download_manifest_sha256"] = file_sha(journal)
            native.update(_new_native_rows(source_root, source, revision, wanted, source_entries, files,
                                          "supplementary/" if source_root != raw_root else ""))
            continue
        if source == "hf-circuitvqa":
            from fine_tuning.notebook_rl.adapters.hf_circuit import SPECS

            if revision != SPECS[source].revision:
                raise ValueError("source evidence: CircuitVQA revision differs from pinned source")
            if supplementary_raw_root is not None:
                source_root = _safe_root(supplementary_raw_root)
                journal = _checked_file(source_root, "download_manifest.jsonl")
                source_entries = [json.loads(line) for line in _read(journal).splitlines() if line.strip()]
                cache["supplementary_download_manifest_sha256"] = file_sha(journal)
        repo_root = source_root / repo.replace("/", "__")
        pattern = ("data/train-*.parquet" if source in _COLUMNS else "train.csv" if source == "hf-chebi-role"
                   else "questions/master_bbox.json" if source == "hf-circuitvqa" else "train.json")
        paths = sorted(repo_root.glob(pattern))
        if not paths:
            continue
        if len(paths) != 1:
            raise ValueError("source evidence: ambiguous native row numbering")
        path = paths[0]
        relative = path.relative_to(source_root).as_posix()
        _checked_file(source_root, relative)
        expected = {row["sha256"] for row in source_entries if row["repo"] == repo and row["revision"] == revision
                    and row.get("path") == path.relative_to(repo_root).as_posix()}
        sha = file_sha(path)
        if expected != {sha}:
            raise ValueError("source evidence: cached source hash/revision mismatch")
        if source_root != raw_root:
            relative = f"supplementary/{relative}"
        files[relative] = sha
        rows: Iterable[dict]
        if source in _COLUMNS:
            import pyarrow.parquet as pq

            table = pq.ParquetFile(path)
            columns = [column for column in _COLUMNS[source] if column in table.schema_arrow.names]
            rows = (row for batch in table.iter_batches(columns=columns) for row in batch.to_pylist())
        elif source == "hf-chebi-role":
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            rows = json.loads(_read(path))
        for index, row in enumerate(rows):
            key = (f"cid:{int(row['CID'])}" if source == "hf-chebi-role" else
                   f"train:{row['qid']}" if source == "hf-slake" else f"train:{index}")
            if key not in wanted:
                continue
            if (source, key) in native:
                raise ValueError("source evidence: duplicate native id")
            native[source, key] = (row, relative, sha)
        if not all((source, key) in native for key in wanted):
            raise ValueError("source evidence: native id absent from pinned train source")
    return native, cache


def _annotations(actor, item, frozen: dict, native: dict) -> tuple[list[dict], str]:
    entry = native.get((actor.source, frozen["native_id"]))
    if entry is None:
        return [], "SOURCE_CACHE_UNAVAILABLE" if actor.source in _REPOS else "LABEL_ONLY"
    row, relative, sha = entry
    source = actor.source
    scope = "LABEL_ONLY"
    if source == "hf-tatqa-table":
        if (tuple(actor.image_sha256) != (row["_normalized_image_sha256"],)
                or actor.question != row["_candidate_question"] or item.gold != row["_candidate_gold"]
                or actor.choices or item.choices or actor.answer_format != row["_candidate_answer_type"]
                or item.answer_type != row["_candidate_answer_type"] or item.accept != row["_candidate_accept"]
                or row["answer_from"] != "table" or row["rel_paragraphs"] not in ([], None)):
            raise ValueError("source evidence: native table question/label/scale/image binding mismatch")
        names = ["uid", "question", "answer", "answer_type", "answer_from", "scale", "rel_paragraphs", "derivation"]
        references = [{"field": name, "value": row[name], "sha256": digest(row[name]),
                       "location": f"raw:{relative}#{row['uid']}:{name}", "file_sha256": sha}
                      for name in names if name in row and row[name] not in (None, "", [])]
        references.extend([{"field": "native_table", "value": row["_native_table"], "sha256": digest(row["_native_table"]),
                            "location": f"raw:{relative}#table:{row['_native_table']['uid']}", "file_sha256": sha},
                           {"field": "canonical_ancestry", "value": row["_origin"], "sha256": digest(row["_origin"]),
                            "location": f"raw:{row['_mirror_relative']}#{frozen['native_id']}",
                            "file_sha256": row["_mirror_sha256"]}])
        return references, "SOURCE_NATIVE_TABLE_ANNOTATIONS_FOR_REVIEW"
    if source in ("hf-clevr", "hf-tatdqa-pdf"):
        expected_images = ((row["_normalized_image_sha256"],) if source == "hf-clevr"
                           else tuple(row["_normalized_image_sha256"]))
        if tuple(actor.image_sha256) != expected_images:
            raise ValueError("source evidence: reconstructed native image binding mismatch")
        if source == "hf-clevr":
            from fine_tuning.notebook_rl.adapters.hf_clevr import program_answer

            scene = row["_scene"]
            answer = program_answer(scene, row["program"])
            choices = ("Yes", "No") if answer in ("yes", "no") else ()
            gold = ("A" if answer == "yes" else "B") if choices else answer
            kind = "mcq" if choices else "int" if re.fullmatch(r"\d+", answer) else "short_text"
            if (row["split"] != "train" or scene["split"] != "train" or answer != row["answer"]
                    or scene["image_filename"] != row["image_filename"] or scene["image_index"] != row["image_index"]
                    or row["question"].strip() != actor.question or choices != actor.choices
                    or gold != item.gold or kind != item.answer_type or kind != actor.answer_format):
                raise ValueError("source evidence: CLEVR native program/question/gold binding mismatch")
            names = ["question", "answer", "program", "image_index", "image_filename", "question_family_index"]
            scope = "SOURCE_EXECUTED_SPATIAL_PROGRAM_FOR_REVIEW"
            extra = [{"field": "scene", "value": scene, "sha256": digest(scene),
                      "location": f"raw:{row['_scene_relative']}#{row['image_index']}",
                      "file_sha256": row["_scene_sha256"]}]
        else:
            from fine_tuning.notebook_rl.adapters.hf_docs import _tatdqa_candidate
            from fine_tuning.notebook_rl.adapters.hf_common import split_for

            doc = row["_doc"]
            candidate = _tatdqa_candidate(doc["uid"], doc["source"], row, b"native-binding-only")
            if (candidate is None or split_for(candidate) != "train"
                    or candidate.question != actor.question or candidate.gold != item.gold
                    or item.answer_type != candidate.answer_type or actor.answer_format != candidate.answer_type
                    or actor.choices or item.choices or item.accept != candidate.accept):
                raise ValueError("source evidence: TAT-DQA native question/scale/gold binding mismatch")
            names = ["uid", "question", "answer", "answer_type", "scale", "facts", "derivation", "block_mapping"]
            scope = "SOURCE_NATIVE_PDF_FINANCIAL_ANNOTATIONS_FOR_REVIEW"
            pdf_page = {"member": row["_pdf_member"], "native_page_count": row["_native_pdf_page_count"],
                        "doc_page_metadata": doc["page"], "pages": list(range(1, row["_native_pdf_page_count"] + 1)),
                        "pdf_sha256": row["_pdf_sha256"], "native_layout_member": row["_layout_member"],
                        "native_layout_sha256": row["_layout_sha256"]}
            extra = [{"field": "doc", "value": doc, "sha256": digest(doc),
                      "location": f"raw:{relative}#{doc['uid']}:doc", "file_sha256": sha},
                     {"field": "pdf_page", "value": pdf_page,
                      "location": f"raw:{row['_archive_relative']}!{row['_pdf_member']}#pages=1..{row['_native_pdf_page_count']}",
                      "file_sha256": row["_archive_sha256"], "sha256": digest(pdf_page)}]
        references = [{"field": name, "value": row[name], "sha256": digest(row[name]),
                       "location": f"raw:{relative}#{frozen['native_id']}:{name}", "file_sha256": sha}
                      for name in names if name in row and row[name] not in (None, "", [])]
        return references + extra, scope
    if source == "hf-scienceqa":
        hint = row.get("hint") or ""
        question = (f"Context: {hint}\n" if hint.strip() else "") + row["question"]
        if (question != actor.question or tuple(row["choices"]) != actor.choices
                or "ABCDEFGHIJ"[row["answer"]] != item.gold):
            raise ValueError("source evidence: ScienceQA native binding mismatch")
        names = _COLUMNS[source]
        if row.get("lecture") or row.get("solution"):
            scope = "SOURCE_EXPLANATION_FOR_REVIEW"
    elif source == "hf-geometry3k":
        if (row["problem"].strip().removeprefix("<image>").strip() != actor.question
                or row["answer"].strip() != item.gold):
            raise ValueError("source evidence: Geometry3K native binding mismatch")
        names = _COLUMNS[source]
    elif source == "hf-chebi-role":
        from fine_tuning.notebook_rl.adapters.hf_chem import parse_roles

        if actor.choices[ord(item.gold) - ord("A")] not in parse_roles(row["description"]):
            raise ValueError("source evidence: ChEBI native role binding mismatch")
        names, scope = ["CID", "description", "SMILES"], "CURATED_ROLE_DESCRIPTION_FOR_REVIEW"
    elif source == "hf-circuitvqa":
        if row.get("qtype") != "count":
            return [], "LABEL_ONLY_UNSUPPORTED_CIRCUIT_TYPE"
        answer, question, bbox, symbol = (row.get(key) for key in ("answer", "question", "bbox", "symbol"))
        if (row.get("splittype") != "train" or not isinstance(question, str)
                or question.strip() != actor.question or not isinstance(answer, str)
                or re.fullmatch(r"\d+(?:\.0+)?", answer) is None or item.answer_type != "int"
                or actor.answer_format != "int" or actor.choices or item.choices
                or str(int(answer.partition(".")[0])) != item.gold
                or not isinstance(bbox, str) or not isinstance(symbol, str)):
            raise ValueError("source evidence: CircuitVQA native binding mismatch")
        boxes = [entry.strip().split() for entry in bbox.split(" . ") if entry.strip()]
        if (not boxes or any(len(box) != 5 for box in boxes)
                or str(sum(box[0].replace(".", "-") == symbol.replace(".", "-") for box in boxes)) != item.gold):
            raise ValueError("source evidence: CircuitVQA count disagrees with source boxes")
        names = ["question", "answer", "qtype", "symbol", "bbox", "bbox_len"]
        scope = "SOURCE_EXACT_CIRCUIT_COUNT_FOR_REVIEW"
    else:
        from fine_tuning.notebook_rl.adapters.hf_medical import _qa_answer

        if _qa_answer(row["question"], row["answer"]) != (actor.question, actor.choices, item.answer_type, item.gold):
            raise ValueError("source evidence: SLAKE native binding mismatch")
        names = ["qid", "question", "answer", "modality", "location", "triple"]
        triple = row.get("triple")
        concrete = (isinstance(triple, list) and len(triple) == 3 and
                    all(isinstance(value, str) and value.strip() and value not in ("_", "vhead", "ktail")
                        for value in triple))
        scope = "SOURCE_TRIPLE_FOR_REVIEW" if concrete else "SOURCE_METADATA_ONLY_PLACEHOLDER_TRIPLE"
    references = [{"field": name, "value": row[name], "sha256": digest(row[name]),
                   "location": f"raw:{relative}#{frozen['native_id']}:{name}", "file_sha256": sha}
                  for name in names if name in row and row[name] not in (None, "", [])]
    return references, scope


def _source_variable(question: str) -> str | None:
    match = re.fullmatch(r"Find\s+(?:the\s+value\s+of\s+)?(?:\$\s*)?([A-Za-z])(?:\s*\$)?[.!?]?", question.strip(), re.I)
    return match[1] if match else None


def _image_unit_references(path: Path | None, expected_sha256: str | None, selected: dict, actors: dict) -> dict:
    """Accept an externally pinned, item-specific observation; never infer units from angle alone."""
    if path is None and expected_sha256 is None:
        return {}
    if path is None or expected_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("image unit evidence: path and expected SHA256 required together")
    _checked_file(_safe_root(path.parent), path.name)
    if file_sha(path) != expected_sha256:
        raise ValueError("image unit evidence: file SHA256 mismatch")
    body = _json(path)
    if (set(body) != {"version", "reviewer_id", "human_verified", "labels_modified", "items"}
            or body["version"] not in ("geometry-image-unit-evidence-v1", "geometry-image-unit-evidence-v2")
            or not isinstance(body["reviewer_id"], str) or not body["reviewer_id"].strip()
            or body["human_verified"] is not False or body["labels_modified"] is not False
            or not isinstance(body["items"], dict) or not body["items"]):
        raise ValueError("image unit evidence: invalid review schema")
    refs = {}
    required = {"source", "source_revision", "native_id", "split", "actor_sha256", "image_sha256",
                "angular_target", "observed_degree_marks"}
    glyph_target = body["version"] == "geometry-image-unit-evidence-v2"
    if glyph_target:
        required = required | {"target_degree_mark"}
    for pid, entry in body["items"].items():
        if pid not in selected or pid not in actors:
            raise ValueError("image unit evidence: unselected problem id")
        frozen, actor = selected[pid], actors[pid]
        if (not isinstance(entry, dict) or set(entry) != required or entry["source"] != "hf-geometry3k"
                or any(entry[name] != frozen[name] for name in ("source", "source_revision", "native_id", "split"))
                or entry["split"] != "train" or entry["actor_sha256"] != text_sha256(to_json(actor))
                or entry["image_sha256"] != list(actor.image_sha256)):
            raise ValueError("image unit evidence: actor/image/source/native binding mismatch")
        target, marks = entry["angular_target"], entry["observed_degree_marks"]
        if (not isinstance(target, str) or not target.strip() or target.strip() != target
                or target.casefold() not in actor.question.casefold()
                or re.search(r"\brad(?:ian)?s?\b", actor.question, re.I)
                or not isinstance(marks, list) or not 1 <= len(marks) <= 16
                or any(not isinstance(mark, str) or re.fullmatch(r"[-+]?\d+(?:\.\d+)?\s*°", mark) is None
                       for mark in marks)):
            raise ValueError("image unit evidence: explicit angular target and observed degree glyphs required")
        if glyph_target:
            if (_source_variable(actor.question) != target or entry["target_degree_mark"] != target + "°"):
                raise ValueError("image unit evidence: exact native variable target and its degree glyph required")
        elif re.search(r"\\angle|∠|\bangle\b", target, re.I) is None:
            raise ValueError("image unit evidence: explicit angular target and observed degree glyphs required")
        value = {**entry, "reviewer_id": body["reviewer_id"], "human_verified": False, "labels_modified": False,
                 "review_scope": "Native image unit marks and selected angular target only; not gold or note correctness"}
        refs[pid] = {"field": "reviewed_image_unit", "value": value, "sha256": digest(value),
                     "location": f"private-image-unit-review:{path.name}#{pid}", "file_sha256": expected_sha256}
    return refs


def _rule(actor, item, lineage, frozen: dict, references: list[dict], binding: dict) -> tuple[dict | None, str]:
    if item.answer_type == "mcq":
        return None, "MCQ_UNCHANGED"
    kind, accept = item.answer_type, dict(item.accept)
    if kind not in ("int", "decimal", "fraction", "unit_number", "short_text"):
        return None, "UNSUPPORTED_TYPE_EXACT_FALLBACK"
    unit = "UNSUPPORTED_NO_EXPLICIT_NATIVE_UNIT" if actor.source == "hf-geometry3k" else "NO_UNIT_EXTENSION"
    problem = next((ref for ref in references if ref["field"] == "problem"), None)
    image_unit = next((ref for ref in references if ref["field"] == "reviewed_image_unit"), None)
    variable = _source_variable(actor.question) if actor.source == "hf-geometry3k" and problem is not None else None
    if variable is not None:
        accept["variable"] = variable
    explicit_text = (problem is not None and re.search(r"\bin degrees\b|\bdegree measure\b", actor.question, re.I)
                     and re.search(r"\b(angle|measure|degrees?)\b|∠", actor.question, re.I))
    unit_ref = None
    if actor.source == "hf-geometry3k" and (explicit_text or image_unit is not None):
        kind = "unit_number"
        accept.update(unit_aliases=["°", "degree", "degrees"], require_unit=False)
        unit = "REVIEWED_NATIVE_IMAGE_DEGREES" if image_unit is not None else "EXPLICIT_NATIVE_DEGREES"
        unit_ref = image_unit if image_unit is not None else problem
    scale = next((ref for ref in references if ref["field"] == "scale"), None)
    if (actor.source == "hf-tatqa-table" and kind == "decimal" and scale is not None
            and scale["value"] in ("percent", "thousand", "million", "billion")):
        kind = "unit_number"
        aliases = ["%", r"\%", "percent"] if scale["value"] == "percent" else [scale["value"]]
        accept.update(unit_aliases=aliases, require_unit=False)
        unit, unit_ref = "NATIVE_AUTHOR_SCALE_NO_RESCALING", scale
    policy = {"gold_sha256": text_sha256(item.gold), "original_accept": item.accept,
              "answer_type": kind, "accept": accept, "unit_evidence": unit_ref,
              "adapter_code_sha256": frozen["adapter_code_sha256"]}
    if variable is not None:
        policy["variable_evidence"] = problem
    rule = {key: binding[key] for key in ("actor_sha256", "image_sha256", "source", "source_revision", "split", "native_id")}
    rule.update(answer_type=kind, accept=accept, provenance={
        "label_provenance": lineage.label_provenance,
        "field": "verifier.gold + adapter.accept + native.unit_evidence" + (" + native.target_variable" if variable else ""),
        "location": f"vault:train/verifier_private.jsonl#{actor.problem_id};adapter:{frozen['adapter_code_sha256']}",
        "sha256": digest(policy), "human_verified": False,
        "review_scope": ("Original label and frozen adapter numeric policy; degree aliases from exact-bound reviewed native image marks; no note review"
                         if image_unit is not None else
                         "Original label and frozen adapter numeric policy; unit aliases only from explicit native degrees; no note review")})
    if variable is not None:
        rule["provenance"]["review_scope"] += "; variable prefix only from the exact native Find-variable target"
    if unit == "NATIVE_AUTHOR_SCALE_NO_RESCALING":
        rule["provenance"]["review_scope"] = "Original native label and frozen numeric policy; suffix only from pinned author scale; no numeric or currency conversion; no note review"
    if compare_training(item.gold, item, rule) is not True:
        return None, "UNSUPPORTED_GOLD_EXACT_FALLBACK"
    return rule, unit


def prepare(*, build_root: Path, manifest: Path, order: Path, limit: int, raw_root: Path, output: Path,
            supplementary_raw_root: Path | None = None, image_unit_evidence: Path | None = None,
            image_unit_evidence_sha256: str | None = None, additional_image_unit_evidence: Path | None = None,
            additional_image_unit_evidence_sha256: str | None = None) -> dict:
    """Write a new private directory; input and output hashes are portable, with no timestamps."""
    if type(limit) is not int or limit < 1:
        raise ValueError("source evidence: positive limit required")
    if output.exists():
        raise ValueError("source evidence: refusing to overwrite output")
    build_root, raw_root = _safe_root(build_root), _safe_root(raw_root)
    frozen, manifest_sha = _json(manifest), file_sha(manifest)
    vault_manifest = _json(build_root / "manifest.yaml")
    vault_sha = file_sha(build_root / "manifest.yaml")
    freeze_binding = _json(build_root / "freeze_binding.json")
    if (frozen.get("version") != "pool-freeze-manifest-v1" or
            freeze_binding.get("version") != "pool-freeze-vault-binding-v1" or
            freeze_binding.get("dataset_manifest_sha256") != manifest_sha or
            freeze_binding.get("vault_manifest_sha256") != vault_sha):
        raise ValueError("source evidence: frozen manifest/vault binding mismatch")
    train = build_root / "train"
    train_sha = file_sha(train / "manifest.yaml")
    if vault_manifest["split_manifests"]["train"] != train_sha:
        raise ValueError("source evidence: train split binding mismatch")
    actors = {actor.problem_id: actor for actor in read_actors(train)}
    items, lineages = read_verifier(train), read_lineage(train)
    fixed_order = read_order(order)
    if limit > len(fixed_order["items"]):
        raise ValueError("source evidence: limit exceeds fixed order")
    manifest_items = {row["problem_id"]: row for row in frozen["items"]}
    if len(manifest_items) != len(frozen["items"]):
        raise ValueError("source evidence: duplicate frozen id")
    selected = {}
    for entry in fixed_order["items"][:limit]:
        pid = entry["problem_id"]
        if pid not in manifest_items or pid not in actors or pid not in items or pid not in lineages:
            raise ValueError("source evidence: selected id absent from frozen train vault")
        row, actor, lineage = manifest_items[pid], actors[pid], lineages[pid]
        if (row["status"] != "INCLUDED" or row["split"] != "train" or lineage.split != "train" or
                row["source"] != actor.source or entry["source"] != actor.source or
                row["source_revision"] != lineage.source_revision or row["subject"] != entry["subject"] or
                row["image_sha256"] != list(actor.image_sha256) or row["image_paths"] != list(actor.image_paths) or
                not isinstance(row["native_id"], str) or not row["native_id"].strip()):
            raise ValueError("source evidence: selected train identity mismatch")
        for name, record in (("actor", actor), ("verifier", items[pid]), ("lineage", lineage)):
            if row["record_sha256"][name] != text_sha256(to_json(record)):
                raise ValueError("source evidence: frozen record hash mismatch")
        for relative, sha in zip(actor.image_paths, actor.image_sha256, strict=True):
            if file_sha(_checked_file(build_root / "images", relative)) != sha:
                raise ValueError("source evidence: image hash mismatch")
        selected[pid] = row
    native, cache = _native_rows(raw_root, selected, supplementary_raw_root=supplementary_raw_root)
    image_units = _image_unit_references(image_unit_evidence, image_unit_evidence_sha256, selected, actors)
    if image_units:
        cache["image_unit_evidence_sha256"] = image_unit_evidence_sha256
    additional_units = _image_unit_references(additional_image_unit_evidence,
                                              additional_image_unit_evidence_sha256, selected, actors)
    if image_units.keys() & additional_units.keys():
        raise ValueError("source evidence: image unit files overlap on problem IDs")
    if additional_units:
        cache["additional_image_unit_evidence_sha256"] = additional_image_unit_evidence_sha256
        image_units.update(additional_units)
    rules, sources = {}, {}
    statuses: Counter[str] = Counter()
    availability: Counter[str] = Counter()
    for pid, row in selected.items():
        actor, item, lineage = actors[pid], items[pid], lineages[pid]
        references, scope = _annotations(actor, item, row, native)
        if pid in image_units:
            references.append(image_units[pid])
        binding = {"actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                   "source": actor.source, "source_revision": lineage.source_revision, "split": "train", "native_id": row["native_id"]}
        rule, status = _rule(actor, item, lineage, row, references, binding)
        if rule is not None:
            rules[pid] = rule
        statuses[status] += 1
        availability[scope] += 1
        sources[pid] = binding | {"label_provenance": lineage.label_provenance, "human_verified": False,
            "vault_manifest_sha256": train_sha, "dataset_manifest_sha256": manifest_sha,
            "record_sha256": row["record_sha256"], "adapter_code_sha256": row["adapter_code_sha256"],
            "label": {"value": item.gold, "sha256": text_sha256(item.gold),
                      "location": f"vault:train/verifier_private.jsonl#{pid}:gold"},
            "adapter_policy": {"accept": item.accept, "sha256": digest(item.accept),
                               "provenance": "Frozen adapter policy, not source-authored tolerance"},
            "annotations": references, "content_verification_available": scope in (
                "SOURCE_EXPLANATION_FOR_REVIEW", "CURATED_ROLE_DESCRIPTION_FOR_REVIEW", "SOURCE_TRIPLE_FOR_REVIEW",
                "SOURCE_EXACT_CIRCUIT_COUNT_FOR_REVIEW", "SOURCE_EXECUTED_SPATIAL_PROGRAM_FOR_REVIEW",
                "SOURCE_NATIVE_PDF_FINANCIAL_ANNOTATIONS_FOR_REVIEW", "SOURCE_NATIVE_TABLE_ANNOTATIONS_FOR_REVIEW"),
            "content_verification_scope": scope, "note_content_status": "UNKNOWN_REVIEW_REQUIRED",
            "reference_use": "PRIVATE_REVIEW_INPUT_ONLY_NOT_TRAINING_TARGET", "unit_policy": status,
            "leak_aliases": [], "acronym_candidates": []}
        if actor.source == "hf-circuitvqa":
            sources[pid]["source_selection_eligible"] = scope == "SOURCE_EXACT_CIRCUIT_COUNT_FOR_REVIEW"
    documents = {"rules.json": {"version": RULES_VERSION, "vault_manifest_sha256": train_sha, "items": rules},
                 "source-provenance.json": {"version": VERSION, "items": sources}}
    payloads = {name: json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=1) + "\n"
                for name, value in documents.items()}
    summary = {"version": "source-evidence-preparation-v1", "count": len(selected), "rules": len(rules),
               "selected_ids_sha256": digest(list(selected)), "order_sha256": file_sha(order),
               "dataset_manifest_sha256": manifest_sha, "vault_manifest_sha256": vault_sha,
               "train_manifest_sha256": train_sha, "cache": cache,
               "answer_policy": dict(statuses), "source_availability": dict(availability),
               "output_sha256": {name: text_sha256(text) for name, text in payloads.items()},
               "human_verified": False, "training_permission_granted": False}
    _safe_root(output.parent)
    output.mkdir(mode=0o700)
    for name, text in (payloads | {"summary.json": json.dumps(summary, sort_keys=True, indent=1) + "\n"}).items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(text)
        (output / name).chmod(0o600)
    return summary


def prepare_dev(*, build_root: Path, manifest: Path, output: Path) -> dict:
    """Pin all frozen dev non-MCQ policies for evaluation only; never extend training rules."""
    if output.exists():
        raise ValueError("dev evidence: refusing to overwrite output")
    build_root = _safe_root(build_root)
    frozen, manifest_sha = _json(manifest), file_sha(manifest)
    vault_sha = file_sha(build_root / "manifest.yaml")
    binding = _json(build_root / "freeze_binding.json")
    if (frozen.get("version") != "pool-freeze-manifest-v1" or
            binding.get("version") != "pool-freeze-vault-binding-v1" or
            binding.get("dataset_manifest_sha256") != manifest_sha or
            binding.get("vault_manifest_sha256") != vault_sha):
        raise ValueError("dev evidence: frozen manifest/vault binding mismatch")
    dev_root = build_root / "dev"
    dev_sha = file_sha(dev_root / "manifest.yaml")
    if _json(build_root / "manifest.yaml")["split_manifests"]["dev"] != dev_sha:
        raise ValueError("dev evidence: dev split binding mismatch")
    actors = {actor.problem_id: actor for actor in read_actors(dev_root)}
    items, lineages = read_verifier(dev_root), read_lineage(dev_root)
    rows = {row["problem_id"]: row for row in frozen["items"]}
    if len(rows) != len(frozen["items"]):
        raise ValueError("dev evidence: duplicate frozen id")
    selected = {pid: row for pid, row in rows.items() if row["status"] == "HELD_OUT_DEV"}
    if not selected or set(selected) != set(actors) or set(selected) != set(items) or set(selected) != set(lineages):
        raise ValueError("dev evidence: dev vault must exactly cover all held-out dev ids")
    rules = {}
    for pid in sorted(selected):
        row, actor, item, lineage = selected[pid], actors[pid], items[pid], lineages[pid]
        if (row["split"] != "dev" or lineage.split != "dev" or row["source"] != actor.source or
                lineage.source != actor.source or row["source_revision"] != lineage.source_revision or
                row["image_sha256"] != list(actor.image_sha256) or row["image_paths"] != list(actor.image_paths) or
                not isinstance(row["native_id"], str) or not row["native_id"].strip()):
            raise ValueError("dev evidence: frozen dev identity mismatch")
        for name, record in (("actor", actor), ("verifier", item), ("lineage", lineage)):
            if row["record_sha256"][name] != text_sha256(to_json(record)):
                raise ValueError("dev evidence: frozen record hash mismatch")
        for relative, sha in zip(actor.image_paths, actor.image_sha256, strict=True):
            if file_sha(_checked_file(build_root / "images", relative)) != sha:
                raise ValueError("dev evidence: image hash mismatch")
        if item.answer_type == "mcq":
            continue
        kind: str = item.answer_type
        accept = dict(item.accept)
        range_rule = {"answer_type": "range", "accept": {}}
        if kind == "short_text" and not accept and compare_training(item.gold, item, range_rule) is True:
            kind = "range"
        rule = {"actor_sha256": text_sha256(to_json(actor)), "image_sha256": list(actor.image_sha256),
                "source": actor.source, "source_revision": lineage.source_revision, "split": "dev",
                "native_id": row["native_id"], "answer_type": kind, "accept": accept,
                "provenance": {"label_provenance": lineage.label_provenance,
                    "field": "verifier.gold + frozen adapter.accept",
                    "location": f"vault:dev/verifier_private.jsonl#{pid};adapter:{row['adapter_code_sha256']}",
                    "sha256": digest({"gold": item.gold, "answer_type": item.answer_type, "accept": item.accept,
                                      "adapter_code_sha256": row["adapter_code_sha256"]}),
                    "human_verified": False,
                    "review_scope": "Evaluation only: frozen native label syntax; numeric tolerance is adapter policy, not source annotation"}}
        if compare_training(item.gold, item, rule) is not True:
            raise ValueError("dev evidence: unsupported frozen answer policy")
        rules[pid] = rule
    document = {"version": "external-dev-answer-rules-v1", "vault_manifest_sha256": dev_sha, "items": rules}
    text = json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=1) + "\n"
    summary = {"version": "external-dev-evidence-preparation-v1", "count": len(selected), "rules": len(rules),
               "selected_ids_sha256": digest(sorted(selected)), "dataset_manifest_sha256": manifest_sha,
               "vault_manifest_sha256": vault_sha, "dev_manifest_sha256": dev_sha,
               "answer_types": dict(Counter(rule["answer_type"] for rule in rules.values())),
               "output_sha256": {"rules.json": text_sha256(text)}, "human_verified": False,
               "training_allowed": False}
    _safe_root(output.parent)
    output.mkdir(mode=0o700)
    for name, value in {"rules.json": text, "summary.json": json.dumps(summary, sort_keys=True, indent=1) + "\n"}.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            stream.write(value)
        (output / name).chmod(0o600)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("build-root", "manifest", "order", "raw-root", "output"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_parser.add_argument("--limit", type=int, required=True)
    prepare_parser.add_argument("--supplementary-raw-root", type=Path,
                                help="explicit second native-source root; no automatic fallback")
    prepare_parser.add_argument("--image-unit-evidence", type=Path)
    prepare_parser.add_argument("--image-unit-evidence-sha256")
    prepare_parser.add_argument("--additional-image-unit-evidence", type=Path)
    prepare_parser.add_argument("--additional-image-unit-evidence-sha256")
    dev_parser = commands.add_parser("prepare-dev")
    for name in ("build-root", "manifest", "output"):
        dev_parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-dev":
        summary = prepare_dev(build_root=args.build_root, manifest=args.manifest, output=args.output)
    else:
        summary = prepare(build_root=args.build_root, manifest=args.manifest, order=args.order,
                          limit=args.limit, raw_root=args.raw_root, output=args.output,
                          supplementary_raw_root=args.supplementary_raw_root,
                          image_unit_evidence=args.image_unit_evidence,
                          image_unit_evidence_sha256=args.image_unit_evidence_sha256,
                          additional_image_unit_evidence=args.additional_image_unit_evidence,
                          additional_image_unit_evidence_sha256=args.additional_image_unit_evidence_sha256)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
