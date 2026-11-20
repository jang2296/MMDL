"""Dataset-provided reference text for the note step, plus a sealed supplement for PTB-XL and SSMR.

Reference text is shown only in the note prompt after the gold answer. It is never part of the solve or
evaluation input, and nothing here reads MMMU or the network. Since SINGLE v12 the human-written solution
fields (HUMAN_SOLUTION_FIELDS) of a source also become H SFT targets; no other reference text is a target.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

from .audit import digest, text_sha256
from .collect_v3 import file_sha

REFERENCE_VERSION = "note-dataset-reference-v1"
SUPPLEMENT_VERSION = "note-reference-supplement-v1"
REFERENCE_MAX_CHARS = 4000
REFERENCE_FIELDS: dict[str, tuple[str, ...]] = {
    "hf-scienceqa": ("lecture", "solution"),
    "hf-clevr": ("scene", "program"),
    "hf-slake": ("modality", "location", "triple"),
    "hf-chebi-role": ("description", "SMILES"),
    "hf-ptbxl": ("report", "statements"),
    "hf-ssmr": ("abc_context",),
}
_LABELS = {"lecture": "Lecture", "solution": "Solution", "scene": "Scene objects",
           "program": "Question program", "abc_context": "Score (ABC notation)",
           "report": "Cardiologist report", "statements": "Diagnostic statements"}
_MULTILINE = {"scene", "program"}
# Same non-concrete SLAKE triple vocabulary as source_evidence.py.
_PLACEHOLDERS = {"_", "vhead", "ktail"}
# source -> (raw repo dir, shard glob, native-id prefix, columns)
_RAW = {"hf-ptbxl": ("tirandazdylan__ptb-xl-ecg", "train-*.parquet", "ecg", ("report", "statements")),
        "hf-ssmr": ("Sylence__SSMR-Bench", "Train_visual-*.parquet", "Train_visual", ("abc_context",))}


def fields_sha256() -> str:
    return digest({"fields": REFERENCE_FIELDS, "max_chars": REFERENCE_MAX_CHARS, "version": REFERENCE_VERSION})


def note_reference_pin(sources: dict[str, dict], supplement: dict[str, dict], supplement_sha256: str,
                       problem_ids: list[str]) -> dict:
    """The protocol's note_reference pin for an order (the campaign's and the local CLI's)."""
    return {"version": REFERENCE_VERSION, "fields_sha256": fields_sha256(), "supplement_sha256": supplement_sha256,
            "max_chars": REFERENCE_MAX_CHARS,
            "items_with_reference": sum(note_reference(sources[pid], supplement.get(pid)) is not None for pid in problem_ids)}


# v12 H rows: human-written solution fields only, as (lecture, solution). Program-generated annotations (CLEVR scene and
# program, SLAKE triples, ...) are never H targets.
HUMAN_SOLUTION_VERSION = "dataset-human-solution-v1"
HUMAN_SOLUTION_FIELDS: dict[str, tuple[str, str]] = {"hf-scienceqa": ("lecture", "solution")}


def human_solution_sha256() -> str:
    return digest({"fields": HUMAN_SOLUTION_FIELDS, "version": HUMAN_SOLUTION_VERSION})


def human_solution(provenance_item: dict) -> dict[str, str] | None:
    """{"lecture", "solution"} text of a mapped source from the item's own provenance annotations (blank when the
    field is absent); None for every other source."""
    names = HUMAN_SOLUTION_FIELDS.get(provenance_item.get("source"))
    if names is None:
        return None
    values: dict[str, object] = {}
    for annotation in provenance_item.get("annotations") or []:
        values.setdefault(annotation.get("field"), annotation.get("value"))
    return {"lecture": _text(values.get(names[0])), "solution": _text(values.get(names[1]))}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value).strip()
    return str(value).strip()


def _parsed(value: object) -> object:
    """Native JSON values pass through; python-repr strings are parsed safely, else kept as text."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return value
    return value


def _number(value: object) -> str:
    return f"{float(value):g}"


def _scene(value: object) -> str:
    scene = _parsed(value)
    objects = scene.get("objects") if isinstance(scene, dict) else None
    if not isinstance(objects, list) or not objects:
        return ""
    lines = []
    try:
        for index, obj in enumerate(objects):
            x, y = obj["pixel_coords"][:2]
            lines.append(f"object {index}: {obj['size']} {obj['color']} {obj['material']} "
                         f"{obj['shape']} at ({_number(x)}, {_number(y)})")
    except (KeyError, TypeError, IndexError, ValueError, AttributeError):
        return ""  # a partial scene would mislead; skip the whole field
    return "\n".join(lines)


def _program(value: object) -> str:
    steps = _parsed(value)
    if not isinstance(steps, list):
        return _text(steps)
    lines = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or "function" not in step:
            return _text(value)
        values = f"({', '.join(str(v) for v in step.get('value_inputs') or [])})" if step.get("value_inputs") else ""
        inputs = f" <- {', '.join(str(i) for i in step['inputs'])}" if step.get("inputs") else ""
        lines.append(f"{index}: {step['function']}{values}{inputs}")
    return "\n".join(lines)


def _triple(value: object) -> str:
    triple = _parsed(value)
    if isinstance(triple, (list, tuple)):
        if not any(str(item).strip() and str(item).strip() not in _PLACEHOLDERS for item in triple):
            return ""
    return _text(triple)


_RENDER = {"scene": _scene, "program": _program, "triple": _triple}


def note_reference(provenance_item: dict, supplement_item: dict | None = None) -> str | None:
    """Reference block for one training item, or None when the source offers nothing usable."""
    source = provenance_item.get("source")
    wanted = REFERENCE_FIELDS.get(source)
    if not wanted:
        return None
    if supplement_item and supplement_item.get("source") not in (None, source):
        raise ValueError("references: supplement item source differs from provenance source")
    values: dict[str, object] = {}
    for item in (provenance_item, supplement_item or {}):
        for annotation in item.get("annotations") or []:
            values.setdefault(annotation.get("field"), annotation.get("value"))
    blocks = []
    for name in wanted:
        text = _RENDER.get(name, _text)(values.get(name))
        if text:
            label = _LABELS.get(name, name[:1].upper() + name[1:])
            blocks.append(f"{label}:\n{text}" if name in _MULTILINE else f"{label}: {text}")
    while blocks and len("\n\n".join(blocks)) > REFERENCE_MAX_CHARS:
        blocks.pop()
    return "\n\n".join(blocks) or None


def _wanted_rows(manifest_path: Path, order_path: Path) -> dict[str, list[tuple[str, str]]]:
    ordered = {item["problem_id"] for item in json.loads(Path(order_path).read_text())["items"]}
    by_id = {item["problem_id"]: item for item in json.loads(Path(manifest_path).read_text())["items"]}
    if ordered - by_id.keys():
        raise ValueError("references: ordered problem ids absent from the manifest")
    rows: dict[str, list[tuple[str, str]]] = {source: [] for source in _RAW}
    for problem_id in sorted(ordered):
        item = by_id[problem_id]
        if item["source"] in _RAW:
            rows[item["source"]].append((problem_id, item["native_id"]))
    return rows


def _entry(source: str, native_id: str, values: dict, repo: str, relative: str, sha: str) -> dict:
    annotations = []
    for field, value in values.items():
        text = _text(value)
        if text:
            annotations.append({"field": field, "value": text, "sha256": text_sha256(text),
                                "location": f"raw:{repo}/{relative}#{native_id}:{field}", "file_sha256": sha})
    return {"source": source, "native_id": native_id, "annotations": annotations}


def build_supplement(*, manifest_path: Path, order_path: Path, raw_root: Path) -> dict:
    """Annotations for the sources the bundled provenance lacks; rows without text go to 'missing'."""
    import pyarrow.parquet as pq

    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise ValueError("references: raw root is not a directory")
    rows = _wanted_rows(manifest_path, order_path)
    items: dict[str, dict] = {}
    missing: set[str] = set()
    for source, (repo, pattern, prefix, columns) in _RAW.items():
        wanted: dict[int, list[str]] = {}
        for problem_id, native_id in rows[source]:
            head, _, index = native_id.partition(":")
            if head != prefix or not index.isdigit():
                raise ValueError(f"references: unexpected {source} native id {native_id!r}")
            wanted.setdefault(int(index), []).append(problem_id)
        shards = sorted((raw_root / repo / "data").glob(pattern))
        found: dict[int, tuple[dict, Path]] = {}
        if source == "hf-ptbxl":
            for shard in shards:
                table = pq.read_table(shard, columns=["ecg_id", *columns])
                for position, ecg_id in enumerate(table.column("ecg_id").to_pylist()):
                    if ecg_id in wanted and ecg_id not in found:
                        found[ecg_id] = ({name: table.column(name)[position].as_py() for name in columns}, shard)
        elif len(shards) > 1:
            raise ValueError("references: expected a single Train_visual shard")
        elif shards:
            table = pq.read_table(shards[0], columns=list(columns))
            for index in wanted:
                if index < table.num_rows:
                    found[index] = ({name: table.column(name)[index].as_py() for name in columns}, shards[0])
        shas: dict[Path, str] = {}
        for index, problem_ids in wanted.items():
            if index not in found:
                missing.update(problem_ids)
                continue
            values, shard = found[index]
            if shard not in shas:
                shas[shard] = file_sha(shard)
            item = _entry(source, f"{prefix}:{index}", values, repo, shard.relative_to(raw_root / repo).as_posix(),
                          shas[shard])
            for problem_id in problem_ids:
                if item["annotations"]:
                    items[problem_id] = item
                else:
                    missing.add(problem_id)
    return {"version": SUPPLEMENT_VERSION, "items": items, "missing": sorted(missing),
            "fields_sha256": fields_sha256()}


def write_supplement(value: dict, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)
    return file_sha(path)


def read_supplement(path: Path, expected_sha256: str) -> dict[str, dict]:
    if file_sha(Path(path)) != expected_sha256:
        raise ValueError("references: supplement sha256 mismatch")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("version") != SUPPLEMENT_VERSION or not isinstance(value.get("items"), dict):
        raise ValueError("references: supplement version/shape mismatch")
    return value["items"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the PTB-XL/SSMR reference supplement from raw files")
    for name in ("manifest", "order", "raw-root", "out"):
        build.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_supplement(manifest_path=args.manifest, order_path=args.order, raw_root=args.raw_root)
    sha = write_supplement(value, args.out)
    print(json.dumps({"status": "OK", "items": len(value["items"]), "missing": len(value["missing"]),
                      "sha256": sha, "by_source": dict(Counter(i["source"] for i in value["items"].values()))},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
