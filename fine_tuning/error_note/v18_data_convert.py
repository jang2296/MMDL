"""Read selected native rows once and reuse the frozen source adapters."""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from contextlib import ExitStack
import csv
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

from .v18_data import VERSION, ROLES, _safe_path, _write, cache_path, canonical_digest, file_sha


class SourceReader:
    def __init__(self, recipe: dict, cache_dir: Path):
        self.recipe, self.cache_dir = recipe, cache_dir
        self.stack = ExitStack()
        self.json_cache: dict = {}
        self.zip_cache: dict = {}
        self.rows: dict = {}
        self.read_timings: list[dict] = []

    def close(self):
        self.stack.close()
        self.rows.clear()
        self.json_cache.clear()
        self.zip_cache.clear()

    def path(self, key: str) -> Path:
        return cache_path(self.cache_dir, self.recipe["files"][key])

    def source_file(self, source: str, suffix: str) -> str:
        repo = self.recipe["sources"][source]["repo"]
        keys = [k for k, v in self.recipe["files"].items()
                if v.get("repo") == repo and v.get("path", "").endswith(suffix)]
        if len(keys) != 1:
            raise ValueError(f"V18_AMBIGUOUS_SOURCE_FILE:{source}:{suffix}")
        return keys[0]

    def json(self, key: str):
        if key not in self.json_cache:
            self.json_cache[key] = json.loads(self.path(key).read_text())
        return self.json_cache[key]

    def member(self, key: str, member: str) -> bytes:
        if Path(member).is_absolute() or ".." in Path(member).parts:
            raise ValueError("V18_UNSAFE_ARCHIVE_MEMBER")
        if key not in self.zip_cache:
            self.zip_cache[key] = self.stack.enter_context(zipfile.ZipFile(self.path(key)))
        archive = self.zip_cache[key]
        if len([x for x in archive.infolist() if x.filename == member]) != 1:
            raise ValueError("V18_AMBIGUOUS_ARCHIVE_MEMBER")
        return archive.read(member)

    def selected_rows(self, items: list[dict]) -> None:
        import pyarrow.parquet as pq

        groups: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            key = item["locator"]["file_key"]
            if self.recipe["files"][key].get("path", "").endswith(".parquet"):
                groups[key].append(item)
        for key, selected in groups.items():
            started = time.monotonic()
            parquet = pq.ParquetFile(self.path(key))
            ends, total = [], 0
            for index in range(parquet.num_row_groups):
                total += parquet.metadata.row_group(index).num_rows
                ends.append(total)
            by_group: dict[int, list[int]] = defaultdict(list)
            for item in selected:
                row = item["locator"]["row_index"]
                if type(row) is not int or not 0 <= row < total:
                    raise ValueError("V18_ROW_OUT_OF_RANGE")
                group = bisect_right(ends, row)
                offset = row - (0 if group == 0 else ends[group - 1])
                if (item["locator"].get("row_group", group) != group
                        or item["locator"].get("row_offset", offset) != offset):
                    raise ValueError("V18_ROW_GROUP_COORDINATES_CHANGED")
                by_group[group].append(row)
            columns = None
            if selected[0]["source"] == "hf-ptbxl":
                from fine_tuning.notebook_rl.adapters.hf_pathology import PTBXL_COLUMNS

                columns = [*PTBXL_COLUMNS, "signal"]
            for group, wanted in by_group.items():
                table = parquet.read_row_group(group, columns=columns)
                start = 0 if group == 0 else ends[group - 1]
                for row in set(wanted):
                    self.rows[key, row] = table.slice(row - start, 1).to_pylist()[0]
            self.read_timings.append({"file_key": key, "sources": sorted({x["source"] for x in selected}),
                "selected_rows": len({x["locator"]["row_index"] for x in selected}),
                "row_groups": len(by_group), "elapsed_seconds": time.monotonic() - started})

    def row(self, item: dict) -> dict:
        key = item["locator"]["file_key"]
        if "question_index" in item["locator"]:
            index_key = (key, "question_index")
            if index_key not in self.json_cache:
                self.json_cache[index_key] = {x["question_index"]: x for x in self.json(key)["questions"]}
            return self.json_cache[index_key][item["locator"]["question_index"]]
        row = item["locator"]["row_index"]
        if (key, row) in self.rows:
            return self.rows[key, row]
        data = self.json(key)
        if isinstance(data, dict):
            data = data[item["locator"].get("array_key", "questions")]
        return data[row]


def candidate(item: dict, reader: SourceReader):
    from fine_tuning.notebook_rl.adapters import (hf_agri as agri, hf_cauldron as cauldron,
        hf_chem as chem, hf_clevr as clevr, hf_docs as docs, hf_graph_vlqa as graph,
        hf_medical as medical, hf_music as music, hf_pathology as pathology, hf_science as science)
    from fine_tuning.notebook_rl.adapters.hf_common import Candidate, program_mcq

    source, native, loc = item["source"], item["native_id"], item["locator"]
    if source == "hf-chebi-role":
        key = loc["file_key"]
        if key not in reader.json_cache:
            with reader.path(key).open(newline="", encoding="utf-8") as stream:
                reader.json_cache[key] = {f"cid:{int(x['CID'])}": x for x in csv.DictReader(stream)}
        row = reader.json_cache[key][native]
        cid = int(row["CID"])
        choices, gold = chem.role_options(cid, row["description"])
        image_key = loc.get("image_file_key") or reader.source_file(source, chem.IMAGE_ZIP)
        return Candidate(source, native, "Pharmacy", chem.ROLE_QUESTION, choices, "mcq", gold,
            images=(reader.member(image_key, f"image/CID_{cid}.png"),), family_key=native), row
    if source == "hf-plantvillage":
        split_key = loc.get("split_file_key") or reader.source_file(source, agri.PLANT_SPLIT_FILE)
        leaves_key = loc.get("leaf_map_file_key") or reader.source_file(source, agri.PLANT_LEAF_MAP)
        pool_key = (source, "pools")
        if pool_key not in reader.json_cache:
            pools: dict = defaultdict(set)
            for line in reader.path(split_key).read_text().splitlines():
                parsed = agri._plant_condition(line)
                if parsed:
                    crop, condition = parsed
                    pools[crop].add(agri._plant_condition_text(condition))
            reader.json_cache[pool_key] = {crop: sorted(values | {agri.HEALTHY_TEXT}) for crop, values in pools.items()}
        crop, condition = agri._plant_condition(native)
        choices, gold = program_mcq(agri._plant_condition_text(condition), reader.json_cache[pool_key][crop],
                                   key=native, n_options=4)
        crop_text = agri._plant_crop_text(crop)
        question = f"This photo shows {agri._crop_article(crop_text)} {crop_text} leaf. Which condition does the leaf show?"
        return Candidate(source, native, agri.SUBJECT, question, choices, "mcq", gold,
            images=(reader.member(loc["file_key"], native),),
            family_key=agri._plant_family_key(native, reader.json(leaves_key))), {}
    row = reader.row(item)
    if source in {"hf-ai2d", "hf-mapqa"}:
        images = tuple(x["bytes"] for x in row["images"])
        image_key = hashlib.sha256(images[0]).hexdigest()
        turn = row["texts"][loc["turn_index"]]
        parse = cauldron._ai2d if source == "hf-ai2d" else cauldron._mapqa
        made = parse(turn["user"] or "", turn["assistant"] or "", image_key)
        if made is None:
            raise ValueError("V18_NATIVE_TURN_REJECTED")
        subject, question, choices, kind, gold, accept = made
        result = Candidate(source, native, subject, question, choices, kind, gold, accept, images, image_key)
    elif source == "hf-chartqa":
        result = docs._chartqa_candidate(row, native)
    elif source == "hf-ssmr":
        result = music._candidate(row, int(native.rsplit(":", 1)[1]))
    elif source == "hf-scienceqa":
        result = science._scienceqa_candidate(row, native)
    elif source == "hf-geometry3k":
        result = science._geometry_candidate(row, native)
    elif source == "hf-tqa":
        result = science._tqa_candidate(row, native)
    elif source == "hf-deepweeds":
        pool_key = (source, "labels")
        if pool_key not in reader.json_cache:
            import pyarrow.parquet as pq

            labels = pq.read_table(reader.path(loc["file_key"]), columns=["species"]).column(0).to_pylist()
            reader.json_cache[pool_key] = sorted({agri._weed_text(x) for x in labels if isinstance(x, str) and x.strip()})
        choices, gold = program_mcq(agri._weed_text(row["species"]), reader.json_cache[pool_key],
            key=f"deepweeds:{int(native.rsplit(':', 1)[1])}", n_options=4)
        raw = row["image"]["bytes"]
        result = Candidate(source, native, agri.SUBJECT, agri.DEEPWEEDS_QUESTION, choices, "mcq", gold,
            images=(raw,), family_key=hashlib.sha256(raw).hexdigest())
    elif source == "hf-nctcrc":
        name = pathology.NCT_NAMES[pathology.NCT_CODES[row["label"]]]
        choices, gold = program_mcq(name, sorted(pathology.NCT_NAMES.values()), key=f"nct:{native}", n_options=4)
        raw = row["image"]["bytes"]
        result = Candidate(source, native, pathology.NCT_SUBJECT, pathology.NCT_QUESTION, choices, "mcq", gold,
            images=(raw,), family_key=hashlib.sha256(raw).hexdigest())
    elif source == "hf-ptbxl":
        if native != f"ecg:{row['ecg_id']}":
            raise ValueError("V18_ECG_ID_MISMATCH")
        label = pathology._ptbxl_label(row)
        signal = pathology._ptbxl_signal(row["signal"], row["fs"])
        if label is None or signal is None:
            raise ValueError("V18_ECG_REJECTED")
        choices, gold = program_mcq(pathology.PTBXL_NAMES[label], sorted(pathology.PTBXL_NAMES.values()),
            key=f"ptbxl:{row['ecg_id']}", n_options=len(pathology.PTBXL_NAMES))
        result = Candidate(source, native, pathology.LAB_SUBJECT, pathology.PTBXL_QUESTION, choices, "mcq", gold,
            images=(pathology.render_ecg(signal, row["leads"], row["fs"]),), family_key=f"patient:{row['patient_id']}")
    elif source == "hf-slake":
        if row["q_lang"] != "en" or native != f"train:{row['qid']}":
            raise ValueError("V18_SLAKE_ID_MISMATCH")
        key = (source, "prefixes")
        if key not in reader.json_cache:
            reader.json_cache[key] = medical.slake_prefix_answers(x["answer"] for x in reader.json(loc["file_key"]) if x["q_lang"] == "en")
        if medical._answer_key(row["answer"]) in reader.json_cache[key]:
            raise ValueError("V18_SLAKE_PREFIX_REJECTED")
        question, choices, kind, gold = medical._qa_answer(row["question"], row["answer"])
        image_key = loc.get("image_file_key") or reader.source_file(source, "imgs.zip")
        result = Candidate(source, native, "Clinical_Medicine" if row["content_type"] == "Abnormality" else "Basic_Medical_Science",
            question, choices, kind, gold, images=(reader.member(image_key, loc.get("image_member", f"imgs/{row['img_name']}")),),
            family_key=row["img_name"].split("/")[0])
    elif source == "hf-gvlqa":
        if row["id"] != native:
            raise ValueError("V18_GRAPH_ID_MISMATCH")
        image = graph._IMAGE_PATH.fullmatch(row["image"])
        if image is None:
            raise ValueError("V18_GRAPH_IMAGE_PATH")
        task = image.group(1)
        choices, kind, gold = graph._answer(task, row["conversations"][1]["value"])
        image_key = loc.get("image_file_key") or reader.source_file(source, row["image"])
        result = Candidate(source, native, "Computer_Science", row["conversations"][0]["value"].removeprefix("<image>\n").strip(),
            choices, kind, gold, images=(reader.path(image_key).read_bytes(),),
            family_key=f"{task}-{image.group(2)}-{image.group(3)}")
    elif source == "hf-clevr":
        scene_key = loc.get("scenes_file_key") or reader.source_file(source, "scenes/CLEVR_train_scenes.json")
        scenes_key = (source, "scenes")
        if scenes_key not in reader.json_cache:
            reader.json_cache[scenes_key] = {x["image_index"]: x for x in reader.json(scene_key)["scenes"]}
        scene = reader.json_cache[scenes_key][row["image_index"]]
        answer = clevr.program_answer(scene, row["program"])
        if native != f"train:{row['question_index']}" or answer != row["answer"] or scene["image_filename"] != row["image_filename"]:
            raise ValueError("V18_CLEVR_BINDING_MISMATCH")
        choices = ("Yes", "No") if answer in {"yes", "no"} else ()
        kind = "mcq" if choices else "int" if re.fullmatch(r"\d+", answer) else "short_text"
        gold = ("A" if answer == "yes" else "B") if choices else answer
        image_key = loc.get("image_file_key") or reader.source_file(source, "images/train/" + row["image_filename"])
        result = Candidate(source, native, "Math", row["question"].strip(), choices, kind, gold,
            images=(reader.path(image_key).read_bytes(),), family_key=row["image_filename"])
        row = dict(row, _scene=scene)
    else:
        raise ValueError(f"V18_UNSUPPORTED_SOURCE:{source}")
    if result is None:
        raise ValueError(f"V18_SELECTED_ROW_REJECTED:{item['problem_id']}")
    return result, row


def _aux_value(ref: dict, reader: SourceReader):
    key, pointer = ref["file_key"], ref.get("pointer")
    if pointer and ".zip" in reader.recipe["files"][key].get("path", ""):
        member, _, entry = pointer.partition("#")
        data = json.loads(reader.member(key, member))
        return data[entry] if entry else data
    return reader.json(key)


def source_annotations(item: dict, row: dict, reader: SourceReader, sidecar: dict, image_path: Path,
                *, offline: bool = False) -> tuple[dict, dict | None, dict | None]:
    source = item["source"]
    metadata, raw, human = {}, None, None
    refs = {x["role"]: x for x in item.get("auxiliary_refs", [])}
    if source == "hf-scienceqa":
        metadata = {name: row[name] for name in ("category", "grade", "skill", "subject", "task", "topic")}
        metadata["native_row_index"] = item["locator"]["row_index"]
        fields = {key: row.get(key) or "" for key in ("lecture", "solution")}
        if any(fields.values()):
            human = dict(sidecar.get("human_solution_metadata", {}), **fields)
    elif source == "hf-chartqa":
        metadata = dict(sidecar.get("native_metadata", {}))
        if "chart_annotation_json" in refs:
            chart = _aux_value(refs["chart_annotation_json"], reader)
            table = reader.path(refs["chart_table_csv"]["file_key"]).read_text()
            visibility = refs.get("chart_visibility_ocr")
            cached_ocr = reader.cache_dir / "objects" / visibility["sha256"] if visibility else None
            if visibility is not None and cached_ocr and cached_ocr.is_file() and file_sha(cached_ocr) == visibility["sha256"]:
                ocr = json.loads(cached_ocr.read_text())
            else:
                from .v18_chart_ocr import visibility_receipt

                ocr = visibility_receipt(image_path, chart, reader.cache_dir, offline=offline)
            raw = {"chart": chart, "table_csv": table, "visibility_receipt": ocr}
            metadata.update(sidecar.get("native_metadata", {}))
    elif source == "hf-geometry3k" and "geometry_problem_zip" in refs:
        data = _aux_value(refs["geometry_problem_zip"], reader)
        raw = _aux_value(refs["geometry_diagram_logic"], reader)
        metadata = {k: data[k] for k in ("is_standard", "problem_type_goal", "problem_type_graph", "source", "unit")}
        metadata.update(native_row_index=item["locator"]["row_index"], original_geometry_id=str(data["id"]),
                        native_diagram_image_sha256=hashlib.sha256(reader.member(refs["geometry_problem_zip"]["file_key"],
                            f"train/{data['id']}/img_diagram.png")).hexdigest())
    elif source == "hf-geometry3k":
        metadata = {"native_row_index": item["locator"]["row_index"],
                    "native_diagram_image_sha256": hashlib.sha256(row["images"][0]["bytes"]).hexdigest()}
    elif source == "hf-gvlqa":
        parts = item["native_id"].split("-")
        task, level, graph = parts[:3]
        group = "-".join((task, level, graph))
        metadata = {"task": task, "level": level, "graph_id": graph, "native_group_id": group,
                    "directed": task in {"flow", "topology"}, "weighted": task in {"flow", "shortest_path"}}
        if "graph_structure" in refs:
            from .content_v16 import parse_gvlqa_structure

            raw = json.loads(json.dumps(parse_gvlqa_structure(
                reader.path(refs["graph_structure"]["file_key"]).read_text())))
            metadata.update(n=raw["n"], m=raw["m"])
    elif source == "hf-clevr":
        scene = row["_scene"]
        raw = {"scene": scene, "program": row["program"]}
        metadata = {"image_filename": row["image_filename"], "image_index": row["image_index"],
            "object_count": len(scene["objects"]), "program_steps": len(row["program"]),
            "question_family_index": row["question_family_index"]}
    return metadata, raw, human


def convert(recipe: dict, sidecars: dict, cache_dir: Path, output: Path, *, tokenizer=None,
            offline: bool = False) -> dict:
    from PIL import Image
    from fine_tuning.notebook_rl.adapters.hf_common import to_problem
    from fine_tuning.notebook_rl.adapters.hf_registry import sources
    from fine_tuning.notebook_rl.contamination import image_fingerprints
    from fine_tuning.notebook_rl.schema import to_json
    from .data_v16b import native_annotation
    from .v17_data import capability_manifest, private_verifier
    from .v18_recipe import portable_payload

    started = time.monotonic()
    per_source: dict = {}
    reader, registry, entries = SourceReader(recipe, cache_dir), sources(), []
    import importlib.util

    common = importlib.util.find_spec(recipe["conversion"]["common_module"])
    if common is None or common.origin is None or file_sha(Path(common.origin)) != recipe["conversion"]["common_sha256"]:
        raise ValueError("V18_COMMON_ADAPTER_CHANGED")
    for source in {x["source"] for x in recipe["items"]}:
        adapter = recipe["sources"][source]["adapter"]
        module = importlib.util.find_spec(adapter["module"])
        if module is None or module.origin is None or file_sha(Path(module.origin)) != adapter["sha256"]:
            raise ValueError(f"V18_SOURCE_ADAPTER_CHANGED:{source}")
    output.mkdir(parents=True, mode=0o700)
    try:
        reader.selected_rows(recipe["items"])
        for selected in recipe["items"]:
            item_started = time.monotonic()
            pid, source, role = selected["problem_id"], selected["source"], selected["role"]
            sidecar = sidecars["items"][pid]
            spec = registry[source][0]
            if spec.revision != selected["source_revision"]:
                raise ValueError("V18_ADAPTER_REVISION_CHANGED")
            made, row = candidate(selected, reader)
            generated = to_problem(made, spec, index=selected["ordinal"])
            if generated.actor.problem_id != pid:
                raise ValueError(f"V18_PROJECT_ID_CHANGED:{pid}")
            actor, verifier, lineage = (asdict(getattr(generated, field)) for field in ("actor", "verifier", "lineage"))
            # JSON tuples/lists are canonicalized identically to the original manifest.
            actor, verifier, lineage = json.loads(json.dumps([actor, verifier, lineage]))
            image_paths, pixels = [], []
            for image, relative, expected in zip(generated.image_bytes, actor["image_paths"], selected["images"], strict=True):
                path = _safe_path(output, "vault/images/" + relative)
                if (relative != expected["path"] or hashlib.sha256(image).hexdigest() != expected["sha256"]):
                    raise ValueError(f"V18_IMAGE_BYTES_CHANGED:{pid}")
                with Image.open(io.BytesIO(image)) as picture:
                    pixel = image_fingerprints(picture)["pixels_sha256"]
                if pixel != expected["pixel_sha256"]:
                    raise ValueError(f"V18_IMAGE_PIXELS_CHANGED:{pid}")
                pixels.append(pixel)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(image)
                path.chmod(0o600)
                image_paths.append(path.relative_to(output).as_posix())
            if spec.label_provenance == "PROGRAM":
                from .v18_chart_ocr import printed_text

                lines = [line for path in image_paths for line in printed_text(output / path, cache_dir, offline=offline)
                         if line]
                generated = replace(generated, verifier=replace(generated.verifier, printed_text=tuple(lines) or ("",)))
                verifier = json.loads(json.dumps(asdict(generated.verifier)))
            metadata, raw, human = source_annotations(selected, row, reader, sidecar,
                output / image_paths[0], offline=offline)
            references_human = dict(human, **sidecar.get("references_human_metadata", {})) if human else None
            entry = {"problem_id": pid, "source": source, "source_revision": spec.revision,
                "native_id": selected["native_id"], "family_id": selected["family_id"],
                "group_id": selected["family_id"], "actor": actor, "verifier": verifier, "lineage": lineage,
                "question": actor["question"], "choices": actor["choices"], "split": role,
                "training_allowed": role in {"train", "prep"}, "benchmark_derived": False, "human_verified": False,
                "image_paths": image_paths, "image_sha256": actor["image_sha256"], "pixel_sha256": pixels,
                "layer": sidecar.get("layer", "unsupported"), "rights": sidecar.get("rights", spec.license),
                "f0_prior": sidecar.get("f0_prior"), "human_solution": human,
                "annotation_refs": sidecar.get("auxiliary_refs", []),
                "contamination_receipt": sidecar.get("contamination_receipt"),
                "inherited_reservations": sidecar.get("inherited_reservations", []),
                "references": {"raw_annotation": raw, "native_metadata": metadata, "human_solution": references_human},
                "record_sha256": {name: hashlib.sha256(to_json(getattr(generated, name)).encode()).hexdigest()
                                  for name in ("actor", "verifier", "lineage")},
                "evaluation": {"gold": verifier["gold"], "answer_type": verifier["answer_type"],
                    "accept": verifier["accept"], "question": actor["question"], "choices": actor["choices"]}}
            entry["annotation"] = native_annotation(entry)
            values = {"actor": actor, "verifier": verifier, "lineage": lineage,
                "question": actor["question"], "choices": actor["choices"], "evaluation": entry["evaluation"],
                "annotation": entry["annotation"], "references": portable_payload(entry["references"]),
                "human_solution": portable_payload(human), "native_metadata": metadata, "raw_annotation": raw}
            for name, value in values.items():
                expected = selected["expected"].get(name + "_sha256")
                if expected and canonical_digest(value) != expected:
                    raise ValueError(f"V18_RECONSTRUCTION_CHANGED:{pid}:{name}")
            entry["v17_private_verifier"] = private_verifier(entry, tokenizer=tokenizer)
            entry["v17_capability"] = capability_manifest(entry, tokenizer=tokenizer)
            entries.append(entry)
            timing = per_source.setdefault(source, {"items": 0, "item_wall_seconds": 0.0})
            timing["items"] += 1
            timing["item_wall_seconds"] += time.monotonic() - item_started
    finally:
        reader.close()
    emission_started = time.monotonic()
    _write(output / "dataset-manifest.json", {"version": VERSION, "items": entries,
        "counts": {role: sum(x["split"] == role for x in entries) for role in ROLES},
        "public_upload_allowed": False, "human_verified": False})
    from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, from_json
    from fine_tuning.notebook_rl.vault import write_split

    for role in ROLES:
        rows = [x for x in entries if x["split"] == role]
        write_split(output / "vault" / role,
            [from_json(ActorItem, json.dumps(x["actor"])) for x in rows],
            [from_json(VerifierItem, json.dumps(x["verifier"])) for x in rows],
            [from_json(Lineage, json.dumps(x["lineage"])) for x in rows])
        _write(output / f"{role}-manifest.json", {"version": VERSION, "split": role,
            "training_allowed": role in {"train", "prep"}, "items": [x for x in entries if x["split"] == role]})
    order = recipe.get("order") or sidecars.get("order")
    if order is None:
        ids = [x["problem_id"] for x in recipe["items"] if x["role"] == "train"]
        order = {"items": [{"problem_id": pid} for pid in ids], "batch_size": 16,
            "batches": [{"batch_index": i // 16, "problem_ids": ids[i:i + 16]} for i in range(0, len(ids), 16)]}
    _write(output / "order.json", order)
    _write(output / "source-provenance.json", {"version": VERSION,
        "items": {x["problem_id"]: {k: x[k] for k in ("source", "source_revision", "native_id", "references",
            "record_sha256", "image_sha256", "human_verified")} for x in entries}})
    if any(x["split"] == "performance" for x in entries):
        from .v17_data import performance_requests

        roles = {role: [x for x in entries if x["split"] == role] for role in ROLES}
        exclusion = canonical_digest({"recipe": recipe.get("content_sha256"),
            "receipts": [x["contamination_receipt"] for x in roles["performance"]]})
        _write(output / "performance-requests.json", performance_requests(roles,
            benchmark_exclusion_sha256=exclusion))
    if sidecars.get("guard"):
        from .guard_v16b import materialize_guard_inputs_v18

        _write(output / "guard-metadata-v18.json", {"guard": sidecars["guard"], "items": {}})
        materialize_guard_inputs_v18(output / "guard-metadata-v18.json", output)
    files = {p.relative_to(output).as_posix(): file_sha(p) for p in output.rglob("*") if p.is_file()}
    _write(output / "payload-receipt.json", {"version": VERSION, "files": files,
        "source_sha256": file_sha(Path(__file__).with_name("v18_data.py")),
        "converter_sha256": file_sha(Path(__file__)), "recipe_content_sha256": recipe.get("content_sha256"), "counts": {role: sum(x["split"] == role for x in entries) for role in ROLES},
        "logical_sha256": canonical_digest(portable_payload(entries))})
    return {"elapsed_seconds": time.monotonic() - started, "per_source": per_source,
        "selected_file_reads": reader.read_timings,
        "manifest_and_vault_wall_seconds": time.monotonic() - emission_started,
        "source_timing_basis": "SERIAL_ITEM_WALL_SECONDS; selected native file reads listed separately"}
