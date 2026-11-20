"""Private, source-bound v16b preparation set and standard notebook vault adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from fine_tuning.notebook_rl import contamination
from fine_tuning.notebook_rl.pod.data_bundle import SPEC_VERSION, expand_spec
from fine_tuning.notebook_rl.schema import ActorItem, Lineage, VerifierItem, from_json, to_json
from fine_tuning.notebook_rl.vault import read_actors, read_lineage, read_verifier, write_split

from . import curation_v16b as curation

VERSION = "v16b-source-bound-data-v1"
FROZEN_VALIDATION_SHA256 = "6c2433be7fb98db217979bcc2c63d45876ac49e3569958b165beb5d7fe5bf215"
LAYERS = {"hf-geometry3k": "geometry", "hf-gvlqa": "graph", "hf-clevr": "scene", "hf-chartqa": "chart"}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=1) + "\n")
    path.chmod(0o600)


def _identities(item: dict) -> set[tuple[str, str]]:
    identities = {("id", item["problem_id"])}
    for key in ("family_id", "connected_family_sha256", "image_sha256", "pixel_sha256"):
        value = item.get(key, [])
        identities.update(("family" if "family" in key else key, x)
                          for x in ([value] if isinstance(value, str) else value))
    if item.get("source") and item.get("native_id"):
        identities.add(("native", f"{item['source']}:{item['native_id']}"))
    return identities


def select_prep(candidates: list[dict], train: list[dict], monitor: list[dict],
                holdouts: list[dict], *, count: int = 512, seed: int = 3407) -> tuple[list[dict], dict]:
    """Condition train proportions on available sources; never use model outcomes."""
    if type(count) is not int or count < 1:
        raise ValueError("prep count must be positive")
    blocked = set().union(*(_identities(x) for x in train + monitor + holdouts))
    buckets: dict[str, list[dict]] = defaultdict(list)
    seen = set(blocked)
    for item in sorted(candidates, key=lambda x: (
            hashlib.sha256(f"{seed}:v16b-prep:{x['problem_id']}".encode()).hexdigest(), x["problem_id"])):
        identities = _identities(item)
        if item.get("eligible") is True and not item.get("exclusion_reasons") and not identities & seen:
            buckets[item["source"]].append(item)
            seen.update(identities)
    weights = Counter(x["source"] for x in train)
    available = sorted(source for source in buckets if weights[source])
    if sum(len(buckets[source]) for source in available) < count:
        raise ValueError("insufficient disjoint eligible preparation families")
    total = sum(weights[source] for source in available)
    quotas = {source: min(count * weights[source] // total, len(buckets[source])) for source in available}
    while sum(quotas.values()) < count:
        source = min((s for s in available if quotas[s] < len(buckets[s])),
                     key=lambda s: (quotas[s] * total - count * weights[s], s))
        quotas[source] += 1
    selected = [item for source in available for item in buckets[source][:quotas[source]]]
    selected.sort(key=lambda x: (hashlib.sha256(f"{seed}:v16b-prep-order:{x['problem_id']}".encode()).hexdigest(), x["problem_id"]))
    return selected, {"rule": "available_train_source_largest_remainder_hash_order_v1", "seed": seed,
        "count": count, "source_counts": quotas, "available_source_counts": {
            source: len(buckets[source]) for source in available},
        "missing_train_sources": sorted(set(weights) - set(available)),
        "selection_uses_model_outcomes": False,
        "supported_observation_items": sum(x.get("annotation_verifier_supported") is True for x in selected)}


def _checked_inputs(root: Path, expected: str) -> tuple[dict, dict, dict, list[dict], list[dict], dict, dict[str, str]]:
    checked: dict[str, str] = {}
    curation._check_ref({"path": str(root / "validation.json"), "sha256": expected}, checked)
    validation = json.loads((root / "validation.json").read_text())
    if validation.get("status") != "PASS_CURATED_NATIVE_3000_WITH_LIMITS":
        raise ValueError("curation is not frozen and validated")
    for ref in validation["artifacts"].values():
        curation._check_ref(ref, checked)
    train = json.loads((root / "train-3000-manifest.json").read_text())
    monitor = json.loads((root / "monitor-manifest.json").read_text())
    order = json.loads((root / "order.json").read_text())
    candidates = curation._rows(root / "candidates.jsonl")
    by_id = curation._by_id(candidates)
    if (train["count"] != len(train["items"]) or monitor["count"] != len(monitor["items"])
            or len({x["problem_id"] for x in train["items"] + monitor["items"]}) != train["count"] + monitor["count"]
            or [x["problem_id"] for x in order["items"]] != [
                pid for block in order["batches"] for pid in block["problem_ids"]]
            or set(x["problem_id"] for x in order["items"]) != set(x["problem_id"] for x in train["items"])):
        raise ValueError("frozen split/order identity mismatch")
    for item in train["items"] + monitor["items"]:
        if curation._manifest_item(by_id[item["problem_id"]]) != item:
            raise ValueError("frozen item differs from audited candidate")
    bindings = train["bindings"]
    for ref in (bindings["inputs"]["holdouts"], bindings["family_supplement"], train["new_text_checks"]):
        curation._check_ref(ref, checked)
    supplement = json.loads(Path(bindings["family_supplement"]["path"]).read_text())
    for ref in supplement["input_refs"].values():
        curation._check_ref(ref, checked)
    original = [dict(x, family_id=x["historical_family_id"]) for x in candidates]
    propagated, holdouts, _ = curation.apply_family_supplement(original,
        curation._rows(Path(bindings["inputs"]["holdouts"]["path"])), supplement)
    if any((a["family_id"], a["eligible"], a["exclusion_reasons"]) !=
           (b["family_id"], b["eligible"], b["exclusion_reasons"])
           for a, b in zip(candidates, propagated, strict=True)):
        raise ValueError("audited family closure mismatch")
    scan = json.loads(Path(train["new_text_checks"]["path"]).read_text())
    if (scan.get("status") != "COMPLETE_FRESH_TEXT_SCAN_WITH_LIMITS"
            or scan.get("thresholds_changed") is not False):
        raise ValueError("fresh text scan incomplete or thresholds changed")
    return train, monitor, order, candidates, holdouts, scan, checked


def _checked_candidate(item: dict, checked: dict[str, str], records: dict,
                       receipts: dict, fields: dict) -> list[dict]:
    pid = item["problem_id"]
    if (item.get("eligible") is not True or item.get("exclusion_reasons")
            or item.get("historical_bindings_status") != "PASS_RECORDS_AND_IMAGES_RECHECKED"):
        raise ValueError("unadmitted candidate")
    rows = []
    for role in ("actor", "verifier", "lineage"):
        path = item["record_paths"][role]
        if path not in records:
            records[path] = curation._by_id(curation._rows(Path(path)))
        row = records[path][pid]
        if curation._digest(row) != item["record_sha256"][role] or row != item[role]:
            raise ValueError("native record binding mismatch")
        rows.append(row)
    actor, verifier, lineage = rows
    if (lineage["benchmark_derived"] is not False or lineage["model_touched"] is not False
            or lineage["label_provenance"] not in {"HUMAN", "PROGRAM"}
            or lineage["split"] != "train" or lineage["source_revision"] != item["source_revision"]
            or actor["source"] != item["source"] or lineage["source"] != item["source"]
            or actor["choices"] != verifier["choices"] or actor["answer_format"] != verifier["answer_type"]
            or actor["image_sha256"] != item["image_sha256"]
            or item["native_id"] != item["manifest_item"]["native_id"]):
        raise ValueError("teacher/benchmark or native identity mismatch")
    for ref in [item["historical_manifest"], item["historical_receipt"],
                *item["native_source_refs"], *item["annotation_refs"]]:
        curation._check_ref(ref, checked)
    receipt_path = item["historical_receipt"]["path"]
    if receipt_path not in receipts:
        receipts[receipt_path] = json.loads(Path(receipt_path).read_text())
    receipt = receipts[receipt_path]
    if pid not in receipt["allowlist"]:
        raise ValueError("candidate not historically allowlisted")
    for path, sha, pixels in zip(item["image_paths"], item["image_sha256"], item["pixel_sha256"], strict=True):
        curation._check_ref({"path": path, "sha256": sha}, checked)
        if receipt["input_files"].get(path) != sha:
            raise ValueError("image exclusion receipt mismatch")
        from PIL import Image

        with Image.open(path) as image:
            if contamination.image_fingerprints(image)["pixels_sha256"] != pixels:
                raise ValueError("native pixel binding mismatch")
    solution = item.get("human_solution")
    if solution:
        if solution.get("authorship") != "dataset_native_expert_curated" or solution.get("human_verified") is not False:
            raise ValueError("human solution lacks native authorship")
        for key in ("source_ref", "provenance_card"):
            curation._check_ref(solution[key], checked)
    payload = {key: item.get(key) for key in ("native_metadata", "raw_annotation")}
    if solution:
        payload["human_solution"] = {key: solution.get(key) for key in ("lecture", "solution", "text")}
    for field, text in curation._strings(payload, "new_native"):
        evidence = fields.get((pid, field))
        if not evidence or evidence["text_sha256"] != hashlib.sha256(text.encode()).hexdigest() or evidence["matches"]:
            raise ValueError("native text not exactly cleared by frozen scan")
    return rows


def prepare(curation_root: Path, output_root: Path, *, prep_count: int = 512,
            seed: int = 3407, validation_sha256: str = FROZEN_VALIDATION_SHA256) -> dict:
    """Create a new bundle directory; keep raw exclusion/model audit files local."""
    curation_root, output_root = curation_root.absolute(), output_root.absolute()
    if output_root.exists():
        raise FileExistsError("refusing to overwrite assembled data")
    train, monitor, order, candidates, holdouts, scan, checked = _checked_inputs(curation_root, validation_sha256)
    prep, selection = select_prep(candidates, train["items"], monitor["items"], holdouts, count=prep_count, seed=seed)
    by_id = curation._by_id(candidates)
    split_items = {"train": [by_id[x["problem_id"]] for x in order["items"]],
                   "monitor": [by_id[x["problem_id"]] for x in monitor["items"]], "prep": prep}
    blocked = set().union(*(_identities(x) for x in holdouts))
    prior = set(blocked)
    for items in split_items.values():
        identities = set().union(*(_identities(x) for x in items))
        if identities & prior:
            raise ValueError("split/family/image/native holdout overlap")
        prior.update(identities)
    records: dict = {}
    receipts: dict = {}
    fields = {(x["problem_id"], x["field"]): x for x in scan["fields"]}
    native = {x["problem_id"]: _checked_candidate(x, checked, records, receipts, fields)
              for items in split_items.values() for x in items}
    output_root.mkdir(mode=0o700, parents=True)
    assembled, provenance = [], {}
    for split, items in split_items.items():
        actors: list = []
        verifiers: list = []
        lineages: list = []
        for item in items:
            pid = item["problem_id"]
            actor, verifier, lineage = native[pid]
            for relative, path in zip(actor["image_paths"], item["image_paths"], strict=True):
                target = output_root / "vault/images" / relative
                if not target.resolve().is_relative_to(output_root / "vault/images"):
                    raise ValueError("image path escapes vault")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(path, target)
                    target.chmod(0o600)
                if curation._sha(target) != item["image_sha256"][actor["image_paths"].index(relative)]:
                    raise ValueError("copied image changed")
            for cls, row, destination in ((ActorItem, actor, actors), (VerifierItem, verifier, verifiers),
                                          (Lineage, lineage, lineages)):
                destination.append(from_json(cls, json.dumps(row)))
            safe_receipt = {"version": VERSION, "problem_id": pid, "family_id": item["family_id"],
                "original_exclusion_sha256": item["historical_receipt"]["sha256"],
                "fresh_text_scan_sha256": train["new_text_checks"]["sha256"],
                "image_sha256": item["image_sha256"], "pixel_sha256": item["pixel_sha256"],
                "fields": [{"field": name, "text_sha256": evidence["text_sha256"], "matches": []}
                           for (key, name), evidence in fields.items() if key == pid],
                "status": "EXCLUSION_CHECKED_WITH_LIMITS", "human_verified": False}
            receipt_path = output_root / "receipts" / f"{hashlib.sha256(pid.encode()).hexdigest()}.json"
            _write(receipt_path, safe_receipt)
            receipt_ref = {"path": str(receipt_path), "sha256": curation._sha(receipt_path)}
            references = {key: item.get(key) for key in ("raw_annotation", "native_metadata")}
            solution = item.get("human_solution")
            if solution:
                references["human_solution"] = {key: value for key, value in solution.items()
                    if key not in {"source_ref", "provenance_card", "contamination_status"}}
                references["human_solution"].update(
                    source_ref={k: v for k, v in solution["source_ref"].items() if k != "path"},
                    provenance_card={k: v for k, v in solution["provenance_card"].items() if k != "path"},
                    contamination_status="PASS_FRESH_TEXT_SCAN_WITH_LIMITS", text_check_receipt=receipt_ref,
                    text_sha256={key: hashlib.sha256(solution[key].encode()).hexdigest()
                                 for key in ("lecture", "solution") if isinstance(solution.get(key), str)})
            entry = {key: item.get(key) for key in ("problem_id", "source", "source_revision", "native_id",
                "family_id", "image_sha256", "pixel_sha256", "rights", "annotation_kind",
                "annotation_support_status", "annotation_verifier_supported")}
            entry.update(group_id=item["family_id"], split=split, training_allowed=split != "monitor",
                benchmark_derived=False, human_verified=False, references=references,
                layer=LAYERS.get(item["source"]) if item["annotation_verifier_supported"] else None,
                image_paths=[str(output_root / "vault/images" / x) for x in actor["image_paths"]],
                record_sha256=item["record_sha256"], contamination_receipt=receipt_ref,
                evaluation={key: value for key, value in verifier.items() if key in {"gold", "answer_type", "accept"}},
                question=actor["question"], choices=actor["choices"],
                f0_prior={key: value for key, value in (item.get("f0_prior") or {}).items() if key != "record_ref"})
            entry["evaluation"].update(question=actor["question"], choices=actor["choices"])
            assembled.append(entry)
            annotations = [{"field": key, "value": value, "sha256": curation._digest(value),
                "location": f"native:{item['source']}#{item['native_id']}:{key}"}
                for key, value in references.items() if value is not None]
            provenance[pid] = {"actor_sha256": item["record_sha256"]["actor"],
                "image_sha256": item["image_sha256"], "source": item["source"], "source_revision": item["source_revision"],
                "split": "train", "native_id": item["native_id"], "label_provenance": lineage["label_provenance"],
                "record_sha256": item["record_sha256"], "human_verified": False,
                "adapter_code_sha256": item["manifest_item"].get("adapter_code_sha256"),
                "label": {"value": verifier["gold"], "sha256": hashlib.sha256(verifier["gold"].encode()).hexdigest(),
                    "location": f"vault:{split}/verifier_private.jsonl#{pid}:gold"},
                "adapter_policy": {"accept": verifier["accept"], "sha256": curation._digest(verifier["accept"]),
                    "provenance": "Frozen adapter policy, not source-authored tolerance"},
                "annotations": annotations, "content_verification_available": item["annotation_verifier_supported"],
                "content_verification_scope": item["annotation_support_status"],
                "note_content_status": "UNKNOWN_REVIEW_REQUIRED", "references": references,
                "reference_use": "PRIVATE_REVIEW_INPUT_ONLY_NOT_TRAINING_TARGET",
                "leak_aliases": [], "acronym_candidates": []}
        write_split(output_root / "vault" / split, actors, verifiers, lineages)
        split_sha = curation._sha(output_root / "vault" / split / "manifest.yaml")
        for item in items:
            provenance[item["problem_id"]]["vault_manifest_sha256"] = split_sha
    _write(output_root / "dataset-manifest.json", {"version": VERSION, "human_verified": False,
        "public_upload_allowed": False, "items": assembled, "counts": {s: len(x) for s, x in split_items.items()}})
    _write(output_root / "prep-manifest.json", {"version": VERSION, "split": "prep", "training_allowed": True,
        "items": [x for x in assembled if x["split"] == "prep"], "selection": selection,
        "curation_validation_sha256": validation_sha256, "human_verified": False})
    dataset_sha = curation._sha(output_root / "dataset-manifest.json")
    for source in provenance.values():
        source["dataset_manifest_sha256"] = dataset_sha
    _write(output_root / "source-provenance.json", {"version": "note-source-provenance-v1", "items": provenance})
    shutil.copyfile(curation_root / "order.json", output_root / "order.json")
    files = sorted(path for path in output_root.rglob("*") if path.is_file())
    inventory = {str(path.relative_to(output_root)): curation._sha(path) for path in files}
    _write(output_root / "payload-receipt.json", {"version": VERSION, "files": inventory,
        "frozen_inputs": {"validation.json": validation_sha256,
            **{name: curation._sha(curation_root / name) for name in ("train-3000-manifest.json", "monitor-manifest.json", "order.json")}},
        "selection": selection, "original_bindings": checked, "human_verified": False,
        "excluded_from_bundle": ["benchmark indices", "raw contamination audits", "historical model outputs", "raw source archives"]})
    _write(output_root / "bundle-spec.json", {"version": SPEC_VERSION,
        "paths": [str(path) for path in files] + [str(output_root / "payload-receipt.json")]})
    result = validate(output_root)
    _write(output_root / "validation.json", result)
    return result


def native_annotation(item: dict) -> dict | None:
    """Expose exact native facts without claiming approval under the earlier O protocol."""
    raw = item.get("references", {}).get("raw_annotation")
    metadata = item.get("references", {}).get("native_metadata") or {}
    if not raw:
        return None
    if item["source"] == "hf-geometry3k":
        return {"diagram_logic_forms": raw["diagram_logic_forms"]}
    if item["source"] == "hf-gvlqa":
        structure = f"{raw['n']} {raw['m']}\n" + "\n".join(" ".join(map(str, edge)) for edge in raw["edges"]) + "\n"
        return {**metadata, "structure_text": structure, "directed": metadata["directed"]}
    if item["source"] == "hf-clevr":
        return raw["scene"]
    return raw


def load_items(root: Path, split: str) -> list[dict]:
    """Read only the requested split, retaining its explicit training permission."""
    if split not in {"train", "prep", "monitor"}:
        raise ValueError("unknown v16b split")
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    actors = {x.problem_id: json.loads(to_json(x)) for x in read_actors(root / "vault" / split)}
    verifiers = {key: json.loads(to_json(x)) for key, x in read_verifier(root / "vault" / split).items()}
    lineages = {key: json.loads(to_json(x)) for key, x in read_lineage(root / "vault" / split).items()}
    items = [x for x in manifest["items"] if x["split"] == split]
    if {x["problem_id"] for x in items} != actors.keys() or actors.keys() != verifiers.keys() or actors.keys() != lineages.keys():
        raise ValueError("assembled manifest/vault identity mismatch")
    return [dict(x, actor=actors[x["problem_id"]], verifier=verifiers[x["problem_id"]],
                 lineage=lineages[x["problem_id"]], annotation=native_annotation(x),
                 **{key: verifiers[x["problem_id"]][key] for key in ("gold", "answer_type", "accept")})
            for x in items]


def validate(root: Path) -> dict:
    receipt = json.loads((root / "payload-receipt.json").read_text())
    for name, sha in receipt["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or curation._sha(path) != sha:
            raise ValueError("assembled payload hash/path mismatch")
    spec_files = expand_spec(json.loads((root / "bundle-spec.json").read_text()))
    if set(spec_files) != {str(root / name) for name in receipt["files"]} | {str(root / "payload-receipt.json")}:
        raise ValueError("bundle contains unbound/audit files")
    counts: dict[str, int] = {}
    previous: set[tuple[str, str]] = set()
    provenance = json.loads((root / "source-provenance.json").read_text())["items"]
    for split in ("train", "monitor", "prep"):
        items = load_items(root, split)
        identities = set().union(*(_identities(x) for x in items))
        if previous & identities:
            raise ValueError("assembled split overlap")
        previous.update(identities)
        counts[split] = len(items)
        for item in items:
            if (item["training_allowed"] is not (split != "monitor")
                    or item["lineage"]["benchmark_derived"] is not False
                    or item["lineage"]["model_touched"] is not False
                    or provenance[item["problem_id"]]["actor_sha256"] != item["record_sha256"]["actor"]):
                raise ValueError("assembled permission/provenance mismatch")
    if counts["prep"] != receipt["selection"]["count"]:
        raise ValueError("preparation count mismatch")
    return {"version": VERSION, "status": "CPU_SOURCE_BOUND_DATA_VALIDATED_WITH_LIMITS",
        "counts": counts, "selection": receipt["selection"], "payload_files": len(receipt["files"]),
        "human_verified": False, "gpu_executed": False,
        "limitations": ["Historical image/OCR/embedding exclusion reused; no new full detector run.",
            "Preparation source proportions conditional on disjoint available sources; missing sources reported.",
            "Native authorship and exact binding are not exhaustive human validation."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--curation-root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare" and args.curation_root is None:
        parser.error("prepare requires --curation-root")
    result = prepare(args.curation_root, args.output_root) if args.command == "prepare" else validate(args.output_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
