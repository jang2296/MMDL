"""HF-pool candidate build -> the existing MMMU/MMMU-Pro exclusion scan, detectors unchanged.

Stages (each refuses an existing --output directory and writes <output>/receipt.json):
  registry  private source registry from the adapters' SourceSpec; no scanner_policy key
  engines   detector input manifests (exclusion_bundle.prepare_manifests), then OCR on CPU
            and/or SSCD + SigLIP on CUDA (--allow-gpu, idle GPU only). Benchmark-side cache
            entries are copied from an existing engine cache, never recomputed.
  assemble  content-identical fresh root (precedent candidates-v1/build -> build-reviewed-v1),
            exclusion_bundle.assemble on it, candidate pairs + answer-free review packet
  apply     adjudication.apply/validate with reviewed decisions; reviewed receipt + manifest
            are written into that fresh root (METADATA_REBIND_IDENTICAL_INPUT_BYTES)

Index, calibration, thresholds and scanner code are the frozen existing ones. MMMU/MMMU-Pro
content is exclusion signal only: the packet carries benchmark question/choices/image paths,
never an explanation, and new items appear through actor fields only (no verifier gold).
AI review stays human_verified=false. Nothing here grants training permission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import adjudication, exclusion_bundle, vault
from . import contamination as c
from .build_exclusion_index import _private_root
from .dataset import SPLITS, _bytes, _write
from .schema import load_protocol

VERSION = "hf-pool-exclusion-v1"
GPU_STAGES = ("sscd", "siglip-images", "siglip-texts")
STAGES = ("ocr", *GPU_STAGES)
_HERE = Path(__file__).resolve().parent
_CODE = ("hf_pool_exclusion.py", "hf_pool.py", "gen_stage.py", "exclusion_bundle.py", "exclusion_engines.py",
         "contamination.py", "adjudication.py", "schema.py", "vault.py", "protocol_v1.yaml")
_SPLIT_FILES = ("actor_inputs.jsonl", "verifier_private.jsonl", "lineage.jsonl", "manifest.yaml")


def _sha_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _desc(path: Path) -> dict:
    path = Path(path).absolute()
    return {"path": str(path), "file_sha256": _sha_file(path)}


def _new_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"output {path} already exists; refusing to overwrite")
    return _private_root(path, create=True)


def _save(path: Path, value: object) -> dict:
    _write(path, _bytes(value) + b"\n")
    return _desc(path)


def _code_hashes() -> dict[str, str]:
    names = [*_CODE, *(p.relative_to(_HERE).as_posix() for p in sorted((_HERE / "generators").glob("*.py")))]
    return {name: _sha_file(_HERE / name) for name in names}


def _receipt(output: Path, stage: str, status: str, inputs: dict, outputs: dict, **extra) -> dict:
    body = {"version": VERSION, "stage": stage, "status": status, "training_allowed": False,
            "human_verified": False, "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": _code_hashes(),
            "inputs": {k: _desc(v) for k, v in inputs.items()},
            "outputs": {k: _desc(v) for k, v in outputs.items()}, **extra}
    _save(output / "receipt.json", body)
    return body


def _tree_digest(root: Path) -> dict:
    files = {str(p.relative_to(root)): _sha_file(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}
    return {"files": len(files), "files_sha256": c._digest(files)}


def _snapshot(build: Path) -> tuple[dict, dict, dict]:
    return c._snapshot({s: Path(build) / s for s in SPLITS}, Path(build) / "images")


def _benchmark_keys(index_data: dict) -> tuple[set[str], set[str]]:
    texts = exclusion_bundle.text_entries(index_data)
    return set(index_data["images"]), {hashlib.sha256(t.encode()).hexdigest() for t in texts.values()}


# ---------------------------------------------------------------- registry

def _generator_entry(source: str, lineages: list) -> dict:
    """Registry entry for a program-generator source, derived from its own lineage records (no SourceSpec)."""
    if not source.startswith("gen-"):
        raise ValueError(f"registry: source {source!r} has no adapter SourceSpec and is not a gen- generator")
    for lineage in lineages:
        if ((lineage.label_provenance, lineage.license, lineage.upstream_origin, lineage.parents)
                != ("PROGRAM", "project-authored", "procedural", ())):
            raise ValueError(f"registry: {lineage.problem_id} is not a PROGRAM project-authored procedural root lineage")
    revisions = {lineage.source_revision for lineage in lineages}
    if len(revisions) != 1:
        raise ValueError(f"registry: generator {source!r} has inconsistent source_revision values")
    revision = revisions.pop()
    if not re.fullmatch(r"[0-9a-f]{64}", revision):  # dataset.generate_pool pins a sha256 code revision
        raise ValueError(f"registry: generator {source!r} source_revision is not a 64-hex code revision")
    return {"rights_approved": True, "source_known": True, "benchmark_derived": False, "model_touched": False,
            "label_provenance": "PROGRAM", "source_revision": revision, "license": "project-authored",
            "upstream_origin": "procedural", "parents": [], "upstream_split": "generated"}


def registry_for(build: Path, specs: dict) -> dict:
    """specs: source id -> adapters.hf_common.SourceSpec. Every lineage must match its spec;
    sources without a spec are accepted only as consistent 'gen-' program generators."""
    sources, generated = {}, defaultdict(list)
    for split in SPLITS:
        for lineage in vault.read_lineage(Path(build) / split).values():
            spec = specs.get(lineage.source)
            if spec is None:
                if not lineage.source.startswith("gen-"):
                    raise ValueError(f"registry: source {lineage.source!r} has no adapter SourceSpec")
                generated[lineage.source].append(lineage)
                continue
            if ((lineage.source_revision, lineage.license, lineage.upstream_origin, lineage.label_provenance,
                 lineage.parents) != (spec.revision, spec.license, spec.upstream_origin, spec.label_provenance, ())):
                raise ValueError(f"registry: {lineage.problem_id} lineage differs from its SourceSpec")
            sources[lineage.source] = {
                "rights_approved": True, "source_known": True, "benchmark_derived": False, "model_touched": False,
                "label_provenance": spec.label_provenance, "source_revision": spec.revision, "license": spec.license,
                "upstream_origin": spec.upstream_origin, "parents": [], "upstream_split": "train"}
    for source, lineages in generated.items():
        sources[source] = _generator_entry(source, lineages)
    registry = {"registry_id": "notebook-rl-private-hf-pool-v1", "status": "SOURCE_POLICY_ONLY_NOT_TRAINING_APPROVAL",
                "training_allowed": False, "human_verified": False, "upstream_origin_blocklist": [],
                "sources": dict(sorted(sources.items()))}
    for split in SPLITS:
        lineages = vault.read_lineage(Path(build) / split)
        for actor in vault.read_actors(Path(build) / split):
            reasons = c._source_checks(actor, lineages[actor.problem_id], registry)
            if reasons:
                raise ValueError(f"registry: {actor.problem_id} fails source checks {reasons}")
    return registry


def run_registry(build: Path, output: Path, specs: dict | None = None) -> dict:
    if specs is None:
        from .hf_pool import default_registry
        specs = {name: spec for name, (spec, _) in default_registry().items()}
    registry = registry_for(build, specs)
    output = _new_dir(output)
    _save(output / "source_registry.json", registry)
    return _receipt(output, "registry", "SOURCE_REGISTRY_WRITTEN", {"build_manifest": Path(build) / "manifest.yaml"},
                    {"source_registry": output / "source_registry.json"}, sources=sorted(registry["sources"]))


# ---------------------------------------------------------------- engines

def gpu_guard(allow_gpu: bool) -> None:
    if not allow_gpu:
        raise ValueError("engines: CUDA stages need --allow-gpu; refusing to touch the GPU")
    try:
        listing = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"engines: cannot confirm an idle GPU with nvidia-smi ({type(exc).__name__})") from exc
    busy = [line.strip() for line in listing.splitlines() if line.strip()]
    if busy:
        raise ValueError(f"engines: GPU already has compute processes {busy}; refusing")


def _seed(identity: dict, name: str, reuse_root: Path, cache_root: Path, keys: dict[str, set[str]]) -> tuple[Path, dict]:
    """Copy existing benchmark-side entries (regular files, nlink 1) into a fresh merged cache root."""
    digest = c._digest(identity)
    source = Path(reuse_root) / name / digest
    if not (source / "identity.json").is_file() or vault._read(source / "identity.json") != c._json(identity):
        raise ValueError(f"engines: no reusable {name} cache with identity {digest} under {reuse_root}; "
                         "refusing to recompute the benchmark side")
    missing = {kind: sorted(k for k in wanted if not (source / kind / f"{k}.json").is_file())
               for kind, wanted in keys.items()}
    if any(missing.values()):
        raise ValueError(f"engines: reuse cache {source} lacks benchmark entries "
                         f"{ {kind: len(v) for kind, v in missing.items()} }; refusing to recompute them")
    target = _private_root(Path(cache_root) / name / digest, create=True)
    for kind in ("images", "texts", "failures"):
        (target / kind).mkdir(mode=0o700, exist_ok=True)
    copies = [("identity.json", "identity.json")] + [(f"{kind}/{k}.json",) * 2 for kind, v in keys.items() for k in v]
    for src, dst in copies:  # new files only ("xb"); no fsync, produce()/assemble re-validate every entry
        with (target / dst).open("xb") as stream:
            stream.write(vault._read(source / src))
    return target, {kind: len(v) for kind, v in keys.items()}


def run_engines(args: argparse.Namespace) -> dict:
    from . import exclusion_engines as e

    stages = list(dict.fromkeys(args.stages))
    gpu = [s for s in stages if s in GPU_STAGES]
    if "sscd" in stages and not args.sscd_artifact or {"siglip-images", "siglip-texts"} & set(stages) and not args.siglip_snapshot:
        raise ValueError("engines: --sscd-artifact / --siglip-snapshot required for the requested stages")
    if "ocr" in stages and not args.tesseract_root:
        raise ValueError("engines: --tesseract-root required for ocr")
    if gpu:
        gpu_guard(args.allow_gpu)
    calibration = exclusion_bundle.read_json(args.calibration)
    output = _new_dir(args.output)
    manifests = exclusion_bundle.prepare_manifests(args.index, args.raw_images, args.build, output / "detector-manifests")
    images = e.load_manifest(output / "detector-manifests/images.json")
    texts = e.load_manifest(output / "detector-manifests/texts.json", texts=True)
    bench_images, bench_texts = _benchmark_keys(exclusion_bundle.read_json(args.index))
    items, _, build_images = _snapshot(args.build)
    human = {sha for item in items.values() if item["lineage"].label_provenance == "HUMAN" for sha in item["image_sha256"]}
    cache_root, report = output / "engine-cache", {}

    def stage(name: str, engine, kind: str, sources: dict, bench: set[str], **kwargs) -> None:
        started = time.monotonic()
        result = e.produce(engine, sources, cache_root, texts=kind == "texts", **kwargs)
        expected_new = len(set(sources) - bench)
        if result["failed_entries"] or result["new_entries"] != expected_new or result["reused_entries"] != len(sources) - expected_new:
            result["status"] = "FAILED_OR_UNEXPECTED_COUNTS"
        report[name] = {**result, "elapsed_seconds": time.monotonic() - started, "benchmark_entries_copied": len(bench & set(sources))}

    if "ocr" in stages:
        ocr = e.OCR(args.tesseract_root, timeout=args.timeout)
        _seed(ocr.identity, "ocr", args.reuse_cache_root, cache_root, {"images": bench_images})
        stage("ocr", ocr, "images", {k: images[k] for k in bench_images | human}, bench_images, workers=args.workers)
    if "sscd" in stages:
        sscd = e.SSCD(args.sscd_artifact, device="cuda")
        if sscd.identity != calibration["engines"]["sscd"]:
            raise ValueError("engines: SSCD identity differs from the calibrated identity; recalibration is out of scope")
        _seed(sscd.identity, "sscd", args.reuse_cache_root, cache_root, {"images": bench_images})
        stage("sscd", sscd, "images", images, bench_images, batch_size=args.batch_size)
    if {"siglip-images", "siglip-texts"} & set(stages):
        siglip = e.SigLIP(args.siglip_snapshot, device="cuda")
        if siglip.identity != calibration["engines"]["siglip"]:
            raise ValueError("engines: SigLIP identity differs from the calibrated identity; recalibration is out of scope")
        wanted = {"images": bench_images if "siglip-images" in stages else set(),
                  "texts": bench_texts if "siglip-texts" in stages else set()}
        _seed(siglip.identity, "siglip", args.reuse_cache_root, cache_root, wanted)
        if "siglip-images" in stages:
            stage("siglip-images", siglip, "images", images, bench_images, batch_size=args.batch_size)
        if "siglip-texts" in stages:
            stage("siglip-texts", siglip, "texts", texts, bench_texts, batch_size=args.text_batch_size)
    roots = {r["cache_root"] for r in report.values()}
    failed = [name for name, r in report.items() if r["status"] != "PARTIAL_CACHE_ONLY"]
    return _receipt(output, "engines", "ENGINE_CACHES_FAILED" if failed else "ENGINE_CACHES_COMPLETE",
                    {"build_manifest": Path(args.build) / "manifest.yaml", "index": args.index, "calibration": args.calibration},
                    {"images_manifest": output / "detector-manifests/images.json",
                     "texts_manifest": output / "detector-manifests/texts.json"},
                    stages=report, failed_stages=failed, detector_manifests=manifests,
                    build_images=len(build_images), human_images=len(human),
                    cache_roots={root: _tree_digest(Path(root)) for root in sorted(roots)})


# ---------------------------------------------------------------- assemble

def _cache_coverage(root: Path | None, required: dict[str, set[str]], expected: dict | None,
                    bench: dict[str, set[str]]) -> dict:
    out: dict = {"cache_root": None if root is None else str(root), "blockers": []}
    if root is None:
        out["blockers"].append("NO_CACHE_ROOT_SUPPLIED")
    else:
        identity = exclusion_bundle.read_json(Path(root) / "identity.json")
        out["identity_sha256"] = c._digest(identity)
        if out["identity_sha256"] != Path(root).name:
            out["blockers"].append("CACHE_DIRECTORY_DIFFERS_FROM_IDENTITY")
        if (identity.get("producer_sha256") != _sha_file(_HERE / "exclusion_engines.py")
                or identity.get("normalizer_sha256") != _sha_file(_HERE / "contamination.py")):
            out["blockers"].append("CACHE_PRODUCER_OR_NORMALIZER_CODE_CHANGED")
        if expected is not None and identity != expected:
            out["blockers"].append("CACHE_IDENTITY_DIFFERS_FROM_CALIBRATION")
    if expected is not None:
        out["calibrated_identity_sha256"] = c._digest(expected)
    for kind, keys in required.items():
        present = set() if root is None else {k for k in keys if (Path(root) / kind / f"{k}.json").is_file()}
        missing = keys - present
        out[kind] = {"required": len(keys), "present": len(present), "missing": len(missing),
                     "missing_benchmark_side": len(missing & bench[kind]),
                     "missing_build_side": len(missing - bench[kind]), "missing_sample": sorted(missing)[:5]}
        if missing:
            out["blockers"].append(f"MISSING_{kind.upper()}_ENTRIES")
    return out


def preflight(build: Path, index_data: dict, calibration: dict, registry: dict, caches: dict[str, Path | None]) -> dict:
    items, _, images = _snapshot(build)
    bench_images, bench_texts = _benchmark_keys(index_data)
    human = {sha for item in items.values() if item["lineage"].label_provenance == "HUMAN" for sha in item["image_sha256"]}
    own_texts = {item["text_sha256"] for item in items.values()}
    blockers = []
    if "scanner_policy" in registry:
        blockers.append("REGISTRY_ALREADY_HAS_SCANNER_POLICY")
    def source_checks(item: dict) -> list[str]:
        try:
            return c._source_checks(item["actor"], item["lineage"], registry)
        except ValueError as exc:
            return [f"PREREQUISITE:{exc}"]
    rejected = {pid: source_checks(item) for pid, item in items.items()}
    if any(rejected.values()):
        blockers.append("SOURCE_CHECKS_REJECTED")
    # contamination.ocr_scan raises for PROGRAM items without printed_text; the scan would be BLOCKED.
    program_no_text = sorted(pid for pid, item in items.items()
                             if item["lineage"].label_provenance == "PROGRAM" and not item["verifier"].printed_text)
    if program_no_text:
        blockers.append("PROGRAM_ITEMS_WITHOUT_PRINTED_TEXT")
    if calibration.get("protocol_sha256") != load_protocol(_HERE / "protocol_v1.yaml")["sha256"]:
        blockers.append("CALIBRATION_PROTOCOL_DIFFERS")
    if calibration.get("index_sha256") != c._digest(index_data):
        blockers.append("CALIBRATION_INDEX_DIFFERS")
    bench = {"images": bench_images, "texts": bench_texts}
    coverage = {
        "ocr": _cache_coverage(caches.get("ocr"), {"images": bench_images | human}, None, bench),
        "sscd": _cache_coverage(caches.get("sscd"), {"images": bench_images | set(images)},
                                calibration["engines"]["sscd"], bench),
        "siglip": _cache_coverage(caches.get("siglip"), {"images": bench_images | set(images),
                                                         "texts": bench_texts | own_texts},
                                  calibration["engines"]["siglip"], bench)}
    blockers += [f"{name.upper()}:{b}" for name, cov in coverage.items() for b in cov["blockers"]]
    return {"status": "PREFLIGHT_BLOCKED" if blockers else "PREFLIGHT_PASSED", "blockers": blockers,
            "items": len(items), "build_images": len(images), "human_images": len(human),
            "label_provenance": {p: sum(i["lineage"].label_provenance == p for i in items.values()) for p in ("HUMAN", "PROGRAM")},
            "program_items_without_printed_text": program_no_text,
            "source_check_rejections": {pid: r for pid, r in rejected.items() if r}, "coverage": coverage}


def copy_build(build: Path, target: Path) -> dict:
    """Byte-identical vault + image copy; no root manifest/receipt (written later by apply)."""
    build, target = Path(build).absolute(), _new_dir(target)
    items, files, _ = _snapshot(build)
    for split in SPLITS:
        for name in _SPLIT_FILES:
            _write(target / split / name, vault._read(build / split / name))
    for item in items.values():
        for rel in item["actor"].image_paths:
            if not (target / "images" / rel).exists():
                _write(target / "images" / rel, vault._read(build / "images" / rel))
    _, new_files, _ = _snapshot(target)
    old_map = {str(Path(p).relative_to(build)): s for p, s in files.items()}
    new_map = {str(Path(p).relative_to(target)): s for p, s in new_files.items()}
    if old_map != new_map:
        raise ValueError("assemble: fresh root differs from the source build")
    return {"status": "SAME_CONTENT_DISTINCT_PATH_BINDING_VERIFIED", "files": len(new_map),
            "relative_file_hash_map_sha256": c._digest(new_map), "old_input_files_sha256": c._digest(files),
            "new_input_files_sha256": c._digest(new_files)}


def _refkey(ref: dict) -> str:
    return "|".join(ref[k] for k in ("dataset_id", "configuration", "split", "id"))


def _matches(candidate: dict, image_refs: dict, text_keys: dict) -> list[tuple[str, str | None]]:
    stage, ref = candidate["stage"], candidate.get("ref", {})
    if stage in ("question_exact", "text_ngrams"):
        return [(_refkey(ref), None)] if isinstance(ref, dict) else []
    if stage in ("ocr_ngrams", "perceptual"):
        return [(_refkey(r), None) for r in ref["benchmark_refs"]]
    if stage == "image_exact":
        return [(_refkey(r), None) for refs in ref.values() for r in refs]
    if stage in ("sscd", "siglip"):
        if candidate["reason"].endswith("_TO_TEXT"):
            return [text_keys[ref]]
        return [(key, None) for key in image_refs.get(ref, [])]
    return []  # source_rights / family: no benchmark record


def review_packet(receipt: dict, receipt_desc: dict, index_data: dict, items: dict,
                  build_root: Path, raw_images: Path) -> dict:
    """Unique candidate pairs with whitelisted fields only: no explanation, no gold, no answer."""
    image_refs: dict[str, list[str]] = defaultdict(list)
    text_keys: dict[str, tuple[str, str]] = {}
    records = {}
    for record in index_data["records"]:
        ref = record["ref"]
        key = _refkey(ref)
        records[key] = record
        for sha in record["image_sha256"]:
            image_refs[sha].append(key)
        for field, values in record["texts"].items():
            for n in range(len(values)):
                text_keys[c._digest([ref["dataset_id"], ref["configuration"], ref["split"], ref["id"], field, n])] = (key, field)
    pairs: dict[str, dict] = {}
    for candidate in receipt["candidates"]:
        cid = adjudication.candidate_id(candidate)
        if cid in pairs:
            pairs[cid]["occurrences"] += 1
            continue
        matches = _matches(candidate, image_refs, text_keys)
        pairs[cid] = {"candidate_id": cid, "occurrences": 1, "problem_id": candidate["problem_id"],
                      "stage": candidate["stage"], "reason": candidate["reason"], "action": candidate["action"],
                      "releasable": (candidate["stage"] in adjudication._RELEASABLE
                                     and candidate["reason"] != "EXACT_RAW_OR_PIXEL_IMAGE"),
                      "benchmark_records": sorted({k for k, _ in matches}),
                      "benchmark_text_fields": sorted({f for _, f in matches if f}), "candidate": candidate}
    used_items = sorted({p["problem_id"] for p in pairs.values()})
    used_records = sorted({k for p in pairs.values() for k in p["benchmark_records"]})
    item_view = {pid: {"problem_id": pid, "split": items[pid]["split"], "source": items[pid]["actor"].source,
                       "question": items[pid]["actor"].question, "choices": list(items[pid]["actor"].choices),
                       "image_paths": [str(Path(build_root) / "images" / rel) for rel in items[pid]["actor"].image_paths],
                       "image_sha256": list(items[pid]["actor"].image_sha256)} for pid in used_items}
    record_view = {key: {"ref": {k: records[key]["ref"][k] for k in ("dataset_id", "configuration", "split", "id")},
                         "question": list(records[key]["texts"]["question"]),
                         "choices": list(records[key]["texts"]["choices"]),
                         "image_fields": list(records[key]["image_fields"]),
                         "image_paths": [str(Path(raw_images) / sha) for sha in records[key]["image_sha256"]],
                         "explanation_withheld": True} for key in used_records}
    return {"format": "hf-pool-exclusion-review-packet-v1", "human_verified": False, "training_allowed": False,
            "instructions": ("One decision per candidate_id: DISTINCT, DUPLICATE or UNRESOLVED, with a reason and "
                             "evidence [{path, file_sha256}] for what was compared. Non-releasable candidates stay "
                             "quarantined whatever the verdict; DUPLICATE/UNRESOLVED quarantine the whole family."),
            "review_template": {"format": adjudication.FORMAT, "raw_receipt_sha256": receipt_desc["file_sha256"],
                                "index_sha256": receipt["index_sha256"], "input_files_sha256": receipt["input_files_sha256"],
                                "provenance": {"kind": "AI_DIRECT_SIMILARITY_REVIEW", "human_verified": False,
                                               "reviewer_id": "", "reviewed_at_utc": ""}, "decisions": []},
            "counts": {"candidate_rows": len(receipt["candidates"]), "unique_candidates": len(pairs),
                       "items": len(used_items), "benchmark_records": len(used_records)},
            "candidates": list(pairs.values()), "items": item_view, "benchmark_records": record_view}


def run_assemble(args: argparse.Namespace) -> tuple[dict, int]:
    index_data = exclusion_bundle.read_json(args.index)
    calibration = exclusion_bundle.read_json(args.calibration)
    registry = exclusion_bundle.read_json(args.registry)
    caches = {"ocr": args.ocr_cache, "sscd": args.sscd_cache, "siglip": args.siglip_cache}
    output = _new_dir(args.output)
    inputs = {"build_manifest": Path(args.build) / "manifest.yaml", "index": args.index,
              "calibration": args.calibration, "registry": args.registry}
    started = time.monotonic()
    checked = preflight(args.build, index_data, calibration, registry, caches)
    _save(output / "preflight.json", checked)
    if checked["blockers"]:
        return _receipt(output, "assemble", "BLOCKED_PREFLIGHT", inputs, {"preflight": output / "preflight.json"},
                        blockers=checked["blockers"], elapsed_seconds=time.monotonic() - started), 2
    fresh = output / "build-reviewed"
    try:
        parity = copy_build(args.build, fresh)
        summary = exclusion_bundle.assemble(args.index, fresh, args.calibration,
                                            {k: Path(v) for k, v in caches.items()}, args.registry, output / "raw-fullscan")
    except ValueError as exc:
        return _receipt(output, "assemble", "FAILED", inputs, {"preflight": output / "preflight.json"},
                        error=str(exc), elapsed_seconds=time.monotonic() - started), 2
    outputs = {name: output / "raw-fullscan" / name for name in ("receipt_v2.json", "source_registry.json", "summary.json")}
    if summary["status"] != "EXCLUSION_CHECKED_WITH_LIMITS":
        return _receipt(output, "assemble", "SCAN_BLOCKED", inputs, outputs, summary=summary, parity=parity,
                        elapsed_seconds=time.monotonic() - started), 2
    receipt_desc = _desc(output / "raw-fullscan/receipt_v2.json")
    receipt = exclusion_bundle.read_json(output / "raw-fullscan/receipt_v2.json")
    items, _, _ = _snapshot(fresh)
    packet = review_packet(receipt, receipt_desc, index_data, items, fresh, args.raw_images)
    outputs["candidates"] = output / "review/candidates.json"
    outputs["review_packet"] = output / "review/review_packet.json"
    _save(outputs["candidates"], [{k: v for k, v in p.items() if k != "candidate"} for p in packet["candidates"]])
    _save(outputs["review_packet"], packet)
    return _receipt(output, "assemble", "RAW_SCAN_COMPLETE_AWAITING_ADJUDICATION", inputs, outputs,
                    summary=summary, parity=parity, review_counts=packet["counts"],
                    evidence_file_sha256=_sha_file(output / "raw-fullscan/evidence.json"),
                    elapsed_seconds=time.monotonic() - started), 0


# ---------------------------------------------------------------- apply

def run_apply(assemble_dir: Path, decisions: Path, output: Path) -> dict:
    assemble_dir = Path(assemble_dir).absolute()
    prior = exclusion_bundle.read_json(assemble_dir / "receipt.json")
    if prior.get("status") != "RAW_SCAN_COMPLETE_AWAITING_ADJUDICATION":
        raise ValueError("apply: assemble stage did not complete a raw scan")
    build_manifest = Path(prior["inputs"]["build_manifest"]["path"])
    if _sha_file(build_manifest) != prior["inputs"]["build_manifest"]["file_sha256"]:
        raise ValueError("apply: source build manifest changed since assemble")
    build, fresh = build_manifest.parent, assemble_dir / "build-reviewed"
    if (fresh / "manifest.yaml").exists() or (fresh / "contamination").exists():
        raise ValueError("apply: fresh root already has a manifest/receipt; refusing to overwrite")
    output = _new_dir(output)
    items, files, _ = _snapshot(fresh)
    raw, review = _desc(assemble_dir / "raw-fullscan/receipt_v2.json"), _desc(decisions)
    reviewed = adjudication.apply(raw, review, items)
    adjudication.validate(reviewed, items)
    if (reviewed["status"] != "EXCLUSION_CHECKED_WITH_LIMITS" or reviewed["input_files"] != files
            or reviewed["input_files_sha256"] != c._digest(files)):
        raise ValueError("apply: reviewed receipt is not a complete binding of the fresh root")
    receipt = _save(fresh / "contamination/receipt_v2.json", reviewed)
    manifest = exclusion_bundle.read_json(build_manifest) | {"exclusion_receipt_sha256": receipt["file_sha256"]}
    new_manifest = _save(fresh / "manifest.yaml", manifest)
    rebind = _save(output / "metadata_rebind.json", {
        "kind": "METADATA_REBIND_IDENTICAL_INPUT_BYTES",
        "old_to_new_files": {str(build / Path(p).relative_to(fresh)): p for p in files},
        "original_build_manifest_sha256": _sha_file(build_manifest),
        "new_build_manifest_sha256": new_manifest["file_sha256"]})
    adj = reviewed["adjudication"]
    return _receipt(output, "apply", reviewed["status"],
                    {"raw_receipt": Path(raw["path"]), "decisions": decisions, "source_build_manifest": build_manifest},
                    {"reviewed_receipt": fresh / "contamination/receipt_v2.json", "build_manifest": fresh / "manifest.yaml",
                     "metadata_rebind": Path(rebind["path"])},
                    build_root=str(fresh), counts=adj["counts"], allowlist=len(reviewed["allowlist"]),
                    blacklisted_problem_ids=adj["blacklisted_problem_ids"],
                    unresolved_problem_ids=adj["unresolved_problem_ids"],
                    binding={k: reviewed[k] for k in ("index_sha256", "index_file_sha256", "source_registry_sha256",
                                                      "evidence_sha256", "input_files_sha256")})


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fine_tuning.notebook_rl.hf_pool_exclusion", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("registry")
    r.add_argument("--build", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("engines")
    for name in ("build", "index", "raw-images", "calibration", "reuse-cache-root", "output"):
        e.add_argument(f"--{name}", type=Path, required=True)
    e.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    e.add_argument("--tesseract-root", type=Path)
    e.add_argument("--sscd-artifact", type=Path)
    e.add_argument("--siglip-snapshot", type=Path)
    e.add_argument("--allow-gpu", action="store_true")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--timeout", type=float, default=60)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--text-batch-size", type=int, default=64)
    a = sub.add_parser("assemble")
    for name in ("build", "index", "raw-images", "calibration", "registry", "output"):
        a.add_argument(f"--{name}", type=Path, required=True)
    for name in ("ocr-cache", "sscd-cache", "siglip-cache"):
        a.add_argument(f"--{name}", type=Path)
    p = sub.add_parser("apply")
    p.add_argument("--assemble-dir", type=Path, required=True)
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    code = 0
    try:
        if args.command == "registry":
            result = run_registry(args.build, args.output)
        elif args.command == "engines":
            result = run_engines(args)
            code = 2 if result["failed_stages"] else 0
        elif args.command == "assemble":
            result, code = run_assemble(args)
        else:
            result = run_apply(args.assemble_dir, args.decisions, args.output)
    except ValueError as exc:
        print(json.dumps({"status": "REFUSED", "stage": args.command, "error": str(exc)}), flush=True)
        return 2
    print(json.dumps({k: result[k] for k in ("stage", "status") if k in result}
                     | {k: result[k] for k in ("blockers", "failed_stages", "counts", "allowlist") if k in result}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
