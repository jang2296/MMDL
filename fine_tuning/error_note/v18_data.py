"""Pinned source acquisition and portable v18 data preparation (CPU only)."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

VERSION = "v18-source-reconstructed-data-v1"
ROLES = ("train", "prep", "monitor", "effect", "performance")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=1)
        stream.write("\n")
    path.chmod(0o600)


def _safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("V18_PATH_ESCAPE")
    return path


def cache_path(cache_dir: Path, spec: dict) -> Path:
    sha = spec["sha256"]
    if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("V18_INVALID_FILE_SHA256")
    return cache_dir / "objects" / sha


def _checked(path: Path, spec: dict) -> bool:
    return (path.is_file() and not path.is_symlink()
        and path.stat().st_size == spec["bytes"] and file_sha(path) == spec["sha256"])


def _url(spec: dict) -> str:
    if spec["provider"] in {"huggingface", "hf"}:
        if re.fullmatch(r"[0-9a-f]{40}", spec["revision"]) is None:
            raise ValueError("V18_UNPINNED_HF_REVISION")
        return (f"https://huggingface.co/datasets/{quote(spec['repo'], safe='/')}/resolve/"
                f"{spec['revision']}/{quote(spec['path'], safe='/')}")
    url = spec["url"]
    if not url.startswith("https://"):
        raise ValueError("V18_SOURCE_REQUIRES_HTTPS")
    return url


def _prefix_member(spec: dict, cache_dir: Path) -> Path:
    """Reproduce the original selected CLEVR prefix without loading 713 MB of JSON."""
    import zipfile

    archive = cache_dir / "objects" / spec["archive_sha256"]
    decoder, rows, buffer = json.JSONDecoder(), [], ""
    with zipfile.ZipFile(archive) as zipped, zipped.open(spec["member"]) as raw:
        import codecs

        text = codecs.getreader("utf-8")(raw)
        while True:
            buffer += text.read(65536)
            match = re.search(r'"' + re.escape(spec["array_key"]) + r'"\s*:\s*\[', buffer)
            if match:
                buffer = buffer[match.end():]
                break
            if len(buffer) > 1024 * 1024 or not buffer:
                raise ValueError("V18_CLEVR_ARRAY_MISSING")
        while True:
            buffer = buffer.lstrip(" \r\n\t,")
            if buffer.startswith("]"):
                break
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                more = text.read(65536)
                if not more:
                    raise ValueError("V18_CLEVR_PREFIX_INCOMPLETE") from None
                buffer += more
                continue
            buffer = buffer[end:]
            if value["image_index"] >= spec["stop_image_index"]:
                break
            rows.append(value)
    encoded = (json.dumps({spec["array_key"]: rows}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    target = cache_path(cache_dir, spec)
    if len(encoded) != spec["bytes"] or hashlib.sha256(encoded).hexdigest() != spec["sha256"]:
        raise ValueError("V18_CLEVR_PREFIX_HASH_MISMATCH")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
        temporary.chmod(0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


FETCH_ATTEMPTS = 5
TRANSIENT_FETCH_ERRORS = (HTTPError, URLError, TimeoutError, ConnectionError, HTTPException)


def _fetch_one(spec: dict, cache_dir: Path, *, offline: bool) -> dict:
    target = cache_path(cache_dir, spec)
    if _checked(target, spec):
        return {"sha256": spec["sha256"], "cache_hit": True, "requests": 0, "received_bytes": 0}
    if offline:
        raise ValueError(f"V18_CACHE_MISSING_OR_CORRUPT:{spec.get('path', spec['sha256'])}")
    if spec["provider"] == "unresolved":
        raise ValueError(f"V18_UNRESOLVED_SOURCE:{spec.get('reason', spec.get('path'))}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if spec["provider"] == "zip-json-prefix":
        _prefix_member(spec, cache_dir)
        return {"sha256": spec["sha256"], "cache_hit": False, "requests": 0, "received_bytes": 0}
    if spec["provider"] == "zip-range":
        from .v18_data_range import fetch_member

        fetch_member(spec, cache_dir)
        if not _checked(target, spec):
            raise ValueError("V18_RANGE_MEMBER_HASH_MISMATCH")
        metrics_path = cache_dir / "range-metrics" / f"{spec['sha256']}.json"
        metrics = json.loads(metrics_path.read_text())
        return {"sha256": spec["sha256"], "cache_hit": False, "requests": metrics["requests"],
                "received_bytes": metrics["received_bytes"]}
    received, requests = 0, 0
    for attempt in range(FETCH_ATTEMPTS):
        fd, temporary = tempfile.mkstemp(prefix=f".{spec['sha256']}.", dir=target.parent)
        try:
            requests += 1
            request = Request(_url(spec), headers={"User-Agent": "mmdl-v18-data/1"})
            with os.fdopen(fd, "wb") as stream, urlopen(request, timeout=120) as response:
                while True:
                    block = response.read(min(1024 * 1024, spec["bytes"] + 1 - stream.tell()))
                    if not block:
                        break
                    received += len(block)
                    stream.write(block)
                    if stream.tell() > spec["bytes"]:
                        raise ValueError("V18_SOURCE_SIZE_EXCEEDED")
                stream.flush()
                os.fsync(stream.fileno())
            if not _checked(Path(temporary), spec):
                raise ValueError(f"V18_SOURCE_HASH_MISMATCH:{spec.get('path')}")
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
            return {"sha256": spec["sha256"], "cache_hit": False, "requests": requests,
                    "received_bytes": received}
        except TRANSIENT_FETCH_ERRORS as error:
            # A mid-stream read timeout or reset is as transient as a 5xx (local empty-cache run 2026-11-18 died
            # on one TimeoutError after 8.2 GB); the partial temporary file is discarded and the file refetched.
            if attempt == FETCH_ATTEMPTS - 1 or isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(2 ** attempt)
        finally:
            Path(temporary).unlink(missing_ok=True)
    raise AssertionError("unreachable")


def fetch(recipe: dict, cache_dir: Path, *, offline: bool = False, workers: int = 4) -> dict:
    """Verify each cached body once; report task sums separately from transfer wall time."""
    import sys

    if not 1 <= workers <= 4:
        raise ValueError("V18_FETCH_WORKERS_MUST_BE_1_TO_4")
    started = time.monotonic()
    unique: dict[str, dict] = {}
    owners: dict[str, set[str]] = {}
    for spec in recipe["files"].values():
        previous = unique.setdefault(spec["sha256"], spec)
        if previous["bytes"] != spec["bytes"]:
            raise ValueError("V18_CONFLICTING_FILE_SIZE")
    for item in recipe.get("items", []):
        keys = [value for key, value in item["locator"].items() if "file_key" in key]
        keys += [ref.get("file_key") for ref in item.get("auxiliary_refs", [])]
        for key in keys:
            if key is not None:
                owners.setdefault(recipe["files"][key]["sha256"], set()).add(item["source"])
    verified, checks = set(), {}
    for sha, spec in unique.items():
        before = time.monotonic()
        if _checked(cache_path(cache_dir, spec), spec):
            verified.add(sha)
        checks[sha] = time.monotonic() - before
    derived = [x for x in unique.values() if x["provider"] == "zip-json-prefix"]
    archives = {}
    for spec in derived:
        if spec["sha256"] not in verified:
            sha = spec["archive_sha256"]
            archives[sha] = {"provider": "url", "url": spec["url"], "bytes": spec["archive_bytes"],
                             "sha256": sha, "path": "CLEVR_v1.0_no_images.zip"}
            owners.setdefault(sha, set()).update(owners.get(spec["sha256"], set()))
    ordinary = {sha: spec for sha, spec in unique.items() if spec["provider"] != "zip-json-prefix"} | archives
    for sha, spec in archives.items():
        if sha not in checks:
            before = time.monotonic()
            if _checked(cache_path(cache_dir, spec), spec):
                verified.add(sha)
            checks[sha] = time.monotonic() - before
    missing = [spec for sha, spec in ordinary.items() if sha not in verified]
    ranges = [spec for spec in missing if spec["provider"] == "zip-range"]
    plan = {"event": "V18_SOURCE_FETCH_PLAN", "files": len(unique), "workers": workers,
        "source_content_bytes": sum(spec["bytes"] for spec in unique.values()),
        "verified_cached_files": sum(sha in verified for sha in unique),
        "verified_cached_bytes": sum(spec["bytes"] for sha, spec in unique.items() if sha in verified),
        "missing_full_file_bytes": sum(spec["bytes"] for spec in missing if spec["provider"] != "zip-range"),
        "missing_range_members": len(ranges),
        "known_compressed_range_payload_bytes": sum(spec.get("compressed_bytes", 0) for spec in ranges),
        "unknown_compressed_range_members": sum("compressed_bytes" not in spec for spec in ranges),
        "range_directory_header_and_retry_bytes": "UNMEASURED_NOT_INCLUDED"}
    print(json.dumps(plan, sort_keys=True), file=sys.stderr, flush=True)

    def acquire(spec):
        before = time.monotonic()
        # Prefix derivation is local work after its whole archive has been verified.
        row = _fetch_one(spec, cache_dir, offline=offline and spec["provider"] != "zip-json-prefix")
        row["file_task_seconds"] = time.monotonic() - before
        return row

    cached = [spec for sha, spec in ordinary.items() if sha in verified]
    cached += [spec for spec in derived if spec["sha256"] in verified]
    rows: list[dict] = [{"sha256": spec["sha256"], "cache_hit": True, "requests": 0, "received_bytes": 0,
             "file_task_seconds": 0.0} for spec in cached]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows += list(pool.map(acquire, missing))
        rows += list(pool.map(acquire, [spec for spec in derived if spec["sha256"] not in verified]))
    source_tasks: dict[str, dict] = {}
    shared: list[dict] = []
    for row in rows:
        row["sources"] = sorted(owners.get(row["sha256"], set()))
        row["cache_verification_seconds"] = checks.get(row["sha256"], 0.0)
        if len(row["sources"]) == 1:
            source = row["sources"][0]
            total = source_tasks.setdefault(source, {"files": 0, "file_task_seconds": 0.0,
                "cache_verification_seconds": 0.0, "requests": 0, "received_bytes": 0})
            total["files"] += 1
            for key in ("file_task_seconds", "cache_verification_seconds", "requests", "received_bytes"):
                total[key] += row[key]
        else:
            shared.append(row)
    required = ordinary | {spec["sha256"]: spec for spec in derived}
    return {"version": VERSION, "files": len(rows), "required_bytes": sum(spec["bytes"] for spec in required.values()),
        "cache_hits": sum(row["cache_hit"] for row in rows), "requests": sum(row["requests"] for row in rows),
        "received_bytes": sum(row["received_bytes"] for row in rows), "elapsed_seconds": time.monotonic() - started,
        "plan": plan, "per_source": source_tasks, "shared_or_unassigned_file_tasks": shared,
        "source_timing_basis": "SUM_OF_FILE_TASKS_NOT_WALL_TIME; shared tasks are listed separately",
        "objects": rows}


def private_verifier(item: dict, *, tokenizer=None) -> dict:
    from .v17_data import private_verifier as verifier

    return verifier(item, tokenizer=tokenizer)


def load_items(root: Path, split: str) -> list[dict]:
    split = {"main": "train", "development": "monitor"}.get(split, split)
    if split not in ROLES:
        raise ValueError("UNKNOWN_V18_ROLE")
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("NOT_V18_DATA")
    items = [dict(x, **{k: x["verifier"][k] for k in ("gold", "answer_type", "accept")})
             for x in manifest["items"] if x["split"] == split]
    if split == "train":
        order = json.loads((root / "order.json").read_text())
        ids = [x["problem_id"] for x in order["items"]]
        by_id = {x["problem_id"]: x for x in items}
        if len(ids) != len(set(ids)) or set(ids) != by_id.keys() or ids != [
                pid for batch in order["batches"] for pid in batch["problem_ids"]]:
            raise ValueError("V18_ORDER_IDS_OR_BATCHES")
        items = [by_id[pid] for pid in ids]
    for item in items:
        item["image_paths"] = [str(_safe_path(root, p)) for p in item["image_paths"]]
    return items


def data_files(root: Path) -> dict[str, str]:
    return json.loads((root / "payload-receipt.json").read_text())["files"]


def validate(root: Path, *, tokenizer=None, verify_legacy_inputs: bool = False,
             recipe: dict | None = None, sidecars: dict | None = None,
             additional_files: dict[str, str] | None = None) -> dict:
    from .v17_data import capability_manifest, identity_components
    from .v18_recipe import expected_digests, load_recipe, load_sidecars, portable_payload

    recipe = recipe if recipe is not None else load_recipe()
    sidecars = sidecars if sidecars is not None else load_sidecars(recipe)
    trusted = {x["problem_id"]: x for x in recipe["items"]}
    receipt = json.loads((root / "payload-receipt.json").read_text())
    if (receipt["version"] != VERSION or receipt["source_sha256"] != file_sha(Path(__file__))
            or receipt["converter_sha256"] != file_sha(Path(__file__).with_name("v18_data_convert.py"))):
        raise ValueError("V18_DATA_SOURCE_CHANGED")
    if receipt.get("recipe_content_sha256") != recipe.get("content_sha256"):
        raise ValueError("V18_RECIPE_BINDING_CHANGED")
    expected = set(receipt["files"]) | {"payload-receipt.json", "validation.json", "execution.json"}
    for relative, sha in (additional_files or {}).items():
        path = _safe_path(root, relative)
        if path.is_symlink() or file_sha(path) != sha:
            raise ValueError("V18_ADDITIONAL_PAYLOAD_CHANGED")
        expected.add(relative)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual - expected:
        raise ValueError("V18_UNBOUND_PAYLOAD_FILES")
    for relative, sha in receipt["files"].items():
        path = _safe_path(root, relative)
        if path.is_symlink() or file_sha(path) != sha:
            raise ValueError(f"V18_PAYLOAD_CHANGED:{relative}")
    roles = {role: load_items(root, role) for role in ROLES}
    counts = {role: len(rows) for role, rows in roles.items()}
    expected_counts = {role: sum(x["role"] == role for x in recipe["items"]) for role in ROLES}
    if counts != expected_counts or counts != receipt["counts"]:
        raise ValueError("V18_ROLE_COUNTS")
    all_items = [x for rows in roles.values() for x in rows]
    if len(set(x["problem_id"] for x in all_items)) != len(all_items):
        raise ValueError("V18_DUPLICATE_PROBLEM")
    components = identity_components(all_items)
    if len(set(components.values())) != len(all_items):
        raise ValueError("V18_IDENTITY_OVERLAP")
    for role, items in roles.items():
        ids = [x["problem_id"] for x in items]
        expected_ids = [x["problem_id"] for x in recipe["items"] if x["role"] == role]
        if ids != expected_ids:
            raise ValueError("V18_FIXED_ROLE_ORDER_CHANGED")
        for item in items:
            pid = item["problem_id"]
            selected = trusted[pid]
            if item["training_allowed"] is not (role in {"train", "prep"}) or item["benchmark_derived"]:
                raise ValueError("V18_TRAINING_PERMISSION")
            safe = sidecars["items"][pid]
            for field in ("f0_prior", "rights", "layer", "contamination_receipt", "inherited_reservations"):
                if item.get(field) != safe.get(field):
                    raise ValueError(f"V18_FIXED_METADATA_CHANGED:{pid}:{field}")
            if item["annotation_refs"] != safe["auxiliary_refs"]:
                raise ValueError("V18_ANNOTATION_REFS_CHANGED")
            if expected_digests(item) != selected["expected"] or item["family_id"] != selected["family_id"]:
                raise ValueError(f"V18_FROZEN_CONTENT_CHANGED:{pid}")
            for path, image in zip(item["image_paths"], selected["images"], strict=True):
                if file_sha(Path(path)) != image["sha256"]:
                    raise ValueError(f"V18_FROZEN_IMAGE_CHANGED:{pid}")
            if (item["v17_capability"] != capability_manifest(item, tokenizer=tokenizer)
                    or item["v17_private_verifier"] != private_verifier(item, tokenizer=tokenizer)):
                raise ValueError("V18_CAPABILITY_CHANGED")
    from dataclasses import asdict
    from fine_tuning.notebook_rl.vault import read_actors, read_lineage, read_verifier

    for role, items in roles.items():
        actors = read_actors(root / "vault" / role)
        verifiers = read_verifier(root / "vault" / role)
        lineages = read_lineage(root / "vault" / role)
        ids = [x["problem_id"] for x in items]
        if [x.problem_id for x in actors] != ids or set(verifiers) != set(ids) or set(lineages) != set(ids):
            raise ValueError("V18_VAULT_ROLE_IDS_CHANGED")
        for actor in actors:
            for name, record in (("actor", actor), ("verifier", verifiers[actor.problem_id]),
                                 ("lineage", lineages[actor.problem_id])):
                if canonical_digest(asdict(record)) != trusted[actor.problem_id]["expected"][name + "_sha256"]:
                    raise ValueError(f"V18_VAULT_CONTENT_CHANGED:{actor.problem_id}:{name}")
    if sidecars.get("guard"):
        guard = json.loads((root / "guard-metadata-v18.json").read_text())
        if guard != {"items": {}, "guard": sidecars["guard"]}:
            raise ValueError("V18_FIXED_GUARD_METADATA_CHANGED")
    persisted = json.loads((root / "dataset-manifest.json").read_text())["items"]
    if counts["performance"]:
        from .v17_data import performance_requests

        persisted_roles = {role: [x for x in persisted if x["split"] == role] for role in ROLES}
        exclusion = canonical_digest({"recipe": recipe.get("content_sha256"),
            "receipts": [x["contamination_receipt"] for x in persisted_roles["performance"]]})
        expected_requests = performance_requests(persisted_roles, benchmark_exclusion_sha256=exclusion)
        if json.loads((root / "performance-requests.json").read_text()) != expected_requests:
            raise ValueError("V18_PERFORMANCE_REQUESTS_CHANGED")
    logical = canonical_digest(portable_payload(persisted))
    if logical != receipt["logical_sha256"]:
        raise ValueError("V18_LOGICAL_DIGEST_CHANGED")
    return {"version": VERSION, "status": "PASS_CPU_V18_DATA_WITH_LIMITS", "counts": counts,
        "logical_sha256": logical, "capability_recomputed": tokenizer is not None,
        "effect_status": "EXPOSURE_HISTORY_INCOMPLETE", "effect_eligible": False,
        "legacy_files_unchanged": None, "human_verified": False}


def prepare(recipe_path: Path, cache_dir: Path, output: Path, *, offline: bool = False,
            workers: int = 4, tokenizer=None) -> dict:
    from .v18_recipe import load_recipe, load_sidecars
    from .v18_data_convert import convert

    started = time.monotonic()
    recipe = load_recipe(recipe_path)
    sidecars = load_sidecars(recipe, recipe_path=recipe_path)
    if output.exists():
        return validate(output, tokenizer=tokenizer, recipe=recipe, sidecars=sidecars)
    import importlib.metadata

    versions = {name: importlib.metadata.version(name) for name in recipe["conversion"]["dependencies"]}
    if versions != recipe["conversion"]["dependencies"]:
        raise ValueError(f"V18_CONVERSION_DEPENDENCIES_CHANGED:{versions}")
    if tokenizer is None:
        raise ValueError("V18_TOKENIZER_REQUIRED: supply --tokenizer with the pinned local model directory")
    tokenizer_root = Path(tokenizer.name_or_path)
    for name, expected_sha in recipe["conversion"]["tokenizer"]["files"].items():
        if file_sha(tokenizer_root / name) != expected_sha:
            raise ValueError("V18_TOKENIZER_FILES_CHANGED")
    acquisition = fetch(recipe, cache_dir, offline=offline, workers=workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.v18-", dir=output.parent))
    stage.rmdir()
    conversion_started = time.monotonic()
    try:
        conversion = convert(recipe, sidecars, cache_dir, stage, tokenizer=tokenizer, offline=offline)
        result = validate(stage, tokenizer=tokenizer, recipe=recipe, sidecars=sidecars)
        _write(stage / "validation.json", result)
        _write(stage / "execution.json", {"version": VERSION, "acquisition": acquisition, "conversion": conversion,
            "conversion_and_validation_seconds": time.monotonic() - conversion_started,
            "elapsed_seconds": time.monotonic() - started})
        stage.rename(output)
    except BaseException:
        # Preserve the failed attempt separately; the next prepare reuses only verified source objects.
        if stage.exists():
            stage.rename(stage.with_name(stage.name + ".failed"))
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fetch"):
        p = sub.add_parser(name)
        p.add_argument("--recipe", type=Path)
        p.add_argument("--cache-dir", type=Path, required=True)
        p.add_argument("--offline", action="store_true")
        p.add_argument("--workers", type=int, default=4)
        if name == "prepare":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--tokenizer", type=Path)
    p = sub.add_parser("verify")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        result = validate(args.output, tokenizer=tokenizer)
    else:
        from .v18_recipe import DEFAULT_RECIPE, load_recipe

        recipe_path = args.recipe or DEFAULT_RECIPE
        if args.command == "fetch":
            result = fetch(load_recipe(recipe_path), args.cache_dir, offline=args.offline, workers=args.workers)
        else:
            tokenizer = None
            if args.tokenizer:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            result = prepare(recipe_path, args.cache_dir, args.output, offline=args.offline,
                             workers=args.workers, tokenizer=tokenizer)
    print(json.dumps({k: v for k, v in result.items() if k != "objects"}, sort_keys=True))


if __name__ == "__main__":
    main()
