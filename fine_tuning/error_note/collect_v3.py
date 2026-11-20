"""Source-reviewed collection binding and gold-blind lesson transfer for the v3 path.

The verifier subprocess checks private note/audit files; the generation process receives
only approved lessons. Review receipts describe source checks, never replacement targets.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT

from .audit import FILE_VERSION, audit_note, digest, read_audits, text_sha256
from .contract import IMAGE, MAX_NEW_TOKENS, NOTE_INSTRUCTION_V3, RESOLVE_HEADER, note_prompt_v3, parse_note_v2

ANSWER_CONTRACT = TRAINING_ANSWER_CONTRACT
COLLECTION_VERSION = "error-note-collection-v3"
LESSONS_VERSION = "error-note-lessons-v3"


def source_identity() -> str:
    """Bind production source, including transitive helpers, without importing model/CUDA code."""
    root = Path(__file__).resolve().parents[1]
    files = {}
    excluded = {"tests", "runs", "generated", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(n for n in names if n.endswith(".py")):
            path = Path(directory) / name
            files[path.relative_to(root).as_posix()] = file_sha(path)
    return digest(files)


def generation_identity(args) -> dict:
    from fine_tuning.modules import BASE_REVISION

    engine = getattr(args, "engine", None)
    if engine not in ("hf", "vllm"):
        raise ValueError("the first v3 collection binding must come from a generation stage")
    model = Path(args.model_path).resolve()
    if model.name != BASE_REVISION:
        raise ValueError("v3 collection must start from the pinned original model snapshot")
    files = sorted(p for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".jinja"))
    if not any(p.suffix == ".safetensors" for p in files) or not (model / "config.json").is_file():
        raise ValueError("v3 model snapshot is incomplete")
    versions: dict[str, str | None] = {}
    for name in ("torch", "transformers", "qwen-vl-utils", "vllm", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"engine": engine, "precision": "nf4" if engine == "hf" else "bf16",
            "model_path": str(model), "model_files_sha256": digest({p.name: file_sha(p) for p in files}),
            "code_commit": getattr(args, "code_commit", None), "packages": versions,
            "python": list(sys.version_info[:3]), "seed": 3407,
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
                         "repetition_penalty": 1.0, "num_beams": 1}}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    from .collect import write_json

    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"v3 immutable evidence changed: {path.name}; use a new output directory")
    else:
        write_json(path, value)


def bind_collection(args, items: list[dict]) -> dict:
    """Pin public settings before generation; private rules are verified only by the service."""
    if getattr(args, "answer_contract", None) != ANSWER_CONTRACT:
        raise ValueError(f"v3 collection requires {ANSWER_CONTRACT}")
    rules, rules_sha = getattr(args, "training_rules", None), getattr(args, "training_rules_sha256", None)
    if rules is None or not isinstance(rules_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", rules_sha):
        raise ValueError("v3 collection requires pinned private training rules")
    from fine_tuning.modules import BASE_ID, BASE_REVISION

    path = Path(args.output) / "collection.json"
    stored = json.loads(path.read_text()) if path.exists() else None
    generation = generation_identity(args) if getattr(args, "engine", None) else \
        stored.get("generation") if isinstance(stored, dict) else None
    if generation is None:
        raise ValueError("audit/rows cannot create a missing generation binding")
    body = {"version": COLLECTION_VERSION, "answer_contract": ANSWER_CONTRACT,
            "training_rules_sha256": rules_sha, "order_sha256": file_sha(args.order),
            "items_sha256": digest(items), "limit": len(items), "model_id": BASE_ID, "revision": BASE_REVISION,
            "max_new_tokens": MAX_NEW_TOKENS, "image": IMAGE,
            "note_prompt_sha256": text_sha256(NOTE_INSTRUCTION_V3),
            "resolve_header_sha256": text_sha256(RESOLVE_HEADER),
            "source_sha256": source_identity(), "generation": generation}
    if not path.exists() and any((Path(args.output) / name).exists()
                                 for name in ("t1.jsonl", "note.jsonl", "resolve.jsonl")):
        raise ValueError("v3 cannot silently reuse unbound legacy collection files")
    _write_once(path, body)
    return body


def _items_file(path: Path, version: str) -> dict:
    data = json.loads(Path(path).read_text())
    if (not isinstance(data, dict) or set(data) != {"version", "items"} or
            data["version"] != version or not isinstance(data["items"], dict)):
        raise ValueError(f"invalid {version} file")
    return data["items"]


def records_and_sources(args) -> tuple[list[dict], dict, dict]:
    from fine_tuning.notebook_rl.prompts import p0_eval_text
    from fine_tuning.notebook_rl.vault import read_actors, read_verifier
    from .collect import SKIPPED_T1_STATUSES, _prefix, make_records_v2, read_jsonl_done

    items, out = _prefix(args), Path(args.output)
    bind_collection(args, items)
    actors = {a.problem_id: a for a in read_actors(Path(args.build_root) / "train")}
    t1 = read_jsonl_done(out / "t1.jsonl", repair=False)
    notes = read_jsonl_done(out / "note.jsonl", repair=False)
    missing = [i["problem_id"] for i in items if i["problem_id"] not in notes]
    if missing:
        raise ValueError(f"v3 audit needs closed note rows for every item: {missing[:3]}")
    gold = read_verifier(Path(args.build_root) / "train")
    for item in items:
        pid, row = item["problem_id"], notes[item["problem_id"]]
        if row.get("note_instruction_sha256") != text_sha256(NOTE_INSTRUCTION_V3):
            raise ValueError("v3 note instruction binding mismatch")
        if str(row.get("status", "")).startswith("SKIPPED_"):
            if t1[pid]["status"] not in SKIPPED_T1_STATUSES or row["status"] != "SKIPPED_" + t1[pid]["status"]:
                raise ValueError("v3 skipped note binding mismatch")
            continue
        if row.get("prompt") != note_prompt_v3(
                p0_eval_text(actors[pid].question, actors[pid].choices), t1[pid].get("text", ""), t1[pid]["status"],
                gold[pid].gold):
            raise ValueError("v3 note prompt binding mismatch")
    records = make_records_v2(items, actors, t1, notes, {})
    sources = _items_file(args.source_provenance, "note-source-provenance-v1")
    if not {r["problem_id"] for r in records} <= set(sources):
        raise ValueError("missing source provenance for selected items")
    return records, sources, actors


def _lessons(records: list[dict], audits: dict) -> dict:
    result = {}
    for record in records:
        pid, audit = record["problem_id"], audits[record["problem_id"]]
        parsed = parse_note_v2(record["note"]["text"]) if record.get("note") else None
        if parsed is None or audit["format"]["result"] != "PASS":
            reason = "NOTE_UNPARSED"
        elif audit["content"]["result"] != "PASS" or audit["content"]["scope"] != "entire_note":
            reason = "CONTENT_" + audit["content"]["result"]
        elif audit["leak"]["result"] != "CLEAR" or audit["leak"]["scope"] != "entire_lesson":
            reason = "HINT_LEAK:" + audit["leak"]["result"]
        else:
            reason = None
        result[pid] = {"lesson": parsed.lesson if reason is None and parsed is not None else None, "skip": reason}
    return result


def _lesson_document(args, records: list[dict], audits: dict) -> dict:
    out = Path(args.output)
    return {"version": LESSONS_VERSION, "bindings": {
        "collection": file_sha(out / "collection.json"), "t1": file_sha(out / "t1.jsonl"),
        "note": file_sha(out / "note.jsonl"), "audit": file_sha(args.audit),
        "source": file_sha(args.source_provenance)}, "items": _lessons(records, audits)}


def audit_collection(args) -> dict:
    from fine_tuning.notebook_rl.vault import read_verifier

    records, sources, actors = records_and_sources(args)
    reviews = _items_file(args.reviews, "note-reviews-v1")
    if set(reviews) - {r["problem_id"] for r in records}:
        raise ValueError("reviews contain ids outside this collection")
    gold = read_verifier(Path(args.build_root) / "train")
    audits = {}
    for record in records:
        pid = record["problem_id"]
        source = sources[pid]
        audits[pid] = audit_note(record, gold=gold[pid].gold, choices=actors[pid].choices,
                                 provenance=source, review=reviews.get(pid),
                                 aliases=source.get("leak_aliases", []),
                                 acronym_candidates=source.get("acronym_candidates", []))
    args.audit = Path(args.audit) if args.audit else Path(args.output) / "audit.json"
    _write_once(args.audit, {"version": FILE_VERSION, "items": audits})
    value = _lesson_document(args, records, audits)
    _write_once(Path(args.output) / "lessons-v3.json", value)
    return {"version": "note-audit-summary-v1", "items": len(audits), "audit_sha256": file_sha(args.audit),
            "content": dict(Counter(a["content"]["result"] for a in audits.values())),
            "leak": dict(Counter(a["leak"]["result"] for a in audits.values())),
            "usable_lessons": sum(v["skip"] is None for v in value["items"].values())}


def verify_lessons(args) -> dict:
    """Trusted private process: recheck all bindings and return only gold-free lesson items."""
    records, sources, _ = records_and_sources(args)
    args.audit = Path(args.audit) if args.audit else Path(args.output) / "audit.json"
    audits = read_audits(args.audit, records, sources)
    expected = _lesson_document(args, records, audits)
    actual = json.loads((Path(args.output) / "lessons-v3.json").read_text())
    if actual != expected:
        raise ValueError("v3 lessons are stale, altered, or not approved by the bound audit")
    return actual["items"]


def verified_lessons_for_actor(args) -> dict:
    """Keep private note/source/audit bytes out of the model process, including during validation."""
    command = [sys.executable, "-m", "fine_tuning.error_note.collect", "verify-lessons",
               "--build-root", str(args.build_root), "--order", str(args.order), "--limit", str(args.limit),
               "--output", str(args.output), "--answer-contract", ANSWER_CONTRACT,
               "--training-rules", str(args.training_rules),
               "--training-rules-sha256", args.training_rules_sha256,
               "--source-provenance", str(args.source_provenance),
               "--audit", str(args.audit or Path(args.output) / "audit.json")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        # Private traceback may include source text: keep it out of the actor's public stream.
        raise ValueError("v3 private lesson verification failed")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("v3 lesson verifier returned invalid data")
    for entry in value.values():
        if not isinstance(entry, dict) or set(entry) != {"lesson", "skip"}:
            raise ValueError("v3 lesson verifier returned private or unexpected fields")
    return value


def collect_rows_v3(args, tokenize) -> tuple[list[dict], list[dict], dict]:
    from .collect import length_report, read_jsonl_done, summarize
    from .contract import build_rows_v3

    records, sources, _ = records_and_sources(args)
    args.audit = Path(args.audit) if args.audit else Path(args.output) / "audit.json"
    audits = read_audits(args.audit, records, sources)
    lessons = verify_lessons(args)
    resolves = read_jsonl_done(Path(args.output) / "resolve.jsonl", repair=False)
    rows, drops = [], []
    for record in records:
        pid = record["problem_id"]
        record.update(audit=audits[pid], source_provenance=sources[pid])
        if pid in resolves:
            third = resolves[pid]
            if "skipped" in third:
                record["resolve"] = {"skipped": third["skipped"]}
            else:
                lesson = lessons[pid]["lesson"]
                record["resolve"] = {"text": third.get("text", ""),
                                     "finish_reason": third.get("finish_reason") or "",
                                     "status": third["status"], "prompt": third["prompt"]}
                record["resolve"]["lesson_sha256"] = text_sha256(lesson) if lesson else None
        if record["t1"]["status"] != "CORRECT" and pid not in resolves:
            raise ValueError(f"v3 missing terminal resolve row for {pid}")
        built, dropped = build_rows_v3(record, tokenize)
        rows.extend(built)
        drops.extend(dropped)
    summary = summarize(records, rows, drops, parse_note_v2)
    summary.update(rows_version="v3", audit_sha256=file_sha(args.audit),
                   rows_by_source=dict(Counter(r["source"] for r in rows)),
                   note_content=dict(Counter(a["content"]["result"] for a in audits.values())),
                   lesson_leak=dict(Counter(a["leak"]["result"] for a in audits.values())))
    ids = {r["problem_id"] for r in records}
    outputs = {stage: {pid: row for pid, row in read_jsonl_done(
        Path(args.output) / f"{stage}.jsonl", repair=False).items() if pid in ids}
        for stage in ("t1", "note", "resolve")}
    summary["length_finishes"] = {stage: length_report(values) for stage, values in outputs.items()}
    attempted = [r for r in outputs["resolve"].values() if "text" in r]
    correct = sum(r["status"] == "CORRECT" for r in attempted)
    summary["resolve"] = {"candidates": sum(r["t1"]["status"] != "CORRECT" for r in records),
                          "attempted": len(attempted), "correct": correct,
                          "success_rate": correct / len(attempted) if attempted else None,
                          "skipped_by_reason": dict(Counter(r["skipped"] for r in resolves.values() if "skipped" in r)),
                          "context_blocked": sum(r.get("status") == "CONTEXT_BLOCKED" for r in resolves.values())}
    return rows, drops, summary
