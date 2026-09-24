"""Durable, resumable per-sample evaluation output."""

from __future__ import annotations

import csv
import fcntl
import html
import io
import json
import math
import os
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TextIO

from mmdl.runtime.artifacts import atomic_text, digest, read_json, resolve_image, sha256_file, write_json
from mmdl.runtime.contracts import SUBJECTS


def _safe_id(sample_id: object) -> str:
    if not isinstance(sample_id, str) or not sample_id or any(char in sample_id for char in "/\\\x00"):
        raise ValueError("Sample ID is unsafe")
    return sample_id


def _record_path(directory: Path, sample_id: str) -> Path:
    # Sample IDs are only used after validation and kept readable for local recovery.
    return directory / f"{_safe_id(sample_id)}.json"


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Create one JSON object once: link gives atomic no-overwrite publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Record already exists: {path.name}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_record(directory: Path, record: dict[str, Any]) -> None:
    path = _record_path(directory, record["id"])
    payload = {key: value for key, value in record.items() if key != "_record_sha256"}
    _write_json_exclusive(path, payload | {"_record_sha256": digest(payload)})


def _load_records(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.json")):
        record = read_json(path)
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record: {path.name}")
        sample_id = _safe_id(record.get("id"))
        if path.name != f"{sample_id}.json":
            raise ValueError(f"Record filename does not match sample ID: {path.name}")
        if "_record_sha256" in record:
            valid_hash = record["_record_sha256"] == digest(
                {key: value for key, value in record.items() if key != "_record_sha256"})
        else:
            # Preserve read access to the first smoke's already-running writer format.
            sidecar = path.with_suffix(".json.sha256")
            valid_hash = sidecar.exists() and read_json(sidecar) == {"sha256": sha256_file(path)}
        if not valid_hash:
            raise ValueError(f"Record hash verification failed: {path.name}")
        if sample_id in records:
            raise ValueError(f"Duplicate persisted sample ID: {sample_id}")
        records[sample_id] = record
    return records


class RunWriter:
    """Own one run directory for its lifetime and append immutable sample records."""

    def __init__(self, run_dir: Path, identity: dict[str, Any], expected_ids: list[str], resume: bool = False):
        self.run_dir = Path(run_dir)
        self.identity = identity
        self.expected_ids = list(expected_ids)
        if len(self.expected_ids) != len(set(self.expected_ids)):
            raise ValueError("Expected IDs contain duplicates")
        for sample_id in self.expected_ids:
            _safe_id(sample_id)
        if resume:
            if not self.run_dir.is_dir():
                raise FileNotFoundError(f"Cannot resume absent run: {self.run_dir}")
        else:
            try:
                self.run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise FileExistsError(f"Refusing existing run directory: {self.run_dir}") from exc
        self._lock_stream: TextIO | None = (self.run_dir / ".lock").open("a+")
        assert self._lock_stream is not None
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_stream.close()
            raise RuntimeError(f"Run already has an active writer: {self.run_dir}")
        manifest = self.run_dir / "run_manifest.json"
        expected_manifest = {"identity": identity, "expected_ids": self.expected_ids}
        if resume:
            if not manifest.exists() or read_json(manifest) != expected_manifest:
                self.close()
                raise ValueError("Resume identity, environment, model, protocol, or IDs differ")
        else:
            write_json(manifest, expected_manifest)
        try:
            self.completed = _load_records(self.run_dir / "samples")
            for record in self.completed.values():
                _validate_record(record)
        except Exception:
            self.close()
            raise
        unexpected = set(self.completed) - set(self.expected_ids)
        if unexpected:
            self.close()
            raise ValueError(f"Persisted records are outside expected IDs: {sorted(unexpected)}")

    def save(self, record: dict[str, Any]) -> None:
        self._ensure_open()
        if not isinstance(record, dict):
            raise TypeError("Record must be a dictionary")
        sample_id = _safe_id(record.get("id"))
        if sample_id not in self.expected_ids:
            raise ValueError(f"Unexpected sample ID: {sample_id}")
        if sample_id in self.completed:
            raise FileExistsError(f"Record already completed: {sample_id}")
        _validate_record(record)
        _write_record(self.run_dir / "samples", record)
        self.completed[sample_id] = record

    def failure(self, sample_id: str, exc: BaseException) -> None:
        self._ensure_open()
        sample_id = _safe_id(sample_id)
        if sample_id not in self.expected_ids:
            raise ValueError(f"Unexpected sample ID: {sample_id}")
        record = {
            "id": sample_id,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        _write_record(self.run_dir / "external_failures", record)

    def close(self) -> None:
        if getattr(self, "_lock_stream", None) is not None:
            stream = self._lock_stream
            assert stream is not None
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
            self._lock_stream = None

    def _ensure_open(self) -> None:
        if self._lock_stream is None:
            raise RuntimeError("Writer is closed")

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _ordered_records(
    run_dir: Path, expected_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    records = _load_records(run_dir / "samples")
    failures = _load_records(run_dir / "external_failures")
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("Expected IDs contain duplicates")
    missing = set(expected) - set(records)
    unexpected = set(records) - set(expected)
    if unexpected:
        raise ValueError(f"Unexpected persisted sample IDs: {sorted(unexpected)}")
    ordered = [records[sample_id] for sample_id in expected if sample_id in records]
    return ordered, [failures[sample_id] for sample_id in sorted(failures)], missing


def _validate_record(record: dict[str, Any]) -> None:
    fields = ("id", "subject", "correct", "parse_status", "generated_tokens", "generation_seconds", "finish_reason")
    absent = [field for field in fields if field not in record]
    if absent:
        raise ValueError(f"Record {record.get('id')} lacks required fields: {absent}")
    if not isinstance(record["correct"], bool):
        raise ValueError(f"Record {record['id']} has non-boolean correctness")


def _subject_from_id(sample_id: str) -> str | None:
    for subject in sorted(SUBJECTS, key=len, reverse=True):
        if sample_id.startswith(f"validation_{subject}_"):
            return subject
    return None


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n" for record in records)


def finalize(run_dir: Path, expected_ids: list[str], mode: str) -> dict[str, Any]:
    """Materialize score artifacts without treating missing/system-failed items as a smaller run."""
    run_dir = Path(run_dir)
    records, failures, missing = _ordered_records(run_dir, expected_ids)
    for record in records:
        _validate_record(record)
    expected = list(expected_ids)
    full = mode.lower() == "full"
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[record["subject"]].append(record)
    correct = sum(record["correct"] for record in records)
    completed_ids = {record["id"] for record in records}
    unresolved_failures = [failure for failure in failures if failure["id"] not in completed_ids]
    incomplete = bool(missing or unresolved_failures)
    status = "INCOMPLETE" if incomplete else ("COMPLETE" if full else mode.upper())
    denominator = 900 if full else len(expected)
    if full and len(expected) != 900:
        raise ValueError("Full mode requires exactly 900 expected IDs")
    selected_totals: Counter[str] = Counter()
    for sample_id in expected:
        subject = _subject_from_id(sample_id)
        if subject is not None:
            selected_totals[subject] += 1
    for subject, items in by_subject.items():
        if subject not in selected_totals:
            selected_totals[subject] = len(items)
    if full and not incomplete:
        if set(selected_totals) != set(SUBJECTS) or any(selected_totals[subject] != 30 for subject in SUBJECTS):
            raise ValueError("Full mode requires 30 actual MMMU subjects with 30 samples each")
        for subject in SUBJECTS:
            items = by_subject.get(subject, [])
            if len(items) != 30 or any(_subject_from_id(item["id"]) != subject for item in items):
                raise ValueError(f"Full mode ID/subject mapping mismatch: {subject}")
    subject_scores = []
    for subject in sorted(selected_totals):
        items = by_subject.get(subject, [])
        total = selected_totals[subject]
        subject_scores.append({"subject": subject, "correct": sum(item["correct"] for item in items), "total": total,
                               "accuracy": sum(item["correct"] for item in items) / total,
                               "generation_seconds": math.fsum(item["generation_seconds"] for item in items),
                               "sample_seconds": math.fsum(item.get("total_sample_seconds", item["generation_seconds"]) for item in items)})
    macro = None
    if full and not incomplete:
        macro = sum(item["accuracy"] for item in subject_scores) / 30
        if not math.isclose(macro, correct / 900, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("Full-run macro average differs from correct/900")
    summary = {
        "mode": mode,
        "status": status,
        "expected_count": len(expected),
        "completed_count": len(records),
        "missing_count": len(missing),
        "missing_ids": sorted(missing),
        "external_failure_count": len(failures),
        "unresolved_failure_count": len(unresolved_failures),
        "correct": correct,
        "denominator": denominator,
        "accuracy": correct / denominator if denominator else None,
        "macro_average": macro,
        "subject_scores": subject_scores,
        "subject_timing_policy": "sum of per-sample generation and preprocess/generate/score timers; excludes persistence and model loading",
    }
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        existing = read_json(summary_path)
        if {key: existing.get(key) for key in summary} == summary:
            return existing
        if existing.get("status") == "COMPLETE":
            raise ValueError("Refusing to replace a completed final summary")
        summary = {key: value for key, value in existing.items() if key not in summary} | summary
    atomic_text(run_dir / "predictions.jsonl", _jsonl(records))
    atomic_text(run_dir / "failures.jsonl", _jsonl(failures))
    columns = ["subject", "correct", "total", "accuracy", "generation_seconds", "sample_seconds"]
    rows = [columns]
    rows.extend([[item[key] for key in columns] for item in subject_scores])
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)
    atomic_text(run_dir / "subject_scores.csv", stream.getvalue())
    write_json(summary_path, summary)
    return summary


def render(run_dir: Path, *, include_images: bool = True) -> Path:
    """Produce a local, escaped inspection page without embedding benchmark images."""
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    records, failures, _ = _ordered_records(run_dir, manifest["expected_ids"])

    def escaped(value: Any) -> str:
        return html.escape("null" if value is None else str(value))

    parts = [
        "<!doctype html><meta charset=utf-8><title>MMDL run review</title>",
        "<style>body{font-family:sans-serif;max-width:1000px;margin:auto}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "dl{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1rem}dd{margin:0}"
        "img{max-width:100%;height:auto}</style>",
        "<body><h1>MMDL run review</h1>",
    ]
    if not include_images:
        parts.append("<p>Text-only rescoring review; original images remain with the source run.</p>")
    for record in records + failures:
        sample_id = escaped(record["id"])
        parts.append(f"<section><h2>{sample_id}</h2>")
        if "exception_type" in record:
            failure_href = html.escape(f"external_failures/{record['id']}.json", quote=True)
            parts.append(f'<p><a href="{failure_href}">Full failure record</a></p>')
            parts.append(f"<h3>Failure</h3><pre>{escaped(record.get('traceback'))}</pre>")
        else:
            parts.append("<dl>")
            for label, key in (
                ("ID", "id"), ("Type", "question_type"), ("Subject", "subject"),
                ("Parsed answer", "parsed_answer"), ("Gold answer", "answer"),
                ("Correct", "correct"), ("Parse status", "parse_status"),
                ("Status", "status"), ("Finish reason", "finish_reason"),
                ("Generated tokens", "generated_tokens"),
            ):
                parts.append(f"<dt>{label}</dt><dd>{escaped(record.get(key))}</dd>")
            parts.append("</dl><h3>Question</h3>")
            parts.append(f"<pre>{escaped(record.get('question'))}</pre>")
            options = record.get("options", [])
            if isinstance(options, list) and options:
                parts.append('<h3>Choices</h3><ol type="A">')
                parts.extend(f"<li>{escaped(option)}</li>" for option in options)
                parts.append("</ol>")
        for image in record.get("images", []) if include_images else []:
            resolve_image(run_dir, image)
            href = html.escape(image, quote=True)
            parts.append(f'<p><a href="{href}"><img src="{href}" loading="lazy" alt="{sample_id} image"></a></p>')
        if "exception_type" not in record:
            parts.append(f"<h3>Raw response</h3><pre>{escaped(record.get('raw_response'))}</pre>")
            parts.append(f"<h3>Model explanation</h3><pre>{escaped(record.get('model_explanation'))}</pre>")
            parts.append(f"<h3>Gold explanation</h3><pre>{escaped(record.get('gold_explanation'))}</pre>")
        metadata = html.escape(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), quote=False)
        parts.append(f"<details><summary>Complete machine metadata</summary><pre>{metadata}</pre></details></section>")
    parts.append("</body>")
    target = run_dir / "review.html"
    atomic_text(target, "\n".join(parts))
    return target
