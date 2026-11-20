"""Collect stage of the error-note path: one gold-blind solve per item, one note per item with the gold visible, rows.

Plan: claudedocs/research_self_note_finetuning_plan_20261101.md section 3; version 2: claudedocs/plan_self_note_v2_20261102.md.
CLI: python -m fine_tuning.error_note.collect {solve,note,sweep,resolve,rows} ... (see _parser). Heavy imports stay inside functions.
Version 2: `note --note-version v2` writes the note without a corrected solution and, from note.jsonl, the gold-free
lessons.json; `resolve` re-solves each item whose T1 is not correct once, gold-blind like `solve`, with only the
lesson in front of the P0 text; `rows --rows-version v2` builds rows from T1 or that re-solve plus the note.
Generation is a continuous add_request/step loop with the evaluation backend's prompt shape and sampling arguments;
every completion is appended to its JSONL immediately, so a killed run resumes by skipping the ids already written.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from fine_tuning.notebook_rl.answers import TRAINING_ANSWER_CONTRACT
from fine_tuning.notebook_rl.prompts import p0_eval_text, user_message

from .contract import (CONTEXT_LIMIT, IMAGE, MAX_NEW_TOKENS, NOTE_INSTRUCTION_V3, TURN1_ANSWER_CONTRACT,
                       build_rows, build_rows_v2, hint_leak, is_loop, note_prompt, note_prompt_v2, note_prompt_v3,
                       parse_note, parse_note_v2, resolve_prompt, sample_seed)
from .order import read_order
from .v4 import BlockRepetition

LOOP_STOP_FINISH = "loop_stop"  # finish_reason of a version-4 training rollout stopped by v4.BlockRepetition
WAIT = object()  # a request iterator's "nothing to add until the next engine step" (version-4 adaptive rollouts)

def loop_stop_info(input_tokens: int) -> dict:
    """Per-request state of a version-4 loop-stopped vLLM request (RequestOutputKind.DELTA outputs)."""
    return {"input_tokens": input_tokens, "ids": [], "detector": BlockRepetition()}


def observe_delta(output: Any, info: dict) -> bool:
    """An unfinished DELTA output carries only its new tokens (vLLM 0.11 output_processor): collect them and report
    whether the generation now only repeats one block."""
    piece = list(output.outputs[0].token_ids)
    info["ids"].extend(piece)
    return info["detector"].feed(piece)


def delta_ids(output: Any, info: dict) -> list[int]:
    """All completion ids of a finished DELTA request: the collected pieces plus the final piece."""
    return info["ids"] + list(output.outputs[0].token_ids)


PREFETCH_WORKERS = 8
THROUGHPUT_EVERY = 60.0
SKIPPED_T1_STATUSES = ("UNVERIFIABLE", "CONTEXT_BLOCKED")
MIB = 1024 * 1024
# Auto sizing (--kv-cache-gib auto, --seqs auto) from measured Qwen3-VL-4B BF16 vLLM runs; keep these numbers in sync with them.
WEIGHTS_GIB = 8.68            # model weights
OTHER_VLLM_GIB = 5.0          # other vLLM memory at 32-48 concurrency (A40 peak 41.66 GiB with KV 28 GiB)
KV_TOKENS_PER_GIB = 7280      # 8 GiB of KV cache gave 58,240 tokens
MEAN_CONTEXT_TOKENS = 4800    # time-weighted mean context of a running sequence
AUTO_RESERVE_GIB = 16         # kept outside the KV cache: WEIGHTS_GIB + OTHER_VLLM_GIB plus about 2.3 GiB of margin
MIN_AUTO_KV_GIB = 8
SEQS_STEP = 8
SEQS_FACTORS = (0.75, 1.0, 1.25)
LESSONS_VERSION = "error-note-lessons-v1"


class InputTooLong(ValueError):
    """The prepared prompt plus the output cap exceeds the context limit; the item is recorded, never truncated."""


@dataclass
class StreamStats:
    steps: int = 0
    step_seconds: float = 0.0
    active_sum: int = 0
    startup_seconds: float = 0.0
    stopped_early: bool = False
    metrics: Callable[[], dict | None] | None = None  # set by the engine builder: {"generation_tokens", "preemptions"}
    on_step: Callable[[], None] | None = None
    receipt_extra: dict = field(default_factory=dict)  # merged into the final receipt (resolved KV size and its source)


Stream = Callable[[Iterable[dict], StreamStats, float | None], Iterator[tuple[dict, dict]]]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ files and small utilities

def append_jsonl(path: Path, row: dict) -> None:
    data = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    with open(path, "ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl_done(path: Path, *, repair: bool = True) -> dict[str, dict]:
    """{problem_id: row}. A final line cut by a kill is dropped with a warning (and cut from the file when repair)."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = path.read_bytes()
    *lines, tail = raw.split(b"\n")  # tail is b"" when the file ends with a newline
    rows = [json.loads(line) for line in lines if line.strip()]
    if tail.strip():
        try:
            rows.append(json.loads(tail))
            if repair:
                with open(path, "ab") as stream:
                    stream.write(b"\n")
        except ValueError:
            warnings.warn(f"{path}: dropped a truncated final line of {len(tail)} bytes", RuntimeWarning)
            if repair:
                with open(path, "r+b") as stream:
                    stream.truncate(len(raw) - len(tail))
    done: dict[str, dict] = {}
    for row in rows:
        if row["problem_id"] in done:
            raise ValueError(f"{path}: duplicate problem_id {row['problem_id']}")
        done[row["problem_id"]] = row
    return done


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def bounded_map(fn: Callable, iterable: Iterable, *, workers: int, buffer: int) -> Iterator:
    """fn over iterable on `workers` threads, results in input order, at most `buffer` futures outstanding."""
    if workers < 1 or buffer < 1:
        raise ValueError("workers and buffer must be at least 1")
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        pending: collections.deque = collections.deque()
        try:
            for item in iterable:
                pending.append(pool.submit(fn, item))
                if len(pending) >= buffer:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


def _gpu_sample() -> tuple[int | None, int | None, int | None]:
    """(utilization percent, memory used MiB, memory total MiB) of the first GPU; nulls when nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5)
        util, used, total = (int(v) for v in result.stdout.splitlines()[0].split(","))
        return util, used, total
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None, None


def _cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat") as stream:
            values = [int(v) for v in stream.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    return sum(values), values[3] + (values[4] if len(values) > 4 else 0)  # total, idle + iowait


def _mem_available_kib() -> int | None:
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


class ResourceLog:
    """Background thread: one JSON line per interval (GPU, CPU, load, free RAM), plus one at start and one at exit."""

    def __init__(self, path: Path, every: float = 60.0):
        self.path, self.every = Path(path), every
        self._prev: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        cpu = _cpu_times()
        percent = None
        if cpu and self._prev and cpu[0] > self._prev[0]:
            percent = round(100 * (1 - (cpu[1] - self._prev[1]) / (cpu[0] - self._prev[0])), 1)
        self._prev = cpu
        util, used, total = _gpu_sample()
        try:
            load = list(os.getloadavg())
        except OSError:
            load = None
        append_jsonl(self.path, {"utc": _utc(), "gpu_util_percent": util, "gpu_mem_used_mib": used,
                                 "gpu_mem_total_mib": total, "cpu_percent": percent, "loadavg": load,
                                 "mem_available_kib": _mem_available_kib()})

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.every):
            self._sample()

    def __enter__(self) -> "ResourceLog":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


# ------------------------------------------------------------------ guards (build-root mode of passrate._install_guards)

def install_training_guards(build_root: Path, *, vllm_schema: dict | None = None,
                            private_files: Iterable[Path] = ()) -> list[Path]:
    """Lock checks, optional vLLM schema step, HF imports, then the file guards. Returns the forbidden benchmark roots.

    vllm_schema: a dict that receives "schema_dependency_guard" (pass {} when vLLM will be imported); None for no vLLM.
    """
    from importlib import import_module

    from fine_tuning.notebook_rl.dataset import SPLITS
    from fine_tuning.notebook_rl.guards import install_guards, prepare_vllm_schema_dependency
    from fine_tuning.notebook_rl.quantized_policy import _versions
    from fine_tuning.notebook_rl.shared_smoke import _prepare_hf_dependencies

    _versions()
    if vllm_schema is not None:
        vllm_schema["schema_dependency_guard"] = prepare_vllm_schema_dependency()
        # Imported before the file guards: a first import after them dies on any hard-linked stdlib or package file.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        for module in ("vllm", "vllm.sampling_params", "logging.config"):
            import_module(module)
    _prepare_hf_dependencies()
    import_module("bitsandbytes")
    hubs = {Path.home() / "mmdl-cache/huggingface/hub",
            Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")}
    pod_eval = Path("/workspace/mmdl-volume/eval-data")  # the pod volume's MMMU predictions/proof files
    forbidden = [Path.home() / "mmdl-data/evaluation",
                 *(hub / name for hub in sorted(hubs) for name in ("datasets--MMMU--MMMU", "datasets--MMMU--MMMU_Pro")),
                 *dict.fromkeys((pod_eval, Path(os.path.realpath(pod_eval))))]
    vaults = [Path(build_root) / split / name for split in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
    vaults.extend(Path(path) for path in private_files)
    install_guards(forbidden_roots=forbidden, vault_files=vaults, allow_vault=False, allow_benchmark=False)
    return forbidden


# ------------------------------------------------------------------ continuous generation

def continuous_stream(engine: Any, requests: Iterable[dict], limit: int, prepare: Callable, finish: Callable,
                      stats: StreamStats, *, stop_after: float | None = None,
                      clock: Callable[[], float] = time.monotonic, observe: Callable | None = None,
                      partial: Callable | None = None) -> Iterator[tuple[dict, dict]]:
    """Keep `limit` requests active on `engine` (add_request/abort_request/step), yield (request, output dict) per finish.

    prepare(request) -> (prompt, params, info), raising InputTooLong to record the item without generating.
    finish(engine_output, request, info) -> output dict. Request dicts need a unique "request_id".
    After `stop_after` seconds the active requests are aborted and the stream ends with stats.stopped_early.
    Optional (version 4 training rollouts only): observe(engine_output, info) -> True aborts that unfinished request,
    which is then yielded as partial(engine_output, request, info). Without observe every output is unchanged.
    The request iterator may yield WAIT while requests are active: filling resumes after the next engine step.
    """
    if (observe is None) != (partial is None):
        raise ValueError("observe and partial go together")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    stop_at = None if stop_after is None else clock() + stop_after
    iterator = iter(requests)
    active: dict[str, dict] = {}
    blocked: collections.deque[tuple[dict, dict]] = collections.deque()
    seen: set[str] = set()
    stopped: set[str] = set()  # aborted by observe: a late output of theirs is ignored, never misread as unknown
    exhausted = False

    def fill() -> None:
        nonlocal exhausted
        while not exhausted and len(active) < limit:
            request = next(iterator, None)
            if request is None:
                exhausted = True
                return
            if request is WAIT:
                if not active and not blocked:
                    raise RuntimeError("request iterator waits with nothing active")
                return
            request_id = request["request_id"]
            if request_id in seen:
                raise ValueError(f"duplicate request_id {request_id}")
            seen.add(request_id)
            try:
                prompt, params, info = prepare(request)
            except InputTooLong as exc:
                blocked.append((request, {"blocked": True, "error": str(exc)}))
                continue
            engine.add_request(request_id, prompt, params)
            active[request_id] = {"request": request, "info": info, "seconds": 0.0}

    try:
        fill()
        while True:
            while blocked:
                yield blocked.popleft()
            if not active:
                if exhausted:
                    break
                fill()  # only after WAIT: the iterator may have waited on the rows just yielded
                if not active and not blocked:
                    break
                continue
            if stop_at is not None and clock() >= stop_at:
                stats.stopped_early = True
                break
            count, began = len(active), clock()
            outputs = engine.step()
            elapsed = clock() - began
            stats.steps += 1
            stats.step_seconds += elapsed
            stats.active_sum += count
            for running_state in active.values():
                running_state["seconds"] += elapsed / count
            for output in outputs:
                state = active.get(output.request_id)
                if state is None and output.request_id in stopped:
                    continue
                if state is None:
                    raise RuntimeError(f"engine returned an unknown request id: {output.request_id}")
                if output.finished:
                    del active[output.request_id]
                    row = finish(output, state["request"], state["info"])
                    row["seconds"] = state["seconds"]
                    yield state["request"], row
                elif observe is not None and partial is not None and observe(output, state["info"]):
                    engine.abort_request([output.request_id])
                    del active[output.request_id]
                    stopped.add(output.request_id)
                    row = partial(output, state["request"], state["info"])
                    row["seconds"] = state["seconds"]
                    yield state["request"], row
            if stats.on_step is not None:
                stats.on_step()
            fill()
    finally:
        if active:
            handling_error = sys.exc_info()[0] is not None
            try:
                engine.abort_request(list(active))
            except Exception as exc:
                if not handling_error:
                    raise
                warnings.warn(f"Failed to abort outstanding requests: {exc}", RuntimeWarning)


def run_generation(stage: str, out: Path, requests: Iterable[dict], stream: Stream, stats: StreamStats,
                   handle: Callable[[dict, dict], dict], *, total: int, S: int, B: int, engine: str,
                   stop_after: float | None = None, every: float = THROUGHPUT_EVERY,
                   gpu_probe: Callable | None = None) -> dict:
    """Append handle(request, output) rows to <stage>.jsonl as they finish; throughput.jsonl every `every` seconds;
    receipt-<stage>.json at exit (also after an error or a stop deadline)."""
    out = Path(out)
    counts = {"done": 0, "tokens": 0, "length": 0}
    peak: list[int | None] = [None]
    last = [time.monotonic()]

    def snapshot() -> dict:
        probe = gpu_probe or _gpu_sample
        used = probe()[1]
        if used is not None:
            peak[0] = used if peak[0] is None else max(peak[0], used)
        try:
            metrics = stats.metrics() if stats.metrics else None
        except Exception:
            metrics = None
        generated = metrics["generation_tokens"] if metrics else stats.active_sum
        seconds = stats.step_seconds
        return {"utc": _utc(), "stage": stage, "engine": engine, "S": S, "B": B, "items_total": total,
                "items_done": counts["done"], "finished_tokens": counts["tokens"], "generation_seconds": seconds,
                "startup_seconds": stats.startup_seconds,
                "tokens_per_second": generated / seconds if seconds > 0 else None,
                "finished_tokens_per_second": counts["tokens"] / seconds if seconds > 0 else None,
                "steps": stats.steps, "mean_step_ms": 1000 * seconds / stats.steps if stats.steps else None,
                "mean_active": stats.active_sum / stats.steps if stats.steps else None,
                "preemptions": metrics["preemptions"] if metrics else None,
                "generation_tokens": metrics["generation_tokens"] if metrics else None,
                "token_source": "vllm_metrics" if metrics else "active_times_steps",
                "peak_gpu_mem_mib": peak[0], "length_finishes": counts["length"], "stopped_early": stats.stopped_early}

    def tick(force: bool = False) -> None:
        now = time.monotonic()
        if force or now - last[0] >= every:
            last[0] = now
            append_jsonl(out / "throughput.jsonl", snapshot())

    stats.on_step = tick
    error = None
    try:
        iterator = iter(requests)
        first = next(iterator, None)
        if first is not None:  # nothing to generate means no engine is ever built
            output_stream: Any = stream(itertools.chain([first], iterator), stats, stop_after)
            with contextlib.closing(output_stream) as outputs:
                for request, output in outputs:
                    row = handle(request, output)
                    append_jsonl(out / f"{stage}.jsonl", row)
                    counts["done"] += 1
                    if not output.get("blocked"):
                        counts["tokens"] += output["completion_tokens"]
                        counts["length"] += output["finish_reason"] == "length"
                    tick()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        tick(force=True)
        receipt = snapshot() | stats.receipt_extra | {"completed": error is None and not stats.stopped_early, "error": error}
        write_json(out / f"receipt-{stage}.json", receipt)
    return receipt


# ------------------------------------------------------------------ solve and note orchestration

def _blocked_fields(output: dict) -> dict:
    return {"status": "CONTEXT_BLOCKED", "error": output["error"]}


def run_solve(items: list[dict], actors: dict, load: Callable, stream: Stream, stats: StreamStats, grade: Callable,
              out: Path, *, S: int, B: int, engine: str, stop_after: float | None = None,
              every: float = THROUGHPUT_EVERY, gpu_probe: Callable | None = None) -> dict:
    """Gold-blind turn 1. grade(problem_id, text, finish_reason, mode) -> (status, extracted); there is no gold here."""
    out = Path(out)
    done = read_jsonl_done(out / "t1.jsonl")
    todo = [item["problem_id"] for item in items if item["problem_id"] not in done]

    def requests() -> Iterator[dict]:
        for actor, images in bounded_map(lambda pid: (actors[pid], load(actors[pid])), todo,
                                         workers=PREFETCH_WORKERS, buffer=2 * S):
            yield {"request_id": actor.problem_id, "messages": [user_message(p0_eval_text(actor.question, actor.choices), len(images))],
                   "images": images, "seed": sample_seed(actor.problem_id, "t1"), "max_tokens": MAX_NEW_TOKENS,
                   "mode": actor.mode}

    def handle(request: dict, output: dict) -> dict:
        row = {"problem_id": request["request_id"], "stage": "t1", "seed": request["seed"]}
        if output.get("blocked"):
            return row | _blocked_fields(output) | {"prompt_tokens": None, "completion_tokens": 0, "extracted": None,
                                                    "seconds": 0.0, "engine": engine}
        status, extracted = grade(request["request_id"], output["text"], output["finish_reason"], request["mode"])
        return row | {"prompt_tokens": output["prompt_tokens"], "completion_tokens": output["completion_tokens"],
                      "finish_reason": output["finish_reason"], "text": output["text"], "status": status,
                      "extracted": extracted, "seconds": output["seconds"], "engine": engine}

    return run_generation("t1", out, requests(), stream, stats, handle, total=len(todo), S=S, B=B, engine=engine,
                          stop_after=stop_after, every=every, gpu_probe=gpu_probe)


def run_note(items: list[dict], actors: dict, t1: dict[str, dict], gold: dict[str, str], load: Callable, stream: Stream,
             stats: StreamStats, grade: Callable, out: Path, *, S: int, B: int, engine: str,
             stop_after: float | None = None, every: float = THROUGHPUT_EVERY, gpu_probe: Callable | None = None,
             version: str = "v1", answer_contract: str = TURN1_ANSWER_CONTRACT) -> dict:
    """Turn 2: the note prompt shows the gold. Items without a usable T1 get a SKIPPED row and no generation.

    version "v2": the note has no corrected solution; its lesson is checked for hints (contract.hint_leak) here, where
    the gold is known, and lessons.json is rewritten from note.jsonl at the end, also after a stop deadline."""
    if version not in ("v1", "v2"):
        raise ValueError(f"unknown note version {version!r}")
    v3 = answer_contract == TRAINING_ANSWER_CONTRACT
    if v3 and version != "v2":
        raise ValueError(f"{TRAINING_ANSWER_CONTRACT} requires the v2 note schema")
    instruction = {"note_instruction_sha256": hashlib.sha256(NOTE_INSTRUCTION_V3.encode()).hexdigest()} if v3 else {}
    out = Path(out)
    ids = [item["problem_id"] for item in items]
    missing = [pid for pid in ids if pid not in t1]
    if missing:
        raise ValueError(f"no T1 row for {len(missing)} items, first {missing[:3]}")
    done = read_jsonl_done(out / "note.jsonl")
    if any(("note_version" in row) != (version == "v2") for row in done.values() if "status" not in row):
        raise ValueError(f"note.jsonl holds notes of another version than {version}")
    if any((row.get("note_instruction_sha256") != instruction["note_instruction_sha256"] if v3
            else "note_instruction_sha256" in row) for row in done.values()):
        raise ValueError("note.jsonl instruction differs; use a new output directory")
    if v3:
        for pid, row in done.items():
            if pid not in ids:
                continue
            if str(row.get("status", "")).startswith("SKIPPED_"):
                if t1[pid]["status"] not in SKIPPED_T1_STATUSES or row["status"] != "SKIPPED_" + t1[pid]["status"]:
                    raise ValueError("note.jsonl skipped status differs from its T1 input")
                continue
            if row.get("prompt") != note_prompt_v3(
                    p0_eval_text(actors[pid].question, actors[pid].choices), t1[pid].get("text", ""), t1[pid]["status"], gold[pid]):
                raise ValueError("note.jsonl prompt differs from the bound v3 instruction/input")
    for pid in ids:
        if pid not in done and t1[pid]["status"] in SKIPPED_T1_STATUSES:
            done[pid] = {"problem_id": pid, "stage": "note", "status": "SKIPPED_" + t1[pid]["status"], **instruction}
            append_jsonl(out / "note.jsonl", done[pid])
    todo = [pid for pid in ids if pid not in done]

    def requests() -> Iterator[dict]:
        for actor, images in bounded_map(lambda pid: (actors[pid], load(actors[pid])), todo,
                                         workers=PREFETCH_WORKERS, buffer=2 * S):
            pid = actor.problem_id
            make = note_prompt_v3 if v3 else note_prompt_v2 if version == "v2" else note_prompt
            prompt = make(p0_eval_text(actor.question, actor.choices), t1[pid]["text"], t1[pid]["status"], gold[pid])
            yield {"request_id": pid, "messages": [user_message(prompt, len(images))], "images": images,
                   "seed": sample_seed(pid, "note"), "max_tokens": MAX_NEW_TOKENS, "prompt": prompt, "mode": actor.mode,
                   "choices": actor.choices}

    def handle(request: dict, output: dict) -> dict:
        pid = request["request_id"]
        row = {"problem_id": pid, "stage": "note", "seed": request["seed"], **instruction}
        if version == "v2":
            row["note_version"] = "v2"
        if output.get("blocked"):
            return row | _blocked_fields(output) | {"prompt": request["prompt"], "engine": engine}
        if version == "v2":
            note = parse_note_v2(output["text"])
            return row | {"prompt": request["prompt"], "prompt_tokens": output["prompt_tokens"],
                          "completion_tokens": output["completion_tokens"], "finish_reason": output["finish_reason"],
                          "text": output["text"], "image_reading": note.image_reading if note else None,
                          "cause": note.cause if note else None, "lesson": note.lesson if note else None,
                          "hint_leak": hint_leak(note.lesson, gold[pid], request["choices"]) if note else None,
                          "seconds": output["seconds"], "engine": engine}
        parsed = parse_note(output["text"])
        corrected = grade(pid, parsed.corrected_text, output["finish_reason"], request["mode"]) if parsed else (None, None)
        return row | {"prompt": request["prompt"], "prompt_tokens": output["prompt_tokens"],
                      "completion_tokens": output["completion_tokens"], "finish_reason": output["finish_reason"],
                      "text": output["text"], "image_reading": parsed.image_reading if parsed else None,
                      "cause": parsed.cause if parsed else None, "corrected_status": corrected[0],
                      "corrected_extracted": corrected[1], "seconds": output["seconds"], "engine": engine}

    receipt = run_generation("note", out, requests(), stream, stats, handle, total=len(todo), S=S, B=B, engine=engine,
                             stop_after=stop_after, every=every, gpu_probe=gpu_probe)
    if version == "v2":
        write_lessons(out)
    return receipt


def _lesson_entry(row: dict) -> dict:
    """What the gold-blind resolve stage may see of one note row: a usable lesson, or the reason there is none."""
    if "status" in row:  # SKIPPED_* (no usable T1) or CONTEXT_BLOCKED
        return {"lesson": None, "skip": "NO_NOTE"}
    if row.get("note_version") != "v2":
        raise ValueError(f"{row['problem_id']}: not a v2 note")
    if row["finish_reason"] != "stop":
        return {"lesson": None, "skip": "NOTE_NOT_STOPPED"}
    if row["lesson"] is None:
        return {"lesson": None, "skip": "NOTE_UNPARSED"}
    if row["hint_leak"] is not None:
        return {"lesson": None, "skip": f"HINT_LEAK:{row['hint_leak']}"}
    return {"lesson": row["lesson"], "skip": None}


def write_lessons(out: Path) -> dict:
    """lessons.json from note.jsonl. note.jsonl holds the gold (in each prompt); lessons.json holds only lessons that
    passed the hint check, so the resolve process never opens a file with the gold. note_bytes ties it to note.jsonl."""
    out = Path(out)
    notes = read_jsonl_done(out / "note.jsonl", repair=False)
    value = {"version": LESSONS_VERSION, "note_bytes": (out / "note.jsonl").stat().st_size,
             "items": {pid: _lesson_entry(row) for pid, row in notes.items()}}
    write_json(out / "lessons.json", value)
    return value


def read_lessons(out: Path) -> dict[str, dict]:
    """lessons.json, refused unless note.jsonl still has the size it was written from (size only; never its content)."""
    out = Path(out)
    value = json.loads((out / "lessons.json").read_text(encoding="utf-8"))
    if value.get("version") != LESSONS_VERSION:
        raise ValueError("lessons.json has an unexpected version")
    if value["note_bytes"] != (out / "note.jsonl").stat().st_size:
        raise ValueError("lessons.json is older than note.jsonl; rerun the v2 note stage to rewrite it")
    return value["items"]


def run_resolve(items: list[dict], actors: dict, t1: dict[str, dict], lessons: dict[str, dict], load: Callable,
                stream: Stream, stats: StreamStats, grade: Callable, out: Path, *, S: int, B: int, engine: str,
                stop_after: float | None = None, every: float = THROUGHPUT_EVERY, gpu_probe: Callable | None = None) -> dict:
    """One gold-blind re-solve per item whose T1 is not correct: the lesson block, then the P0 text; nothing else.

    No retry and no second sample: an item is generated once (resume skips every id already in resolve.jsonl).
    An item without a usable lesson gets a row {"skipped": reason} and no generation; a correct T1 gets no row."""
    out = Path(out)
    ids = [item["problem_id"] for item in items]
    missing = [pid for pid in ids if pid not in t1]
    if missing:
        raise ValueError(f"no T1 row for {len(missing)} items, first {missing[:3]}")
    missing = [pid for pid in ids if pid not in lessons]
    if missing:
        raise ValueError(f"no note row for {len(missing)} items, first {missing[:3]}")
    done = read_jsonl_done(out / "resolve.jsonl")
    candidates = [pid for pid in ids if t1[pid]["status"] != "CORRECT"]
    for pid in candidates:
        if pid not in done and lessons[pid]["skip"] is not None:
            done[pid] = {"problem_id": pid, "stage": "resolve", "skipped": lessons[pid]["skip"]}
            append_jsonl(out / "resolve.jsonl", done[pid])
    todo = [pid for pid in candidates if pid not in done]
    stats.receipt_extra |= {"candidates": len(candidates),
                            "skipped": dict(Counter(done[pid]["skipped"] for pid in candidates
                                                    if "skipped" in done.get(pid, {})))}

    def requests() -> Iterator[dict]:
        for actor, images in bounded_map(lambda pid: (actors[pid], load(actors[pid])), todo,
                                         workers=PREFETCH_WORKERS, buffer=2 * S):
            pid = actor.problem_id
            prompt = resolve_prompt(p0_eval_text(actor.question, actor.choices), lessons[pid]["lesson"])
            yield {"request_id": pid, "messages": [user_message(prompt, len(images))], "images": images,
                   "seed": sample_seed(pid, "resolve"), "max_tokens": MAX_NEW_TOKENS, "prompt": prompt, "mode": actor.mode}

    def handle(request: dict, output: dict) -> dict:
        row = {"problem_id": request["request_id"], "stage": "resolve", "seed": request["seed"], "prompt": request["prompt"]}
        if output.get("blocked"):
            return row | _blocked_fields(output) | {"prompt_tokens": None, "completion_tokens": 0, "extracted": None,
                                                    "seconds": 0.0, "engine": engine}
        status, extracted = grade(request["request_id"], output["text"], output["finish_reason"], request["mode"])
        return row | {"prompt_tokens": output["prompt_tokens"], "completion_tokens": output["completion_tokens"],
                      "finish_reason": output["finish_reason"], "text": output["text"], "status": status,
                      "extracted": extracted, "seconds": output["seconds"], "engine": engine}

    return run_generation("resolve", out, requests(), stream, stats, handle, total=len(todo), S=S, B=B, engine=engine,
                          stop_after=stop_after, every=every, gpu_probe=gpu_probe)


# ------------------------------------------------------------------ auto sizing from the GPU's memory

def gpu_total_gib() -> float:
    """Total memory of GPU 0 in GiB from nvidia-smi, so the calling process holds no CUDA context."""
    total_mib = _gpu_sample()[2]
    if total_mib is None:
        raise RuntimeError("--kv-cache-gib auto: nvidia-smi did not report the GPU memory")
    return total_mib / 1024


def auto_kv_gib(total_gib: float) -> int:
    kv = math.floor(total_gib - AUTO_RESERVE_GIB)
    if kv < MIN_AUTO_KV_GIB:
        raise ValueError(f"--kv-cache-gib auto: a {total_gib:.1f} GiB GPU leaves {kv} GiB for the KV cache, "
                         f"below the minimum of {MIN_AUTO_KV_GIB}")
    return kv


def auto_seqs(kv_gib: int) -> list[int]:
    """Sweep candidates around the concurrency the KV cache holds at the mean context: multiples of 8, sorted, distinct."""
    estimate = kv_gib * KV_TOKENS_PER_GIB / MEAN_CONTEXT_TOKENS
    return sorted({max(SEQS_STEP, round(factor * estimate) // SEQS_STEP * SEQS_STEP) for factor in SEQS_FACTORS})


def resolve_kv(kv_arg: float | str, total_gib: Callable[[], float] = gpu_total_gib) -> dict:
    """The GPU is read only for "auto"."""
    if kv_arg != "auto":
        return {"gpu_total_gib": None, "kv_cache_gib_resolved": kv_arg, "kv_source": "fixed"}
    total = total_gib()
    return {"gpu_total_gib": total, "kv_cache_gib_resolved": auto_kv_gib(total), "kv_source": "auto"}


# ------------------------------------------------------------------ engines (GPU; not exercised by the unit tests)

def vllm_metrics(llm: Any) -> dict | None:
    try:
        values = {m.name: m.value for m in llm.get_metrics() if hasattr(m, "value")}
        return {"generation_tokens": int(values["vllm:generation_tokens"]),
                "preemptions": int(values.get("vllm:num_preemptions", 0))}
    except Exception:
        return None


def _vllm_stream_builder(args: argparse.Namespace, stats: StreamStats) -> Stream:
    def stream(requests: Iterable[dict], stats_: StreamStats, stop_after: float | None) -> Iterator[tuple[dict, dict]]:
        kv = resolve_kv(args.kv_cache_gib)  # before the heavy imports: a GPU too small for "auto" fails fast
        stats_.receipt_extra |= kv
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from importlib.metadata import version

        from transformers import AutoConfig, AutoProcessor
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from fine_tuning.notebook_rl.engines import QWEN_OFFICIAL, vllm_kwargs
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

        if version("vllm") != "0.11.0":
            raise RuntimeError("collect: vllm==0.11.0 required")
        began = time.monotonic()
        model_path = str(args.model_path)
        loop_stop = bool(getattr(args, "loop_stop", False))
        kwargs = vllm_kwargs(Path(model_path), IMAGE, int(kv["kv_cache_gib_resolved"] * 2**30), args.max_num_seqs, CONTEXT_LIMIT,
                             gpu_memory_utilization=float(args.gpu_memory_utilization),
                             enable_prefix_caching=bool(getattr(args, "prefix_caching", False)))
        if kwargs["enable_prefix_caching"] or loop_stop:  # version-4 options only; earlier receipts are unchanged
            stats_.receipt_extra |= {"prefix_caching": kwargs["enable_prefix_caching"], "loop_stop": loop_stop}
        kwargs |= {"max_num_batched_tokens": args.max_num_batched_tokens, "disable_log_stats": False}
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, min_pixels=IMAGE["min_pixels"],
                                                  max_pixels=IMAGE["max_pixels"])
        processor.image_processor.size = {"shortest_edge": IMAGE["min_pixels"], "longest_edge": IMAGE["max_pixels"]}
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        llm = LLM(**kwargs)
        stats.startup_seconds = time.monotonic() - began
        stats.metrics = lambda: vllm_metrics(llm)

        def prepare(request: dict) -> tuple[dict, Any, Any]:
            item, images = prepare_protocol_inputs(processor, config, request["messages"], request["images"], IMAGE)
            if item["input_tokens"] + request["max_tokens"] > CONTEXT_LIMIT:
                raise InputTooLong(f"{item['input_tokens']} input + {request['max_tokens']} output tokens exceed {CONTEXT_LIMIT}")
            # The same arguments as the evaluation backend's _sampling_params; only the seed and the cap differ per call.
            params = SamplingParams(n=1, temperature=QWEN_OFFICIAL.temperature, top_p=QWEN_OFFICIAL.top_p,
                                    top_k=QWEN_OFFICIAL.top_k, repetition_penalty=QWEN_OFFICIAL.repetition_penalty,
                                    presence_penalty=QWEN_OFFICIAL.presence_penalty, max_tokens=request["max_tokens"],
                                    seed=request["seed"],
                                    output_kind=RequestOutputKind.DELTA if loop_stop else RequestOutputKind.FINAL_ONLY)
            info = loop_stop_info(item["input_tokens"]) if loop_stop else item["input_tokens"]
            return {"prompt": item["chat_prompt"], "multi_modal_data": {"image": images}}, params, info

        def finish(output: Any, request: dict, info: Any) -> dict:
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM did not return exactly one sampled completion")
            completion = output.outputs[0]
            if loop_stop:  # DELTA outputs: the earlier pieces were collected by observe
                ids = delta_ids(output, info)
                prompt_tokens = info["input_tokens"]
            else:
                ids = list(completion.token_ids)
                prompt_tokens = len(output.prompt_token_ids)
            return {"text": processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "finish_reason": completion.finish_reason or "unknown",
                    "prompt_tokens": prompt_tokens, "completion_tokens": len(ids)}

        def partial(output: Any, request: dict, info: dict) -> dict:
            return {"text": processor.tokenizer.decode(info["ids"], skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False),
                    "finish_reason": LOOP_STOP_FINISH, "prompt_tokens": info["input_tokens"],
                    "completion_tokens": len(info["ids"])}

        yield from continuous_stream(llm.llm_engine, requests, args.max_num_seqs, prepare, finish, stats_,
                                     stop_after=stop_after, observe=observe_delta if loop_stop else None,
                                     partial=partial if loop_stop else None)

    return stream


class GeneratedOnlyPresencePenalty:
    """src/mmdl/runtime/generation.py GeneratedOnlyPresencePenalty (copied: fine_tuning must work alone)."""

    def __init__(self, prompt_length: int, penalty: float):
        if prompt_length < 0 or penalty < 0:
            raise ValueError("Invalid prompt length or penalty")
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        generated = input_ids[:, self.prompt_length:]
        if generated.numel() == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


def _hf_stream_builder(args: argparse.Namespace, stats: StreamStats) -> Stream:
    """Local NF4 functional runs: one sequence at a time, like pod/gpu_smoke.py."""
    def stream(requests: Iterable[dict], stats_: StreamStats, stop_after: float | None) -> Iterator[tuple[dict, dict]]:
        import torch
        from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

        from fine_tuning.error_note.protocol import ARM, plain_protocol
        from fine_tuning.notebook_rl.engines import QWEN_OFFICIAL as SAMPLING
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
        from fine_tuning.notebook_rl.quantized_policy import build_quantized_policy

        began = time.monotonic()
        bundle = build_quantized_policy(Path(args.model_path), ARM, plain_protocol("nf4"))
        model, processor = bundle.model.eval(), bundle.processor
        stats_.startup_seconds = time.monotonic() - began
        base = model.generation_config
        eos_ids = set(base.eos_token_id) if isinstance(base.eos_token_id, list) else {base.eos_token_id}
        stop_at = None if stop_after is None else time.monotonic() + stop_after
        for request in requests:
            if request is WAIT:  # one sequence at a time: every earlier request is already finished and handled
                raise RuntimeError("the sequential HF engine cannot wait for unfinished requests")
            if stop_at is not None and time.monotonic() >= stop_at:
                stats_.stopped_early = True
                return
            item, _ = prepare_protocol_inputs(processor, model.config, request["messages"], request["images"], IMAGE)
            length = item["input_tokens"]
            if length + request["max_tokens"] > CONTEXT_LIMIT:
                yield request, {"blocked": True, "error": f"{length} input + {request['max_tokens']} output tokens exceed {CONTEXT_LIMIT}"}
                continue
            config = GenerationConfig(
                do_sample=True, num_beams=1, temperature=SAMPLING.temperature, top_p=SAMPLING.top_p, top_k=SAMPLING.top_k,
                repetition_penalty=SAMPLING.repetition_penalty, max_new_tokens=request["max_tokens"],
                eos_token_id=base.eos_token_id, pad_token_id=base.pad_token_id, use_cache=True)
            torch.manual_seed(request["seed"])
            began = time.monotonic()
            detector = BlockRepetition() if getattr(args, "loop_stop", False) else None

            class _LoopStop(StoppingCriteria):  # version-4 training rollouts only
                def __call__(self, input_ids, scores, **kwargs):  # type: ignore[no-untyped-def]
                    assert detector is not None
                    return torch.full((input_ids.shape[0],), detector.feed([int(input_ids[0, -1])]),
                                      dtype=torch.bool, device=input_ids.device)

            stopping = StoppingCriteriaList([_LoopStop()]) if detector is not None else None
            # The Merger masters are FP32 while the frozen base computes in BF16 (policy._logprobs uses the same autocast).
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    **item["inputs"].to("cuda:0"), generation_config=config, use_model_defaults=False,
                    logits_processor=LogitsProcessorList([GeneratedOnlyPresencePenalty(length, SAMPLING.presence_penalty)]),
                    **({"stopping_criteria": stopping} if stopping is not None else {}))
            torch.cuda.synchronize()
            seconds = time.monotonic() - began
            ids = getattr(generated, "sequences", generated)[0, length:].tolist()
            finish_reason = "stop" if ids and ids[-1] in eos_ids else "length"
            if detector is not None and detector.found and finish_reason == "length" and len(ids) < request["max_tokens"]:
                finish_reason = LOOP_STOP_FINISH
            if finish_reason == "stop":
                ids = ids[:-1]
            stats_.steps += len(ids)
            stats_.step_seconds += seconds
            stats_.active_sum += len(ids)
            if stats_.on_step is not None:
                stats_.on_step()
            yield request, {"text": processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                            "finish_reason": finish_reason, "prompt_tokens": length, "completion_tokens": len(ids),
                            "seconds": seconds}

    return stream


# Explicit SINGLE path; the legacy builders above retain their original contracts.
SINGLE_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


def single_request_binding(request: dict) -> dict:
    """The identity of one permitted completion, shared by collection and resume validation."""
    if not isinstance(request, dict):
        raise ValueError("SINGLE: a complete request dictionary is required")
    pid = request.get("problem_id", request.get("item"))
    cap = MAX_NEW_TOKENS
    if request.get("evaluation_monitor"):
        if (request["evaluation_monitor"] is not True or request.get("protocol_version") != "error-note-v18b"
                or request.get("stage") not in ("v18b-b2-short-monitor", "v18b-b2-reasoning-monitor")
                or request.get("block_id") != request["stage"] or request.get("sampling") != SINGLE_SAMPLING):
            raise ValueError("only a v18b held-out short B2 monitor can use this diagnostic contract")
        cap = 64
    if (not isinstance(pid, str) or not pid or any(not isinstance(request.get(k), str) or not request[k]
            for k in ("request_id", "stage")) or type(request.get("seed")) is not int
            or not 0 <= request["seed"] < 2**31 or type(request.get("max_tokens")) is not int
            or not 1 <= request["max_tokens"] <= cap
            or not isinstance(request.get("messages"), list) or not isinstance(request.get("images"), list)):
        raise ValueError("SINGLE: complete request identity, messages, images, seed and token cap required")
    if "item" in request and request["item"] != pid:
        raise ValueError("SINGLE: conflicting problem identity")
    block = request.get("block_id")
    if not ((isinstance(block, str) and block) or (type(block) is int and block >= 0)):
        raise ValueError("SINGLE: block_id required")
    if any(k in request for k in ("n", "seeds", "num_return_sequences", "num_beams")):
        raise ValueError("SINGLE: candidate counts, beam counts and multiple seeds are forbidden")
    return {"request_id": request["request_id"], "problem_id": pid, "block_id": block,
            "stage": request["stage"], "seed": request["seed"], "max_tokens": request["max_tokens"]}


def single_request_sha256(request: dict) -> str:
    """Bind the original request and decoded images before a request may enter generation."""
    from fine_tuning.notebook_rl.input_preparation import digest

    binding = single_request_binding(request)
    images = []
    for image in request["images"]:
        if getattr(image, "mode", None) != "RGB":
            raise ValueError("SINGLE: original decoded RGB images required")
        images.append({"mode": image.mode, "size": list(image.size),
                       "sha256": hashlib.sha256(image.tobytes()).hexdigest()})
    return digest(binding | {"messages": request["messages"], "images": images})


class SingleGenerationDeadline(TimeoutError):
    """A configured deadline stopped one original draw stream; scalar traces are audit-only, never resumable."""

    def __init__(self, sampled: dict, request_ids: list[str]):
        super().__init__("SINGLE: generation deadline reached; unfinished requests cannot be resampled")
        self.sampled = sampled
        self.request_ids = list(request_ids)
        self.partials: list[dict] = []


class _SingleActionSampler:
    """Draw each distinct question from its own RNG and retain only scalar action evidence."""

    def __init__(self, requests: list[dict], prompt_width: int, eos: set[int], pad: int, device: Any,
                 *, loop_stop: bool, stop_at: float | None):
        import torch

        self.requests, self.prompt_width, self.eos, self.pad = requests, prompt_width, eos, pad
        self.sampling = dict(requests[0].get("sampling", SINGLE_SAMPLING))
        if any(dict(r.get("sampling", SINGLE_SAMPLING)) != self.sampling for r in requests):
            raise ValueError("SINGLE: one sampling distribution required per HF batch")
        self.full_vocabulary = self.sampling["top_p"] == 1.0 and self.sampling["top_k"] == 0
        self.tail_flags: list[list[bool]] = [[] for _ in requests]
        self.generators = [torch.Generator(device=device).manual_seed(r["seed"]) for r in requests]
        self.tokens: list[list[int]] = [[] for _ in requests]
        self.history = torch.full((len(requests), max(r["max_tokens"] for r in requests)), pad,
                                  dtype=torch.long, device=device)
        self.logprobs: list[list[float]] = [[] for _ in requests]
        self.supports: list[list[str]] = [[] for _ in requests]
        self.support_ids: list[list[list[int]]] = [[] for _ in requests]
        self.reasons = [""] * len(requests)
        self.detectors = [BlockRepetition() if loop_stop else None for _ in requests]
        self.steps, self.stop_at = 0, stop_at

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        from .rl import digest, processed_scores

        if (input_ids.shape != (len(self.requests), self.prompt_width + self.steps)
                or scores.ndim != 2 or scores.shape[0] != len(self.requests)):
            raise RuntimeError("SINGLE: generation prefix or score shape changed")
        if self.steps and not torch.equal(input_ids[:, self.prompt_width:], self.history[:, :self.steps]):
            raise RuntimeError("SINGLE: generated actions differ from the recorded draws")
        if self.stop_at is not None and time.monotonic() >= self.stop_at:
            raise SingleGenerationDeadline({"tokens": [list(row) for row in self.tokens],
                "logprobs": [list(row) for row in self.logprobs], "supports": [list(row) for row in self.supports],
                "support_ids": [list(row) for row in self.support_ids],
                "reasons": [reason or "deadline" for reason in self.reasons], "steps": self.steps},
                [request["request_id"] for request in self.requests])
        forced = torch.full_like(scores, float("-inf"))
        active = [i for i, reason in enumerate(self.reasons) if not reason]
        for i, reason in enumerate(self.reasons):
            if reason:
                forced[i, self.pad] = 0.0
        if active:
            # One batched transform and one host sync per step; each question keeps its own RNG stream.
            rows = torch.tensor(active, dtype=torch.long, device=scores.device)
            transformed = processed_scores(scores[rows], self.history[rows, :self.steps], **self.sampling)
            probabilities = transformed.softmax(-1)
            picks = torch.cat([torch.multinomial(probabilities[j:j + 1], 1, generator=self.generators[i])
                               for j, i in enumerate(active)])
            chosen = transformed.log_softmax(-1).gather(-1, picks).flatten()
            self.history[rows, self.steps] = picks.flatten()
            forced[rows, picks.flatten()] = 0.0
            supports: list[list[int]] = [[] for _ in active]
            if not self.full_vocabulary:
                for row, column in torch.isfinite(transformed).nonzero().tolist():
                    supports[row].append(column)
            else:
                diagnostic = processed_scores(scores[rows], self.history[rows, :self.steps], **SINGLE_SAMPLING)
                flags = ~torch.isfinite(diagnostic.gather(-1, picks).flatten())
            for j, (token, logprob) in enumerate(zip(picks.flatten().tolist(), chosen.tolist(), strict=True)):
                i = active[j]
                self.logprobs[i].append(float(logprob))
                if self.full_vocabulary:
                    self.tail_flags[i].append(bool(flags[j]))
                else:
                    self.support_ids[i].append(supports[j])
                    self.supports[i].append(digest(supports[j]))
                self.tokens[i].append(token)
                detector = self.detectors[i]
                repeated = detector.feed([token]) if detector is not None else False
                if token in self.eos:
                    self.reasons[i] = "stop"
                elif repeated:
                    self.reasons[i] = LOOP_STOP_FINISH
                elif len(self.tokens[i]) == self.requests[i]["max_tokens"]:
                    self.reasons[i] = "length"
        self.steps += 1
        # HF remains in sampling mode; these one-token distributions emit the independent stochastic draws above.
        return forced

    def rows_done(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        import torch

        return torch.tensor([bool(r) for r in self.reasons], dtype=torch.bool, device=input_ids.device)

    def finish(self, sequences: Any, prompts: Any) -> None:
        import torch

        if (not all(self.reasons) or sequences.shape != (len(self.requests), self.prompt_width + self.steps)
                or not torch.equal(sequences[:, :self.prompt_width], prompts)
                or any(sequences[i, self.prompt_width:self.prompt_width + len(row)].tolist() != row
                       for i, row in enumerate(self.tokens))):
            raise RuntimeError("SINGLE: missing, duplicated or altered completion")


def _single_batch_inputs(prepared: list[dict], pad: int, device: Any) -> dict:
    """Left-pad different prompts; concatenate image tensors in the same question/image order."""
    import torch

    width = max(p["input_tokens"] for p in prepared)
    keys = set(prepared[0]["inputs"])
    allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "token_type_ids", "mm_token_type_ids"}
    if not {"input_ids", "attention_mask"} <= keys or keys - allowed or any(set(p["inputs"]) != keys for p in prepared):
        raise ValueError("SINGLE: unsupported or inconsistent processor tensors")
    batch = {}
    for key in keys:
        values = [p["inputs"][key] for p in prepared]
        if key in ("pixel_values", "image_grid_thw"):
            batch[key] = torch.cat(values, dim=0).to(device)
        else:
            if any(v.shape != (1, p["input_tokens"]) for v, p in zip(values, prepared, strict=True)):
                raise ValueError("SINGLE: processor sequence tensor shape changed")
            batch[key] = torch.cat([torch.nn.functional.pad(v, (width-v.shape[1], 0), value=pad if key == "input_ids" else 0)
                                    for v in values], dim=0).to(device)
    return batch


def _single_generate_batch(bundle: Any, rows: list[dict], prepared: list[dict], dimensions: list[dict], *,
                           eos: Any, eos_ids: set[int], pad: int, stop_at: float | None,
                           loop_stop: bool, capture: Any = None) -> dict:
    """Own batch GPU tensors locally so every measurement starts before inputs/history and releases them on return."""
    import torch
    from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
    from .capacity import CapacityHold, shape

    model = bundle.model
    device = next(model.parameters()).device
    measurement = (capture.measure("generation", shape(dimensions, generation=True),
        hashlib.sha256(json.dumps([p["input_sha256"] for p in prepared]).encode()).hexdigest())
        if capture is not None else contextlib.nullcontext(None))
    with measurement as measured:
        inputs = _single_batch_inputs(prepared, pad, device)
        sampler = _SingleActionSampler(rows, inputs["input_ids"].shape[1], eos_ids, pad, device,
                                       loop_stop=loop_stop, stop_at=stop_at)
        config = GenerationConfig(do_sample=True, num_beams=1, num_return_sequences=1, temperature=1.0,
            top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=max(r["max_tokens"] for r in rows),
            eos_token_id=eos, pad_token_id=pad, use_cache=True, renormalize_logits=False,
            return_dict_in_generate=True, output_scores=False, output_logits=False)
        began, generated = time.monotonic(), None
        devices = [device.index or 0] if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices), torch.inference_mode(), (
                    torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()):
                generated = model.generate(**inputs, generation_config=config, use_model_defaults=False,
                    logits_processor=LogitsProcessorList([sampler]),
                    stopping_criteria=StoppingCriteriaList([sampler.rows_done]))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            sampler.finish(getattr(generated, "sequences", generated), inputs["input_ids"])
            if measured is not None:
                cache = getattr(generated, "past_key_values", None)
                if cache is None or not hasattr(cache, "layers") or any(
                        value.dtype != torch.bfloat16 for layer in cache.layers for value in (layer.keys, layer.values)):
                    raise CapacityHold("capacity HOLD: unmeasured/non-BF16 KV cache")
                del cache
                measured["shape"] = shape([row | {"target_tokens": len(tokens)}
                    for row, tokens in zip(dimensions, sampler.tokens, strict=True)], generation=True)
                measured["work_ids"] = [row["request_id"] for row in rows]
                measured["complete"] = True
        except SingleGenerationDeadline as error:
            error.sampled["seconds"] = time.monotonic() - began
            raise
        finally:
            if generated is not None and hasattr(generated, "past_key_values"):
                generated["past_key_values"] = None
        return {"tokens": sampler.tokens, "logprobs": sampler.logprobs, "supports": sampler.supports,
                "support_ids": sampler.support_ids,
                "reasons": sampler.reasons, "steps": sampler.steps, "seconds": time.monotonic() - began,
                "b2_tail_flags": sampler.tail_flags}


def hf_single_stream_builder(bundle: Any, args: argparse.Namespace, stats: StreamStats) -> Stream:
    """Reuse the live policy for bounded batches of different SINGLE questions, with exact action evidence.

    max_num_batched_tokens bounds rows * (padded prompt width + maximum output cap). Optional measured admission
    splits only before execution; no sampling retry or unmeasured sizing.
    Construct a new stream after a policy update, using the same bundle and its new policy SHA.
    """
    from .rl import policy_sha256, sampling_sha256
    from .capacity import CapacityHold, shape as capacity_shape

    seqs = getattr(args, "max_num_seqs", 1)
    context = getattr(args, "context_limit", CONTEXT_LIMIT)
    budget = getattr(args, "max_num_batched_tokens", context)
    if (type(seqs) is not int or not 1 <= seqs <= 16 or type(context) is not int or context < 1
            or type(budget) is not int or budget < 1):
        raise ValueError("SINGLE: explicit sequence capacity 1..16, context and padded token budget required")
    policy, protocol = args.policy_sha256, args.protocol_sha256
    if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
           for h in (policy, protocol)):
        raise ValueError("SINGLE: policy and protocol SHA256 required")
    if policy_sha256(bundle) != policy:
        raise ValueError("SINGLE: stale live policy")
    sampling = dict(getattr(args, "sampling", SINGLE_SAMPLING))
    full_vocabulary = sampling["top_p"] == 1.0 and sampling["top_k"] == 0
    sampling_hash = sampling_sha256(sampling)
    capacity, capture = getattr(args, "capacity_session", None), getattr(args, "capacity_capture", None)
    if capacity is not None and capture is not None:
        raise CapacityHold("capacity HOLD: measurement and measured admission are separate runs")
    prepared_by_id: dict[str, dict] = {}
    planned: list[list[str]] = []
    admitted: list[tuple[str, str]] = []
    start_batch: Callable | None = None

    def set_start_callback(callback: Callable | None) -> None:
        nonlocal start_batch
        start_batch = callback

    def row_shape(request: dict, item: dict) -> dict:
        return {"problem_id": single_request_binding(request)["problem_id"], "prompt_tokens": item["input_tokens"],
                "target_tokens": request["max_tokens"], "image_grids": item["inputs"]["image_grid_thw"].tolist()}

    def admit(requests: list[dict]) -> None:
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

        nonlocal planned, admitted
        assert capacity is not None
        prepared_by_id.clear()
        shapes: list[dict] = []
        base_plan: list[list[int]] = []
        current: list[int] = []
        for index, request in enumerate(requests):
            binding_ = single_request_binding(request)
            item, _ = prepare_protocol_inputs(bundle.processor, bundle.model.config, request["messages"], request["images"],
                                               getattr(args, "image_cfg", IMAGE))
            prepared_by_id[binding_["request_id"]] = item
            dimensions = row_shape(request, item)
            if dimensions["prompt_tokens"] + dimensions["target_tokens"] > min(context, budget):
                raise CapacityHold("capacity HOLD: singleton exceeds context/token budget")
            shapes.append(dimensions)
            changed_stage = current and any(single_request_binding(requests[current[0]])[key] != binding_[key]
                                            for key in ("block_id", "stage"))
            overflow = False
            if not changed_stage and len(current) < seqs:
                cost = capacity_shape([shapes[i] for i in [*current, index]], generation=True)
                overflow = cost["rows"] * cost["total_tokens"] > budget
            if current and (changed_stage or len(current) == seqs or overflow):
                base_plan.append(current)
                current = []
            current.append(index)
        if current:
            base_plan.append(current)
        indices = capacity.partition(shapes, base_plan, "generation")
        planned = [[requests[i]["request_id"] for i in batch] for batch in indices]
        admitted = [(request["request_id"], single_request_sha256(request)) for request in requests]
    seen: set[str] = set()
    submitted: set[tuple[Any, str, str]] = set()
    block_items: dict[Any, set[str]] = {}
    stats.receipt_extra.update(single_sampler="hf-independent-question-v1", single_batch_rows=seqs,
                               single_padded_token_budget=budget)

    def stream(requests: Iterable[dict], stats_: StreamStats, stop_after: float | None) -> Iterator[tuple[dict, dict]]:
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

        model, processor = bundle.model, bundle.processor
        if policy_sha256(bundle) != policy:
            raise ValueError("SINGLE: stale live policy")
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else () if eos is None else (eos,))
        pad = model.generation_config.pad_token_id
        if type(pad) is not int or any(type(token) is not int for token in eos_ids):
            raise ValueError("SINGLE: integer padding and EOS token IDs required")
        stop_at = None if stop_after is None else time.monotonic() + stop_after
        if capacity is not None:
            requests = list(requests)
            if [(r["request_id"], single_request_sha256(r)) for r in requests] != admitted:
                admit(requests)
        iterator = iter(requests)
        waiting: dict | None = None
        was_training = model.training
        model.eval()
        batch_number = 0
        try:
            while True:
                if stop_at is not None and time.monotonic() >= stop_at:
                    stats_.stopped_early = True
                    return
                batch: list[tuple[dict, dict]] = []
                batch_limit = len(planned[batch_number]) if capacity is not None and batch_number < len(planned) else seqs
                while len(batch) < batch_limit:
                    if waiting is None:
                        try:
                            request = next(iterator)
                        except StopIteration:
                            break
                    else:
                        request = waiting
                    waiting = None
                    binding = single_request_binding(request)
                    if full_vocabulary:
                        request = request | {"sampling": sampling}
                    if batch and any(binding[k] != single_request_binding(batch[0][0])[k] for k in ("block_id", "stage")):
                        waiting = request
                        break
                    if capacity is not None:
                        item = prepared_by_id[binding["request_id"]]
                    else:
                        item, _ = prepare_protocol_inputs(processor, model.config, request["messages"], request["images"],
                                                           getattr(args, "image_cfg", IMAGE))
                    length, cap = item["input_tokens"], request["max_tokens"]
                    if length + cap > context or length + cap > budget:
                        raise ValueError("SINGLE: prompt plus completion exceeds explicit context/token budget")
                    cost = (len(batch)+1) * (max([p["input_tokens"] for _, p in batch] + [length])
                                           + max([r["max_tokens"] for r, _ in batch] + [cap]))
                    if batch and cost > budget:
                        waiting = request
                        break
                    key = (binding["block_id"], binding["stage"], binding["problem_id"])
                    if (binding["request_id"] in seen or key in submitted
                            or any(r["request_id"] == binding["request_id"] for r, _ in batch)):
                        raise ValueError("SINGLE: duplicate request or second candidate for a question/stage")
                    if any(binding["problem_id"] == single_request_binding(r)["problem_id"] for r, _ in batch):
                        raise ValueError("SINGLE: concurrent requests must be different questions")
                    members = block_items.get(binding["block_id"], set()) | {binding["problem_id"]}
                    members |= {single_request_binding(r)["problem_id"] for r, _ in batch}
                    if len(members) > 16:
                        raise ValueError("SINGLE: more than 16 different questions in a frozen block")
                    batch.append((request, item))
                if not batch:
                    return
                rows, prepared = [r for r, _ in batch], [p for _, p in batch]
                dimensions = [row_shape(r, item) for r, item in batch]
                if capacity is not None:
                    if batch_number >= len(planned) or [r["request_id"] for r in rows] != planned[batch_number]:
                        raise CapacityHold("capacity HOLD: admitted FIFO plan changed")
                    receipt = capacity.check(dimensions, "generation")
                    stats_.receipt_extra.setdefault("capacity_admissions", []).append(receipt)
                    if start_batch is not None:
                        start_batch(rows)
                for request in rows:
                    record = single_request_binding(request)
                    seen.add(record["request_id"])
                    submitted.add((record["block_id"], record["stage"], record["problem_id"]))
                    block_items.setdefault(record["block_id"], set()).add(record["problem_id"])
                batch_number += 1
                deadline = None
                try:
                    sampled = _single_generate_batch(bundle, rows, prepared, dimensions, eos=eos, eos_ids=eos_ids,
                        pad=pad, stop_at=stop_at, loop_stop=getattr(args, "loop_stop", False), capture=capture)
                except SingleGenerationDeadline as error:
                    stats_.stopped_early = True
                    deadline, sampled = error, error.sampled
                except TimeoutError:
                    stats_.stopped_early = True
                    raise
                stats_.steps += sampled["steps"]
                stats_.step_seconds += sampled["seconds"]
                stats_.active_sum += sum(map(len, sampled["tokens"]))
                if policy_sha256(bundle) != policy:
                    raise ValueError("SINGLE: policy changed during collection")
                for i, (request, item) in enumerate(batch):
                    ids = sampled["tokens"][i]
                    text = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    generation = single_request_binding(request) | {"prompt_token_ids": list(item["prompt_token_ids"]),
                        "completion_token_ids": list(ids), "sampling_logprobs": sampled["logprobs"][i],
                        "support_sha256": sampled["supports"][i], "support_token_ids": sampled["support_ids"][i],
                        "sampling": dict(sampling),
                        "policy_sha256": policy, "protocol_sha256": protocol, "sampling_sha256": sampling_hash,
                        "input_sha256": item["input_sha256"], "finish_reason": sampled["reasons"][i],
                        "request_sha256": single_request_sha256(request), "text": text,
                        "eos_token_ids": sorted(eos_ids), "completion_count": 1}
                    if full_vocabulary:
                        generation.pop("support_sha256")
                        generation.pop("support_token_ids")
                        generation.update(probability_space="processed-full-vocab-v1",
                                          b2_tail_flags=sampled["b2_tail_flags"][i])
                    output = {"text": text, "finish_reason": sampled["reasons"][i], "prompt_tokens": item["input_tokens"],
                        "completion_tokens": len(ids), "seconds": sampled["seconds"], "completion_count": 1, "generation": generation}
                    if deadline is not None:
                        generation.update(incomplete=True, input_tensors=item["input_tensors"])
                        output["incomplete"] = True
                        deadline.partials.append(output)
                    else:
                        yield request, output
                if deadline is not None:
                    raise deadline
                if stats_.on_step is not None:
                    stats_.on_step()
        finally:
            model.train(was_training)

    if capacity is not None:
        setattr(stream, "admit", admit)
        setattr(stream, "set_start_callback", set_start_callback)
    return stream


# ------------------------------------------------------------------ sweep

def _child_command(args: argparse.Namespace, seqs: int, stop_after: float | None) -> list[str]:
    command = [sys.executable, "-m", "fine_tuning.error_note.collect", "solve",
               "--build-root", str(args.build_root), "--order", str(args.order), "--limit", str(args.limit),
               "--model-path", str(args.model_path), "--output", str(args.output), "--engine", args.engine,
               "--kv-cache-gib", str(args.kv_cache_gib), "--max-num-batched-tokens", str(args.max_num_batched_tokens),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--max-num-seqs", str(seqs)]
    if getattr(args, "answer_contract", TURN1_ANSWER_CONTRACT) != TURN1_ANSWER_CONTRACT:
        command += ["--answer-contract", args.answer_contract, "--training-rules", str(args.training_rules),
                    "--training-rules-sha256", args.training_rules_sha256]
        if getattr(args, "code_commit", None):
            command += ["--code-commit", args.code_commit]
    return command + ([] if stop_after is None else ["--stop-after-seconds", str(stop_after)])


def _run_child(command: list[str]) -> int:
    return subprocess.run(command).returncode


def sweep(args: argparse.Namespace, run_child: Callable[[list[str]], int] = _run_child,
          total_gib: Callable[[], float] = gpu_total_gib) -> int:
    """One short solve child per --seqs value (or "auto": candidates from the KV size), then a final child with the
    fastest S that runs the rest to completion. The children get --kv-cache-gib as given and resolve it themselves."""
    out, items = Path(args.output), _prefix(args)
    kv = resolve_kv(args.kv_cache_gib, total_gib)
    seqs_values = (auto_seqs(int(kv["kv_cache_gib_resolved"])) if args.seqs == "auto"
                   else [int(v) for v in args.seqs.split(",")])
    out.mkdir(parents=True, exist_ok=True)
    total, ids = len(items), {item["problem_id"] for item in items}

    def finished() -> int:
        return len(ids & set(read_jsonl_done(out / "t1.jsonl", repair=False)))

    runs: list[dict] = []
    for seqs in seqs_values:
        if finished() >= total:
            break  # everything finished early; the remaining values are skipped
        (out / "receipt-t1.json").unlink(missing_ok=True)
        code = run_child(_child_command(args, seqs, args.seconds))
        if code:
            return code
        receipt = json.loads((out / "receipt-t1.json").read_text())
        write_json(out / f"receipt-t1-s{seqs}.json", receipt)
        runs.append(receipt)
    measured = [run for run in runs if run.get("tokens_per_second") is not None]
    best = max(measured, key=lambda run: (run["tokens_per_second"], -run["S"]))["S"] if measured else None
    result = {"runs": runs, "best_seqs": best} | kv | {"seqs_resolved": seqs_values}
    write_json(out / "sweep.json", result)
    if best is not None and finished() < total:
        code = run_child(_child_command(args, best, None))
        if code:
            return code
        result["final"] = json.loads((out / "receipt-t1.json").read_text())
        write_json(out / "sweep.json", result)
    return 0


# ------------------------------------------------------------------ rows

def make_records(items: list[dict], actors: dict, t1: dict[str, dict], note: dict[str, dict]) -> list[dict]:
    """The inputs of contract.build_rows. A note row with a status (SKIPPED / CONTEXT_BLOCKED) counts as no note."""
    records = []
    for item in items:
        pid = item["problem_id"]
        if pid not in t1:
            raise ValueError(f"no T1 row for {pid}")
        actor, first, second = actors[pid], t1[pid], note.get(pid)
        record = {"problem_id": pid, "subject": item["subject"], "source": item["source"],
                  "p0_text": p0_eval_text(actor.question, actor.choices), "image_paths": list(actor.image_paths),
                  "t1": {"text": first.get("text", ""), "finish_reason": first.get("finish_reason") or "",
                         "status": first["status"]}}
        if second is not None and "status" not in second:
            record["note"] = {key: second[key] for key in ("prompt", "text", "finish_reason", "corrected_status")}
        records.append(record)
    return records


def summarize(records: list[dict], rows: list[dict], drops: list[dict], parse: Callable = parse_note) -> dict:
    """Counts only; no text."""
    parsed = [parse(r["note"]["text"]) if "note" in r else "NO_NOTE" for r in records]
    by_kind: dict[str, list[int]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["target_tokens"])
    return {
        "items": len(records), "rows": len(rows), "rows_by_kind": dict(Counter(r["kind"] for r in rows)),
        "rows_by_target_source": dict(Counter(r["target_source"] for r in rows)),
        "rows_by_route": dict(Counter(r["route"] for r in rows)),
        "rows_by_subject": dict(Counter(r["subject"] for r in rows)),
        "drops": len(drops), "drops_by_reason": dict(Counter(d["reason"] for d in drops)),
        "image_reading": dict(Counter(p if isinstance(p, str) else "UNPARSED" if p is None else p.image_reading for p in parsed)),
        "cause": dict(Counter(p if isinstance(p, str) else "UNPARSED" if p is None else p.cause for p in parsed)),
        "t1_status": dict(Counter(r["t1"]["status"] for r in records)),
        "mean_target_tokens": {kind: round(sum(v) / len(v), 1) for kind, v in by_kind.items()},
    }


def make_records_v2(items: list[dict], actors: dict, t1: dict[str, dict], note: dict[str, dict],
                    resolve: dict[str, dict]) -> list[dict]:
    """The inputs of contract.build_rows_v2: note {prompt, text, finish_reason}, resolve {text, finish_reason, status}
    or {skipped}. A note or resolve row with a status other than a grade (SKIPPED / CONTEXT_BLOCKED note) is no note."""
    records = []
    for item in items:
        pid = item["problem_id"]
        if pid not in t1:
            raise ValueError(f"no T1 row for {pid}")
        actor, first, second, third = actors[pid], t1[pid], note.get(pid), resolve.get(pid)
        record = {"problem_id": pid, "subject": item["subject"], "source": item["source"],
                  "p0_text": p0_eval_text(actor.question, actor.choices), "image_paths": list(actor.image_paths),
                  "t1": {"text": first.get("text", ""), "finish_reason": first.get("finish_reason") or "",
                         "status": first["status"]}}
        if second is not None and "status" not in second:
            if second.get("note_version") != "v2":
                raise ValueError(f"{pid}: rows v2 needs v2 notes")
            record["note"] = {key: second[key] for key in ("prompt", "text", "finish_reason")}
        if third is not None:
            record["resolve"] = ({"skipped": third["skipped"]} if "skipped" in third else
                                 {"text": third.get("text", ""), "finish_reason": third.get("finish_reason") or "",
                                  "status": third["status"]})
        records.append(record)
    return records


def length_report(outputs: dict[str, dict]) -> dict:
    """Completions cut at the token limit, split into loops (contract.is_loop) and the rest: the rest is the share
    that a higher limit might finish (user decision 2026-11-02: report it; the limit is never raised automatically)."""
    cut = [row["text"] for row in outputs.values() if row.get("finish_reason") == "length"]
    loops = sum(is_loop(text) for text in cut)
    return {"generated": sum("text" in row for row in outputs.values()), "length": len(cut), "length_loop": loops,
            "length_not_loop": len(cut) - loops, "not_loop_share": (len(cut) - loops) / len(cut) if cut else None}


def collect_rows_v2(records: list[dict], tokenize: Callable[[str], list[int]],
                    outputs: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict], dict]:
    """outputs: {"t1", "note", "resolve"} -> the rows of the ordered prefix, for the counts that need more than records."""
    rows: list[dict] = []
    drops: list[dict] = []
    for record in records:
        built, dropped = build_rows_v2(record, tokenize)
        rows += built
        drops += dropped
    resolves = [r["resolve"] for r in records if "resolve" in r]
    attempted = [r for r in resolves if "skipped" not in r]
    correct = sum(r["status"] == "CORRECT" for r in attempted)
    skipped = Counter(r["skipped"] for r in resolves if "skipped" in r)
    notes = [row for row in outputs["note"].values() if "status" not in row]
    summary = summarize(records, rows, drops, parse_note_v2) | {
        "rows_version": "v2",
        "resolve": {"candidates": sum(r["t1"]["status"] != "CORRECT" for r in records),
                    "no_row": sum(r["t1"]["status"] != "CORRECT" and "resolve" not in r for r in records),
                    "attempted": len(attempted), "correct": correct,
                    "success_rate": correct / len(attempted) if attempted else None,
                    "skipped_by_reason": dict(skipped),
                    "hint_leak_skips": sum(n for reason, n in skipped.items() if reason.startswith("HINT_LEAK:"))},
        "note_hint_leak_by_reason": dict(Counter((row["hint_leak"] or "NONE").split(":")[0] for row in notes
                                                 if row["lesson"] is not None)),
        "length_finishes": {stage: length_report(rows_) for stage, rows_ in outputs.items()},
    }
    return rows, drops, summary


def collect_rows(records: list[dict], tokenize: Callable[[str], list[int]]) -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    drops: list[dict] = []
    for record in records:
        built, dropped = build_rows(record, tokenize)
        rows += built
        drops += dropped
    return rows, drops, summarize(records, rows, drops)


# ------------------------------------------------------------------ CLI

def _prefix(args: argparse.Namespace) -> list[dict]:
    items = read_order(Path(args.order))["items"]
    if not 1 <= args.limit <= len(items):
        raise ValueError(f"--limit must be 1..{len(items)}")
    return items[:args.limit]


def _grader(service: Any, answer_contract: str = TURN1_ANSWER_CONTRACT) -> Callable:
    def grade(problem_id: str, text: str, finish_reason: str, mode: str) -> tuple[str, str | None]:
        result = service.grade(problem_id, text, finish_reason, mode, answer_contract=answer_contract)
        return result.status, result.extracted
    return grade


def _generate(args: argparse.Namespace, stage: str) -> int:
    from fine_tuning.notebook_rl.dataset import load_images
    from fine_tuning.notebook_rl.vault import read_actors, read_verifier
    from fine_tuning.notebook_rl.verifier_service import VerifierService

    build_root, out, vllm = Path(args.build_root), Path(args.output), args.engine == "vllm"
    out.mkdir(parents=True, exist_ok=True)
    items = _prefix(args)
    answer_contract = getattr(args, "answer_contract", TURN1_ANSWER_CONTRACT)
    v3 = answer_contract == TRAINING_ANSWER_CONTRACT
    lessons = None
    service_options = {}
    private_files = []
    if v3:
        from .collect_v3 import bind_collection, verified_lessons_for_actor

        bind_collection(args, items)
        if stage == "note" and args.note_version != "v2":
            raise ValueError("v3 requires the note-only v2 prompt")
        service_options = {"training_rules_path": args.training_rules,
                           "training_rules_sha256": args.training_rules_sha256}
        private_files = [args.training_rules, out / "note.jsonl", out / "audit.json"]
        private_files += [p for p in (args.source_provenance, args.audit, args.reviews) if p is not None]
        if stage == "resolve":
            if args.source_provenance is None:
                raise ValueError("v3 resolve requires source provenance")
            lessons = verified_lessons_for_actor(args)
    seqs = args.max_num_seqs
    gold = None
    if stage in ("note", "resolve") and args.max_num_seqs_from:
        seqs = json.loads(Path(args.max_num_seqs_from).read_text())["best_seqs"]
        if type(seqs) is not int:
            raise ValueError("--max-num-seqs-from: the sweep has no best_seqs")
    auto = None
    if seqs == "auto":  # no sweep (T1 reused): the middle sweep candidate for the KV size
        auto = auto_seqs(int(resolve_kv(args.kv_cache_gib)["kv_cache_gib_resolved"]))
        seqs = auto[len(auto) // 2]
    args.max_num_seqs = seqs
    if stage == "note":
        verifier = read_verifier(build_root / "train")  # the gold is read before any guard, in this process only
        gold = {item["problem_id"]: verifier[item["problem_id"]].gold for item in items}
        del verifier
    guard_options = {"private_files": private_files} if v3 else {}
    # The note writer is explicitly gold-aware; solve and resolve may not open its files.
    if v3 and stage == "note":
        guard_options["private_files"] = [p for p in private_files if p != out / "note.jsonl"]
    forbidden = install_training_guards(build_root, vllm_schema={} if vllm else None, **guard_options)
    actors = {actor.problem_id: actor for actor in read_actors(build_root / "train")}
    unknown = [item["problem_id"] for item in items if item["problem_id"] not in actors]
    if unknown:
        raise ValueError(f"{len(unknown)} ordered items are not in the train split, first {unknown[:3]}")
    stats = StreamStats()
    if auto is not None:
        stats.receipt_extra["max_num_seqs_auto"] = {"candidates": auto, "chosen": seqs}
    stream = (_vllm_stream_builder if vllm else _hf_stream_builder)(args, stats)
    options = dict(S=seqs, B=args.max_num_batched_tokens, engine=args.engine, stop_after=args.stop_after_seconds)
    load = lambda actor: load_images(actor, build_root / "images")  # noqa: E731
    with VerifierService(build_root / "train", forbidden_roots=forbidden, **service_options) as service, ResourceLog(out / "resource.jsonl"):
        grade = _grader(service, answer_contract)
        if stage == "solve":
            receipt = run_solve(items, actors, load, stream, stats, grade, out, **options)
        elif stage == "resolve":  # gold-blind like solve: lessons.json, never note.jsonl
            t1 = read_jsonl_done(out / "t1.jsonl", repair=False)
            approved_lessons = lessons if v3 else read_lessons(out)
            assert approved_lessons is not None
            receipt = run_resolve(items, actors, t1, approved_lessons, load, stream, stats, grade, out,
                                  **options)
        else:
            assert gold is not None
            t1 = read_jsonl_done(out / "t1.jsonl", repair=False)
            receipt = run_note(items, actors, t1, gold, load, stream, stats, grade, out, **options,
                               version=args.note_version, answer_contract=answer_contract)
    print(json.dumps(receipt))
    return 0


def _rows(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from fine_tuning.notebook_rl.vault import read_actors

    out = Path(args.output)
    destination = Path(args.rows_output) if args.rows_output else out
    destination.mkdir(parents=True, exist_ok=True)
    items = _prefix(args)
    actors = {actor.problem_id: actor for actor in read_actors(Path(args.build_root) / "train")}
    t1, note = read_jsonl_done(out / "t1.jsonl", repair=False), read_jsonl_done(out / "note.jsonl", repair=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True)
    tokenize = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    if args.rows_version == "v3":
        from .collect_v3 import collect_rows_v3

        rows, drops, summary = collect_rows_v3(args, tokenize)
    elif args.rows_version == "v2":
        resolve = read_jsonl_done(out / "resolve.jsonl", repair=False)
        ids = {item["problem_id"] for item in items}
        outputs = {name: {pid: row for pid, row in rows_.items() if pid in ids}
                   for name, rows_ in (("t1", t1), ("note", note), ("resolve", resolve))}
        rows, drops, summary = collect_rows_v2(make_records_v2(items, actors, t1, note, resolve), tokenize, outputs)
    else:
        rows, drops, summary = collect_rows(make_records(items, actors, t1, note), tokenize)
    write_atomic(destination / "rows.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_atomic(destination / "drops.jsonl", "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in drops))
    write_json(destination / "rows_summary.json", summary)
    print(json.dumps(summary))
    return 0


def _kv_arg(text: str) -> float | str:
    if text == "auto":
        return text
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number or 'auto'")
    return value


def _seqs_arg(text: str) -> int | str:
    if text == "auto":
        return text
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer or 'auto'")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--build-root", type=Path, required=True)
        p.add_argument("--order", type=Path, required=True)
        p.add_argument("--limit", type=int, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--answer-contract", choices=(TURN1_ANSWER_CONTRACT, TRAINING_ANSWER_CONTRACT),
                       default=TURN1_ANSWER_CONTRACT)
        p.add_argument("--training-rules", type=Path)
        p.add_argument("--training-rules-sha256")
        p.add_argument("--source-provenance", type=Path)
        p.add_argument("--audit", type=Path)
        p.add_argument("--reviews", type=Path)
        p.add_argument("--code-commit", help="declared source commit; production source bytes are independently bound")

    def engine(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model-path", type=Path, required=True)
        p.add_argument("--engine", choices=("vllm", "hf"), required=True)
        p.add_argument("--kv-cache-gib", type=_kv_arg, default=28, help="GiB of KV cache, or auto (GPU memory - 16)")
        p.add_argument("--max-num-batched-tokens", type=int, default=16384)
        p.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    def run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-num-seqs", type=_seqs_arg, default=32,
                       help="concurrent sequences, or auto (the middle sweep candidate for the KV size)")
        p.add_argument("--stop-after-seconds", type=float, default=None)

    for name in ("solve", "note", "sweep", "resolve"):
        p = sub.add_parser(name)
        common(p)
        engine(p)
        if name != "sweep":
            run_options(p)
    for name in ("note", "resolve"):
        sub.choices[name].add_argument("--max-num-seqs-from", type=Path, default=None)
    sub.choices["note"].add_argument("--note-version", choices=("v1", "v2"), default="v1")
    sub.choices["sweep"].add_argument("--seqs", required=True, help="comma list of --max-num-seqs values, or auto")
    sub.choices["sweep"].add_argument("--seconds", type=float, default=300.0)
    rows = sub.add_parser("rows")
    common(rows)
    rows.add_argument("--tokenizer-path", type=Path, required=True)
    rows.add_argument("--rows-output", type=Path, default=None)
    rows.add_argument("--rows-version", choices=("v1", "v2", "v3"), default="v1")
    for name in ("audit", "verify-lessons"):
        common(sub.add_parser(name))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "sweep":
        return sweep(args)
    if args.command == "rows":
        return _rows(args)
    if args.command in ("audit", "verify-lessons"):
        from .collect_v3 import audit_collection, verify_lessons

        if args.source_provenance is None or (args.command == "audit" and args.reviews is None):
            raise ValueError("audit needs source provenance and explicit review receipts")
        value = audit_collection(args) if args.command == "audit" else verify_lessons(args)
        print(json.dumps(value))
        return 0
    return _generate(args, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
