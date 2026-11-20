"""Source-bound F0 preparation and monitor collection for the v16b campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from .audit import digest
from .collect import append_jsonl, write_json
from .contract import MASTER_SEED, sample_seed

VERSION = "v16b-preparation-v1"
MODERN = {"error-note-v17", "error-note-v18", "error-note-v18b"}


def __getattr__(name: str) -> Any:
    if name == "single":
        from . import single
        return single
    raise AttributeError(name)


def preparation_version(version: str) -> str:
    return version.removeprefix("error-note-") + "-preparation-v1"


def preparation_stage(version: str, role: str) -> str:
    # Accuracy monitoring keeps v18 seeds; short diagnostics use a separate fixed draw.
    if version == "error-note-v18b":
        if role in {"short_monitor", "reasoning_monitor"}:
            return "v18b-b2-" + role.replace("_", "-")
        version = "error-note-v18"
    return version.removeprefix("error-note-") + "-" + role


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal(path: Path, body: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = body | {"sha256": digest(body)}
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError(f"single runtime: immutable binding changed: {path.name}")
    else:
        write_json(path, result)
    return result


def read_sealed(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"v16b: sealed artifact changed: {path.name}")
    return value

def data_api(version: str = "error-note-v16b"):
    if version == "error-note-v18b":
        from . import v18b_data
        return v18b_data
    if version == "error-note-v18":
        from . import v18_data
        return v18_data
    if version in MODERN:
        from . import v17_data
        return v17_data
    if version != "error-note-v16b":
        raise ValueError("unknown preparation version")
    from . import data_v16b
    return data_v16b


def preflight_items(items: list[dict], *, version: str = "error-note-v16b") -> dict:
    """Check every native note contract and optional observation shape without loading a model."""
    from .v16b_rows import build_obs_prompt_v16b, observation_eligibility_v16b, problem_contract_v16b
    prompt_builder: Any = build_obs_prompt_v16b

    if version in MODERN:
        from .v17_rows import problem_contract_v17, observation_eligibility_v17, build_obs_request_v17
        problem_contract_v16b = problem_contract_v17
        observation_eligibility_v16b = observation_eligibility_v17
        prompt_builder = build_obs_request_v17
        if version == "error-note-v18":
            from .v18_rows import problem_contract_v17, observation_eligibility_v17, build_obs_request_v17
            problem_contract_v16b = problem_contract_v17
            observation_eligibility_v16b = observation_eligibility_v17
            prompt_builder = build_obs_request_v17
        elif version == "error-note-v18b":
            from .v18b_rows import problem_contract_v17, observation_eligibility_v17, build_obs_request_v17
            problem_contract_v16b = problem_contract_v17
            observation_eligibility_v16b = observation_eligibility_v17
            prompt_builder = build_obs_request_v17
    elif version != "error-note-v16b":
        raise ValueError("unknown data preflight version")
    records = []
    for item in items:
        contract = problem_contract_v16b(item)
        eligibility = observation_eligibility_v16b(item)
        if eligibility["eligible"]:
            prompt_builder(item)
        records.append({"problem_id": item["problem_id"], "note_contract_sha256": digest(contract),
                        "observation_eligibility": eligibility})
        if version in MODERN:
            records[-1]["repair_supported"] = bool(contract.get("supported"))
    return {"version": version.removeprefix("error-note-") + "-data-contract-preflight-v1", "count": len(records), "items": records,
            "observation_eligible": sum(row["observation_eligibility"]["eligible"] for row in records),
            "first_solution_count": len(records), "model_loaded": False}


def select(data_root: Path, output: Path, *, local_sequential: bool = False,
           version: str = "error-note-v16b") -> dict:
    from .data_v16b import _identities

    data = data_api(version)
    load_items = data.load_items
    validation = data.validate(data_root)
    roles = (("train", "prep", "monitor", "effect", "performance", "reasoning_monitor")
             if version == "error-note-v18b" else ("train", "prep", "monitor"))
    splits = {name: load_items(data_root, name) for name in roles}
    previous: set[tuple[str, str]] = set()
    for name, items in splits.items():
        identities = set().union(*(_identities(item) for item in items))
        if previous & identities or len({item["problem_id"] for item in items}) != len(items):
            raise ValueError("v16b preparation: overlapping split/family/image identities")
        previous.update(identities)
        if any(item["training_allowed"] is not (name in {"train", "prep"}) for item in items):
            raise ValueError("v16b preparation: invalid split training permission")
    if len(splits["prep"]) != 512 or len(splits["train"]) != 3000:
        raise ValueError("v16b preparation: the frozen 512/3000 payload is required")
    preflight = seal(output / "data-contract-preflight.json", preflight_items(
        [item for items in splits.values() for item in items], version=version))
    frozen_role_ids = {role: [item["problem_id"] for item in entries] for role, entries in splits.items()}
    excluded_ceiling_ids: list[str] = []
    if version in MODERN and not local_sequential and version != "error-note-v18b":
        from .guard_v16b import SignalGateV17
        supported = [r["problem_id"] for r in preflight["items"] if r.get("repair_supported")
                     and r["problem_id"] in {item["problem_id"] for item in splits["train"]}]
        order = [item["problem_id"] for item in splits["train"]]
        SignalGateV17([order[i:i + 16] for i in range(0, len(order), 16)], supported)
    if local_sequential:
        # A native human target guarantees that the short functional run can exercise preparation SFT.
        splits["prep"] = sorted(splits["prep"], key=lambda x: (
            not bool(x.get("references", {}).get("human_solution")), digest(x["problem_id"])))[:2]
        splits["monitor"] = splits["monitor"][:2]
        if "reasoning_monitor" in splits:
            splits["reasoning_monitor"] = splits["reasoning_monitor"][:2]
    return seal(output / "selection.json", {"version": preparation_version(version), "local_sequential": local_sequential,
        "data_root": str(data_root.absolute()), "payload_sha256": file_sha(data_root / "payload-receipt.json"),
        "prep_manifest_sha256": file_sha(data_root / "prep-manifest.json"),
        "validation": validation, "data_contract_preflight_sha256": preflight["sha256"], "prep_ids": [x["problem_id"] for x in splits["prep"]],
        "monitor_ids": [x["problem_id"] for x in splits["monitor"]], "human_verified": False,
        **({"role_ids": {role: [x["problem_id"] for x in entries] for role, entries in splits.items()},
            "frozen_role_ids": frozen_role_ids, "excluded_ceiling_monitor_ids": excluded_ceiling_ids,
            "monitor_generation_policy": "full_200_accuracy_8192_and_264_short_warning_64",
            "reasoning_monitor_ids": [x["problem_id"] for x in splits["reasoning_monitor"]]}
           if version == "error-note-v18b" else {})})


def p0(item: dict) -> str:
    from .run_v16 import _p0

    return _p0(item)


def images(item: dict) -> list:
    from PIL import Image

    result = []
    for name, expected in zip(item["image_paths"], item["image_sha256"], strict=True):
        path = Path(name)
        if file_sha(path) != expected:
            raise ValueError("v16b: image identity changed")
        with Image.open(path) as image:
            result.append(image.convert("RGB"))
    return result


def grader(data_root: Path, split: str, *, raw_v8: bool = False):
    from fine_tuning.notebook_rl.verifier import Verifier
    from .single_grading import grade_single

    verifier = Verifier(data_root / "vault" / split)

    def grade(item: dict, output: dict) -> tuple[str, Any]:
        actor = verifier.actors[item["problem_id"]]
        if raw_v8:
            from fine_tuning.notebook_rl.answers import TURN1_EVAL_CONTRACT
            result = verifier.grade(actor.problem_id, output["text"], output["finish_reason"], actor.mode,
                                    answer_contract=TURN1_EVAL_CONTRACT)
            return result.status, result.extracted
        return grade_single(verifier, actor, actor.problem_id, output["text"], output["finish_reason"],
                            actor.mode, units=True)
    return grade


def make_request(item: dict, prompt: str, stage: str, *, cap: int, sampling: dict,
                 version: int = 0, protocol_sha256: str, policy_sha256: str, engine_step: int = 0,
                 evaluation_monitor: bool = False) -> dict:
    from fine_tuning.notebook_rl.prompts import user_message
    from .collect import single_request_sha256

    request = {"request_id": digest({"problem_id": item["problem_id"], "stage": stage}),
        "problem_id": item["problem_id"], "block_id": stage, "stage": stage,
        "seed": sample_seed(item["problem_id"], stage, MASTER_SEED), "max_tokens": cap,
        "messages": [user_message(prompt, len(item["image_paths"]))], "images": images(item),
        "kind": stage, "sampling": dict(sampling), "policy_version": version,
        "policy_sha256": policy_sha256, "protocol_sha256": protocol_sha256, "admit_step": engine_step}
    if evaluation_monitor:
        request.update(evaluation_monitor=True, protocol_version="error-note-v18b")
    request["request_sha256"] = single_request_sha256(request)
    return request


def collect(bundle: Any, sampler: Any, items: list[dict], *, data_root: Path, split: str, out: Path,
            stage: str, protocol_sha256: str, sampling: dict, cap: int, version: int = 0,
            observation: bool = False, algorithm_version: str = "error-note-v16b",
            baseline_receipts: dict[str, dict] | None = None,
            cached_records: list[dict] | None = None, resume_incomplete: bool = False) -> list[dict]:
    """One durable request per parent/stage; ambiguous interrupted requests require recovery."""
    if resume_incomplete and not (algorithm_version == "error-note-v18b"
                                 and split in {"monitor", "reasoning_monitor", "effect"}):
        raise ValueError("only heldout v18b diagnostics may retry an unfinished draw")
    started = time.perf_counter()
    fresh_ids: set[str] = set()
    def measured(result: list[dict]) -> list[dict]:
        elapsed = time.perf_counter() - started
        tokens = sum(len(row["output"]["generation"]["completion_token_ids"]) for row in result
                     if row["problem_id"] in fresh_ids)
        out.mkdir(parents=True, exist_ok=True)
        append_jsonl(out / "collection-timing.jsonl", {"stage": stage, "split": split, "cap": cap,
            "n": len(result), "new_draws": len(fresh_ids), "cached_draws": len(result) - len(fresh_ids),
            "new_output_tokens": tokens, "elapsed_seconds": elapsed,
            "tokens_per_second": tokens / elapsed if elapsed > 0 and fresh_ids else None,
            "algorithm_version": algorithm_version, "sampling": sampling, "reportable": False})
        return result
    from .rl import policy_sha256
    from .v16b_rows import build_obs_prompt_v16b, observation_eligibility_v16b
    observation_prompt: Any = build_obs_prompt_v16b

    if algorithm_version in MODERN:
        from .v17_rows import build_obs_request_v17, observation_eligibility_v17
        observation_eligibility_v16b = observation_eligibility_v17
        if algorithm_version == "error-note-v18":
            from .v18_rows import build_obs_request_v17, observation_eligibility_v17
            observation_eligibility_v16b = observation_eligibility_v17
        elif algorithm_version == "error-note-v18b":
            from .v18b_rows import build_obs_request_v17, observation_eligibility_v17
            observation_eligibility_v16b = observation_eligibility_v17
        def modern_observation_prompt(item: dict) -> str:
            return build_obs_request_v17(item)["prompt"]
        observation_prompt = modern_observation_prompt
    elif algorithm_version != "error-note-v16b":
        raise ValueError("unknown collection version")
    if algorithm_version == "error-note-v18b" and cached_records is not None:
        cached = {row["problem_id"]: row for row in cached_records}
        if len(cached) != len(cached_records) or set(cached) - {item["problem_id"] for item in items}:
            raise ValueError("v18b preparation: duplicate or foreign imported role")
        for item in items:
            row = cached.get(item["problem_id"])
            if row is not None and (row.get("stage") != stage or row.get("prompt") != p0(item)
                    or not row.get("reuse") or row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"})):
                raise ValueError("v18b preparation: imported record changed")
        if not observation and len(cached) == len(items):
            return measured([cached[item["problem_id"]] for item in items])
    if algorithm_version == "error-note-v18b" and observation:
        skipped = [observation_eligibility_v16b(item) for item in items]
        if all(not receipt["eligible"] for receipt in skipped):
            seal(out / "observation-skips.json", {"version": "v18b-preparation-v1", "items": skipped})
            return []
    if baseline_receipts is not None:
        if algorithm_version not in MODERN or set(baseline_receipts) != {item["problem_id"] for item in items}:
            raise ValueError("calibration baseline membership must be sealed before collection")
        for receipt in baseline_receipts.values():
            if receipt.get("sha256") != digest({k: v for k, v in receipt.items() if k != "sha256"}):
                raise ValueError("calibration baseline receipt changed")
    policy = policy_sha256(bundle)
    if hasattr(sampler, "protocol_sha256"):
        sampler.protocol_sha256 = protocol_sha256
    grade = grader(data_root, split, **({"raw_v8": True} if algorithm_version == "error-note-v18b"
                                      and split in {"monitor", "reasoning_monitor"} else {}))
    registry = sampler.live_registry()
    if str(version) not in registry:
        sampler.add_version(bundle, version)
    elif registry[str(version)]["policy_sha256"] != policy:
        raise ValueError("v16b: monitor sampler has a stale policy")
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "requests.jsonl"
    logged = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    records: dict[str, dict] = {}
    cache = {row["problem_id"]: row for row in cached_records or []}
    if (len(cache) != len(cached_records or []) or set(cache) - {item["problem_id"] for item in items}
            or cached_records and (algorithm_version != "error-note-v18b" or observation)):
        raise ValueError("v18b preparation: invalid imported role membership")
    pending: dict[str, tuple[dict, str]] = {}
    eligibility = {item["problem_id"]: observation_eligibility_v16b(item) for item in items} if observation else {}
    if observation:
        seal(out / "observation-skips.json", {"version": VERSION,
            "items": [receipt for receipt in eligibility.values() if not receipt["eligible"]]})
    def finish_pending(keep: int = 0) -> None:
        # One sampler.step() per pass, so a freed slot is refilled at once instead of idling for a full drain.
        while len(pending) > keep:
            if not sampler.has_unfinished_requests():
                raise RuntimeError("v16b: collector lost an admitted request")
            for request, output in sampler.step():
                rid = request["request_id"]
                if rid not in pending:
                    raise ValueError("v16b: unknown/duplicate preparation completion")
                item, prompt = pending.pop(rid)
                output = dict(output, prompt=prompt)
                if sampling["top_p"] < 1 or sampling["top_k"]:
                    output["generation"]["probability_space"] = "processed-truncated-b2"
                if not observation:
                    status, extracted = grade(item, output)
                    output.update(status=status, extracted=extracted)
                body = {"problem_id": item["problem_id"], "stage": stage, "prompt": prompt,
                    "request_sha256": request["request_sha256"], "policy_sha256": policy,
                    "protocol_sha256": protocol_sha256, "output": output}
                stored = seal(out / "generations" / (rid + ".json"), body)
                records[item["problem_id"]] = stored
                append_jsonl(ledger, {"request_id": rid, "status": "COMPLETED", "sha256": stored["sha256"]})

    for item in items:
        if observation and not eligibility[item["problem_id"]]["eligible"]:
            continue
        prompt = observation_prompt(item) if observation else p0(item)
        if item["problem_id"] in cache:
            stored = cache[item["problem_id"]]
            if stored.get("stage") != stage or stored.get("prompt") != prompt or not stored.get("reuse"):
                raise ValueError("v18b preparation: imported raw request changed")
            records[item["problem_id"]] = stored
            continue
        request = make_request(item, prompt, stage, cap=cap, sampling=sampling, version=version,
            protocol_sha256=protocol_sha256, policy_sha256=policy, engine_step=sampler.engine_steps,
            evaluation_monitor=algorithm_version == "error-note-v18b" and stage.startswith("v18b-b2-"))
        if baseline_receipts is not None:
            from .collect import single_request_sha256
            key = "obs_baseline_receipt" if observation else "baseline_receipt"
            request[key] = baseline_receipts[item["problem_id"]]
            request["request_sha256"] = single_request_sha256(request)
        rid = request["request_id"]
        path = out / "generations" / (rid + ".json")
        previous = [row for row in logged if row["request_id"] == rid]
        if previous:
            starts = [row for row in previous if row["status"] == "STARTED"]
            if (resume_incomplete and algorithm_version == "error-note-v18b" and split in {"monitor", "reasoning_monitor", "effect"}
                    and not path.exists() and starts and not any(row["status"] == "COMPLETED" for row in previous)):
                if any(row.get("request_sha256") != request["request_sha256"] for row in starts):
                    raise ValueError("v18b: interrupted diagnostic request identity changed")
                append_jsonl(ledger, {"request_id": rid, "status": "RETRY_INTERRUPTED_DIAGNOSTIC",
                    "request_sha256": request["request_sha256"], "prior_starts": len(starts), "reportable": False})
                previous = []
            elif not path.is_file() or len(starts) != 1 and not resume_incomplete:
                raise ValueError("v16b: STARTED request lacks a durable single result; recovery required")
            if previous:
                stored = read_sealed(path)
                if (not starts or any(row.get("request_sha256") != request["request_sha256"] for row in starts)
                        or stored["request_sha256"] != request["request_sha256"] or stored["policy_sha256"] != policy
                        or stored["protocol_sha256"] != protocol_sha256
                        or any(row["sha256"] != stored["sha256"] for row in previous if row["status"] == "COMPLETED")):
                    raise ValueError("v16b: stored preparation request identity changed")
                records[item["problem_id"]] = stored
                continue
        if path.exists():
            raise ValueError("v16b: result has no request ownership ledger")
        append_jsonl(ledger, {"request_id": rid, "status": "STARTED",
                              "request_sha256": request["request_sha256"]})
        if algorithm_version in MODERN and hasattr(sampler, "prepare_request_v17"):
            prepared = sampler.prepare_request_v17(request)
            while not sampler.can_submit_v17(prepared):
                if not pending:
                    raise ValueError("v17 collection: request cannot fit the actual KV reservation")
                finish_pending(keep=len(pending) - 1)
            sampler.submit(request, prepared=prepared)
        else:
            sampler.submit(request)
        fresh_ids.add(item["problem_id"])
        pending[rid] = (item, prompt)
        # Bound host image memory as well as the engine's active request count: keep one slot free.
        slots = getattr(sampler, "max_num_seqs", 1)
        if len(pending) >= slots:
            finish_pending(keep=slots - 1)
    finish_pending()
    return measured([records[item["problem_id"]] for item in items if item["problem_id"] in records])




def summary(records: list[dict]) -> dict:
    lengths = [len(row["output"]["generation"]["completion_token_ids"]) for row in records]
    return {"count": len(records), "correct": sum(row["output"].get("status") == "CORRECT" for row in records),
        "missing": sum(row["output"].get("extracted") is None for row in records),
        "length": sum(row["output"]["finish_reason"] == "length" for row in records),
        "median_tokens": statistics.median(lengths) if lengths else None}


def generate(data_root: Path, output: Path, bundle: Any, sampler: Any, *, image: dict,
             local_sequential: bool = False, version: str = "error-note-v16b",
             reuse_records: dict[str, list[dict]] | None = None, phase: str = "all") -> dict:
    from .protocol import EVAL_SAMPLING_V16B, TRAIN_SAMPLING_V16B

    data = data_api(version)
    selection = select(data_root, output, local_sequential=local_sequential, version=version)
    is_v17 = version in MODERN
    context = {"version": preparation_version(version),
               "selection_sha256": selection["sha256"],
               "image": image, "cap": 256 if local_sequential else 8192, "initial_policy": "fresh_F0"}
    contract = digest(context)
    if phase not in {"all", "legacy", "monitor"} or phase != "all" and version != "error-note-v18b":
        raise ValueError("v18b preparation: explicit generation phase required")
    splits = {name: {item["problem_id"]: item for item in data.load_items(data_root, name)}
              for name in ("prep", "monitor")}
    preps = [splits["prep"][pid] for pid in selection["prep_ids"]]
    if version == "error-note-v18b":
        from .v18b_rows import bind_graph_observation_targets_v18b
        preps = bind_graph_observation_targets_v18b(preps)
    monitors = [splits["monitor"][pid] for pid in selection["monitor_ids"]]
    kwargs = dict(data_root=data_root, protocol_sha256=contract, cap=context["cap"])
    if is_v17:
        kwargs["algorithm_version"] = version
    reused = reuse_records or {}
    result = dict(context)
    if phase in {"all", "legacy"}:
        records = collect(bundle, sampler, preps, split="prep", out=output / "prep", stage=preparation_stage(version, "prep"),
            sampling=EVAL_SAMPLING_V16B, **kwargs, cached_records=reused.get("prep"))
        obs = collect(bundle, sampler, preps, split="prep", out=output / "prep-obs", stage=preparation_stage(version, "prep-obs"),
            sampling=TRAIN_SAMPLING_V16B, observation=True, **(kwargs | {"cap": 256 if local_sequential else 1024}))
        result.update(prep=records, obs=obs)
    if phase in {"all", "monitor"}:
        monitor = collect(bundle, sampler, monitors, split="monitor", out=output / "monitor-f0",
            stage=preparation_stage(version, "monitor"), sampling=EVAL_SAMPLING_V16B,
            **kwargs)
        result.update(monitor=monitor, monitor_summary=summary(monitor))
        if version == "error-note-v18b":
            short = collect(bundle, sampler, monitors, split="monitor", out=output / "short-monitor-f0",
                stage=preparation_stage(version, "short_monitor"), sampling=EVAL_SAMPLING_V16B,
                **(kwargs | {"cap": 64}))
            result.update(short_monitor=short, short_monitor_summary=summary(short))
    if is_v17 and phase in {"all", "legacy"}:
        effects = data.load_items(data_root, "effect")
        if "effect_ids" in selection:
            effects_by_id = {item["problem_id"]: item for item in effects}
            effects = [effects_by_id[pid] for pid in selection["effect_ids"]]
        elif local_sequential:
            effects = effects[:2]
        effect = collect(bundle, sampler, effects, split="effect", out=output / "effect-f0",
                         stage=preparation_stage(version, "effect"), sampling=EVAL_SAMPLING_V16B, **kwargs,
                         cached_records=reused.get("effect"))
        result.update(effect=effect, effect_summary=summary(effect))
    if version == "error-note-v18b" and phase in {"all", "monitor"}:
        panel = {item["problem_id"]: item for item in data.load_items(data_root, "reasoning_monitor")}
        records = collect(bundle, sampler, [panel[pid] for pid in selection["reasoning_monitor_ids"]],
            split="reasoning_monitor", out=output / "reasoning-monitor-f0",
            stage=preparation_stage(version, "reasoning_monitor"), sampling=EVAL_SAMPLING_V16B,
            **(kwargs | {"cap": 64}))
        result.update(reasoning_monitor=records, reasoning_monitor_summary=summary(records))
    if reuse_records is not None:
        result["raw_reuse_receipt_sha256"] = read_sealed(output / "f0-reuse.json")["sha256"]
    return seal(output / ("f0-generations.json" if phase == "all" else f"f0-{phase}-generations.json"), result)


def rows(data_root: Path, output: Path, tokenizer: Any, *, local_sequential: bool = False,
         version: str = "error-note-v16b") -> dict:
    load_items = data_api(version).load_items
    from .v16b_rows import obs_score, prep_prior_v16b, prep_rows

    selection = select(data_root, output, local_sequential=local_sequential, version=version)
    generated = read_sealed(output / "f0-generations.json")
    if generated["selection_sha256"] != selection["sha256"]:
        raise ValueError("v16b: F0 preparation selection changed")
    prep_items = load_items(data_root, "prep")
    if version == "error-note-v18b":
        from .v18b_rows import bind_graph_observation_targets_v18b
        prep_items = bind_graph_observation_targets_v18b([item for item in prep_items if item["problem_id"] in selection["prep_ids"]])
    items = {item["problem_id"]: item for item in prep_items}
    records = [dict(items[row["problem_id"]], p0_text=row["prompt"], t1=row["output"])
               for row in generated["prep"]]
    if [row["problem_id"] for row in records] != selection["prep_ids"]:
        raise ValueError("v16b: incomplete F0 preparation")
    if version in MODERN:
        from .v17_rows import prep_rows_with_receipts_v17
        if version == "error-note-v18":
            from .v18_rows import prep_rows_with_receipts_v17
        elif version == "error-note-v18b":
            from .v18b_rows import prep_rows_with_receipts_v17
        selected, receipts = prep_rows_with_receipts_v17(records, tokenizer=tokenizer)
        seal(output / "prep-admission.json", {"version": version.removeprefix("error-note-") + "-prep-admission-v1",
             "selection_sha256": selection["sha256"], "receipts": receipts})
    else:
        selected = prep_rows(records, tokenizer=tokenizer)
    if not selected:
        raise ValueError("v16b: preparation has no admissible SFT rows")
    batches = [selected[index:index + 16] for index in range(0, len(selected), 16)]
    sealed = seal(output / "prep-rows.json", {"version": preparation_version(version), "selection_sha256": selection["sha256"],
        "f0_generations_sha256": generated["sha256"], "rows": selected, "batches": batches,
        "prep_updates": len(batches), "local_sequential": local_sequential})
    prior = prep_prior_v16b(records)
    scores: dict[str, list[float]] = defaultdict(list)
    for row in generated["obs"]:
        item, out = items[row["problem_id"]], row["output"]
        if version in MODERN:
            from .v17_rows import audit_observation_v17
            from .v17_data import private_verifier
            if version == "error-note-v18":
                from .v18_rows import audit_observation_v17
                from .v18_data import private_verifier
            elif version == "error-note-v18b":
                from .v18b_rows import audit_observation_v17
                from .v18b_data import private_verifier
            receipt = audit_observation_v17(dict(item, obs=out),
                verifier=private_verifier(item), tokenizer=tokenizer)
            score = {"score": receipt["reward"]}
        else:
            score = obs_score(item, item["layer"], out["text"], finish_reason=out["finish_reason"])
        if version not in {"error-note-v18", "error-note-v18b"} and score.get("score") is not None:
            scores[item["layer"]].append(score["score"])
    seal(output / "baseline-init.json", {"version": preparation_version(version), "prep_rows_sha256": sealed["sha256"],
        "source_rates": prior["source_rates"], "overall": prior["overall"],
        "layer_rates": {layer: sum(values) / len(values) for layer, values in scores.items()},
        "missing_obs_policy": "binary-task-beta-prior-0.5" if version in {"error-note-v18", "error-note-v18b"} else "first-admission-batch-zero-advantage",
        **({"obs_task": "v18-single-fact-observation-v1"} if version in {"error-note-v18", "error-note-v18b"} else {}),
        "prep_prior": prior})
    return sealed


def eta_report(block_seconds: list[float], *, done: int, total: int) -> dict:
    if (type(done) is not int or type(total) is not int or not 0 <= done <= total or not block_seconds
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                   for x in block_seconds)):
        raise ValueError("v16b: valid observed block times and fixed progress are required")
    mean = sum(block_seconds) / len(block_seconds)
    return {"version": "v16b-eta-v1", "done": done, "total": total, "mean_block_seconds": mean,
            "remaining_seconds": mean * (total - done), "changes_plan": False}


def _generate_in_phases(args: Any, argv: list[str], reuse: dict | None) -> dict:
    """A process boundary releases the 24K engine before constructing the 40K B2 engine."""
    import subprocess
    import sys

    extra = []
    if reuse and reuse.get("sha256"):
        extra = ["--f0-import-receipt-sha256", file_sha(args.output / "f0-reuse.json")]
    for phase in ("legacy", "monitor"):
        subprocess.run([sys.executable, "-m", "fine_tuning.error_note.prep_v16b", *argv,
                        "--generation-phase", phase, *extra], check=True)
    legacy = read_sealed(args.output / "f0-legacy-generations.json")
    monitor = read_sealed(args.output / "f0-monitor-generations.json")
    if any(legacy.get(key) != monitor.get(key) for key in ("version", "selection_sha256", "image", "cap", "initial_policy")):
        raise ValueError("v18b preparation: generation phases belong to different selections")
    return seal(args.output / "f0-generations.json", {k: v for k, v in (legacy | monitor).items() if k != "sha256"})


def _legacy_cache_complete(args: Any, records: dict[str, list[dict]]) -> bool:
    from .v18b_rows import observation_eligibility_v17, bind_graph_observation_targets_v18b

    data = data_api(args.version)
    selected = select(args.data_root, args.output, local_sequential=args.local_sequential, version=args.version)
    preps = bind_graph_observation_targets_v18b([x for x in data.load_items(args.data_root, "prep") if x["problem_id"] in selected["prep_ids"]])
    effects = data.load_items(args.data_root, "effect")[:2] if args.local_sequential else data.load_items(args.data_root, "effect")
    return (all({x["problem_id"] for x in records.get(role, [])} == {x["problem_id"] for x in items}
                for role, items in (("prep", preps), ("effect", effects)))
            and all(not observation_eligibility_v17(item)["eligible"] for item in preps))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "generate", "rows", "index-f0", "reuse-f0", "rescore-f0"))
    parser.add_argument("--version", choices=("error-note-v16b", "error-note-v17", "error-note-v18", "error-note-v18b"), default="error-note-v16b")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--precision", choices=("nf4", "bf16"), default="bf16")
    parser.add_argument("--engine", choices=("hf", "vllm"), default="vllm")
    parser.add_argument("--nf4-cache", type=Path)
    parser.add_argument("--local-sequential", action="store_true")
    parser.add_argument("--allow-model-run", action="store_true")
    parser.add_argument("--allow-f0-raw-reuse", action="store_true")
    parser.add_argument("--f0-reuse-source", type=Path)
    parser.add_argument("--f0-reuse-sha256")
    parser.add_argument("--f0-reuse-evidence", type=Path)
    parser.add_argument("--f0-reuse-evidence-sha256")
    parser.add_argument("--f0-producer-plan", type=Path)
    parser.add_argument("--f0-producer-plan-sha256")
    parser.add_argument("--f0-producer-source-root", type=Path)
    parser.add_argument("--f0-model-manifest", type=Path)
    parser.add_argument("--f0-model-manifest-sha256")
    parser.add_argument("--f0-producer-doctor", type=Path)
    parser.add_argument("--f0-producer-doctor-sha256")
    parser.add_argument("--f0-producer-capacity", type=Path)
    parser.add_argument("--f0-producer-capacity-sha256")
    parser.add_argument("--generation-phase", choices=("all", "legacy", "monitor"), default="all")
    parser.add_argument("--f0-import-receipt-sha256")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--kv-cache-gib", type=float, default=16.)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.65)
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(actual_argv)
    if args.command == "select":
        result = select(args.data_root, args.output, local_sequential=args.local_sequential, version=args.version)
    elif args.command in {"index-f0", "reuse-f0", "rescore-f0"}:
        from .prep_reuse_v18b import index_f0_source, rescore_f0, reuse_f0
        if args.command == "index-f0":
            result = index_f0_source(args)
        elif args.command == "reuse-f0":
            result = reuse_f0(args)
        else:
            result = rescore_f0(args)
    elif args.command == "rows":
        from transformers import AutoProcessor
        if args.model_path is None:
            raise ValueError("v16b preparation: --model-path is required")
        processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
        result = rows(args.data_root, args.output, processor.tokenizer, local_sequential=args.local_sequential, version=args.version)
    else:
        reuse = None
        if args.version == "error-note-v18b":
            if args.f0_import_receipt_sha256:
                from . import single
                path = args.output / "f0-reuse.json"
                if file_sha(path) != args.f0_import_receipt_sha256:
                    raise ValueError("v18b preparation: imported receipt bytes changed")
                reuse = read_sealed(path)
                selected = select(args.data_root, args.output, local_sequential=args.local_sequential, version=args.version)
                current = single.source_manifest()
                if (reuse.get("selection_sha256") != selected["sha256"]
                        or reuse.get("validator_commit") != current["base_commit"]
                        or reuse.get("validator_source_sha256") != digest(current["actual_files"])):
                    raise ValueError("v18b preparation: imported receipt selection/validator changed")
            else:
                from .prep_reuse_v18b import reuse_f0
                reuse = reuse_f0(args)
            if args.generation_phase == "all":
                if not args.allow_model_run:
                    raise ValueError("v18b preparation: missing B2 draws need --allow-model-run")
                result = _generate_in_phases(args, actual_argv, reuse)
                print(json.dumps({"status": "PREPARED", "sha256": result["sha256"], "command": args.command}))
                return 0
            if (args.generation_phase == "legacy" and reuse and reuse.get("sha256")
                    and _legacy_cache_complete(args, reuse["records"])):
                from .protocol import SINGLE_IMAGE_V14
                result = generate(args.data_root, args.output, None, None, image=dict(SINGLE_IMAGE_V14),
                    local_sequential=args.local_sequential, version=args.version,
                    reuse_records=reuse["records"], phase="legacy")
                print(json.dumps({"status": "PREPARED_CPU_REUSE", "sha256": result["sha256"], "command": args.command}))
                return 0
        if not args.allow_model_run or args.model_path is None:
            raise ValueError("v16b preparation: explicit --allow-model-run and --model-path are required")
        select(args.data_root, args.output, local_sequential=args.local_sequential, version=args.version)
        from .campaign_v16b import load_f0, make_sampler
        from .protocol import SINGLE_IMAGE_V14, SMOKE_IMAGE_V16B, TRAIN_SAMPLING_V16B
        from .run_v16 import _setup_execution
        _setup_execution(MASTER_SEED)
        image = dict(SMOKE_IMAGE_V16B if args.local_sequential else SINGLE_IMAGE_V14)
        args._f0_legacy_generation = args.version == "error-note-v18b" and args.generation_phase == "legacy"
        if args._f0_legacy_generation and reuse and reuse.get("sha256"):
            args.kv_cache_gib = reuse["engine_settings"]["kv_cache_gib"]
            args.gpu_memory_utilization = reuse["engine_settings"]["gpu_memory_utilization"]
        bundle, protocol = load_f0(args, image=image)
        sampler = make_sampler(bundle, args, protocol, image=image, sampling=TRAIN_SAMPLING_V16B)
        if not args.local_sequential:  # fail fast: the two-version gate on identical weights, minutes into the pod
            sampler.add_version(bundle, 0)
            sampler.add_version(bundle, 1)
        try:
            result = generate(args.data_root, args.output, bundle, sampler, image=image,
                              local_sequential=args.local_sequential, version=args.version,
                              **({"reuse_records": reuse["records"]} if reuse and reuse.get("sha256") else {}),
                              **({"phase": args.generation_phase} if args.version == "error-note-v18b" else {}))
        finally:
            sampler.release()
    print(json.dumps({"status": result.get("status", "PREPARED"), "sha256": result.get("sha256"), "command": args.command}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
