"""Replay saved native generations through v18 grading and row admission, without generation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from typing import Any, Callable

from .audit import digest
from .prep_v16b import file_sha, grader, p0, seal


def read_archive(path: Path, *, allow_partial: bool = False) -> tuple[dict, dict[str, dict]]:
    """Read only JSON members; never extract an archive or follow its links."""
    plan, records = None, {}

    def accept(name: str, raw: bytes) -> None:
        nonlocal plan
        if name.endswith("/campaign/campaign-plan.json") or name == "campaign-plan.json":
            if plan is not None:
                raise ValueError("v18 replay: archive contains multiple campaign plans")
            plan = json.loads(raw)
        elif "/campaign/runtime/items/" in name or name.startswith("runtime/items/"):
            row = json.loads(raw)
            pid = row["problem_id"]
            if pid in records:
                raise ValueError("v18 replay: duplicate native parent record")
            records[pid] = row

    if path.is_dir():
        roots = [path] if (path / "campaign-plan.json").is_file() else [p.parent for p in path.rglob("campaign-plan.json")]
        if len(roots) != 1:
            raise ValueError("v18 replay: one campaign root required")
        root = roots[0]
        accept("campaign-plan.json", (root / "campaign-plan.json").read_bytes())
        for member_path in sorted((root / "runtime" / "items").glob("*.json")):
            if member_path.is_symlink():
                raise ValueError("v18 replay: symlink record forbidden")
            accept("runtime/items/" + member_path.name, member_path.read_bytes())
    else:
        process = subprocess.Popen(["zstd", "-dc", "--", str(path)], stdout=subprocess.PIPE) if path.name.endswith(".zst") else None
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") if process else tarfile.open(path, mode="r|*") as archive:
                for member in archive:
                    selected = (member.name.endswith("/campaign/campaign-plan.json")
                                or "/campaign/runtime/items/" in member.name and member.name.endswith(".json"))
                    if not selected:
                        continue
                    if not member.isfile() or ".." in PurePosixPath(member.name).parts or member.size > 64 * 1024 * 1024:
                        raise ValueError("v18 replay: unsafe or oversized JSON member")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("v18 replay: missing JSON member")
                    accept(member.name, stream.read())
        finally:
            if process is not None:
                assert process.stdout is not None
                process.stdout.close()
                process.wait()
        if process is not None and process.returncode != 0:
            raise ValueError("v18 replay: zstd stream failed")
    if plan is None or not records:
        raise ValueError("v18 replay: campaign plan and native item records required")
    if plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("v18 replay: original plan seal changed")
    ids = [item["problem_id"] for item in plan["items"]]
    expected = set(ids[:len(records)]) if allow_partial else set(ids)
    if len(ids) != len(set(ids)) or expected != set(records):
        raise ValueError("v18 replay: native records do not match the frozen campaign")
    return plan, records


def replay_parent(item: dict, native: dict, *, tokenizer: Any,
                  grade: Callable[[dict, dict], tuple[str, Any]],
                  grade_v8: Callable[[dict, dict], tuple[str, Any]] | None = None,
                  version: str = "v18") -> tuple[dict, list[dict]]:
    """Use the same v18 policy, verifier, admission and sampled-token row builders as campaign."""
    if version not in {"v18", "v18b"}:
        raise ValueError("Unknown replay version")
    if version == "v18b":
        from . import v18b_rows
        rows: Any = v18b_rows
    else:
        from . import v18_rows
        rows = v18_rows

    native_sha256 = digest(native)
    outputs = {key: deepcopy(native[key]) for key in ("t1", "obs", "note", "resolve") if key in native}
    record = deepcopy(item)
    record.update(p0_text=p0(item), subject=item.get("subject", item["source"]), t1=outputs["t1"])
    if version == "v18b" and "obs" in outputs:
        record["obs"] = outputs["obs"]
        try:
            record = rows.bind_historical_observation_v18b(record)
        except ValueError:
            pass  # Legacy v17 OBS remains ungraded under the new input contract.
    contract = rows.problem_contract_v17(item)
    kwargs = {"verifier": contract["verifier"], "tokenizer": tokenizer,
              "fact_catalog": contract["fact_catalog"], "rules": contract["rules"]}
    gold, choices = item["evaluation"]["gold"], item["evaluation"]["choices"]
    before = {key: digest(value) for key, value in outputs.items()}
    old_status = record["t1"].get("status")
    def regrade(stage: str) -> None:
        status, extracted = grade(item, record[stage])
        record[stage].update(status=status, extracted=extracted)
        if grade_v8 is not None:
            raw_status, raw_extracted = grade_v8(item, record[stage])
            record[stage].update(v8_status=raw_status, v8_extracted=raw_extracted)
    regrade("t1")
    policy = rows.t1_policy_receipt_v18(record, gold=gold, choices=choices, **kwargs)
    record["t1_policy_receipt"] = policy
    record["t1_ledger"] = rows.audit_t1_v17(record, **kwargs)
    record["a1_receipt"] = rows.verify_solution_v17(record, gold=gold, choices=choices, **kwargs)
    ledgers = [record["t1_ledger"]]
    if "obs" in outputs:
        record["obs"] = outputs["obs"]
        observation = rows.audit_observation_v17(record, verifier=contract["verifier"], tokenizer=tokenizer)
        record.update(obs_score=observation, obs_ledger=observation["ledger"])
        ledgers.append(observation["ledger"])
    result: dict[str, Any] = {"problem_id": item["problem_id"], "source": item["source"],
        "answer_type": item["evaluation"]["answer_type"], "old_status": old_status,
        "replayed_status": record["t1"]["status"], "v8_status": record["t1"].get("v8_status"), "policy": policy,
        "historical_contradictions": sum(c.get("verdict") == "CONTRADICTED"
            for key in ("t1_ledger", "obs_ledger") for c in (native.get(key) or {}).get("claims", [])),
        "historical_verified_repairs": (native.get("note_receipt") or {}).get("verified_repair_count", 0),
        "a1_receipt": record["a1_receipt"], "native_output_sha256": before,
        "claims": {k: sum(c.get("verdict") == k for ledger in ledgers for c in ledger.get("claims", []))
                   for k in ("VERIFIED", "CONTRADICTED", "UNKNOWN")},
        "supported": bool(contract.get("supported")), "historical_note_protocol": True}
    admitted = []
    a1 = rows.a1_row_v17(record, tokenizer=tokenizer, receipt=record["a1_receipt"])
    if a1 is not None:
        admitted.append(a1)
    try:
        request = rows.build_note_request_v17(record, ledgers=ledgers, gold=gold, tokenizer=tokenizer)
        result["note_requested"] = request is not None
        result["new_note_prompt_sha256"] = digest(request["prompt"]) if request else None
    except ValueError as exc:
        result.update(note_requested=False, note_skip=str(exc))
    if "note" in outputs:
        record["note"] = outputs["note"]
        receipt = rows.verify_note_v17(record, ledgers=ledgers, gold=gold, choices=choices, **kwargs)
        record["note_receipt"] = receipt
        result["note_receipt"] = receipt
        nrow = rows.n_row_v17(record, tokenizer=tokenizer, receipt=receipt)
        if nrow is not None:
            admitted.append(nrow)
        if "resolve" in outputs and record["note_receipt"].get("resolve_allowed"):
            record["resolve"] = outputs["resolve"]
            regrade("resolve")
            receipt = rows.verify_solution_v17(record, stage="resolve", gold=gold, choices=choices,
                note_receipt=record["note_receipt"], **kwargs)
            result["resolve_receipt"] = receipt
            crow = rows.c_row_v17(record, tokenizer=tokenizer, receipt=receipt)
            if crow is not None:
                admitted.append(crow)
        elif "resolve" in outputs:
            result["resolve_receipt"] = {"accepted": False, "reasons": ["NOT_SCHEDULED_UNDER_V18"],
                                         "native_output_retained": True}
    result["row_counts"] = dict(Counter(row["kind"] for row in admitted))
    # Regrading is local metadata. Never rewrite the sampled text, token IDs, prompt or archived receipts.
    result["native_archive_unchanged"] = digest(native) == native_sha256
    if not result["native_archive_unchanged"]:
        raise ValueError("Replay altered original native records")
    result["replay_version"] = version
    result["generated_answers"] = 0
    result["gpu_updates"] = 0
    return result, admitted


def replay(archive: Path, data_root: Path, tokenizer: Any, *, output: Path,
           limit: int | None = None, parent_ids: list[str] | None = None, version: str = "v18",
           allow_partial: bool = False) -> dict:
    source_files: tuple[str, ...] = ("v18_replay.py", "v18_rows.py", "v17_rows.py", "v16b_rows.py", "single_grading.py", "prep_v16b.py")
    if version == "v18b":
        source_files += ("v18b_rows.py", "v18b_claims.py")
    source_before = {name: file_sha(Path(__file__).with_name(name)) for name in source_files}
    plan, native = read_archive(archive, allow_partial=allow_partial)
    ids = [item["problem_id"] for item in plan["items"] if item["problem_id"] in native]
    if parent_ids is not None:
        if len(parent_ids) != len(set(parent_ids)) or not set(parent_ids) <= set(ids):
            raise ValueError("v18 replay: requested IDs must belong to the original campaign")
        ids = [pid for pid in ids if pid in parent_ids]
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("v18 replay: positive explicit limit required")
        ids = ids[:limit]
    by_id, grade = {item["problem_id"]: item for item in plan["items"]}, grader(data_root, "train")
    grade_v8 = grader(data_root, "train", raw_v8=True)
    from fine_tuning.notebook_rl.vault import read_actors, read_verifier
    actors = {actor.problem_id: actor for actor in read_actors(data_root / "vault" / "train")}
    private = read_verifier(data_root / "vault" / "train")
    for pid in ids:
        item, spec, actor, truth = by_id[pid], by_id[pid]["evaluation"], actors[pid], private[pid]
        if (truth.gold != spec["gold"] or truth.answer_type != spec["answer_type"]
                or list(truth.choices) != list(spec["choices"]) or actor.question != spec["question"]
                or list(actor.image_sha256) != list(item["image_sha256"])):
            raise ValueError("v18 replay: grading vault differs from the original native parent")
    counts: dict[str, Counter] = defaultdict(Counter)
    rejections: dict[str, Counter] = defaultdict(Counter)
    details = []
    for pid in ids:
        item = dict(by_id[pid], block_index=next(i for i, block in enumerate(plan["blocks"]) if pid in block))
        result, rows = replay_parent(item, native[pid], tokenizer=tokenizer, grade=grade, grade_v8=grade_v8, version=version)
        details.append(result)
        for key in ("ALL", item["source"] + ":" + item["evaluation"]["answer_type"]):
            counts[key].update(parents=1, supported=int(result["supported"]), A1=0, N=0, C=0,
                format_certified=int(result["policy"]["reason"] == "FORMAT_ONLY_CERTIFIED"),
                pg_excluded=int(not result["policy"]["pg"]),
                attributable_pg_tokens=sum(result["policy"]["loss_token_mask"]),
                native_notes=int("note" in native[pid]),
                historical_contradicted_parents=int(result["historical_contradictions"] > 0),
                contradictions=result["claims"]["CONTRADICTED"], historical_contradictions=result["historical_contradictions"],
                v8_training_disagreement=int(result["v8_status"] != result["replayed_status"]),
                regraded=int(result["old_status"] != result["replayed_status"]))
            counts[key].update(result["row_counts"])
        for stage in ("a1", "note", "resolve"):
            receipt = result.get(stage + "_receipt") or {}
            rejections[stage].update(receipt.get("reasons", []))
        seal(output / "parents" / (digest(pid) + ".json"), result)
    if source_before != {name: file_sha(Path(__file__).with_name(name)) for name in source_files}:
        raise ValueError("v18 replay: source changed during replay; partial records retained, summary unsealed")
    tokenizer_root = Path(getattr(tokenizer, "name_or_path", ""))
    tokenizer_files = {name: file_sha(tokenizer_root / name) for name in ("tokenizer.json", "tokenizer_config.json")
                       if (tokenizer_root / name).is_file()}
    return seal(output / "replay-summary.json", {"version": "v18-native-replay-v1",
        "source_files": source_before, "source_sha256": digest(source_before), "tokenizer_files": tokenizer_files,
        "archive_sha256": file_sha(archive) if archive.is_file() else digest({pid: native[pid] for pid in ids}),
        "plan_sha256": plan["sha256"],
        "grading_vault_files": {name: file_sha(data_root / "vault" / "train" / name)
                                for name in ("actor_inputs.jsonl", "verifier_private.jsonl", "manifest.yaml")},
        "counts": {k: dict(v) for k, v in sorted(counts.items())},
        "rejections_by_stage": {k: dict(v) for k, v in rejections.items()},
        "selected_parent_ids": ids, "full_campaign": len(ids) == len(plan["items"]),
        "replay_version": version, "available_native_records": len(native), "planned_parents": len(plan["items"]),
        "consumption_status": "NOT_INFERRED_FROM_GENERATION",
        "records_sha256": digest(details), "human_verified": False, "generated_answers": 0,
        "gpu_updates": 0, "effect_measured": False,
        "note_replay_limit": "v17 note prompts are retained; v18 binding rejection is reported, not repaired"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parent-id", action="append")
    parser.add_argument("--version", choices=("v18", "v18b"), default="v18")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), local_files_only=True)
    result = replay(args.archive, args.data_root, tokenizer, output=args.output,
                    limit=args.limit, parent_ids=args.parent_id, version=args.version, allow_partial=args.allow_partial)
    print(json.dumps({"sha256": result["sha256"], "full_campaign": result["full_campaign"], "counts": result["counts"].get("ALL", {})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
