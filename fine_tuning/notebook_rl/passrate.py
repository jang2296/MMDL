"""Pass-rate pre-filter and hard-negative re-selection; measurement only, never training approval.

A group whose first attempts are all right or all wrong has zero reward variance and gives RL
no signal. `measure` samples k first attempts per item exactly like the repair collector's T1
call, `select` keeps items with some but not all passes, and `hard_negatives` keeps the items
a trained policy still fails. Gold stays in the verifier; benchmark lineage is refused by the vault.

Every report binds the weights it measured: `weights` = {"checkpoint": null, "trainable_sha256": null} for F0, or the
checkpoint path and the sha256 of the loaded trainable tensors (summary.json, hence the selection's report hash).
A checkpoint-measured report can only feed `hard-negatives`; local_data.propose_proof_rule refuses it as a
training selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
import traceback
from collections import Counter
from contextlib import ExitStack
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterator

from . import prompts as prompt_builders
from .dataset import SPLITS, _bytes, _sha, _write, load_images
from .engines import RolloutRequest, RolloutResult, _completion
from .local_data import project_observation_actors
from .local_profile import DEFAULT_NOTE_CONTRACT, NOTE_CONTRACT_CHOICES, build_protocol, local_prompts
from .rollout import BLOCKING, RolloutConfig, _hash, seeds
from .vault import _read, _unique, read_actors, read_lineage
from .verifier import Verifier

VERSION = "notebook-rl-passrate-v1"
RULES_VERSION = "passrate-rules-v1"
SELECTION_STATUS = "SELECTION_FROZEN_NOT_TRAINING_APPROVAL"
MASTER_SEED = 3407
K_RANGE = (4, 8)
# Training seed branches are "T1:root", "T2:<sample id>", ...; none can start with this prefix.
SEED_BRANCH = "PASSRATE"
REASONS = ("ALL_CORRECT", "ALL_WRONG", "BLOCKING", "OUT_OF_RANGE")
_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_RULE_KEYS = {"version", "min_pass", "max_pass", "max_blocking"}
F0_WEIGHTS = {"checkpoint": None, "trainable_sha256": None}


def first_attempt_request(actor, images: list, prompts: dict, *, policy_version: int, k: int,
                          max_new_tokens: int, seed_namespace: str) -> RolloutRequest:
    """The repair collector's T1 request (same prompt path), with k namespaced seeds."""
    prompts = copy.deepcopy(prompt_builders.for_actor(actor, prompts))
    branch = f"{SEED_BRANCH}:{seed_namespace}"
    return RolloutRequest(
        f"{actor.problem_id}:v{policy_version}:{branch}", None,
        [prompt_builders.user_message(prompt_builders.turn1_text(actor, prompts), len(images))],
        images, k, max_new_tokens,
        seeds(actor.problem_id, policy_version, 1, branch, k, master_seed=MASTER_SEED))


def _sample_batch(requests: list[RolloutRequest], engine: Any, policy_version: int,
                  context_limit: int) -> list[RolloutResult | None]:
    """rollout._sample_request's checks for several requests sampled in ONE engine.sample call.

    Each request keeps its own messages, images, n, cap and seeds; None marks a CONTEXT_BLOCKED request,
    which is never sent to the sampler. One request is exactly `_sample_request` (without its hooks)."""
    prepared: dict[str, Any] = {}
    for request in requests:
        try:
            item = engine.prepare(request)
            if len(item.token_ids) + request.max_new_tokens > context_limit:
                raise ValueError("CONTEXT_BLOCKED: prompt plus completion reservation")
        except ValueError as exc:
            if not str(exc).startswith("CONTEXT_BLOCKED:"):
                raise
            continue
        prepared[request.request_id] = item
    live = [r for r in requests if r.request_id in prepared]
    if not live:
        return [None] * len(requests)
    if engine.policy_version != policy_version:
        raise RuntimeError("passrate: policy changed before sampling")
    outputs = engine.sample(live)
    if (len(outputs) != len(live) or engine.policy_version != policy_version
            or any(out.request_id != r.request_id or len(out.completions) != r.n
                   or out.prompt_token_ids != prepared[r.request_id].token_ids
                   or r.prompt_token_ids is not None and out.prompt_token_ids != r.prompt_token_ids
                   for r, out in zip(live, outputs))):
        raise RuntimeError("passrate: sampler altered the actual state, version or group size")
    by_id = {r.request_id: out for r, out in zip(live, outputs)}
    return [by_id.get(r.request_id) for r in requests]


def _measure_items(actors: list, images: list[list], prompts: dict, engine: Any, verifier: Any, split: str, *,
                   policy_version: int, k: int, cfg: RolloutConfig, seed_namespace: str) -> Iterator[dict]:
    """The first-attempt requests of all `actors` in one engine.sample call; rows graded and yielded in order.

    A row's `seconds` runs from its batch's start to the end of its own grading (one item: the item's time)."""
    from .notebook_synthesis import grade_options

    began = time.monotonic()
    plans = []
    for actor, item_images in zip(actors, images, strict=True):
        # The collector's T1 (rollout.build_tree): a bound eval-aligned contract sets its own cap and V8 grading.
        turn1 = prompt_builders.turn1_contract(actor, prompts)
        request = first_attempt_request(actor, item_images, prompts, policy_version=policy_version, k=k,
            max_new_tokens=cfg.max_new_tokens[1] if turn1 is None else turn1["max_new_tokens"],
            seed_namespace=seed_namespace)
        plans.append((actor, request, grade_options(1, None if turn1 is None else turn1["version"])))
    outputs = _sample_batch([request for _, request, _ in plans], engine, policy_version, cfg.context_limit)
    for (actor, request, grading), output in zip(plans, outputs, strict=True):
        statuses: list[str] = []
        finishes: list[str] = []
        tokens: list[int] = []
        raw: list[str] = []
        prompt_tokens = prompt_sha = None
        if output is None:
            statuses = ["CONTEXT_BLOCKED"] * k
        else:
            prompt_tokens, prompt_sha = len(output.prompt_token_ids), _hash(output.prompt_token_ids)
            for completion in output.completions:
                _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
                # Same call the collector's native audit makes for a T1 sample.
                statuses.append(verifier.grade(actor.problem_id, completion.text, completion.finish_reason,
                                               actor.mode, **grading).status)
                finishes.append(completion.finish_reason)
                tokens.append(len(completion.token_ids))
                raw.append(hashlib.sha256(completion.text.encode("utf-8")).hexdigest())
        yield {"problem_id": actor.problem_id, "family": actor.family, "source": actor.source, "split": split,
               "policy_version": policy_version, "k": k, "passes": statuses.count("CORRECT"),
               "blocking": sum(status in BLOCKING for status in statuses),
               "statuses": dict(sorted(Counter(statuses).items())),
               "finish_reasons": dict(sorted(Counter(finishes).items())), "completion_tokens": tokens,
               "prompt_tokens": prompt_tokens, "prompt_sha256": prompt_sha, "request_id": request.request_id,
               "seeds": list(request.seeds), "response_sha256": raw,
               "seconds": round(time.monotonic() - began, 3)}


def _check_weights(weights: Any) -> None:
    """F0 (both null) or a checkpoint path with the sha256 of the trainable tensors that were loaded."""
    if type(weights) is not dict or set(weights) != set(F0_WEIGHTS):
        raise ValueError("passrate: the report must bind its weights (F0 nulls, or checkpoint and trainable_sha256)")
    checkpoint, sha = weights["checkpoint"], weights["trainable_sha256"]
    if (checkpoint, sha) != (None, None) and not (
            type(checkpoint) is str and os.path.isabs(checkpoint) and type(sha) is str and re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise ValueError("passrate: weights are F0 (both null) or an absolute checkpoint path with its trainable sha256")


def _trainable_sha256(bundle: Any) -> str:
    """sha256 over the loaded trainable tensors (name -> tensor sha256, what checkpoint.save records per tensor)."""
    from fine_tuning.modules import tensor_sha256

    from .checkpoint import _parameters

    return _sha(_bytes({name: tensor_sha256(p) for name, p in sorted(_parameters(bundle).items())}))


def _outside_build(build_root: Path, output: Path) -> None:
    if Path(output).resolve().is_relative_to(Path(build_root).resolve()):
        raise ValueError("passrate: output cannot be inside the data build")


def measure(build_root: Path, split: str, ids: list[str] | None, *, protocol: dict, engine: Any, k: int,
            seed_namespace: str, output: Path, on_record: Callable[[dict], None] | None = None,
            verifier: Any = None, weights: dict | None = None, lineage: dict | None = None,
            item_batch: int = 1) -> dict:
    """k first attempts per item; records.jsonl is fsynced per item, summary.json closes the run.

    `verifier` defaults to the in-process vault Verifier; the CLI passes the isolated service.
    `weights`: what was loaded (F0_WEIGHTS by default); recorded in summary.json.
    `lineage`: the split's lineage read before the actor guards (the CLI); None reads it here (unguarded callers).
    `item_batch`: items whose first-attempt requests go to the sampler in ONE engine.sample call (runtime only:
    every request keeps its own messages, images, n, cap and seeds, so records do not depend on it). After the
    call returns each item is graded, written and fsynced in `ids` order before the next. There is no resume
    (records.jsonl is created exclusively): a crash keeps every written item and loses the unwritten items of
    the batch in flight (at most item_batch); a new run needs a new output directory.
    """
    build_root, output = Path(build_root), Path(output)
    weights = dict(F0_WEIGHTS) if weights is None else weights
    _check_weights(weights)
    if (split not in SPLITS or type(k) is not int or not K_RANGE[0] <= k <= K_RANGE[1]
            or not isinstance(seed_namespace, str) or not _NAMESPACE.fullmatch(seed_namespace)
            or type(item_batch) is not int or item_batch < 1):
        raise ValueError("passrate: known split, k in 4..8, a plain seed namespace and item_batch >= 1 required")
    _outside_build(build_root, output)
    manifest_sha = _sha(_read(build_root / "manifest.yaml"))
    actors = {a.problem_id: a for a in project_observation_actors(read_actors(build_root / split), protocol)}
    chosen = list(actors) if ids is None else list(ids)
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= actors.keys():
        raise ValueError("passrate: nonempty unique ids from the vault split required")
    lineage = read_lineage(build_root / split) if lineage is None else lineage
    if any(lineage[pid].split != split for pid in chosen):
        raise ValueError("passrate: lineage split differs from the requested split")
    prompts = local_prompts(protocol)
    cfg = RolloutConfig.from_protocol(protocol, "ALG-NB3", master_seed=MASTER_SEED)
    policy_version = engine.policy_version
    verifier = verifier if verifier is not None else Verifier(build_root / split)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    began = time.monotonic()
    rows: list[dict] = []
    with (output / "records.jsonl").open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        for start in range(0, len(chosen), item_batch):
            batch = chosen[start:start + item_batch]
            with ExitStack() as stack:
                images = []
                for pid in batch:
                    images.append(load_images(actors[pid], build_root / "images"))
                    for image in images[-1]:
                        stack.callback(image.close)
                for row in _measure_items([actors[pid] for pid in batch], images, prompts, engine, verifier, split,
                                          policy_version=policy_version, k=k, cfg=cfg, seed_namespace=seed_namespace):
                    # A crash after this point still keeps every completed item.
                    stream.write(_bytes(row) + b"\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    if on_record is not None:
                        on_record(copy.deepcopy(row))
    statuses: Counter = Counter()
    for row in rows:
        statuses.update(row["statuses"])
    summary = {"version": VERSION, "status": "MEASURED_NOT_TRAINING_APPROVAL", "split": split, "k": k,
               "seed_namespace": seed_namespace, "master_seed": MASTER_SEED, "policy_version": policy_version,
               "item_batch": item_batch,
               "weights": weights, "protocol_sha256": protocol["sha256"],
               "build_manifest_sha256": manifest_sha, "items": len(rows), "ids": chosen,
               "status_counts": dict(sorted(statuses.items())),
               "pass_histogram": {str(p): n for p, n in sorted(Counter(r["passes"] for r in rows).items())},
               "passes_total": sum(r["passes"] for r in rows), "samples_total": k * len(rows),
               "blocking_total": sum(r["blocking"] for r in rows),
               "completion_tokens_total": sum(sum(r["completion_tokens"]) for r in rows),
               "records_sha256": _sha((output / "records.jsonl").read_bytes()),
               "wall_seconds": round(time.monotonic() - began, 3)}
    _write(output / "summary.json", _bytes(summary) + b"\n")
    return summary


def _report(report_dir: Path) -> tuple[dict, list[dict], str]:
    report_dir = Path(report_dir)
    raw = _read(report_dir / "summary.json")
    summary = json.loads(raw, object_pairs_hook=_unique)
    records = _read(report_dir / "records.jsonl")
    if (type(summary) is not dict or summary.get("version") != VERSION
            or summary.get("records_sha256") != _sha(records)):
        raise ValueError("passrate: complete measurement report with matching records required")
    rows = [json.loads(line, object_pairs_hook=_unique) for line in records.decode("utf-8").splitlines()]
    k = summary["k"]
    if (len(rows) != summary["items"] or len({r["problem_id"] for r in rows}) != len(rows)
            or any(r["k"] != k or type(r["passes"]) is not int or type(r["blocking"]) is not int
                   or not 0 <= r["passes"] <= k or not 0 <= r["blocking"] <= k
                   or r["passes"] + r["blocking"] > k for r in rows)):
        raise ValueError("passrate: malformed or duplicate measurement records")
    _check_weights(summary.get("weights"))
    return summary, rows, _sha(raw)


def _check_rules(rules: Any, k: int) -> None:
    if (type(rules) is not dict or set(rules) != _RULE_KEYS or rules["version"] != RULES_VERSION
            or any(type(rules[key]) is not int for key in _RULE_KEYS - {"version"})
            or not 0 <= rules["min_pass"] <= rules["max_pass"] <= k or rules["max_blocking"] < 0):
        raise ValueError("passrate: invalid selection rules")


def _freeze(kind: str, report_dir: Path, rules: dict, output: Path) -> dict:
    result = _derive(kind, report_dir, rules)
    _write(Path(output), _bytes(result) + b"\n")
    return result


def verify_selection(selection: Path, report_dir: Path) -> tuple[dict, dict, str]:
    """A saved selection re-derived from its measurement report -> (selection, report summary, file sha256).

    Any edit to the selection, the summary or the records (each bound by hash) is refused."""
    raw = _read(Path(selection))
    value = json.loads(raw, object_pairs_hook=_unique)
    if (type(value) is not dict or value.get("version") != VERSION or type(value.get("kind")) is not str
            or type(value.get("rules")) is not dict):
        raise ValueError("passrate: saved selection required")
    if _derive(value["kind"], report_dir, value["rules"]) != value:
        raise ValueError("passrate: selection differs from the one its measurement report derives")
    return value, _report(report_dir)[0], _sha(raw)


def _derive(kind: str, report_dir: Path, rules: dict) -> dict:
    summary, rows, report_sha = _report(report_dir)
    _check_rules(rules, summary["k"])
    selected: list[str] = []
    excluded: list[dict] = []
    for row in rows:
        passes, k = row["passes"], row["k"]
        if row["blocking"] > rules["max_blocking"]:
            reason = "BLOCKING"
        elif rules["min_pass"] <= passes <= rules["max_pass"]:
            selected.append(row["problem_id"])
            continue
        else:
            reason = "ALL_CORRECT" if passes == k else "ALL_WRONG" if passes == 0 else "OUT_OF_RANGE"
        excluded.append({"id": row["problem_id"], "reason": reason})
    return {"version": VERSION, "kind": kind, "status": SELECTION_STATUS, "split": summary["split"],
            "k": summary["k"], "policy_version": summary["policy_version"], "rules": dict(rules),
            "report_sha256": report_sha, "selected_ids": selected, "excluded": excluded,
            "training_allowed": False}


def select(report_dir: Path, rules: dict, output: Path) -> dict:
    """Keep items with min_pass <= passes <= max_pass and at most max_blocking blocking samples."""
    return _freeze("PASSRATE_SELECTION", report_dir, rules, output)


def hard_negatives(report_dir: Path, *, max_pass: int = 1, output: Path) -> dict:
    """From a post-training measurement keep the items the trained policy still fails (all-wrong included)."""
    return _freeze("HARD_NEGATIVES", report_dir, {"version": RULES_VERSION, "min_pass": 0,
                   "max_pass": max_pass, "max_blocking": 0}, output)


# ---------------------------------------------------------------- CLI (real model; GPU)

def _install_guards(build_root: Path, prepared: dict | None, request: dict | None, *,
                    vllm_schema: dict | None = None) -> list[Path]:
    """`vllm_schema`: see local_run._install_actor_guards (None: HF sampler, openai never imported)."""
    from .local_run import _install_actor_guards

    if prepared is not None and request is not None:
        return _install_actor_guards(prepared, request, vllm_schema=vllm_schema)
    # Build-root mode has no request descriptors: same benchmark roots and every split's private files.
    from .guards import install_guards, prepare_vllm_schema_dependency
    from .quantized_policy import _versions
    from .shared_smoke import _prepare_hf_dependencies

    _versions()
    if vllm_schema is not None:
        vllm_schema["schema_dependency_guard"] = prepare_vllm_schema_dependency()
    _prepare_hf_dependencies()
    import_module("bitsandbytes")
    hubs = {Path.home() / "mmdl-cache/huggingface/hub",
            Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")}
    forbidden = [Path.home() / "mmdl-data/evaluation",
                 *(hub / name for hub in sorted(hubs) for name in ("datasets--MMMU--MMMU", "datasets--MMMU--MMMU_Pro"))]
    vaults = [build_root / split / name for split in SPLITS for name in ("verifier_private.jsonl", "lineage.jsonl")]
    install_guards(forbidden_roots=forbidden, vault_files=vaults, allow_vault=False, allow_benchmark=False)
    return forbidden


def _probe_actors(build_root: Path, split: str, ids: list[str] | None, protocol: dict,
                  prepared: dict | None = None) -> list:
    """The fixed single-image + paired-image probe actors; ValueError when either kind is missing.

    With a --preparation, local_run.probe_set names them (v2: the bound coverage set, probe-only actors included,
    single-image only for a single-image-only pool); v1 and build-root mode take the selection's own minimal pair.
    """
    from .local_run import minimal_actors, probe_set

    actors = project_observation_actors(read_actors(build_root / split), protocol)
    selected = [a for a in actors if ids is None or a.problem_id in ids]
    return minimal_actors(selected) if prepared is None else probe_set(prepared, selected)


def _synchronize_sampler(bundle, sampler: Any, hf_engine: Any, probe_actors: list, protocol: dict,
                         build_root: Path, output: Path, prepared: dict | None = None) -> None:
    """Forward-only HF-vs-vLLM probes at policy version 0; nothing is sampled before they pass.

    `prepared`: the --preparation whose bound probe coverage the probes cover (None: single and two-image)."""
    from . import scale as scale_options
    from .local_run import make_probes, probe_counts, sync_options

    with ExitStack() as stack:
        images = {a.problem_id: load_images(a, build_root / "images") for a in probe_actors}
        for values in images.values():
            for image in values:
                stack.callback(image.close)
        probes = make_probes(probe_actors, images, hf_engine, local_prompts(protocol), image_counts=probe_counts(prepared))
        _, report = scale_options.synchronize(bundle, sampler, hf_engine, probes, protocol, 0, push=False,
                                              **sync_options(prepared))
    _outside_build(build_root, output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write(output / "sampler-synchronization.json", _bytes(report) + b"\n")


def _cli_measure(args: argparse.Namespace) -> dict:
    from . import checkpoint, scale as scale_options
    from .hf_pool_exclusion import gpu_guard
    from .local_profile import repair_cli_keywords
    from .local_run import cache_readiness, read_frozen_preparation, read_preparation
    from .quantized_cache import block_original_access
    from .quantized_policy import build_quantized_policy, load_quantized_policy
    from .verifier_service import VerifierService

    began = time.monotonic()
    gpu_guard(True)  # refuses unless nvidia-smi lists no other compute process
    protocol = build_protocol(repair_notebook=True, context_limit=24576, **repair_cli_keywords(args))
    bf16 = scale_options.bf16(protocol)
    if bf16 and args.standalone_base:
        raise ValueError("passrate: a bf16 policy loads the verified original snapshot, not an NF4 cache")
    prepared = request = verification = None
    ids = args.ids or None
    frozen = getattr(args, "frozen_approval_descriptor", None)
    if args.preparation is not None:
        if frozen is None:
            prepared, request, actors = read_preparation(args.preparation, protocol)
        else:
            prepared, request, actors, verification = read_frozen_preparation(args.preparation, protocol, frozen)
        build_root, split = Path(prepared["build_root"]), "train"
        ids = ids or [a.problem_id for a in actors]
    else:
        build_root, split = args.build_root, args.split
    readiness = cache_readiness(args.cache_check, args.model_path, protocol) if args.standalone_base else None
    schema = {} if scale_options.options(protocol)["rollout_engine"] == "vllm" else None
    # The guards deny lineage.jsonl: read it now, never after installation.
    lineage = read_lineage(build_root / split)
    forbidden = _install_guards(build_root, prepared, request, vllm_schema=schema)
    # A vLLM sampler is probe-synchronized with the HF policy before sampling: refuse first without both probe kinds.
    probe_actors = (_probe_actors(build_root, split, ids, protocol, prepared)
                    if scale_options.options(protocol)["rollout_engine"] == "vllm" else None)
    original_access = block_original_access(Path(readiness["original_model_root"])) if readiness else None
    factory = (scale_options.build_bf16_policy if bf16 else
               load_quantized_policy if args.standalone_base else build_quantized_policy)
    weights = dict(F0_WEIGHTS)
    if args.checkpoint is None:
        bundle = factory(args.model_path, "MOD-RV", protocol)
    else:
        binding = json.loads(args.binding.read_text())
        bundle = checkpoint.load_eval(args.model_path, args.checkpoint, binding=binding.get("binding", binding),
                                      policy_factory=partial(factory, device_index=0))
        weights = {"checkpoint": str(Path(args.checkpoint).absolute()), "trainable_sha256": _trainable_sha256(bundle)}
    # HF (serial or bound row batching) or the bound vLLM sampler; version 0 as before (seed namespace).
    engine, hf_engine = scale_options.make_engines(bundle, protocol, args.model_path)
    if engine is not hf_engine:
        if args.checkpoint is not None:
            from .policy import merged_named_tensors

            # The vLLM sampler starts from base weights; load the checkpoint's merged BF16 tensors once.
            engine.initialize_weights(merged_named_tensors(bundle), 0)
        # Also at F0: initialize_weights leaves the sampler unsynchronized, and base weights are never trusted unprobed.
        _synchronize_sampler(bundle, engine, hf_engine, probe_actors, protocol, build_root, Path(args.output), prepared)
    startup = time.monotonic() - began
    manifest = json.loads((build_root / "manifest.yaml").read_text())
    with VerifierService(build_root / split, forbidden_roots=forbidden, timeout=60) as service:
        if service.metadata["vault_manifest_sha256"] != manifest["split_manifests"][split]:
            raise ValueError("passrate: verifier dataset changed")
        summary = measure(build_root, split, ids, protocol=protocol, engine=engine, k=args.k,
                          seed_namespace=args.seed_namespace, output=args.output, verifier=service, weights=weights,
                          lineage=lineage, item_batch=args.item_batch,
                          on_record=lambda row: print(json.dumps({"event": "ITEM", "problem_id": row["problem_id"],
                                                                  "passes": row["passes"], "k": row["k"]}), flush=True))
    if original_access and original_access["denied_original_access_attempts"]:
        raise ValueError("passrate: original BF16 model access attempted")
    scale_options.write_receipt(Path(args.output), items=summary["items"],
        completion_tokens=summary["completion_tokens_total"], wall_seconds=time.monotonic() - began, updates=0,
        update_seconds=0., n_samples=summary["k"], retries_per_item=0., startup_seconds=startup,
        extra={"run": "passrate", "sampler": scale_options.sampler_label(protocol), **(schema or {}),
               "completion_scope": "first_attempt_records", "item_batch": summary["item_batch"],
               **({"data_verification": verification} if verification is not None else {})})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    run = sub.add_parser("measure")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--preparation", type=Path)
    source.add_argument("--build-root", type=Path)
    run.add_argument("--split", choices=SPLITS, default="train")
    run.add_argument("--ids", nargs="*", default=[])
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--standalone-base", action="store_true")
    run.add_argument("--cache-check", type=Path)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--binding", type=Path)
    run.add_argument("--k", type=int, required=True)
    run.add_argument("--seed-namespace", required=True)
    run.add_argument("--item-batch", type=int, default=1,
                     help="items per engine.sample call (runtime only; records do not depend on it)")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--note-contract", choices=NOTE_CONTRACT_CHOICES, default=DEFAULT_NOTE_CONTRACT)
    from .local_profile import add_repair_cli_arguments
    from .local_run import add_frozen_approval_arguments, frozen_approval_option

    add_repair_cli_arguments(run)
    add_frozen_approval_arguments(run)
    pick = sub.add_parser("select")
    pick.add_argument("--report", type=Path, required=True)
    pick.add_argument("--min-pass", type=int, required=True)
    pick.add_argument("--max-pass", type=int, required=True)
    pick.add_argument("--max-blocking", type=int, default=0)
    pick.add_argument("--output", type=Path, required=True)
    hard = sub.add_parser("hard-negatives")
    hard.add_argument("--report", type=Path, required=True)
    hard.add_argument("--max-pass", type=int, default=1)
    hard.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "measure":
        if args.standalone_base != (args.cache_check is not None):
            parser.error("--standalone-base requires exactly its --cache-check")
        if args.item_batch < 1:
            parser.error("--item-batch must be at least 1")
        if (args.checkpoint is None) != (args.binding is None):
            parser.error("--checkpoint requires --binding (a launch.json or a binding object) and vice versa")
        if args.answer_verified_route != bool(args.slotless_source):
            parser.error("--slotless-source is required exactly with --answer-verified-route")
        if (args.answer_verified_route or args.finalization_reward) and args.note_contract == DEFAULT_NOTE_CONTRACT:
            parser.error("--answer-verified-route/--finalization-reward require the compact --note-contract")
        args.frozen_approval_descriptor = frozen_approval_option(parser, args)
        if args.frozen_approval_descriptor is not None and args.preparation is None:
            parser.error("--frozen-approval verifies a pool --preparation; it does not apply to --build-root")
    try:
        if args.stage == "measure":
            result = _cli_measure(args)
        elif args.stage == "select":
            result = select(args.report, {"version": RULES_VERSION, "min_pass": args.min_pass,
                                          "max_pass": args.max_pass, "max_blocking": args.max_blocking}, args.output)
        else:
            result = hard_negatives(args.report, max_pass=args.max_pass, output=args.output)
        print(json.dumps({k: v for k, v in result.items() if k not in ("ids", "selected_ids", "excluded")}), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
