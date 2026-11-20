"""CPU replay of original repair journals and inputs; never a model or VJP proof."""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .checkpoint import digest
from .local_profile import REPAIR_PROTOCOL_ID, local_prompts, restore_protocol
from .schema import ActorItem, ProblemTree, from_json, to_json
from .vault import _read, _unique

VERSION = "repair-capture-replay-v1"


def validate_probe_evidence(probes: dict, problem_ids: list[str], image_counts: list[int], *,
                            batch_rows: int | None = None) -> None:
    """Check saved single(/multi)-image probe structure; actual forward comparison is separate.

    `image_counts` is the caller's bound probe coverage: [1, 2], or [1] for a v2 single-image-only pool.
    A row-seeded batched result is accepted only when the protocol binds `batch_rows`."""
    from .engines import ROW_SEEDED_BATCH_SAMPLER, _completion

    labels = {"batch_rows": batch_rows, "sampler_version": ROW_SEEDED_BATCH_SAMPLER} if batch_rows is not None else {}

    failure = "repair replay: exact original single/multi-image forward probes required"
    if (type(problem_ids) is not list or type(image_counts) is not list
            or digest(image_counts) not in (digest([1]), digest([1, 2])) or len(problem_ids) != len(image_counts)
            or any(type(pid) is not str or not pid for pid in problem_ids) or len(set(problem_ids)) != len(problem_ids)
            or type(probes) is not dict or set(probes) != {"inputs", "reference_logprobs", "generated", "task_accuracy_measured"}
            or probes["task_accuracy_measured"] is not False
            or any(type(probes[key]) is not list or len(probes[key]) != len(problem_ids)
                   for key in ("inputs", "reference_logprobs", "generated"))):
        raise ValueError(failure)
    for ordinal, (pid, count) in enumerate(zip(problem_ids, image_counts, strict=True)):
        row, generated, reference = (probes[key][ordinal] for key in ("inputs", "generated", "reference_logprobs"))
        request_id = "local-sync:" + pid
        if (type(row) is not dict or set(row) != {"request_id", "input_tokens", "image_tokens", "image_grid_thw",
                "processed_image_sizes", "pixel_tensors"} or row["request_id"] != request_id
                or type(row["input_tokens"]) is not int or row["input_tokens"] <= 0
                or type(row["image_tokens"]) is not int or not 0 < row["image_tokens"] <= row["input_tokens"]
                or type(row["image_grid_thw"]) is not list or len(row["image_grid_thw"]) != count
                or any(type(grid) is not list or len(grid) != 3
                       or any(type(value) is not int or value <= 0 for value in grid) for grid in row["image_grid_thw"])
                or type(row["processed_image_sizes"]) is not list or len(row["processed_image_sizes"]) != count
                or any(type(size) is not list or len(size) != 2
                       or any(type(value) is not int or value <= 0 for value in size) for size in row["processed_image_sizes"])
                or type(row["pixel_tensors"]) is not dict or set(row["pixel_tensors"]) != {"pixel_values", "image_grid_thw"}
                or type(generated) is not dict or set(generated) != {"request_id", "prompt_token_ids", "completions", *labels}
                or any(digest(generated[key]) != digest(value) for key, value in labels.items())
                or generated["request_id"] != request_id or type(generated["prompt_token_ids"]) is not list
                or len(generated["prompt_token_ids"]) != row["input_tokens"]
                or any(type(token) is not int or token < 0 for token in generated["prompt_token_ids"])
                or type(generated["completions"]) is not list or len(generated["completions"]) != 1
                or type(reference) is not list or not reference
                or any(type(value) not in (int, float) or not math.isfinite(value) or value > 1e-5 for value in reference)):
            raise ValueError(failure)
        for name, tensor in row["pixel_tensors"].items():
            if (type(tensor) is not dict or set(tensor) != {"shape", "dtype", "sha256"}
                    or type(tensor["shape"]) is not list or len(tensor["shape"]) != 2
                    or any(type(value) is not int or value <= 0 for value in tensor["shape"])
                    or tensor["dtype"] not in ({"torch.int64"} if name == "image_grid_thw" else
                        {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"})
                    or type(tensor["sha256"]) is not str or len(tensor["sha256"]) != 64
                    or set(tensor["sha256"]) - set("0123456789abcdef")):
                raise ValueError(failure)
        if (row["pixel_tensors"]["image_grid_thw"]["shape"] != [count, 3]
                or row["pixel_tensors"]["pixel_values"]["shape"][0] != sum(math.prod(grid) for grid in row["image_grid_thw"])):
            raise ValueError(failure)
        completion = generated["completions"][0]
        if (type(completion) is not dict or set(completion) != {"token_ids", "logprobs", "finish_reason", "text"}
                or type(completion["token_ids"]) is not list or not 0 < len(completion["token_ids"]) <= 8
                or type(completion["logprobs"]) is not list or type(completion["text"]) is not str
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in completion["logprobs"])):
            raise ValueError(failure)
        try:
            _completion(completion["token_ids"], completion["logprobs"], completion["finish_reason"], completion["text"])
        except (ValueError, RuntimeError):
            raise ValueError(failure) from None


def read_snapshot(path: Path) -> tuple[Any, str]:
    """Parse and hash the same guarded bytes before any replay callbacks."""
    def invalid(value: str) -> None:
        raise ValueError("repair replay: nonfinite JSON")

    raw = _read(path)
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid)
    digest(value)
    return value, hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> Any:
    return read_snapshot(path)[0]


def _sha(path: Path) -> str:
    return hashlib.sha256(_read(path)).hexdigest()


def checkpoint_snapshot(capture_root: Path, name: str, *, binding: dict) -> tuple[dict, dict[str, str]]:
    """Bind metadata and all three payloads to one stable, capture-local checkpoint."""
    from .checkpoint import file_hash, read_metadata

    if type(name) is not str or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("repair replay: capture-local checkpoint name required")
    root = capture_root / name
    names = ("metadata.json", "metadata.sha256", "adapter.safetensors", "mergers.safetensors", "runtime.pt")
    if (root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()} != set(names)
            or any(p.is_symlink() or not p.is_file() for p in root.iterdir())):
        raise ValueError("repair replay: exact original checkpoint files required")
    parsed, metadata_sha = read_snapshot(root / "metadata.json")
    hashes = {entry: file_hash(root / entry) for entry in names}
    metadata = read_metadata(root, binding=binding)
    if (hashes["metadata.json"] != metadata_sha or digest(parsed) != digest(metadata)
            or {p.name for p in root.iterdir()} != set(names)
            or any(p.is_symlink() or not p.is_file() for p in root.iterdir())
            or any(file_hash(root / entry) != sha for entry, sha in hashes.items())):
        raise ValueError("repair replay: checkpoint changed during snapshot")
    return metadata, {f"{name}/{entry}": sha for entry, sha in hashes.items()}


def _wire(value: Any) -> Any:
    return json.loads(to_json(value))


def _expected_journal(trees: list[ProblemTree]) -> list[dict]:
    from . import fresh_observation as fresh

    expected = []
    for tree in trees:
        collector = tree.notebook["repair_collector"]
        record = tree.notebook["rewarded_notebook"]
        observation = collector["fresh_observation"]
        events = fresh.events_for(observation) if observation is not None else []
        if (type(collector["fresh_observation_calls"]) is not int
                or collector["fresh_observation_calls"] != int(observation is not None)
                or digest(collector["fresh_observation_events"]) != digest(events)):
            raise ValueError("repair replay: original fresh count, phase or order differs")
        groups = {group.group_id: group for group in tree.groups}
        used = []
        for call in collector["calls"]:
            if call["turn"] == 2:
                for event in collector["fresh_observation_events"]:
                    expected.append({"kind": "REPAIR_RAW", "raw": {"version": collector["version"],
                        "purpose": "RETRY_FRESH_OBSERVATION", "problem_id": tree.problem_id,
                        "policy_version": tree.policy_version, "event": event}})
            completions = call["completions"]
            if type(call["n"]) is not int or len(completions) != call["n"]:
                raise ValueError("repair replay: incomplete native call")
            for index, completion in enumerate(completions):
                event = {"version": collector["version"], "purpose": "NATIVE_SOLUTION_COMPLETION",
                    "problem_id": tree.problem_id, "policy_version": tree.policy_version,
                    **call, "completions": [], "index": index, "seed": call["seeds"][index],
                    "completion": completion}
                expected.append({"kind": "REPAIR_RAW", "raw": event})
            used.append(call["group_id"])
            expected.append({"kind": "GROUP", "raw": _wire(groups[call["group_id"]])})
        if not any(call["turn"] == 2 for call in collector["calls"]):
            for event in collector["fresh_observation_events"]:
                expected.append({"kind": "REPAIR_RAW", "raw": {"version": collector["version"],
                    "purpose": "RETRY_FRESH_OBSERVATION", "problem_id": tree.problem_id,
                    "policy_version": tree.policy_version, "event": event}})
        call = record["call"]
        base = {"version": record["notebook_generation_policy"], "problem_id": tree.problem_id,
            "policy_version": tree.policy_version, "source_audit_sha256": record["source_audit"]["sha256"]}
        if call is not None:
            if type(call["n"]) is not int or call["n"] != 4 or len(call["completions"]) != 4:
                raise ValueError("repair replay: incomplete final call")
            expected.append({"kind": "REPAIR_RAW", "raw": base | {"purpose": "REPAIR_FINAL_STARTED",
                "status": "STARTED", "call": call | {"completions": []}}})
            for index, completion in enumerate(call["completions"]):
                expected.append({"kind": "REPAIR_RAW", "raw": base | {"purpose": "REPAIR_FINAL_COMPLETION",
                    "status": "STARTED", "call": call | {"completions": call["completions"][:index + 1]},
                    "index": index, "seed": call["seeds"][index], "completion": completion}})
        expected.append({"kind": "REPAIR_RAW", "raw": base | {"purpose": "REPAIR_FINAL_CLOSED",
            "status": record["status"], "call": call}})
        for gid in record["group_ids"]:
            used.append(gid)
            expected.append({"kind": "GROUP", "raw": _wire(groups[gid])})
        if used != [group.group_id for group in tree.groups] or len(set(used)) != len(used):
            raise ValueError("repair replay: journal group identity or order differs")
        expected.append({"kind": "TREE", "raw": _wire(tree)})
    return expected


def raw_journal_paths(root: Path) -> list[Path]:
    """Capture-journal files in write order: the writer's `:04d` widens past raw-9999, so order by (width, name)."""
    return sorted(root.glob("raw-*.json"), key=lambda path: (len(path.name), path.name))


def validate_journal(capture_root: Path, raw_records: list, trees: list[ProblemTree], *,
                     complete: bool = True) -> dict[str, str]:
    """Match every durable external row to original embedded calls, including all siblings.

    `complete=False` (resume prefix) requires the records to be the contiguous journal start only;
    later raw files of an abandoned uncheckpointed batch stay in their own segment."""
    if (type(raw_records) is not list or type(trees) is not list or not trees
            or any(type(tree) is not ProblemTree for tree in trees)
            or len({tree.problem_id for tree in trees}) != len(trees)):
        raise ValueError("repair replay: original journal and unique trees required")
    files = [path.name for path in raw_journal_paths(capture_root)]
    names = [f"raw-{index:04d}.json" for index in range(len(raw_records))]
    if (files if complete else files[:len(names)]) != names:
        raise ValueError("repair replay: contiguous complete raw journal required")
    expected = _expected_journal(trees)
    if len(expected) != len(raw_records):
        raise ValueError("repair replay: original journal event count differs")
    paths = {}
    for name, receipt, event in zip(names, raw_records, expected, strict=True):
        path = capture_root / name
        actual, sha = read_snapshot(path)
        if (type(receipt) is not dict or set(receipt) != {"file", "kind", "sha256"}
                or receipt["file"] != name or receipt["kind"] != event["kind"]
                or receipt["sha256"] != sha or digest(actual) != digest(event)):
            raise ValueError("repair replay: raw journal differs from original embedded call or group")
        paths[name] = receipt["sha256"]
    return paths


def replay_trees(trees: list[ProblemTree], actors: list[ActorItem], protocol: dict, prepared: dict,
                 model_path: Path, *, control: bool = False) -> dict[str, dict]:
    """Reprocess original images/token prompts on CPU and freshly verify native semantics."""
    from transformers import AutoConfig, AutoProcessor

    from .dataset import load_images
    from .engines import InputPreparation
    from .launch import read_descriptor
    from .repair_training import prepare_tree_credit
    from .verifier_service import VerifierService

    if (restore_protocol(protocol)["protocol_id"] != REPAIR_PROTOCOL_ID or type(control) is not bool
            or type(trees) is not list or not trees or type(actors) is not list
            or any(type(tree) is not ProblemTree for tree in trees)
            or any(type(actor) is not ActorItem for actor in actors)
            or len({tree.problem_id for tree in trees}) != len(trees)
            or len({actor.problem_id for actor in actors}) != len(actors)):
        raise ValueError("repair replay: explicit repair protocol and original inputs required")
    snapshot = digest({"trees": [_wire(tree) for tree in trees], "actors": [_wire(actor) for actor in actors],
                       "protocol": protocol, "prepared": prepared})
    current = [from_json(ProblemTree, to_json(tree)) for tree in trees]
    actor_map = {actor.problem_id: from_json(ActorItem, to_json(actor)) for actor in actors}
    if any(tree.problem_id not in actor_map for tree in current):
        raise ValueError("repair replay: source actor unavailable")
    if any(digest(tree.notebook["rewarded_notebook"]["source_prompts"]) != digest(local_prompts(protocol))
           for tree in current):
        raise ValueError("repair replay: source prompt differs from canonical protocol")
    root = Path(prepared["build_root"])
    if prepared["image_root"] != str(root / "images"):
        raise ValueError("repair replay: original image root differs")
    request = read_descriptor(prepared["data_request"])
    forbidden = [Path(request["index"]["path"]).parent, Path.home() / "mmdl-data/evaluation"]
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
        min_pixels=protocol["image"]["min_pixels"], max_pixels=protocol["image"]["max_pixels"])
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    preparation = InputPreparation(processor, config, protocol["image"], protocol["context_limit"])
    receipts = {}
    with ExitStack() as stack:
        verifier = stack.enter_context(VerifierService(root / "train", forbidden_roots=forbidden, timeout=60))
        if verifier.metadata["vault_manifest_sha256"] != _json(root / "manifest.yaml")["split_manifests"]["train"]:
            raise ValueError("repair replay: independent verifier dataset differs")
        for tree in current:
            actor = actor_map[tree.problem_id]
            images = load_images(actor, root / "images")
            try:
                preparation.policy_version = tree.policy_version
                receipts[tree.problem_id] = prepare_tree_credit(tree, actor, verifier, preparation.tokenizer,
                    preparation=preparation, images=images, master_seed=3407, notebook_reward_enabled=not control)
            finally:
                for image in images:
                    image.close()
    if snapshot != digest({"trees": [_wire(tree) for tree in trees], "actors": [_wire(actor) for actor in actors],
                           "protocol": protocol, "prepared": prepared}):
        raise ValueError("repair replay: original replay inputs mutated")
    return json.loads(json.dumps(receipts, allow_nan=False))


def replay_capture(capture_root: Path, *, protocol: dict, prepared: dict, actors: list[ActorItem],
                   parameter_inventory: list, require_s: bool = False,
                   probe_actors: list[ActorItem] | None = None) -> dict:
    """Replay a closed minimum capture; actual fresh model reload remains separate evidence.

    With `probe_actors` (v2 pool preparations) `actors` is the exact trained pair and the probes are the
    separate set covering the preparation's bound image counts; without it both are the first single/two-image `actors`."""
    from .local_run import covering_actors, minimal_actors, minimum_binding, probe_counts, verify_changes
    from .repair_evidence import repair_update_evidence

    if restore_protocol(protocol)["protocol_id"] != REPAIR_PROTOCOL_ID or type(require_s) is not bool:
        raise ValueError("repair replay: explicit canonical repair minimum required")
    names = ("launch.json", "result.json", "reload_expected.json", "repair_preparation_index.json",
             "repair_update_evidence.json", "update.json", "inventory.json", "parameter_changes.json",
             "updated_forward_probes.json", "updated_sync.json")
    snapshots = {name: read_snapshot(capture_root / name) for name in names}
    files = {name: value for name, (value, _) in snapshots.items()}
    if any(type(value) is not dict for name, value in files.items() if name != "updated_sync.json"):
        raise ValueError("repair replay: original object receipts required")
    paths = {name: sha for name, (_, sha) in snapshots.items()}
    launch, result, expected, index, envelope, update = (files[name] for name in names[:6])
    selected = minimal_actors(actors) if probe_actors is None else list(actors)
    probed = selected if probe_actors is None else covering_actors(probe_actors, probe_counts(prepared))
    ids, probe_ids = [actor.problem_id for actor in selected], [actor.problem_id for actor in probed]
    from .scale import options as scale_options

    validate_probe_evidence(files["updated_forward_probes.json"], probe_ids, [len(actor.image_paths) for actor in probed],
                            batch_rows=scale_options(protocol)["rollout_batch_rows"])
    sync = files["updated_sync.json"]
    if (type(sync) is not list or len(sync) != len(probe_ids) or any(type(row) is not dict
            or set(row) != {"request_id", "mean_abs_delta", "max_abs_delta", "policy_version"}
            or row["request_id"] != "local-sync:" + probe_ids[ordinal]
            or type(row["policy_version"]) is not int or row["policy_version"] != 1
            or any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0
                   for key in ("mean_abs_delta", "max_abs_delta"))
            or row["mean_abs_delta"] > row["max_abs_delta"]
            or row["mean_abs_delta"] > protocol["engine"]["probe_max_mean_abs_logprob_diff"]
            for ordinal, row in enumerate(sync))):
        raise ValueError("repair replay: actual synchronized single/multi-image probe reports required")
    binding = minimum_binding(prepared, protocol)
    if (digest(launch.get("protocol")) != digest(protocol) or launch.get("binding") != binding
            or launch.get("problem_ids") != ids or len(set(ids)) != 2
            or any(key in launch for key in ("execution_contract", "execution_contract_sha256"))
            or type(launch.get("pid")) is not int or launch["pid"] <= 0
            or result.get("status") != "UPDATED_SAVED_PENDING_FRESH_RELOAD"
            or type(result.get("optimizer_updates")) is not int or result["optimizer_updates"] != 1
            or result.get("attempted_ids") != ids or result.get("fresh_reload_verified") is not False
            or result.get("repair_functional_verified") is not False or result.get("repair_learning_verified") is not False
            or type(result.get("S_gradient_verified")) is not bool
            or type(expected.get("step")) is not int or expected["step"] != 1
            or expected.get("repair_protocol_sha256") != protocol["sha256"]):
        raise ValueError("repair replay: closed original minimum/protocol/binding differs")
    if (type(index) is not dict or set(index) != {"version", "protocol_sha256", "capture_binding_sha256", "trees", "raw_records"}
            or index["version"] != "repair-local-minimum-preparation-v1"
            or index["protocol_sha256"] != protocol["sha256"] or index["capture_binding_sha256"] != digest(binding)
            or type(index["trees"]) is not list or len(index["trees"]) != 2
            or type(index["raw_records"]) is not list
            or any(type(row) is not dict or set(row) != {"file", "kind", "sha256"}
                   or row["file"] != f"raw-{ordinal:04d}.json" for ordinal, row in enumerate(index["raw_records"]))
            or result.get("repair_preparation_index_sha256") != paths["repair_preparation_index.json"]
            or result.get("repair_update_evidence_sha256") != paths["repair_update_evidence.json"]
            or expected.get("repair_update_evidence_sha256") != paths["repair_update_evidence.json"]):
        raise ValueError("repair replay: original minimum receipt links differ")
    trees, saved = [], {}
    for ordinal, link in enumerate(index["trees"]):
        if (type(link) is not dict or set(link) != {"problem_id", "tree_file", "tree_file_sha256", "preparation_file",
                "preparation_file_sha256", "preparation_receipt_sha256"}
                or link["problem_id"] != ids[ordinal]
                or link["preparation_file"] != f"repair-preparation-{ordinal:04d}.json"
                or link["tree_file"] not in {row["file"] for row in index["raw_records"] if row.get("kind") == "TREE"}):
            raise ValueError("repair replay: original tree/preparation identity differs")
        tree_file, prep_file = capture_root / link["tree_file"], capture_root / link["preparation_file"]
        (row, tree_sha), (receipt, prep_sha) = read_snapshot(tree_file), read_snapshot(prep_file)
        if (set(row) != {"kind", "raw"} or row["kind"] != "TREE"
                or tree_sha != link["tree_file_sha256"] or prep_sha != link["preparation_file_sha256"]
                or receipt.get("sha256") != link["preparation_receipt_sha256"]):
            raise ValueError("repair replay: original tree/preparation file hash differs")
        tree = from_json(ProblemTree, json.dumps(row["raw"], allow_nan=False))
        if tree.problem_id != ids[ordinal] or type(tree.policy_version) is not int or tree.policy_version != 0:
            raise ValueError("repair replay: original policy-zero minimal tree required")
        trees.append(tree)
        saved[tree.problem_id] = receipt
        paths[link["tree_file"]], paths[link["preparation_file"]] = tree_sha, prep_sha
    paths.update(validate_journal(capture_root, index["raw_records"], trees))
    fresh = replay_trees(trees, selected, protocol, prepared, Path(launch["model_path"]))
    if digest(fresh) != digest(saved):
        raise ValueError("repair replay: fresh original preparation differs from saved receipt")
    eligible = [tree for tree in trees if fresh[tree.problem_id]["admitted"] and fresh[tree.problem_id]["has_signal"]]
    if result.get("discarded_ids") != [tree.problem_id for tree in trees if tree not in eligible]:
        raise ValueError("repair replay: recorded minimum eligibility differs")
    summary = repair_update_evidence(update, protocol, preparations={tree.problem_id: fresh[tree.problem_id] for tree in eligible},
        trees=eligible, parameter_inventory=parameter_inventory, require_s=require_s, control=False)
    physical = files["parameter_changes.json"]
    changes = verify_changes(physical["before"], physical["after"], update)
    if (digest(physical) != digest({"before": physical["before"], "after": physical["after"], **changes})
            or digest(physical["after"]) != digest(expected.get("trainable_sha256"))
            or digest(files["inventory.json"].get("trainable")) != digest(parameter_inventory)
            or digest(files["updated_forward_probes.json"]) != digest(expected.get("forward_probes"))
            or digest({key: result.get(key) for key in changes}) != digest(changes)):
        raise ValueError("repair replay: captured parameter changes, inventory or forward probes differ")
    metadata, checkpoint_paths = checkpoint_snapshot(capture_root, "checkpoint", binding=binding)
    if (digest(metadata["state_sha256"]) != digest({name: sha for group in physical["after"].values()
                                                  for name, sha in group.items()})
            or digest(metadata["inventory"]) != digest(files["inventory.json"])
            or digest(metadata["protocol"]) != digest(protocol)
            or digest(metadata["base_sha256"]) != digest(expected["frozen_base_sha256"])
            or digest(metadata["inventory"].get("quantization")) != digest(expected["quantization"])
            or type(metadata["step"]) is not int or metadata["step"] != 1
            or digest(metadata["cursor"]) != digest({"epoch": 0, "position": 2,
                "discarded_problem_ids": result["discarded_ids"], "problem_ids": ids, "sync_pending": False})):
        raise ValueError("repair replay: minimum checkpoint differs from recorded update")
    paths.update(checkpoint_paths)
    wanted = {"version": "repair-local-minimum-update-v1", "protocol_sha256": protocol["sha256"],
        "capture_binding_sha256": digest(binding), "preparation_index_sha256": paths["repair_preparation_index.json"],
        "update_file_sha256": paths["update.json"], "retained_repair_signal_sha256": digest(update.get("retained_repair_signal")),
        "repair_aggregate_gradient_sha256": digest(update.get("repair_aggregate_gradient")), "validation": summary}
    if (digest(envelope) != digest(wanted) or result["S_gradient_verified"] is not summary["S_gradient_verified"]
            or result.get("S_text_gradient_verified") is not summary.get("S_text_gradient_verified")
            or ("repair_route_policy" in protocol) is not ("answer_verified_ids" in result)
            or "answer_verified_ids" in result and result["answer_verified_ids"] != [
                tree.problem_id for tree in trees if fresh[tree.problem_id]["admission"] == "ANSWER_VERIFIED"]):
        raise ValueError("repair replay: independently replayed update evidence differs")
    from .checkpoint import file_hash

    if (any((file_hash if name in checkpoint_paths else _sha)(capture_root / name) != sha for name, sha in paths.items())
            or [path.name for path in raw_journal_paths(capture_root)] != [row["file"] for row in index["raw_records"]]):
        raise ValueError("repair replay: original evidence changed during independent replay")
    report = {"version": VERSION, "problem_ids": ids, "eligible_problem_ids": [tree.problem_id for tree in eligible],
        "protocol_sha256": protocol["sha256"], "capture_binding_sha256": digest(binding),
        "update_sha256": summary["update_sha256"], "preparation_sha256": digest(fresh),
        "journal_sha256": digest(index["raw_records"]), "paths": paths, "update_evidence": summary,
        "S_gradient_verified": summary["S_gradient_verified"], "semantic_replay_verified": True,
        "fresh_cpu_inputs_verified": True, "vjp_remeasured": False, "repair_functional_verified": False,
        "repair_learning_verified": False, "benefit_verified": False}
    return report | {"sha256": digest(report)}
