"""Trusted CPU data audit and public-only binding for native ladder diagnosis.

This permits the fixed evaluation cohort, never training or a GPU launch. The
runtime separately requires actual notebook learning, reload and cost controls.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

from . import local_ladder as ladder
from .cli import artifact_path
from .dataset import _bytes, _sha, _write
from .generators import circuit
from .vault import _read, _safe_root, read_lineage, read_verifier

STATUS = "LOCAL_LADDER_DIAGNOSTIC_DATA_VERIFIED"
_CHECKS = ("native_reconstruction", "two_exact_solvers", "source_disjoint", "image_disjoint",
           "structure_disjoint", "complete_graph_node_count_disjoint", "all_images_ai_reviewed")


def _sources() -> dict[str, str]:
    from .approval import code_hashes
    root = Path(__file__).resolve().parent
    return ladder._sources() | {str(root / name): sha for name, sha in code_hashes().items()}


def _validate_protocol(protocol: dict) -> None:
    from . import local_ladder_generation, local_ladder_recovery
    if (type(protocol) is not dict or any(type(protocol.get(key)) is not str
            or re.fullmatch(r"[0-9a-f]{64}", protocol[key]) is None
            for key in ("local_protocol_sha256", "prompt_templates_sha256"))):
        raise ValueError("ladder gate: canonical native protocol and SHA bindings required")
    for module in (local_ladder_generation, local_ladder_recovery):
        if (module is local_ladder_generation and protocol.get("evaluation_prompt_policy")
                == local_ladder_generation.CIRCUIT_REPAIR_PROMPT_POLICY):
            continue
        templates = (local_ladder_generation.circuit_evaluation_prompts(protocol["evaluation_prompt_policy"])
                     if "evaluation_prompt_policy" in protocol else {})
        note = (local_ladder_recovery._note_contract(protocol) if module is local_ladder_recovery
                and local_ladder_recovery._repair(templates) else None)
        expected = module.protocol(templates, protocol["local_protocol_sha256"],
                                   context_limit=protocol.get("context_limit"),
                                   **({"note_contract": note} if note is not None else {}))
        if not templates:
            expected["prompt_templates_sha256"] = protocol["prompt_templates_sha256"]
        if _bytes(protocol) == _bytes(expected):
            return
    raise ValueError("ladder gate: canonical native first-attempt or recovery protocol required")


def _public_inputs(cohort: Path, preparation: Path) -> tuple[list, dict[str, str]]:
    actors = ladder.load_ladder_actors(cohort)
    prepared = ladder._json(preparation)
    files = [cohort / "preparation.json", cohort / "records/manifest.yaml",
             cohort / "records/actor_inputs.jsonl", preparation]
    for key in ("actors", "selection", "data_request", "validated_data", "previous_selection"):
        value = prepared[key]
        if ladder._descriptor(Path(value["path"])) != value:
            raise ValueError("ladder gate: changed training descriptor")
        files.append(Path(value["path"]))
    if "execution_actors" in prepared:
        value = prepared["execution_actors"]
        if ladder._descriptor(Path(value["path"])) != value:
            raise ValueError("ladder gate: changed execution actors")
        files.append(Path(value["path"]))
    files.extend(cohort / "images" / rel for a in actors for rel in a.image_paths)
    return actors, {str(p): ladder._descriptor(p)["file_sha256"] for p in files}


def _review(path: Path, cohort: Path, actors: list) -> dict:
    review = ladder._json(path)
    expected = [{"problem_id": a.problem_id, "image_sha256": list(a.image_sha256),
                 "verdict": "NO_MISMATCH_OBSERVED"} for a in actors]
    if (review.get("status") != "AI_REVIEWED_NOT_HUMAN_VERIFIED"
            or review.get("human_verified") is not False
            or type(review.get("gold_exposed_to_reviewer")) is not bool
            or review.get("model_payload_contains_review") is not False
            or review.get("preparation") != ladder._descriptor(cohort / "preparation.json")
            or review.get("scope") != "all-images-values-units-polarity-connections-query-target"
            or review.get("rows") != expected):
        raise ValueError("ladder gate: complete bound AI image review required")
    return review


def build_gate(cohort: Path, training_preparation: Path, review: Path,
               output: Path, protocol: dict) -> dict:
    """Run only in a trusted CPU process: private reference data is read here."""
    root = _safe_root(cohort)
    preparation, review, output = map(ladder._path, (training_preparation, review, output))
    if output.exists() or output.is_relative_to(root) or output in (preparation, review):
        raise ValueError("ladder gate: preserve cohort and prior evidence")
    _validate_protocol(protocol)
    sources = _sources()
    manifest = ladder._json(root / "preparation.json")
    actors, public = _public_inputs(root, preparation)
    prepared, train, worlds, training_inputs = ladder._training(preparation)
    if output.is_relative_to(Path(prepared["build_root"])) or output.is_relative_to(preparation.parent):
        raise ValueError("ladder gate: output must preserve training inputs")
    if (manifest.get("status") != "CPU_COHORT_PREPARED_PENDING_GATES"
            or manifest.get("source_before_sha256") != manifest.get("source_after_sha256")
            or manifest.get("source_after_sha256") != ladder._sources()
            or manifest.get("source_revision") != _sha(_bytes(ladder._sources()))
            or protocol.get("local_protocol_sha256") != prepared["local_protocol_sha256"]):
        raise ValueError("ladder gate: current native source and canonical protocol required")
    _review(review, root, actors)
    private_paths = [root / "records" / name for name in ("verifier_private.jsonl", "lineage.jsonl")]
    private = {str(p): ladder._descriptor(p)["file_sha256"] for p in private_paths}
    verifiers, lineage = read_verifier(root / "records"), read_lineage(root / "records")
    if set(verifiers) != {a.problem_id for a in actors} or set(lineage) != set(verifiers):
        raise ValueError("ladder gate: private reference coverage differs")
    node_counts, structures = set(), set()
    for index, actor in zip(ladder.INDICES, actors, strict=True):
        first = circuit.generate(index, ladder.SEED, "sealed", "obs", "default")
        second = circuit.generate(index, ladder.SEED, "sealed", "obs", "default")
        circuit.self_check(first)
        circuit.self_check(second)
        expected_lineage = replace(first.lineage, source_revision=manifest["source_revision"])
        if (first != second or first.actor != actor or first.verifier != verifiers[actor.problem_id]
                or expected_lineage != lineage[actor.problem_id] or expected_lineage.benchmark_derived
                or expected_lineage.model_touched or expected_lineage.parents
                or expected_lineage.label_provenance != "PROGRAM"):
            raise ValueError("ladder gate: native ancestry, reference or deterministic reconstruction differs")
        for rel, raw in zip(actor.image_paths, first.image_bytes, strict=True):
            if _read(root / "images" / rel) != raw:
                raise ValueError("ladder gate: rendered native image differs")
        world = first.verifier.accept["circuit_world"]
        if world["kind"] != "ladder":
            raise ValueError("ladder gate: native ladder required")
        node_counts.add(len(world["sources"]) + len(world["series"]) + 1)
        structures.add(first.lineage.structure_hash)
    checks = dict.fromkeys(_CHECKS, True)
    checks.update(source_disjoint=not {a.source for a in train} & {a.source for a in actors},
        image_disjoint=not {h for a in train for h in a.image_sha256} & {h for a in actors for h in a.image_sha256},
        structure_disjoint=not {w["structure_id"] for w in worlds.values()} & structures,
        complete_graph_node_count_disjoint=not {w["node_count"] for w in worlds.values()} & node_counts)
    if not all(checks.values()):
        raise ValueError("ladder gate: training/evaluation overlap")
    if (_sources() != sources or _public_inputs(root, preparation)[1] != public
            or any(ladder._descriptor(Path(p))["file_sha256"] != h
                   for p, h in (training_inputs | private).items())):
        raise ValueError("ladder gate: sources or inputs changed during trusted audit")
    result = {"status": STATUS, "preparation": ladder._descriptor(root / "preparation.json"),
        "training_preparation": ladder._descriptor(preparation), "protocol": protocol,
        "protocol_sha256": _sha(_bytes(protocol)), "problem_ids": [a.problem_id for a in actors],
        "source_sha256": sources, "public_inputs_sha256": public,
        "trusted_private_inputs_sha256": training_inputs | private,
        "render_review": ladder._descriptor(review), "checks": checks,
        "evaluation_allowed": True, "training_allowed": False, "gpu_execution_authorized": False,
        "human_verified": False, "model_generation_provenance_verified": False,
        "new_benchmark_exclusion_scan_performed": False,
        "limitations": ["Native program-authored evaluation only; not training authorization.",
            "Three topology families, not ten independent structures; only complete-graph nonisomorphism is proved.",
            "AI render review is not human certification. No semantic, subgraph or pretraining-overlap guarantee.",
            "Private reference data is checked here and by the later scorer, never passed to the inference actor."]}
    _write(artifact_path(output), _bytes(result) + b"\n")
    return result


def validate_gate(cohort: Path, training_preparation: Path, receipt: Path, protocol: dict) -> dict:
    """Public binding check: does not open gold/lineage or load a model."""
    root = _safe_root(cohort)
    _validate_protocol(protocol)
    preparation, receipt = map(ladder._path, (training_preparation, receipt))
    result = ladder._json(receipt)
    actors, public = _public_inputs(root, preparation)
    if (result.get("status") != STATUS or result.get("evaluation_allowed") is not True
            or result.get("training_allowed") is not False or result.get("gpu_execution_authorized") is not False
            or result.get("human_verified") is not False
            or result.get("preparation") != ladder._descriptor(root / "preparation.json")
            or result.get("training_preparation") != ladder._descriptor(preparation)
            or result.get("protocol") != protocol or result.get("protocol_sha256") != _sha(_bytes(protocol))
            or result.get("problem_ids") != [a.problem_id for a in actors]
            or result.get("checks") != dict.fromkeys(_CHECKS, True)
            or result.get("source_sha256") != _sources() or result.get("public_inputs_sha256") != public):
        raise ValueError("ladder gate: pending, changed or stale diagnostic data receipt")
    review = result["render_review"]
    if ladder._descriptor(Path(review["path"])) != review:
        raise ValueError("ladder gate: render review changed")
    _review(Path(review["path"]), root, actors)
    return result
