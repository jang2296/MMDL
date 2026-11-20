"""Prepare ten native sealed ladders on CPU; grants no evaluation/training authority."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from fine_tuning import ablation, circuit_bridge, graph_bridge, nvw

from . import cli, dataset, schema, vault
from .dataset import _bytes, _sha, _write, load_images
from .generators import circuit, common, graph, graph_pair
from .schema import ActorItem, to_json
from .vault import _read, _safe_root, _unique, read_actors, read_lineage, read_verifier

SEED = 20261021
INDICES = (2, 5, 8, 10, 11, 13, 14, 16, 17, 20)
_HERE = Path(__file__).resolve().parent


def _path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("local ladder: absolute unaliased paths required")
    _safe_root(path.parent)
    return path


def _descriptor(path: Path) -> dict:
    path = _path(path)
    return {"path": str(path), "file_sha256": _sha(_read(path))}


def _json(path: Path):
    return json.loads(_read(_path(path)), object_pairs_hook=_unique)


def _bound(descriptor: dict, bindings: dict[str, str]):
    if type(descriptor) is not dict or set(descriptor) != {"path", "file_sha256"}:
        raise ValueError("local ladder: exact path/SHA descriptor required")
    path = _path(descriptor["path"])
    actual = _descriptor(path)
    if actual != descriptor:
        raise ValueError("local ladder: bound input changed")
    bindings[str(path)] = actual["file_sha256"]
    return _json(path)


def _sources() -> dict[str, str]:
    modules = (circuit, common, graph, graph_pair, circuit_bridge, nvw, ablation,
               graph_bridge, cli, dataset, schema, vault)
    paths = []
    for module in modules:
        if module.__file__ is None:
            raise ValueError("local ladder: source-bound modules required")
        paths.append(Path(module.__file__).absolute())
    paths.extend((Path(__file__).absolute(), _HERE / "protocol_v1.yaml", _HERE / "guards.py"))
    return {str(path): _descriptor(path)["file_sha256"] for path in sorted(paths)}


def _graph_nodes(world: object) -> int:
    """Count the actual native world, rejecting extensions to its bounded contract."""
    try:
        if not isinstance(world, dict):
            raise ValueError("graph world object required")
        graph._validate_world(world)
        if world["kind"] in ("bfs", "shortest"):
            return len(set(world["nodes"]))
        if world["kind"] == "dfa":
            return len(set(world["trans"]) | {v for row in world["trans"].values() for v in row.values()})
        return len({world["root"]} | set(world["children"]) |
                   {v for row in world["children"].values() for v in row.values()})
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("local ladder: unsupported training graph_world") from exc


def _training(preparation: Path) -> tuple[dict, list[ActorItem], dict, dict[str, str]]:
    bindings: dict[str, str] = {}
    prep = _bound(_descriptor(preparation), bindings)
    if (prep.get("format") != "notebook-rl-local-data-v1"
            or prep.get("status") != "LOCAL_DATA_REVALIDATED"
            or prep.get("training_allowed") is not False):
        raise ValueError("local ladder: bound local preparation required")
    selection = _bound(prep["selection"], bindings)
    public = _bound(prep["actors"], bindings)
    request = _bound(prep["data_request"], bindings)
    validated = _bound(prep["validated_data"], bindings)
    original_selection = _bound(prep["previous_selection"], bindings)
    root = _safe_root(Path(prep["build_root"]))
    if (not Path(prep["build_root"]).is_absolute() or request.get("build_root") != str(root)
            or prep.get("image_root") != str(root / "images")
            or validated.get("status") != "VALIDATED_FOR_MEASUREMENT"
            or validated.get("training_allowed") is not False
            or validated.get("measurement_allowed") is not True or validated.get("blockers") != []
            or selection.get("status") != "LOCAL_FIXED_TEN_NOT_TRAINING_APPROVAL"
            or selection.get("local_protocol_sha256") != prep.get("local_protocol_sha256")
            or selection.get("previous_selection") != prep["previous_selection"]
            or any(selection.get(key) != original_selection.get(key)
                   for key in ("problem_ids", "structures", "image_counts"))):
        raise ValueError("local ladder: preparation build/selection binding mismatch")
    build = _bound({"path": str(root / "manifest.yaml"), "file_sha256":
                   validated.get("input_binding", {}).get("build_manifest_sha256")}, bindings)
    if (build.get("format") != "notebook-rl-candidate-build-v1"
            or build.get("training_allowed") is not False or build.get("counts", {}).get("train") != 16):
        raise ValueError("local ladder: validated native train16 build required")
    _bound({"path": str(root / "train/manifest.yaml"), "file_sha256":
            build.get("split_manifests", {}).get("train")}, bindings)
    for name in ("actor_inputs.jsonl", "verifier_private.jsonl", "lineage.jsonl"):
        descriptor = _descriptor(root / "train" / name)
        bindings[descriptor["path"]] = descriptor["file_sha256"]
    actors = {actor.problem_id: actor for actor in read_actors(root / "train")}
    verifiers, lineage = read_verifier(root / "train"), read_lineage(root / "train")
    validated_ids = validated.get("selected_ids")
    if (type(validated_ids) is not list or len(validated_ids) != 16 or len(set(validated_ids)) != 16
            or set(validated_ids) != set(actors) or set(actors) != set(lineage)
            or validated.get("actors") != [json.loads(to_json(actors[pid])) for pid in validated_ids]
            or validated.get("structures") != {pid: lineage[pid].structure_hash for pid in validated_ids}):
        raise ValueError("local ladder: actual vault differs from validated train16")
    ids = selection.get("problem_ids")
    if (type(ids) is not list or len(ids) != 10 or len(set(ids)) != 10
            or not set(ids) <= actors.keys() or set(actors) != set(verifiers) or set(actors) != set(lineage)
            or type(public) is not list or public != [json.loads(to_json(actors[pid])) for pid in ids]
            or selection.get("structures") != {pid: lineage[pid].structure_hash for pid in ids}
            or selection.get("image_counts") != {pid: len(actors[pid].image_paths) for pid in ids}):
        raise ValueError("local ladder: exact selected ten actors/structures required")
    selected, worlds = [actors[pid] for pid in ids], {}
    if sorted(actor.source for actor in selected) != [graph_pair.GENERATOR_VERSION] * 5 + [graph.GENERATOR_VERSION] * 5:
        raise ValueError("local ladder: selected graph/graph_pair five each required")
    for actor in selected:
        row, verifier = lineage[actor.problem_id], verifiers[actor.problem_id]
        world = verifier.accept.get("graph_world")
        if not isinstance(world, dict):
            raise ValueError("local ladder: unsupported training graph_world")
        nodes = _graph_nodes(world)
        structure = common.structure_hash({k: v for k, v in world.items() if k != "positions"})
        template = ("graph-pair-shortest" if actor.source == graph_pair.GENERATOR_VERSION else
                    f"graph-{world['kind']}" + (f"-{row.template_id.rsplit('-', 1)[-1]}" if world["kind"] == "tree" else ""))
        if (row.split != "train" or row.source != actor.source or row.generator_version != actor.source
                or actor.family != "graph" or row.template_id != template or row.structure_hash != structure
                or actor.answer_format != verifier.answer_type or actor.choices != verifier.choices
                or len(actor.image_paths) != (2 if actor.source == graph_pair.GENERATOR_VERSION else 1)
                or (actor.source == graph_pair.GENERATOR_VERSION and world["kind"] != "shortest")):
            raise ValueError("local ladder: selected native graph metadata mismatch")
        for image in load_images(actor, root / "images"):
            image.close()
        for rel, digest in zip(actor.image_paths, actor.image_sha256, strict=True):
            bindings[str(root / "images" / rel)] = digest
        worlds[actor.problem_id] = {"kind": world["kind"], "node_count": nodes,
            "structure_id": row.structure_hash, "template_id": row.template_id,
            "generator_version": row.generator_version}
    return prep, selected, worlds, bindings


def _preserved(sources: dict[str, str], bindings: dict[str, str]) -> dict[str, str]:
    current = _sources()
    if current != sources or any(_descriptor(Path(path))["file_sha256"] != digest
                                 for path, digest in bindings.items()):
        raise ValueError("local ladder: source/input changed during preparation")
    return current


def prepare_ladder_data(preparation: Path, out: Path) -> dict:
    preparation, out = _path(Path(preparation).absolute()), Path(out).absolute()
    if ".." in out.parts or any(path.is_symlink() for path in (out, *out.parents)):
        raise ValueError("local ladder: output aliases forbidden")
    if out.exists():
        raise FileExistsError("local ladder: preserve previous output")
    source_before = _sources()
    prep, train, worlds, bindings = _training(preparation)
    if any(out.is_relative_to(root) for root in
           (preparation.parent, Path(prep["build_root"]), *(Path(path).parent for path in bindings))):
        raise ValueError("local ladder: output must be outside input roots")
    rows = []
    revision = _sha(_bytes(source_before))
    for index in INDICES:
        first = circuit.generate(index, SEED, "sealed", "obs", "default")
        second = circuit.generate(index, SEED, "sealed", "obs", "default")
        circuit.self_check(first)
        circuit.self_check(second)
        if first != second:
            raise ValueError("local ladder: nondeterministic native generator")
        rows.append(replace(first, lineage=replace(first.lineage, source_revision=revision)))
    topologies = []
    for row in rows:
        world = row.verifier.accept["circuit_world"]
        if world["kind"] != "ladder":
            raise ValueError("local ladder: only native ladders allowed")
        topologies.append((len(world["sources"]), len(world["series"])))
    train_nodes = {world["node_count"] for world in worlds.values()}
    ladder_nodes = {sources + series + 1 for sources, series in topologies}
    separation = {
        "source_disjoint": not {a.source for a in train} & {r.actor.source for r in rows},
        "generator_disjoint": not {w["generator_version"] for w in worlds.values()} & {r.lineage.generator_version for r in rows},
        "template_disjoint": not {w["template_id"] for w in worlds.values()} & {r.lineage.template_id for r in rows},
        "structure_id_disjoint": not {w["structure_id"] for w in worlds.values()} & {r.lineage.structure_hash for r in rows},
        "image_sha256_disjoint": not {h for a in train for h in a.image_sha256} & {h for r in rows for h in r.actor.image_sha256},
        "complete_graph_node_count_disjoint": not train_nodes & ladder_nodes,
    }
    if not all(separation.values()):
        raise ValueError("local ladder: evaluation/training overlap")
    if (len(rows) != 10 or len({r.actor.problem_id for r in rows}) != 10
            or len({r.lineage.structure_hash for r in rows}) != 10 or set(topologies) != {(1, 3), (2, 2), (2, 3)}):
        raise ValueError("local ladder: fixed ten/three topology families changed")
    source_after = _preserved(source_before, bindings)
    # No output is created until all bindings, solvers and separation checks pass.
    out = cli.artifact_path(out)
    pool = cli.write_pool(out, rows, recipe={"generator": "circuit", "seed": SEED,
        "indices": list(INDICES), "split": "sealed", "style": "default", "mode": "obs"})
    result = {"format": "notebook-rl-local-ladder-preparation-v1",
        "status": "CPU_COHORT_PREPARED_PENDING_GATES", "training_allowed": False,
        "evaluation_allowed": False, "gpu_evaluation_completed": False,
        "preparation": _descriptor(preparation), "input_sha256": bindings,
        "source_before_sha256": source_before, "source_after_sha256": source_after,
        "source_revision": revision, "pool": _descriptor(out / "pool.json"),
        "records_manifest": _descriptor(out / "records/manifest.yaml"),
        "image_root": str(out / "images"), "count": pool["count"],
        "problem_ids": [row.actor.problem_id for row in rows], "training_worlds": worlds,
        "training_node_counts": sorted(train_nodes), "ladder_node_counts": sorted(ladder_nodes),
        "topology_families": [{"sources": ns, "series_stages": n, "node_count": ns + n + 1}
                              for ns, n in sorted(set(topologies))], "topology_family_count": 3,
        "separation": separation, "deterministic_generation_checked": True,
        "two_exact_solvers_checked": True, "new_exclusion_scan_performed": False,
        "new_rights_review_performed": False, "new_render_review_performed": False,
        "limitations": ["Ten value/query instances from exactly three native ladder topology families.",
            "Node-count separation proves complete-graph nonisomorphism only; no subgraph, semantic or pretraining-overlap guarantee.",
            "Program checks do not certify render semantics or rights; no new exclusion scan or human review.",
            "Actual evaluation requires new separate data, protocol and execution gates; no model or scorer runs here."]}
    result["source_after_sha256"] = _preserved(source_before, bindings)
    _write(out / "preparation.json", _bytes(result) + b"\n")
    return result


def load_ladder_actors(cohort: Path) -> list[ActorItem]:
    """Public vault loader: never opens verifier/gold or lineage payloads."""
    root = _safe_root(cohort)
    manifest = _json(root / "preparation.json")
    if (manifest.get("format") != "notebook-rl-local-ladder-preparation-v1"
            or manifest.get("training_allowed") is not False or manifest.get("evaluation_allowed") is not False
            or manifest.get("records_manifest") != _descriptor(root / "records/manifest.yaml")):
        raise ValueError("local ladder: incomplete or changed cohort")
    actors = read_actors(root / "records")
    if len(actors) != 10 or [a.problem_id for a in actors] != manifest.get("problem_ids"):
        raise ValueError("local ladder: actor coverage changed")
    for actor in actors:
        for image in load_images(actor, root / "images"):
            image.close()
    return actors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_ladder_data(args.preparation, args.output)
    print(json.dumps({key: result[key] for key in ("status", "count", "training_allowed", "evaluation_allowed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
