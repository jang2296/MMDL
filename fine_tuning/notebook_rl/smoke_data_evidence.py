"""Actual CPU data measurements for the separately authorized, program-only smoke.

Components are not approvals. Assembly requires the real full exclusion receipt;
an explicit metadata rebind can preserve an earlier review of identical inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image

from . import approval, contamination
from .cli import artifact_path
from .dataset import SPLITS, _bytes, _write
from .generators import get_generator, graph, graph_pair
from .schema import ActorItem, Lineage, VerifierItem, load_protocol
from .vault import _read, _safe_root, _unique, read_actors, read_lineage, read_verifier

_HERE = Path(__file__).resolve().parent
_SOURCES = {"gen-graph-v1": "graph", "gen-graph-pair-v1": "graph_pair"}
_LIMITS = ["AI/program review is fallible; human_verified=false.",
           "Python guards are not an OS or hostile same-user isolation boundary.",
           "graph_pair plain-mode question still requests observation annotations; this is not a pure answer-only control.",
           "D4 counterfactual probes are not approved training examples or model-performance measurements."]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    return _sha(_bytes(value))


def _json(path):
    return json.loads(_read(Path(path)), object_pairs_hook=_unique)


def _descriptor(path):
    return {"path": str(Path(path).absolute()), "file_sha256": _sha(_read(Path(path)))}


def _bound_json(descriptor):
    raw = _read(Path(descriptor["path"]))
    if _sha(raw) != descriptor["file_sha256"]:
        raise ValueError("smoke evidence: artifact changed")
    return json.loads(raw, object_pairs_hook=_unique)


def _inputs(root):
    root = _safe_root(root)
    items, files, manifests = {}, {}, {}
    for split in SPLITS:
        part = root / split
        manifests[split] = _json(part / "manifest.yaml")
        actors, lineages, verifiers = read_actors(part), read_lineage(part), read_verifier(part)
        if {actor.problem_id for actor in actors} != set(lineages) or set(lineages) != set(verifiers):
            raise ValueError("smoke evidence: mismatched vault identities")
        for name in ("manifest.yaml", "actor_inputs.jsonl", "lineage.jsonl", "verifier_private.jsonl"):
            files[str(part / name)] = _sha(_read(part / name))
        for actor in actors:
            pid = actor.problem_id
            lineage = lineages[pid]
            if (pid in items or lineage.split != split or lineage.label_provenance != "PROGRAM"
                    or lineage.benchmark_derived or lineage.model_touched or lineage.source not in _SOURCES):
                raise ValueError("smoke evidence: unsupported or forbidden candidate")
            images = tuple(_read(root / "images" / rel) for rel in actor.image_paths)
            if tuple(_sha(raw) for raw in images) != actor.image_sha256:
                raise ValueError("smoke evidence: original image changed")
            for rel, digest in zip(actor.image_paths, actor.image_sha256, strict=True):
                files[str(root / "images" / rel)] = digest
            items[pid] = (actor, verifiers[pid], lineage, images)
    if not items or len(items) > 64:
        raise ValueError("smoke evidence: bounded nonempty pool required")
    return items, files, manifests


def _revision(module):
    return _sha(_read(Path(module.__file__)) + b"\0" + _read(_HERE / "generators/common.py")
                + b"".join(b"\0" + _read(Path(dep.__file__)) for dep in getattr(module, "SOURCE_DEPENDENCIES", ())))


def _solvers(world, template):
    traversal = template.rsplit("-", 1)[-1] if world["kind"] == "tree" else "postorder"
    graph._validate_world(world)
    return graph._answers(world, traversal)


def intervene(world, template):
    """One predetermined, label-changing structural/weight intervention; no model selection."""
    changed = copy.deepcopy(world)
    kind = world["kind"]
    if kind == "shortest":
        # Every simple A->H path leaves A exactly once. The all-1 and all-9
        # outgoing-A settings differ by exactly eight, so one differs from gold.
        low = copy.deepcopy(world)
        low["edges"] = [[a, b, 1 if a == "A" else weight] for a, b, weight in world["edges"]]
        use_low = _solvers(low, template)[0] != _solvers(world, template)[0]
        changed["edges"] = [[a, b, (1 if use_low else 9) if a == "A" else weight]
                            for a, b, weight in world["edges"]]
    elif kind == "dfa":
        final = world["start"]
        for symbol in graph._WORD:
            final = world["trans"][final][symbol]
        accepting = set(world["accepting"])
        accepting.symmetric_difference_update({final})
        others = sorted(set(world["trans"]) - {final})
        if not accepting:
            accepting.add(others[0])
        if len(accepting) == 4:
            accepting.remove(others[0])
        changed["accepting"] = sorted(accepting)
    else:
        if kind == "bfs":
            reached = graph._bfs(world)
            pair = (next(node for node in reached if node != "A"),
                    next(node for node in world["nodes"] if node not in reached))
        else:
            leaves = sorted(set("ABCDEFG") - set(world["children"]))
            pair = (leaves[0], leaves[1])
        def rename(name):
            return pair[1] if name == pair[0] else pair[0] if name == pair[1] else name
        if kind == "bfs":
            changed["edges"] = [[rename(a), rename(b), weight] for a, b, weight in world["edges"]]
            changed["positions"] = {rename(name): position for name, position in world["positions"].items()}
        else:
            changed["children"] = {rename(parent): {side: rename(child) for side, child in children.items()}
                                   for parent, children in world["children"].items()}
    _solvers(changed, template)
    return changed


def measure_interventions(source, template, example, output, *, count=100):
    """Keep every attempted pair, including errors and unchanged answers, on disk."""
    module = get_generator(_SOURCES[source])
    actor, _, lineage, _ = example
    start = int(actor.problem_id.split(":")[-2])
    style = "shift" if lineage.split == "style_sealed" else "default"
    rows = []
    for number in range(count):
        row = {"attempt": number, "source": source, "template": template, "status": "FAIL"}
        try:
            index = start + number * (12 if source == "gen-graph-v1" else 1)
            p = module.generate(index, lineage.generator_seed, lineage.split, actor.mode, style)
            module.self_check(p)
            if p.lineage.template_id != template:
                raise ValueError("intervention template changed")
            world = p.verifier.accept["graph_world"]
            altered = intervene(world, template)
            before, before_independent = _solvers(world, template)
            after, after_independent = _solvers(altered, template)
            if source == "gen-graph-pair-v1":
                images, printed = graph_pair._render(altered, style)
            else:
                raw, printed = graph._render(altered, style)
                images = (raw,)
            after_actor = replace(p.actor, image_paths=tuple(_sha(raw) + ".png" for raw in images),
                                  image_sha256=tuple(_sha(raw) for raw in images))
            fields = {key: value for key, value in asdict(p.actor).items() if key not in ("image_paths", "image_sha256")}
            after_fields = {key: value for key, value in asdict(after_actor).items() if key not in ("image_paths", "image_sha256")}
            row.update(problem_id=p.actor.problem_id, original_structure_hash=p.lineage.structure_hash,
                       changed_world=altered, before=before, after=after,
                       independent_before=before_independent, independent_after=after_independent,
                       question_unchanged=p.actor.question == after_actor.question,
                       nonimage_inputs_unchanged=fields == after_fields, nonimage_inputs_sha256=_hash(fields),
                       before_actor=asdict(p.actor), after_actor=asdict(after_actor),
                       before_images_sha256=list(p.actor.image_sha256), after_images_sha256=[_sha(raw) for raw in images],
                       printed_text_sha256=_hash(printed))
            for raw in (*p.image_bytes, *images):
                path = output / "images" / (_sha(raw) + ".png")
                if not path.exists():
                    _write(path, raw)
                elif _sha(_read(path)) != _sha(raw):
                    raise ValueError("intervention image changed")
            if (before == before_independent == p.verifier.gold and after == after_independent
                    and before != after and tuple(images) != p.image_bytes):
                row["status"] = "PASS"
        except Exception as exc:
            row["error_type"] = type(exc).__name__
        _write(output / "attempts" / f"{source}-{template}-{number:03d}.json", _bytes(row) + b"\n")
        rows.append(row)
    return {"pairs": len(rows), "answer_changed": sum(row["status"] == "PASS" for row in rows),
            "question_unchanged": all(row.get("question_unchanged") is True for row in rows),
            "nonimage_inputs_unchanged": all(row.get("nonimage_inputs_unchanged") is True for row in rows),
            "attempts_sha256": _hash(rows), "attempts_recorded": len(rows)}


def _guard_worker(root, benchmark):
    from .guards import install_guards
    private = root / "train/verifier_private.jsonl"
    install_guards(forbidden_roots=[benchmark.parent], vault_files=[private], allow_vault=False, allow_benchmark=False)
    result = {"separate_process": True, "python_guard_not_os_sandbox": True}
    for key, action in (("private_open_denied", lambda: private.read_bytes()),
                        ("benchmark_open_denied", lambda: benchmark.read_bytes()),
                        ("teacher_import_denied", lambda: __import__("openai"))):
        try:
            action()
        except PermissionError:
            result[key] = True
        else:
            result[key] = False
    return {"pid": os.getpid(), "measurements": result}


def measure_isolation(root, benchmark):
    if not benchmark.exists():
        raise ValueError("smoke evidence: an existing benchmark file is required for actual denial test")
    result = subprocess.run([sys.executable, "-m", "fine_tuning.notebook_rl.smoke_data_evidence", "--guard-worker", str(root), str(benchmark)],
                            capture_output=True, check=False, timeout=30)
    if result.returncode:
        raise ValueError("smoke evidence: isolated denial worker failed")
    record = json.loads(result.stdout, object_pairs_hook=_unique)
    if record["pid"] == os.getpid() or not all(record["measurements"].values()):
        raise ValueError("smoke evidence: isolation denial failed")
    return record


def produce_components(root, review_descriptor, benchmark, output):
    root = _safe_root(Path(root))
    output = artifact_path(Path(output))
    if output.exists():
        raise FileExistsError("preserve existing evidence")
    items, files, manifests = _inputs(root)
    source_paths = [Path(__file__), Path(graph.__file__), Path(graph_pair.__file__),
                    *(_HERE / name for name in ("dataset.py", "schema.py", "vault.py", "guards.py", "prompts.py", "prompts_v1.yaml", "generators/common.py"))]
    source_files = {str(path): _sha(_read(path)) for path in source_paths}
    original_manifest = _json(root / "manifest.yaml")
    original_sha = _sha(_read(root / "manifest.yaml"))
    review = _bound_json(review_descriptor)
    if (review.get("status") != "AI_REVIEWED_PROGRAM_VERIFIED" or review.get("build_manifest_sha256") != original_sha
            or review.get("build_root") != str(root) or review.get("blockers") != []
            or review["provenance"].get("human_verified") is not False
            or review["provenance"].get("labels_modified") is not False):
        raise ValueError("smoke evidence: original AI review binding required")
    for path, digest in files.items():
        # The review binds PNG bytes through each sample's ordered image SHA;
        # its separate file ledger covers vaults/manifests, not image paths.
        if Path(path).is_relative_to(root / "images"):
            continue
        if review["bound_files_sha256"].get(path) != digest:
            raise ValueError("smoke evidence: review input binding changed")
    samples = review["samples"]
    if ({sample["problem_id"] for sample in samples} != set(items) or len(samples) != len(items)
            or any(sample["image_sha256"] != list(items[sample["problem_id"]][0].image_sha256)
                   or sample.get("mismatches") != 0 or sample.get("unresolved") != []
                   or sample.get("printed_text_slots_consistent") is not True for sample in samples)):
        raise ValueError("smoke evidence: complete original-image review required")
    output.mkdir(mode=0o700)
    checks = []
    templates: dict[str, tuple[ActorItem, VerifierItem, Lineage, tuple[bytes, ...]]] = {}
    for pid, (actor, verifier, lineage, images) in sorted(items.items()):
        module = get_generator(_SOURCES[lineage.source])
        style = "shift" if lineage.split == "style_sealed" else "default"
        p = module.generate(int(pid.split(":")[-2]), lineage.generator_seed, lineage.split, actor.mode, style)
        module.self_check(p)
        expected_lineage = replace(p.lineage, source_revision=_revision(module))
        before, independent = _solvers(verifier.accept["graph_world"], lineage.template_id)
        checks.append({"problem_id": pid, "repeat_match": (p.actor, p.verifier, expected_lineage, p.image_bytes)
                       == (actor, verifier, lineage, images), "answer": before, "independent_answer": independent,
                       "solver_match": before == independent == verifier.gold})
        templates.setdefault(lineage.source + ":" + lineage.template_id, (actor, verifier, lineage, images))
    d4 = {key: measure_interventions(example[2].source, example[2].template_id, example, output)
          for key, example in sorted(templates.items())}
    leaks = {"structure_cross_split": 0, "image_cross_split": 0}
    for key, extract in (("structure_cross_split", lambda row: [row[2].structure_hash]),
                         ("image_cross_split", lambda row: row[0].image_sha256)):
        owners = defaultdict(set)
        for row in items.values():
            for value in extract(row):
                owners[value].add(row[2].split)
        leaks[key] = sum(len(splits) > 1 for splits in owners.values())
    isolation = measure_isolation(root, Path(benchmark))
    ids_sha = _hash(sorted(items))
    gates = {
        "D1": {"checked_ids_sha256": ids_sha, "items_checked": len(items),
               "repeat_mismatches": sum(not row["repeat_match"] for row in checks)},
        "D2": {"checked_ids_sha256": ids_sha, "items_checked": len(items),
               "solver_mismatches": sum(not row["solver_match"] for row in checks), "independent_solvers": True},
        "D3": {"sampling_seed": 3407, "samples": samples}, "D4": {"templates": d4}, "D5": leaks,
        "D7": {key: review["actor_leak_review"][key] for key in ("reviewed_actor_files", "warnings_sha256", "unresolved")},
        "D8": isolation["measurements"]}
    if any(_sha(_read(Path(path))) != digest for path, digest in {**files, **source_files}.items()):
        raise ValueError("smoke evidence: inputs changed during measurement")
    result = {"format": "notebook-rl-smoke-data-components-v1", "status": "COMPONENTS_MEASURED_NOT_APPROVAL",
              "training_allowed": False, "build_root": str(root), "original_build_manifest": original_manifest,
              "original_build_manifest_sha256": original_sha, "input_files": files, "input_files_sha256": _hash(files),
              "source_files": source_files,
              "review": review_descriptor, "review_provenance": review["provenance"], "gates": gates,
              "repeat_solver_records": checks, "guard_process": isolation, "limitations": _LIMITS,
              "pending": ["D6 requires actual full exclusion receipt and final metadata bindings"],
              "producer": _descriptor(Path(__file__))}
    _write(output / "components.json", _bytes(result) + b"\n")
    return result


def assemble(components_descriptor, root, receipt_descriptor, binding, scope, output, *, source_registry, metadata_rebind=None):
    """Bind measured components only after real exclusion; never create GPU approval."""
    components = _bound_json(components_descriptor)
    receipt = _bound_json(receipt_descriptor)
    root = _safe_root(Path(root))
    items, files, manifests = _inputs(root)
    manifest = _json(root / "manifest.yaml")
    if components["producer"] != _descriptor(Path(__file__)) or components["input_files_sha256"] != _hash(components["input_files"]):
        raise ValueError("smoke evidence: stale components or producer")
    if any(_sha(_read(Path(path))) != digest for path, digest in components["source_files"].items()):
        raise ValueError("smoke evidence: measured source changed")
    if (receipt.get("version") != contamination.VERSION
            or receipt.get("status") != "EXCLUSION_CHECKED_WITH_LIMITS" or receipt.get("blockers") != []
            or receipt.get("training_allowed") is not False or receipt.get("approval_granted") is not False
            or receipt.get("partial_name") is not None or receipt.get("receipt_sha256") != _hash({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or set(receipt.get("steps", {})) != set(contamination._STEPS)
            or any(step.get("status") != "COMPLETE" for step in receipt["steps"].values())
            or receipt.get("input_files") != files or receipt.get("input_files_sha256") != _hash(files)
            or manifest.get("exclusion_receipt_sha256") != receipt_descriptor["file_sha256"]):
        raise ValueError("smoke evidence: actual complete exclusion receipt required")
    from .adjudication import validate as reviewed_candidates
    inventory = {}
    for pid, row in items.items():
        pixels = []
        for raw in row[3]:
            with Image.open(io.BytesIO(raw)) as image:
                pixels.append(contamination.audit.image_fingerprint(image)["pixels_sha256"])
        inventory[pid] = {"structure_hash": row[2].structure_hash,
                          "image_sha256": list(row[0].image_sha256), "pixel_sha256": pixels}
    quarantined = contamination.family_quarantine(inventory, reviewed_candidates(receipt, inventory))
    if (receipt.get("quarantined_ids") != quarantined["quarantined_ids"]
            or receipt.get("allowlist") != quarantined["allowlist"]
            or not any(pid in quarantined["allowlist"] and row[2].split == "train"
                       for pid, row in items.items())):
        raise ValueError("smoke evidence: reviewed blacklist/allowlist mismatch or no eligible train items")
    old = components["original_build_manifest"]
    if _hash(old) != _hash({**manifest, "exclusion_receipt_sha256": old["exclusion_receipt_sha256"]}):
        raise ValueError("smoke evidence: changed candidate manifest beyond exclusion binding")
    if metadata_rebind is None:
        if (root != Path(components["build_root"]) or files != components["input_files"]
                or _sha(_read(root / "manifest.yaml")) != components["original_build_manifest_sha256"]):
            raise ValueError("smoke evidence: explicit old/new metadata mapping required")
    else:
        mapping = _bound_json(metadata_rebind)
        old_root = Path(components["build_root"])
        expected = {str(old_root / Path(path).relative_to(root)): digest for path, digest in files.items()}
        if (mapping.get("old_to_new_files") != {str(old_root / Path(path).relative_to(root)): path for path in files}
                or expected != components["input_files"]
                or any(_sha(_read(Path(path))) != digest for path, digest in expected.items())
                or mapping.get("kind") != "METADATA_REBIND_IDENTICAL_INPUT_BYTES"
                or mapping.get("new_build_manifest_sha256") != _sha(_read(root / "manifest.yaml"))
                or mapping.get("original_build_manifest_sha256") != components["original_build_manifest_sha256"]):
            raise ValueError("smoke evidence: invalid immutable metadata rebind")
    expected_binding = {"build_manifest_sha256": _sha(_read(root / "manifest.yaml")),
                        "exclusion_receipt_sha256": receipt_descriptor["file_sha256"],
                        "input_files_sha256": _hash(files), "protocol_sha256": load_protocol(_HERE / "protocol_v1.yaml")["sha256"],
                        **{key: receipt[key] for key in ("index_sha256", "index_file_sha256", "source_registry_sha256", "evidence_sha256")}}
    if binding != expected_binding or scope.get("input_files_sha256") != _hash(files) or scope.get("producer") != _descriptor(Path(__file__)):
        raise ValueError("smoke evidence: final binding or scope mismatch")
    gates = copy.deepcopy(components["gates"])
    gates["D6"] = {"exclusion_receipt_sha256": receipt_descriptor["file_sha256"]}
    gates["D7"]["reviewed_actor_files"] = {path: sha for path, sha in files.items() if Path(path).name == "actor_inputs.jsonl"}
    if gates["D7"]["warnings_sha256"] != _hash({split: value["leak_review"] for split, value in manifests.items()}):
        raise ValueError("smoke evidence: leak warning metadata changed")
    proofs = {key: {"measurements": value, **({"provenance": components["review_provenance"]} if key in ("D3", "D7") else {})}
              for key, value in gates.items()}
    # Reuse the real scalar gate checks; no failed measurement becomes PASS.
    registry = _bound_json(source_registry)
    if _hash(registry) != binding["source_registry_sha256"]:
        raise ValueError("smoke evidence: source registry binding changed")
    approval._measurement_scope(scope, binding, load_protocol(_HERE / "protocol_v1.yaml"), {}, limited_smoke=True)
    as_items = {pid: {"actor": row[0], "verifier": row[1], "lineage": row[2]} for pid, row in items.items()}
    approval._data_gates(proofs, as_items, manifests, files, registry, receipt_descriptor["file_sha256"], 3407, limited_smoke=True)
    output = artifact_path(Path(output))
    output.mkdir(mode=0o700, exist_ok=False)
    report = {"format": "notebook-rl-data-gates-v1", "status": "COMPLETE", "binding": binding, "scope": scope, "gates": {}}
    for key, proof in proofs.items():
        proof.update(gate=key, status="PASS", binding_sha256=_hash(binding), scope_sha256=_hash(scope),
                     producer_sha256=scope["producer"]["file_sha256"], components=components_descriptor,
                     review=components["review"], metadata_rebind=metadata_rebind)
        path = output / (key + ".json")
        _write(path, _bytes(proof) + b"\n")
        report["gates"][key] = {"status": "PASS", "artifact": _descriptor(path)}
    _write(output / "data_report.json", _bytes(report) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--benchmark-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = produce_components(args.build, {"path": str(args.review), "file_sha256": args.review_sha256}, args.benchmark_probe, args.out)
    print(json.dumps({"status": result["status"], "training_allowed": False, "pending": result["pending"]}))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--guard-worker"]:
        print(json.dumps(_guard_worker(Path(sys.argv[2]), Path(sys.argv[3]))))
    else:
        raise SystemExit(main())
