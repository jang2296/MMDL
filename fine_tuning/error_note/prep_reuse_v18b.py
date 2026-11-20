"""CPU-only, request-by-request import of one predeclared F0 preparation source.

The sealed producer evidence names attempt 7, model files, generator dependencies,
precision and backend. Missing evidence is a cache miss, never inferred hardware proof.
"""
from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .audit import digest, text_sha256
from . import prep_v16b as prep

VERSION = "v18b-f0-raw-reuse-v1"
SOURCE_VERSION = "v18b-f0-source-evidence-v1"
ATTEMPT7_F0_SHA256 = "fff61c99a9e111aadd56c83175385ca9f2610b8e24c3850405c200b7aee22d17"
LEGACY_ENGINE = {"context_limit": 24576, "max_num_seqs": 16, "kv_cache_gib": 24.,
                 "gpu_memory_utilization": .65, "precision": "bf16", "backend": "vllm-0.11.0"}
# Hash generation dependencies separately from grading, admission and the campaign plan.
DEPENDENCIES = ("error_note/single_vllm.py", "error_note/llm_lora.py",
                "notebook_rl/engines.py", "notebook_rl/input_preparation.py",
                "notebook_rl/prompts.py")


def _legacy_generator_ast(source: str, *, version: str) -> str:
    """Compare complete generator code specialized to the fixed, non-monitor 24K F0 path."""
    class FixedF0(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == "release_version":
                cleanup = ast.parse('if self.lora_request is not None and self.lora_request.lora_int_id == entry["lora_int_id"]:\n'
                                    '    self.lora_request, self.loaded_policy = None, None').body[0]
                expected = ast.dump(cleanup, include_attributes=False)
                # The acknowledged deletion only clears a stale alias; F0 token generation is identical.
                node.body = [statement for statement in node.body
                             if ast.dump(statement, include_attributes=False) != expected]
            return self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id == "rehearsal_profile":
                return ast.Constant(None)
            if isinstance(node.ctx, ast.Load) and node.id in {"protocol_version", "generation_context_limit", "CONTEXT_LIMIT"}:
                return ast.Constant(version if node.id == "protocol_version" else 24576)
            return node

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self" and isinstance(node.ctx, ast.Load):
                if node.attr == "protocol_version":
                    return ast.Constant(version)
                if node.attr == "context_limit":
                    return ast.Constant(24576)
                if node.attr == "rehearsal_profile":
                    return ast.Constant(None)
            return self.generic_visit(node)

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == "self"
                    and isinstance(node.args[1], ast.Constant)):
                if node.args[1].value == "protocol_version":
                    return ast.Constant(version)
                if node.args[1].value == "context_limit":
                    return ast.Constant(24576)
                if node.args[1].value == "rehearsal_profile":
                    return ast.Constant(None)
            if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "request" and node.func.attr == "get"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "evaluation_monitor"):
                return ast.Constant(False)
            return self.generic_visit(node)

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
                   and ((t.attr == "protocol_version" and isinstance(node.value, ast.Constant) and node.value.value == version)
                        or (t.attr == "rehearsal_profile" and isinstance(node.value, ast.Constant) and node.value.value is None)
                        or (t.attr == "context_limit" and isinstance(node.value, ast.Constant)
                            and node.value.value in (24576, 40960))) for t in node.targets):
                return None
            return node

        def visit_arguments(self, node):
            pairs = [(a, d) for a, d in zip(node.kwonlyargs, node.kw_defaults, strict=True)
                     if a.arg not in {"generation_context_limit", "rehearsal_profile"}]
            node.kwonlyargs, node.kw_defaults = [a for a, _ in pairs], [d for _, d in pairs]
            return self.generic_visit(node)

        def visit_Compare(self, node):
            node = self.generic_visit(node)
            if len(node.ops) != 1:
                return node
            try:
                left, right = ast.literal_eval(node.left), ast.literal_eval(node.comparators[0])
            except (ValueError, TypeError):
                return node
            op = node.ops[0]
            if isinstance(op, (ast.In, ast.NotIn)):
                value = left in right
                return ast.Constant(not value if isinstance(op, ast.NotIn) else value)
            if isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                value = left == right
                return ast.Constant(not value if isinstance(op, (ast.NotEq, ast.IsNot)) else value)
            return node

        def visit_BoolOp(self, node):
            node = self.generic_visit(node)
            if all(isinstance(v, ast.Constant) and type(v.value) is bool for v in node.values):
                values = [v.value for v in node.values]
                return ast.Constant(all(values) if isinstance(node.op, ast.And) else any(values))
            if isinstance(node.op, ast.And) and isinstance(node.values[0], ast.Constant) and node.values[0].value is False:
                return ast.Constant(False)
            if isinstance(node.op, ast.Or) and isinstance(node.values[0], ast.Constant) and node.values[0].value is True:
                return ast.Constant(True)
            return node

        def visit_IfExp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
                return node.body if node.test.value else node.orelse
            return node

        def visit_If(self, node):
            node = self.generic_visit(node)
            if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
                return node.body if node.test.value else node.orelse
            return node

    return ast.dump(FixedF0().visit(ast.parse(source)), include_attributes=False)


def generation_dependencies() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: prep.file_sha(root / name) for name in DEPENDENCIES}


def _bound(path: Any, expected: Any) -> dict:
    if (path is None or not re.fullmatch(r"[a-f0-9]{64}", str(expected or ""))
            or Path(path).is_symlink() or prep.file_sha(Path(path)) != expected):
        raise ValueError("v18b F0 import: explicit, unchanged source file and SHA256 required")
    return prep.read_sealed(Path(path))


def _validate_evidence(evidence: dict, source: dict, source_file_sha256: str,
                       model_path: Path) -> list[str]:
    from fine_tuning.modules import BASE_ID, BASE_REVISION

    if (evidence.get("version") != SOURCE_VERSION or evidence.get("producer_attempt") != 7
            or not re.fullmatch(r"[a-f0-9]{40}", str(evidence.get("producer_commit", "")))
            or evidence.get("source_file_sha256") != source_file_sha256
            or evidence.get("source_seal_sha256") != source["sha256"]
            or evidence.get("initial_policy") != "fresh_F0" or source.get("initial_policy") != "fresh_F0"
            or source.get("version") != "v18-preparation-v1"):
        raise ValueError("v18b F0 import: original attempt-7 source evidence does not bind this raw bundle")
    if evidence.get("context_limit") != 24576:
        raise ValueError("v18b F0 import: original 24K prep/effect generation context required")
    model = evidence.get("model", {})
    if (model.get("repo_id") != BASE_ID or model.get("revision") != BASE_REVISION
            or model.get("processor_revision") != BASE_REVISION or not model.get("files")):
        raise ValueError("v18b F0 import: pinned model and processor file inventory required")
    paths = [entry["path"] for entry in model["files"]]
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json",
                "preprocessor_config.json", "model.safetensors.index.json"}
    if (len(paths) != len(set(paths)) or not required <= set(paths)
            or not any(name.endswith(".safetensors") for name in paths)):
        raise ValueError("v18b F0 import: complete model, tokenizer and processor inventory required")
    for entry in model["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("v18b F0 import: unsafe model inventory path")
        path = model_path / relative
        if path.stat().st_size != entry["bytes"] or prep.file_sha(path) != entry["sha256"]:
            raise ValueError("v18b F0 import: model/processor bytes changed")
    misses = []
    expected, actual = evidence.get("generation_dependencies", {}), generation_dependencies()
    changed = {name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name)}
    if "error_note/single_vllm.py" in changed:
        original = evidence.get("generator_source", "")
        current = (Path(__file__).resolve().parent / "single_vllm.py").read_text(encoding="utf-8")
        if (hashlib.sha256(original.encode()).hexdigest() == expected.get("error_note/single_vllm.py")
                and _legacy_generator_ast(original, version="error-note-v18")
                == _legacy_generator_ast(current, version="error-note-v18b")):
            changed.remove("error_note/single_vllm.py")
    if changed:
        misses.append("GENERATOR_DEPENDENCY_MISMATCH:" + ",".join(sorted(changed)))
    if evidence.get("precision") != "bf16" or evidence.get("backend") != "vllm-0.11.0":
        misses.append("PRECISION_OR_BACKEND_MISMATCH")
    if evidence.get("engine_settings") != LEGACY_ENGINE:
        misses.append("LEGACY_ENGINE_SETTINGS_MISMATCH")
    return misses


def index_f0_source(args: Any) -> dict:
    """Make new migration evidence from the unchanged original plan, source files and doctor."""
    source = _bound(args.f0_reuse_source, args.f0_reuse_sha256)
    if args.f0_reuse_sha256 != ATTEMPT7_F0_SHA256:
        raise ValueError("v18b source index: only the predeclared attempt-7 F0 bundle is eligible")
    producer = _bound(args.f0_producer_plan, args.f0_producer_plan_sha256)
    capacity = _bound(args.f0_producer_capacity, args.f0_producer_capacity_sha256)
    if (producer.get("protocol", {}).get("version") != "error-note-v18"
            or producer.get("inputs", {}).get("f0", {}).get("file_sha256") != args.f0_reuse_sha256):
        raise ValueError("v18b source index: original v18 plan does not bind this raw file")
    if (capacity.get("source_sha256") != digest(producer["source"]["actual_files"])
            or capacity.get("version") != "v18-capacity-v1"):
        raise ValueError("v18b source index: original generation capacity receipt does not bind its source")
    dependencies = {name: producer["source"]["actual_files"][name] for name in DEPENDENCIES}
    root = Path(args.f0_producer_source_root)
    for name, expected in dependencies.items():
        if prep.file_sha(root / name) != expected:
            raise ValueError("v18b source index: original generator bytes differ from producer plan")
    for field in ("f0_model_manifest", "f0_producer_doctor"):
        if prep.file_sha(Path(getattr(args, field))) != getattr(args, field + "_sha256"):
            raise ValueError("v18b source index: model/environment evidence bytes changed")
    model = json.loads(Path(args.f0_model_manifest).read_text(encoding="utf-8"))
    doctor = json.loads(Path(args.f0_producer_doctor).read_text(encoding="utf-8"))
    if (producer["protocol"]["initial_checkpoint"] != {"model_id": model["repo_id"], "revision": model["revision"]}
            or doctor.get("environment", {}).get("packages", {}).get("vllm") != "0.11.0"):
        raise ValueError("v18b source index: pinned original model and vLLM 0.11 evidence required")
    return prep.seal(args.output / "f0-source-evidence.json", {"version": SOURCE_VERSION,
        "producer_attempt": 7, "producer_commit": producer["source"]["base_commit"],
        "producer_plan_file_sha256": args.f0_producer_plan_sha256,
        "source_file_sha256": args.f0_reuse_sha256, "source_seal_sha256": source["sha256"],
        "initial_policy": source["initial_policy"], "model": model,
        "model_manifest_file_sha256": args.f0_model_manifest_sha256,
        "precision": producer["protocol"]["precision"], "backend": "vllm-0.11.0",
        "producer_environment_file_sha256": args.f0_producer_doctor_sha256,
        "producer_capacity_file_sha256": args.f0_producer_capacity_sha256,
        "engine_settings": {"context_limit": producer["protocol"]["context_limit"],
            "max_num_seqs": producer["capacity"]["max_num_seqs"],
            "gpu_memory_utilization": producer["capacity"]["gpu_memory_utilization"],
            "kv_cache_gib": capacity["report"]["kv_cache_gib"], "precision": producer["protocol"]["precision"],
            "backend": "vllm-0.11.0"},
        "generation_dependencies": dependencies, "context_limit": 24576,
        "generator_source": (root / "error_note/single_vllm.py").read_text(encoding="utf-8"),
        "migration_only": True, "upload_authorized": False})


def _validate_record(row: dict, item: dict, role: str, *, processor: Any, model_config: Any,
                     image: dict, cap: int, sampling: dict) -> None:
    from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs

    if row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"}):
        raise ValueError("v18b F0 import: raw record seal changed")
    output, stage = row["output"], prep.preparation_stage("error-note-v18b", role)
    generation = output["generation"]
    request = prep.make_request(item, prep.p0(item), stage, cap=cap, sampling=sampling,
        protocol_sha256=row["protocol_sha256"], policy_sha256=row["policy_sha256"])
    prepared, _ = prepare_protocol_inputs(processor, model_config, request["messages"], request["images"], image)
    if (row.get("problem_id") != item["problem_id"] or row.get("stage") != stage
            or row.get("prompt") != prep.p0(item) or row.get("request_sha256") != request["request_sha256"]
            or any(generation.get(key) != request[key] for key in ("request_id", "problem_id", "stage", "seed", "max_tokens"))
            or generation.get("request_sha256") != row["request_sha256"]
            or generation.get("protocol_sha256") != row["protocol_sha256"]
            or generation.get("policy_sha256") != row["policy_sha256"]
            or generation.get("policy_version") != 0 or generation.get("visual_version") != 0
            or generation.get("mixed_version") is not False or generation.get("completion_count") != 1
            or generation.get("input_sha256") != prepared["input_sha256"]
            or generation.get("prompt_token_ids") != prepared["prompt_token_ids"]
            or generation.get("sampling") != sampling or generation.get("sampling_sha256") != digest(sampling)
            or output.get("text") != generation.get("text") or output.get("finish_reason") != generation.get("finish_reason")
            or not isinstance(generation.get("completion_token_ids"), list)
            or not generation["completion_token_ids"]
            or len(generation["completion_token_ids"]) > cap):
        raise ValueError("v18b F0 import: request/model input/seed/sampling/raw identity mismatch")
    tokens = generation["completion_token_ids"]
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("v18b F0 import: invalid original completion tokens")
    if processor.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False) != output["text"]:
        raise ValueError("v18b F0 import: raw text differs from original tokens")
    if (output["finish_reason"] not in {"stop", "length"}
            or output["finish_reason"] == "stop" and tokens[-1] not in generation.get("eos_token_ids", [])
            or output["finish_reason"] == "length" and len(tokens) != cap):
        raise ValueError("v18b F0 import: incomplete original stop/token-cap evidence")


def reuse_f0(args: Any, *, processor: Any = None, model_config: Any = None) -> dict:
    """Import matching prep/effect requests and regrade on CPU; never load model weights or CUDA.

    Explicit --allow-f0-raw-reuse gates local import; it does not authorize upload.
    --f0-reuse-source/--f0-reuse-evidence and their SHA256 values pin the only source.
    """
    if getattr(args, "allow_f0_raw_reuse", False) is not True:
        if any(getattr(args, key, None) is not None for key in ("f0_reuse_source", "f0_reuse_evidence")):
            raise ValueError("v18b F0 import: --allow-f0-raw-reuse is required")
        return {"status": "DISABLED", "records": {}}
    if getattr(args, "version", None) != "error-note-v18b" or getattr(args, "precision", None) != "bf16":
        raise ValueError("v18b F0 import: BF16 v18b generation only")
    if getattr(args, "f0_reuse_source", None) is None or getattr(args, "model_path", None) is None:
        raise ValueError("v18b F0 import: explicit raw source and local model path required")
    source_path = Path(args.f0_reuse_source)
    source = _bound(source_path, getattr(args, "f0_reuse_sha256", None))
    evidence = _bound(getattr(args, "f0_reuse_evidence", None), getattr(args, "f0_reuse_evidence_sha256", None))
    selection = prep.select(args.data_root, args.output, local_sequential=args.local_sequential, version=args.version)
    source_sha = prep.file_sha(source_path)
    current_source = prep.single.source_manifest()
    global_misses = _validate_evidence(evidence, source, source_sha, args.model_path)
    from .protocol import EVAL_SAMPLING_V16B, SINGLE_IMAGE_V14

    if source.get("image") != SINGLE_IMAGE_V14 or source.get("cap") != 8192 or args.local_sequential:
        global_misses.append("IMAGE_OR_CAP_MISMATCH")
    if not global_misses and (processor is None or model_config is None):
        from transformers import AutoConfig, AutoProcessor
        processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True)
        model_config = AutoConfig.from_pretrained(str(args.model_path), local_files_only=True)
    data = prep.data_api(args.version)
    accepted: dict[str, list[dict]] = {}
    decisions: list[dict] = []
    for role in ("prep", "effect"):
        items = data.load_items(args.data_root, role)
        if args.local_sequential:
            items = ([item for item in items if item["problem_id"] in selection["prep_ids"]]
                     if role == "prep" else items[:2])
        old_rows = source.get(role, [])
        lookup = {row["problem_id"]: row for row in old_rows}
        if len(lookup) != len(old_rows):
            raise ValueError("v18b F0 import: duplicate source request; selecting an answer is forbidden")
        grade, accepted[role] = prep.grader(args.data_root, role), []
        for item in items:
            row = lookup.get(item["problem_id"])
            reason = global_misses[0] if global_misses else "MISSING_SOURCE_REQUEST" if row is None else None
            if reason is None and row is not None:
                try:
                    _validate_record(row, item, role, processor=processor, model_config=model_config,
                        image=SINGLE_IMAGE_V14, cap=8192, sampling=EVAL_SAMPLING_V16B)
                except ValueError as error:
                    reason = str(error)
            decision = {"problem_id": item["problem_id"], "role": role, "status": "MISS" if reason else "REUSED", "reason": reason}
            if reason is None and row is not None:
                output = deepcopy(row["output"])
                status, extracted = grade(item, output)
                output.update(status=status, extracted=extracted)
                raw = row["output"]["generation"]
                fingerprints = {"raw_text_sha256": text_sha256(output["text"]),
                    "completion_tokens_sha256": digest(raw["completion_token_ids"]),
                    "prompt_tokens_sha256": digest(raw["prompt_token_ids"]), "request_sha256": row["request_sha256"],
                    "input_sha256": raw["input_sha256"], "sampling_sha256": raw["sampling_sha256"]}
                imported = {key: deepcopy(value) for key, value in row.items() if key not in {"sha256", "output"}}
                imported.update(output=output, reuse={"producer_commit": evidence["producer_commit"],
                    "producer_attempt": 7, "producer_record_sha256": row["sha256"],
                    "source_file_sha256": source_sha, "evidence_sha256": evidence["sha256"],
                    "validator_source_sha256": digest(current_source["actual_files"]),
                    "rescore_only": True, "original_generation_unchanged": True, **fingerprints})
                imported["sha256"] = digest(imported)
                accepted[role].append(imported)
                decision.update(fingerprints)
            decisions.append(decision)
    original_code = evidence.get("generator_source")
    current_code = (Path(__file__).resolve().parent / "single_vllm.py").read_text(encoding="utf-8")
    code_comparison = ({"producer_file_sha256": evidence["generation_dependencies"]["error_note/single_vllm.py"],
        "current_file_sha256": generation_dependencies()["error_note/single_vllm.py"],
        "producer_legacy_ast_sha256": text_sha256(_legacy_generator_ast(original_code, version="error-note-v18")),
        "current_legacy_ast_sha256": text_sha256(_legacy_generator_ast(current_code, version="error-note-v18b")),
        "scope": "complete source specialized to non-monitor F0, context24576"} if original_code else None)
    receipt = prep.seal(args.output / "f0-reuse.json", {"version": VERSION,
        "status": "VERIFIED_PARTIAL" if any(accepted.values()) else "NO_MATCH",
        "source_file_sha256": source_sha, "source_seal_sha256": source["sha256"],
        "selection_sha256": selection["sha256"],
        "producer_commit": evidence["producer_commit"], "evidence_sha256": evidence["sha256"],
        "validator_commit": current_source["base_commit"],
        "validator_source_sha256": digest(current_source["actual_files"]),
        "generation_dependencies": generation_dependencies(), "global_misses": global_misses,
        "engine_settings": LEGACY_ENGINE,
        "generator_code_comparison": code_comparison,
        "role_permissions": {"prep": "training", "effect": "evaluation_only", "monitor": "evaluation_only",
                             "reasoning_monitor": "evaluation_only"},
        "monitor_reuse": "SEPARATE_ACCURACY_8192_AND_SHORT_64_REQUIRE_MATCHED_DRAWS", "decisions": decisions,
        "upload_authorized": False, "model_loaded": False, "records": accepted})
    return receipt


def rescore_f0(args: Any) -> dict:
    """Diagnostic CPU rescore of unchanged raw records; it never admits training or reuse."""
    source = _bound(getattr(args, "f0_reuse_source", None), getattr(args, "f0_reuse_sha256", None))
    data, results = prep.data_api(args.version), {}
    for role in ("prep", "monitor", "effect"):
        items = {item["problem_id"]: item for item in data.load_items(args.data_root, role)}
        grade, raw_grade = prep.grader(args.data_root, role), prep.grader(args.data_root, role, raw_v8=True)
        records, seen = [], set()
        for row in source.get(role, []):
            pid = row["problem_id"]
            if pid in seen or pid not in items or row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"}):
                raise ValueError("v18b CPU rescore: duplicate/unknown parent or changed original record")
            seen.add(pid)
            status, extracted = grade(items[pid], row["output"])
            raw_status, raw_extracted = raw_grade(items[pid], row["output"])
            records.append({"problem_id": pid, "original_record_sha256": row["sha256"],
                "raw_text_sha256": text_sha256(row["output"]["text"]), "training_status": status,
                "training_extracted": extracted, "raw_v8_status": raw_status, "raw_v8_extracted": raw_extracted})
        results[role] = records
    current = prep.single.source_manifest()
    return prep.seal(args.output / "f0-rescore.json", {"version": "v18b-f0-cpu-rescore-v1",
        "source_file_sha256": prep.file_sha(Path(args.f0_reuse_source)), "source_seal_sha256": source["sha256"],
        "validator_commit": current["base_commit"], "validator_source_sha256": digest(current["actual_files"]),
        "diagnostic_only": True, "training_admitted": False, "model_loaded": False, "roles": results})
