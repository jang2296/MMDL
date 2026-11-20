"""Explicit continuation across reviewed operational fixes with the training contract retained."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess

from .audit import digest
from .collect import write_json
from .prep_v16b import file_sha, read_sealed, seal

VERSION = "v18b-hotfix-resume-v1"
# Only these operational bodies may differ. Their signatures and all other source stay bound.
OPERATIONAL_BODIES = {
    "fine_tuning/error_note/single_vllm.py": {
        "SingleVllmSampler.release_version", "SingleVllmSampler.release_idle_versions",
        "SingleVllmSampler._swap_adapter", "SingleVllmSampler.release"},
    "fine_tuning/error_note/v18b_recovery.py": {"RecoveryCopyHook.poll", "RecoveryCopyHook.close"},
    # 2026-11-19 (user decision after the TRUNC_UP stop at RL block 49): the truncation stop rule became a recorded
    # warning and a campaign stopped only on it continues from that boundary. Stop rules are operational; prompts,
    # rewards, masks, data and schedule stay bound. The allowlist itself is reviewed source (see _normalized).
    "fine_tuning/error_note/guard_v16b.py": {"GuardV16B.update", "GuardV17.__init__", "GuardV17.update"},
    "fine_tuning/error_note/campaign_v16b.py": {"_run_campaign"},
    "fine_tuning/error_note/v18b_hotfix.py": {"_normalized"},
}
EXCLUDED = {"tests", "runs", "generated", "__pycache__"}


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def _commit(repo: Path, commit: str) -> None:
    if not isinstance(commit, str) or re.fullmatch(r"[a-f0-9]{40}", commit) is None:
        raise ValueError("hotfix resume requires exact 40-hex commits")
    if _git(repo, "rev-parse", commit + "^{commit}").decode().strip() != commit:
        raise ValueError("hotfix resume commit is missing")


def committed_source(repo: Path, commit: str) -> dict:
    _commit(repo, commit)
    names = [name for name in _git(repo, "ls-tree", "-r", "--name-only", commit, "--", "fine_tuning").decode().splitlines()
             if Path(name).suffix == ".py" and not EXCLUDED.intersection(Path(name).parts[1:])]
    if any("\n" in name or "\r" in name for name in names):
        raise ValueError("hotfix resume unsafe source path")
    stream = BytesIO(subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"],
        input="".join(f"{commit}:{name}\n" for name in names).encode(), check=True, capture_output=True).stdout)
    records = {}
    for name in names:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("hotfix resume source blob is missing")
        content = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("hotfix resume source framing differs")
        records[Path(name).relative_to("fine_tuning").as_posix()] = hashlib.sha256(content).hexdigest()
    if not records:
        raise ValueError("hotfix resume has no committed training source")
    return {"base_commit": commit, "actual_files": records}


def _normalized(source: bytes, path: str) -> str:
    tree = ast.parse(source.decode("utf-8"))
    allowed = OPERATIONAL_BODIES.get(path, set())
    if path == "fine_tuning/error_note/v18b_hotfix.py":
        # An operator fix may extend the reviewed allowlist; every other validator statement stays bound.
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "OPERATIONAL_BODIES" for target in node.targets))]

    def visit(body: list, parents: tuple = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, parents + (node.name,))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ".".join(parents + (node.name,)) in allowed:
                    node.body = [ast.Pass()]
    visit(tree.body)
    return ast.dump(tree, include_attributes=False)


def contract_diff(repo: Path, old: str, new: str) -> list[dict]:
    """Reject changed inputs, prompts, objectives, masks, module signatures and nonoperational code."""
    _commit(repo, old)
    _commit(repo, new)
    if old == new or subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", old, new],
                                    capture_output=True).returncode != 0:
        raise ValueError("hotfix resume requires a descendant operational fix")
    changed = []
    for name in _git(repo, "diff", "--name-only", "--no-renames", old, new).decode().splitlines():
        path = Path(name)
        documentation = path.suffix == ".md" and (path.name == "README.md" or path.parts[0] in {"docs", "claudedocs"})
        if documentation or path.parts[0] == "tests" or path.parts[:2] == ("fine_tuning", "tests"):
            continue
        if path.suffix != ".py" or path.parts[0] != "fine_tuning":
            raise ValueError(f"hotfix resume changes a bound input: {name}")
        try:
            before, after = (_git(repo, "show", f"{commit}:{name}") for commit in (old, new))
        except subprocess.CalledProcessError as error:
            raise ValueError(f"hotfix resume adds or removes source: {name}") from error
        if _normalized(before, name) != _normalized(after, name):
            raise ValueError(f"hotfix resume changes the training contract: {name}")
        changed.append({"path": name, "before_sha256": hashlib.sha256(before).hexdigest(),
                        "after_sha256": hashlib.sha256(after).hexdigest()})
    return changed


def _checked(ref: dict) -> Path:
    from .v18b_recovery import resolve_recovery_path
    path = resolve_recovery_path(ref["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or file_sha(path) != ref["file_sha256"]:
        raise ValueError("hotfix resume bound artifact changed")
    return path


def closed_boundary(output: Path, plan: dict) -> dict:
    state = read_sealed(output / "campaign-state.json")
    cursor = state["cursor"]
    if (state.get("plan_sha256") != plan["sha256"] or type(cursor.get("slot")) is not int
            or not 0 < cursor["slot"] <= plan["total_slots"]):
        raise ValueError("hotfix resume needs a saved boundary under the original plan")
    consumed = cursor.get("consumed_problem_ids", [])
    ids = [pid for block in plan["blocks"] for pid in block]
    rows = cursor.get("consumed_row_ids", [])
    if consumed != ids[:len(consumed)] or len(rows) != len(set(rows)):
        raise ValueError("hotfix resume consumption cursor is inconsistent")
    metadata_path = _checked(state["checkpoint"])
    metadata = json.loads(metadata_path.read_text())
    if ((metadata_path.parent / "metadata.sha256").read_text().strip() != file_sha(metadata_path)
            or metadata.get("cursor", {}).get("single") != cursor or metadata.get("protocol") != plan["protocol"]
            or metadata.get("step") != cursor["slot"]
            or set(metadata.get("files", {})) != {"adapter.safetensors", "mergers.safetensors", "runtime.pt"}):
        raise ValueError("hotfix resume checkpoint metadata differs")
    for name, checksum in metadata["files"].items():
        path = metadata_path.parent / name
        if path.is_symlink() or file_sha(path) != checksum:
            raise ValueError("hotfix resume checkpoint payload differs")
    for ref in state.get("probe", {}).values():
        _checked(ref)
    for ref in plan["inputs"].values():
        _checked(ref)
    if cursor.get("last_transaction_sha256") is not None:
        transaction = read_sealed(output / "transactions" / f"slot-{cursor['slot'] - 1}.json")
        if (transaction["sha256"] != cursor["last_transaction_sha256"]
                or transaction.get("plan_sha256") != plan["sha256"] or transaction.get("slot") != cursor["slot"] - 1):
            raise ValueError("hotfix resume transaction differs")
    if state.get("runtime") is not None:
        envelope = json.loads(_checked(state["runtime"]).read_text())
        runtime = envelope["state"]
        journal = output / "runtime" / "requests.jsonl"
        if (envelope.get("sha256") != digest(runtime) or runtime.get("plan_sha256") != digest(plan)
                or runtime.get("protocol_sha256") != plan["protocol"]["sha256"]
                or set(runtime["requests"]) != set(runtime["closed"])
                or any(entry["refcount"] for entry in runtime["registry"].values())
                or hashlib.sha256(journal.read_bytes()[:runtime["journal_bytes"]]).hexdigest() != runtime["journal_sha256"]):
            raise ValueError("hotfix resume runtime has open requests or changed evidence")
    return state


def _clean_current(repo: Path, current: dict) -> None:
    if (_git(repo, "rev-parse", "HEAD").decode().strip() != current["base_commit"]
            or _git(repo, "status", "--porcelain", "--untracked-files=all").strip()
            or committed_source(repo, current["base_commit"])["actual_files"] != current["actual_files"]):
        raise ValueError("hotfix resume requires the clean exact published source")


def lineage(output: Path, plan: dict) -> list[dict]:
    """Read the immutable transition records retained in state and in recovery closures."""
    state_path = output / "campaign-state.json"
    state = read_sealed(state_path) if state_path.exists() else {}
    result: list[dict] = []
    previous = plan["source"]["base_commit"]
    for ref in state.get("hotfix_resumes", []):
        record = read_sealed(_checked(ref))
        if (record.get("version") != VERSION or record.get("plan_sha256") != plan["sha256"]
                or record.get("old_commit") != previous
                or record.get("training_contract_sha256") != digest(plan)
                or record.get("previous_authorization_sha256") != (result[-1]["sha256"] if result else None)
                or record.get("new_source", {}).get("base_commit") != record.get("new_commit")):
            raise ValueError("hotfix resume lineage differs from the original training contract")
        result.append(record)
        previous = record["new_commit"]
    if result and (state.get("execution_code_commit") != result[-1]["new_commit"]
                   or state.get("hotfix_lineage_sha256") != digest(result)):
        raise ValueError("hotfix resume saved execution lineage differs")
    return result


def authorize(output: Path, repo: Path, old: str, new: str, reason: str) -> dict:
    plan = read_sealed(output / "campaign-plan.json")
    if plan["protocol"].get("version") != "error-note-v18b" or not isinstance(reason, str) or not reason.strip():
        raise ValueError("hotfix resume needs v18b and an explicit operational reason")
    state = closed_boundary(output, plan)
    previous = lineage(output, plan)
    expected = previous[-1]["new_source"] if previous else plan["source"]
    current = committed_source(repo, new)
    _clean_current(repo, current)
    if expected["base_commit"] != old or committed_source(repo, old)["actual_files"] != expected["actual_files"]:
        raise ValueError("hotfix resume previous committed source differs")
    changes = contract_diff(repo, old, new)
    return {"version": VERSION, "plan_sha256": plan["sha256"], "training_contract_sha256": digest(plan),
            "old_commit": old, "new_commit": new, "new_source": current, "changed_source": changes,
            "reason": reason.strip(), "boundary_state_sha256": state["sha256"],
            "checkpoint": state["checkpoint"], "cursor_sha256": digest(state["cursor"]),
            "slot": state["cursor"]["slot"], "previous_authorization_sha256": previous[-1]["sha256"] if previous else None}


def execution_source(args, plan: dict, current: dict) -> dict:
    """Adopt one explicit authorization before model load; keep original checkpoint binding intact."""
    output = Path(args.output)
    requested = getattr(args, "hotfix_resume", None)
    state_path = output / "campaign-state.json"
    previous = lineage(output, plan)
    repo = Path(getattr(args, "repo", None) or Path(__file__).resolve().parents[2])
    if requested is not None:
        if plan["protocol"].get("version") != "error-note-v18b":
            raise ValueError("hotfix resume is v18b only")
        requested = Path(requested)
        if requested.is_symlink():
            raise ValueError("hotfix resume symlink authorization refused")
        record = read_sealed(requested)
        already = next((row for row in previous if row["sha256"] == record["sha256"]), None)
        if already is None:
            old, new, reason = record.get("old_commit"), record.get("new_commit"), record.get("reason")
            if not isinstance(old, str) or not isinstance(new, str) or not isinstance(reason, str):
                raise ValueError("hotfix resume authorization requires both commits and a reason")
            body = authorize(output, repo, old, new, reason)
            if record != body | {"sha256": digest(body)}:
                raise ValueError("hotfix resume authorization or boundary changed")
            destination = output / ("hotfix-resume-" + record["sha256"] + ".json")
            if destination.exists() and read_sealed(destination) != record:
                raise ValueError("hotfix resume immutable authorization changed")
            if not destination.exists():
                write_json(destination, record)
            state = read_sealed(state_path)
            state = {k: deepcopy(v) for k, v in state.items() if k != "sha256"}
            state.setdefault("hotfix_resumes", []).append({"path": str(destination.absolute()), "file_sha256": file_sha(destination)})
            state.update(execution_code_commit=record["new_commit"], hotfix_lineage_sha256=digest(previous + [record]))
            write_json(state_path, state | {"sha256": digest(state)})
            previous = lineage(output, plan)
    if previous:
        if current["base_commit"] != previous[-1]["new_commit"] or current["actual_files"] != previous[-1]["new_source"]["actual_files"]:
            raise ValueError("hotfix resume current source differs from its authorization")
        _clean_current(repo, current)
        # Recheck each semantic diff so a copied receipt cannot authorize arbitrary source.
        expected_source = plan["source"]["actual_files"]
        for row in previous:
            if (committed_source(repo, row["old_commit"])["actual_files"] != expected_source
                    or committed_source(repo, row["new_commit"]) != row["new_source"]
                    or contract_diff(repo, row["old_commit"], row["new_commit"]) != row["changed_source"]):
                raise ValueError("hotfix resume authorized diff differs")
            expected_source = row["new_source"]["actual_files"]
        return {"execution_code_commit": current["base_commit"], "hotfix_lineage_sha256": digest(previous)}
    if requested is not None or (current["base_commit"] != plan["source"]["base_commit"]
                                 or current["actual_files"] != plan["source"]["actual_files"]):
        raise ValueError("v16b campaign: execution source differs from the prepared source; explicit hotfix authorization required")
    return {}


def resume_spec(original: dict, *, campaign_output: Path, repo: Path, new_commit: str,
                authorization: Path, attempt: Path, run_dir: Path) -> dict:
    """Continue existing campaign artifacts through the normal advance/reload/export/seal commands."""
    from fine_tuning.notebook_rl.pod.chain import check_spec

    original = check_spec(original)
    names = ("campaign-advance", "require-complete", "reload", "export", "training-artifact-seal")
    selected = {row["name"]: deepcopy(row) for row in original["steps"] if row["name"] in names}
    if set(selected) != set(names):
        raise ValueError("hotfix resume requires the original complete v18b training tail")
    advance = selected["campaign-advance"]["argv"]
    if ("fine_tuning.error_note.campaign_v16b" not in advance or "--output" not in advance
            or advance[advance.index("--output") + 1] != str(campaign_output)):
        raise ValueError("hotfix resume spec belongs to another campaign")
    py = advance[0]
    old_export = selected["export"]["argv"][selected["export"]["argv"].index("--export-output") + 1]
    exported = str(Path(old_export).with_name("export-final-" + attempt.name))
    artifact = selected["training-artifact-seal"]["argv"]
    artifact_path = Path(artifact[artifact.index("--artifact-output") + 1])
    if artifact_path.exists():
        raise ValueError("hotfix resume training is already sealed; use its evaluation entry")
    steps = [{"name": "verify-published-source", "argv": [py, "-m", "fine_tuning.error_note.plan", "verify-source",
              "--repo-root", str(repo), "--commit", new_commit, "--output", str(attempt / "source.json")],
              "expect_files": [str(attempt / "source.json")], "timeout_seconds": 300}]
    for name in names:
        row = selected[name]
        argv = row["argv"]
        index = argv.index("fine_tuning.error_note.campaign_v16b")
        tail = argv[index + 1:]
        if tail[tail.index("--output") + 1] != str(campaign_output):
            raise ValueError("hotfix resume tail changes the campaign output")
        tail[tail.index("--code-commit") + 1] = new_commit
        if "--repo" in tail:
            tail[tail.index("--repo") + 1] = str(repo)
        else:
            tail += ["--repo", str(repo)]
        if name == "campaign-advance":
            tail += ["--hotfix-resume", str(authorization)]
        argv[index + 1:] = tail
        row["argv"] = [exported + arg[len(old_export):] if arg == old_export or arg.startswith(old_export + "/") else arg
                       for arg in argv]
        row["expect_files"] = [exported + path[len(old_export):] if path == old_export or path.startswith(old_export + "/") else path
                               for path in row["expect_files"]]
        if name == "reload":
            state = read_sealed(campaign_output / "campaign-state.json")
            plan = read_sealed(campaign_output / "campaign-plan.json")
            prior = campaign_output / f"reload-{state['cursor']['slot']}.json"
            if state.get("status") == "COMPLETE" and state["cursor"]["slot"] == plan["total_slots"] and prior.exists():
                proof = read_sealed(prior)
                if (proof.get("checkpoint") != state["checkpoint"] or proof.get("runtime_equal") is not True
                        or proof.get("greedy_equal") is not True or proof.get("probe_max_abs_difference") != 0):
                    raise ValueError("hotfix resume prior reload differs from its checkpoint")
                row = {"name": name, "argv": [py, "-m", "fine_tuning.notebook_rl.pod.chain", "pin",
                    "--file", str(prior), "--sha256", file_sha(prior)], "expect_files": [str(prior)]}
            else:
                capture = str(attempt / "reload-final.json")
                argv = row["argv"]
                if "capture" not in argv[:index] or "fine_tuning.notebook_rl.pod.chain" not in argv[:index]:
                    raise ValueError("hotfix resume reload needs the original capture wrapper")
                argv[argv.index("--output") + 1] = capture
                row["expect_files"] = [capture]
        steps.append(row)
    meta = deepcopy(original.get("meta", {}))
    meta.update(continuation=True, hotfix_resume=True, code_commit=new_commit, repo_root=str(repo),
                run_dir=str(run_dir), original_campaign_output=str(campaign_output),
                hotfix_authorization={"path": str(authorization), "file_sha256": file_sha(authorization)})
    return check_spec({"version": original["version"], "meta": meta, "steps": steps})


def operator_plan(args) -> dict:
    """Create a new job around the stopped campaign without changing its first-bill clock or inputs."""
    from fine_tuning.budget import measured_balance
    from fine_tuning.notebook_rl.pod.job import content_sha256, validate_job
    old_job = json.loads(Path(args.job).read_text())
    source_spec = Path(args.source_spec)
    argv = old_job["main"]
    if (old_job.get("financial_mode") != "measured_balance_reserve"
            or old_job.get("approval", {}).get("approved_by_user") is not True
            or old_job["approval"].get("approved_sha256") != content_sha256(old_job)
            or argv[argv.index("--spec") + 1] != str(source_spec)
            or argv[argv.index("--spec-sha256") + 1] != file_sha(source_spec)
            or not args.approval.strip()):
        raise ValueError("hotfix resume requires the original approved source-bound measured-balance job")
    plan = read_sealed(args.campaign_output / "campaign-plan.json")
    settings = plan.get("execution_settings") or {}
    if settings.get("limit_action") != "warn" or settings.get("training_cutoff_seconds") is not None:
        raise ValueError("hotfix resume retains the original schedule; the bounded legacy contract requires a fresh run")
    state = read_sealed(args.campaign_output / "campaign-state.json")
    old = state.get("execution_code_commit", plan["source"]["base_commit"])
    body = authorize(args.campaign_output, args.repo, old, args.new_commit, args.reason)
    balance_path = Path(args.balance_observation)
    balance = measured_balance(str(balance_path), old_job["financial"]["owned_pod_id"])
    if balance <= old_job["financial"]["reserve_usd"]:
        raise ValueError("hotfix resume balance has reached the recovery reserve")
    observation = json.loads(balance_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    authorization = args.output / "authorization.json"
    seal(authorization, body)
    run_dir = args.output / "job-run"
    spec = resume_spec(json.loads(source_spec.read_text()), campaign_output=args.campaign_output,
        repo=args.repo, new_commit=args.new_commit, authorization=authorization,
        attempt=args.output, run_dir=run_dir)
    spec_path = args.output / "spec.json"
    write_json(spec_path, spec)
    now = datetime.now(timezone.utc).isoformat()
    job = deepcopy(old_job)
    job.update(commit=args.new_commit, repo_root=str(args.repo), run_dir=str(run_dir), max_main_seconds=0,
        main=[argv[0], "-m", "fine_tuning.notebook_rl.pod.chain", "run", "--spec", str(spec_path),
              "--spec-sha256", file_sha(spec_path)])
    job["financial"].update(balance_usd=balance, balance_snapshot_utc=observation["observed_at"])
    job["approval"] = {"approved_by_user": True, "approved_utc": now,
        "approval_note": args.approval + " Reviewed hotfix " + old + " -> " + args.new_commit + ". " + args.reason,
        "approved_sha256": ""}
    job["approval"]["approved_sha256"] = content_sha256(job)
    validate_job(job)
    job_path = args.output / "job.json"
    write_json(job_path, job)
    return {"status": "RESUME_PLANNED", "job": str(job_path), "spec": str(spec_path),
        "authorization": str(authorization), "start_log": str(args.output / "start-train.log"),
        "code_commit": args.new_commit, "campaign_output": str(args.campaign_output),
        "slot": body["slot"], "first_billed_at": job["financial"]["billing_start_utc"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("authorize", "resume-plan"))
    for name in ("campaign-output", "repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--old-commit")
    for name in ("new-commit", "reason"):
        parser.add_argument("--" + name, required=True)
    for name in ("source-spec", "job", "balance-observation"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--approval")
    args = parser.parse_args(argv)
    from .single import _runtime_lock
    from .v18b_recovery import recovery_paths
    with _runtime_lock(args.campaign_output, ".v16b-campaign.lock"), recovery_paths(args.campaign_output):
        if args.command == "resume-plan":
            if any(getattr(args, key) is None for key in ("source_spec", "job", "balance_observation", "approval")):
                raise ValueError("resume-plan needs original --source-spec --job, a fresh --balance-observation and --approval")
            print(json.dumps(operator_plan(args)))
            return 0
        body = authorize(args.campaign_output, args.repo, args.old_commit, args.new_commit, args.reason)
        result = seal(args.output, body)
    print(json.dumps({"status": "AUTHORIZED", "sha256": result["sha256"], "slot": result["slot"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
