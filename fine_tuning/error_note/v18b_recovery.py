"""Portable, immutable drained-boundary copies; sealed training identities stay unchanged."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
from threading import Event
import time
from typing import Iterator

from .audit import digest
from .collect import write_json
from .prep_v16b import file_sha, read_sealed

VERSION = "v18b-recovery-closure-v1"
_locations: ContextVar[dict[str, str]] = ContextVar("recovery_locations", default={})


def _mapped(path: Path | str, mappings: dict[str, str]) -> Path:
    path = Path(path)
    for old, new in sorted(mappings.items(), key=lambda entry: len(entry[0]), reverse=True):
        if path.is_relative_to(old):
            return Path(new) / path.relative_to(old)
    return path


def resolve_recovery_path(path: Path | str) -> Path:
    return _mapped(path, _locations.get())


def original_recovery_path(path: Path | str) -> Path:
    return _mapped(path, {new: old for old, new in _locations.get().items()})


@contextmanager
def recovery_paths(output: Path) -> Iterator[None]:
    marker = Path(output) / "recovery-locations.json"
    record = read_sealed(marker) if marker.exists() else None
    mappings = record["mappings"] if record is not None else {}
    if record is not None and record["manifest_sha256"] != read_sealed(Path(output) / "recovery-manifest.json")["sha256"]:
        raise ValueError("v18b recovery: relocation marker belongs to another closure")
    token = _locations.set(mappings)
    try:
        yield
    finally:
        _locations.reset(token)


def _inside(path: Path, root: Path) -> Path:
    if (not path.is_absolute() or not path.is_relative_to(root) or path == root
            or ".." in path.parts or not path.resolve().is_relative_to(root.resolve())):
        raise ValueError(f"v18b recovery: artifact outside campaign: {path}")
    return path.relative_to(root)


def _checked(record: dict, mappings: dict[str, str]) -> Path:
    path = _mapped(record["path"], mappings)
    if path.is_symlink() or not path.is_file() or file_sha(path) != record["file_sha256"]:
        raise ValueError(f"v18b recovery: missing or changed reference: {path}")
    return path


def check_row_admission(root: Path, transaction: dict, update: dict) -> Path | None:
    """Bind replay exclusions to the committed update, including restored histories."""
    checksum = transaction.get("row_admission_sha256")
    stats = update.get("result", {}).get("stats", {})
    if checksum is None:
        if "row_quarantine" in stats:
            raise ValueError("v18b recovery: row quarantine has no bound admission")
        return None
    slot = transaction.get("slot")
    if type(slot) is not int or slot < 0:
        raise ValueError("v18b recovery: row admission has an invalid slot")
    path = Path(root) / "row-admissions" / f"slot-{slot}.json"
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("v18b recovery: unsafe row admission")
    admission = read_sealed(path)
    report = admission.get("report", {})
    admitted, rejected = report.get("admitted_row_ids", []), report.get("rejected_rows", [])
    rejected_ids = [row.get("row_id") for row in rejected]
    attempted, optional = report.get("attempted_rows"), report.get("auxiliary_rows")
    if (admission.get("sha256") != checksum or admission.get("slot") != slot
            or update.get("global_slot") != slot or update.get("transaction_sha256") != transaction.get("sha256")
            or update.get("plan_sha256") != transaction.get("plan_sha256")
            or admission.get("plan_sha256") != transaction.get("plan_sha256")
            or admission.get("parent_checkpoint") != transaction.get("parent_checkpoint")
            or report != stats.get("row_quarantine") or report.get("schema") != "v18b-row-quarantine-v1"
            or report.get("status") != "CONTINUE" or type(attempted) is not int or type(optional) is not int
            or not 0 <= optional <= attempted or len(rejected) * 2 > optional
            or any(not isinstance(rid, str) or not rid for rid in admitted + rejected_ids)
            or len(set(admitted + rejected_ids)) != len(admitted) + len(rejected)
            or len(admitted) + len(rejected) != attempted
            or any(row.get("kind") not in {"A1", "C", "N"}
                   or not re.fullmatch(r"[a-f0-9]{64}", str(row.get("source_row_sha256", "")))
                   or not row.get("reason_code") for row in rejected)):
        raise ValueError("v18b recovery: row admission or quarantine evidence differs")
    counts, attempts = update.get("row_counts", {}), update.get("attempted_row_counts", {})
    if (any(type(value) is not int or value < 0 for value in list(counts.values()) + list(attempts.values()))
            or sum(counts.values()) != len(admitted) or sum(attempts.values()) != attempted
            or sum(attempts.get(kind, 0) for kind in ("A1", "C", "N")) != optional
            or any(attempts.get(kind, 0) != counts.get(kind, 0) + sum(row["kind"] == kind for row in rejected)
                   for kind in set(counts) | set(attempts) | {row["kind"] for row in rejected})):
        raise ValueError("v18b recovery: admitted row counts differ from the update")
    consumed = stats.get("consumed_row_ids")
    held = bool(stats.get("HOLD", stats.get("hold", False)))
    if (not isinstance(consumed, list) or len(consumed) != len(set(consumed))
            or set(consumed) - set(admitted) or held and consumed or not held and set(consumed) != set(admitted)):
        raise ValueError("v18b recovery: row admission consumption differs")
    return path


def verify_closure(root: Path, *, external: bool = True, mappings: dict[str, str] | None = None) -> dict:
    """Verify inventory, both physical checkpoints and live resume references, without a GPU."""
    root = Path(root).absolute()
    manifest = read_sealed(root / "recovery-manifest.json")
    if manifest.get("version") != VERSION:
        raise ValueError("v18b recovery: unsupported closure version")
    saved_locations = root / "recovery-locations.json"
    previous = read_sealed(saved_locations)["mappings"] if saved_locations.exists() else {}
    locations = {**previous, **(mappings or {}), manifest["source_root"]: str(root)}
    expected = set(manifest["files"]) | {"recovery-manifest.json"}
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    actual.discard("recovery-locations.json")
    if actual != expected or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("v18b recovery: inventory or symlink mismatch")
    for name, entry in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("v18b recovery: unsafe inventory path")
        path = root / relative
        if path.stat().st_size != entry["bytes"] or file_sha(path) != entry["sha256"]:
            raise ValueError(f"v18b recovery: file hash mismatch: {name}")
    plan = read_sealed(root / "campaign-plan.json")
    state = read_sealed(root / "campaign-state.json")
    if state["plan_sha256"] != plan["sha256"] or state["cursor"]["slot"] != manifest["slot"]:
        raise ValueError("v18b recovery: campaign/plan/cursor mismatch")
    for path in (root / "transactions").glob("slot-*.json"):
        transaction = read_sealed(path)
        update_path = root / "update-results" / path.name
        if transaction.get("row_admission_sha256") is not None or update_path.exists():
            update = read_sealed(update_path)
            check_row_admission(root, transaction, update)
    for index_path in root.glob("*.json"):
        value = json.loads(index_path.read_text())
        latest = value.get("latest") if isinstance(value, dict) else None
        if isinstance(latest, dict) and {"path", "file_sha256"} <= set(latest):
            target = _checked(latest, locations)
            if not target.is_relative_to(root):
                raise ValueError("v18b recovery: latest evidence is outside the closure")
            evidence = read_sealed(target)
            if evidence.get("plan_sha256") != plan["sha256"]:
                raise ValueError("v18b recovery: latest evidence belongs to another plan")
    descriptors = [state["checkpoint"], *state["probe"].values()]
    normal = state.get("v18b_normal_state")
    if normal is not None:
        descriptors += [normal["checkpoint"], *normal["probe"].values()]
        if normal.get("runtime") is not None:
            descriptors.append(normal["runtime"])
    prep = json.loads((root / "prep-checkpoint.json").read_text())
    descriptors.append(prep)
    for record in descriptors:
        _checked(record, locations)
    for record in plan["inputs"].values():
        if external or Path(record["path"]).is_relative_to(manifest["source_root"]):
            _checked(record, locations)
    for record in [state["checkpoint"], prep] + ([normal["checkpoint"]] if normal is not None else []):
        metadata_path = _checked(record, locations)
        metadata = json.loads(metadata_path.read_text())
        if ((metadata_path.parent / "metadata.sha256").read_text().strip() != file_sha(metadata_path)
                or set(metadata["files"]) != {"adapter.safetensors", "mergers.safetensors", "runtime.pt"}):
            raise ValueError("v18b recovery: checkpoint metadata mismatch")
        for name, checksum in metadata["files"].items():
            if file_sha(metadata_path.parent / name) != checksum:
                raise ValueError("v18b recovery: checkpoint payload mismatch")
        if record == state["checkpoint"] and metadata["cursor"]["single"] != state["cursor"]:
            raise ValueError("v18b recovery: checkpoint cursor mismatch")
    if state.get("runtime") is not None:
        envelope = json.loads(_checked(state["runtime"], locations).read_text())
        runtime = envelope["state"]
        if (envelope["sha256"] != digest(runtime) or runtime["plan_sha256"] != digest(plan)
                or runtime["protocol_sha256"] != plan["protocol"]["sha256"]
                or set(runtime["requests"]) != set(runtime["closed"])
                or any(entry["refcount"] for entry in runtime["registry"].values())):
            raise ValueError("v18b recovery: runtime identity/open requests mismatch")
        journal = root / "runtime" / "requests.jsonl"
        if journal.stat().st_size != runtime["journal_bytes"] or file_sha(journal) != runtime["journal_sha256"]:
            raise ValueError("v18b recovery: sealed journal mismatch")
        for descriptor in runtime["pending"]:
            if digest(runtime["closed"][descriptor["request_id"]]) != descriptor["output_sha256"]:
                raise ValueError("v18b recovery: delayed result mismatch")
        for pid, record in runtime["records"].items():
            stored = json.loads((root / "runtime" / "items" / (digest(pid) + ".json")).read_text())
            if stored != {"problem_id": pid, **record}:
                raise ValueError("v18b recovery: delayed record mismatch")
    external_refs = [record for record in plan["inputs"].values()
                     if not Path(record["path"]).is_relative_to(manifest["source_root"])]
    if external:
        for record in external_refs:
            _checked(record, locations)
        for field in ("model_path", "data_root", "prep_root", "nf4_cache"):
            if plan.get(field) and not _mapped(plan[field], locations).exists():
                raise ValueError(f"v18b recovery: external mount required: {field}={plan[field]}")
    return {"status": "RESUME_REFERENCES_VERIFIED" if external else "RECOVERY_CLOSURE_VERIFIED",
            "slot": manifest["slot"], "manifest_sha256": manifest["sha256"],
            "external_references_verified": external, "external_references": external_refs,
            "model_reload_verified": False, "next_slot_verified": False}


def stage_closure(output: Path, state: dict, stage: Path) -> None:
    """Pin immutable files before checkpoint pruning; copy only the sealed journal prefix."""
    output, stage = Path(output).absolute(), Path(stage).absolute()
    if state != read_sealed(output / "campaign-state.json"):
        raise ValueError("v18b recovery: checkpoint changed before staging")
    stage.mkdir(parents=True, exist_ok=False)

    def pin(source: Path) -> None:
        relative = _inside(source, output)
        target = stage / relative
        if target.exists():
            return
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"v18b recovery: missing/unsafe closure file: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)

    for name in ("campaign-plan.json", "campaign-state.json", "prep-checkpoint.json"):
        pin(output / name)
    for source in output.glob("*.json"):
        if source.name not in {"recovery-manifest.json", "recovery-locations.json", "recovery-copy-failure.json",
                               "recovery-copy-warning.json"}:
            pin(source)
    for directory in output.iterdir():
        if directory.is_dir() and (directory.name == "training-bindings" or directory.name.startswith(
                ("monitor-", "short-monitor-", "reasoning-monitor-", "effect-"))):
            for source in directory.rglob("*"):
                if source.is_file() and not source.name.endswith(".tmp"):
                    pin(source)
    plan = read_sealed(output / "campaign-plan.json")
    prep = json.loads((output / "prep-checkpoint.json").read_text())
    normal = state.get("v18b_normal_state")
    for descriptor in [state["checkpoint"], prep] + ([normal["checkpoint"]] if normal is not None else []):
        metadata_path = _checked(descriptor, _locations.get())
        for name in ("adapter.safetensors", "mergers.safetensors", "runtime.pt", "metadata.json", "metadata.sha256"):
            pin(metadata_path.parent / name)
    for descriptor in list(state["probe"].values()) + (list(normal["probe"].values()) if normal is not None else []):
        pin(_checked(descriptor, _locations.get()))
    if normal is not None and normal.get("runtime") is not None:
        pin(_checked(normal["runtime"], _locations.get()))
    for descriptor in plan["inputs"].values():
        if resolve_recovery_path(descriptor["path"]).is_relative_to(output):
            pin(_checked(descriptor, _locations.get()))
    # These are append-only per-slot evidence; do not copy live logs or regenerated export models.
    for evidence_directory in ("transactions", "update-results", "row-admissions", "signal-reports", "supply-reports"):
        for source in (output / evidence_directory).glob("*.json"):
            match = re.fullmatch(r"(?:v18b-gate-|v18b-)?slot-(\d+)", source.stem)
            main_evidence = source.stem.startswith("v18b-") or evidence_directory == "supply-reports"
            limit = state["cursor"].get("policy_version", state["cursor"]["slot"]) if main_evidence else state["cursor"]["slot"]
            if match and int(match[1]) < limit:
                pin(source)
    for index_path in output.glob("*.json"):
        value = json.loads(index_path.read_text())
        latest = value.get("latest") if isinstance(value, dict) else None
        if isinstance(latest, dict) and {"path", "file_sha256"} <= set(latest):
            target = _checked(latest, _locations.get())
            if not (stage / _inside(target, output)).is_file():
                raise ValueError("v18b recovery: latest evidence is missing from the committed boundary")
    if state.get("runtime") is not None:
        snapshot = _checked(state["runtime"], _locations.get())
        pin(snapshot)
        envelope = json.loads(snapshot.read_text())
        runtime = envelope["state"]
        data = (output / "runtime" / "requests.jsonl").read_bytes()[:runtime["journal_bytes"]]
        import hashlib
        if len(data) != runtime["journal_bytes"] or hashlib.sha256(data).hexdigest() != runtime["journal_sha256"]:
            raise ValueError("v18b recovery: journal prefix changed before staging")
        (stage / "runtime" / "items").mkdir(parents=True)
        (stage / "runtime" / "requests.jsonl").write_bytes(data)
        (stage / "runtime" / "drain-state.json").write_bytes(snapshot.read_bytes())
        for pid, record in runtime["records"].items():
            write_json(stage / "runtime" / "items" / (digest(pid) + ".json"), {"problem_id": pid, **record})
        for source in (output / "runtime" / "rolled-back").rglob("*"):
            if source.is_file():
                pin(source)


def _publish(stage: Path, destination: Path, slot: int, source_root: Path, transfer_command: list[str] | None,
             mappings: dict[str, str], stop_transfer: Event) -> dict:
    body = {"version": VERSION, "source_root": str(source_root), "slot": slot,
            "campaign_state_sha256": read_sealed(stage / "campaign-state.json")["sha256"],
            "files": {str(path.relative_to(stage)): {"bytes": path.stat().st_size, "sha256": file_sha(path)}
                      for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "recovery-manifest.json"}}
    write_json(stage / "recovery-manifest.json", body | {"sha256": digest(body)})
    verification = verify_closure(stage, mappings=mappings)
    archive = destination / f"slot-{slot}-{digest(body)[:16]}.tar"
    temporary = archive.with_suffix(".tar.tmp")
    with tarfile.open(temporary, "w", dereference=True) as stream:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                info = stream.gettarinfo(str(path), arcname=str(path.relative_to(stage)))
                info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                info.pax_headers = {}
                with path.open("rb") as source:
                    stream.addfile(info, source)
    checksum = file_sha(temporary)
    if archive.exists():
        if file_sha(archive) != checksum:
            raise ValueError("v18b recovery: existing bundle differs")
        temporary.unlink()
    else:
        os.replace(temporary, archive)
    verified_stat = archive.stat()
    receipt = verification | {"bundle_sha256": checksum, "archive": str(archive), "slot": slot,
                              "campaign_state_sha256": body["campaign_state_sha256"], "receiver_verified": False}
    receipt_path = archive.with_suffix(".receipt.pending.json")
    write_json(receipt_path, receipt)
    if transfer_command:
        try:
            argv = [part.format(archive=str(archive), receipt=str(receipt_path), slot=slot, bundle_sha256=checksum)
                    for part in transfer_command]
            remote = _receiver_result(argv, stop_transfer)
            if remote.get("bundle_sha256") != checksum or remote.get("status") not in {
                    "RECOVERY_CLOSURE_VERIFIED", "RESUME_REFERENCES_VERIFIED"}:
                raise ValueError("v18b recovery: receiver did not verify this bundle")
            receipt["receiver"] = remote
            receipt["receiver_verified"] = True
        except Exception as exc:
            receipt["transfer_status"] = "RECEIVER_UNCONFIRMED"
            receipt["warning"] = str(exc)
        # A delivery fault can coincide with damage to the pod copy. Only the former is recoverable.
        current_stat = archive.lstat()
        if (archive.is_symlink() or current_stat.st_size != verified_stat.st_size
                or current_stat.st_ctime_ns != verified_stat.st_ctime_ns and file_sha(archive) != checksum):
            raise ValueError("v18b recovery: local archive changed during receiver verification")
        if not receipt["receiver_verified"]:
            write_json(_mapped(source_root, mappings) / "recovery-copy-warning.json", {
                "status": "BACKUP_RECEIVER_UNCONFIRMED", "slot": slot,
                "archive": str(archive), "error": receipt["warning"], "training_may_continue": True})
        write_json(receipt_path, receipt)
    if not transfer_command or receipt["receiver_verified"]:
        os.replace(receipt_path, archive.with_suffix(".receipt.json"))
    shutil.rmtree(stage)
    return receipt


def _receiver_result(argv: list[str], stop: Event) -> dict:
    """Cancel only our receiver waiter at shutdown; leave its verified local bundle available."""
    if stop.is_set():
        raise RuntimeError("receiver wait deferred at campaign close")
    with tempfile.TemporaryFile(mode="w+t") as output:
        process = subprocess.Popen(argv, stdout=output, stderr=output, text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + 1800
            while process.poll() is None:
                if stop.wait(.1):
                    raise RuntimeError("receiver wait deferred at campaign close")
                if time.monotonic() >= deadline:
                    raise TimeoutError("receiver verification timed out")
            if process.returncode:
                raise RuntimeError(f"receiver command exited with status {process.returncode}")
            output.seek(0)
            if os.fstat(output.fileno()).st_size > 1024 * 1024:
                raise ValueError("receiver acknowledgement exceeds 1 MiB")
            return json.loads(output.read().strip().splitlines()[-1])
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()


def restore_bundle(archive: Path, output: Path, *, mappings: dict[str, str] | None = None,
                   external: bool = True) -> dict:
    """Extract into an unused path, verify before publishing, and retain sealed source bytes."""
    archive, output = Path(archive).absolute(), Path(output).absolute()
    if output.exists():
        marker = output / "recovery-locations.json"
        if marker.exists() and read_sealed(marker).get("bundle_sha256") == file_sha(archive):
            return verify_closure(output, external=external, mappings=mappings) | {
                "bundle_sha256": file_sha(archive), "output": str(output)}
        raise ValueError("v18b recovery: restore destination must be unused")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".recovery-", dir=output.parent))
    try:
        with tarfile.open(archive, "r") as stream:
            members = stream.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or any(not member.isfile() or Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts for member in members):
                raise ValueError("v18b recovery: unsafe archive entry")
            stream.extractall(staging, members=members, filter="data")
        verification = verify_closure(staging, external=external, mappings=mappings)
        manifest = read_sealed(staging / "recovery-manifest.json")
        body = {"version": VERSION, "mappings": {manifest["source_root"]: str(output), **(mappings or {})},
                "manifest_sha256": manifest["sha256"], "bundle_sha256": file_sha(archive)}
        write_json(staging / "recovery-locations.json", body | {"sha256": digest(body)})
        os.replace(staging, output)
        return verification | {"bundle_sha256": file_sha(archive), "output": str(output)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def receive_bundle(archive: Path, destination: Path, *, mappings: dict[str, str] | None = None,
                   external: bool = True) -> dict:
    """Keep two verified off-pod closures; publish each receipt before retiring the oldest."""
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as stream:
        source = stream.extractfile("recovery-manifest.json")
        if source is None:
            raise ValueError("v18b recovery: missing archive manifest")
        manifest = json.load(source)
    slot, identity = manifest.get("slot"), manifest.get("sha256", "")
    if type(slot) is not int or slot < 0 or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("v18b recovery: invalid receiver identity")
    name = f"slot-{slot}-{identity[:16]}"
    result = restore_bundle(archive, destination / name, mappings=mappings, external=external)
    write_json(destination / (name + ".received.json"), result)
    receipts = sorted(destination.glob("slot-*.received.json"),
                      key=lambda path: (int(path.name.split("-")[1]), path.name), reverse=True)
    for receipt in receipts[2:]:
        value = json.loads(receipt.read_text())
        previous = destination / receipt.name.removesuffix(".received.json")
        if (value.get("output") != str(previous) or not previous.is_dir()
                or read_sealed(previous / "recovery-locations.json").get("bundle_sha256") != value.get("bundle_sha256")):
            raise ValueError("v18b recovery: old receiver copy ownership changed")
        shutil.rmtree(previous)
        receipt.unlink()
    return result


class RecoveryCopyHook:
    """Stage at commit; archive/copy/receiver verification run off the training thread."""

    def __init__(self, output: Path, destination: Path, *, transfer_command: list[str] | None = None,
                 every_slots: int = 16):
        self.output, self.destination = Path(output).absolute(), Path(destination).absolute()
        if self.destination.is_relative_to(self.output) or every_slots < 1:
            raise ValueError("v18b recovery: separate backup root and positive interval required")
        self.destination.mkdir(parents=True, exist_ok=True)
        self.transfer_command, self.every_slots = transfer_command, every_slots
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="boundary-copy")
        self.pending: list[Future] = []
        self.submitted: set[str] = set()
        self.stop_transfer = Event()

    def checkpoint(self, state: dict, *, force: bool = False) -> dict:
        self.poll()
        slot, identity = state["cursor"]["slot"], state["sha256"]
        if (not force and slot % self.every_slots) or identity in self.submitted:
            return {"status": "NOT_DUE_OR_ALREADY_STAGED", "slot": slot}
        if self.pending:
            self.stop_transfer.set()
            self.poll(wait=True)
            self.stop_transfer = Event()
        for path in self.destination.glob("slot-*.receipt.json"):
            receipt = json.loads(path.read_text())
            if receipt.get("campaign_state_sha256") == identity and (
                    not self.transfer_command or receipt.get("receiver")):
                archive = Path(receipt["archive"])
                if archive.is_file() and file_sha(archive) == receipt["bundle_sha256"]:
                    self.submitted.add(identity)
                    return {"status": "ALREADY_VERIFIED", "slot": slot}
        stage = self.output.parent / (".recovery-staging-" + self.output.name) / identity
        if stage.exists():
            if not (stage / "campaign-state.json").exists() or read_sealed(stage / "campaign-state.json") != state:
                raise ValueError("v18b recovery: interrupted staging needs inspection")
            shutil.rmtree(stage)
        try:
            stage_closure(self.output, state, stage)
            future = self.executor.submit(_publish, stage, self.destination, slot, original_recovery_path(self.output),
                                          self.transfer_command, dict(_locations.get()), self.stop_transfer)
            self.pending.append(future)
            self.submitted.add(identity)
        except Exception as exc:
            self._failure(exc)
            raise
        return {"status": "COPY_STARTED", "slot": slot}

    def _failure(self, error: Exception) -> None:
        write_json(self.output / "recovery-copy-failure.json", {"status": "BACKUP_FAILED", "error": str(error)})
        (self.output / "RECOVERY_COPY_FAILED").write_text(str(error) + "\n")

    def poll(self, *, wait: bool = False) -> list[dict]:
        completed = []
        for future in list(self.pending):
            if wait or future.done():
                try:
                    completed.append(future.result())
                except Exception as exc:
                    self._failure(exc)
                    raise RuntimeError("v18b recovery: boundary backup failed; see recovery-copy-failure.json") from exc
                self.pending.remove(future)
        # The notebook receiver may acknowledge a pending archive after its waiter timed out.
        pending_receipts = [] if self.pending else list(self.destination.glob("slot-*.receipt.pending.json"))
        for path in pending_receipts:
            receipt = json.loads(path.read_text())
            archive = path.with_name(path.name.removesuffix(".receipt.pending.json") + ".tar")
            if (receipt.get("archive") != str(archive) or archive.is_symlink() or not archive.is_file()):
                error = ValueError("v18b recovery: pending local copy ownership changed")
                self._failure(error)
                raise error
            remote_path = Path(receipt["archive"] + ".remote.json")
            if not remote_path.is_file():
                continue
            if remote_path.is_symlink() or remote_path.stat().st_size > 1024 * 1024:
                continue
            try:
                remote = json.loads(remote_path.read_text())
            except (ValueError, UnicodeError):
                continue
            if not isinstance(remote, dict):
                continue
            if (remote.get("bundle_sha256") == receipt["bundle_sha256"] and remote.get("status") in {
                    "RECOVERY_CLOSURE_VERIFIED", "RESUME_REFERENCES_VERIFIED"}):
                if file_sha(archive) != receipt["bundle_sha256"]:
                    error = ValueError("v18b recovery: pending local archive hash changed")
                    self._failure(error)
                    raise error
                receipt.update(receiver=remote, receiver_verified=True, transfer_status="RECEIVER_VERIFIED")
                receipt.pop("warning", None)
                write_json(path, receipt)
                os.replace(path, path.with_name(path.name.replace(".receipt.pending.json", ".receipt.json")))
                completed.append(receipt)
        if completed:
            try:
                # Keep two local closures and the newest confirmed remote closure through receiver outages.
                receipts = sorted([*self.destination.glob("slot-*.receipt.json"),
                                   *self.destination.glob("slot-*.receipt.pending.json")],
                                  key=lambda path: (int(path.name.split("-")[1]), path.name), reverse=True)
                remote = next((path for path in receipts if json.loads(path.read_text()).get("receiver_verified")), None)
                for receipt in receipts[2:]:
                    if receipt == remote:
                        continue
                    suffix = ".receipt.pending.json" if receipt.name.endswith(".receipt.pending.json") else ".receipt.json"
                    archive = receipt.with_name(receipt.name.removesuffix(suffix) + ".tar")
                    value = json.loads(receipt.read_text())
                    if value.get("archive") != str(archive) or archive.is_symlink() or not archive.is_file():
                        raise ValueError("v18b recovery: old local copy ownership changed")
                    archive.unlink()
                    receipt.unlink()
            except Exception as exc:
                self._failure(exc)
                raise RuntimeError("v18b recovery: backup retention failed") from exc
        return completed

    def close(self) -> list[dict]:
        self.stop_transfer.set()
        try:
            return self.poll(wait=True)
        finally:
            self.executor.shutdown(wait=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("restore", "receive", "verify"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--closure-only", action="store_true", help="external mounts remain unverified")
    args = parser.parse_args(argv)
    mappings = dict(entry.split("=", 1) for entry in args.map)
    if args.command == "verify":
        result = verify_closure(args.output, external=not args.closure_only, mappings=mappings)
    else:
        if args.archive is None:
            parser.error("--archive is required")
        operation = receive_bundle if args.command == "receive" else restore_bundle
        result = operation(args.archive, args.output, mappings=mappings, external=not args.closure_only)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
