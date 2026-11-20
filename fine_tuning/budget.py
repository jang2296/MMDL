"""Standard-library owned-process watchdog, usable before dependency installation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

BILLING_SKEW_SECONDS = 300  # clock skew tolerated for a billing start in the future


def measured_balance(path: str, owned_pod_id: str, *, now: dt.datetime | None = None) -> float:
    """Read a fresh operator observation delivered over the existing authenticated SSH channel."""
    observation = json.loads(Path(path).read_text())
    if (not isinstance(observation, dict) or set(observation) != {"version", "owned_pod_id", "observed_at", "balance_usd"}
            or observation["version"] != "v18b-balance-observation-v1" or observation["owned_pod_id"] != owned_pod_id):
        raise ValueError("Balance observation belongs to a different owned pod or contract")
    instant = now or dt.datetime.now(dt.timezone.utc)
    age = (instant - utc(observation["observed_at"])).total_seconds()
    if not -60 <= age <= 600:
        raise ValueError("Balance observation must be recent and cannot be in the future")
    value = observation["balance_usd"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("RunPod returned an invalid measured balance")
    return float(value)


def utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC deadlines require explicit timezone")
    return parsed.astimezone(dt.timezone.utc)


def effective_deadline(gpu_cutoff: dt.datetime, billing_start: dt.datetime,
                       balance_usd: float, reserve_usd: float, hourly_usd: float) -> dt.datetime:
    if not all(math.isfinite(v) for v in (balance_usd, reserve_usd, hourly_usd)):
        raise ValueError("Nonfinite financial budget")
    if hourly_usd <= 0 or reserve_usd < 0 or balance_usd <= reserve_usd:
        raise ValueError("Positive rate and balance above recovery reserve required")
    financial = billing_start + dt.timedelta(seconds=(balance_usd-reserve_usd)/hourly_usd*3600)
    return min(gpu_cutoff, financial)


def check_billing_start(billing_start: dt.datetime, now: dt.datetime | None = None,
                        skew_seconds: float = BILLING_SKEW_SECONDS) -> None:
    """A billing start in the future moves the financial deadline later than the money allows."""
    now = now or dt.datetime.now(dt.timezone.utc)
    if billing_start > now + dt.timedelta(seconds=skew_seconds):
        raise ValueError("Billing start is in the future; give the Pod's actual first billing time")


def reserve_deadline(snapshot: dt.datetime, balance_usd: float, reserve_usd: float,
                     hourly_usd: float, now: dt.datetime | None = None) -> dt.datetime:
    """The measured balance reaches the fixed recovery reserve at this instant."""
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (balance_usd, reserve_usd, hourly_usd))
            or reserve_usd != 1.5 or balance_usd < 0 or hourly_usd <= 0):
        raise ValueError("Reserve-only mode needs a finite balance, positive rate and USD 1.5 reserve")
    if snapshot > (now or dt.datetime.now(dt.timezone.utc)) + dt.timedelta(seconds=BILLING_SKEW_SECONDS):
        raise ValueError("Balance snapshot is in the future")
    return snapshot + dt.timedelta(seconds=(balance_usd - reserve_usd) / hourly_usd * 3600)


def balance_extension_sha256(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def check_balance_extension(value: dict, *, previous: str, deadline: dt.datetime,
                            snapshot: dt.datetime, owned_pod_id: str, hourly_usd: float,
                            now: dt.datetime | None = None) -> tuple[dt.datetime, dt.datetime]:
    """Validate an explicit top-up against the same Pod, price and preceding receipt."""
    keys = {"version", "previous", "owned_pod_id", "balance_snapshot_utc", "balance_usd",
            "reserve_usd", "hourly_usd", "old_deadline_utc", "new_deadline_utc",
            "topup_reference", "authorization", "confirmed_by", "sha256"}
    if (not isinstance(value, dict) or set(value) != keys
            or value["version"] != "pod-balance-extension-v1"
            or value["sha256"] != balance_extension_sha256(value)
            or value["previous"] != previous or value["owned_pod_id"] != owned_pod_id
            or value["hourly_usd"] != hourly_usd or value["reserve_usd"] != 1.5
            or value["old_deadline_utc"] != deadline.isoformat()
            or value["confirmed_by"] not in ("user", "claude")
            or any(not isinstance(value[k], str) or not value[k].strip()
                   for k in ("topup_reference", "authorization"))):
        raise ValueError("Invalid approved balance extension receipt")
    current = utc(value["balance_snapshot_utc"])
    extended = reserve_deadline(current, value["balance_usd"], value["reserve_usd"],
                                value["hourly_usd"], now)
    if (current <= snapshot or extended <= deadline
            or value["new_deadline_utc"] != extended.isoformat()):
        raise ValueError("Top-up must use a newer measured snapshot and extend the reserve deadline")
    return extended, current


def balance_extensions(directory: str | Path, *, root_sha256: str, owned_pod_id: str,
                       snapshot: dt.datetime, balance_usd: float, hourly_usd: float,
                       seen: tuple[str, ...] = (), now: dt.datetime | None = None
                       ) -> tuple[dt.datetime, dt.datetime, tuple[str, ...]]:
    deadline = reserve_deadline(snapshot, balance_usd, 1.5, hourly_usd, now)
    previous, hashes = root_sha256, []
    for index, path in enumerate(sorted(Path(directory).glob("*.json")), 1):
        if path.is_symlink() or path.name != f"{index:06d}.json":
            raise ValueError("Balance extensions must be append-only numbered JSON receipts")
        value = json.loads(path.read_text())
        deadline, snapshot = check_balance_extension(
            value, previous=previous, deadline=deadline, snapshot=snapshot,
            owned_pod_id=owned_pod_id, hourly_usd=hourly_usd, now=now)
        hashes.append(value["sha256"])
        previous = value["sha256"]
    if tuple(hashes[:len(seen)]) != seen:
        raise ValueError("An accepted balance extension was removed or replaced")
    return deadline, snapshot, tuple(hashes)


def write_result(path: str, result: dict) -> None:
    """Atomic replace in an existing directory (never creates one)."""
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".budget-result-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def terminate_group(process: subprocess.Popen, grace_seconds: float = 15) -> None:
    """Terminate a start_new_session child and kill residual workers after leader exit."""
    if not math.isfinite(grace_seconds) or grace_seconds <= 0:
        raise ValueError("Positive finite termination grace required")
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("owned-pod-id", "gpu-cutoff-utc", "billing-start-utc"):
        parser.add_argument("--"+name, required=True)
    for name in ("balance-usd", "reserve-usd", "hourly-usd"):
        parser.add_argument("--"+name, type=float, required=True)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--financial-mode", choices=("legacy", "balance_reserve_only", "measured_balance_reserve"), default="legacy")
    parser.add_argument("--balance-snapshot-utc")
    parser.add_argument("--balance-observation")
    parser.add_argument("--extension-receipts")
    parser.add_argument("--extension-root-sha256")
    parser.add_argument("--reserve-stop-receipt")
    parser.add_argument("--result-file", help="also write the result JSON here: when a budget stop is "
                        "decided (before the group is stopped) and at exit; the directory must exist")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    process, previous = None, {}
    started = time.monotonic()
    result: dict[str, object] = {}

    def persist() -> None:
        paths = [args.result_file] if args.result_file else []
        if result.get("status") == "BUDGET_STOP" and args.financial_mode in ("balance_reserve_only", "measured_balance_reserve"):
            paths += [args.reserve_stop_receipt]
        for path in paths:
            try:
                write_result(path, result)
            except OSError as exc:
                result["result_file_error"] = f"{type(exc).__name__}: {exc}"
    try:
        if not args.owned_pod_id or os.environ.get("RUNPOD_POD_ID") != args.owned_pod_id:
            raise ValueError("Exact owned RunPod identity required")
        measured_only = args.financial_mode == "measured_balance_reserve"
        if not math.isfinite(args.max_seconds) or args.max_seconds < 0 or args.max_seconds == 0 and not measured_only:
            raise ValueError("Positive finite maximum seconds required")
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise ValueError("A command after -- is required")
        check_billing_start(utc(args.billing_start_utc))
        reserve_only = args.financial_mode == "balance_reserve_only"
        seen: tuple[str, ...] = ()
        if measured_only:
            if args.reserve_usd != 1.5 or not args.reserve_stop_receipt or not Path(args.reserve_stop_receipt).is_absolute():
                raise ValueError("Measured balance watchdog needs the 1.5 USD reserve and an absolute stop receipt")
            if not args.balance_observation or not Path(args.balance_observation).is_absolute():
                raise ValueError("Measured balance watchdog requires an absolute observation path")
            balance = measured_balance(args.balance_observation, args.owned_pod_id)
            if balance <= args.reserve_usd:
                raise TimeoutError("Measured balance reached the USD 1.5 reserve")
            allowed = None
            result.update(financial_mode=args.financial_mode, deadline_utc=None, timeout_seconds=None)
        elif reserve_only:
            if (args.reserve_usd != 1.5 or not args.extension_receipts
                    or not Path(args.extension_receipts).is_absolute()
                    or not args.reserve_stop_receipt or not Path(args.reserve_stop_receipt).is_absolute()
                    or not args.extension_root_sha256 or len(args.extension_root_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in args.extension_root_sha256)):
                raise ValueError("Reserve-only watchdog needs fixed reserve and bound extension receipts")
            snapshot = utc(args.balance_snapshot_utc)
            deadline, _, seen = balance_extensions(
                args.extension_receipts, root_sha256=args.extension_root_sha256,
                owned_pod_id=args.owned_pod_id, snapshot=snapshot, balance_usd=args.balance_usd,
                hourly_usd=args.hourly_usd)
            allowed = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
            result.update(financial_mode=args.financial_mode, extension_sha256=list(seen))
        else:
            deadline = effective_deadline(utc(args.gpu_cutoff_utc), utc(args.billing_start_utc),
                                          args.balance_usd, args.reserve_usd, args.hourly_usd)
            allowed = min(args.max_seconds, (deadline-dt.datetime.now(dt.timezone.utc)).total_seconds()-120)
        if not measured_only:
            assert allowed is not None
            result.update(deadline_utc=deadline.isoformat(), timeout_seconds=max(0, allowed))
        if allowed is not None and allowed <= 0:
            raise TimeoutError("Measured balance reached the USD 1.5 reserve" if reserve_only else
                               "Budget cannot retain the 120-second recovery reserve")
        def interrupted(signum, _frame):
            raise SystemExit(128+signum)

        def hangup(_signum, _frame):
            # A dropped SSH session must not kill the watchdog and leave its own-session child
            # running unbounded; a Python handler (not SIG_IGN) is reset to default in the child.
            result["sighup_ignored"] = int(result.get("sighup_ignored", 0)) + 1
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, hangup)
        environment = dict(os.environ, MMDL_BUDGET_WATCHDOG_PID=str(os.getpid()))
        process = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL,
                                   env=environment)
        result["pid"] = process.pid
        if measured_only:
            while True:
                if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds:
                    result.update(status="TECHNICAL_TIMEOUT", technical_timeout_seconds=args.max_seconds)
                    code = 124
                    break
                try:
                    balance = measured_balance(args.balance_observation, args.owned_pod_id)
                except Exception as exc:
                    result.update(balance_telemetry_warning=type(exc).__name__,
                                  balance_telemetry_warning_at=dt.datetime.now(dt.timezone.utc).isoformat())
                else:
                    result.update(measured_balance_usd=balance,
                                  balance_measured_at=dt.datetime.now(dt.timezone.utc).isoformat())
                    result.pop("balance_telemetry_warning", None)
                    if balance <= args.reserve_usd:
                        raise TimeoutError("Measured balance reached the USD 1.5 reserve")
                    if balance < args.reserve_usd + args.hourly_usd * 2:
                        result["recharge_requested"] = True
                    else:
                        result.pop("recharge_requested", None)
                persist()
                try:
                    wait = min(60, max(.001, args.max_seconds - (time.monotonic() - started))) if args.max_seconds > 0 else 60
                    code = process.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        elif reserve_only:
            while True:
                deadline, _, seen = balance_extensions(
                    args.extension_receipts, root_sha256=args.extension_root_sha256,
                    owned_pod_id=args.owned_pod_id, snapshot=snapshot, balance_usd=args.balance_usd,
                    hourly_usd=args.hourly_usd, seen=seen)
                result.update(deadline_utc=deadline.isoformat(), extension_sha256=list(seen))
                remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
                if remaining <= 0:
                    raise TimeoutError("Measured balance reached the USD 1.5 reserve")
                try:
                    code = process.wait(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        else:
            code = process.wait(timeout=allowed)
        result.setdefault("status", "EXITED")
        result["returncode"] = code
        code = code if code >= 0 else 128-code
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        result.update(status="BUDGET_STOP", reason=str(exc))
        if args.financial_mode in ("balance_reserve_only", "measured_balance_reserve"):
            result.update(required_action="POD_STOP_REQUIRED", recovery_status="RECOVERY_REQUIRED",
                          owned_pod_id=args.owned_pod_id, result_file=args.result_file,
                          automatic_pod_stop=False)
        persist()  # before the group is stopped, so the child can record BUDGET_STOP, not a kill
        if args.financial_mode in ("balance_reserve_only", "measured_balance_reserve"):
            try:
                print(f"POD_STOP_REQUIRED: preserve results, then stop owned Pod {args.owned_pod_id}; "
                      f"receipt {args.reserve_stop_receipt}. Workload stop does not stop Pod billing.",
                      flush=True)
            except OSError:
                pass  # the durable receipt survives a dropped terminal
        code = 124
    except (KeyboardInterrupt, SystemExit) as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else (exc.code if isinstance(exc.code, int) else 1)
        result.update(status="INTERRUPTED", signal=code-128)
    except Exception as exc:
        result.update(status="BLOCKED", reason=f"{type(exc).__name__}: {exc}")
        code = 2
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if process is not None:
                terminate_group(process, grace_seconds=45)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    result["seconds"] = time.monotonic()-started
    persist()
    try:
        print(json.dumps(result), flush=True)
    except OSError:  # stdout was a terminal that hung up; the result file holds the record
        pass
    return code


if __name__ == "__main__":
    raise SystemExit(main())
