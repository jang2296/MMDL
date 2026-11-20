"""Run-dir bookkeeping for start.sh: append-only state, owner record, bounded logged command.

CLI: runner state STATE_FILE STATE [--exit-code N]
     runner last STATE_FILE                      prints the current state
     runner owner OWNER_FILE --pid PID
     runner run --log FILE --timeout SECONDS [--disk-path DIR --min-free-gib G --disk-log FILE
                --state-file STATE_FILE [--disk-interval SECONDS]] -- CMD...
With --disk-path the free space is sampled every interval into --disk-log (JSON lines); below
--min-free-gib the command group is stopped, DISK_LOW is recorded and the exit code is 75.
A command is never restarted and its limits are never changed here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

from fine_tuning.budget import terminate_group

from .common import process_startticks, read_json, utc_now, write_json

STATE_RE = re.compile(r"^(PREFLIGHT_OK|PREFLIGHT_FAILED|SMOKE_OK|SMOKE_FAILED|MAIN_RUNNING|"
                      r"MAIN_EXIT -?\d+|INTERRUPTED|BUDGET_STOP|DISK_LOW)$")
TERMINAL_PREFIXES = ("PREFLIGHT_FAILED", "SMOKE_FAILED", "MAIN_EXIT", "INTERRUPTED",
                     "BUDGET_STOP", "DISK_LOW")
TIMEOUT_CODE = 124
DISK_LOW_CODE = 75
DISK_INTERVAL_SECONDS = 60.0


def record_state(path: str | Path, state: str, **extra: Any) -> dict[str, Any]:
    if not STATE_RE.match(state):
        raise ValueError(f"unknown run state: {state!r}")
    target = Path(path)
    record = (read_json(target) if target.exists()
              else {"version": "pod-run-state-v1", "history": []})
    record["history"].append({"state": state, "utc": utc_now(), **extra})
    record["state"] = state
    write_json(target, record)
    return record


def last_state(path: str | Path) -> str:
    return str(read_json(path)["state"])


def disk_sample(path: str | Path, log: str | Path) -> float:
    """Free GiB at path, appended to the JSON-lines log."""
    free = shutil.disk_usage(path).free / 2**30
    with open(log, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": utc_now(), "path": str(path),
                                 "free_gib": round(free, 3)}) + "\n")
    return free


def write_owner(path: str | Path, pid: int) -> dict[str, Any]:
    ticks = process_startticks(pid)
    if ticks is None:
        raise ValueError(f"cannot read start time of pid {pid}")
    owner = {"pid": pid, "startticks": ticks, "utc": utc_now()}
    write_json(path, owner)
    return owner


def run_logged(argv: list[str], log: str | Path, timeout: float, grace: float = 30, *,
               disk_path: str | Path | None = None, min_free_gib: float = 0.0,
               disk_log: str | Path | None = None, state_file: str | Path | None = None,
               disk_interval: float = DISK_INTERVAL_SECONDS) -> int:
    """Run argv in its own session, append stdout+stderr to log, hard timeout -> 124.

    SIGTERM/SIGINT to this runner stops the whole child group via budget.terminate_group.
    Handlers are live before the child exists and signals are ignored while the group is being
    stopped (as budget.py does): a default-action death here would orphan the child's session,
    which the watchdog's killpg of our group does not reach.
    With disk_path, free space is sampled every disk_interval seconds; below min_free_gib the
    group is stopped, DISK_LOW is recorded in state_file and DISK_LOW_CODE returned.
    """
    if disk_path is not None and (disk_log is None or state_file is None
                                  or not disk_interval > 0 or not min_free_gib > 0):
        raise ValueError("disk monitoring needs --disk-log, --state-file, a positive interval "
                         "and a positive --min-free-gib")
    previous: dict[int, Any] = {}
    process: subprocess.Popen[bytes] | None = None

    def interrupted(signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        with open(log, "ab") as stream:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            if disk_path is not None:
                free = disk_sample(disk_path, disk_log)  # type: ignore[arg-type]
                if free < min_free_gib:
                    record_state(state_file, "DISK_LOW",  # type: ignore[arg-type]
                                 free_gib=round(free, 3), min_free_gib=min_free_gib)
                    return DISK_LOW_CODE
            remaining = deadline - time.monotonic() if deadline is not None else disk_interval
            if deadline is not None and remaining <= 0:
                return TIMEOUT_CODE
            try:
                step = remaining if disk_path is None else min(disk_interval, remaining)
                code = process.wait(timeout=step)
                return code if code >= 0 else 128 - code
            except subprocess.TimeoutExpired:
                continue
    finally:
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            if process is not None:
                terminate_group(process, grace_seconds=grace)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    state = sub.add_parser("state")
    state.add_argument("file")
    state.add_argument("state")
    state.add_argument("--exit-code", type=int)
    last = sub.add_parser("last")
    last.add_argument("file")
    owner = sub.add_parser("owner")
    owner.add_argument("file")
    owner.add_argument("--pid", type=int, required=True)
    run = sub.add_parser("run")
    run.add_argument("--log", required=True)
    run.add_argument("--timeout", type=float, required=True)
    run.add_argument("--disk-path")
    run.add_argument("--min-free-gib", type=float, default=0.0)
    run.add_argument("--disk-log")
    run.add_argument("--state-file")
    run.add_argument("--disk-interval", type=float, default=DISK_INTERVAL_SECONDS)
    run.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == "state":
            extra = {} if args.exit_code is None else {"exit_code": args.exit_code}
            record_state(args.file, args.state, **extra)
            return 0
        if args.command == "last":
            print(last_state(args.file))
            return 0
        if args.command == "owner":
            write_owner(args.file, args.pid)
            return 0
        cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
        if not cmd or args.timeout < 0 or args.timeout == 0 and not os.environ.get("MMDL_BUDGET_WATCHDOG_PID"):
            raise ValueError("a command after -- and a positive timeout are required")
        return run_logged(cmd, args.log, args.timeout, disk_path=args.disk_path,
                          min_free_gib=args.min_free_gib, disk_log=args.disk_log,
                          state_file=args.state_file, disk_interval=args.disk_interval)
    except (OSError, ValueError, KeyError) as exc:
        print(f"runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
