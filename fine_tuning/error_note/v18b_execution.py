"""Operator decisions and a single clock for both v18b chains, including recovery."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time

from .audit import digest

def _epoch(value: str) -> float:
    if not isinstance(value, str):
        raise ValueError("v18b billing needs a UTC timestamp")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    offset = instant.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("v18b billing timestamps must use UTC")
    return instant.timestamp()


def _number(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_settings(settings: object) -> dict:
    if isinstance(settings, dict) and settings.get("execution_mode") == "on_prem":
        keys = {"execution_mode", "run_id", "evaluation_strategy", "training_cutoff_seconds",
                "monitor_stop_action", "reuse_f0_raw", "limit_action"}
        if (set(settings) != keys or not isinstance(settings["run_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", settings["run_id"])
                or settings["evaluation_strategy"] != "one_pod"
                or settings["training_cutoff_seconds"] is not None
                or settings["monitor_stop_action"] not in {"evaluate_checkpoint", "stop"}
                or settings["reuse_f0_raw"] is not False or settings["limit_action"] != "warn"):
            raise ValueError("on-prem v18b requires fresh F0, no billing/cutoff and explicit monitor decisions")
        return settings
    keys = {"run_id", "first_billed_at", "total_timeout_seconds", "recovery_reserve_seconds",
            "post_eval_reserve_seconds", "evaluation_strategy", "training_cutoff_seconds",
            "monitor_stop_action", "reuse_f0_raw", "billing"}
    optional = {"limit_action", "budget_limit_usd"}
    if not isinstance(settings, dict) or not keys <= set(settings) <= keys | optional:
        raise ValueError("v18b requires the complete explicit run decisions and billing schedule")
    if not isinstance(settings["run_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", settings["run_id"]):
        raise ValueError("invalid v18b run ID")
    start = _epoch(settings["first_billed_at"])
    total, recovery, post_eval = (settings[k] for k in (
        "total_timeout_seconds", "recovery_reserve_seconds", "post_eval_reserve_seconds"))
    if not all(type(v) is int for v in (total, recovery, post_eval)) or not (0 < recovery <= post_eval < total):
        raise ValueError("expected positive recovery reservation within the declared planning allowance")
    action = settings.get("limit_action", "stop")
    budget = settings.get("budget_limit_usd", 22.)  # Legacy sealed settings retain their original ceiling.
    if action not in {"warn", "stop"} or budget is not None and not _number(budget):
        raise ValueError("choose a finite planning budget or null and limit_action warn/stop")
    strategy = settings["evaluation_strategy"]
    if strategy not in {"one_pod", "two_pod", "training_cutoff"} or type(settings["reuse_f0_raw"]) is not bool:
        raise ValueError("explicit evaluation strategy and F0 reuse decision required")
    if settings["monitor_stop_action"] not in {"evaluate_checkpoint", "stop"}:
        raise ValueError("choose how to handle a quality stop before training")
    cutoff = settings["training_cutoff_seconds"]
    if action == "warn" and (strategy != "one_pod" or cutoff is not None):
        raise ValueError("completion mode requires one pod with no training cutoff")
    if strategy == "training_cutoff":
        if type(cutoff) is not int or not 0 < cutoff < total - post_eval:
            raise ValueError("training cutoff must leave time for evaluation and recovery")
    elif cutoff is not None:
        raise ValueError("a training cutoff requires explicit partial-evaluation policy")
    leases = settings["billing"]
    if not isinstance(leases, list) or len(leases) != (2 if strategy == "two_pod" else 1):
        raise ValueError("supply a billing row for each authorized pod")
    for lease in leases:
        if not isinstance(lease, dict) or set(lease) != {"first_billed_at", "usd_per_hour"} or not _number(lease["usd_per_hour"]):
            raise ValueError("pod rate must include GPU and storage")
        lease_start = _epoch(lease["first_billed_at"])
        if not start <= lease_start < start + total:
            raise ValueError("pod start outside the bounded run")
    if _epoch(leases[0]["first_billed_at"]) != start:
        raise ValueError("first billed timestamp must belong to the primary pod")
    return settings


def projected_cost(settings: dict, at: float) -> float:
    """Conservative cost: scheduled pods accrue through the deadline, including handoff time."""
    return sum(max(0., at - _epoch(row["first_billed_at"])) * row["usd_per_hour"] / 3600
               for row in settings["billing"])


def enforce_limits(settings: dict) -> bool:
    """Old registrations retain their bounds; completion runs use the same clock for warnings."""
    return settings.get("execution_mode") != "on_prem" and settings.get("limit_action", "stop") == "stop"


def planning_status(settings: dict, *, now: float | None = None) -> dict:
    validate_settings(settings)
    if settings.get("execution_mode") == "on_prem":
        return {"execution_mode": "on_prem", "elapsed_seconds": None, "estimated_cost_usd": None,
                "limit_action": "warn", "warnings": []}
    current = time.time() if now is None else now
    elapsed = current - _epoch(settings["first_billed_at"])
    cost = projected_cost(settings, current)
    budget = settings.get("budget_limit_usd", 22.)
    warnings = []
    if elapsed >= settings["total_timeout_seconds"]:
        warnings.append("PLANNING_TIME_TARGET_EXCEEDED")
    if budget is not None and cost >= budget:
        warnings.append("PLANNING_COST_TARGET_EXCEEDED")
    return {"elapsed_seconds": elapsed, "estimated_cost_usd": cost,
            "limit_action": settings.get("limit_action", "stop"), "warnings": warnings}


def deadline_epoch(settings: dict, *, reserve_seconds: float = 0) -> float:
    validate_settings(settings)
    if settings.get("execution_mode") == "on_prem":
        raise ValueError("on-prem execution has no billing deadline")
    if type(reserve_seconds) not in (float, int) or not math.isfinite(reserve_seconds) or reserve_seconds < 0:
        raise ValueError("invalid time reservation")
    start = _epoch(settings["first_billed_at"])
    end = start + settings["total_timeout_seconds"]
    budget = settings.get("budget_limit_usd", 22.)
    if enforce_limits(settings) and budget is not None and projected_cost(settings, end) > budget:
        # Price is monotonic and changes at most once (the second pod's earliest start).
        lo, hi = start, end
        for _ in range(48):
            mid = (lo + hi) / 2
            if projected_cost(settings, mid) > budget:
                hi = mid
            else:
                lo = mid
        end = lo
    return end - reserve_seconds


def remaining(settings: dict, *, reserve_seconds: float = 0, now: float | None = None) -> float:
    return deadline_epoch(settings, reserve_seconds=reserve_seconds) - (time.time() if now is None else now)


def settings_sha256(settings: dict) -> str:
    return digest(validate_settings(settings))


def read_settings(path: Path) -> dict:
    from .prep_v16b import read_sealed
    record = read_sealed(path)
    if set(record) != {"version", "settings", "sha256"} or record["version"] != "v18b-execution-settings-v1":
        raise ValueError("invalid v18b execution settings receipt")
    return validate_settings(record["settings"])


def main(argv: list[str] | None = None) -> int:
    from .prep_v16b import seal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    settings = validate_settings(json.loads(args.settings_json))
    seal(args.output, {"version": "v18b-execution-settings-v1", "settings": settings})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
