"""Cost/time projection from a MEASURED throughput receipt. Pure stdlib, deterministic.

CLI: python -m fine_tuning.notebook_rl.pod.cost --receipt R.json --plan P.json
         [--cap-usd X] [--output OUT.json]
Every output number is a projection (the receipt is a measurement from a smoke/pilot run dir;
its sha256 is recorded). Exit 3 when the high-end projected cost plus reserve exceeds the cap.

Receipt `throughput-receipt-v1` (written by the smoke/pilot command, never invented here):
  items, completion_tokens, wall_seconds, updates, update_seconds (subset of wall_seconds),
  peak_memory_gib, n_samples, retries_per_item
  optional: startup_seconds (default 0; one-time model/engine load inside wall_seconds, so
  update_seconds + startup_seconds < wall_seconds). The schema version name is unchanged.
Plan `cost-plan-v1`:
  hourly_usd, setup_minutes, reserve_usd, range (0..1, default 0.5), optional cap_usd,
  gpu_memory_gib, stages: [{name, items, n_samples, retries_per_item}]
Scaling: generation time per item scales with n_samples and (1 + retries_per_item) relative
to the receipt; update time per item is taken as measured (not scaled); startup_seconds is taken
out of wall_seconds before the per-item rates and added once per stage.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from .common import read_json, sha256_file, write_json

RECEIPT_VERSION = "throughput-receipt-v1"
PLAN_VERSION = "cost-plan-v1"
MEMORY_HEADROOM = 0.92  # warn when peak memory exceeds this share of the planned GPU


class CostError(ValueError):
    pass


def _number(source: dict[str, Any], key: str, *, positive: bool = False,
            integer: bool = False, default: float | None = None) -> float:
    value = source.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CostError(f"{key} must be a finite number")
    if integer and value != int(value):
        raise CostError(f"{key} must be an integer")
    if value < 0 or (positive and value == 0):
        raise CostError(f"{key} must be {'positive' if positive else 'non-negative'}")
    return float(value)


def _spread(mid: float, span: float, unit: float = 1.0) -> dict[str, float]:
    return {"low": round(mid * (1 - span) * unit, 4), "mid": round(mid * unit, 4),
            "high": round(mid * (1 + span) * unit, 4)}


def project(receipt: dict[str, Any], plan: dict[str, Any], cap_usd: float | None = None,
            *, receipt_sha256: str, plan_sha256: str) -> dict[str, Any]:
    if receipt.get("version") != RECEIPT_VERSION or plan.get("version") != PLAN_VERSION:
        raise CostError(f"versions must be {RECEIPT_VERSION} and {PLAN_VERSION}")
    items = _number(receipt, "items", positive=True, integer=True)
    tokens = _number(receipt, "completion_tokens", positive=True)
    wall = _number(receipt, "wall_seconds", positive=True)
    updates = _number(receipt, "updates", integer=True)
    update_seconds = _number(receipt, "update_seconds", default=0)
    startup = _number(receipt, "startup_seconds", default=0)
    peak = _number(receipt, "peak_memory_gib", positive=True)
    measured_n = _number(receipt, "n_samples", positive=True)
    measured_retries = _number(receipt, "retries_per_item", default=0)
    if update_seconds + startup >= wall or (updates == 0 and update_seconds > 0):
        raise CostError("update_seconds + startup_seconds must be < wall_seconds and "
                        "update_seconds needs updates > 0")
    hourly = _number(plan, "hourly_usd", positive=True)
    setup_hours = _number(plan, "setup_minutes", default=0) / 60
    reserve = _number(plan, "reserve_usd", default=0)
    span = _number(plan, "range", default=0.5)
    if span >= 1:
        raise CostError("range must be < 1")
    if cap_usd is not None:  # NaN would make `with_reserve > cap` False and never refuse
        cap_usd = _number({"cap_usd": cap_usd}, "cap_usd", positive=True)
    elif plan.get("cap_usd") is not None:
        cap_usd = _number(plan, "cap_usd", positive=True)
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise CostError("plan.stages must be a non-empty list")
    generation_wall = wall - update_seconds - startup
    generation_per_item = generation_wall / items
    update_per_item = update_seconds / items
    rows, total_mid = [], setup_hours
    for stage in stages:
        if not isinstance(stage.get("name"), str) or not stage["name"]:
            raise CostError("every stage needs a name")
        count = _number(stage, "items", positive=True, integer=True)
        scale = (_number(stage, "n_samples", positive=True) / measured_n
                 * (1 + _number(stage, "retries_per_item", default=0)) / (1 + measured_retries))
        hours = (count * (generation_per_item * scale + update_per_item) + startup) / 3600
        total_mid += hours
        rows.append({"name": stage["name"], "label": "projection", "items": int(count),
                     "hours": _spread(hours, span), "usd": _spread(hours, span, hourly)})
    total_usd = _spread(total_mid, span, hourly)
    with_reserve = round(total_usd["high"] + reserve, 4)
    warnings = []
    if plan.get("gpu_memory_gib") is not None:
        capacity = _number(plan, "gpu_memory_gib", positive=True)
        if peak > capacity * MEMORY_HEADROOM:
            warnings.append(f"measured peak {peak} GiB exceeds {MEMORY_HEADROOM:.0%} of "
                            f"{capacity} GiB")
    return {
        "version": "cost-projection-v1", "label": "projection", "receipt_sha256": receipt_sha256,
        "plan_sha256": plan_sha256,
        "measured_receipt": {"label": "measured", "items": int(items), "completion_tokens": tokens,
                             "wall_seconds": wall, "updates": int(updates),
                             "update_seconds": update_seconds, "startup_seconds": startup,
                             "peak_memory_gib": peak,
                             "n_samples": measured_n, "retries_per_item": measured_retries},
        "derived": {"label": "projection", "completion_tokens_per_generation_second":
                    round(tokens / generation_wall, 4),
                    "generation_seconds_per_item": round(generation_per_item, 4),
                    "update_seconds_per_item": round(update_per_item, 4)},
        "assumptions": {"label": "projection", "hourly_usd": hourly, "setup_minutes":
                        round(setup_hours * 60, 4), "range": span, "reserve_usd": reserve},
        "stages": rows,
        "setup": {"label": "projection", "hours": _spread(setup_hours, span),
                  "usd": _spread(setup_hours, span, hourly)},
        "total": {"label": "projection", "hours": _spread(total_mid, span), "usd": total_usd,
                  "usd_high_plus_reserve": with_reserve},
        "cap_usd": cap_usd, "refuse": cap_usd is not None and with_reserve > cap_usd,
        "warnings": warnings}


def table(result: dict[str, Any]) -> str:
    lines = [f"ALL FIGURES ARE PROJECTIONS from receipt sha256={result['receipt_sha256']}",
             f"{'stage':<24}{'items':>7}  {'hours low/mid/high':<26}{'usd low/mid/high'}"]
    for row in [*result["stages"], {"name": "setup", "items": "", **result["setup"]},
                {"name": "TOTAL", "items": "", **result["total"]}]:
        hours, usd = row["hours"], row["usd"]
        h = f"{hours['low']:.2f}/{hours['mid']:.2f}/{hours['high']:.2f}"
        u = f"{usd['low']:.2f}/{usd['mid']:.2f}/{usd['high']:.2f}"
        lines.append(f"{row['name']:<24}{row['items']:>7}  {h:<26}{u}")
    lines.append(f"usd high + reserve = {result['total']['usd_high_plus_reserve']:.2f} (projection)"
                 f"; cap = {result['cap_usd']}; REFUSE = {result['refuse']}")
    lines.extend(f"WARNING: {warning}" for warning in result["warnings"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cap-usd", type=float)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.output and Path(args.output).exists():
            raise CostError("refusing to overwrite an existing projection file")
        result = project(read_json(args.receipt), read_json(args.plan), args.cap_usd,
                         receipt_sha256=sha256_file(args.receipt),
                         plan_sha256=sha256_file(args.plan))
    except (CostError, OSError, ValueError, AttributeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(table(result))
    if args.output:
        write_json(args.output, result)
    return 3 if result["refuse"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
