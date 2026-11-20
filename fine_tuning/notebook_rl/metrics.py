"""Finite, append-only operational evidence and predeclared health thresholds."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

REQUIRED = frozenset(("step", "policy_version", "n_problems", "n_discarded", "n_groups_by_kind",
    "reward_mean_by_kind", "degenerate_fraction", "kl_mean", "masked_fraction", "mean_abs_lp_diff",
    "grad_norm", "vision_on_fraction", "len_median", "truncation_rate", "short_fraction_le32",
    "unverifiable_fraction", "dev_first_attempt_acc", "seconds"))
FRACTIONS = ("degenerate_fraction", "masked_fraction", "vision_on_fraction", "truncation_rate",
             "short_fraction_le32", "unverifiable_fraction", "dev_first_attempt_acc")


def step_record(**fields) -> dict:
    if not REQUIRED <= fields.keys():
        raise ValueError("metrics: missing required observations")
    for key in ("step", "policy_version", "n_problems", "n_discarded"):
        if type(fields[key]) is not int or fields[key] < 0:
            raise ValueError("metrics: invalid counter")
    if fields["policy_version"] != fields["step"]:
        raise ValueError("metrics: record only a fully synchronized step")
    def finite(value):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("metrics: nonfinite observation; record a separate failure event")
        if type(value) is dict:
            for child in value.values():
                finite(child)
        if type(value) in (tuple, list):
            for child in value:
                finite(child)
    finite(fields)
    for key in FRACTIONS:
        value = fields[key]
        if value is not None and (type(value) not in (float, int) or not 0 <= value <= 1):
            raise ValueError("metrics: invalid fraction")
    for key in ("kl_mean", "mean_abs_lp_diff", "len_median"):
        if fields[key] is not None and (type(fields[key]) not in (int, float) or fields[key] < 0):
            raise ValueError("metrics: invalid nonnegative measurement")
    for key in ("n_groups_by_kind", "reward_mean_by_kind", "grad_norm", "seconds"):
        if type(fields[key]) is not dict:
            raise ValueError("metrics: named measurements required")
    if (set(fields["grad_norm"]) != {"L", "M", "V"}
            or set(fields["seconds"]) != {"rollout", "old", "ref", "train", "push"}):
        raise ValueError("metrics: incomplete partition/time measurements")
    for value in (*fields["grad_norm"].values(), *fields["seconds"].values()):
        if value is not None and (type(value) not in (int, float) or value < 0):
            raise ValueError("metrics: invalid partition/time measurement")
    # JSON serialization also rejects unknown non-wire objects.
    return json.loads(json.dumps(fields, allow_nan=False))


def health_check(history: list[dict], protocol: dict, f0_reference: dict) -> list[str]:
    if not history:
        return []
    for record in history:
        step_record(**record)
    if any(b["step"] != a["step"]+1 for a, b in zip(history, history[1:])):
        raise ValueError("metrics: missing or repeated steps")
    limits, last, stops = protocol["health"], history[-1], []
    def above(key, threshold, count=3):
        return len(history) >= count and all(r[key] is not None and r[key] > threshold for r in history[-count:])
    if above("kl_mean", limits["kl_max"]):
        stops.append("KL")
    if above("truncation_rate", limits["truncation_max"]):
        stops.append("TRUNCATION")
    mismatch = protocol["loss"]["mismatch"]
    if above("masked_fraction", mismatch["stop_if_masked_fraction_gt"], mismatch["consecutive_steps"]):
        stops.append("MISMATCH")
    reference_length = f0_reference.get("len_median")
    if (reference_length is None or type(reference_length) not in (int, float)
            or not math.isfinite(reference_length) or reference_length <= 0):
        raise ValueError("metrics: actual positive Phase-A length reference required")
    if (last["len_median"] is not None and last["len_median"] < reference_length*limits["length_median_min_ratio"]
            or last["short_fraction_le32"] is not None and last["short_fraction_le32"] > limits["short_response_max_fraction"]):
        stops.append("LENGTH_COLLAPSE")
    if last["unverifiable_fraction"] is not None and last["unverifiable_fraction"] > limits["unverifiable_max"]:
        stops.append("VERIFIER")
    dev = [r["dev_first_attempt_acc"] for r in history if r["dev_first_attempt_acc"] is not None]
    baseline = f0_reference.get("dev_first_attempt_acc")
    if dev:
        if baseline is None or type(baseline) not in (float, int) or not 0 <= baseline <= 1:
            raise ValueError("metrics: actual F0 dev accuracy required")
        if len(dev) >= 2 and all(baseline-value >= limits["dev_drop_pp"]/100 for value in dev[-2:]):
            stops.append("DEV_DROP")
    return stops


def append_record(path: Path, record: dict) -> None:
    payload = (json.dumps(step_record(**record), sort_keys=True, allow_nan=False)+"\n").encode()
    if path.is_symlink():
        raise ValueError("metrics: symlink log forbidden")
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("metrics: incomplete prior record; preserve and recover before resuming")
        if raw and json.loads(raw.splitlines()[-1])["step"]+1 != record["step"]:
            raise ValueError("metrics: append must advance exactly one step")
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
