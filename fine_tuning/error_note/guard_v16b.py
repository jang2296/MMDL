"""Frozen rule-C training guards and funding decisions; no cloud calls or evaluation launch."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

RULES = ("ACC_DROP_SELF", "TRUNC_UP", "NOFINAL_DRIFT", "COLLAPSE_LEN", "COLLAPSE_SHORT",
         "TAIL_RATE", "IS_DRIFT", "ORDER_MISMATCH")
GUARD_VERSION = "error-note-v16b-guard-c-v1"
EVAL_SECONDS = 23400


class GuardV16B:
    """Append one completed RL block; serializable state belongs in the campaign checkpoint."""

    version = GUARD_VERSION

    def __init__(self, order: list[list[str]], state: dict | None = None):
        ids = [item for block in order for item in block]
        if (not order or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids)
                or any(not 1 <= len(block) <= 16 for block in order)):
            raise ValueError("v16b guard: unique planned IDs in blocks of at most 16 required")
        self.order = copy.deepcopy(order)
        binding = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
        self.state = copy.deepcopy(state) if state is not None else {
            "version": self.version, "order_sha256": binding, "history": [], "reasons": [],
            "collapse_windows": 0, "tail_streak": 0, "hold_streak": 0}
        if (self.state.get("version") != self.version or self.state.get("order_sha256") != binding
                or len(self.state["history"]) > len(order)):
            raise ValueError("v16b guard: state/order binding mismatch")

    def update(self, block: int, items: list[dict], *, tail_rate: float | None = None,
               is_hold: bool = False) -> dict:
        if type(block) is not int or block != len(self.state["history"]) + 1 or block > len(self.order):
            raise ValueError("v16b guard: append completed blocks in order, starting at 1")
        if not items:
            raise ValueError("v16b guard: empty completed block")
        if type(is_hold) is not bool or (tail_rate is not None and (
                isinstance(tail_rate, bool) or not isinstance(tail_rate, (int, float))
                or not math.isfinite(tail_rate) or not 0 <= tail_rate <= 1)):
            raise ValueError("v16b guard: finite tail proportion and boolean HOLD required")
        for item in items:
            if (any(type(item.get(k)) is not bool for k in ("correct", "truncated", "no_final"))
                    or type(item.get("tokens")) is not int or item["tokens"] < 0
                    or (item.get("f0_tokens") is not None and (
                        type(item["f0_tokens"]) is not int or item["f0_tokens"] < 0))):
                raise ValueError("v16b guard: exact T1 booleans and nonnegative lengths required")
        reasons = []
        if [i["id"] for i in items] != self.order[block - 1]:
            reasons.append("ORDER_MISMATCH")
        row = {"n": len(items), "correct": sum(i["correct"] for i in items),
               "truncated": sum(i["truncated"] for i in items), "no_final": sum(i["no_final"] for i in items),
               "short": sum(i["tokens"] < 32 for i in items), "tail_rate": tail_rate,
               "length_pairs": [[i["tokens"], i["f0_tokens"]] for i in items if i.get("f0_tokens") is not None]}
        if self.version == "error-note-v17-guard-v1":
            row["lengths"] = [i["tokens"] for i in items]
            row["sources"] = {source: [i["tokens"] for i in items if i.get("source", "UNKNOWN") == source]
                              for source in sorted({i.get("source", "UNKNOWN") for i in items})}
        history = self.state["history"]
        history.append(row)
        if block >= 2 and all(r["short"] >= 12 for r in history[-2:]):
            reasons.append("COLLAPSE_SHORT")
        self.state["hold_streak"] = self.state["hold_streak"] + 1 if is_hold else 0
        if self.state["hold_streak"] >= 2:
            reasons.append("IS_DRIFT")
        if block > 4 and tail_rate is not None and all(r["tail_rate"] is not None for r in history[:4]):
            baseline = statistics.mean(r["tail_rate"] for r in history[:4])
            self.state["tail_streak"] = self.state["tail_streak"] + 1 if tail_rate > 2 * baseline else 0
        else:
            self.state["tail_streak"] = 0
        if self.state["tail_streak"] >= 2:
            reasons.append("TAIL_RATE")
        if block % 6 == 0:
            pairs = [pair for r in history[-6:] for pair in r["length_pairs"]]
            collapsed = bool(pairs) and statistics.median(p[0] for p in pairs) < 0.5 * statistics.median(p[1] for p in pairs)
            self.state["collapse_windows"] = self.state["collapse_windows"] + 1 if collapsed else 0
            if block >= 12 and self.state["collapse_windows"] >= 2:
                if getattr(self, "collapse_stop", True):
                    reasons.append("COLLAPSE_LEN")
                elif block not in self.state.setdefault("collapse_warning_blocks", []):
                    self.state["collapse_warning_blocks"].append(block)
        thresholds = {}
        if block >= 36:
            reference, recent = history[:24], history[-12:]
            ref_n, recent_n = sum(r["n"] for r in reference), sum(r["n"] for r in recent)
            ref_acc = sum(r["correct"] for r in reference) / ref_n
            thresholds["accuracy"] = ref_acc - 0.15
            if sum(r["correct"] for r in recent) / recent_n < thresholds["accuracy"]:
                reasons.append("ACC_DROP_SELF")
            for field, rule in (("truncated", "TRUNC_UP"), ("no_final", "NOFINAL_DRIFT")):
                ref = sum(r[field] for r in reference) * (recent_n / ref_n if self.version == "error-note-v17-guard-v1" else 12 / 24)
                minimum = (max(4, math.ceil(2 * ref), math.ceil(ref + 4 * math.sqrt(ref)))
                           if self.version == "error-note-v17-guard-v1" else max(2 * ref, ref + 4 * math.sqrt(ref)))
                thresholds[field] = {"ref": ref, "minimum": minimum}
                if sum(r[field] for r in recent) >= thresholds[field]["minimum"]:
                    if rule == "TRUNC_UP" and not getattr(self, "trunc_stop", True):
                        if block not in self.state.setdefault("trunc_warning_blocks", []):
                            self.state["trunc_warning_blocks"].append(block)
                    else:
                        reasons.append(rule)
        if self.version == "error-note-v17-guard-v1":
            trend = short_drift_v17(history)
            if trend["status"] == "STOP_REQUESTED":
                reasons.append("SHORT_DRIFT_STOP")
            self.state["short_drift"] = trend
        self.state["reasons"] = sorted(set(self.state["reasons"] + reasons))
        return {"version": self.version, "block": block, "status": "STOP_REQUESTED" if self.state["reasons"] else "CONTINUE",
                "reasons": self.state["reasons"][:], "thresholds": thresholds, "auto_evaluate": False}

def length_distribution_v17(lengths: list[int]) -> dict:
    bounds = (32, 128, 256, 1024, 4096, 8192)
    histogram = dict.fromkeys(("<32", "32..127", "128..255", "256..1023",
                              "1024..4095", "4096..8191", ">=8192"), 0)
    values = sorted(lengths)
    for value in values:
        histogram[list(histogram)[sum(value >= bound for bound in bounds)]] += 1
    def quantile(p: float) -> float | None:
        if not values:
            return None
        pos = (len(values) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"n": len(values), "short_lt32": histogram["<32"], "histogram": histogram,
            "p10": quantile(.1), "median": quantile(.5), "p90": quantile(.9), "p95": quantile(.95)}


def short_drift_v17(history: list[dict]) -> dict:
    if len(history) < 36:
        return {"status": "WARMUP", "first24": None, "recent12": None, "delta": None,
                "se": None, "stop_threshold": None, "warnings": []}
    reference, recent = history[:24], history[-12:]
    ref_n, recent_n = sum(r["n"] for r in reference), sum(r["n"] for r in recent)
    ref_short, recent_short = sum(r["short"] for r in reference), sum(r["short"] for r in recent)
    delta = recent_short / recent_n - ref_short / ref_n
    pooled = (ref_short + recent_short) / (ref_n + recent_n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / ref_n + 1 / recent_n))
    threshold = max(.15, 4 * se)
    warning = ["SHORT_DRIFT_WARNING"] if delta >= .05 else []
    if warning and sum(r["truncated"] for r in recent) / recent_n > sum(r["truncated"] for r in reference) / ref_n:
        warning.append("LENGTH_POLARIZATION_WARNING")
    return {"status": "STOP_REQUESTED" if delta >= threshold else "WARNING" if warning else "CONTINUE",
            "first24": {"short": ref_short, "n": ref_n}, "recent12": {"short": recent_short, "n": recent_n},
            "delta": delta, "se": se, "stop_threshold": threshold, "warnings": warning}


class GuardV17(GuardV16B):
    version = "error-note-v17-guard-v1"

    def __init__(self, order: list[list[str]], state: dict | None = None, *, telemetry: bool = True,
                 collapse_stop: bool = True):
        super().__init__(order, state)
        self.telemetry = telemetry
        # v18 (user 2026-11-19 KST): COLLAPSE_LEN compares medians with F0 priors that the false-alarm simulation
        # cannot exercise (it sets tokens = f0_tokens). The first v18 pod run stopped on it at block 66 while
        # training accuracy rose (50% -> 57-65%) and <32-token answers fell (26% -> 12%); for v18 it is a recorded
        # warning. Accuracy, truncation, no-final, short-drift, tail and HOLD rules still stop the run.
        self.collapse_stop = collapse_stop
        # v18b (user 2026-11-19 KST): the second v18b pod run stopped on TRUNC_UP at RL block 49 (8 length-capped first
        # solutions in the last 12 blocks, 11 of 14 in one source, hf-ssmr) while accuracy rules stayed quiet and the
        # external monitor held (138 -> 140 of 200). Where COLLAPSE_LEN is a warning, TRUNC_UP is a recorded warning
        # too; accuracy, no-final, short-drift, tail and HOLD rules still stop the run.
        self.trunc_stop = collapse_stop
        if not self.trunc_stop and "TRUNC_UP" in self.state.get("reasons", []):
            # A stop saved before the demotion is the warning it now is; the stopped checkpoint files keep the record.
            self.state["reasons"] = [reason for reason in self.state["reasons"] if reason != "TRUNC_UP"]
            blocks = self.state.setdefault("trunc_warning_blocks", [])
            if len(self.state["history"]) not in blocks:
                blocks.append(len(self.state["history"]))

    def update(self, block: int, items: list[dict], *, tail_rate: float | None = None,
               is_hold: bool = False) -> dict:
        receipt = super().update(block, items, tail_rate=tail_rate, is_hold=is_hold)
        receipt["short_drift"] = copy.deepcopy(self.state["short_drift"])
        receipt["warnings"] = receipt["short_drift"]["warnings"] + (
            ["COLLAPSE_LEN_WARNING"] if self.state.get("collapse_warning_blocks") else []) + (
            ["TRUNC_UP_WARNING"] if self.state.get("trunc_warning_blocks") else [])
        receipt["collapse_warning_blocks"] = list(self.state.get("collapse_warning_blocks", []))
        receipt["trunc_warning_blocks"] = list(self.state.get("trunc_warning_blocks", []))
        if self.telemetry:
            history = self.state["history"]
            receipt["length_summary"] = length_distribution_v17([v for r in history for v in r["lengths"]])
            receipt["length_n"] = receipt["length_summary"]["n"]
            receipt["short_lt32"] = receipt["length_summary"]["short_lt32"]
            receipt["no_final"] = sum(r["no_final"] for r in history)
            receipt["truncated"] = sum(r["truncated"] for r in history)
            receipt["cap_rate_8192"] = receipt["length_summary"]["histogram"][">=8192"] / receipt["length_n"]
            receipt["sources"] = {source: length_distribution_v17([v for r in history for v in r["sources"].get(source, [])])
                                  for source in sorted({s for r in history for s in r["sources"]})}
        return receipt


class SignalGateV17:
    """Fixed supported-parent gate; only checkpoint-linked actual N content updates count."""

    def __init__(self, order: list[list[str]], supported_parent_ids: list[str], state: dict | None = None,
                 *, local_smoke: bool = False):
        ids = [pid for block in order for pid in block]
        if (not ids or len(ids) != len(set(ids)) or len(supported_parent_ids) != len(set(supported_parent_ids))
                or not set(supported_parent_ids) <= set(ids)):
            raise ValueError("v17 signal gate: unique ordered parent and supported IDs required")
        supported = [pid for pid in ids if pid in set(supported_parent_ids)]
        if len(supported) < 32 and not (local_smoke and len(ids) <= 8):
            raise ValueError("SIGNAL_COVERAGE_INSUFFICIENT: fewer than 32 source-bound repair parents")
        self.parents = supported[:32]
        self.due_slot = max((b for b, block in enumerate(order) if set(block) & set(self.parents)), default=0) + 3
        binding = hashlib.sha256(json.dumps({"order": order, "parents": self.parents, "local_smoke": local_smoke},
                                            sort_keys=True).encode()).hexdigest()
        self.state = copy.deepcopy(state) if state is not None else {
            "version": "v17-signal-gate-v1", "binding_sha256": binding, "parents": self.parents,
            "due_slot": self.due_slot, "pending_parent_ids": [], "verified_repair_parent_ids": [],
            "verified_content_tokens": 0, "content_gradient_linked": False, "checkpoint_ids": [],
            "status": "LOCAL_SMOKE_UNVERIFIED" if local_smoke else "WARMUP", "local_smoke": local_smoke}
        if self.state.get("binding_sha256") != binding or self.state.get("version") != "v17-signal-gate-v1":
            raise ValueError("v17 signal gate: resume binding mismatch")

    def consume(self, stats: dict) -> None:
        if (stats.get("optimizer_step") is not True or stats.get("hold") or stats.get("HOLD")
                or stats.get("n_content_gradient_finite_nonzero") is not True
                or stats.get("note_gradient_retained") is not True):
            return
        tokens = stats.get("n_verified_content_tokens")
        parents = stats.get("verified_repair_parent_ids", [])
        if type(tokens) is not int or tokens <= 0 or not isinstance(parents, list):
            return
        accepted = sorted(set(parents) & set(self.parents))
        if accepted:
            self.state["pending_parent_ids"] = sorted(set(self.state["pending_parent_ids"]) | set(accepted))
            self.state["verified_content_tokens"] += tokens

    def mark_checkpoint(self, checkpoint_id: str) -> None:
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("v17 signal gate: checkpoint identity required")
        pending = self.state["pending_parent_ids"]
        if pending:
            self.state["verified_repair_parent_ids"] = sorted(set(self.state["verified_repair_parent_ids"]) | set(pending))
            self.state["content_gradient_linked"] = True
            self.state["pending_parent_ids"] = []
        if checkpoint_id not in self.state["checkpoint_ids"]:
            self.state["checkpoint_ids"].append(checkpoint_id)

    def evaluate(self, slot: int, closed_parent_ids: list[str], opportunity_parent_ids: list[str]) -> dict:
        if self.state["local_smoke"]:
            return copy.deepcopy(self.state)
        if self.state["status"] in ("PASS", "FAIL", "INCONCLUSIVE_SIGNAL"):
            return copy.deepcopy(self.state)
        if slot < self.due_slot:
            return copy.deepcopy(self.state)
        if not set(self.parents) <= set(closed_parent_ids):
            raise RuntimeError("v17 signal gate: scheduled NOTE/C closure is incomplete")
        if self.state["pending_parent_ids"]:
            return self.state | {"status": "CHECKPOINT_REQUIRED"}
        opportunities = set(opportunity_parent_ids) & set(self.parents)
        self.state["opportunity_parent_ids"] = sorted(opportunities)
        if not opportunities:
            self.state["status"] = "INCONCLUSIVE_SIGNAL"
        elif (len(self.state["verified_repair_parent_ids"]) >= 3 and self.state["verified_content_tokens"] > 0
              and self.state["content_gradient_linked"]):
            self.state["status"] = "PASS"
        else:
            self.state["status"] = "FAIL"
        return copy.deepcopy(self.state)



class SignalGateV18B:
    """Delayed per-parent N evidence; counts become saved only at a checkpoint boundary."""

    def __init__(self, order: list[list[str]], supported_parent_ids: list[str], state: dict | None = None,
                 *, local_smoke: bool = False):
        ids = [pid for block in order for pid in block]
        if (len(ids) != len(set(ids)) or not set(supported_parent_ids) <= set(ids)
                or len(supported_parent_ids) != len(set(supported_parent_ids))):
            raise ValueError("V18B_SIGNAL_PARENT_BINDING")
        self.positions = {pid: slot for slot, block in enumerate(order) for pid in block}
        self.supported = [pid for pid in ids if pid in set(supported_parent_ids)]
        self.parents = self.supported[:32]
        self.due_slot = max((self.positions[pid] for pid in self.parents), default=0) + 3
        if len(self.parents) < 32 and not local_smoke:
            raise ValueError("SIGNAL_COVERAGE_INSUFFICIENT")
        binding = hashlib.sha256(json.dumps([order, self.supported, local_smoke], sort_keys=True).encode()).hexdigest()
        self.state = copy.deepcopy(state) if state is not None else {
            "version": "v18b-signal-gate-v1", "binding_sha256": binding,
            "parents": self.parents, "pending_rows": {}, "saved_rows": {}, "checkpoint_ids": [],
            "local_smoke": local_smoke, "status": "WARMUP"}
        if self.state.get("version") != "v18b-signal-gate-v1" or self.state.get("binding_sha256") != binding:
            raise ValueError("V18B_SIGNAL_RESUME_BINDING")

    def consume(self, stats: dict) -> None:
        if (stats.get("optimizer_step") is not True or stats.get("hold") or stats.get("HOLD")
                or stats.get("n_content_gradient_finite_nonzero") is not True
                or stats.get("note_gradient_retained") is not True):
            return
        # Batch totals cannot certify an individual parent's supervision or retained gradient.
        for row in stats.get("n_parent_evidence", []):
            pid = row.get("problem_id")
            if pid not in self.supported:
                continue
            if (row.get("row_id") != pid + ":N" or type(row.get("supervised_tokens")) is not int
                    or row["supervised_tokens"] <= 0 or row.get("gradient_finite_nonzero") is not True
                    or row.get("note_gradient_retained") is not True):
                raise ValueError("V18B_SIGNAL_INVALID_PARENT_EVIDENCE")
            rid = row["row_id"]
            previous = self.state["pending_rows"].get(rid) or self.state["saved_rows"].get(rid)
            if previous is not None:
                if {k: v for k, v in previous.items() if k != "checkpoint_id"} != row:
                    raise ValueError("V18B_SIGNAL_ROW_EVIDENCE_CHANGED")
            else:
                self.state["pending_rows"][rid] = copy.deepcopy(row)

    def mark_checkpoint(self, checkpoint_id: str) -> None:
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("V18B_SIGNAL_CHECKPOINT_REQUIRED")
        for rid, row in self.state["pending_rows"].items():
            self.state["saved_rows"][rid] = row | {"checkpoint_id": checkpoint_id}
        self.state["pending_rows"] = {}
        if checkpoint_id not in self.state["checkpoint_ids"]:
            self.state["checkpoint_ids"].append(checkpoint_id)

    def evaluate(self, slot: int, closed_parent_ids: list[str], opportunity_parent_ids: list[str] | dict,
                 *, contract_failure_reproduced: bool = False) -> dict:
        if type(slot) is not int or slot < 0 or type(contract_failure_reproduced) is not bool:
            raise ValueError("V18B_SIGNAL_SLOT_OR_CONTRACT_FAILURE")
        stages = opportunity_parent_ids if isinstance(opportunity_parent_ids, dict) else {"UNKNOWN": opportunity_parent_ids}
        if not set(stages) <= {"T1", "OBS", "UNKNOWN"}:
            raise ValueError("V18B_SIGNAL_OPPORTUNITY_STAGE")
        opportunities = set().union(*(set(pids) for pids in stages.values()))
        if not opportunities <= set(self.supported):
            raise ValueError("V18B_SIGNAL_UNSUPPORTED_OPPORTUNITY")
        closed = set(closed_parent_ids)
        mature = {pid for pid in opportunities if pid in closed and slot >= self.positions[pid] + 3}
        fixed = set(self.parents)
        fixed_mature = {pid for pid in fixed if pid in closed and slot >= self.positions[pid] + 3}
        saved = self.state["saved_rows"]
        linked = {row["problem_id"] for row in saved.values()}
        fixed_rows = [row for row in saved.values() if row["problem_id"] in fixed]
        if self.state["local_smoke"]:
            status = "LOCAL_SMOKE_UNVERIFIED"
        elif fixed_mature == fixed:
            status = ("INCONCLUSIVE_SIGNAL" if not mature & fixed else "PASS" if len(linked & fixed) >= 3 else "FAIL")
        else:
            status = "WARMUP"
        self.state.update(status=status,
            opportunity_parent_ids=sorted(opportunities), mature_opportunity_parent_ids=sorted(mature),
            opportunity_stages={stage: {"observed": len(set(pids)), "mature": len(set(pids) & mature)}
                                for stage, pids in stages.items()},
            verified_repair_parent_ids=sorted(linked & fixed),
            verified_content_tokens=sum(row["supervised_tokens"] for row in fixed_rows),
            content_gradient_linked=bool(fixed_rows),
            early_zero_diagnostic=len(mature) >= 8 and not linked & mature,
            fixed32_zero=status == "FAIL" and not linked & fixed,
            stop_requested=bool(status == "FAIL" and not linked & fixed and contract_failure_reproduced),
            contract_failure_reproduced=contract_failure_reproduced,
            pending_parent_ids=sorted({r["problem_id"] for r in self.state["pending_rows"].values()}))
        return copy.deepcopy(self.state)


def funding_decision(balance_usd: float, hourly_usd: float, now: float,
                     requested_at: float | None = None) -> dict:
    """Caller polls actual balance every ten minutes; recharge never authorizes a new run."""
    values = [balance_usd, hourly_usd, now] + ([] if requested_at is None else [requested_at])
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values)
            or hourly_usd <= 0 or (requested_at is not None and requested_at > now)):
        raise ValueError("v16b funding: finite actual balance, rate and monotonic timestamps required")
    protected = 5.65 * hourly_usd + 1.5
    low = balance_usd - protected < hourly_usd
    started = requested_at if requested_at is not None else now
    status = "CONTINUE" if not low else "STOP_REQUESTED" if now - started >= 1800 else "RECHARGE_REQUESTED"
    return {"status": status, "protected_usd": protected, "next_hour_usd": hourly_usd,
            "requested_at": started if low else None, "auto_evaluate": False}

def funding_decision_v17(balance_usd: float, hourly_usd: float, now: float, *,
                         evaluation_seconds: float, other_post_train_seconds: float,
                         recovery_seconds: float, storage_usd_per_hour: float, reserve_usd: float,
                         next_train_seconds: float, requested_at: float | None = None) -> dict:
    """Protect calibrated post-training work; low actual balance gets a fixed 30-minute grace."""
    values = (balance_usd, hourly_usd, now, evaluation_seconds, other_post_train_seconds,
              recovery_seconds, storage_usd_per_hour, reserve_usd, next_train_seconds)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values)
            or hourly_usd <= 0 or evaluation_seconds <= 0 or recovery_seconds <= 0 or next_train_seconds <= 0
            or (requested_at is not None and (isinstance(requested_at, bool)
                or not isinstance(requested_at, (int, float)) or not math.isfinite(requested_at)
                or requested_at < 0 or requested_at > now))):
        raise ValueError("v17 funding: finite actual balance, calibrated costs and consistent timestamps required")
    rate = hourly_usd + storage_usd_per_hour
    protected_seconds = evaluation_seconds + other_post_train_seconds + recovery_seconds
    protected = reserve_usd + rate * protected_seconds / 3600
    next_train = rate * next_train_seconds / 3600
    if not all(math.isfinite(v) for v in (rate, protected_seconds, protected, next_train, protected + next_train)):
        raise ValueError("v17 funding: derived cost is not finite")
    low = balance_usd < protected + next_train
    started = requested_at if requested_at is not None else now
    status = "CONTINUE" if not low else "STOP_REQUESTED" if now - started >= 1800 else "RECHARGE_REQUESTED"
    return {"version": "v17-funding-decision-v1", "status": status,
            "balance_usd": balance_usd, "hourly_usd": hourly_usd,
            "storage_usd_per_hour": storage_usd_per_hour, "all_in_hourly_usd": rate,
            "evaluation_seconds": evaluation_seconds, "other_post_train_seconds": other_post_train_seconds,
            "recovery_seconds": recovery_seconds, "post_train_seconds": protected_seconds,
            "reserve_usd": reserve_usd, "protected_usd": protected, "next_train_usd": next_train,
            "requested_at": started if low else None, "recharge_wait_seconds": 1800,
            "stop_requested_at": started + 1800 if low else None, "auto_evaluate": False,
            "auto_recharge": False, "auto_restart": False, "training_elapsed_cutoff": None}


def falsealarm_receipt(manifest: Path, *, trials: int = 2000, seed: int = 3407) -> dict:
    """Source-stratified F0 bootstrap, using only exact prior bindings in the frozen 3,000 pool."""
    if type(trials) is not int or trials < 1:
        raise ValueError("v16b guard: positive bootstrap trials required")
    raw = Path(manifest).read_bytes()
    items = json.loads(raw)["items"]
    if len(items) != 3000 or len({i["problem_id"] for i in items}) != 3000:
        raise ValueError("v16b guard: frozen 3,000 unique parents required")
    sources = defaultdict(list)
    for item in items:
        prior = item.get("f0_prior")
        if prior is not None:
            if prior.get("binding") != "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD":
                raise ValueError("v16b guard: exact F0 prior binding required")
            sources[item["source"]].append(prior)
    n = sum(map(len, sources.values()))
    if not n:
        raise ValueError("v16b guard: no F0 priors")
    rng = random.Random(seed)
    hits = dict.fromkeys(RULES, 0)
    any_hits = 0
    order = [[str(i) for i in range(j, min(j + 16, 3000))] for j in range(0, 3000, 16)]
    for _ in range(trials):
        keyed = []
        for source in sorted(sources):
            rows = sources[source]
            picks = [rng.choice(rows) for _ in range(max(1, round(3000 * len(rows) / n)))]
            phase = rng.random()
            keyed += [((i + phase) / len(picks), rng.random(), prior) for i, prior in enumerate(picks)]
        keyed.sort(key=lambda r: r[:2])
        sampled = [r[2] for r in keyed]
        while len(sampled) < 3000:
            sampled.append(rng.choice(sampled))
        guard = GuardV16B(order)
        for block, ids in enumerate(order, 1):
            batch = []
            for item_id in ids:
                prior = sampled[int(item_id)]
                batch.append({"id": item_id, "correct": prior["status"] == "CORRECT",
                              "tokens": prior["completion_tokens"], "f0_tokens": prior["completion_tokens"],
                              "truncated": prior["finish_reason"] == "length",
                              "no_final": prior["status"] == "INVALID_NO_FINAL"})
            guard.update(block, batch)
        for rule in guard.state["reasons"]:
            hits[rule] += 1
        any_hits += bool(guard.state["reasons"])
    rates = {rule: count / trials for rule, count in hits.items()}
    # F0 priors contain no sampler tail or IS statistics: their false-alarm rate cannot be certified here.
    unmeasured = ["TAIL_RATE", "IS_DRIFT"]
    measured_pass = max([any_hits / trials, *rates.values()]) <= 0.02
    return {"version": "v16b-guard-falsealarm-v2", "guard_version": GUARD_VERSION,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "pool_items": 3000, "prior_items": n,
            "trials": trials, "seed": seed, "method": "source-stratified-interleaved-bootstrap-v1",
            "rule_hits": {rule: None if rule in unmeasured else value for rule, value in hits.items()},
            "rule_rates": {rule: None if rule in unmeasured else value for rule, value in rates.items()},
            "overall_rate": any_hits / trials,
            "unmeasured_rules": unmeasured, "false_alarm_max": 0.02,
            "measured_status": "PASS" if measured_pass else "FAIL", "measured_pass": measured_pass,
            "status": "INCOMPLETE" if measured_pass else "FAIL", "reason": "TAIL_RATE and IS_DRIFT unavailable in frozen f0_prior; no threshold tuning",
            "auto_evaluate": False}


SCOPED_UNMEASURED_V17 = ("IS_DRIFT", "TAIL_RATE")  # run-relative rules: no stored F0 statistic can measure them


def falsealarm_receipt_v17(manifest: Path, *, trials: int = 2000, seed: int = 3407,
                           fill_missing: bool = False, extra_priors: Path | None = None) -> dict:
    """Bootstrap the frozen source/capability positions; missing observations remain unmeasured.

    fill_missing (2026-11-17, the v16b method): a position without an exact F0 prior draws from the priors of the
    same (source, layer, capability) stratum, else the same source, else every prior. The filled positions are
    recorded; with only TAIL_RATE/IS_DRIFT unmeasured a passing bootstrap is PASS_SCOPED, never PASS."""
    if fill_missing:
        return _falsealarm_filled_v17(Path(manifest), trials=trials, seed=seed, extra_priors=extra_priors)
    if type(trials) is not int or trials < 1:
        raise ValueError("v17 guard: positive bootstrap trials required")
    raw = Path(manifest).read_bytes()
    items = json.loads(raw)["items"]
    ids = [item["problem_id"] for item in items]
    if len(ids) != 3000 or len(set(ids)) != 3000:
        raise ValueError("v17 guard: frozen ordered 3,000 unique parents required")
    sources: dict[str, list[dict]] = defaultdict(list)
    keys, missing = [], []
    for pos, item in enumerate(items):
        key = json.dumps([item["source"], item.get("layer"), item.get("capability", item.get("capability_type"))],
                         sort_keys=True, separators=(",", ":"))
        keys.append(key)
        prior = item.get("f0_prior")
        if prior is None:
            missing.append(pos)
            continue
        if (prior.get("binding") != "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD"
                or type(prior.get("completion_tokens")) is not int or prior["completion_tokens"] < 0
                or prior.get("finish_reason") not in ("stop", "length") or not isinstance(prior.get("status"), str)):
            raise ValueError("v17 guard: exact F0 prior response binding required")
        sources[key].append(prior)
    rules = (*RULES, "SHORT_DRIFT_STOP")
    receipt = {"version": "v17-guard-falsealarm-v1", "guard_version": GuardV17.version,
               "manifest_sha256": hashlib.sha256(raw).hexdigest(), "pool_items": len(items),
               "prior_items": len(items) - len(missing), "trials": trials, "seed": seed,
               "method": "fixed-source-capability-position-bootstrap-v1",
               "strata_order_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
               "missing_positions": missing, "block_sizes": [min(16, len(items) - p) for p in range(0, len(items), 16)],
               "false_alarm_max": .02, "auto_evaluate": False}
    if missing:
        return receipt | {"status": "INCOMPLETE", "measured_status": "INCOMPLETE",
                          "overall_rate": None, "overall_upper95": None,
                          "rule_hits": dict.fromkeys(rules), "rule_rates": dict.fromkeys(rules),
                          "unmeasured_rules": list(rules), "reason": "missing frozen F0 priors; no imputation"}
    unmeasured = []
    if any(prior.get("tail_rate") is None for rows in sources.values() for prior in rows):
        unmeasured.append("TAIL_RATE")
    if any(type(prior.get("is_hold")) is not bool for rows in sources.values() for prior in rows):
        unmeasured.append("IS_DRIFT")
    rng, hits, any_hits = random.Random(seed), dict.fromkeys(rules, 0), 0
    order = [ids[p:p + 16] for p in range(0, len(ids), 16)]
    for _ in range(trials):
        sampled = [rng.choice(sources[key]) for key in keys]
        guard = GuardV17(order, telemetry=False)
        for block, planned in enumerate(order, 1):
            priors = sampled[(block - 1) * 16:block * 16]
            batch = [{"id": pid, "source": items[(block - 1) * 16 + j]["source"],
                      "correct": prior["status"] == "CORRECT", "tokens": prior["completion_tokens"],
                      "f0_tokens": prior["completion_tokens"], "truncated": prior["finish_reason"] == "length",
                      "no_final": prior["status"] == "INVALID_NO_FINAL"}
                     for j, (pid, prior) in enumerate(zip(planned, priors, strict=True))]
            tail = None if "TAIL_RATE" in unmeasured else statistics.mean(p["tail_rate"] for p in priors)
            hold = False if "IS_DRIFT" in unmeasured else any(p["is_hold"] for p in priors)
            guard.update(block, batch, tail_rate=tail, is_hold=hold)
        measured_reasons = set(guard.state["reasons"]) - set(unmeasured)
        for rule in measured_reasons:
            hits[rule] += 1
        any_hits += bool(measured_reasons)
    rate, z = any_hits / trials, 1.959963984540054
    upper = (rate + z*z/(2*trials) + z*math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials))) / (1 + z*z/trials)
    passed = trials >= 2000 and upper <= .02
    return receipt | {"rule_hits": {r: None if r in unmeasured else v for r, v in hits.items()},
                      "rule_rates": {r: None if r in unmeasured else v / trials for r, v in hits.items()},
                      "overall_rate": rate, "overall_upper95": upper, "unmeasured_rules": unmeasured,
                      "measured_status": "PASS" if passed else "INCOMPLETE" if trials < 2000 else "FAIL",
                      "status": "PASS" if passed and not unmeasured else "INCOMPLETE" if passed or trials < 2000 else "FAIL",
                      "reason": "unmeasured rules remain null; thresholds were not tuned"}


def _falsealarm_filled_v17(manifest: Path, *, trials: int, seed: int, extra_priors: Path | None = None) -> dict:
    """extra_priors: {"version": "v17-guard-extra-priors-v1", "sources": {source: [{status, completion_tokens,
    finish_reason}, ...]}} of real F0 responses on OTHER items of the same source (e.g. recovered B2-sampled F0
    preparation/monitor generations). Used only for a source without any exact prior, before the all-prior pool."""
    extra: dict[str, list[dict]] = {}
    extra_sha = None
    if extra_priors is not None:
        extra_raw = Path(extra_priors).read_bytes()
        extra_sha = hashlib.sha256(extra_raw).hexdigest()
        body = json.loads(extra_raw)
        if body.get("version") != "v17-guard-extra-priors-v1":
            raise ValueError("v17 guard: versioned extra prior pool required")
        for source, rows in body["sources"].items():
            for prior in rows:
                if (type(prior.get("completion_tokens")) is not int or prior["completion_tokens"] < 0
                        or prior.get("finish_reason") not in ("stop", "length") or not isinstance(prior.get("status"), str)):
                    raise ValueError("v17 guard: extra prior response fields required")
            extra[source] = list(rows)
    raw = manifest.read_bytes()
    items = json.loads(raw)["items"]
    ids = [item["problem_id"] for item in items]
    if len(ids) != 3000 or len(set(ids)) != 3000:
        raise ValueError("v17 guard: frozen ordered 3,000 unique parents required")
    if type(trials) is not int or trials < 1:
        raise ValueError("v17 guard: positive bootstrap trials required")
    keys, by_key, by_source, everything, filled = [], defaultdict(list), defaultdict(list), [], []
    for item in items:
        key = json.dumps([item["source"], item.get("layer"), item.get("capability", item.get("capability_type"))],
                         sort_keys=True, separators=(",", ":"))
        keys.append(key)
        prior = item.get("f0_prior")
        if prior is None:
            continue
        if (prior.get("binding") != "EXACT_NATIVE_ID_IMAGE_QUESTION_CHOICES_GOLD"
                or type(prior.get("completion_tokens")) is not int or prior["completion_tokens"] < 0
                or prior.get("finish_reason") not in ("stop", "length") or not isinstance(prior.get("status"), str)):
            raise ValueError("v17 guard: exact F0 prior response binding required")
        by_key[key].append(prior)
        by_source[item["source"]].append(prior)
        everything.append(prior)
    if not everything:
        raise ValueError("v17 guard: no F0 priors")
    pools = []  # every position resamples its stratum, as the unfilled v17 bootstrap does
    for pos, item in enumerate(items):
        level = ("stratum" if by_key[keys[pos]] else "source" if by_source[item["source"]]
                 else "source_extra" if extra.get(item["source"]) else "all")
        if item.get("f0_prior") is None:
            filled.append({"position": pos, "level": level})
        pools.append(by_key[keys[pos]] if level == "stratum" else by_source[item["source"]] if level == "source"
                     else extra[item["source"]] if level == "source_extra" else everything)
    rules = (*RULES, "SHORT_DRIFT_STOP")
    unmeasured = list(SCOPED_UNMEASURED_V17)
    rng, hits, any_hits = random.Random(seed), dict.fromkeys(rules, 0), 0
    order = [ids[p:p + 16] for p in range(0, len(ids), 16)]
    for _ in range(trials):
        sampled = [rng.choice(pool) for pool in pools]
        guard = GuardV17(order, telemetry=False)
        for block, planned in enumerate(order, 1):
            priors = sampled[(block - 1) * 16:block * 16]
            batch = [{"id": pid, "source": items[(block - 1) * 16 + j]["source"],
                      "correct": prior["status"] == "CORRECT", "tokens": prior["completion_tokens"],
                      "f0_tokens": prior["completion_tokens"], "truncated": prior["finish_reason"] == "length",
                      "no_final": prior["status"] == "INVALID_NO_FINAL"}
                     for j, (pid, prior) in enumerate(zip(planned, priors, strict=True))]
            guard.update(block, batch, tail_rate=None, is_hold=False)
        measured = set(guard.state["reasons"]) - set(unmeasured)
        for rule in measured:
            hits[rule] += 1
        any_hits += bool(measured)
    rate, z = any_hits / trials, 1.959963984540054
    upper = (rate + z*z/(2*trials) + z*math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials))) / (1 + z*z/trials)
    passed = trials >= 2000 and upper <= .02
    return {"version": "v17-guard-falsealarm-v1", "guard_version": GuardV17.version,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "pool_items": len(items),
            "prior_items": len(everything), "trials": trials, "seed": seed,
            "method": "fixed-position-stratified-fill-bootstrap-v1",
            "strata_order_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
            "filled_positions": len(filled), "filled_levels": {lv: sum(f["level"] == lv for f in filled)
                                                              for lv in ("stratum", "source", "source_extra", "all")},
            "extra_priors_sha256": extra_sha,
            "extra_prior_counts": {source: len(rows) for source, rows in sorted(extra.items())},
            "block_sizes": [min(16, len(items) - p) for p in range(0, len(items), 16)],
            "false_alarm_max": .02, "auto_evaluate": False,
            "rule_hits": {r: None if r in unmeasured else v for r, v in hits.items()},
            "rule_rates": {r: None if r in unmeasured else v / trials for r, v in hits.items()},
            "overall_rate": rate, "overall_upper95": upper, "unmeasured_rules": unmeasured,
            "measured_status": "PASS" if passed else "INCOMPLETE" if trials < 2000 else "FAIL",
            "status": "PASS_SCOPED" if passed else "INCOMPLETE" if trials < 2000 else "FAIL",
            "scope": {"uncertified_rules": unmeasured,
                      "why": "TAIL_RATE compares the run with its own first blocks and IS_DRIFT counts HOLD updates; "
                             "stored F0 records carry neither statistic. Thresholds unchanged.",
                      "consequence": "a TAIL_RATE or IS_DRIFT stop ends training at the next save point; "
                                     "no automatic evaluation"},
            "reason": "missing priors filled by stratified resampling (v16b method); run-relative rules unmeasured"}


RISK_ACCEPTED_UPPER95_MAX_V17 = .05  # explicit, recorded acceptance ceiling; the measured status stays as measured


def receipt_passes_v17(receipt: dict, manifest_raw: bytes) -> bool:
    """PASS; PASS_SCOPED whose only unmeasured rules are the run-relative TAIL_RATE and IS_DRIFT; or RISK_ACCEPTED:
    the same scoped bootstrap whose measured upper95 exceeds 2% but is at most 5%, carrying a dated, signed
    risk_acceptance (who, why, consequence). The measured_status (FAIL) is never rewritten."""
    filled = (sorted(receipt.get("unmeasured_rules") or []) == list(SCOPED_UNMEASURED_V17)
              and receipt.get("method") == "fixed-position-stratified-fill-bootstrap-v1")
    upper = receipt.get("overall_upper95")
    if not isinstance(upper, (int, float)) or isinstance(upper, bool):
        return False
    acceptance = receipt.get("risk_acceptance") or {}
    accepted = (receipt.get("status") == "RISK_ACCEPTED" and filled and upper <= RISK_ACCEPTED_UPPER95_MAX_V17
                and receipt.get("measured_status") == "FAIL"
                and all(isinstance(acceptance.get(k), str) and acceptance[k].strip()
                        for k in ("by", "at_utc", "reason", "consequence")))
    passed = ((receipt.get("status") == "PASS" and not receipt.get("unmeasured_rules")
               or receipt.get("status") == "PASS_SCOPED" and filled) and upper <= .02)
    return (receipt.get("version") == "v17-guard-falsealarm-v1" and (passed or accepted)
            and receipt.get("trials", 0) >= 2000
            and receipt.get("manifest_sha256") == hashlib.sha256(manifest_raw).hexdigest())



def materialize_guard_inputs_v18(sidecars: Path, output: Path) -> dict:
    """Stage portable safe metadata, never a historical acceptance decision."""
    from .audit import digest
    from .prep_v16b import file_sha, seal

    raw = Path(sidecars).read_bytes()
    body = json.loads(raw)
    guard = body["guard"]
    by_id = body["items"]
    if isinstance(by_id, list):
        by_id = {item["problem_id"]: item for item in by_id}
    ordered = guard.get("items")
    ids = [item["problem_id"] for item in ordered] if isinstance(ordered, list) else guard["training_order_ids"]
    if isinstance(ordered, list):
        by_id = {item["problem_id"]: item for item in ordered}
    if len(ids) != 3000 or len(set(ids)) != 3000 or not set(ids) <= set(by_id):
        raise ValueError("v18 guard: exact 3,000 ordered parents required")
    fields = ("source", "layer", "capability", "capability_type", "f0_prior")
    items = [{"problem_id": pid, **{k: by_id[pid][k] for k in fields if k in by_id[pid]}} for pid in ids]
    if any("source" not in item for item in items):
        raise ValueError("v18 guard: source metadata required")
    manifest = seal(output / "guard-manifest-v18.json", {"version": "v18-guard-manifest-v1",
        "order_sha256": digest(ids), "sidecars_sha256": hashlib.sha256(raw).hexdigest(), "items": items})
    extra = guard.get("extra_priors", {})
    if "sources" in extra:
        extra = extra["sources"]
    # The legacy bootstrap's version identifies its input schema; v18 receipt identifies this new execution.
    seal(output / "guard-extra-priors-v18.json", {"version": "v17-guard-extra-priors-v1", "sources": extra})
    return {"version": "v18-guard-inputs-v1", "manifest": str(output / "guard-manifest-v18.json"),
            "extra_priors": str(output / "guard-extra-priors-v18.json"),
            "manifest_sha256": file_sha(output / "guard-manifest-v18.json"),
            "extra_priors_sha256": file_sha(output / "guard-extra-priors-v18.json"),
            "order_sha256": manifest["order_sha256"]}


def falsealarm_receipt_v18(manifest: Path, *, trials: int = 2000, seed: int = 3407,
                           extra_priors: Path | None = None) -> dict:
    receipt = falsealarm_receipt_v17(manifest, trials=trials, seed=seed,
                                    fill_missing=True, extra_priors=extra_priors)
    return receipt | {"version": "v18-guard-falsealarm-v1", "inherited_acceptance": False}


def receipt_passes_v18(receipt: dict, manifest_raw: bytes) -> bool:
    if receipt.get("version") != "v18-guard-falsealarm-v1" or receipt.get("inherited_acceptance") is not False:
        return False
    return receipt_passes_v17(receipt | {"version": "v17-guard-falsealarm-v1"}, manifest_raw)



def falsealarm_receipt_v18b(manifest: Path, *, trials: int = 2000, seed: int = 3407,
                            extra_priors: Path | None = None) -> dict:
    legacy = falsealarm_receipt_v17(manifest, trials=trials, seed=seed,
                                    fill_missing=True, extra_priors=extra_priors)
    return legacy | {"version": "v18b-guard-falsealarm-v1", "inherited_acceptance": False,
        "legacy_unmeasured_rules": legacy.get("unmeasured_rules", []),
        "unmeasured_rules": sorted(set(legacy.get("unmeasured_rules", [])) |
            {"COLLAPSE_LEN", "MONITOR_DEGRADED", "B2_SHORT_DRIFT", "N_SUPPLY"}),
        "external_b2_actions": {"MONITOR_DEGRADED": "STOP_REQUESTED", "B2_SHORT_DRIFT": "WARNING"},
        "new_rule_calibration": "UNMEASURED_SINGLE_F0; source-paired independent F0 repeats unavailable",
        "scope": "LEGACY_TRAINING_RULES_ONLY; new monitor and supply thresholds are not calibrated"}


def receipt_passes_v18b(receipt: dict, manifest_raw: bytes) -> bool:
    if (receipt.get("version") != "v18b-guard-falsealarm-v1" or receipt.get("inherited_acceptance") is not False
            or receipt.get("new_rule_calibration") != "UNMEASURED_SINGLE_F0; source-paired independent F0 repeats unavailable"
            or not {"COLLAPSE_LEN", "B2_SHORT_DRIFT", "N_SUPPLY"} <= set(receipt.get("unmeasured_rules", []))):
        return False
    return receipt_passes_v17(receipt | {"version": "v17-guard-falsealarm-v1",
        "unmeasured_rules": receipt.get("legacy_unmeasured_rules", [])}, manifest_raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--materialize-data-root", type=Path)
    parser.add_argument("--risk-acceptance", help="Explicit operator decision; measured FAIL is retained")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--version", choices=("error-note-v16b", "error-note-v17", "error-note-v18", "error-note-v18b"), default="error-note-v16b")
    parser.add_argument("--check-receipt", type=Path)
    parser.add_argument("--fill-missing", action="store_true", help="v17 only: stratified fill of missing F0 priors")
    parser.add_argument("--extra-priors", type=Path, help="v17 fill only: same-source F0 responses on other items")
    args = parser.parse_args(argv)
    if args.materialize_data_root is not None:
        if args.version not in {"error-note-v18", "error-note-v18b"} or args.output is None or args.manifest is not None:
            parser.error("materialization requires explicit v18, --output and no --manifest")
        from .v18_data import validate
        if args.version == "error-note-v18b":
            from .v18b_data import validate
        validate(args.materialize_data_root)
        inputs = materialize_guard_inputs_v18(args.materialize_data_root / "guard-metadata-v18.json", args.output.parent)
        args.manifest, args.extra_priors = Path(inputs["manifest"]), Path(inputs["extra_priors"])
    if args.manifest is None:
        parser.error("--manifest or --materialize-data-root is required")
    if args.risk_acceptance is not None and args.version not in {"error-note-v18", "error-note-v18b"}:
        parser.error("risk acceptance applies to a fresh v18 receipt only")
    if args.check_receipt is not None:
        receipt = json.loads(args.check_receipt.read_text())
        passed = (receipt_passes_v18b(receipt, args.manifest.read_bytes()) if args.version == "error-note-v18b" else
                  receipt_passes_v18(receipt, args.manifest.read_bytes()) if args.version == "error-note-v18" else
                  args.version == "error-note-v17" and receipt_passes_v17(receipt, args.manifest.read_bytes()))
        return 0 if passed else 3
    if args.output is None:
        parser.error("--output is required when generating a receipt")
    if args.version == "error-note-v18b":
        receipt = falsealarm_receipt_v18b(args.manifest, trials=args.trials, extra_priors=args.extra_priors)
    elif args.version == "error-note-v18":
        receipt = falsealarm_receipt_v18(args.manifest, trials=args.trials, extra_priors=args.extra_priors)
    elif args.version == "error-note-v17":
        receipt = falsealarm_receipt_v17(args.manifest, trials=args.trials, fill_missing=args.fill_missing,
                                         extra_priors=args.extra_priors)
    else:
        receipt = falsealarm_receipt(args.manifest, trials=args.trials)
    if args.risk_acceptance is not None:
        from datetime import datetime, timezone
        if not args.risk_acceptance.strip():
            raise ValueError("v18 guard: nonempty operator decision required")
        if receipt.get("measured_status") == "FAIL":
            upper = receipt.get("overall_upper95")
            if not isinstance(upper, (int, float)) or upper > RISK_ACCEPTED_UPPER95_MAX_V17:
                raise ValueError("v18 guard: operator acceptance requires measured upper95 <= 5%")
            receipt.update(status="RISK_ACCEPTED", risk_acceptance={"by": "operator",
                "at_utc": datetime.now(timezone.utc).isoformat(), "reason": args.risk_acceptance.strip(),
                "consequence": "Fixed thresholds remain active; a guard stop ends training at its save point."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if args.version == "error-note-v18b":
        return 0 if receipt_passes_v18b(receipt, args.manifest.read_bytes()) else 3
    if args.version == "error-note-v18":
        return 0 if receipt_passes_v18(receipt, args.manifest.read_bytes()) else 3
    return 0 if receipt["status"] in ("PASS", "PASS_SCOPED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
