"""Generation workload of the production v18b retry. Timings never control admission."""
from __future__ import annotations

import math

from .protocol import MAX_TOKENS_V16B


def generation_plan(stage: str, params: dict) -> dict:
    limits = MAX_TOKENS_V16B
    rows: dict[str, tuple[int, int, int]]
    if stage == "v18b-train":
        fresh = not params["execution_settings"]["reuse_f0_raw"]
        rows = {
            # Values are calls, maximum requests per call, and per-request output cap.
            "prep_solutions": (1, 512 if fresh else 0, limits["prep"]),
            "prep_observations": (1, 512, limits["obs"]),
            "f0_effect": (1, 128 if fresh else 0, limits["monitor"]),
            "accuracy_monitor": (4, 200, limits["monitor"]),  # F0, post-prep, signal gate, final
            "short_monitor": (5, 264, 64),  # F0, post-prep, blocks 8 and 16, final
            "first_solutions": (1, 3000, limits["t1"]),
            "annotated_observations": (1, 3000, limits["obs"]),
            "error_notes": (1, 3000, limits["note"]),
            "resolutions": (1, 3000, limits["resolve"]),
            "final_effect": (1, 128, limits["monitor"]),
        }
    elif stage == "v18b-eval":
        count = 450 if params["execution_settings"]["evaluation_strategy"] == "two_pod" else 900
        rows = {"mmmu": (1, count, 32768)}
    else:
        raise ValueError("generation workload is defined for v18b production plans")
    measurements = params.get("generation_timing", {})
    if not isinstance(measurements, dict) or set(measurements) - set(rows) - {"reference_only"}:
        raise ValueError("generation timing keys must match generation stages or reference_only")
    rates = {}
    for name, sample in measurements.items():
        if (not isinstance(sample, dict) or set(sample) != {"output_tokens", "elapsed_seconds", "source"}
                or type(sample["output_tokens"]) is not int or sample["output_tokens"] <= 0
                or type(sample["elapsed_seconds"]) not in (int, float)
                or not math.isfinite(sample["elapsed_seconds"]) or sample["elapsed_seconds"] <= 0
                or not isinstance(sample["source"], str) or not sample["source"].strip()):
            raise ValueError(f"{name}: timing requires measured tokens, wall seconds and source")
        rates[name] = sample["output_tokens"] / sample["elapsed_seconds"]
    reference_rate = rates.get("reference_only")
    result = {}
    for name, (calls, count, cap) in rows.items():
        sample = measurements.get(name)
        rate = rates.get(name)
        per_call = count * cap
        result[name] = {"max_calls": calls, "max_requests_per_call": count, "output_cap": cap,
                        "max_output_tokens_per_call": per_call, "max_output_tokens": calls * per_call,
                        "timing_sample": sample, "observed_output_tokens_per_second": rate,
                        "reference_only_cap_seconds": (calls * per_call / reference_rate if reference_rate else None),
                        "cap_scenario_seconds": 0 if not count else (calls * per_call / rate if rate else None)}
    return {"version": "v18b-retry-generation-workload-v1", "stages": result, "advisory_only": True,
            "reference_only": measurements.get("reference_only"),
            "basis": "Upper token counts; seconds assume the supplied workload-specific wall throughput. "
                     "Ineligible, reused and stopped requests can reduce actual generation. "
                     "Engine initialization, validation probes, SFT/RL updates, reload/export, supplementary "
                     "classification, transfer and cleanup require their own wall measurements."}
