"""G8 live two-turn token-boundary measurement, not a correctness experiment.

The caller supplies fixed probe feedback, not an answer key. These unscored
requests test transport and the production action selector, never train a policy.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import torch

from fine_tuning.modules import tensor_sha256
from .engines import RolloutRequest, _completion, _request
from .rollout import continuation_ids, seeds
from .schema import Sample, to_json
from .updates import action_mask


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("G8: " + message)


def _pixels(prepared) -> dict:
    return {"image_key": prepared.image_key, "image_spans": [list(s) for s in prepared.image_spans],
            "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                              "sha256": tensor_sha256(value)} for key, value in prepared.pixels.items()}}


def _sample(engine, request: RolloutRequest, version: int, turn: int,
            parent: Sample | None, record: Callable[[dict], None]) -> tuple[Sample, dict]:
    prepared = engine.prepare(request)
    expected, pixels = list(prepared.token_ids), _pixels(prepared)
    try:
        returned = engine.sample([request])
    except BaseException as exc:
        record({"kind": "G8_SAMPLER_ERROR", "request_id": request.request_id,
                "policy_version": version, "error_type": type(exc).__name__,
                "backend_invalid_payload_preserved": False})
        raise
    record({"kind": "G8_ENGINE_RETURN", "request_id": request.request_id,
            "policy_version": version, "returns": [asdict(value) for value in returned]})
    _require(engine.policy_version == version and engine._synchronized is True,
             "policy version or synchronization changed")
    _require(len(returned) == 1 and returned[0].request_id == request.request_id
             and returned[0].prompt_token_ids == expected and len(returned[0].completions) == 1,
             "sampler changed the actual prompt or request coverage")
    _require(request.prompt_token_ids is None or expected == request.prompt_token_ids,
             "literal retry prompt was changed during preparation")
    c = returned[0].completions[0]
    _completion(c.token_ids, c.logprobs, c.finish_reason, c.text)
    _require(len(c.token_ids) <= request.max_new_tokens, "completion exceeds reserved cap")
    sample = Sample(request.request_id, parent.problem_id if parent else request.request_id,
        request.request_id, "T1" if turn == 1 else "T2", turn,
        parent.sample_id if parent else None, list(expected), list(c.token_ids), list(c.logprobs),
        c.finish_reason, c.text, request.seeds[0], version)
    to_json(sample)
    # Production updates operate on completion-only logprobs, one sample per
    # microbatch. The full-sequence view makes that boundary auditable; no pads.
    selected = action_mask(sample, "ALG-NB", engine.tokenizer)
    _require(selected.dtype == torch.bool and tuple(selected.shape) == (1, len(c.token_ids))
             and bool(selected.all()), "production selector did not retain every generated action")
    inputs = engine.model_inputs(request, sample.completion_token_ids, torch.device("cpu"))
    ids = sample.prompt_token_ids + sample.completion_token_ids
    _require(inputs["input_ids"].tolist() == [ids]
             and inputs["attention_mask"].tolist() == [[1] * len(ids)],
             "training preparation differs from actual prompt plus actions")
    _require(set(inputs) == {"input_ids", "attention_mask", *prepared.pixels}
             and {key: {"shape": list(inputs[key].shape), "dtype": str(inputs[key].dtype),
                         "sha256": tensor_sha256(inputs[key])} for key in prepared.pixels} == pixels["tensors"],
             "training preparation changed image tensors")
    _require(_pixels(engine.prepare(request)) == pixels, "image inputs changed during probe")
    full_mask = [False] * len(expected) + selected[0].tolist()
    row = {"sample_id": sample.sample_id, "turn": turn, "seed": sample.seed,
           "prompt_token_ids": expected, "completion_token_ids": sample.completion_token_ids,
           "finish_reason": sample.finish_reason, "input": pixels,
           "full_sequence_action_mask": full_mask, "padding_tokens": 0,
           "selected_tokens": sum(full_mask)}
    record({"kind": "G8_ACTION_BOUNDARY", **row})
    return sample, row


def _measure_tokens(engine, requests: list[RolloutRequest], *, feedback: str,
                    retry_max_new_tokens: int, record: Callable[[dict], None]) -> dict:
    """Private seam also permits real tiny-HF CPU tests; never native evidence."""
    _require(callable(record) and isinstance(feedback, str) and bool(feedback.strip())
             and type(retry_max_new_tokens) is int and retry_max_new_tokens > 0,
             "journal, explicit feedback and positive retry reservation required")
    _require(len(requests) == 2 and {len(r.images) for r in requests} == {1, 2}
             and len({r.request_id for r in requests}) == 2
             and all(r.n == 1 and r.messages is not None and r.prompt_token_ids is None for r in requests),
             "distinct initial single/two-image probes with n=1 required")
    for request in requests:
        _request(request)
    version = engine.policy_version
    _require(type(version) is int and version >= 0 and engine._synchronized is True,
             "stable synchronized policy required")
    tokenizer = engine.tokenizer
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    _require(type(end) is int and end >= 0 and end != getattr(tokenizer, "unk_token_id", None),
             "actual Qwen conversation boundary required")
    # Independently compose the template, then compare the production helper.
    suffix = tokenizer.encode("\n<|im_start|>user\n" + feedback
                              + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    _require(bool(suffix) and all(type(t) is int and t >= 0 for t in suffix), "valid feedback IDs required")
    rows = []
    for request in requests:
        first, first_row = _sample(engine, request, version, 1, None, record)
        generated_end = first.completion_token_ids[-1] == end
        expected = first.prompt_token_ids + first.completion_token_ids + ([] if generated_end else [end]) + suffix
        actual = continuation_ids(first, feedback, tokenizer)
        _require(actual == expected, "production continuation differs from intended token template")
        retry_id = request.request_id + ":G8-retry"
        _require(retry_id not in {r.request_id for r in requests}, "retry request identity collision")
        retry = RolloutRequest(retry_id, actual, None, request.images, 1, retry_max_new_tokens,
                               seeds(request.request_id, version, 2, "G8", 1))
        record({"kind": "G8_RETRY_REQUEST", "request_id": retry_id, "parent_sample_id": first.sample_id,
                "prompt_token_ids": actual, "feedback": feedback, "feedback_token_ids": suffix,
                "generated_im_end": generated_end, "inserted_im_end": not generated_end,
                "reserved_output_tokens": retry_max_new_tokens})
        # An overflowing retry is a failed measurement, not a truncated prompt.
        second, second_row = _sample(engine, retry, version, 2, first, record)
        _require(second.prompt_token_ids == expected and second_row["input"] == first_row["input"],
                 "retry changed original history, image order or pixels")
        _require(not any(second_row["full_sequence_action_mask"][:len(expected)]),
                 "history or feedback entered current action mask")
        rows.append({"request_id": request.request_id, "first": first_row, "second": second_row,
                     "generated_im_end": generated_end, "inserted_im_end": not generated_end})
    return {"tokens_exact": True, "generated_only_mask": True, "eos_checked": True,
            "policy_version": version, "probes": rows, "actual_calls": 4,
            "conversation_boundary_token_id": end,
            "closure_branches_observed": sorted({"GENERATED" if r["generated_im_end"] else "INSERTED" for r in rows}),
            "native_execution": False, "training_allowed": False, "gpu_gates_certified": False,
            "weights_updated": False, "answers_scored": False,
            "mask_scope": "ALG-NB_COMPLETION_ONLY_UNPADDED_MICROBATCH",
            "raw_return_scope": "ENGINE_VALIDATED_RETURNS_ONLY"}


def measure_tokens(engine, requests: list[RolloutRequest], *, feedback: str,
                   retry_max_new_tokens: int, record: Callable[[dict], None]) -> dict:
    """Run four actual unscored requests on an already-owned native engine."""
    from .gate_weights import _pinned_config, require_idle_engine

    runtime = require_idle_engine(engine, require_push=False)
    _pinned_config(engine.model_config)
    result = _measure_tokens(engine, requests, feedback=feedback,
                             retry_max_new_tokens=retry_max_new_tokens, record=record)
    _require(require_idle_engine(engine, require_push=False) == runtime, "native runtime changed")
    result.update(runtime=runtime, native_execution=True,
        limitations=["Fixed-feedback token transport probe; not correctness feedback or learned self-correction",
                     "Observes the actual EOS branch; authored tests separately cover both closure branches",
                     "Unpadded microbatch selector; note-only and padded-loss checks are separate tests",
                     "No complete GPU/data approval or performance claim"])
    return result
