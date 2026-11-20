"""Token-faithful rollout engines. vLLM operation still requires actual GPU gates.

Images in requests are original decoded PIL images, not pre-resized derivatives.
Both engines share the pinned CPU preprocessing; retry token IDs are never decoded
and retokenized. Prepared pixels are also the trainer's input, not reconstructed
from a model's description. The HF implementation is for bounded smoke checks.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0

    def validate(self) -> None:
        values = (self.temperature, self.top_p, self.presence_penalty, self.repetition_penalty)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or type(self.top_k) is not int
                or (self.temperature, self.top_p, self.top_k, self.presence_penalty,
                    self.repetition_penalty) not in ((1., 1., 0, 0., 1.), (.7, .8, 20, 1.5, 1.))):
            raise ValueError("engine: only the pinned T1 and official evaluation samplers are supported")

    @property
    def name(self) -> str:
        self.validate()
        return "temperature_1" if self.temperature == 1 else "qwen_official"


QWEN_OFFICIAL = SamplingSettings(.7, .8, 20, 1.5, 1.)


@dataclass(frozen=True)
class RolloutRequest:
    request_id: str
    prompt_token_ids: list[int] | None
    messages: list[dict] | None
    images: list
    n: int
    max_new_tokens: int
    seeds: list[int]


@dataclass(frozen=True)
class Completion:
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    text: str


@dataclass(frozen=True)
class RolloutResult:
    request_id: str
    prompt_token_ids: list[int]
    completions: list[Completion]


ROW_SEEDED_BATCH_SAMPLER = "hf-row-seeded-batch-v1"


@dataclass(frozen=True)
class BatchedRolloutResult(RolloutResult):
    """Opt-in HF output: the n seeds of one request in chunks of `batch_rows` rows per generate.

    For more than one row it is reproducible only for the same seeds, chunking and kernels, and
    is not token-equal to the serial sampler on GPU. A distinct type, so it never compares equal
    to a serial `RolloutResult`, and `asdict()` carries both labels into receipts.
    """
    batch_rows: int
    sampler_version: str = ROW_SEEDED_BATCH_SAMPLER


@dataclass
class Prepared:
    token_ids: list[int]
    pixels: dict
    images: list
    image_key: str
    image_spans: tuple[tuple[int, int], ...] = ()


def _request(request: RolloutRequest) -> None:
    if (not request.request_id or (request.prompt_token_ids is None) == (request.messages is None)
            or type(request.n) is not int or request.n < 1 or type(request.max_new_tokens) is not int
            or request.max_new_tokens < 1 or len(request.seeds) != request.n
            or any(type(s) is not int or not 0 <= s < 2**31 for s in request.seeds)
            or len(set(request.seeds)) != request.n or len(request.images) > 2):
        raise ValueError("engine: invalid request or ambiguous prompt representation")
    if request.prompt_token_ids is not None and (not request.prompt_token_ids
            or any(type(t) is not int or t < 0 for t in request.prompt_token_ids)):
        raise ValueError("engine: invalid literal token prompt")


def _completion(ids: list[int], logprobs: list[float], finish: str, text: str) -> Completion:
    if (not ids or len(ids) != len(logprobs) or finish not in ("stop", "length")
            or any(type(t) is not int or t < 0 for t in ids)
            or any(not math.isfinite(p) or p > 1e-5 for p in logprobs)):
        raise RuntimeError("engine: invalid actual completion/log probability contract")
    return Completion(ids, logprobs, finish, text)


class InputPreparation:
    # HF<->HF probe threshold cap (fixed). Only VLLMRolloutEngine overrides it (cross-engine calibration).
    sync_threshold_cap = 0.02

    def __init__(self, processor: Any, model_config: Any, image_cfg: dict, context_limit: int):
        self.processor, self.tokenizer, self.model_config = processor, processor.tokenizer, model_config
        self.image_cfg, self.context_limit = dict(image_cfg), context_limit
        self.policy_version = 0
        self._initialization_open = True
        self._synchronized = True
        self._push_complete = True
        # At most one problem's CPU image tensors; this is preprocessing, not an encoder cache.
        self._image_key: str | None = None
        self._pixels: dict = {}
        self._processed_images: list = []

    def initialize_weights(self, named_weights: Iterable[tuple[str, torch.Tensor]], policy_version: int) -> None:
        """Restore a fresh engine once; actual single/multi-image probes must follow."""
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("engine: initial policy version must be a nonnegative integer")
        if not self._initialization_open:
            raise RuntimeError("engine: initialization requires an unused fresh engine and is allowed only once")
        self._initialization_open = False
        self._synchronized = self._push_complete = False
        engine: Any = self
        engine._load_weights(named_weights, policy_version)
        self._push_complete = True

    def validate_sync(self, probes: list[tuple[RolloutRequest, list[int], list[float]]], threshold: float,
                      image_counts: tuple[int, ...] = (1, 2)) -> list[dict]:
        """`image_counts` (1,) only for a preparation that binds single-image-only probe coverage.

        The threshold may not exceed the engine's `sync_threshold_cap` (0.02; a non-subclass caller gets 0.02)."""
        self._initialization_open = False
        self._synchronized = False
        if not self._push_complete:
            raise RuntimeError("engine: a failed/partial push cannot be certified by probes")
        if (image_counts not in ((1,), (1, 2)) or sorted(len(p[0].images) for p in probes) != list(image_counts)
                or type(threshold) not in (int, float)
                or not 0 < threshold <= getattr(self, "sync_threshold_cap", 0.02)):
            raise ValueError("engine: fixed single/multi-image HF probes and strict threshold required")
        reports = []
        engine: Any = self
        for request, ids, reference in probes:
            actual = engine.probe_logprobs(request, ids)
            if len(actual) != len(reference) or not reference:
                raise RuntimeError("engine: incomplete synchronization probe")
            delta = [abs(a-b) for a, b in zip(actual, reference, strict=True)]
            if any(not math.isfinite(x) for x in delta) or sum(delta)/len(delta) > threshold:
                raise RuntimeError("engine: synchronized sampler/HF probabilities disagree")
            reports.append({"request_id": request.request_id, "mean_abs_delta": sum(delta)/len(delta),
                            "max_abs_delta": max(delta), "policy_version": self.policy_version})
        self._synchronized = True
        return reports

    def prepare(self, request: RolloutRequest) -> Prepared:
        from .input_preparation import prepare_protocol_inputs

        _request(request)
        hashes = []
        for image in request.images:
            if image.mode != "RGB":
                raise ValueError("engine: decoded RGB images required")
            hashes.append((image.size, hashlib.sha256(image.tobytes()).hexdigest()))
        key = hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()
        if request.messages is not None:
            if request.images:
                item, processed = prepare_protocol_inputs(self.processor, self.model_config,
                    request.messages, request.images, self.image_cfg)
                ids = item["prompt_token_ids"]
                self._image_key, self._processed_images = key, processed
                self._pixels = {k: item["inputs"][k] for k in ("pixel_values", "image_grid_thw")}
            else:
                text = self.processor.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
                ids = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            ids = list(request.prompt_token_ids or [])
            if request.images and self._image_key != key:
                # Build image tensors through the same processor path, but never reuse
                # its freshly tokenized text as the literal multi-turn request.
                messages = [{"role": "user", "content": [
                    *({"type": "image"} for _ in request.images), {"type": "text", "text": ""}]}]
                item, processed = prepare_protocol_inputs(self.processor, self.model_config,
                    messages, request.images, self.image_cfg)
                self._image_key, self._processed_images = key, processed
                self._pixels = {k: item["inputs"][k] for k in ("pixel_values", "image_grid_thw")}
        if not ids or len(ids) + request.max_new_tokens > self.context_limit:
            raise ValueError("CONTEXT_BLOCKED: prompt plus reserved completion exceeds context")
        pixels = self._pixels if request.images else {}
        expected = (int(pixels["image_grid_thw"].prod(-1).sum())
                    // self.processor.image_processor.merge_size**2) if pixels else 0
        if ids.count(self.model_config.image_token_id) != expected:
            raise ValueError("engine: literal prompt/image grid mismatch")
        spans, offset = [], 0
        if pixels:
            for grid in pixels["image_grid_thw"]:
                count = int(grid.prod()) // self.processor.image_processor.merge_size**2
                try:
                    start = ids.index(self.model_config.vision_start_token_id, offset) + 1
                except ValueError as exc:
                    raise ValueError("engine: missing registered image boundary") from exc
                end = start + count
                if (count < 1 or ids[start:end] != [self.model_config.image_token_id] * count
                        or end >= len(ids) or ids[end] != self.model_config.vision_end_token_id):
                    raise ValueError("engine: registered image span differs from processor grid")
                spans.append((start, end))
                offset = end + 1
        return Prepared(ids, pixels, self._processed_images if request.images else [], key, tuple(spans))

    def model_inputs(self, request: RolloutRequest, completion_ids: list[int], device: torch.device) -> dict:
        prepared = self.prepare(request)
        if len(prepared.token_ids) + len(completion_ids) > self.context_limit:
            raise ValueError("engine: actual sequence exceeds context")
        ids = torch.tensor([prepared.token_ids + completion_ids], dtype=torch.long, device=device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                **{k: v.to(device) for k, v in prepared.pixels.items()}}


class _SelectedTokenLogprobs:
    """Keep one score vector; the next prefix reveals its actual sampled token."""

    def __init__(self, prompt_length: int):
        self.prompt_length = prompt_length
        self.scores: torch.Tensor | None = None
        self.input_ids: torch.Tensor | None = None
        self.logprobs: list[torch.Tensor] = []

    def _append(self, token: torch.Tensor) -> None:
        if self.scores is None:
            raise RuntimeError("engine: missing streamed generation scores")
        # gather owns a scalar allocation; indexing would retain the vocabulary tensor.
        self.logprobs.append(self.scores[0].float().log_softmax(-1).gather(0, token))

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        expected = self.prompt_length + len(self.logprobs) + int(self.scores is not None)
        if (input_ids.ndim != 2 or input_ids.shape != (1, expected)
                or scores.ndim != 2 or scores.shape[0] != 1):
            raise RuntimeError("engine: streamed generation prefix/score mismatch")
        if self.scores is not None:
            self._append(input_ids[0, -1:])
        self.scores, self.input_ids = scores, input_ids
        return scores

    def finish(self, sequences: torch.Tensor) -> list[float]:
        if (self.input_ids is None or sequences.ndim != 2
                or sequences.shape != (1, self.prompt_length + len(self.logprobs) + 1)
                or not torch.equal(self.input_ids, sequences[:, :-1])):
            raise RuntimeError("engine: missing or mismatched streamed generation scores")
        self._append(sequences[0, -1:])
        self.scores = self.input_ids = None
        return torch.cat(self.logprobs).tolist()

    def clear(self) -> None:
        self.scores = self.input_ids = None
        self.logprobs.clear()


def _batch_rows(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("engine: batch_rows must be None (serial sampler) or a positive int")
    return value


class _RowSeededSampler:
    """Temperature-one draw for every live row from that row's own seeded generator.

    Placed directly after LogitNormalization, so each row sees the serial path's scores: the draw
    is softmax -> multinomial on that (1, V) row and the recorded log probability is
    float().log_softmax().gather() of it, the same operations as HF sampling and
    `_SelectedTokenLogprobs`. The drawn token is forced (0 vs -inf) so greedy decoding emits it.
    A row stops at its first EOS id or at `cap` tokens; HF pads finished rows afterwards.
    """

    def __init__(self, seeds: list[int], prompt_length: int, cap: int, eos_ids: frozenset[int],
                 pad_id: int, device: torch.device):
        self.generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        self.prompt_length, self.cap, self.eos_ids, self.pad_id = prompt_length, cap, eos_ids, pad_id
        self.tokens: list[list[int]] = [[] for _ in seeds]
        self.logprobs: list[list[torch.Tensor]] = [[] for _ in seeds]
        self.done = [False] * len(seeds)
        self.steps = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        rows = len(self.generators)
        if (input_ids.ndim != 2 or input_ids.shape != (rows, self.prompt_length + self.steps)
                or scores.ndim != 2 or scores.shape[0] != rows):
            raise RuntimeError("engine: batched generation prefix/score mismatch")
        if self.steps:
            emitted = input_ids[:, -1].tolist()
            if any(len(row) == self.steps and row[-1] != emitted[i] for i, row in enumerate(self.tokens)):
                raise RuntimeError("engine: generate did not emit the row-seeded token")
        live = [i for i, done in enumerate(self.done) if not done]
        picks = []
        for i in live:
            row = scores[i:i + 1]
            token = torch.multinomial(torch.softmax(row, -1), 1, generator=self.generators[i])[0]
            # gather owns a scalar allocation; indexing would retain the vocabulary tensor.
            self.logprobs[i].append(row[0].float().log_softmax(-1).gather(0, token))
            picks.append(token)
        forced = torch.full_like(scores, float("-inf"))
        for i, token in zip(live, torch.cat(picks).tolist() if picks else [], strict=True):
            self.tokens[i].append(token)
            forced[i, token] = 0.
            self.done[i] = token in self.eos_ids or len(self.tokens[i]) >= self.cap
        for i in set(range(rows)) - set(live):
            forced[i, self.pad_id] = 0.  # HF overwrites finished rows with its pad id anyway.
        self.steps += 1
        return forced

    def rows_done(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return torch.tensor(self.done, dtype=torch.bool, device=input_ids.device)

    def finish(self, sequences: torch.Tensor, prompts: torch.Tensor) -> list[tuple[list[int], list[float]]]:
        rows = len(self.generators)
        if (not all(self.done) or sequences.ndim != 2
                or sequences.shape != (rows, self.prompt_length + self.steps)
                or not torch.equal(sequences[:, :self.prompt_length], prompts)
                or any(sequences[i, self.prompt_length:self.prompt_length + len(row)].tolist() != row
                       for i, row in enumerate(self.tokens))):
            raise RuntimeError("engine: batched sequences differ from the row-seeded draws")
        return [(list(row), torch.cat(lps).tolist()) for row, lps in zip(self.tokens, self.logprobs, strict=True)]


class HFRolloutEngine(InputPreparation):
    """Actual sequential HF generation with bounded vocabulary-score retention.

    `batch_rows=None` (default) is the serial one-seed-per-generate sampler. An int opts into
    `_RowSeededSampler`: the n seeds of one request in chunks of at most `batch_rows` rows,
    returned as `BatchedRolloutResult`.
    """
    supported_sampling = frozenset(("temperature_1",))
    batch_rows: int | None = None

    def __init__(self, model: Any, processor: Any, image_cfg: dict, max_model_len: int, *,
                 batch_rows: int | None = None):
        super().__init__(processor, model.config, image_cfg, max_model_len)
        self.model = model
        self._weight_shapes = _weight_inventory(model.config)
        self.batch_rows = _batch_rows(batch_rows)

    def sample(self, requests: list[RolloutRequest], *,
               on_completion: Callable[[str, int, int, Completion], None] | None = None) -> list[RolloutResult]:
        if on_completion is None:
            return self.sample_with_settings(requests, SamplingSettings())
        return self.sample_with_settings(requests, SamplingSettings(), on_completion=on_completion)

    def sample_with_settings(self, requests: list[RolloutRequest], sampling: SamplingSettings, *,
            on_completion: Callable[[str, int, int, Completion], None] | None = None) -> list[RolloutResult]:
        from transformers import GenerationConfig, LogitNormalization, LogitsProcessorList

        self._initialization_open = False
        if not self._synchronized:
            raise RuntimeError("engine: post-push logprob probes have not passed")
        sampling.validate()
        if sampling.name not in self.supported_sampling:
            raise NotImplementedError("engine: HF official evaluation needs a reviewed generated-only presence penalty adapter")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("engine: duplicate request ID")
        results: list[RolloutResult] = []
        device, was_training = next(self.model.parameters()).device, self.model.training
        self.model.eval()
        try:
            for request in requests:
                prepared = self.prepare(request)
                batch = self.model_inputs(request, [], device)
                if self.batch_rows is not None:
                    results.append(BatchedRolloutResult(request.request_id, prepared.token_ids,
                        self._sample_rows(request, prepared, batch, device, on_completion), self.batch_rows))
                    continue
                completions = []
                eos = self.model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                cfg = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                    repetition_penalty=1.0, max_new_tokens=request.max_new_tokens,
                    eos_token_id=eos, pad_token_id=self.model.generation_config.pad_token_id,
                    return_dict_in_generate=True, output_scores=False, use_cache=True,
                    renormalize_logits=False)
                for index, seed in enumerate(request.seeds):
                    devices = [device.index or 0] if device.type == "cuda" else []
                    collector = _SelectedTokenLogprobs(len(prepared.token_ids))
                    output = None
                    try:
                        with torch.random.fork_rng(devices=devices), torch.no_grad():
                            torch.manual_seed(seed)
                            # T1 has no warpers: this is HF's original final normalizer,
                            # followed only by an observer that returns the same scores.
                            output = self.model.generate(**batch, generation_config=cfg,
                                use_model_defaults=False,
                                logits_processor=LogitsProcessorList([LogitNormalization(), collector]))
                        # ModelOutput attribute assignment leaves the mapping's cache alive.
                        output["past_key_values"] = None
                        ids = output.sequences[0, len(prepared.token_ids):].tolist()
                        lps = collector.finish(output.sequences)
                    finally:
                        collector.clear()
                        if output is not None:
                            output["past_key_values"] = None
                        del output
                    finish = "stop" if ids and ids[-1] in eos_ids else "length"
                    completions.append(_completion(ids, lps, finish,
                        self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)))
                    if on_completion is not None:
                        completion = completions[-1]
                        on_completion(request.request_id, index, seed,
                            Completion(list(completion.token_ids), list(completion.logprobs),
                                       completion.finish_reason, completion.text))
                results.append(RolloutResult(request.request_id, prepared.token_ids, completions))
        finally:
            self.model.train(was_training)
        return results

    def _sample_rows(self, request: RolloutRequest, prepared: Prepared, batch: dict, device: torch.device,
                     on_completion: Callable[[str, int, int, Completion], None] | None) -> list[Completion]:
        """One generate per chunk of seeds over the same unpadded prompt; callbacks after each chunk."""
        from transformers import GenerationConfig, LogitNormalization, LogitsProcessorList, StoppingCriteriaList

        if self.batch_rows is None:
            raise RuntimeError("engine: row-seeded sampling requires batch_rows")
        eos = self.model.generation_config.eos_token_id
        eos_ids = frozenset(eos if isinstance(eos, list) else () if eos is None else (eos,))
        pad = self.model.generation_config.pad_token_id
        # Greedy over the forced row draws; warpers stay off, so no sampling knob reaches HF.
        cfg = GenerationConfig(do_sample=False, repetition_penalty=1.0, max_new_tokens=request.max_new_tokens,
            eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=True, output_scores=False,
            use_cache=True, renormalize_logits=False)
        devices = [device.index or 0] if device.type == "cuda" else []
        completions: list[Completion] = []
        for start in range(0, request.n, self.batch_rows):
            seeds = request.seeds[start:start + self.batch_rows]
            rows = {key: value.repeat(len(seeds), 1) for key, value in batch.items()}
            sampler = _RowSeededSampler(seeds, len(prepared.token_ids), request.max_new_tokens, eos_ids,
                                        0 if pad is None else pad, device)
            output = None
            try:
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    output = self.model.generate(**rows, generation_config=cfg, use_model_defaults=False,
                        logits_processor=LogitsProcessorList([LogitNormalization(), sampler]),
                        stopping_criteria=StoppingCriteriaList([sampler.rows_done]))
                output["past_key_values"] = None
                drawn = sampler.finish(output.sequences, rows["input_ids"])
            finally:
                if output is not None:
                    output["past_key_values"] = None
                del output
            for offset, (seed, (ids, lps)) in enumerate(zip(seeds, drawn, strict=True)):
                completions.append(_completion(ids, lps, "stop" if ids[-1] in eos_ids else "length",
                    self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)))
                if on_completion is not None:
                    completion = completions[-1]
                    on_completion(request.request_id, start + offset, seed,
                        Completion(list(completion.token_ids), list(completion.logprobs),
                                   completion.finish_reason, completion.text))
        return completions

    def probe_logprobs(self, request: RolloutRequest, completion_ids: list[int]) -> list[float]:
        self._initialization_open = False
        if not completion_ids:
            raise ValueError("engine: empty probe")
        device, was_training = next(self.model.parameters()).device, self.model.training
        batch = self.model_inputs(request, completion_ids, device)
        self.model.eval()
        try:
            with torch.no_grad():
                logits = self.model(**batch, use_cache=False).logits[:, -len(completion_ids)-1:-1].float()
                actions = batch["input_ids"][:, -len(completion_ids):]
                return logits.log_softmax(-1).gather(-1, actions.unsqueeze(-1))[0, :, 0].tolist()
        finally:
            self.model.train(was_training)

    def push_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        if type(version) is not int or version != self.policy_version + 1 or not self._synchronized:
            raise ValueError("engine: weight versions must advance by one")
        self._initialization_open = False
        self._load_weights(named, version)

    def _load_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        parameters = dict(self.model.named_parameters())
        if {n: tuple(p.shape) for n, p in parameters.items()} != self._weight_shapes:
            raise ValueError("engine: model identity differs from the configured architecture")
        weights = {}
        for name, value in named:
            if (name in weights or name not in parameters or not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != self._weight_shapes[name] or value.dtype != parameters[name].dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("engine: duplicate/unexpected/nonfinite or wrong shape/dtype weight")
            weights[name] = value
        if weights.keys() != parameters.keys():
            raise ValueError("engine: complete named parameters required")
        # HF calls create fresh generation caches; this engine retains no KV/encoder state.
        with torch.no_grad():
            for name, value in weights.items():
                parameters[name].copy_(value)
        self.policy_version = version


def vllm_kwargs(model_path: Path, image_cfg: dict, kv_cache_memory_bytes: int,
                 max_num_seqs: int, max_model_len: int, *, enable_prefix_caching: bool = False,
                 gpu_memory_utilization: float = 0.7) -> dict:
    if (any(type(v) is not int or v <= 0 for v in (kv_cache_memory_bytes, max_num_seqs, max_model_len))
            or image_cfg.get("preprocessing") != "qwen_vl_utils_0_0_14"
            or type(enable_prefix_caching) is not bool
            or type(gpu_memory_utilization) is not float or not 0 < gpu_memory_utilization <= 1):
        raise ValueError("engine: explicit positive budgets and pinned image preprocessing required")
    return dict(model=str(model_path), tokenizer=str(model_path), dtype="bfloat16", quantization=None,
        # v0.11 startup checks this fraction before honoring the explicit KV bytes.
        # Leave room for the colocated HF policy; this does not shrink the 10GiB KV.
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False, enable_prefix_caching=enable_prefix_caching, mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 2, "video": 0}, generation_config="vllm", tensor_parallel_size=1,
        mm_processor_kwargs={"size": {"shortest_edge": image_cfg["min_pixels"],
            "longest_edge": image_cfg["max_pixels"]}, "do_resize": False},
        kv_cache_memory_bytes=kv_cache_memory_bytes, max_num_seqs=max_num_seqs, max_model_len=max_model_len)


def _weight_inventory(config: Any) -> dict[str, tuple[int, ...]]:
    """The real HF architecture on meta, not a second allocation of the model weights."""
    from transformers import Qwen3VLForConditionalGeneration

    if config.model_type != "qwen3_vl":
        raise ValueError("engine: exact Qwen3-VL architecture required")
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    return {name: tuple(p.shape) for name, p in model.named_parameters()}


def _clear_encoder_cache(worker: Any) -> bool:
    # vLLM 0.11 UniProcExecutor.collective_rpc runs this synchronously; exceptions propagate.
    cache = worker.model_runner.encoder_cache
    if not isinstance(cache, dict):
        raise RuntimeError("engine: unsupported worker encoder cache")
    cache.clear()
    return not cache


class VLLMRolloutEngine(InputPreparation):
    supported_sampling = frozenset(("temperature_1", "qwen_official"))
    # User decision 2026-10-31 "vLLM 전용 기준 0.04로 보정": vLLM-sampler<->HF-policy probes only (the exact
    # value is bound by scale.VLLM_SYNC_THRESHOLD). A40 vLLM 0.11.0 vs HF BF16, identical F0 weights: max measured
    # mean |delta| 0.0228 (ScienceQA 001890, 1,120 image tokens) -- a cross-engine numeric gap, not a weight error.
    sync_threshold_cap = 0.04

    def __init__(self, model_path: Path, image_cfg: dict, kv_cache_memory_bytes: int,
                 max_num_seqs: int, max_model_len: int, *, enable_prefix_caching: bool = False,
                 gpu_memory_utilization: float = 0.7):
        if version("vllm") != "0.11.0":
            raise RuntimeError("engine: vllm==0.11.0 required")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from transformers import AutoConfig, AutoProcessor
        LLM = import_module("vllm").LLM

        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
            min_pixels=image_cfg["min_pixels"], max_pixels=image_cfg["max_pixels"])
        processor.image_processor.size = {"shortest_edge": image_cfg["min_pixels"],
                                          "longest_edge": image_cfg["max_pixels"]}
        super().__init__(processor, AutoConfig.from_pretrained(str(model_path), local_files_only=True),
                         image_cfg, max_model_len)
        self._weight_shapes = _weight_inventory(self.model_config)
        self.model_dir = Path(model_path).resolve()
        self.llm = LLM(**vllm_kwargs(model_path, image_cfg, kv_cache_memory_bytes, max_num_seqs, max_model_len,
                                     enable_prefix_caching=enable_prefix_caching,
                                     gpu_memory_utilization=gpu_memory_utilization))
        self._call = 0
        self._synchronized = True
        self._push_complete = True
        self._last_loaded_version: int | None = None

    def _prompt(self, prepared: Prepared, ids: list[int]) -> dict:
        self._call += 1
        if ids[:len(prepared.token_ids)] != prepared.token_ids:
            raise ValueError("engine: transport must preserve the actual prompt prefix")
        # vLLM 0.11 re-expands each TokenPrompt placeholder even if HF already expanded it.
        # Collapse only registered image spans; prior generated IDs are never re-tokenized.
        transport, offset = [], 0
        if len(prepared.image_spans) != len(prepared.images):
            raise ValueError("engine: registered image spans required for token transport")
        for start, end in prepared.image_spans:
            if not offset <= start < end <= len(prepared.token_ids):
                raise ValueError("engine: invalid registered image span")
            transport.extend(ids[offset:start])
            transport.append(self.model_config.image_token_id)
            offset = end
        transport.extend(ids[offset:])
        prompt: dict = {"prompt_token_ids": transport}
        if prepared.images:
            prompt.update(multi_modal_data={"image": prepared.images},
                multi_modal_uuids={"image": [f"v{self.policy_version}:{self._call}:{prepared.image_key}:{i}"
                                            for i in range(len(prepared.images))]})
        return prompt

    def sample(self, requests: list[RolloutRequest], *,
               on_completion: Callable[[str, int, int, Completion], None] | None = None) -> list[RolloutResult]:
        if on_completion is None:
            return self.sample_with_settings(requests, SamplingSettings())
        return self.sample_with_settings(requests, SamplingSettings(), on_completion=on_completion)

    def sample_with_settings(self, requests: list[RolloutRequest], sampling: SamplingSettings, *,
            on_completion: Callable[[str, int, int, Completion], None] | None = None) -> list[RolloutResult]:
        """One vLLM call; `on_completion` then sees each validated output in request, index order."""
        self._initialization_open = False
        SamplingParams = import_module("vllm").SamplingParams

        sampling.validate()
        if not self._synchronized:
            raise RuntimeError("engine: post-push logprob probes have not passed")
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError("engine: duplicate request ID")
        prepared = [self.prepare(r) for r in requests]
        prompts, params = [], []
        for request, item in zip(requests, prepared, strict=True):
            for seed in request.seeds:
                prompts.append(self._prompt(item, item.token_ids))
                params.append(SamplingParams(n=1, temperature=sampling.temperature, top_p=sampling.top_p,
                    top_k=sampling.top_k, presence_penalty=sampling.presence_penalty,
                    repetition_penalty=sampling.repetition_penalty, max_tokens=request.max_new_tokens,
                    seed=seed, logprobs=0))
        outputs = iter(self.llm.generate(prompts, params, use_tqdm=False))
        results = []
        for request, item in zip(requests, prepared, strict=True):
            completions = []
            for index, seed in enumerate(request.seeds):
                output = next(outputs)
                if list(output.prompt_token_ids) != item.token_ids or len(output.outputs) != 1:
                    raise RuntimeError("engine: actual vLLM input IDs/count differ")
                c = output.outputs[0]
                ids = list(c.token_ids)
                if c.logprobs is None:
                    raise RuntimeError("engine: sampler logprobs missing")
                lps = [float(row[t].logprob) for t, row in zip(ids, c.logprobs, strict=True)]
                completions.append(_completion(ids, lps, c.finish_reason,
                    self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)))
                if on_completion is not None:
                    completion = completions[-1]
                    on_completion(request.request_id, index, seed,
                        Completion(list(completion.token_ids), list(completion.logprobs),
                                   completion.finish_reason, completion.text))
            results.append(RolloutResult(request.request_id, item.token_ids, completions))
        if next(outputs, None) is not None:
            raise RuntimeError("engine: unexpected extra vLLM outputs")
        return results

    def probe_logprobs(self, request: RolloutRequest, completion_ids: list[int]) -> list[float]:
        self._initialization_open = False
        SamplingParams = import_module("vllm").SamplingParams

        prepared = self.prepare(request)
        ids = prepared.token_ids + completion_ids
        if not completion_ids or len(ids) + 1 > self.context_limit:
            raise ValueError("engine: empty or overflowing teacher-forced probe")
        output = self.llm.generate([self._prompt(prepared, ids)], SamplingParams(n=1, temperature=1,
            top_p=1, top_k=0, presence_penalty=0, repetition_penalty=1, max_tokens=1,
            seed=request.seeds[0], prompt_logprobs=0), use_tqdm=False)[0]
        if list(output.prompt_token_ids) != ids or output.prompt_logprobs is None:
            raise RuntimeError("engine: probe input mismatch or missing teacher-forced probabilities")
        rows = output.prompt_logprobs[len(prepared.token_ids):]
        return [float(row[t].logprob) for row, t in zip(rows, completion_ids, strict=True)]

    def push_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        if type(version) is not int or version != self.policy_version + 1 or not self._synchronized:
            raise ValueError("engine: only sequential validated weight pushes are allowed")
        self._initialization_open = False
        self._load_weights(named, version)

    def _load_weights(self, named: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        self._synchronized = False
        self._push_complete = False
        self._last_loaded_version = None
        InprocClient = import_module("vllm.v1.engine.core_client").InprocClient
        client = self.llm.llm_engine.engine_core
        if not isinstance(client, InprocClient):
            raise RuntimeError("engine: verified in-process vLLM V1 required")
        scheduler = client.engine_core.scheduler
        manager = scheduler.encoder_cache_manager
        if scheduler.has_unfinished_requests() or any(manager.cached.values()):
            raise RuntimeError("engine: weight push requires an idle scheduler/encoder")
        seen: set[str] = set()

        def checked_weights():
            for name, value in named:
                if (name in seen or name not in self._weight_shapes or not isinstance(value, torch.Tensor)
                        or tuple(value.shape) != self._weight_shapes[name] or value.dtype != torch.bfloat16
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError("engine: duplicate/unexpected/nonfinite or wrong shape/dtype weight")
                seen.add(name)
                yield name, value
            if seen != self._weight_shapes.keys():
                raise ValueError("engine: incomplete canonical weight inventory")

        def load(model):
            loaded = set(model.load_weights(checked_weights()))
            # Native QKV/gate-up packing changes names, so also check the actual worker inventory.
            return loaded == {n for n, _ in model.named_parameters()} and bool(loaded)

        if self.llm.apply_model(load) != [True] or seen != self._weight_shapes.keys():
            raise RuntimeError("engine: complete native weight loading not confirmed")
        # Public LLM.reset_prefix_cache discards this bool in vLLM 0.11 V1.
        if scheduler.reset_prefix_cache() is not True:
            raise RuntimeError("engine: prefix cache invalidation not acknowledged")
        if self.llm.collective_rpc(_clear_encoder_cache) != [True]:
            raise RuntimeError("engine: encoder cache invalidation not acknowledged")
        # Clearing only GPU tensors leaves stale scheduler cache-hit bookkeeping.
        replacement = type(manager)(cache_size=manager.cache_size)
        scheduler.encoder_cache_manager = replacement
        if replacement.cached or replacement.freeable or replacement.freed:
            raise RuntimeError("engine: encoder bookkeeping invalidation failed")
        self.policy_version = version
        self._push_complete = True
        self._last_loaded_version = version
