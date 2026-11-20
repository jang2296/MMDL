"""One unverified, same-policy image reread; context only, never a rewarded action."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict
from typing import Any, Callable

from .engines import RolloutRequest, _completion
from .prompts import p0_body, user_message
from .rollout import _hash, _sample_request, seeds
from .schema import ActorItem, from_json, to_json

POLICY = "fresh-image-reread-before-full-history-v1"
MAX_NEW_TOKENS = 1024
PROMPT = """Inspect the original images and report their observable facts only.
Do not solve the question, give a final answer, diagnose attempts, or write a lesson.
Put the literal heading Observed facts: on its own line. Then put each slot assignment
on its own line. End with the literal line Solution: and write nothing after it.
Read each arrow's source and target from the image, and each printed value from its own image.
Question and choices (context only):
{question}
Public observation schema (format only, no correct values):
{schema}
"""


def request_for(actor: ActorItem, schema: str, images: list, policy_version: int,
                master_seed: int) -> RolloutRequest:
    to_json(actor)
    if (not isinstance(schema, str) or not schema.strip()
            or len(images) != len(actor.image_paths) or not images):
        raise ValueError("fresh observation: public schema and original images required")
    question = p0_body(actor.question, actor.choices).removesuffix(
        "Please select the correct answer from the options above.\n")
    return RolloutRequest(f"{actor.problem_id}:v{policy_version}:FRESH_OBSERVATION", None,
        [user_message(PROMPT.format(question=question, schema=schema), len(images))], images,
        1, MAX_NEW_TOKENS, seeds(actor.problem_id, policy_version, 1, POLICY, 1, master_seed=master_seed))


def generate(actor: ActorItem, schema: str, images: list, engine: Any, *,
             master_seed: int, context_limit: int,
             on_event: Callable[[dict], None] | None = None) -> dict:
    # No verifier is accepted: neither sample selection nor the input can depend on its answers.
    from PIL.Image import Image
    from .notebook_layout_probe import _image_identity

    policy_version = engine.policy_version
    actor_wire = to_json(actor)
    request = request_for(actor, schema, images, policy_version, master_seed)

    def state() -> str:
        return _hash({"actor": to_json(actor), "schema": schema, "policy_version": engine.policy_version,
            "request": {"request_id": request.request_id, "prompt_token_ids": request.prompt_token_ids,
                "messages": request.messages, "n": request.n, "max_new_tokens": request.max_new_tokens,
                "seeds": request.seeds},
            "images": [(id(image), image.mode, image.size, hashlib.sha256(image.tobytes()).hexdigest())
                if isinstance(image, Image) else (id(image),) for image in images],
            "request_images": [id(image) for image in request.images]})

    original = state()

    def unchanged() -> None:
        if state() != original or to_json(actor) != actor_wire:
            raise ValueError("fresh observation: original request, actor, images or policy changed")

    prepared = engine.prepare(request)
    unchanged()
    if len(prepared.token_ids) + request.max_new_tokens > context_limit:
        raise ValueError("CONTEXT_BLOCKED: fresh observation reservation")
    prepared_images = _hash(_image_identity(actor, prepared))
    receipt = {"policy": POLICY, "policy_version": policy_version,
        "master_seed": master_seed, "actor": json.loads(actor_wire), "schema": schema,
        "request_id": request.request_id, "messages": copy.deepcopy(request.messages), "seeds": list(request.seeds),
        "max_new_tokens": MAX_NEW_TOKENS, "prompt_token_ids": list(prepared.token_ids),
        "image_key": prepared.image_key, "input_tokens": len(prepared.token_ids),
        "used_for_routing": False, "eligible_for_rl_loss": False, "verified": False}

    def emit(phase: str) -> None:
        if on_event is not None:
            before = _hash(receipt)
            event = copy.deepcopy({"phase": phase, **receipt})
            event_sha = _hash(event)
            on_event(event)
            if _hash(receipt) != before or _hash(event) != event_sha:
                raise ValueError("fresh observation: receipt changed in persistence callback")
            unchanged()
            if _hash(_image_identity(actor, prepared)) != prepared_images:
                raise ValueError("fresh observation: prepared image values changed in callback")

    def prepared_matches(current) -> None:
        unchanged()
        if (current.token_ids != receipt["prompt_token_ids"] or current.image_key != receipt["image_key"]
                or _hash(_image_identity(actor, current)) != prepared_images):
            raise ValueError("fresh observation: prepared prompt or image changed during sampling")

    emit("STARTED")
    began = time.monotonic()
    output = _sample_request(request, engine, policy_version, context_limit,
                             validate_request=unchanged, validate_prepared=prepared_matches)
    if output.prompt_token_ids != receipt["prompt_token_ids"]:
        raise ValueError("fresh observation: prepared prompt changed during sampling")
    completion = output.completions[0]
    _completion(completion.token_ids, completion.logprobs, completion.finish_reason, completion.text)
    receipt.update(completion=copy.deepcopy(asdict(completion)), output_tokens=len(completion.token_ids),
                   seconds=time.monotonic() - began)
    receipt["sha256"] = _hash(receipt)
    validate(receipt, policy_version=policy_version, tokenizer=engine.tokenizer,
             preparation=engine, images=images, actor=actor, schema=schema, master_seed=master_seed)
    unchanged()
    emit("COMPLETED")
    return receipt


def validate(receipt: dict, *, policy_version: int, tokenizer: Any | None = None,
             preparation: Any | None = None, images: list | None = None,
             actor: ActorItem | None = None, schema: str | None = None, master_seed: int | None = None) -> None:
    try:
        if set(receipt) != {"policy", "policy_version", "master_seed", "actor", "schema", "request_id",
                "messages", "seeds", "max_new_tokens", "prompt_token_ids", "image_key", "input_tokens",
                "used_for_routing", "eligible_for_rl_loss", "verified", "completion", "output_tokens", "seconds", "sha256"}:
            raise ValueError("receipt fields differ")
        if (any(type(receipt[key]) is not int for key in
                ("policy_version", "master_seed", "max_new_tokens", "input_tokens", "output_tokens"))
                or type(receipt["seeds"]) is not list or any(type(seed) is not int for seed in receipt["seeds"])
                or type(receipt["prompt_token_ids"]) is not list
                or type(receipt["completion"]) is not dict
                or set(receipt["completion"]) != {"token_ids", "logprobs", "finish_reason", "text"}):
            raise ValueError("receipt scalar or completion types differ")
        recorded_actor = from_json(ActorItem, json.dumps(receipt["actor"], allow_nan=False))
        if (actor is not None and actor != recorded_actor
                or schema is not None and schema != receipt["schema"]
                or master_seed is not None and master_seed != receipt["master_seed"]):
            raise ValueError("approved actor/schema/seed differs")
        request = request_for(recorded_actor, receipt["schema"], [None] * len(recorded_actor.image_paths),
                              policy_version, receipt["master_seed"])
        if (receipt.get("sha256") != _hash({k: v for k, v in receipt.items() if k != "sha256"})
                or receipt["policy"] != POLICY or receipt["policy_version"] != policy_version
                or receipt["request_id"] != request.request_id or receipt["messages"] != request.messages
                or receipt["seeds"] != request.seeds or receipt["max_new_tokens"] != MAX_NEW_TOKENS
                or receipt["input_tokens"] != len(receipt["prompt_token_ids"])
                or not receipt["prompt_token_ids"]
                or any(type(t) is not int or t < 0 for t in receipt["prompt_token_ids"])
                or type(receipt["image_key"]) is not str or not receipt["image_key"]
                or any(receipt[k] is not False for k in ("used_for_routing", "eligible_for_rl_loss", "verified"))
                or type(receipt["seconds"]) not in (int, float)
                or not math.isfinite(receipt["seconds"]) or receipt["seconds"] < 0):
            raise ValueError("changed receipt")
        c = receipt["completion"]
        _completion(c["token_ids"], c["logprobs"], c["finish_reason"], c["text"])
        if (receipt["output_tokens"] != len(c["token_ids"]) or len(c["token_ids"]) > MAX_NEW_TOKENS
                or tokenizer is not None and tokenizer.decode(c["token_ids"], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) != c["text"]):
            raise ValueError("actual completion differs")
        if preparation is not None:
            if images is None or preparation.policy_version != policy_version:
                raise ValueError("original images and same-policy preparation required")
            current = preparation.prepare(request_for(recorded_actor, receipt["schema"], images,
                                                       policy_version, receipt["master_seed"]))
            if current.token_ids != receipt["prompt_token_ids"] or current.image_key != receipt["image_key"]:
                raise ValueError("actual image/prompt differs")
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"fresh observation: invalid binding ({exc})") from exc


def quoted_context(receipt: dict) -> str:
    # Reversible quoting preserves the actual raw text, including malformed or truncated outputs.
    value = json.dumps({"finish_reason": receipt["completion"]["finish_reason"],
                        "raw_response": receipt["completion"]["text"]}, ensure_ascii=False)
    value = value.replace("<", "\\u003c").replace(">", "\\u003e")
    return ("An independent reread by this same policy, without prior attempts, follows. "
            "It is model-generated and UNVERIFIED, not a correct oracle or an additional solution attempt. "
            "Compare it with the original images and ALL original attempts; never assume either is correct. "
            "Do not include this reread in the attempts array or change the designated successful retry.\n"
            "<unverified_fresh_observation>\n" + value + "\n</unverified_fresh_observation>\n\n")


RETRY_CONTEXT = ("An independent image reread by this same policy, without prior attempts, follows. "
    "It is model-generated and UNVERIFIED, including any malformed or truncated text. "
    "Reinspect the original images and evaluate both this reread and the prior response yourself. "
    "This is context, not an additional solution attempt or verified observation. "
    "Continue under the original retry output contract.\n")


def quoted_retry_context(receipt: dict) -> str:
    completion = receipt["completion"]
    if completion["finish_reason"] not in ("stop", "length") or type(completion["text"]) is not str:
        raise ValueError("fresh observation: actual reread text and termination required")
    value = json.dumps({"finish_reason": completion["finish_reason"], "raw_response": completion["text"]},
                       ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    return RETRY_CONTEXT + "<unverified_fresh_observation>\n" + value + "\n</unverified_fresh_observation>\n"


def events_for(receipt: dict) -> list[dict]:
    started = {key: value for key, value in receipt.items()
               if key not in ("completion", "output_tokens", "seconds", "sha256")}
    return [{"phase": "STARTED", **started}, {"phase": "COMPLETED", **receipt}]
