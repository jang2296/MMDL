"""Opt-in local GPU check of the v15 unmerged-LoRA rollout path on a tiny random Qwen3-VL in vLLM 0.11.0.

MMDL_TEST_VLLM_LORA_CUDA=1 enables it (MMDL_TEST_VLLM_LORA_SLEEP=0 runs without sleep mode). Tiny random weights with
the real tokenizer/processor: this proves the wiring and the preflight's power, not 4B numerics, CUDA-graph sizes,
throughput or pod memory.

Run it as `python -m fine_tuning.tests.test_error_note_single_vllm_lora_gpu` for a truthful exit status: on WSL an
in-process vLLM engine with sleep mode (CuMem) segfaults in the interpreter's final garbage collection after the
tests have finished, on the merged v14 path as well, so `python -m unittest` reports OK and then exits 139.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SNAPSHOT = (Path.home() / "mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots"
            / "ebb281ec70b05090aa6165b016eac8ec08e71b17")
ENABLED = os.environ.get("MMDL_TEST_VLLM_LORA_CUDA") == "1"
SLEEP = os.environ.get("MMDL_TEST_VLLM_LORA_SLEEP", "1") == "1"
EOS = [151643, 151645]
SCALE = {"L": 0.25, "V": 1.0, "M": 0.02}  # LoRA B / Merger noise large enough for PF5 to test both slopes
SMALL = 2.0 ** -5  # LoRA B factor of the small-adapter case: real effect near MIN_EFFECT, like the v14 4B adapter
LINEAR = 2.0 ** -8  # LoRA B factor of the alpha-16 control: its x AMPLIFY copy stays in the tiny model's linear range


def build_tiny(dest: Path, *, printable_output: bool = False) -> Path:
    import torch
    from transformers import AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = json.loads((SNAPSHOT / "config.json").read_text())
    config["text_config"].update(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                                 head_dim=64, num_key_value_heads=2)
    config["text_config"]["rope_scaling"]["mrope_section"] = [8, 12, 12]
    config["vision_config"].update(depth=3, hidden_size=128, intermediate_size=256, num_heads=2, out_hidden_size=256,
                                   deepstack_visual_indexes=[0, 1, 2])
    if printable_output:
        config["tie_word_embeddings"] = False
        config["text_config"]["tie_word_embeddings"] = False
    torch.manual_seed(3407)
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config)).to(torch.bfloat16)
    fixture = None
    if printable_output:
        import string
        tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
        safe = [tokenizer.encode(char, add_special_tokens=False) for char in string.ascii_letters + string.digits]
        if any(len(tokens) != 1 for tokens in safe):
            raise ValueError("printable fixture requires single-token ASCII letters and digits")
        token_ids = [tokens[0] for tokens in safe] + EOS
        generator = torch.Generator().manual_seed(3407)
        vectors = torch.randn((len(token_ids) // 2, config["text_config"]["hidden_size"]), generator=generator)
        vectors = vectors / vectors.norm(dim=1, keepdim=True) * 64
        weights = model.get_output_embeddings().weight
        with torch.no_grad():
            weights.zero_()
            weights[token_ids[::2]] = vectors.to(weights.dtype)
            weights[token_ids[1::2]] = -vectors.to(weights.dtype)
        fixture = {"reportable": False, "profile": "tiny-bf16", "kind": "synthetic-printable-output-head",
                   "sampling_filter": False, "output_head_row_norm": 64, "safe_token_ids": token_ids}
    model.save_pretrained(dest, safe_serialization=True)
    for name in ("chat_template.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
                 "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"):
        shutil.copyfile(SNAPSHOT / name, dest / name)
    if fixture is not None:
        shutil.copyfile(SNAPSHOT / "generation_config.json", dest / "generation_config.json")
        (dest / "rehearsal-fixture.json").write_text(json.dumps(fixture, sort_keys=True))
    return dest


def printable_fixture_cpu_check(path: Path, data_root: Path, output: Path) -> dict:
    """Check full-vocabulary sampled bytes before and after one actual masked-note update."""
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    from mmdl.runtime.generation import GeneratedOnlyPresencePenalty
    from fine_tuning.error_note import campaign_v16b as campaign, rl, single
    from fine_tuning.error_note.protocol import ARM, V16B_CONTRACT_HASHES, plain_protocol_v18b, v18b_contract_for_update
    from fine_tuning.error_note.prep_v16b import images, p0, seal
    from fine_tuning.error_note.train_v17 import _sample_mask, _end_ids
    from fine_tuning.error_note.v18b_data import load_items
    from fine_tuning.error_note.v18b_rehearsal import policy_factory
    from fine_tuning.notebook_rl import checkpoint
    from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
    from fine_tuning.notebook_rl.prompts import user_message
    from fine_tuning.tests.test_error_note_v18b_row_quarantine import mechanical_note_row

    torch.set_num_threads(4)
    torch.manual_seed(3407)
    started = time.monotonic()
    marker = json.loads((path / "rehearsal-fixture.json").read_text())
    termination = GenerationConfig.from_pretrained(str(path), local_files_only=True)
    if ((path / "generation_config.json").read_bytes() != (SNAPSHOT / "generation_config.json").read_bytes()
            or not isinstance(termination.eos_token_id, list)
            or set(termination.eos_token_id) != set(EOS) or termination.pad_token_id != 151643):
        raise ValueError("printable fixture must preserve the pinned Qwen generation configuration")
    safe = set(marker["safe_token_ids"])
    protocol = plain_protocol_v18b("bf16", rl_blocks=8, prep_updates=1,
        contracts={key: "0" * 64 for key in V16B_CONTRACT_HASHES}, rehearsal_profile="tiny-bf16")
    bundle = policy_factory(path, ARM, protocol)
    candidates = load_items(data_root, "train")
    items = [item for item in candidates if item.get("layer") == "graph"][:8]
    if len(items) != 8:
        raise ValueError("the printable fixture needs eight fixed graph inputs")
    unsafe = torch.ones(bundle.model.config.text_config.vocab_size, dtype=torch.bool)
    unsafe[list(safe)] = False
    reports = []
    for phase in ("before-update", "after-update"):
        bundle.model.eval()
        for index, item in enumerate(items):
            torch.manual_seed(3407 + index)
            prepared, _ = prepare_protocol_inputs(bundle.processor, bundle.model.config,
                [user_message(p0(item), len(item["image_paths"]))], images(item), protocol["image"])
            count = prepared["input_tokens"]
            cfg = GenerationConfig(do_sample=True, temperature=.7, top_p=1., top_k=0,
                max_new_tokens=32, eos_token_id=EOS, pad_token_id=EOS[0], repetition_penalty=1., num_beams=1)
            processor = LogitsProcessorList([GeneratedOnlyPresencePenalty(count, 1.5)])
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                generated = bundle.model.generate(**prepared["inputs"], generation_config=cfg,
                    logits_processor=processor, use_model_defaults=False, use_cache=True,
                    return_dict_in_generate=True, output_scores=True)
            actions = generated.sequences[0, count:].tolist()
            text = bundle.processor.tokenizer.decode(actions, skip_special_tokens=True)
            _sample_mask(bundle.processor.tokenizer, actions, text, _end_ids(bundle.processor.tokenizer))
            mass = max((float(score.float().softmax(-1)[0, unsafe].sum()) for score in generated.scores), default=0.)
            if not set(actions) <= safe or mass > 1e-10:
                raise AssertionError({"phase": phase, "unsafe_sampled_ids": sorted(set(actions) - safe), "unsafe_mass": mass})
            reports.append({"phase": phase, "problem_id": item["problem_id"], "tokens": len(actions),
                            "max_unsafe_probability_mass": mass, "sampled_byte_binding": True})
        if phase == "before-update":
            row, item = mechanical_note_row(bundle, "printable-fixture-note", block_index=7)
            encoded, _ = campaign._encode_update_rows(bundle, [row], {item["problem_id"]: item}, protocol["image"])
            optimizer, scheduler = checkpoint.make_optimizer(bundle, protocol)
            for _ in range(9):
                optimizer.step()
                scheduler.step()
            cursor = {"slot": 9, "updates": 0, "policy_sha256": rl.policy_sha256(bundle), "consumed_problem_ids": []}
            definition = v18b_contract_for_update(protocol, 9, phase="rl", policy_sha256=rl.policy_sha256(bundle),
                expected_problem_ids=[], note_problem_ids=[item["problem_id"]])
            bundle.model.train()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                update = single.run_joint_single_update(bundle, optimizer, scheduler, encoded, definition,
                    cursor=cursor, block_ids=[], allow_update=True)
            if update["optimizer_steps"] != 1 or update["stats"].get("n_content_gradient_finite_nonzero") is not True:
                raise AssertionError("the printable fixture was not tested after actual note learning")
    return seal(output, {"status": "PASS", "reportable": False, "model_path": str(path),
        "source": marker, "sampling": "full-vocabulary top-p=1 top-k=0 temperature=.7 presence=1.5",
        "actual_gradient_updates": 1, "inputs": reports, "seconds": time.monotonic() - started})


def sampler(path: Path):
    from fine_tuning.error_note.single_vllm import SingleVllmSampler

    return SingleVllmSampler(path, kv_cache_gib=.25, max_num_seqs=2, gpu_memory_utilization=.3, sleep=SLEEP,
                             eos_token_ids=EOS, lora=True)


def policy(path: Path):
    """HF policy on the same BF16 snapshot with perturbed L/V LoRA B and Merger masters."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    from fine_tuning.notebook_rl.policy import configure_policy
    from fine_tuning.notebook_rl.schema import load_protocol

    base = Qwen3VLForConditionalGeneration.from_pretrained(str(path), dtype=torch.bfloat16).to("cuda")
    protocol = load_protocol(Path(__file__).parents[1] / "notebook_rl/protocol_v1.yaml")
    bundle = configure_policy(base, None, "MOD-RV", protocol, enforce_counts=False)
    generator = torch.Generator(device="cuda").manual_seed(5)
    with torch.no_grad():
        for name, parameter in bundle.model.named_parameters():
            group = "M" if ".modules_to_save.default." in name else "V" if ".visual." in name else "L"
            if ".lora_B." in name or group == "M":
                noise = torch.randn(parameter.shape, generator=generator, device="cuda") * SCALE[group]
                parameter.add_(noise.to(parameter.dtype))
    return bundle


def scale_lora_b(bundle, factor: float) -> None:
    """Every LoRA B (L and V) x factor in place; powers of two keep the round trip exact."""
    import torch

    with torch.no_grad():
        for name, parameter in bundle.model.named_parameters():
            if ".lora_B." in name:
                parameter.mul_(factor)


def unscoped_main(path: str) -> None:
    """Negative control in its own process: the engine built without the scope patch must fail PF1."""
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration as Q

    from fine_tuning.error_note import llm_lora

    upstream = getattr(Q.get_mm_mapping, "mmdl_original", Q.get_mm_mapping)
    with patch.object(llm_lora, "scope_qwen3vl_lora", lambda: None), patch.object(Q, "get_mm_mapping", upstream):
        sampler(Path(path))


def report_line(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.startswith("SINGLE_LORA_PREFLIGHT ")]
    if not lines:
        raise AssertionError("no SINGLE_LORA_PREFLIGHT line")
    return json.loads(lines[-1].removeprefix("SINGLE_LORA_PREFLIGHT "))


def registry_lifecycle(path: Path) -> dict:
    """Replay the live-preflight cleanup and trained monitor on one real vLLM engine."""
    import torch

    from fine_tuning.error_note import rl
    from fine_tuning.error_note.collect import single_request_sha256
    from fine_tuning.error_note.single_vllm import SingleVllmSampler, probe_requests

    started = time.monotonic()
    image = {"min_pixels": 4096, "max_pixels": 16384, "preprocessing": "qwen_vl_utils_0_0_14"}
    sampling = {"temperature": .7, "top_p": 1., "top_k": 0, "presence_penalty": 1.5}
    value = SingleVllmSampler(path, kv_cache_gib=.25, max_num_seqs=2, gpu_memory_utilization=.3,
        sleep=False, eos_token_ids=EOS, image=image, lora=True, version_registry=True,
        sampling=sampling, protocol_version="error-note-v18b", preflight_stage="f0")
    bundle = policy(path)
    trained = {name: p.detach().clone() for name, p in bundle.model.named_parameters() if ".lora_B." in name}
    scale_lora_b(bundle, 0.)
    try:
        value.add_version(bundle, 0)
        value.add_version(bundle, 1)
        f0 = value.preflight_report
        assert f0["pass"] and f0["pf5"]["stage_contract"]["stage"] == "f0"
        value.release_idle_versions()
        value.set_preflight_stage_v17("trained")
        with torch.no_grad():
            for name, parameter in bundle.model.named_parameters():
                if name in trained:
                    parameter.copy_(trained[name])
        print("LORA_REGISTRY_STAGE trained-monitor", file=sys.stderr, flush=True)
        value.add_version(bundle, 0)
        trained_report = value.preflight_report
        assert trained_report["pass"] and trained_report["pf5"]["stage_contract"]["stage"] == "trained"
        value.release_idle_versions()
        admitted_policy = value.add_version(bundle, 3)
        probe = probe_requests(image)[0]
        request = {"request_id": "registry-lifecycle:T1", "problem_id": "registry-lifecycle", "block_id": 0,
            "stage": "v18b-t1", "seed": 3407, "max_tokens": 32, "messages": probe.messages,
            "images": probe.images, "kind": "T1", "sampling": sampling, "policy_version": 3,
            "policy_sha256": admitted_policy, "protocol_sha256": "a" * 64, "admit_step": value.engine_steps}
        request["request_sha256"] = single_request_sha256(request)
        value.submit(request)
        outputs = []
        while not value.prefill_barrier():
            outputs.extend(value.step())
        assert value.pending, "the lifetime regression must preserve an unfinished request"
        old_id = value.versions[3]["lora_int_id"]
        with torch.no_grad():
            next(p for n, p in bundle.model.named_parameters() if ".lora_B." in n).add_(.001)
        next_policy = value.add_version(bundle, 4)
        assert next_policy != admitted_policy and old_id in value.engine.llm.llm_engine.list_loras()
        assert value.versions[3]["refcount"] == 1
        while value.has_unfinished_requests():
            outputs.extend(value.step())
        assert len(outputs) == 1
        generation = outputs[0][1]["generation"]
        assert generation["policy_sha256"] == admitted_policy and generation["lora_int_id"] == old_id
        assert rl.policy_sha256(bundle) == next_policy
        value.release_idle_versions()
        assert value.lora_request is None and value.loaded_policy is None
        report = {"status": "PASS", "reportable": False, "engine": "vllm-0.11.0", "precision": "bf16",
            "f0_pf5": f0["pf5"], "trained_pf5": trained_report["pf5"], "generated_tokens": len(generation["completion_token_ids"]),
            "preserved_admission_policy": True, "seconds": time.monotonic() - started}
        print("LORA_REGISTRY_LIFECYCLE " + json.dumps(report, sort_keys=True), file=sys.stderr, flush=True)
        return report
    finally:
        shutil.rmtree(value.lora_dir, ignore_errors=True)


@unittest.skipUnless(ENABLED, "vLLM LoRA GPU check requires explicit MMDL_TEST_VLLM_LORA_CUDA=1")
class TinyQwenLoraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="tiny-qwen3vl-")
        cls.path = build_tiny(Path(cls.tmp.name) / "model")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_engine_without_scope_patch_fails_pf1(self):
        code = "import sys; from fine_tuning.tests.test_error_note_single_vllm_lora_gpu import unscoped_main; " \
               "unscoped_main(sys.argv[1])"
        run = subprocess.run([sys.executable, "-c", code, str(self.path)], cwd=Path(__file__).parents[2],
                             capture_output=True, text=True, timeout=900)
        sys.stderr.write(run.stderr[-6000:])
        self.assertNotEqual(run.returncode, 0)
        if "SINGLE_LORA_PREFLIGHT " in run.stderr:  # engine started: PF1 must see the vision wrappers
            report = report_line(run.stderr)
            self.assertFalse(report["pf1"]["pass"])
            self.assertTrue(report["pf1"]["foreign_modules"] or report["pf1"]["renamed_vision_weights"])
        else:  # vLLM 0.11.0 fails at start-up: a vision proj LoRA wrapper runs punica with the decoder token map
            self.assertIn("self.proj(context_layer)", run.stderr)
            self.assertIn("assert token_lora_mapping.size(0) == M", run.stderr)

    def test_b_preflight_passes_and_controls_fail(self):
        from fine_tuning.error_note import llm_lora

        value, bundle = sampler(self.path), policy(self.path)
        self.addCleanup(shutil.rmtree, value.lora_dir, True)
        value.sync(bundle)  # first LoRA sync runs PF1-PF5 and raises on failure: adapter 1, probe 2, restored 3
        report = value.preflight_report
        self.assertTrue(report["pass"])
        amplified = report["pf5"]["amplified"]
        self.assertTrue(amplified["llm"]["pass"] and amplified["vision"]["pass"])  # both amplified slopes tested
        self.assertTrue(report["pf2_restored"]["pass"])
        self.assertEqual((report["lora_int_id"], report["restored_lora_int_id"], value.lora_request.lora_int_id), (1, 3, 3))
        self.assertIs(report["pf4"]["pass"], True if SLEEP else None)

        value.vision_hooks.on = False  # control 1: vision hooks off (ids 4, 5)
        with self.assertRaises(RuntimeError):
            value.verify_lora(bundle)
        self.assertFalse(value.preflight_report["pf2"]["pass"])
        self.assertIs(value.preflight_report["pf5"]["amplified"]["vision"]["pass"], False)  # tested and failed
        self.assertFalse(value.vision_hooks.on)  # the probe restores the hooks as it found them
        value.vision_hooks.on = True

        scale_lora_b(bundle, SMALL)  # a trained-size adapter: sync id 6, preflight ids 7, 8 (raises on failure)
        value.sync(bundle)
        small = value.verify_lora(bundle)
        print("SINGLE_LORA_SMALL " + json.dumps({"small": SMALL, **{k: small["pf5"][k] for k in
                                                ("real", "amplified", "mean_abs_on")}}, sort_keys=True),
              file=sys.stderr, flush=True)
        self.assertLess(small["pf5"]["real"]["total"]["rms_effect"], 3 * llm_lora.MIN_EFFECT)
        self.assertTrue(small["pf5"]["amplified"]["llm"]["pass"] and small["pf5"]["amplified"]["vision"]["pass"])

        # control 2: every adapter file written with alpha 16 (vLLM scaling 1, not 2), on an adapter whose x AMPLIFY
        # copy stays in the tiny model's near-linear range: x16 of the full tiny adapter saturates the log-probs and
        # hid the halved scale there (amplified LLM slope 0.985; PF2 alone failed it)
        scale_lora_b(bundle, LINEAR / SMALL)  # sync id 9
        value.sync(bundle)
        original = llm_lora.adapter_config
        with patch.object(llm_lora, "adapter_config", lambda b: original(b) | {"lora_alpha": 16}):
            value._swap_adapter(bundle, value.loaded_policy)  # id 10
            with self.assertRaises(RuntimeError):
                value.verify_lora(bundle)  # ids 11, 12
        control = value.preflight_report
        print("SINGLE_LORA_ALPHA16 " + json.dumps({"linear": LINEAR, **{k: control["pf5"][k] for k in
                                                  ("real", "amplified")}}, sort_keys=True), file=sys.stderr, flush=True)
        self.assertFalse(control["pf2"]["pass"])
        self.assertIs(control["pf5"]["amplified"]["llm"]["pass"], False)  # tested (effect >= MIN_EFFECT) and failed
        # A halved LLM scale: ~0.5 in the linear limit; the greedy (high-probability) probe tokens respond
        # superlinearly, so the measured slope sits lower (0.28 at LINEAR 2^-8, 0.52 at 2^-9 below MIN_EFFECT).
        self.assertLess(control["pf5"]["amplified"]["llm"]["slope"], .6)

        scale_lora_b(bundle, 1 / LINEAR)  # the original adapter again (exact): sync id 13
        value.sync(bundle)
        value.release()  # real level-1 sleep in LoRA mode, then wake + Merger push + new adapter
        value.sync(bundle)
        self.assertEqual(value.lora_request.lora_int_id, 14 if SLEEP else 13)
        self.assertTrue(value.verify_lora(bundle)["pass"])
        self.check_stream(value, bundle)

    def test_c_registry_preflight_cleanup_then_trained_monitor(self):
        self.assertEqual(registry_lifecycle(self.path)["status"], "PASS")

    def check_stream(self, value, bundle):
        """The campaign's own stream (continuous add_request/step through _LoraEngine) samples the unmerged-LoRA
        policy: HF replays the sampled actions' log q within the training replay bound with the adapter on, not off."""
        import torch

        from fine_tuning.error_note import llm_lora, rl
        from fine_tuning.error_note.collect import SINGLE_SAMPLING, StreamStats
        from fine_tuning.error_note.collect_v4 import validate_single_output
        from fine_tuning.error_note.single_vllm import probe_requests
        from fine_tuning.notebook_rl.engines import RolloutRequest

        args = SimpleNamespace(policy_sha256=rl.policy_sha256(bundle), protocol_sha256="a" * 64)
        requests = [{"request_id": f"p{i}:t1", "problem_id": f"p{i}", "block_id": 0, "stage": "t1", "seed": 5 + i,
                     "max_tokens": 32, "messages": probe.messages, "images": probe.images}
                    for i, probe in enumerate(probe_requests(value.image))]
        stats = StreamStats()
        outputs = list(value.stream_builder(bundle, args, stats)(requests, StreamStats(), None))
        self.assertEqual(sorted(request["request_id"] for request, _ in outputs), ["p0:t1", "p1:t1"])
        self.assertEqual(stats.receipt_extra["single_sampler"], llm_lora.ENGINE_LORA)
        base, device = bundle.model.get_base_model(), next(bundle.model.parameters()).device

        def replay(request, generation):
            ids = generation["completion_token_ids"]
            probe = RolloutRequest(request["request_id"], None, request["messages"], request["images"], 1, len(ids),
                                   [request["seed"]])
            inputs = value.engine.model_inputs(probe, ids, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = base.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
                scores = base.lm_head(hidden[0, -len(ids) - 1:-1]).float()
            done = torch.tensor([ids], device=device)
            return [float(rl.stored_support_logprob(scores[t:t + 1], done[:, :t], ids[t], support,
                                                     temperature=SINGLE_SAMPLING["temperature"],
                                                     presence_penalty=SINGLE_SAMPLING["presence_penalty"]))
                    for t, support in enumerate(generation["support_token_ids"])]

        on = off = count = 0.0
        for request, output in outputs:
            validate_single_output(request, output, policy_sha256=args.policy_sha256, protocol_sha256="a" * 64)
            generation = output["generation"]
            self.assertEqual(generation["engine"], llm_lora.ENGINE_LORA)
            sampled = generation["sampling_logprobs"]
            on += sum(abs(a - b) for a, b in zip(replay(request, generation), sampled, strict=True))
            with llm_lora.adapters_off(bundle, "L", "V"):
                off += sum(abs(a - b) for a, b in zip(replay(request, generation), sampled, strict=True))
            count += len(sampled)
        print(f"SINGLE_LORA_STREAM tokens {int(count)} replay_on {on / count:.5f} replay_off {off / count:.5f}",
              file=sys.stderr, flush=True)
        self.assertLessEqual(on / count, 0.05)  # protocol replay block bound
        self.assertGreater(off / count, 5 * max(on / count, 0.01))


if __name__ == "__main__":
    outcome = unittest.main(exit=False).result.wasSuccessful()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if outcome else 1)  # skip the final GC that segfaults under WSL CuMem (see module docstring)
