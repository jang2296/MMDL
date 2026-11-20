"""CPU reference objectives, real tiny-policy gradients and sampled-byte encoder contracts."""
import copy
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.error_note import rl, train_v17 as train
from fine_tuning.error_note.contract import ROUTE_FULL, ROUTE_LM
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests.test_error_note_train import END, encoded, tiny_factory
from fine_tuning.tests.test_error_note_v16b_loss import TOKENS, contract as old_contract, row as old_row


def row(bundle, index=0, kind="RL", route=ROUTE_LM, advantage=1., version=1):
    original_kind = "RL" if kind == "A1" else kind
    value = old_row(bundle, index, original_kind, route, advantage, version)
    if kind == "A1":
        value = {key: val for key, val in value.items()
                 if key not in {"reward", "baseline", "advantage", "baseline_receipt", "pg"}}
    pid = value["problem_id"]
    n = value["target_len"]
    value.update(schema=train.ROW_SCHEMA, row_id=f"{pid}:{kind}", parent_id=pid, kind=kind,
                 reason="CORRECT" if advantage > 0 else "INCORRECT", prompt="question", target="target",
                 training_prompt_sha256=hashlib.sha256(b"question").hexdigest(),
                 loss_probability_space="processed-full-vocab-v1" if kind in train.PG_KINDS else "raw-full-vocab-v1")
    value["input_sha256"] = train.encoded_input_sha256_v17(value)
    value["loss_mask"] = torch.tensor([True] * n if kind in train.PG_KINDS else [True] * (n - 1) + [False])
    if kind == "PREP_HUMAN":
        value["target_origin"] = "human"
    else:
        value["generation"]["policy_sha256"] = rl.policy_sha256(bundle)
    if kind == "N":
        value["mask_mode"] = "verified-corrected-value-content-only-v1"
        value["verified_byte_spans"] = [[0, 5]]
        value["verified_content_token_mask"] = value["loss_mask"].tolist()
    if kind in train.SFT_KINDS:
        bind_sft(value)
    return value


def bind_sft(value):
    kind = value["kind"]
    value["generation_prompt_sha256"] = hashlib.sha256(value["prompt"].encode()).hexdigest()
    receipt = {"accepted": True, "problem_id": value["problem_id"],
               "output_binding_sha256": digest({"generation": value.get("generation"), "text": value["target"],
                                                "prompt_sha256": value["generation_prompt_sha256"]}),
               "stage": "resolve" if kind == "C" else "t1",
               "record_binding_sha256": digest({"problem_id": value["problem_id"], "p0_text": value["prompt"],
                                                "image_paths": value.get("image_paths", []),
                                                "image_sha256": value.get("image_sha256", [])})}
    if kind == "N":
        receipt.update(n_accepted=True, verified_repair_count=1,
                       verified_byte_spans=value.get("verified_byte_spans", []),
                       verified_content_token_mask=value["verified_content_token_mask"])
    if kind == "PREP_HUMAN":
        receipt.update(target_sha256=hashlib.sha256(value["target"].encode()).hexdigest(),
                       native_reference_sha256="a" * 64, text_check_receipt={"sha256": "b" * 64}, human_verified=False)
    receipt["sha256"] = digest(receipt)
    value.update(verification_receipt=receipt, admission_receipt_sha256=receipt["sha256"],
                 verifier_receipt_sha256=receipt["sha256"])


def contract(bundle, rows, *, version=1, phase="rl"):
    definition = old_contract(bundle, rows, version=version, phase=phase)
    return definition | {"version": "error-note-v17", "master_seed": 3407,
                         "f0_position_salt": train.F0_POSITION_SALT}


def gradients(bundle):
    return {name: p.grad.detach().clone() for name, p in bundle.model.named_parameters() if p.grad is not None}


class V17LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_pg_reference_uses_fixed_sixteen_and_detached_importance(self):
        bundle = tiny_factory()
        rows = [row(bundle, i, advantage=.7 if i % 2 else -.6) for i in range(8)]
        definition = contract(bundle, rows)
        report = train.backward_v17_block(bundle, rows, definition, components={"RL"})
        actual = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        values = train.v17_row_logprobs(bundle, rows)
        terms = []
        for item, value in zip(rows, values, strict=True):
            mu = value.new_tensor(item["generation"]["sampling_logprobs"])
            ratio = (value.detach() - mu).exp().clamp(max=2)
            mask = rl.negative_token_mask(value) if item["advantage"] < 0 else torch.ones_like(value, dtype=torch.bool)
            terms.append(-item["advantage"] * (ratio * mask * value).sum() / (16 * 1024))
        expected = sum(terms)
        expected.backward()
        self.assertAlmostEqual(report["loss"], float(expected.detach()), places=7)
        self.assertEqual(report["rl"]["denominator"], 16 * 1024)
        for name, value in gradients(bundle).items():
            torch.testing.assert_close(value, actual[name], atol=2e-8, rtol=2e-4)

    def test_raw_sft_fixed_denominator_missing_channels_and_prep_actual_rows(self):
        bundle = tiny_factory()
        for kinds, phase, denominator in ((["C"], "rl", 16), (["N", "C", "C"], "rl", 16),
                                           (["PREP_SELF", "PREP_HUMAN"], "prep", 2)):
            bundle.model.zero_grad(set_to_none=True)
            rows = [row(bundle, i, kind=kind) for i, kind in enumerate(kinds)]
            definition = contract(bundle, rows, phase=phase)
            report = train.backward_v17_block(bundle, rows, definition, components=set(kinds))
            with torch.no_grad():
                values = train.v17_row_logprobs(bundle, rows)
                manual = sum(-train.SFT_WEIGHTS[r["kind"]] * value[r["loss_mask"]].mean() / denominator
                             for r, value in zip(rows, values, strict=True))
            self.assertAlmostEqual(report["loss"], float(manual), places=6)
            self.assertTrue(all(item["parent_denominator"] == denominator for item in report["sft"].values()))
            if "N" in kinds:
                self.assertTrue(report["n_content_gradient_finite_nonzero"])
                self.assertEqual(report["verified_repair_parent_ids"], [rows[0]["problem_id"]])

    def test_shared_rl_a1_equals_separate_forward_and_microbatch_gradients(self):
        bundle = tiny_factory()
        rows = [row(bundle), row(bundle, kind="A1"), row(bundle, 1, kind="C", route=ROUTE_FULL)]
        definition = contract(bundle, rows)
        shared = train.backward_v17_block(bundle, rows, definition, components={"RL", "A1", "C"})
        actual = gradients(bundle)
        self.assertEqual(shared["shared_t1_a1_pairs"], 1)
        bundle.model.zero_grad(set_to_none=True)
        reference = 0
        for item in rows:
            value = train.v17_row_logprobs(bundle, [item])[0]
            if item["kind"] == "RL":
                ratio = (value.detach() - value.new_tensor(item["generation"]["sampling_logprobs"])).exp().clamp(max=2)
                reference = reference - (ratio * value).sum() / (16 * 1024)
            else:
                reference = reference - train.SFT_WEIGHTS[item["kind"]] * value[item["loss_mask"]].mean() / 16
        reference.backward()
        self.assertAlmostEqual(shared["loss"], float(reference.detach()), places=6)
        for name, value in gradients(bundle).items():
            torch.testing.assert_close(value, actual[name], atol=2e-7, rtol=4e-4)
        bundle.model.zero_grad(set_to_none=True)
        divided = train.backward_v17_block(bundle, rows, definition, token_budget=1, components={"RL", "A1", "C"})
        self.assertAlmostEqual(shared["loss"], divided["loss"], places=6)
        for name, value in gradients(bundle).items():
            torch.testing.assert_close(value, actual[name], atol=2e-7, rtol=4e-4)

    def test_negative_obs_keeps_high_probability_actions_while_t1_masks_them(self):
        bundle = tiny_factory()
        for kind in ("RL", "OBS"):
            bundle.model.zero_grad(set_to_none=True)
            item = row(bundle, kind=kind, advantage=-.5)
            logq = torch.tensor([-.1, -1., -.1, -1., -.1, -1.], requires_grad=True)
            item["generation"]["sampling_logprobs"] = logq.detach().tolist()
            states = torch.zeros((6, 8))
            with patch.object(train, "_row_values", return_value=[(torch.stack([logq - .2, logq]), states)]):
                report = train.backward_v17_block(bundle, [item], contract(bundle, [item]), components={kind})
            self.assertEqual(report["obs" if kind == "OBS" else "rl"]["tokens"], 6 if kind == "OBS" else 3)
            self.assertGreater(float(logq.grad[1]), 0)
            self.assertEqual(float(logq.grad[0]) == 0., kind == "RL")
            self.assertEqual(report["t1_negative_candidate_tokens"], 6 if kind == "RL" else 0)
            self.assertEqual(report["t1_negative_retained_tokens"], 3 if kind == "RL" else 0)
            self.assertEqual(report["t1_negative_retained_ratio"], .5 if kind == "RL" else 0.)

    def test_hold_clears_previously_accumulated_sft_kl_and_restores_state(self):
        bundle = tiny_factory()
        bundle.model.eval()
        bundle.route.apply(False)
        rows = [row(bundle, kind="N"), row(bundle, 1)]
        rows[1]["generation"]["sampling_logprobs"] = [x - 1 for x in rows[1]["generation"]["sampling_logprobs"]]
        result = train.backward_v17_block(bundle, rows, contract(bundle, rows), token_budget=1)
        self.assertTrue(result["hold"])
        self.assertFalse(result["task_signal"] or result["note_gradient_retained"])
        self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))
        self.assertFalse(bundle.model.training or bundle.route.vision_on)

    def test_one_mixed_sft_ancestor_holds_entire_block_before_forward(self):
        bundle = tiny_factory()
        rows = [row(bundle, i) for i in range(16)] + [row(bundle, i, "A1") for i in range(16)]
        rows[-1]["source_generations"] = [{"mixed_version": True}]
        with patch.object(train, "_row_values", side_effect=AssertionError("no forward on mixed")):
            result = train.backward_v17_block(bundle, rows, contract(bundle, rows))
        self.assertEqual(result["hold_reason"], "MIXED_VERSION")
        self.assertFalse(result["consumed_row_ids"])

    def test_sft_receipt_rebinds_target_and_p0_and_only_n_gradient_channels_default(self):
        bundle = tiny_factory()
        item = row(bundle, kind="C")
        definition = contract(bundle, [item])
        changed = copy.deepcopy(item)
        changed["target"] = "altered rationale"
        with self.assertRaisesRegex(ValueError, "admission receipt"):
            train.backward_v17_block(bundle, [changed], definition)
        changed = copy.deepcopy(item)
        changed["prompt"] = "altered P0"
        with self.assertRaisesRegex(ValueError, "P0 input"):
            train.backward_v17_block(bundle, [changed], definition)
        result = train.backward_v17_block(bundle, [item], definition, components={"C"})
        self.assertEqual(result["component_gradient_channels_measured"], [])
        self.assertEqual(result["component_gradient_norms"], {})

    def test_f0_bins_reproduce_cover_all_positions_and_weight_to_one(self):
        for n in (0, 1, 7, 64, 65, 8192):
            positions = train.f0_positions_v17(n, "parent")
            self.assertEqual(positions, train.f0_positions_v17(n, "parent"))
            self.assertEqual(len(positions), min(n, 64))
            if positions:
                self.assertAlmostEqual(sum(p["weight"] for p in positions), 1.)
                self.assertTrue(all(p["start"] <= p["position"] < p["stop"] for p in positions))
            if n <= 64:
                self.assertEqual([p["position"] for p in positions], list(range(n)))
        self.assertNotEqual(train.f0_positions_v17(8192, "parent"), train.f0_positions_v17(8192, "other"))

    def test_f0_dense_kl_all_t1_once_including_pg_excluded_and_a1(self):
        from fine_tuning.error_note import train_v4
        bundle = tiny_factory()
        with torch.no_grad():
            for parameter in bundle.params["L"]:
                parameter.add_(.01)
        rows = [row(bundle, i) for i in range(3)]
        rows[1].update(pg=False, reward=0., advantage=0.)
        rows.append(row(bundle, 0, "A1"))
        definition = contract(bundle, rows)
        reference = train_v4._single_reference_hidden
        lengths = []

        def recording(bundle, values):
            lengths.extend(r["target_len"] for r in values)
            result = reference(bundle, values)
            self.assertFalse(result.requires_grad)
            return result

        with patch.object(train_v4, "_single_reference_hidden", recording):
            result = train.backward_v17_block(bundle, rows, definition, components={"KL"})
        self.assertEqual(lengths, [6, 6, 6])
        self.assertEqual(len(result["f0_positions"]), 3)
        self.assertGreater(result["loss"], 0.)
        self.assertFalse(result["task_signal"])
        bundle.model.zero_grad(set_to_none=True)
        manual = []
        for item in rows[:3]:
            hidden, _, lengths = train_v4._single_hidden(bundle, [item])
            original = reference(bundle, [item])
            start = lengths[0] - item["target_len"] - 1
            head = bundle.model.get_base_model().lm_head
            current_p = head(hidden[0, start:start + item["target_len"]]).float().log_softmax(-1)
            original_p = head(original[0, start:start + item["target_len"]]).float().log_softmax(-1)
            manual.append((original_p.exp() * (original_p - current_p)).sum(-1).mean() * .02 / 16)
        self.assertAlmostEqual(result["loss"], float(sum(manual).detach()), places=7)

    def test_preparation_uses_only_first_four_reference_positions(self):
        from fine_tuning.error_note import train_v4
        bundle = tiny_factory()
        rows = [row(bundle, kind="PREP_SELF"), row(bundle, 1, "PREP_HUMAN")]
        reference = train_v4._single_reference_hidden
        lengths = []

        def recording(bundle, values):
            lengths.extend(r["target_len"] for r in values)
            return reference(bundle, values)

        with patch.object(train_v4, "_single_reference_hidden", recording):
            result = train.backward_v17_block(bundle, rows, contract(bundle, rows, phase="prep"))
        self.assertEqual(lengths, [4, 4])
        self.assertTrue(result["task_signal"])

    def test_chunk_history_raw_and_processed_spaces_and_reference_exception_restore(self):
        from fine_tuning.error_note import train_v4
        bundle = tiny_factory()
        rows = [row(bundle)]
        value = train._row_values(bundle, rows, chunk=1)[0][0]
        full = train._row_values(bundle, rows, chunk=64)[0][0]
        torch.testing.assert_close(value, full)
        self.assertFalse(torch.allclose(value[0], value[1]))
        bundle.model.eval()
        bundle.route.apply(False)
        modes = [m.training for m in bundle.model.modules()]
        flags = [p.requires_grad for p in bundle.model.parameters()]
        with patch.object(train_v4, "_single_raw_kl", side_effect=RuntimeError("reference projection failed")):
            with self.assertRaisesRegex(RuntimeError, "reference projection"):
                train.backward_v17_block(bundle, rows, contract(bundle, rows))
        self.assertEqual(modes, [m.training for m in bundle.model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in bundle.model.parameters()])
        self.assertFalse(bundle.route.vision_on)
        self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))

    def test_empty_flush_and_tampered_advantage_fail_without_gradients(self):
        bundle = tiny_factory()
        result = train.backward_v17_block(bundle, [], contract(bundle, []))
        self.assertEqual(result["loss"], 0.)
        self.assertFalse(result["has_gradient"] or result["task_signal"])
        item = row(bundle)
        definition = contract(bundle, [item])
        item["advantage"] = .123
        with self.assertRaises(ValueError):
            train.backward_v17_block(bundle, [item], definition)

    def test_loop_penalty_ignores_probability_filter_and_caps_by_window_tokens(self):
        bundle = tiny_factory()
        item = row(bundle, advantage=-.5)
        count = 4096
        item.update(target="x" * (count - 1), reason="LOOP_CONFIRMED", target_len=count,
                    input_ids=torch.tensor([7, 8] + [12] * (count - 1) + [END]),
                    loss_mask=torch.tensor([False] * (count - 3001) + [True] * 3000 + [False]))
        item["generation"].update(completion_token_ids=item["input_ids"][-count:].tolist(),
                                  finish_reason="length", sampling_logprobs=[-.01] * count)
        item["input_sha256"] = train.encoded_input_sha256_v17(item)
        logq = torch.full((count,), -.01, requires_grad=True)
        with patch.object(train, "_row_values", return_value=[(torch.stack([logq, logq]), torch.zeros((count, 8)))]):
            result = train.backward_v17_block(bundle, [item], contract(bundle, [item]), components={"RL"})
        self.assertEqual(result["loop"]["tokens"], 3000)
        self.assertEqual(result["loop_tokens_penalized"], 3000)
        self.assertEqual(result["loop_confirmed"], 1)
        self.assertEqual(result["t1_negative_candidate_tokens"], 0)
        manual = .5 * (2048 / 3000) * 3000 * -.01 / (16 * 1024)
        self.assertAlmostEqual(result["loss"], manual, places=7)
        self.assertEqual(int((logq.grad != 0).sum()), 3000)
        self.assertFalse(bool(logq.grad[-1]))

    def test_capacity_probe_shared_full_graph_restores_state_and_rejects_unfit_budget(self):
        from contextlib import ExitStack
        from fine_tuning.error_note import capacity, train_v4
        from fine_tuning.error_note.protocol import ARM, plain_protocol
        from fine_tuning.error_note.train import _all_parameter_hashes
        from fine_tuning.notebook_rl.checkpoint import capture_rng
        from fine_tuning.notebook_rl.policy import configure_policy
        from fine_tuning.tests.test_training import tiny_model

        bundle = configure_policy(tiny_model().to(dtype=torch.bfloat16), None, ARM,
                                  plain_protocol("bf16"), enforce_counts=False)
        specimen = encoded(kind="RL", route=ROUTE_FULL)
        bounds = {"max_images": 1, "max_vision_patches": 16, "max_prompt_tokens": 12, "dataset_sha256": "f" * 64}
        incoming = bundle.params["L"][0]
        incoming.grad = torch.ones_like(incoming)
        gradient = incoming.grad
        weights, rng = _all_parameter_hashes(bundle), capture_rng()
        readings = {"free": 70 * 2**30, "total": 80 * 2**30, "allocated": 8 * 2**30, "reserved": 9 * 2**30}
        with ExitStack() as stack:
            stack.enter_context(patch.object(train, "MEMORY_PROBE_ACTIONS", 8))
            stack.enter_context(patch.object(train, "MEMORY_PROBE_PROMPT", 12))
            stack.enter_context(patch.object(capacity, "memory", return_value=readings))
            for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(torch.cuda, name))
            reserved = stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=26 * 2**30))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=25 * 2**30))
            stack.enter_context(patch.object(train_v4, "_autocast", side_effect=lambda _: torch.autocast("cpu", dtype=torch.bfloat16)))
            stack.enter_context(patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("no optimizer")))
            result = train.measure_update_memory_v17(bundle, specimen, shape_bounds=bounds,
                                                      max_target_tokens=8, max_input_tokens=12)
            self.assertEqual(result["shared_t1_a1_pairs"], 1)
            self.assertEqual(len(result["f0_positions"]), 8)
            self.assertEqual(result["extra_reserved_gib"], 17.)
            self.assertGreater(result["gradient_norms"]["V"], 0.)
            self.assertEqual(result["optimizer_steps"], 0)
            reserved.return_value = 46 * 2**30
            with self.assertRaises(capacity.CapacityHold):
                train.measure_update_memory_v17(bundle, specimen, shape_bounds=bounds,
                                                 max_target_tokens=8, max_input_tokens=12)
        self.assertEqual(weights, _all_parameter_hashes(bundle))
        self.assertIs(incoming.grad, gradient)
        torch.testing.assert_close(rng["torch"], capture_rng()["torch"])

    def test_calibration_no_optimizer_state_change_or_rng_consumption(self):
        from fine_tuning.error_note.train import _all_parameter_hashes
        from fine_tuning.notebook_rl.checkpoint import capture_rng
        bundle = tiny_factory()
        rows = [row(bundle), row(bundle, kind="A1")]
        definition = contract(bundle, rows)
        parameter = bundle.params["L"][0]
        parameter.grad = torch.ones_like(parameter)
        incoming = parameter.grad
        rng, before = capture_rng(), _all_parameter_hashes(bundle)
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("no optimizer")):
            result = train.calibrate_update_v17(bundle, rows, definition)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(all(len(value["samples"]) == 3 for value in result["measurements"].values()))
        self.assertEqual(before, _all_parameter_hashes(bundle))
        self.assertIs(parameter.grad, incoming)
        torch.testing.assert_close(rng["torch"], capture_rng()["torch"])


class V17EncodingTests(unittest.TestCase):
    def encode(self, target="hello", kind="A1", terminal=256, spans=None):
        tokenizer = copy.copy(TOKENS)
        tokenizer.convert_tokens_to_ids = lambda value: {"<|im_end|>": 256, "<|endoftext|>": 257}.get(value)
        processor = SimpleNamespace(tokenizer=tokenizer)
        actions = tokenizer.encode(target) + ([terminal] if terminal is not None else [])
        mask = [True] * len(target.encode()) + ([False] if terminal is not None else [])
        if kind == "N":
            mask = train._sample_mask(tokenizer, actions, target, {256, 257}, spans)
        value = {"schema": train.ROW_SCHEMA, "kind": kind, "route": ROUTE_LM, "problem_id": "encoder",
                 "prompt": "question", "target": target, "loss_token_mask": mask,
                 "training_prompt_sha256": hashlib.sha256(b"question").hexdigest(),
                 "generation": {"text": target, "completion_token_ids": actions},
                 "mask_mode": "verified-corrected-value-content-only-v1" if kind == "N" else "all-non-eos"}
        if kind == "N":
            value.update(verified_byte_spans=spans, verified_content_token_mask=mask)
        value["loss_mask"] = mask
        bind_sft(value)
        prepared = {"inputs": {"input_ids": torch.tensor([[7, 8]])}, "prompt_token_ids": [7, 8], "input_sha256": "prepared"}
        with patch.object(train, "_prepare_row_inputs", return_value=prepared), \
                patch.object(train, "encoded_input_sha256_v17", return_value="prepared"):
            result = train.encode_v17_row(processor, SimpleNamespace(), value, [])
        return result

    def test_sampled_ids_preserved_both_eos_masked_and_utf8_spans_recomputed(self):
        for end in (256, 257, None):
            result = self.encode(terminal=end)
            self.assertEqual(result["input_ids"][-result["target_len"]:].tolist(), list(b"hello") + ([end] if end else []))
            if end:
                self.assertFalse(bool(result["loss_mask"][-1]))
        result = self.encode("α=7", kind="N", spans=[[3, 4]])
        self.assertEqual(result["loss_mask"].tolist(), [False, False, False, True, False])

    def test_span_boundary_crossing_token_is_excluded(self):
        tokenizer = copy.copy(TOKENS)
        tokenizer.convert_ids_to_tokens = lambda _: ["AA", "A" * 2999, "<|im_end|>"]
        mask = train._sample_mask(tokenizer, [1, 2, 256], "A" * 3001, {256, 257}, [[1, 3001]])
        self.assertEqual(mask, [False, True, False])

    def test_sampled_control_or_reencoded_bytes_rejected(self):
        with self.assertRaises(ValueError):
            self.encode("hello", terminal=999)
        with self.assertRaises(ValueError):
            self.encode("<|im_end|>")


if __name__ == "__main__":
    unittest.main()
