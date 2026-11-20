"""Real tiny Qwen3-VL PolicyBundle checks for the v16b joint objective."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from fine_tuning.error_note import rl, train_v16b as train, v16b_rows
from fine_tuning.error_note.contract import ROUTE_FULL, ROUTE_LM
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests.test_error_note_train import END, encoded, tiny_factory

SAMPLING = {"temperature": .7, "presence_penalty": 1.5, "top_p": 1., "top_k": 0}


def row(bundle, index=0, kind="RL", route=ROUTE_LM, advantage=1., version=1):
    item = encoded(f"{kind}-{index}", target=(12, 13, 14, 15, 16), kind=kind, route=route)
    if route == ROUTE_LM:
        item["input_ids"] = torch.tensor([7, 8, 12, 13, 14, 15, 16, END])
        item["pixel_values"] = item["image_grid_thw"] = None
    item.update(schema=train.ROW_SCHEMA, problem_id=f"problem-{index}", end_token_id=END,
                loss_mask=torch.tensor([True, True, True, True, True, False]),
                block_index=version - ({"N": 1, "C": 3}.get(kind, 0)))
    actions = item["input_ids"][-item["target_len"]:].tolist()
    item["generation"] = {"completion_token_ids": actions, "probability_space": "processed-full-vocab-v1",
                          "sampling_sha256": rl.sampling_sha256(SAMPLING), "finish_reason": "stop",
                          "policy_version": version if kind == "N" else max(0, version - 1)}
    if kind == "PREP_SELF":
        item["generation"]["probability_space"] = "processed-truncated-b2"
    elif kind == "PREP_HUMAN":
        del item["generation"]
    if kind in train.PG_KINDS:
        reward = 1. if advantage > 0 else 0.
        baseline = reward - advantage
        receipt = {"version": "test-baseline-v1", "problem_id": item["problem_id"], "baseline": baseline,
                   "history_cutoff": item["block_index"] - 2}
        receipt["sha256"] = digest(receipt)
        item.update(pg=True, reward=reward, baseline=baseline, advantage=advantage, baseline_receipt=receipt)
        bundle.route.apply(route == ROUTE_FULL)
        with torch.no_grad():
            item["generation"]["sampling_logprobs"] = train.v16b_row_logprobs(bundle, [item])[0].tolist()
    return item


def contract(bundle, rows, *, version=1, phase="rl"):
    def ids(kind):
        return [r["problem_id"] for r in rows if r["kind"] == kind]
    return {"version": "error-note-v16b", "sampling": SAMPLING, "policy_sha256": rl.policy_sha256(bundle),
            "phase": phase, "block_id": version,
            "expected_problem_ids": ids("RL") if phase == "rl" else sorted({r["problem_id"] for r in rows}),
            "expected_obs_problem_ids": ids("OBS"), "note_problem_ids": ids("N"), "resolve_problem_ids": ids("C"),
            "baseline_receipts": {f'{r["kind"]}:{r["problem_id"]}': r["baseline_receipt"]
                                  for r in rows if r["kind"] in train.PG_KINDS},
            "reward_values": {f'{r["kind"]}:{r["problem_id"]}': {"reward": r["reward"], "pg": r["pg"]}
                              for r in rows if r["kind"] in train.PG_KINDS}}


def gradients(bundle):
    return {name: value.grad.clone() for name, value in bundle.model.named_parameters() if value.grad is not None}


class JointLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_rl_matches_manual_gradient_fixed_sixteen_denominator_and_detached_is(self):
        bundle = tiny_factory()
        rows = [row(bundle)]
        rows[0]["generation"]["sampling_logprobs"] = [x - .1 for x in rows[0]["generation"]["sampling_logprobs"]]
        report = train.backward_v16b_block(bundle, rows, contract(bundle, rows), components=("RL",))
        actual = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        value = train.v16b_row_logprobs(bundle, rows)[0]
        weights = (value.detach() - value.new_tensor(rows[0]["generation"]["sampling_logprobs"])).exp().clamp(max=2)
        manual = -(weights * value).sum() / (16 * 1024)
        manual.backward()
        self.assertAlmostEqual(report["loss"], float(manual.detach()), places=7)
        for name, gradient in gradients(bundle).items():
            torch.testing.assert_close(actual[name], gradient, atol=1e-7, rtol=1e-5)
        self.assertEqual(report["rl"]["denominator"], 16 * 1024)
        self.assertTrue(report["task_signal"] and report["has_gradient"])

    def test_microbatch_partition_preserves_joint_gradients_and_full_vision_route(self):
        bundle = tiny_factory()
        rows = [row(bundle, kind="RL", route=ROUTE_FULL), row(bundle, 1, kind="OBS", route=ROUTE_FULL),
                row(bundle, 2, kind="N", route=ROUTE_FULL), row(bundle, 3, kind="C", version=4)]
        # C participates in update 4; all other schedule fields are adjusted to that same update.
        for item in rows[:3]:
            item["block_index"] += 3
            item["generation"]["policy_version"] += 3
            if item["kind"] in train.PG_KINDS:
                item["baseline_receipt"]["history_cutoff"] += 3
                item["baseline_receipt"]["sha256"] = digest({k: v for k, v in item["baseline_receipt"].items() if k != "sha256"})
        definition = contract(bundle, rows, version=4)
        a = train.backward_v16b_block(bundle, rows, definition, token_budget=0)
        actual = gradients(bundle)
        self.assertGreater(a["gradient_norms"]["V"], 0)
        bundle.model.zero_grad(set_to_none=True)
        b = train.backward_v16b_block(bundle, rows, definition, token_budget=12000)
        self.assertAlmostEqual(a["loss"], b["loss"], places=6)
        for name, gradient in gradients(bundle).items():
            torch.testing.assert_close(actual[name], gradient, atol=3e-7, rtol=3e-4)

    def test_hold_erases_sft_and_pg_gradients_and_restores_mode_route(self):
        bundle = tiny_factory()
        bundle.model.eval()
        rows = [row(bundle, kind="N"), row(bundle, 1)]
        rows[1]["generation"]["sampling_logprobs"] = [x - .8 for x in rows[1]["generation"]["sampling_logprobs"]]
        report = train.backward_v16b_block(bundle, rows, contract(bundle, rows))
        self.assertTrue(report["hold"])
        self.assertFalse(report["task_signal"] or report["has_gradient"])
        self.assertTrue(report["sft"]["N"]["weighted_loss"] > 0)
        self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))
        self.assertFalse(bundle.model.training)

    def test_negative_row_drift_excludes_only_pg_and_clip_threshold_holds_block(self):
        bundle = tiny_factory()
        item = row(bundle, advantage=-1.)
        item["generation"]["sampling_logprobs"] = [x + .4 for x in item["generation"]["sampling_logprobs"]]
        report = train.backward_v16b_block(bundle, [item], contract(bundle, [item]), components=("RL",))
        self.assertEqual(report["importance"]["row_excluded"], 1)
        self.assertFalse(report["hold"] or report["has_gradient"])
        item = row(bundle)
        # One clipped token of six exceeds 10%, while mean absolute log-ratio stays below .5.
        item["generation"]["sampling_logprobs"][0] -= 1.
        report = train.backward_v16b_block(bundle, [item], contract(bundle, [item]), components=("RL",))
        self.assertEqual(report["hold_reason"], "IS_CLIP")

    def test_preparation_uses_processed_self_raw_human_and_excludes_eos(self):
        bundle = tiny_factory()
        rows = [row(bundle, kind="PREP_SELF"), row(bundle, 1, kind="PREP_HUMAN")]
        report = train.backward_v16b_block(bundle, rows, contract(bundle, rows, phase="prep"),
                                          components=("PREP_SELF", "PREP_HUMAN"))
        bundle.model.zero_grad(set_to_none=True)
        values = train.v16b_row_logprobs(bundle, rows)
        manual = sum(-value[item["loss_mask"]].mean() / 2 for item, value in zip(rows, values, strict=True))
        self.assertAlmostEqual(report["loss"], float(manual.detach()), places=6)
        self.assertFalse(report["importance"]["tokens"])

    def test_anchor_reference_forward_is_limited_to_first_four_actions(self):
        bundle = tiny_factory()
        rows = [row(bundle)]
        with torch.no_grad():
            for parameter in bundle.params["L"]:
                parameter.add_(.01)
        definition = contract(bundle, rows)
        from fine_tuning.error_note import train_v4
        original = train_v4._single_reference_hidden
        lengths = []

        def recording(bundle, prefixes):
            lengths.extend(r["target_len"] for r in prefixes)
            return original(bundle, prefixes)

        with patch.object(train_v4, "_single_reference_hidden", recording):
            report = train.backward_v16b_block(bundle, rows, definition, components=("ANCHOR",))
        self.assertEqual(lengths, [4])
        self.assertGreater(report["rl"]["anchor"], 0)
        self.assertFalse(report["task_signal"])

    def test_tampered_advantage_missing_membership_and_wrong_note_version_fail(self):
        bundle = tiny_factory()
        rows = [row(bundle), row(bundle, 1, kind="N")]
        definition = contract(bundle, rows)
        for mutate in (lambda x: x[0].update(advantage=.3),
                       lambda x: x.pop(0),
                       lambda x: x[1]["generation"].update(policy_version=0)):
            changed = copy.deepcopy(rows)
            mutate(changed)
            with self.assertRaises(ValueError):
                train.backward_v16b_block(bundle, changed, definition)
            self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))

    def test_schema_dispatch_encodes_real_prepared_input_and_excludes_end_from_human_sft(self):
        from PIL import Image
        from fine_tuning.error_note.train_v4 import encode_single_row
        from fine_tuning.notebook_rl.input_preparation import prepare_protocol_inputs
        from fine_tuning.notebook_rl.prompts import user_message
        from fine_tuning.tests.test_error_note_train import tiny_processor

        bundle, processor = tiny_factory(), tiny_processor()
        images = [Image.new("RGB", (8, 8))]
        image = {"min_pixels": 64, "max_pixels": 64}
        prepared, _ = prepare_protocol_inputs(processor, bundle.model.config,
                                              [user_message("question", 1)], images, image)
        actions = processor.tokenizer.encode("a b", add_special_tokens=False) + [
            processor.tokenizer.convert_tokens_to_ids("<|im_end|>")]
        raw = {"schema": train.ROW_SCHEMA, "kind": "RL", "problem_id": "encoded", "route": ROUTE_FULL,
               "prompt": "question", "target": "a b",
               "generation": {"text": "a b", "completion_token_ids": actions,
                              "prompt_token_ids": prepared["prompt_token_ids"],
                              "input_sha256": prepared["input_sha256"]}}
        encoded_row = encode_single_row(processor, bundle.model.config, raw, images, image=image)
        self.assertEqual(encoded_row["input_ids"][-len(actions):].tolist(), actions)
        self.assertEqual(encoded_row["route"], ROUTE_FULL)
        human = {k: v for k, v in raw.items() if k != "generation"} | {"kind": "PREP_HUMAN"}
        encoded_human = encode_single_row(processor, bundle.model.config, human, images, image=image)
        self.assertEqual(encoded_human["loss_mask"].tolist(), [True, True, False])
        empty = copy.deepcopy(raw)
        empty["target"] = empty["generation"]["text"] = ""
        empty["generation"]["completion_token_ids"] = [actions[-1]]
        self.assertEqual(encode_single_row(processor, bundle.model.config, empty, images, image=image)["target_len"], 1)
        changed = copy.deepcopy(raw)
        changed["generation"]["input_sha256"] = "changed"
        with self.assertRaises(ValueError):
            encode_single_row(processor, bundle.model.config, changed, images, image=image)

    def test_memory_probe_runs_real_backward_restores_every_state_and_refuses_an_unfit_budget(self):
        from contextlib import ExitStack
        from fine_tuning.error_note import capacity
        from fine_tuning.error_note.protocol import ARM, plain_protocol
        from fine_tuning.error_note.train import _all_parameter_hashes
        from fine_tuning.notebook_rl.checkpoint import capture_rng
        from fine_tuning.notebook_rl.policy import configure_policy
        from fine_tuning.tests.test_training import tiny_model

        bundle = configure_policy(tiny_model().to(dtype=torch.bfloat16), None, ARM,
                                  plain_protocol("bf16"), enforce_counts=False)
        bundle.route.apply(False)
        bundle.model.eval()
        specimen = encoded(kind="RL", route=ROUTE_FULL)
        bounds = {"max_images": 1, "max_vision_patches": 16, "max_prompt_tokens": 12,
                  "dataset_sha256": "f" * 64}
        incoming = bundle.params["L"][0]
        incoming.grad = torch.ones_like(incoming)
        gradient = incoming.grad
        weights = _all_parameter_hashes(bundle)
        rng = capture_rng()
        readings = {"free": 70 * 2**30, "total": 80 * 2**30, "allocated": 8 * 2**30,
                    "reserved": 9 * 2**30}
        from fine_tuning.error_note import train_v4
        with ExitStack() as stack:
            stack.enter_context(patch.object(train, "MEMORY_PROBE_ACTIONS", 8))
            stack.enter_context(patch.object(train, "MEMORY_PROBE_PROMPT", 12))
            stack.enter_context(patch.object(capacity, "memory", return_value=readings))
            for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(torch.cuda, name))
            reserved = stack.enter_context(patch.object(torch.cuda, "max_memory_reserved", return_value=26 * 2**30))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=25 * 2**30))
            stack.enter_context(patch.object(train_v4, "_autocast", side_effect=lambda _: torch.autocast("cpu", dtype=torch.bfloat16)))
            stack.enter_context(patch.object(torch.optim, "AdamW", side_effect=AssertionError("no optimizer in probe")))
            report = train.measure_update_memory_v16b(bundle, specimen, shape_bounds=bounds,
                                                       max_target_tokens=8, max_input_tokens=12)
            self.assertEqual(report["extra_reserved_gib"], 17.)
            self.assertEqual(report["kv_cache_gib"], 24.)
            self.assertEqual(report["prompt_tokens"], 12)
            self.assertEqual(report["action_tokens"], 8)
            self.assertGreater(report["gradient_norms"]["V"], 0)
            self.assertEqual(report["optimizer_steps"], 0)
            reserved.return_value = 46 * 2**30
            with self.assertRaisesRegex(capacity.CapacityHold, "less than 16"):
                train.measure_update_memory_v16b(bundle, specimen, shape_bounds=bounds,
                                                max_target_tokens=8, max_input_tokens=12)
        self.assertEqual(weights, _all_parameter_hashes(bundle))
        self.assertIs(incoming.grad, gradient)
        self.assertFalse(bundle.model.training or bundle.route.vision_on)
        after = capture_rng()
        self.assertEqual(rng["python"], after["python"])
        torch.testing.assert_close(rng["torch"], after["torch"])
        torch.testing.assert_close(rng["numpy"]["keys"], after["numpy"]["keys"])
        with self.assertRaisesRegex(ValueError, "production-length"):
            train.measure_update_memory_v16b(bundle, specimen, shape_bounds=bounds,
                                            max_target_tokens=8, max_input_tokens=12)

    def test_measured_single_full_shape_splits_batches_without_changing_loss_or_gradients(self):
        bundle = tiny_factory()
        rows = [row(bundle, i, route=ROUTE_FULL) for i in range(2)]
        definition = contract(bundle, rows)
        unrestricted = train.backward_v16b_block(bundle, rows, definition, token_budget=12000)
        actual = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        bounded = definition | {"memory_shape_bounds": {
            "max_images": 1, "max_vision_patches": 16, "max_prompt_tokens": 8}}
        report = train.backward_v16b_block(bundle, rows, bounded, token_budget=12000)
        self.assertEqual(report["microbatches"], [[0], [1]])
        self.assertAlmostEqual(report["loss"], unrestricted["loss"], places=6)
        for name, gradient in gradients(bundle).items():
            torch.testing.assert_close(actual[name], gradient, atol=3e-7, rtol=3e-4)
        bundle.model.zero_grad(set_to_none=True)
        bounded["memory_shape_bounds"]["max_vision_patches"] = 15
        with self.assertRaisesRegex(ValueError, "exceeds the measured shape"):
            train.backward_v16b_block(bundle, rows, bounded, token_budget=12000)
        self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))


class _ByteTokenizer:
    """Byte tokens plus both Qwen EOS controls: 256 <|im_end|>, 257 <|endoftext|>."""
    special = {256: "<|im_end|>", 257: "<|endoftext|>"}
    all_special_tokens = list(special.values())
    unk_token_id = None
    byte_tokens = bytes_to_unicode()

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def convert_tokens_to_ids(self, token):
        return {value: key for key, value in self.special.items()}[token]

    def convert_ids_to_tokens(self, ids):
        return [self.special.get(value) or self.byte_tokens[value] for value in ids]


TOKENS = _ByteTokenizer()
PROSE = ("The displayed quantities describe separate groups whose sizes can be compared through their ratio. "
         "Before choosing an answer, identify the numerator and denominator from the question, preserve their "
         "original roles, perform the division, and inspect the resulting units for consistency. "
         "This reasoning connects the available evidence to a single numerical conclusion.")
LESSON = "Recompute the division before committing the result."
ERROR = {"earlier_span": [0, 10], "earlier_quote": "12 / 3 = 5", "kind": "ARITHMETIC", "operand_fact_refs": [],
         "operation": "DIV", "old_value": "5", "corrected_value": "4", "rounding": "none", "unit": "",
         "rule_id": "DIV"}


def sampled(text, *, status="INCORRECT", extracted="5", actions=None):
    actions = actions or TOKENS.encode(text) + [256]
    generation = {"text": text, "completion_token_ids": actions, "finish_reason": "stop", "policy_version": 0,
                  "sampling_logprobs": [-.5] * len(actions), "probability_space": "processed-full-vocab-v1"}
    return {"text": text, "status": status, "extracted": extracted, "prompt": "generation prompt",
            "completion_tokens": len(actions), "finish_reason": "stop", "generation": generation}


def note_record(note, *, gold="4", **outputs):
    return {"problem_id": "synthetic:1", "source": "creator", "image_paths": ["image.png"], "image_sha256": ["a" * 64],
            "p0_text": "Question: Divide the displayed counts.", "evaluation": {"gold": gold, "choices": []},
            "t1": sampled("12 / 3 = 5\nFinal answer: 5"), "note": sampled(json.dumps(note)), **outputs}


def verify(record, *, gold="4", facts=(), rules=None):
    return v16b_rows.verify_note_v16b(record, fact_catalog=list(facts), rules=rules or {}, choices=(), gold=gold)


def resolve(record, gold="4"):
    return v16b_rows.c_row_with_receipt(record, tokenizer=TOKENS, fact_catalog=[], rules={}, choices=(), gold=gold)


class V16bRowRepairs(unittest.TestCase):
    def test_string_lesson_and_claim_schema_slips_allow_resolve_but_train_nothing_unverified(self):
        lesson = "Percentages must be computed accurately; recheck each arithmetic step before answering."
        note = {"observed_facts": [], "error": ERROR | {"earlier_span": [0, 100], "rounding": 0}, "lesson": lesson}
        receipt = verify(note_record(note))
        self.assertTrue(receipt["resolve_allowed"], receipt)
        self.assertFalse(receipt["accepted"])
        self.assertEqual([(c["status"], c["reason"]) for c in receipt["claims"]], [("UNVERIFIABLE", "ERROR_SCHEMA")])
        self.assertEqual(receipt["verified_spans"], [])
        self.assertEqual(receipt["lesson"], {"text": lesson})
        self.assertIn("Abstract lesson: " + lesson, v16b_rows.build_resolve_prompt_v16b("P0", receipt["lesson"]))
        # Malformed observed facts stay untrained while the verified error still makes an N row.
        note = {"observed_facts": [{"fact_ref": "f0"}, "f1 = 3"], "error": ERROR, "lesson": LESSON}
        receipt = verify(note_record(note))
        self.assertTrue(receipt["accepted"] and receipt["resolve_allowed"], receipt)
        self.assertEqual([(c["status"], c.get("reason")) for c in receipt["claims"]],
                         [("UNVERIFIABLE", "OBSERVED_FACT_SCHEMA")] * 2 + [("VERIFIED", "SOURCE_BOUND_CORRECTION")])
        self.assertEqual(verify(note_record(note | {"observed_facts": {"f0": "12"}}))["claims"][0]["path"],
                         "observed_facts")
        # A contradicted claim, an over-long or leaking lesson, bad JSON and missing keys still block everything.
        blocked = [note | {"error": ERROR | {"corrected_value": "8"}}, note | {"lesson": "word " * 41},
                   note | {"lesson": "The answer is 4."}, {"error": None, "lesson": LESSON}]
        for candidate in blocked:
            receipt = verify(note_record(candidate))
            self.assertFalse(receipt["resolve_allowed"] or receipt["accepted"], receipt)
        record = note_record(note)
        record["note"] = sampled("lesson: recompute")
        self.assertFalse(verify(record)["resolve_allowed"])

    def test_truncated_f0_prior_counts_as_not_correct(self):
        prior = {"source_rates": {"s": .6}, "overall": .6}
        for status, expected in (("TRUNCATED", 0), ("INCORRECT", 0), ("CORRECT", 1)):
            receipt = v16b_rows.baseline_receipt_v16b({"problem_id": "p", "source": "s", "f0_prior": {"status": status}},
                                                      [], prior=prior, history_cutoff=-2)
            self.assertEqual(receipt["f0_prior"], expected)
            self.assertAlmostEqual(receipt["baseline"], (2 * .6 + expected) / 3)

    def test_endoftext_terminated_self_rows_are_kept_and_other_control_tokens_reject_the_row(self):
        text = PROSE + "\n12 / 3 = 4\nConclusion: 4\nFinal answer: 4"
        good = sampled(text, status="CORRECT", extracted="4", actions=TOKENS.encode(text) + [257])
        note = {"observed_facts": [], "error": ERROR, "lesson": LESSON}
        record = note_record(note, resolve=good)
        record["note"] = sampled(json.dumps(note), actions=TOKENS.encode(json.dumps(note)) + [257])
        row, receipt = resolve(record)
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(row["loss_mask"], [True] * len(TOKENS.encode(text)) + [False])
        noted = v16b_rows.note_row(record, tokenizer=TOKENS, fact_catalog=[], rules={}, choices=(), gold="4")
        self.assertFalse(noted["loss_mask"][-1])
        self.assertTrue(any(noted["loss_mask"]))
        inner = sampled(text, status="CORRECT", extracted="4", actions=TOKENS.encode(text) + [256, 257])
        row, receipt = resolve(record | {"resolve": inner})
        self.assertIsNone(row)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "SFT_MASK:v16 train: generated control token")

    def test_graph_note_catalog_keeps_only_image_read_values(self):
        def contract(structure):
            return v16b_rows.native_note_contract({"problem_id": "graph:1", "layer": "graph",
                                                   "annotation": {"structure_text": structure, "directed": False}})
        weighted, plain = contract("4 2\n0 1 5\n1 2 7"), contract("4 2\n0 1\n1 2")
        self.assertEqual([f["fact_id"] for f in weighted["fact_catalog"]], ["edge_weight_0", "edge_weight_1"])
        self.assertEqual([f["fact_id"] for f in plain["fact_catalog"]], ["visible_node_count", "edge_count"])
        self.assertTrue(weighted["supported"] and plain["supported"])
        note = {"observed_facts": [{"fact_ref": "edge_0", "value": "present"}, {"fact_ref": "edge_weight_0", "value": "5"}],
                "error": None, "lesson": LESSON}
        receipt = verify(note_record(note), facts=weighted["fact_catalog"], rules=weighted["rules"])
        self.assertEqual([c["status"] for c in receipt["claims"]], ["UNVERIFIABLE", "VERIFIED"])
        self.assertEqual(receipt["verified"], 1)

    def test_multi_word_and_range_open_gold_resolves_are_accepted(self):
        note = {"observed_facts": [], "error": None, "lesson": LESSON}
        for gold in ("South Carolina", "53.6%-55.6%"):
            with self.subTest(gold=gold):
                record = note_record(note, gold=gold, resolve=sampled(PROSE + "\nFinal answer: " + gold,
                                                                      status="CORRECT", extracted=gold))
                row, receipt = resolve(record, gold)
                self.assertTrue(receipt["accepted"], receipt)
                self.assertEqual(row["prompt"], record["p0_text"])
                wrong = record | {"resolve": sampled(PROSE + "\nFinal answer: North Carolina",
                                                     status="CORRECT", extracted="North Carolina")}
                self.assertEqual(resolve(wrong, gold)[1]["reason"], "PARSE_GOLD_MISMATCH")


if __name__ == "__main__":
    unittest.main()
