"""CPU gradients for v18's source-grounded masks and actual note-bearing optimizer step."""
import copy
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fine_tuning.error_note import rl, single, train_v17, train_v18 as train
from fine_tuning.error_note.contract import ROUTE_LM
from fine_tuning.notebook_rl.input_preparation import digest
from fine_tuning.tests.test_error_note_train import tiny_factory
from fine_tuning.tests.test_error_note_v16b_loss import TOKENS
from fine_tuning.tests.test_error_note_v17_loss import bind_sft, contract as old_contract, gradients, row as old_row


def bundle_with_bytes():
    bundle = tiny_factory()
    tokenizer = copy.copy(TOKENS)
    tokenizer.special = {2: "<|im_end|>", 31: "<|endoftext|>"}
    tokenizer.all_special_tokens = list(tokenizer.special.values())
    tokenizer.pad_token_id = 0
    bundle.processor = SimpleNamespace(tokenizer=tokenizer)
    return bundle


def row(bundle, index=0, kind="RL", *, grounded=False, used=False, version=1, advantage=1.):
    value = old_row(bundle, index, kind, ROUTE_LM, advantage, version)
    value.update(schema=train.ROW_SCHEMA, target="".join(map(chr, [12, 13, 14, 15, 16])), end_token_ids=[2, 31],
                 generation_prompt_sha256=hashlib.sha256(value["prompt"].encode()).hexdigest())
    if kind != "PREP_HUMAN":
        value["generation"]["text"] = value["target"]
    if kind in train_v17.SFT_KINDS:
        bind_sft(value)
        receipt = value["verification_receipt"]
        if kind in {"A1", "C"}:
            spans = [[0, 2]] if grounded else []
            use_spans = [[1, 2]] if used else []
            value.update(process_evidence_byte_spans=spans,
                         verified_process_token_mask=[grounded, grounded, False, False, False, False],
                         corrected_use_byte_spans=use_spans,
                         corrected_use_token_mask=[False, used, False, False, False, False])
            receipt.update(evidence_grade="SOURCE_GROUNDED" if grounded else "RESULT_ONLY",
                           verified_evidence_count=int(grounded), verified_process_byte_spans=spans,
                           corrected_use_byte_spans=use_spans)
        receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
        value.update(admission_receipt_sha256=receipt["sha256"], verifier_receipt_sha256=receipt["sha256"])
    return value


def contract(bundle, rows, **kwargs):
    return old_contract(bundle, rows, **kwargs) | {"version": "error-note-v18", "row_schema": train.ROW_SCHEMA,
        "sft_weights": train.SFT_WEIGHTS, "process_weights": train.PROCESS_WEIGHTS}


def localized(value, reason="FORMAT_ONLY_CERTIFIED"):
    spans, mask = [[1, 3]], [False, True, True, False, False, False]
    value.update(pg_mask_reason=reason, pg_byte_spans=spans, loss_mask=torch.tensor(mask))
    body = {"target_sha256": hashlib.sha256(value["target"].encode()).hexdigest(),
            "mask_reason": reason, "byte_spans": spans, "token_mask": mask,
            "output_binding_sha256": digest({"generation": value["generation"], "text": value["target"],
                                             "prompt_sha256": value["generation_prompt_sha256"]})}
    value["policy_mask_receipt"] = body | {"sha256": digest(body)}
    body = {"accepted": True, "content_equivalent": True, "byte_spans": spans,
            "output_sha256": body["target_sha256"], "output_binding_sha256": body["output_binding_sha256"]}
    value["format_receipt"] = body | {"sha256": digest(body)}
    return value


class V18LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_a1_and_c_weights_match_independent_raw_gradient_reference(self):
        for kind, grounded, used in [("A1", False, False), ("A1", True, False),
                                     ("C", False, False), ("C", True, True), ("N", False, False)]:
            with self.subTest(kind=kind, grounded=grounded):
                bundle = bundle_with_bytes()
                candidate = row(bundle, kind=kind, grounded=grounded, used=used)
                rows = [row(bundle), candidate] if kind == "A1" else [candidate]
                report = train.backward_v18_block(bundle, rows, contract(bundle, rows), components={kind})
                actual = gradients(bundle)
                bundle.model.zero_grad(set_to_none=True)
                raw = train_v17.v17_row_logprobs(bundle, [candidate])[0]
                coefficient = .18 if kind == "A1" and grounded else train.SFT_WEIGHTS[kind]
                manual = -coefficient * raw[candidate["loss_mask"]].mean() / 16
                if grounded and kind == "A1":
                    manual -= .02 * raw[:2].mean() / 16
                elif used:
                    manual -= .05 * raw[1] / 16
                manual.backward()
                self.assertAlmostEqual(report["loss"], float(manual.detach()), places=7)
                for name, expected in gradients(bundle).items():
                    torch.testing.assert_close(actual[name], expected, atol=1e-7, rtol=1e-5)
                self.assertEqual(report["sft"][kind]["parent_denominator"], 16)

    def test_process_and_note_mask_tampering_is_rejected_before_backward(self):
        for kind, field in [("A1", "verified_process_token_mask"), ("C", "corrected_use_token_mask"),
                            ("N", "verified_content_token_mask")]:
            bundle = bundle_with_bytes()
            candidate = row(bundle, kind=kind, grounded=True, used=kind == "C")
            candidate[field][4] = not candidate[field][4]
            rows = [row(bundle), candidate] if kind == "A1" else [candidate]
            with self.assertRaises(ValueError):
                train.backward_v18_block(bundle, rows, contract(bundle, rows))
            self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))

    def test_format_penalty_keeps_only_certified_tokens_after_probability_filter(self):
        bundle = bundle_with_bytes()
        candidate = localized(row(bundle, advantage=-1.))
        definition = contract(bundle, [candidate])
        with patch.object(rl, "negative_token_mask", return_value=torch.tensor([True, False, True, True, True, True])):
            report = train.backward_v18_block(bundle, [candidate], definition, components={"RL"})
        actual = gradients(bundle)
        bundle.model.zero_grad(set_to_none=True)
        value = train_v17.v17_row_logprobs(bundle, [candidate])[0]
        manual = value[2] / (16 * 1024)
        manual.backward()
        for name, expected in gradients(bundle).items():
            torch.testing.assert_close(actual[name], expected, atol=1e-7, rtol=1e-5)
        self.assertEqual(report["policy_mask_counts"]["FORMAT_ONLY_CERTIFIED"],
                         {"candidate_tokens": 2, "retained_tokens": 1, "negative_tokens": 1})

    def test_ungraded_policy_rows_are_excluded_with_an_all_false_mask(self):
        bundle = bundle_with_bytes()
        obs = row(bundle, 0, kind="OBS") | {"pg": False, "advantage": 0.}
        obs["loss_mask"] = torch.zeros_like(obs["loss_mask"])
        empty = localized(row(bundle, 1, advantage=0.))
        empty.update(pg=False, loss_mask=torch.zeros_like(empty["loss_mask"]))
        rows = [obs, empty]
        train.backward_v18_block(bundle, rows, contract(bundle, rows), components={"RL"})
        bundle.model.zero_grad(set_to_none=True)
        kept = row(bundle, 0, kind="OBS") | {"pg": False, "advantage": 0.}  # all-True mask on an unscored row
        with self.assertRaisesRegex(ValueError, "ordinary policy mask"):
            train.backward_v18_block(bundle, [kept], contract(bundle, [kept]), components={"RL"})

    def test_ordinary_rows_with_no_mask_reason_keep_the_report_sealable(self):
        bundle = bundle_with_bytes()
        plain = row(bundle, 0) | {"pg_mask_reason": None}
        candidate = localized(row(bundle, 1, advantage=-1.))
        report = train.backward_v18_block(bundle, [plain, candidate], contract(bundle, [plain, candidate]),
                                          components={"RL"})
        self.assertNotIn(None, report["policy_mask_counts"])
        self.assertIn("FORMAT_ONLY_CERTIFIED", report["policy_mask_counts"])
        digest(report["policy_mask_counts"])

    def test_partial_policy_mask_needs_a_content_and_token_bound_receipt(self):
        bundle = bundle_with_bytes()
        candidate = localized(row(bundle, advantage=-1.))
        candidate["format_receipt"]["content_equivalent"] = False
        with self.assertRaisesRegex(ValueError, "content certification"):
            train.backward_v18_block(bundle, [candidate], contract(bundle, [candidate]))
        candidate = row(bundle, advantage=-1.)
        candidate["loss_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "ordinary policy mask"):
            train.backward_v18_block(bundle, [candidate], contract(bundle, [candidate]))

    def test_actual_optimizer_step_records_signed_n_gradient_and_delta(self):
        bundle = bundle_with_bytes()
        candidate = row(bundle, kind="N")
        definition = contract(bundle, [candidate]) | {"global_slot": 1, "prep_updates": 0,
                    "rl_blocks": 1, "total_slots": 4}
        optimizer = torch.optim.SGD([parameter for ps in bundle.params.values() for parameter in ps], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        cursor = {"slot": 1, "updates": 0, "policy_sha256": rl.policy_sha256(bundle), "consumed_problem_ids": []}
        result = single.run_joint_single_update(bundle, optimizer, scheduler, [candidate], definition,
                                               cursor=cursor, block_ids=[], allow_update=True)
        evidence = result["stats"]["note_update_alignment"]
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertTrue(evidence["gradient_measured"] and evidence["optimizer_step"])
        self.assertGreater(evidence["signed_n_gradient_norm"], 0)
        self.assertGreater(evidence["combined_gradient_dot"], 0)
        self.assertLess(evidence["parameter_delta_dot"], 0)
        self.assertTrue(evidence["locally_descends_note_loss"])
        self.assertEqual(evidence["parent_ids"], [candidate["problem_id"]])
        self.assertEqual(result["stats"]["consumed_row_ids"], [candidate["row_id"]])
        self.assertTrue(all(parameter.grad is None for parameter in bundle.model.parameters()))

    def test_external_hold_explicitly_settles_no_rows_and_has_no_measured_note_gradient(self):
        bundle = bundle_with_bytes()
        definition = contract(bundle, []) | {"global_slot": 1, "prep_updates": 0,
                    "rl_blocks": 1, "total_slots": 4, "external_hold": "POLICY_DRIFT"}
        optimizer = torch.optim.SGD([parameter for ps in bundle.params.values() for parameter in ps], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        cursor = {"slot": 1, "updates": 0, "policy_sha256": rl.policy_sha256(bundle), "consumed_problem_ids": []}
        before = cursor["policy_sha256"]
        result = single.run_joint_single_update(bundle, optimizer, scheduler, [], definition,
                                               cursor=cursor, block_ids=[], allow_update=True)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["stats"]["consumed_row_ids"], [])
        self.assertFalse(result["stats"]["note_update_alignment"]["gradient_measured"])
        self.assertIsNone(result["stats"]["note_update_alignment"]["parameter_delta_dot"])
        self.assertEqual(cursor["policy_sha256"], before)

    def test_calibration_binds_v18_identity_and_leaves_parameters_unchanged(self):
        bundle = bundle_with_bytes()
        rows = [row(bundle), row(bundle, kind="A1", grounded=True)]
        before = rl.policy_sha256(bundle)
        report = train.calibrate_update_v18(bundle, rows, contract(bundle, rows))
        self.assertEqual(report["version"], "v17-update-calibration-v1")
        self.assertEqual(report["objective_version"], "error-note-v18")
        self.assertEqual(report["row_schema"], train.ROW_SCHEMA)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(rl.policy_sha256(bundle), before)
        self.assertEqual(report["sha256"], digest({key: value for key, value in report.items() if key != "sha256"}))

    def test_a_delayed_row_cannot_be_consumed_twice_by_the_optimizer(self):
        bundle = bundle_with_bytes()
        candidate = row(bundle, kind="N")
        definition = contract(bundle, [candidate]) | {"global_slot": 1, "prep_updates": 0,
                    "rl_blocks": 1, "total_slots": 4}
        optimizer = torch.optim.SGD([parameter for ps in bundle.params.values() for parameter in ps], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        cursor = {"slot": 1, "updates": 0, "policy_sha256": rl.policy_sha256(bundle),
                  "consumed_problem_ids": [], "consumed_row_ids": [candidate["row_id"]]}
        with self.assertRaisesRegex(ValueError, "repeated consumed"):
            single.run_joint_single_update(bundle, optimizer, scheduler, [candidate], definition,
                                           cursor=cursor, block_ids=[], allow_update=True)
        self.assertEqual(cursor["updates"], 0)


class V18EncodingTests(unittest.TestCase):
    def encode(self, target="α=7", kind="A1", spans=None, *, mask_override=None):
        tokenizer = copy.copy(TOKENS)
        actions = tokenizer.encode(target) + [256]
        full = [True] * len(target.encode()) + [False]
        selected = train_v17._sample_mask(tokenizer, actions, target, {256, 257}, spans) if spans else [False] * len(actions)
        mask = selected if kind == "N" else full
        candidate = {"schema": train.ROW_SCHEMA, "kind": kind, "route": ROUTE_LM, "problem_id": "encoder",
                     "prompt": "question", "target": target, "generation": {"text": target, "completion_token_ids": actions},
                     "training_prompt_sha256": hashlib.sha256(b"question").hexdigest(),
                     "loss_mask": mask, "loss_token_mask": mask,
                     "mask_mode": "verified-corrected-value-content-only-v1" if kind == "N" else "all-non-eos"}
        if kind == "N":
            candidate.update(verified_byte_spans=spans, verified_content_token_mask=mask)
        bind_sft(candidate)
        if kind in {"A1", "C"}:
            candidate.update(process_evidence_byte_spans=spans or [], verified_process_token_mask=selected,
                             corrected_use_byte_spans=[], corrected_use_token_mask=[False] * len(actions))
            receipt = candidate["verification_receipt"]
            receipt.update(evidence_grade="SOURCE_GROUNDED" if spans else "RESULT_ONLY",
                           verified_evidence_count=int(bool(spans)), verified_process_byte_spans=spans or [],
                           corrected_use_byte_spans=[])
            receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
            candidate.update(admission_receipt_sha256=receipt["sha256"], verifier_receipt_sha256=receipt["sha256"])
        if mask_override is not None:
            candidate["verified_process_token_mask"] = mask_override
        prepared = {"inputs": {"input_ids": torch.tensor([[7, 8]])}, "prompt_token_ids": [7, 8], "input_sha256": "prepared"}
        with patch.object(train_v17, "_prepare_row_inputs", return_value=prepared), \
                patch.object(train_v17, "encoded_input_sha256_v17", return_value="prepared"):
            return train.encode_v18_row(SimpleNamespace(tokenizer=tokenizer), SimpleNamespace(), candidate, [])

    def test_source_and_note_masks_bind_actual_utf8_actions_and_exclude_terminal(self):
        for kind in ("A1", "C", "N"):
            with self.subTest(kind=kind):
                result = self.encode(kind=kind, spans=[[3, 4]])
                self.assertEqual(result["schema"], train.ROW_SCHEMA)
                self.assertEqual(result["input_ids"][-5:].tolist(), [206, 177, 61, 55, 256])
                selected = result["loss_mask"].tolist() if kind == "N" else result["verified_process_token_mask"]
                self.assertEqual(selected, [False, False, False, True, False])
                self.assertFalse(bool(result["loss_mask"][-1]))

    def test_source_mask_cannot_include_unverified_or_terminal_tokens(self):
        with self.assertRaisesRegex(ValueError, "source-grounded process mask"):
            self.encode(spans=[[3, 4]], mask_override=[True, False, False, True, True])


if __name__ == "__main__":
    unittest.main()
