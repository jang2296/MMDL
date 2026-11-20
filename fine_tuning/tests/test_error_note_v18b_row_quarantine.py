"""Mechanical replay fixtures exercise admission and accounting, without claiming self-correction."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from fine_tuning.error_note import campaign_v16b as campaign, rl, single, train_v17, v18b_rows
from fine_tuning.error_note.audit import digest
from fine_tuning.error_note.v16b_runtime import V16bScheduler
from fine_tuning.tests.test_error_note_v17_rows import output, verifier
from fine_tuning.tests.test_error_note_v18_loss import contract
from fine_tuning.tests.test_error_note_v18_rows import source_record
from fine_tuning.tests import test_error_note_v18b_note_evidence as note_evidence
from fine_tuning.error_note.contract import IMAGE


def mechanical_note_row(bundle, problem_id="fixture", *, block_index=0):
    """A reportable=false, synthetic NOTE in the same token/input form as vLLM output."""
    tokenizer = bundle.processor.tokenizer
    item = source_record()
    item.update(problem_id=problem_id, image_paths=[], image_sha256=[], block_index=block_index)
    private = verifier(item)

    def completion(text, prompt, status, version):
        prepared = train_v17._prepare_row_inputs(bundle.processor, bundle.model.config, prompt, [], IMAGE)
        actions = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
        result = output(text, prompt=prompt, status=status, actions=actions)
        result["generation"].update(prompt_token_ids=prepared["prompt_token_ids"], input_sha256=prepared["input_sha256"],
            policy_version=version, policy_sha256=rl.policy_sha256(bundle), completion_tokens=len(actions),
            probability_space="processed-full-vocab-v1", sampling_sha256=rl.sampling_sha256(train_v17.SAMPLING),
            engine="vllm", mixed_version=False, sampling_logprobs=[-0.1] * len(actions))
        return result

    item["t1"] = completion(item["t1"]["text"], item["p0_text"], "INCORRECT", block_index)
    ledger = v18b_rows.audit_t1_v17(item, verifier=private, tokenizer=tokenizer)
    request = v18b_rows.build_note_request_v17(item, ledgers=[ledger], gold="14", tokenizer=tokenizer)
    text = json.dumps({"repairs": [{"stage": "T1", "quote": "AB = 9", "corrected_value": "7", "unit": "length"}],
                       "lesson": "Inspect each printed label before calculating."})
    item["note"] = completion(text, request["prompt"], "NOTE", block_index + 1)
    receipt = v18b_rows.verify_note_v17(item, ledgers=[ledger], verifier=private, tokenizer=tokenizer, gold="14", choices=[])
    row = v18b_rows.n_row_v17(item, tokenizer=tokenizer, receipt=receipt)
    if row is None:
        raise AssertionError(receipt)
    return row, item


class RowQuarantine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.bundle = note_evidence.V18BNoteEvidence().bundle()
        self.prompt_ids = torch.tensor([7, 8])
        self.input_sha = train_v17.encoded_input_sha256_v17({"input_ids": self.prompt_ids, "target_len": 0})
        self.prepared = {"inputs": {"input_ids": self.prompt_ids[None]}, "prompt_token_ids": self.prompt_ids.tolist(),
                         "input_sha256": self.input_sha}
        self.prepare = patch.object(train_v17, "_prepare_row_inputs", return_value=self.prepared)
        self.prepare.start()
        self.addCleanup(self.prepare.stop)

    def specimens(self, count=3):
        pairs = [mechanical_note_row(self.bundle, f"fixture-{i}") for i in range(count)]
        return [row for row, _ in pairs], {item["problem_id"]: item for _, item in pairs}

    def test_one_bad_mask_is_quarantined_but_two_valid_notes_update_and_save(self):
        rows, items = self.specimens()
        rows[0]["loss_mask"] = [False] * len(rows[0]["loss_mask"])
        encoded, report = campaign._encode_update_rows(self.bundle, rows, items, IMAGE)
        self.assertEqual(report["status"], "CONTINUE")
        self.assertEqual([r["row_id"] for r in report["rejected_rows"]], [rows[0]["row_id"]])
        self.assertEqual(report["rejected_rows"][0]["source_row_sha256"], digest(rows[0]))
        self.assertEqual(len(encoded), 2)
        definition = contract(self.bundle, encoded) | {"version": "error-note-v18b", "global_slot": 1,
            "prep_updates": 0, "rl_blocks": 1, "total_slots": 4}
        optimizer = torch.optim.AdamW([p for ps in self.bundle.params.values() for p in ps], lr=.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        cursor = {"slot": 1, "updates": 0, "policy_sha256": rl.policy_sha256(self.bundle), "consumed_problem_ids": []}
        result = single.run_joint_single_update(self.bundle, optimizer, scheduler, encoded, definition,
            cursor=cursor, block_ids=[], allow_update=True)
        stats = result["stats"]
        self.assertFalse(stats["hold"])
        self.assertEqual(set(stats["consumed_row_ids"]), {rows[1]["row_id"], rows[2]["row_id"]})
        evidence = stats["note_update_alignment"]
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertLess(evidence["parameter_delta_dot"], 0)
        stats["row_quarantine"] = report
        runtime = object.__new__(V16bScheduler)
        runtime.v18, runtime.v18b, runtime.row_settlements = True, True, {}
        runtime.settle_rows(1, rows, stats)
        settlement = runtime.row_settlements["1"]
        self.assertEqual(settlement["discarded_row_ids"], [rows[0]["row_id"]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            torch.save({"model": self.bundle.model.state_dict(), "optimizer": optimizer.state_dict(),
                        "settlement": runtime.row_settlements, "n_update": evidence}, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["settlement"], runtime.row_settlements)
            self.assertEqual(saved["n_update"]["row_ids"], [row["row_id"] for row in encoded])

    def test_strict_majority_rejected_is_systematic_without_model_changes(self):
        rows, items = self.specimens()
        for row in rows[:2]:
            row["loss_mask"] = [False] * len(row["loss_mask"])
        before = rl.policy_sha256(self.bundle)
        _, report = campaign._encode_update_rows(self.bundle, rows, items, IMAGE)
        self.assertEqual(report["status"], "SYSTEMATIC_AUXILIARY_FAILURE")
        self.assertEqual(rl.policy_sha256(self.bundle), before)
        self.assertTrue(all(p.grad is None for ps in self.bundle.params.values() for p in ps))

    def test_exactly_half_is_allowed_but_mandatory_rl_or_obs_failure_is_fatal(self):
        rows, items = self.specimens(2)
        rows[0]["loss_mask"] = [False] * len(rows[0]["loss_mask"])
        self.assertEqual(campaign._encode_update_rows(self.bundle, rows, items, IMAGE)[1]["status"], "CONTINUE")
        for kind in ("RL", "OBS", "PREP_SELF", "PREP_HUMAN"):
            candidate = rows[0] | {"kind": kind}
            with self.subTest(kind=kind), patch.object(campaign, "_encode", side_effect=train_v17.RowEncodingError("TOKEN_MASK", "bad")):
                with self.assertRaises(train_v17.RowEncodingError):
                    campaign._encode_update_rows(self.bundle, [candidate], items, IMAGE)

    def test_provenance_and_unexpected_failures_remain_fatal(self):
        rows, items = self.specimens()
        rows[0]["generation"]["text"] += "tampered"
        with self.assertRaises(ValueError):
            campaign._encode_update_rows(self.bundle, rows, items, IMAGE)
        with patch.object(campaign, "_encode", side_effect=RuntimeError("OOM")), self.assertRaisesRegex(RuntimeError, "OOM"):
            campaign._encode_update_rows(self.bundle, rows[1:], items, IMAGE)

    def test_cannot_claim_an_rl_row_was_quarantined(self):
        rows, _ = self.specimens(1)
        row = rows[0] | {"kind": "RL"}
        runtime = object.__new__(V16bScheduler)
        runtime.v18, runtime.v18b, runtime.row_settlements = True, True, {}
        rejected = {"row_id": row["row_id"], "kind": "RL", "source_row_sha256": digest(row)}
        with self.assertRaisesRegex(RuntimeError, "quarantine evidence"):
            runtime.settle_rows(0, [row], {"consumed_row_ids": [], "row_quarantine": {
                "status": "CONTINUE", "rejected_rows": [rejected]}})

    def test_production_artifact_seal_checks_the_admission_receipt(self):
        from fine_tuning.error_note import v17_artifacts, v17_preflight
        from fine_tuning.error_note.prep_v16b import seal

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            definition = {"sha256": "a" * 64, "source": {"base_commit": "b" * 40, "actual_files": {}},
                          "protocol": {"sha256": "c" * 64}}
            report = {"schema": "v18b-row-quarantine-v1", "status": "CONTINUE", "attempted_rows": 1,
                      "auxiliary_rows": 1, "admitted_row_ids": ["p:N"], "rejected_rows": []}
            admission = seal(root / "row-admissions/slot-0.json", {"plan_sha256": definition["sha256"],
                "slot": 0, "parent_checkpoint": None, "report": report})
            transaction = seal(root / "transactions/slot-0.json", {"plan_sha256": definition["sha256"],
                "slot": 0, "parent_checkpoint": None, "row_admission_sha256": admission["sha256"]})
            seal(root / "update-results/slot-0.json", {"version": "v18b-update-result-v1",
                "plan_sha256": definition["sha256"], "global_slot": 0, "producer_pid": 11,
                "timing_scope": "encode_backward_optimizer", "wall_seconds": .1, "phase": "rl",
                "transaction_sha256": transaction["sha256"], "row_counts": {"N": 1}, "attempted_row_counts": {"N": 1},
                "result": {"optimizer_steps": 1, "stats": {"consumed_row_ids": ["p:N"], "row_quarantine": report}}})
            identity = {"code_commit": definition["source"]["base_commit"], "source_sha256": digest({}),
                        "campaign_plan_sha256": definition["sha256"], "protocol_sha256": definition["protocol"]["sha256"],
                        "training_protocol_version": "error-note-v18b"}
            seal(root / "training-bindings/live-11.json", {"version": "v18b-live-training-preflight-v1", **identity,
                "status": "BOUND", "production_admitted": True, "producer_pid": 11, "maximum_shape_measurements": 1,
                "optimizer_steps": 0, "production_parents_consumed": 0, "capacity": {},
                "actual_abc_status": "NOT_RUN_USE_PRODUCTION_SLOT_TIMINGS", "f0_transport": {"pass": True},
                "coexistence": [{"pass": True}]})
            with patch.object(v17_preflight, "_validate_live_capacity"):
                v17_artifacts._v18b_training_evidence(root, definition, 1)
                (root / "row-admissions/slot-0.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    v17_artifacts._v18b_training_evidence(root, definition, 1)


if __name__ == "__main__":
    unittest.main()
