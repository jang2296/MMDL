"""CPU fixture proves N lineage, not natural Qwen self-correction or task accuracy."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fine_tuning.error_note import rl, train_v17, train_v18
from fine_tuning.error_note.audit import digest
from fine_tuning.tests.test_error_note_train import tiny_factory
from fine_tuning.tests.test_error_note_v17_rows import TOKENIZER, verifier
from fine_tuning.tests.test_error_note_v18_loss import contract
from fine_tuning.tests.test_error_note_v18_rows import source_record
from fine_tuning.tests.test_error_note_v18b_rows import note, rows


def actual_fixture_row(bundle, change_prompt=False):
    item = source_record()
    item.update(image_paths=[], image_sha256=[])
    private = verifier(item)
    note(item, private)
    prompt_ids = torch.tensor([7, 8])
    input_hash = train_v17.encoded_input_sha256_v17({"input_ids": prompt_ids, "target_len": 0})
    item["note"]["generation"].update(prompt_token_ids=prompt_ids.tolist(), input_sha256=input_hash,
        policy_version=1, probability_space="processed-full-vocab-v1",
        sampling_sha256=rl.sampling_sha256(train_v17.SAMPLING), policy_sha256=rl.policy_sha256(bundle))
    ledgers = [rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)]
    receipt = rows.verify_note_v17(item, ledgers=ledgers, verifier=private, tokenizer=TOKENIZER, gold="14", choices=[])
    row = rows.n_row_v17(item, tokenizer=TOKENIZER, receipt=receipt)
    assert row is not None, receipt
    if change_prompt:
        row.update(prompt="unrelated context", training_prompt_sha256=hashlib.sha256(b"unrelated context").hexdigest())
    prepared = {"inputs": {"input_ids": prompt_ids[None]}, "prompt_token_ids": prompt_ids.tolist(), "input_sha256": input_hash}
    with patch.object(train_v17, "_prepare_row_inputs", return_value=prepared):
        encoded = train_v18.encode_v18_row(bundle.processor, bundle.model.config, row, [])
    return encoded


class V18BNoteEvidence(unittest.TestCase):
    def test_note_rejects_a_different_training_context(self):
        with self.assertRaisesRegex(ValueError, "note training context"):
            actual_fixture_row(self.bundle(), change_prompt=True)

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def bundle(self):
        bundle = tiny_factory()
        bundle.model.resize_token_embeddings(258, mean_resizing=False)
        tokenizer = copy.copy(TOKENIZER)
        tokenizer.pad_token_id = 0
        bundle.processor = SimpleNamespace(tokenizer=tokenizer)
        return bundle

    def test_original_n_row_encodes_backpropagates_updates_and_survives_resume_state(self):
        bundle = self.bundle()
        row = actual_fixture_row(bundle)
        self.assertEqual(row["n_source_evidence"]["token_ids"], row["input_ids"][-row["target_len"]:].tolist())
        definition = contract(bundle, [row]) | {"version": "error-note-v18b"}
        gradients = {}
        stats = train_v18.backward_v18_block(bundle, [row], definition, components={"N"}, note_gradient_capture=gradients)
        self.assertFalse(stats["hold"], stats)
        parent = stats["n_parent_evidence"][0]
        self.assertEqual((parent["row_id"], parent["supervised_tokens"]), (row["row_id"], 1))
        self.assertTrue(parent["gradient_finite_nonzero"] and parent["note_gradient_retained"])
        self.assertEqual(parent["source_evidence_sha256"], row["n_source_evidence"]["sha256"])
        before, report = train_v18.note_step_before_v18(bundle, gradients, stats)
        optimizer = torch.optim.AdamW([p for ps in bundle.params.values() for p in ps], lr=.001)
        optimizer.step()
        report = train_v18.note_step_after_v18(bundle, gradients, before, report, step=True)
        self.assertLess(report["parameter_delta_dot"], 0.)
        self.assertEqual(report["row_ids"], [row["row_id"]])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            cursor = {"consumed_row_ids": stats["consumed_row_ids"], "n_parent_evidence": stats["n_parent_evidence"]}
            torch.save({"model": bundle.model.state_dict(), "optimizer": optimizer.state_dict(),
                        "cursor": cursor, "n_update": report}, checkpoint)
            saved_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restored = copy.deepcopy(bundle.model)
            restored.load_state_dict(loaded["model"], strict=True)
            self.assertEqual(loaded["cursor"], cursor)
            self.assertEqual(loaded["n_update"]["row_ids"], [row["row_id"]])
            self.assertEqual(loaded["n_update"]["n_parent_evidence"][0]["source_evidence_sha256"], row["n_source_evidence"]["sha256"])
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), saved_sha)
            self.assertEqual(loaded["optimizer"]["state"].keys(), optimizer.state_dict()["state"].keys())

    def test_altered_row_identity_or_token_mask_cannot_keep_source_receipt(self):
        bundle = self.bundle()
        for field in ("row_id", "supervised_tokens", "token_mask"):
            candidate = actual_fixture_row(bundle)
            evidence = candidate["n_source_evidence"]
            evidence[field] = {"row_id": "different:N", "supervised_tokens": 100,
                               "token_mask": [True] * candidate["target_len"]}[field]
            evidence["sha256"] = digest({key: value for key, value in evidence.items() if key != "sha256"})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "note source evidence changed"):
                train_v18.validate_v18_block(bundle, [candidate], contract(bundle, [candidate]) | {"version": "error-note-v18b"})


if __name__ == "__main__":
    unittest.main()
