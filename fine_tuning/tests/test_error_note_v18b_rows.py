"""Synthetic admission checks; none of these texts are training targets."""
import json
import unittest
from unittest.mock import patch

from fine_tuning.error_note import v18_rows as old, v18b_rows as rows
from fine_tuning.error_note.audit import digest
from fine_tuning.tests.test_error_note_v17_rows import TOKENIZER, output, seal, verifier
from fine_tuning.tests.test_error_note_v18_rows import source_record


def note(item, private, *, stage="T1", quote="AB = 9", corrected="7", unit="length", escaped=False):
    ledgers = [rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)]
    if item.get("obs"):
        ledgers.append(rows.audit_observation_v17(item, verifier=private, tokenizer=TOKENIZER)["ledger"])
    request = rows.build_note_request_v17(item, ledgers=ledgers, gold="14", tokenizer=TOKENIZER)
    text = json.dumps({"repairs": [{"stage": stage, "quote": quote, "corrected_value": corrected, "unit": unit}],
                       "lesson": "Check each relevant visual detail before calculating."}, ensure_ascii=escaped)
    item["note"] = output(text, prompt=request["prompt"], status="NOTE")
    receipt = rows.verify_note_v17(item, ledgers=ledgers, gold="14", choices=[], tokenizer=TOKENIZER, verifier=private)
    return receipt, request


def graph_item(text="Path: 0 -> 1 -> 2\nFinal answer: A", *, complete=False, directed=False):
    item = source_record(text, status="CORRECT")
    item.update(layer="graph", question="Find a path through the graph.", source="hf-gvlqa")
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]
    facts = [{"kind": "VISIBLE_NODE", "entity": n, "value": True} for n in range(4)]
    facts += [{"kind": "EDGE_PRESENT", "entity": pair, "value": True} for pair in edges if max(pair) < 4]
    raw = {"n": 5, "m": len(edges), "edges": edges}
    if complete:
        item.update(raw_annotation=raw, native_metadata={"directed": directed})
    private = verifier(item, facts, directed=directed, closed_world_kinds=["EDGE_PRESENT"],
        closed_world_scopes={"visible_nodes": list(range(4))}, obs_eligible=True,
        annotation_sha256=digest(raw), repair_catalog_eligible=True)
    item["v17_private_verifier"] = private
    return item, private


class V18BRows(unittest.TestCase):
    def test_note_has_original_public_obs_target_unit_and_response(self):
        item = source_record()
        private = verifier(item)
        item["v17_private_verifier"] = private
        request = rows.build_obs_request_v17(item)
        item["obs"] = output('{"value":"9","unit":"length"}', prompt=request["prompt"], status="OBS")
        with patch.object(rows, "render_note_target_last", wraps=rows.render_note_target_last) as rendered:
            receipt, request = note(item, private, stage="OBS", quote="9")
        self.assertTrue(receipt["n_accepted"], receipt)
        body = rendered.call_args.args[0]
        self.assertEqual(request["public_context_sha256"], digest(body))
        self.assertEqual(request["prompt"], rows.render_note_target_last(body))
        self.assertEqual(body["observation_question"]["unit"], "length")
        self.assertEqual(body["observation_question"]["target"], "directly printed length label on segment AB")
        self.assertEqual(body["observation"], item["obs"]["text"])
        self.assertNotIn("expected", json.dumps(body))
        self.assertNotIn("facts", json.dumps(body))
        item["obs"]["prompt"] += "changed"
        with self.assertRaisesRegex(ValueError, "OBS_INPUT_CHANGED"):
            rows.build_note_request_v17(item, ledgers=[rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)], gold="14")

    def test_absent_observation_limits_available_stages(self):
        item = source_record()
        _, request = note(item, verifier(item))
        body = json.loads(request["prompt"].split("\n\n", 1)[1])
        self.assertEqual(body["available_stages"], ["T1"])

    def test_unique_passage_contains_one_complete_wrong_claim(self):
        item = source_record("The printed segment AB = 9.\nFinal answer: B")
        receipt, _ = note(item, verifier(item), quote="The printed segment AB = 9.")
        self.assertTrue(receipt["n_accepted"], receipt)
        for quote in ("9", "AB =", "Final answer: B"):
            receipt, _ = note(item, verifier(item), quote=quote)
            self.assertFalse(receipt["n_accepted"], receipt)
        item = source_record("AB = 9 and CD = 8\nFinal answer: B")
        private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "7"},
                                  {"kind": "SEGMENT_LABEL", "entity": ["C", "D"], "value": "6"}])
        receipt, _ = note(item, private, quote="AB = 9 and CD = 8")
        self.assertFalse(receipt["n_accepted"], receipt)

    def test_wrong_stage_wrong_value_wrong_unit_and_same_value_are_not_repairs(self):
        for kwargs in ({"stage": "OBS"}, {"corrected": "9"}, {"corrected": "8"}, {"unit": "m"}):
            item = source_record()
            receipt, _ = note(item, verifier(item), **kwargs)
            self.assertFalse(receipt["n_accepted"], receipt)

    def test_unicode_repair_supervises_original_json_escape_tokens(self):
        item = source_record("AB = 9\nFinal answer: B")
        private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "칠"}])
        receipt, _ = note(item, private, corrected="칠", escaped=True)
        self.assertTrue(receipt["n_accepted"], receipt)
        row = rows.n_row_v17(item, tokenizer=TOKENIZER, receipt=receipt)
        supervised = bytes(t for t, keep in zip(row["generation"]["completion_token_ids"], row["loss_mask"]) if keep).decode()
        self.assertEqual(supervised, r"\uce60")
        self.assertEqual(row["n_source_evidence"]["row_id"], row["row_id"])

    def test_symbolic_printing_and_source_bound_substitution_are_separate(self):
        for text, expected in (("x = 9\nBC = 23\nUse the computed length carefully.\nFinal answer: 14", "VERIFIED"),
                               ("x = 9\nBC = 24\nFinal answer: 14", "CONTRADICTED"),
                               ("BC = 23\nFinal answer: 14", "UNKNOWN"),
                               ("BC = 3x - 4\nFinal answer: 14", "VERIFIED"),
                               ("BC = 3x + 4\nFinal answer: 14", "CONTRADICTED")):
            with self.subTest(text=text):
                item = source_record(text, status="CORRECT")
                item["t1"]["extracted"] = "14"
                item["question"] = "Given x = 9, find the requested side."
                private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["B", "C"], "value": "3x−4", "unit": "length"}])
                ledger = rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)
                claims = [c for c in ledger["claims"] if c["kind"] == "SEGMENT_LABEL"]
                self.assertEqual(len(claims), 1, claims)
                self.assertEqual(claims[0]["verdict"], expected, claims)
                receipt = rows.verify_solution_v17(item, verifier=private, tokenizer=TOKENIZER, gold="14")
                self.assertEqual("PROCESS_CONTRADICTION" in receipt["reasons"], expected == "CONTRADICTED", receipt)

    def test_observation_symbolic_whitespace_and_null_domain(self):
        item = source_record()
        private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "x+5", "unit": "length"}])
        item["v17_private_verifier"] = private
        request = rows.build_obs_request_v17(item)
        item["obs"] = output('{"value":"x + 5","unit":"length"}', prompt=request["prompt"], status="OBS")
        self.assertEqual(rows.audit_observation_v17(item, verifier=private, tokenizer=TOKENIZER)["score"], 1.)
        item.update(layer="scene", question="What color is shown?")
        private = verifier(item, [{"kind": "OBJECT_ATTRIBUTE", "entity": [1, "material"], "value": "metal"}], supported_claim_kinds=["OBJECT_ATTRIBUTE"])
        item["v17_private_verifier"] = private
        request = rows.build_obs_request_v17(item)
        item["obs"] = output('{"value":"null","unit":""}', prompt=request["prompt"], status="OBS")
        receipt = rows.audit_observation_v17(item, verifier=private, tokenizer=TOKENIZER)
        self.assertIsNone(receipt["score"])
        self.assertIn("INVALID_ATTRIBUTE_DOMAIN", receipt["reasons"])

    def test_complete_graph_checks_paths_beyond_old_four_nodes(self):
        item, private = graph_item("Path: 0 -> 4 -> 3\nFinal answer: A", complete=True)
        ledger = rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)
        self.assertEqual([c["verdict"] for c in ledger["claims"] if c["kind"] == "GRAPH_PATH"], ["VERIFIED"])
        item["t1"] = output("Path: 0 -> 4 -> 2\nFinal answer: A", prompt=item["p0_text"])
        receipt = rows.verify_solution_v17(item, verifier=private, tokenizer=TOKENIZER, gold="A")
        self.assertFalse(receipt["accepted"])
        self.assertIn("PROCESS_CONTRADICTION", receipt["reasons"])
        item, private = graph_item("Path: 0 -> 4 -> 2\nFinal answer: A", complete=False)
        claims = rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)["claims"]
        self.assertEqual([c["verdict"] for c in claims if c["kind"] == "GRAPH_PATH"], ["UNKNOWN"])

    def test_directed_graph_reverse_is_not_a_valid_path(self):
        item, private = graph_item("Path: 2 -> 1 -> 0\nFinal answer: A", complete=True, directed=True)
        claims = rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)["claims"]
        self.assertEqual([c["verdict"] for c in claims if c["kind"] == "GRAPH_PATH"], ["CONTRADICTED"])

    def test_prose_wrapped_path_and_hypothetical_negative(self):
        for text, expected in (("This sequence — 0 → 2 → 3 — forms a valid path, even though it does not use every edge.", "CONTRADICTED"),
                               ("Therefore, 0 → 2 → 3 forms a valid path.", "CONTRADICTED"),
                               ("Suppose 0 → 2 → 3 forms a path.", "UNKNOWN")):
            item, private = graph_item(text + "\nFinal answer: A", complete=True)
            claims = rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)["claims"]
            self.assertEqual([claim["verdict"] for claim in claims if claim["kind"] == "GRAPH_PATH"], [expected])

    def test_correct_answer_with_wrong_graph_path_is_excluded_from_a1_and_c(self):
        item, private = graph_item("Path: 0 -> 2 -> 3\nFinal answer: A", complete=True)
        receipt = rows.verify_solution_v17(item, verifier=private, tokenizer=TOKENIZER, gold="A")
        self.assertFalse(receipt["accepted"])
        self.assertIsNone(rows.a1_row_v17(item, tokenizer=TOKENIZER, receipt=receipt))
        lesson = {"text": "Inspect each connection before following a route."}
        item["note"] = output('{}', status="NOTE")
        nreceipt = seal({"note_output_sha256": digest(item["note"]), "lesson": lesson, "resolve_allowed": True, "repairs": []})
        item["resolve"] = output(item["t1"]["text"], prompt=old.previous.build_resolve_prompt_v16b(item["p0_text"], lesson))
        receipt = rows.verify_solution_v17(item, stage="resolve", verifier=private, tokenizer=TOKENIZER, gold="A", note_receipt=nreceipt)
        self.assertFalse(receipt["accepted"])
        self.assertIsNone(rows.c_row_v17(item, tokenizer=TOKENIZER, receipt=receipt))

    def test_model_proposed_path_repair_is_checked_edge_by_edge_and_used_in_c(self):
        item, private = graph_item("Path: 0 -> 2 -> 3\nFinal answer: A", complete=True)
        receipt, _ = note(item, private, quote="0 -> 2 -> 3", corrected="0 -> 4 -> 3", unit="")
        self.assertTrue(receipt["n_accepted"], receipt)
        self.assertEqual(receipt["repairs"][0]["kind"], "GRAPH_PATH")
        item["resolve"] = output("Path: 0 -> 4 -> 3\nFinal answer: A",
            prompt=old.previous.build_resolve_prompt_v16b(item["p0_text"], receipt["lesson"]))
        admitted = rows.verify_solution_v17(item, stage="resolve", verifier=private, tokenizer=TOKENIZER,
            gold="A", note_receipt=receipt)
        self.assertTrue(admitted["accepted"], admitted)
        self.assertTrue(admitted["corrected_use_byte_spans"])
        for corrected in ("3 -> 4 -> 0", "0 -> 2 -> 3", "0 -> 4 -> 2 -> 3"):
            bad, _ = note(item, private, quote="0 -> 2 -> 3", corrected=corrected, unit="")
            self.assertFalse(bad["n_accepted"], bad)

    def test_whole_arithmetic_not_suffix_and_retraction_are_preserved(self):
        item = source_record("(20025/100) * (10/95) = (20025)/(9500)\nFinal answer: B")
        claims = rows.audit_t1_v17(item, verifier=None, tokenizer=TOKENIZER)["claims"]
        whole = [c for c in claims if c.get("reason") == "WHOLE_EXACT_EQUATION"]
        self.assertEqual(len(whole), 1, claims)
        self.assertEqual(whole[0]["verdict"], "CONTRADICTED")
        item = source_record("AB = 9\nActually AB = 7\nFinal answer: B")
        receipt, _ = note(item, verifier(item))
        self.assertFalse(receipt["n_accepted"], receipt)

    def test_balanced_graph_classes_and_constant_baseline(self):
        items = []
        for index in range(10):
            item, _ = graph_item(complete=True)
            item["problem_id"] = f"graph-{index}"
            private = {k: v for k, v in item["v17_private_verifier"].items() if k != "sha256"}
            private["problem_id"] = item["problem_id"]
            item["v17_private_verifier"] = seal(private)
            items.append(item)
        bound = rows.bind_graph_observation_targets_v18b(items)
        self.assertEqual(sum(item["v18b_obs_balance"]["present"] for item in bound), 5)
        self.assertEqual(bound, rows.bind_graph_observation_targets_v18b(list(reversed(items)))[::-1])
        records = []
        for item in bound:
            private = rows.problem_contract_v17(item)["verifier"]
            request = rows.build_obs_request_v17(item)
            item["obs"] = output('{"value":false,"unit":""}', prompt=request["prompt"], status="OBS")
            item["obs_score"] = rows.audit_observation_v17(item, verifier=private, tokenizer=TOKENIZER)
            records.append(item)
        report = rows.observation_baseline_v18b(records)
        self.assertEqual((report["graded"], report["correct"], report["constant_best_correct"]), (10, 5, 5))

    def test_historical_obs_target_is_retained_without_reinterpretation(self):
        item, private = graph_item(complete=True)
        request = old.build_obs_request_v18(item)
        item["obs"] = output('{"value":true,"unit":""}', prompt=request["prompt"], status="OBS")
        item = rows.bind_historical_observation_v18b(item)
        receipt = rows.audit_observation_v17(item, verifier=private, tokenizer=TOKENIZER)
        self.assertEqual(receipt["score"], 1.)
        self.assertTrue(receipt["historical_query"])
        self.assertEqual(receipt["ledger"]["claims"][0]["original_entity"], [0, 1])

    def test_contradicted_correct_solution_only_reinforces_bound_final_or_holds_pg(self):
        import torch
        from fine_tuning.error_note import train_v18

        for final in ("Final answer: A", "The final result is A."):
            item, private = graph_item("Path: 0 -> 2 -> 3\n" + final, complete=True)
            policy = rows.t1_policy_receipt_v18b(item, verifier=private, tokenizer=TOKENIZER, gold="A")
            selected = bytes(token for token, keep in zip(item["t1"]["generation"]["completion_token_ids"], policy["loss_token_mask"]) if keep).decode()
            self.assertEqual(selected, final if final.startswith("Final answer:") else "")
            candidate = {"kind": "RL", "target": item["t1"]["text"], "problem_id": item["problem_id"],
                "generation": item["t1"]["generation"], "image_sha256": item["image_sha256"],
                "loss_mask": torch.tensor(policy["loss_token_mask"]), "advantage": .5 if policy["pg"] else 0.,
                "t1_policy_receipt": policy, **{key: policy[key] for key in
                    ("pg", "pg_mask_reason", "pg_byte_spans", "policy_mask_receipt")}}
            train_v18._validate_masks(candidate, TOKENIZER)
            if policy["pg"]:
                candidate["pg_byte_spans"] = [[0, 4]]
                with self.assertRaises(ValueError):
                    train_v18._validate_masks(candidate, TOKENIZER)

    def test_partial_archive_requires_an_explicit_contiguous_prefix(self):
        from pathlib import Path
        import tempfile
        from fine_tuning.error_note.v18_replay import read_archive

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runtime/items").mkdir(parents=True)
            (root / "campaign-plan.json").write_text(json.dumps(seal({"items": [{"problem_id": "a"}, {"problem_id": "b"}]})))
            record_path = root / "runtime/items/a.json"
            record_path.write_text(json.dumps({"problem_id": "a"}))
            with self.assertRaisesRegex(ValueError, "native records"):
                read_archive(root)
            self.assertEqual(set(read_archive(root, allow_partial=True)[1]), {"a"})
            record_path.write_text(json.dumps({"problem_id": "b"}))
            with self.assertRaisesRegex(ValueError, "native records"):
                read_archive(root, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
