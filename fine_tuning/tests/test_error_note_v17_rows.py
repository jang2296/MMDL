"""Synthetic CPU contracts; these fixtures are never training material."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from fine_tuning.error_note import v17_rows as rows
from fine_tuning.error_note.audit import digest


class Bytes:
    all_special_tokens = ["<|im_end|>", "<|endoftext|>"]
    unk_token_id = None
    byte_encoder = bytes_to_unicode()

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def convert_tokens_to_ids(self, value):
        return 256 if value == "<|im_end|>" else 257

    def convert_ids_to_tokens(self, values):
        return [self.byte_encoder[x] if x < 256 else self.all_special_tokens[x - 256] for x in values]


TOKENIZER = Bytes()
CATALOG = [{"fact_id": "ab", "kind": "OBSERVATION", "entity": "segment AB", "value": "7", "unit": "length",
            "evidence_origin": "synthetic-fixture"}]


def seal(body):
    return body | {"sha256": digest(body)}


def output(text, prompt="Question: synthetic?", status="CORRECT", extracted="A", finish="stop", actions=None):
    actions = list(text.encode()) + ([256] if finish == "stop" else []) if actions is None else actions
    generation = {"text": text, "finish_reason": finish, "completion_token_ids": actions,
                  "prompt_token_ids": [1, 2], "policy_sha256": "a" * 64, "policy_version": 0,
                  "input_sha256": "b" * 64, "request_sha256": "c" * 64, "protocol_sha256": "d" * 64,
                  "sampling_logprobs": [-0.1] * len(actions)}
    return {"text": text, "prompt": prompt, "generation": generation, "finish_reason": finish,
            "status": status, "extracted": extracted}


def record(text="AB = 9\nFinal answer: B", status="INCORRECT", finish="stop"):
    return {"problem_id": "fixture", "source": "Geometry3K", "layer": "geometry", "p0_text": "Question: synthetic?",
            "image_paths": ["synthetic.png"], "image_sha256": ["e" * 64], "block_index": 0,
            "t1": output(text, status=status, finish=finish, extracted="B" if status != "CORRECT" else "A")}


def verifier(item, facts=None, **extra):
    facts = [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "7"}] if facts is None else facts
    return seal({"schema": "v17-private-verifier-v1", "problem_id": item["problem_id"],
                 "image_sha256": item["image_sha256"], "facts": facts,
                 "supported_claim_kinds": ["SEGMENT_LABEL", "VISIBLE_NODE", "EDGE_PRESENT", "EDGE_WEIGHT"],
                 "closed_world_kinds": [], **extra})


def observe(item, claims, private):
    item["obs"] = output(json.dumps({"claims": claims}, ensure_ascii=False), prompt="generic OBS", status="OBS")
    return rows.audit_observation_v17(item, verifier=private, tokenizer=TOKENIZER)


def note(item, private, *, claim_id="t0", corrected="7", lesson="Inspect each printed label before computing.", ledgers=None):
    ledgers = ledgers or [rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER, fact_catalog=CATALOG)]
    prompt = rows.build_note_request_v17(item, ledgers=ledgers, gold="A", tokenizer=TOKENIZER)["prompt"]
    item["note"] = output(json.dumps({"repairs": [{"claim_id": claim_id, "corrected_value": corrected}], "lesson": lesson}, ensure_ascii=False),
                          prompt=prompt, status="NOTE")
    return rows.verify_note_v17(item, verifier=private, ledgers=ledgers, choices=["first", "second"], gold="A",
                                tokenizer=TOKENIZER, fact_catalog=CATALOG), ledgers


class V17Rows(unittest.TestCase):
    def test_typed_chart_t1_claims_are_evidence_and_wrong_values_reject(self):
        item = record("SeriesX at Monday = 7\nFinal answer: A", status="CORRECT")
        private = verifier(item, [{"kind": "CHART_VALUE", "entity": ["SeriesX", "Monday"], "value": "7", "unit": "kg"}],
                           supported_claim_kinds=["CHART_VALUE"], repair_catalog_eligible=True)
        receipt = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], verifier=private)
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["verified_evidence_count"], 1)
        item["t1"] = output("SeriesX at Monday = 9\nFinal answer: A")
        receipt = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], verifier=private)
        self.assertIn("PROCESS_CONTRADICTION", receipt["reasons"])
        ledger = rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER)
        result, _ = note(item, private, corrected="7", ledgers=[ledger])
        self.assertTrue(result["n_accepted"], result)

    def test_new_scope_excludes_old_catalog_facts_and_unsupported_prose(self):
        item = record("AB = 7\nFinal answer: A", status="CORRECT")
        private = verifier(item, [], repair_catalog_eligible=False)
        receipt = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], verifier=private, fact_catalog=CATALOG)
        self.assertEqual(receipt["verified_evidence_count"], 0)
        # 2026-11-17: reasoning prose is admitted on catalog sources too (verified claims are almost never available);
        # a bare answer is still refused.
        item["t1"] = output("I think the label clearly supports this answer.\nFinal answer: A")
        private = verifier(item, repair_catalog_eligible=True)
        receipt = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], verifier=private)
        self.assertNotIn("ANSWER_ONLY", receipt["reasons"])
        item["t1"] = output("A. first")
        receipt = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], verifier=private)
        self.assertIn("ANSWER_ONLY", receipt["reasons"])

    def test_final_answer_comes_from_graded_extraction_or_decorated_final_lines(self):
        choices = ["first", "second"]
        def reasons(text, extracted="A"):
            item = record(text, status="CORRECT")
            item["t1"]["extracted"] = extracted
            return rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=choices,
                                            verifier=verifier(item, []))["reasons"]
        body = "The segment label shows the first option matches the measured length of the side.\n"
        self.assertEqual(reasons(body + "✅ **Final Answer: A**"), [])
        self.assertEqual(reasons(body + "Therefore the correct choice is **A**."), [])
        self.assertIn("NO_EXPLICIT_FINAL", reasons(body + "Therefore it is unclear.", extracted=None))
        self.assertIn("MULTIPLE_FINAL_ANSWERS", reasons(body + "Final answer: A\nFinal answer: A"))
        self.assertIn("MULTIPLE_FINAL_ANSWERS", reasons(body + "Final answer: A\nFinal answer: B"))
        self.assertIn("FINAL_ANSWER_MISMATCH", reasons(body + "Final answer: B", extracted="A"))

    def test_latex_source_values_are_left_out_of_the_checkable_catalog(self):
        item = record("AB = 7\nFinal answer: A", status="CORRECT")
        private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "2\\sqrt{3}"},
                                  {"kind": "SEGMENT_LABEL", "entity": ["A", "C"], "value": "7"}])
        self.assertEqual([f["entity"] for f in rows._source_catalog(private)], ["segment AC"])
        receipt = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], verifier=private)
        self.assertNotIn("UNSUPPORTED_SOURCE_FACT", str(receipt))

    def test_observation_prompt_forbids_an_empty_claim_list(self):
        prompt = rows.build_obs_request_v17({"problem_id": "fixture", "layer": "graph", "image_paths": [],
                                             "image_sha256": []})["prompt"]
        self.assertIn("1 to 16 objects", prompt)
        self.assertIn("an empty list is not allowed", prompt)
        self.assertNotIn("at most 16", prompt)

    def test_public_request_ignores_all_private_values(self):
        item = record()
        before = rows.build_obs_request_v17(item)
        after = rows.build_obs_request_v17(item | {"gold": "private-answer", "references": {"facts": "secret-facts"}})
        self.assertEqual(before, after)
        from fine_tuning.error_note.v17_data import public_schema
        schema = public_schema("geometry") | {"entities": ["A", "B"]}
        with self.assertRaisesRegex(ValueError, "PUBLIC_SCHEMA"):
            rows.build_obs_request_v17(item, public_schema=schema)

    def test_scoped_graph_outside_node_is_unknown_and_bool_value_is_supported(self):
        item = record()
        facts = [{"kind": "VISIBLE_NODE", "entity": n, "value": True} for n in (1, 2)]
        private = verifier(item, facts, closed_world_kinds=["VISIBLE_NODE"])
        result = observe(item, [{"kind": "VISIBLE_NODE", "entity": 1, "value": "true"},
                                {"kind": "VISIBLE_NODE", "entity": 99, "value": True}], private)
        self.assertEqual((result["tp"], result["fp"], result["unknown"]), (1, 0, 1))

    def test_f1_deduplicates_and_distinguishes_missing_unknown_and_wrong(self):
        item = record()
        private = verifier(item)
        receipt = observe(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "7"}] * 2
                          + [{"kind": "SEGMENT_LABEL", "entity": ["C", "D"], "value": "2"}], private)
        self.assertEqual((receipt["tp"], receipt["fp"], receipt["fn"], receipt["unknown"]), (1, 0, 0, 1))
        self.assertEqual(receipt["reward"], 1)
        empty = observe(item, [], private)
        self.assertEqual((empty["reward"], empty["fp"], empty["fn"]), (0, 0, 1))

    def test_conflicting_entity_claims_are_not_merged(self):
        item = record()
        receipt = observe(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": x} for x in ("7", "9")], verifier(item))
        self.assertEqual(receipt["tp"], 0)
        self.assertIn("CONFLICTING_ENTITY_CLAIMS", receipt["reasons"])

    def test_certified_edge_absence_is_true_without_adding_f1_target_tp(self):
        item = record()
        facts = [{"kind": "VISIBLE_NODE", "entity": n, "value": True} for n in (1, 2)]
        facts.append({"kind": "EDGE_PRESENT", "entity": [2, 1], "value": True})
        private = verifier(item, facts, directed=True, closed_world_kinds=["EDGE_PRESENT"])
        for value in (False, "false", "absent"):
            with self.subTest(value=value):
                receipt = observe(item, [{"kind": "EDGE_PRESENT", "entity": [1, 2], "value": value}], private)
                self.assertEqual(receipt["ledger"]["claims"][0]["verdict"], "VERIFIED")
                self.assertEqual((receipt["tp"], receipt["fp"], receipt["fn"]), (0, 0, 3))
        receipt = observe(item, [{"kind": "EDGE_PRESENT", "entity": [1, 2], "value": True}], private)
        self.assertEqual(receipt["ledger"]["claims"][0]["verdict"], "CONTRADICTED")
        self.assertEqual((receipt["tp"], receipt["fp"], receipt["fn"]), (0, 1, 3))
        repaired, _ = note(item, private, claim_id="o0", corrected="false", ledgers=[receipt["ledger"]])
        self.assertTrue(repaired["n_accepted"], repaired)
        row = rows.n_row_v17(item, tokenizer=TOKENIZER, receipt=repaired)
        selected = bytes(a for a, keep in zip(row["generation"]["completion_token_ids"], row["loss_token_mask"]) if keep)
        self.assertEqual(selected, b"false")

    def test_weighted_edge_presence_is_true_but_not_a_weight_target(self):
        item = record()
        facts = [{"kind": "VISIBLE_NODE", "entity": n, "value": True} for n in (1, 2)]
        facts.append({"kind": "EDGE_WEIGHT", "entity": [1, 2], "value": "7"})
        private = verifier(item, facts, directed=True, closed_world_kinds=["EDGE_PRESENT", "EDGE_WEIGHT"])
        receipt = observe(item, [{"kind": "EDGE_PRESENT", "entity": [1, 2], "value": True}], private)
        self.assertEqual(receipt["ledger"]["claims"][0]["verdict"], "VERIFIED")
        self.assertEqual((receipt["tp"], receipt["fp"], receipt["fn"]), (0, 0, 3))
        receipt = observe(item, [{"kind": "EDGE_PRESENT", "entity": [1, 2], "value": False}], private)
        self.assertEqual(receipt["ledger"]["claims"][0]["verdict"], "CONTRADICTED")

    def test_geometric_relation_endpoints_and_segment_order_are_equivalent(self):
        item = record()
        for kind in ("PARALLEL", "PERPENDICULAR"):
            private = verifier(item, [{"kind": kind, "entity": [["A", "B"], ["C", "D"]], "value": True}],
                               supported_claim_kinds=[kind])
            for entity in ([["B", "A"], ["D", "C"]], [["D", "C"], ["B", "A"]], [["C", "D"], ["A", "B"]]):
                with self.subTest(kind=kind, entity=entity):
                    result = observe(item, [{"kind": kind, "entity": entity, "value": True}], private)
                    self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
                    self.assertEqual(result["ledger"]["claims"][0]["verdict"], "VERIFIED")
            wrong = observe(item, [{"kind": kind, "entity": [["B", "A"], ["D", "C"]], "value": False}], private)
            self.assertEqual(wrong["ledger"]["claims"][0]["verdict"], "CONTRADICTED")
            other = observe(item, [{"kind": kind, "entity": [["A", "C"], ["B", "D"]], "value": True}], private)
            self.assertEqual(other["ledger"]["claims"][0]["verdict"], "UNKNOWN")

    def test_geometry_optional_canonical_units_and_chart_separate_unit(self):
        item = record()
        for kind, entity, unit in (("SEGMENT_LABEL", ["A", "B"], "length"), ("ANGLE_LABEL", ["A", "B", "C"], "degree")):
            private = verifier(item, [{"kind": kind, "entity": entity, "value": "7", "unit": unit}], supported_claim_kinds=[kind])
            for field in ({}, {"unit": unit}):
                result = observe(item, [{"kind": kind, "entity": entity[::-1], "value": "7", **field}], private)
                self.assertEqual(result["tp"], 1)
            wrong = observe(item, [{"kind": kind, "entity": entity, "value": "9", "unit": unit}], private)
            self.assertEqual(wrong["fp"], 1)
        private = verifier(item, [{"kind": "CHART_VALUE", "entity": ["SeriesX", "Monday"], "value": "7", "unit": "kg"}],
                           supported_claim_kinds=["CHART_VALUE"])
        correct = observe(item, [{"kind": "CHART_VALUE", "entity": ["SeriesX", "Monday"], "value": "7", "unit": "kg"}], private)
        self.assertEqual(correct["tp"], 1)
        mixed = observe(item, [{"kind": "CHART_VALUE", "entity": ["SeriesX", "Monday"], "value": "7 kg", "unit": "kg"}], private)
        self.assertEqual(mixed["fp"], 1)
        missing_unit = observe(item, [{"kind": "CHART_VALUE", "entity": ["SeriesX", "Monday"], "value": "7"}], private)
        self.assertEqual(missing_unit["unknown"], 1)

    def test_directed_edges_and_closed_world_are_source_bound(self):
        item = record()
        facts = [{"kind": "VISIBLE_NODE", "entity": n, "value": "present"} for n in (1, 2)]
        facts += [{"kind": "EDGE_PRESENT", "entity": [1, 2], "value": "present"}]
        private = verifier(item, facts, directed=True, closed_world_kinds=["EDGE_PRESENT"])
        claims = [{"kind": "EDGE_PRESENT", "entity": [2, 1], "value": "present"},
                  {"kind": "EDGE_PRESENT", "entity": [2, 99], "value": "present"}]
        result = observe(item, claims, private)
        self.assertEqual((result["fp"], result["unknown"]), (1, 1))
        private = verifier(item, facts, directed=False, closed_world_kinds=["EDGE_PRESENT"])
        result = observe(item, claims[:1], private)
        self.assertEqual(result["tp"], 1)

    def test_json_duplicates_depth_nonfinite_and_control_are_rejected(self):
        malformed = ['{"claims":[],"claims":[]}', '{"claims":NaN}', '{"claims":[{"kind":"SEGMENT_LABEL","entity":["A","B"],"value":"Inf"}]}',
                     '{"claims":' + '[' * 34 + '0' + ']' * 34 + '}', '{"claims":[],"x":"<|image_pad|>"}']
        for text in malformed:
            with self.subTest(text=text[:40]):
                item = record()
                item["obs"] = output(text, status="OBS")
                result = rows.audit_observation_v17(item, verifier=verifier(item), tokenizer=TOKENIZER)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reward"], 0)

    def test_private_verifier_and_generation_mutation_fail(self):
        item = record()
        private = verifier(item)
        private["facts"][0]["value"] = "8"
        with self.assertRaisesRegex(ValueError, "EVIDENCE_CHANGED"):
            observe(item, [], private)
        item = record()
        item["t1"]["generation"]["completion_token_ids"][0] = 1
        with self.assertRaisesRegex(ValueError, "SAMPLED_TOKEN_BINDING"):
            rows.audit_t1_v17(item, verifier=verifier(item), tokenizer=TOKENIZER)

    def test_note_whitelist_never_serializes_private_fact_or_hash(self):
        item = record()
        private = verifier(item)
        ledger = rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER, fact_catalog=CATALOG)
        prompt = rows.build_note_request_v17(item, ledgers=[ledger], gold="A")["prompt"]
        self.assertIn('"original_value": "9"', prompt)
        for forbidden in ("canonical_fact_key", "computed_value", "verifier_evidence_sha256", private["sha256"], '"value": "7"', "synthetic-fixture"):
            self.assertNotIn(forbidden, prompt)

    def test_real_wrong_to_correct_masks_only_value_and_is_immutable(self):
        item = record()
        receipt, _ = note(item, verifier(item))
        self.assertTrue(receipt["n_accepted"], receipt)
        row = rows.n_row_v17(item, tokenizer=TOKENIZER, receipt=receipt)
        selected = bytes(a for a, include in zip(row["generation"]["completion_token_ids"], row["loss_token_mask"]) if include)
        self.assertEqual(selected, b"7")
        self.assertFalse(row["loss_token_mask"][-1])
        item["note"]["text"] = "changed"
        self.assertNotEqual(row["target"], item["note"]["text"])

    def test_unicode_content_uses_utf8_bytes(self):
        item = record()
        private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "칠"}])
        obs = observe(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "여덟"}], private)
        receipt, _ = note(item, private, claim_id="o0", corrected="칠", ledgers=[obs["ledger"]])
        self.assertTrue(receipt["n_accepted"], receipt)
        row = rows.n_row_v17(item, tokenizer=TOKENIZER, receipt=receipt)
        selected = bytes(a for a, keep in zip(row["generation"]["completion_token_ids"], row["loss_token_mask"]) if keep)
        self.assertEqual(selected.decode(), "칠")

    def test_unknown_verified_and_forged_id_do_not_count_as_repairs(self):
        for claim_id, value, old in (("missing", "7", "9"), ("o0", "7", "7"), ("o0", "7", "9")):
            item = record()
            private = verifier(item)
            obs = observe(item, [{"kind": "SEGMENT_LABEL", "entity": ["C", "D"] if old == "9" else ["A", "B"], "value": old}], private)
            result, _ = note(item, private, claim_id=claim_id, corrected=value, ledgers=[obs["ledger"]])
            self.assertFalse(result["n_accepted"])
            self.assertEqual(result["verified_repair_count"], 0)

    def test_o0_t0_are_distinct_and_forged_generation_ledger_is_rebuilt(self):
        item = record()
        private = verifier(item)
        obs = observe(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "9"}], private)
        t1 = rows.audit_t1_v17(item, verifier=private, tokenizer=TOKENIZER, fact_catalog=CATALOG)
        result, _ = note(item, private, claim_id="o0", ledgers=[obs["ledger"], t1])
        self.assertTrue(result["n_accepted"])
        fake = deepcopy(t1)
        fake["claims"][0]["original_value"] = "123"
        fake = seal({k: v for k, v in fake.items() if k != "sha256"})
        result = rows.verify_note_v17(item, verifier=private, ledgers=[fake], choices=["first", "second"], gold="A", tokenizer=TOKENIZER, fact_catalog=CATALOG)
        self.assertFalse(result["n_accepted"])
        self.assertFalse(result["resolve_allowed"])

    def test_n_is_independent_of_leaking_lesson_and_wrong_resolve(self):
        item = record()
        result, _ = note(item, verifier(item), lesson="Select option A and use 7.")
        self.assertTrue(result["n_accepted"])
        self.assertFalse(result["lesson_accepted"])
        self.assertFalse(result["resolve_allowed"])
        safe, _ = note(item, verifier(item))
        item["resolve"] = output("Some independent explanation.\nFinal answer: B", status="INCORRECT", extracted="B")
        c = rows.verify_solution_v17(item, stage="resolve", tokenizer=TOKENIZER, gold="A", choices=["first", "second"], note_receipt=safe)
        self.assertFalse(c["accepted"])
        self.assertTrue(safe["n_accepted"])

    def test_failed_n_safe_lesson_can_admit_independent_c(self):
        item = record()
        result, _ = note(item, verifier(item), corrected="8")
        self.assertFalse(result["n_accepted"])
        self.assertTrue(result["resolve_allowed"])
        prompt = rows.previous.build_resolve_prompt_v16b(item["p0_text"], result["lesson"])
        item["resolve"] = output("AB = 7\nFinal answer: A", prompt=prompt)
        receipt = rows.verify_solution_v17(item, stage="resolve", tokenizer=TOKENIZER, gold="A", choices=["first", "second"],
                                           fact_catalog=CATALOG, note_receipt=result)
        self.assertTrue(receipt["accepted"], receipt)
        c = rows.c_row_v17(item, tokenizer=TOKENIZER, receipt=receipt)
        self.assertEqual(c["prompt"], item["p0_text"])
        self.assertNotEqual(c["generation_prompt_sha256"], c["training_prompt_sha256"])
        self.assertEqual(len(c["source_generations"]), 3)

    def test_truncated_t1_with_error_can_note_but_not_a1(self):
        item = record("AB = 9", finish="length")
        result, _ = note(item, verifier(item))
        self.assertTrue(result["n_accepted"], result)
        a1 = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", fact_catalog=CATALOG)
        self.assertFalse(a1["accepted"])
        self.assertIn("PROCESS_CONTRADICTION", a1["reasons"])

    def test_empty_and_eos_only_t1_bind_without_inventing_actions(self):
        for actions in ([256], []):
            with self.subTest(actions=actions):
                item = record("", status="NO_PARSE")
                item["t1"]["extracted"] = None
                item["t1"]["generation"]["completion_token_ids"] = list(actions)
                item["t1"]["generation"]["sampling_logprobs"] = [-.1] * len(actions)
                ledger = rows.audit_t1_v17(item, verifier=verifier(item), tokenizer=TOKENIZER)
                self.assertEqual(ledger["claims"], [])
                self.assertEqual(ledger["original_token_ids_sha256"], digest(actions))
                receipt = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"])
                self.assertFalse(receipt["accepted"])
                self.assertIn("ANSWER_ONLY", receipt["reasons"])
                self.assertIn("NOT_CORRECT", receipt["reasons"])
                self.assertIsNone(rows.a1_row_v17(item, tokenizer=TOKENIZER, receipt=receipt))
                self.assertEqual(item["t1"]["generation"]["completion_token_ids"], actions)
        item = record("", status="NO_PARSE")
        item["t1"]["generation"]["completion_token_ids"] = [42]
        with self.assertRaisesRegex(ValueError, "SAMPLED_TOKEN_BINDING"):
            rows.audit_t1_v17(item, verifier=verifier(item), tokenizer=TOKENIZER)

    def test_public_prompt_declares_generic_entity_types_without_private_examples(self):
        from fine_tuning.error_note.v17_data import public_schema
        item = record() | {"layer": "graph"}
        schema = public_schema("graph")
        prompt = rows.build_obs_request_v17(item)["prompt"]
        self.assertEqual(schema["entity_schema"]["VISIBLE_NODE"]["entity_type"], "integer")
        self.assertIn(json.dumps(schema["entity_schema"], ensure_ascii=False, sort_keys=True), prompt)
        for key in ("facts", "ordered_entities", "target", "private_verifier"):
            self.assertNotIn('"' + key + '"', prompt)

    def test_short_good_solution_and_all_reasons_short_bad_solution(self):
        item = record("AB = 7\nFinal answer: A", status="CORRECT")
        result = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], fact_catalog=CATALOG)
        self.assertTrue(result["accepted"], result)
        self.assertLess(len(item["t1"]["generation"]["completion_token_ids"]), 256)
        item = record("AB = 9\nFinal answer: B", finish="length")
        result = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], fact_catalog=CATALOG)
        for reason in ("NOT_STOPPED", "NOT_CORRECT", "FINAL_ANSWER_MISMATCH", "PROCESS_CONTRADICTION"):
            self.assertIn(reason, result["reasons"])

    def test_answer_only_restatement_and_multiple_conclusions_reject(self):
        for text in ("Final answer: A", "The correct answer is first.\nFinal answer: A", "AB = 7\nThus B\nFinal answer: A", "AB = 7\nFinal answer: A\nFinal answer: A"):
            item = record(text, status="CORRECT")
            result = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], fact_catalog=CATALOG)
            self.assertFalse(result["accepted"], result)

    def test_long_contradiction_and_loops_never_become_sft(self):
        item = record("AB = 9\n" + "different explanations for the drawing. " * 80 + "\nFinal answer: A", status="CORRECT")
        result = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], fact_catalog=CATALOG)
        self.assertIn("PROCESS_CONTRADICTION", result["reasons"])
        self.assertIn("REPETITION", result["reasons"])
        self.assertIsNone(rows.a1_row_v17(item, tokenizer=TOKENIZER, receipt=result))

    def test_arithmetic_repair_requires_current_full_assertion(self):
        for text, accepted in (("2 + 3 = 8", True), ("If 2 + 3 = 8", False), ('"2 + 3 = 8"', False), ("2 + 3 = 8\nActually, 2 + 3 = 5", False)):
            item = record(text)
            ledger = rows.audit_t1_v17(item, verifier=verifier(item), tokenizer=TOKENIZER)
            result, _ = note(item, verifier(item), corrected="5", ledgers=[ledger])
            self.assertEqual(result["n_accepted"], accepted, (text, result))

    def test_sampled_ids_need_not_equal_reencoded_ids(self):
        class Merged(Bytes):
            def convert_ids_to_tokens(self, values):
                return ["AB" if x == 300 else super(Merged, self).convert_ids_to_tokens([x])[0] for x in values]
        item = record("AB = 7\nFinal answer: A", status="CORRECT")
        item["t1"]["generation"]["completion_token_ids"] = [300] + list(item["t1"]["text"].encode())[2:] + [256]
        result = rows.verify_solution_v17(item, tokenizer=Merged(), gold="A", choices=["first", "second"], fact_catalog=CATALOG)
        self.assertTrue(result["accepted"], result)
        a1 = rows.a1_row_v17(item, tokenizer=Merged(), receipt=result)
        self.assertEqual(a1["generation"]["completion_token_ids"][0], 300)

    def test_appended_answer_or_eos_and_changed_parent_are_rejected(self):
        item = record("AB = 7\nFinal answer: A", status="CORRECT")
        result = rows.verify_solution_v17(item, tokenizer=TOKENIZER, gold="A", choices=["first", "second"], fact_catalog=CATALOG)
        item["t1"]["generation"]["completion_token_ids"].append(256)
        with self.assertRaisesRegex(ValueError, "OUTPUT_CHANGED"):
            rows.a1_row_v17(item, tokenizer=TOKENIZER, receipt=result)

    def test_loop_reward_and_last3000_unicode_boundary_without_probability_mask(self):
        prefix = "unique lead " * 30
        text = prefix + "가나다 " * 900
        for finish in ("length", "stop"):
            self.assertEqual(rows.reward_v17("TRUNCATED", finish, text, None), (0, "LOOP_CONFIRMED", True))
        self.assertEqual(rows.reward_v17("TRUNCATED", "length", "Reasoning incomplete but nonrepeating.", None), (0, "LENGTH", False))
        actions = list(text.encode()) + [256]
        mask = rows.loop_token_mask_v17(TOKENIZER, actions, text)
        start = len(text[:-3000].encode())
        self.assertEqual(mask, [False] * start + [True] * (len(text.encode()) - start) + [False])

    def test_n_boundary_token_is_excluded_not_split(self):
        class Boundary(Bytes):
            def convert_ids_to_tokens(self, values):
                return ['"7' if x == 300 else super(Boundary, self).convert_ids_to_tokens([x])[0] for x in values]
        item = record()
        result, ledgers = note(item, verifier(item))
        output_note = item["note"]
        raw = output_note["text"].encode()
        offset = raw.index(b'"7"')
        output_note["generation"]["completion_token_ids"] = list(raw[:offset]) + [300] + list(raw[offset + 2:]) + [256]
        result = rows.verify_note_v17(item, verifier=verifier(item), ledgers=ledgers, choices=["first", "second"], gold="A", tokenizer=Boundary(), fact_catalog=CATALOG)
        self.assertFalse(result["n_accepted"])
        self.assertTrue(any(r.startswith("N_MASK") for r in result["reasons"]))

    def test_common_topic_stem_does_not_disable_number_or_option_leak_checks(self):
        item = record()
        self.assertEqual(rows._lesson("Compare material before computing.", item, ["red material", "blue material"], "A", []), [])
        for lesson in ("Choose option A.", "Compare material 7."):
            self.assertTrue(rows._lesson(lesson, item, ["red material", "blue material"], "A", []))

    def test_prep_uses_same_short_solution_admission_and_records_failures(self):
        good = record("AB = 7\nFinal answer: A", status="CORRECT")
        good["evaluation"] = {"gold": "A", "choices": ["first", "second"]}
        bad = deepcopy(good)
        bad["problem_id"] = "bad"
        bad["t1"] = output("AB = 9\nFinal answer: A")
        contract = {"fact_catalog": CATALOG, "rules": {}, "verifier": None}
        with patch.object(rows, "problem_contract_v17", return_value=contract):
            result, receipts = rows.prep_rows_with_receipts_v17([good, bad], tokenizer=TOKENIZER)
        self.assertEqual([r["kind"] for r in result], ["PREP_SELF"])
        self.assertEqual(len(receipts), 4)
        rejected = next(r for r in receipts if r["problem_id"] == "bad" and r["kind"] == "PREP_SELF")
        self.assertIn("PROCESS_CONTRADICTION", rejected["admission"]["reasons"])


if __name__ == "__main__":
    unittest.main()
