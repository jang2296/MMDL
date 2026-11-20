"""Synthetic CPU admission tests; none of these examples is training material."""
from copy import deepcopy
import json
import unittest

from fine_tuning.error_note import v18_rows as rows
from fine_tuning.tests.test_error_note_v17_rows import TOKENIZER, output, record, seal, verifier


def source_record(text="AB = 9\nFinal answer: B", status="INCORRECT"):
    item = record(text, status=status)
    item.update(question="Use twice the printed segment AB; multiply by 2.", source_revision="f" * 40,
                native_id="synthetic/source:0:0")
    item["p0_text"] = "Question: " + item["question"]
    item["t1"]["prompt"] = item["p0_text"]
    return item


def repair(item, private, *, quote="AB = 9", corrected="7", unit="length", stage="T1",
           lesson="Inspect each printed label before computing."):
    ledgers = [rows.audit_t1_v18(item, verifier=private, tokenizer=TOKENIZER)]
    if item.get("obs"):
        ledgers.append(rows.audit_observation_v18(item, verifier=private, tokenizer=TOKENIZER)["ledger"])
    request = rows.build_note_request_v18(item, ledgers=ledgers, gold="14", tokenizer=TOKENIZER)
    item["note"] = output(json.dumps({"repairs": [{"stage": stage, "quote": quote,
        "corrected_value": corrected, "unit": unit}], "lesson": lesson}),
        prompt=request["prompt"], status="NOTE")
    receipt = rows.verify_note_v18(item, ledgers=ledgers, verifier=private, choices=[],
                                  gold="14", tokenizer=TOKENIZER)
    return receipt, ledgers


def map_record(state="Iowa", anchor="Missouri", region=None):
    question = (f"Among the states that border {anchor} , which have the highest value?" if region is None
                else f"Which states have the highest value in the {region}?")
    text = (f"{state} has the highest value among the states that border {anchor}." if region is None
            else f"{state} has the highest value in the {region}.")
    item = source_record(text)
    item.update(source="hf-mapqa", question=question, p0_text="Question: " + question,
                native_id="mapqa/train-frozen:43:5")
    item["t1"] = output(text, prompt=item["p0_text"], status="INCORRECT", extracted=text)
    return item


class V18Format(unittest.TestCase):
    def test_four_scope_cases_and_no_gold_in_extraction(self):
        for state, anchor, region in [("Iowa", "Missouri", None), ("Maryland", "Virginia", None),
                                      ("Wyoming", None, "West"), ("West Virginia", None, "USA")]:
            with self.subTest(state=state):
                item = map_record(state, anchor, region)
                receipt = rows.format_receipt_v18(item, gold=state)
                self.assertTrue(receipt["accepted"], receipt)
                decorated = deepcopy(item)
                decorated["t1"] = output("✅ **Final Answer**: **" + state + "** has the highest value " +
                    (f"among the states that border {anchor}." if region is None else f"in the {region}."),
                    prompt=item["p0_text"], status="INCORRECT")
                self.assertTrue(rows.format_receipt_v18(decorated, gold=state)["accepted"])
                self.assertEqual(receipt["extracted_without_gold"], state.casefold())
                other = rows.format_receipt_v18(item, gold="Alaska")
                self.assertEqual(other["extracted_without_gold"], state.casefold())
                self.assertFalse(other["content_equivalent"])

    def test_format_policy_only_penalizes_declaration_and_no_a1(self):
        item = map_record()
        item["t1"] = output("Read all plotted labels carefully.\n" + item["t1"]["text"], prompt=item["p0_text"], status="INCORRECT")
        policy = rows.t1_policy_receipt_v18(item, tokenizer=TOKENIZER, gold="Iowa")
        self.assertEqual(policy["reason"], "FORMAT_ONLY_CERTIFIED")
        self.assertEqual(policy["reward"], 0)
        mask = policy["loss_token_mask"]
        start = len("Read all plotted labels carefully.\n".encode())
        self.assertFalse(any(mask[:start]))
        self.assertFalse(any(mask[start:start + len("Iowa")]))
        self.assertTrue(all(mask[start + len("Iowa"):-1]))
        self.assertFalse(mask[-1])
        receipt = rows.verify_solution_v18(item, tokenizer=TOKENIZER, gold="Iowa")
        self.assertFalse(receipt["accepted"])
        self.assertIn("FORMAT_CORRECTION_REQUIRED", receipt["reasons"])

    def test_ambiguous_negation_direction_scope_and_source_rejected(self):
        for text in ["Iowa and Kansas have the highest value among the states that border Missouri.",
                     "Iowa or Kansas has the highest value among the states that border Missouri.",
                     "Iowa does not have the highest value among the states that border Missouri.",
                     "Iowa has the lowest value among the states that border Missouri.",
                     "Iowa has the highest value in the USA.",
                     "The highest value among the states that border Missouri is not Iowa.",
                     "Iowa might have the highest value among the states that border Missouri.",
                     "Among states that border Missouri the highest value includes Iowa."]:
            with self.subTest(text=text):
                item = map_record()
                item["t1"] = output(text, prompt=item["p0_text"], status="INCORRECT")
                self.assertFalse(rows.format_receipt_v18(item, gold="Iowa")["accepted"])
        item = map_record()
        del item["source_revision"]
        self.assertFalse(rows.format_receipt_v18(item, gold="Iowa")["accepted"])

    def test_format_flows_through_note_and_exactly_one_gold_free_retry(self):
        item = map_record()
        policy = rows.t1_policy_receipt_v18(item, tokenizer=TOKENIZER, gold="Iowa")
        item["t1_policy_receipt"] = policy
        ledger = rows.audit_t1_v18(item, verifier=None, tokenizer=TOKENIZER)
        prompt = rows.build_note_request_v18(item, ledgers=[ledger], gold="Iowa")["prompt"]
        item["note"] = output(json.dumps({"repairs": [], "lesson": rows.FORMAT_LESSON_V18}), prompt=prompt, status="NOTE")
        receipt = rows.verify_note_v18(item, ledgers=[ledger], verifier=None, choices=[], gold="Iowa", tokenizer=TOKENIZER)
        self.assertTrue(receipt["resolve_allowed"], receipt)
        self.assertFalse(receipt["n_accepted"])
        retry_prompt = rows.previous.build_resolve_prompt_v16b(item["p0_text"], receipt["lesson"])
        self.assertNotIn("Iowa", retry_prompt)
        item["resolve"] = output("Compare the displayed values in the requested scope.\nFinal answer: Iowa",
                                 prompt=retry_prompt, extracted="Iowa")
        candidate = rows.verify_solution_v18(item, stage="resolve", gold="Iowa", choices=[],
                                             note_receipt=receipt, verifier=None, tokenizer=TOKENIZER)
        self.assertTrue(candidate["accepted"], candidate)
        c = rows.c_row_v18(item, tokenizer=TOKENIZER, receipt=candidate)
        self.assertEqual(c["prompt"], item["p0_text"])
        again = rows.verify_note_v18(item, ledgers=[ledger], verifier=None, choices=[], gold="Iowa", tokenizer=TOKENIZER)
        self.assertFalse(again["resolve_allowed"])
        item["resolve2"] = item["resolve"]
        candidate = rows.verify_solution_v18(item, stage="resolve", gold="Iowa", note_receipt=receipt, tokenizer=TOKENIZER)
        self.assertIn("MULTIPLE_RESOLVE_ATTEMPTS", candidate["reasons"])


class V18Observation(unittest.TestCase):
    def graph(self, present=True):
        item = source_record()
        item.update(layer="graph", question="Find the shortest path from node 0 to node 2.")
        facts = [{"kind": "VISIBLE_NODE", "entity": x, "value": True} for x in range(3)]
        if present:
            facts.append({"kind": "EDGE_PRESENT", "entity": [0, 1], "value": True})
        private = verifier(item, facts, supported_claim_kinds=["VISIBLE_NODE", "EDGE_PRESENT", "EDGE_WEIGHT"],
                           closed_world_kinds=["EDGE_PRESENT"], closed_world_scopes={"visible_nodes": [0, 1, 2]})
        item["v17_private_verifier"] = private
        return item, private

    def audit(self, item, private, value, unit=""):
        request = rows.build_obs_request_v18(item)
        item["obs"] = output(json.dumps({"value": value, "unit": unit}), prompt=request["prompt"], status="OBS")
        return rows.audit_observation_v18(item, verifier=private, tokenizer=TOKENIZER)

    def test_public_query_same_for_presence_absence_and_one_fact_reward(self):
        yes, vy = self.graph(True)
        no, vn = self.graph(False)
        self.assertEqual(rows.build_obs_request_v18(yes)["prompt"], rows.build_obs_request_v18(no)["prompt"])
        self.assertEqual(self.audit(yes, vy, True)["reward"], 1)
        self.assertEqual(self.audit(yes, vy, False)["reward"], 0)
        self.assertEqual(self.audit(no, vn, False)["reward"], 1)
        self.assertEqual(self.audit(no, vn, True)["reward"], 0)

    def test_fenced_observation_is_read_like_bare_json(self):
        # v17 local F0 (2026-11-17): 3 of 6 observation answers came wrapped in one ```json fence.
        yes, private = self.graph(True)
        request = rows.build_obs_request_v18(yes)
        yes["obs"] = output('```json\n{"value": true, "unit": ""}\n```', prompt=request["prompt"], status="OBS")
        self.assertEqual(rows.audit_observation_v18(yes, verifier=private, tokenizer=TOKENIZER)["reward"], 1)

    def test_abstention_unit_unknown_malformed_and_changed_input_no_pg(self):
        item, private = self.graph()
        for value, unit in [(None, ""), (True, "meters"), ("unknown", "")]:
            receipt = self.audit(item, private, value, unit)
            self.assertIsNone(receipt["reward"], receipt)
            self.assertFalse(receipt["pg"])
        item["obs"] = output('{"claims": []}', prompt=rows.build_obs_request_v18(item)["prompt"], status="OBS")
        self.assertFalse(rows.audit_observation_v18(item, verifier=private, tokenizer=TOKENIZER)["pg"])
        self.audit(item, private, True)
        item["obs"]["prompt"] += " secret"
        self.assertFalse(rows.audit_observation_v18(item, verifier=private, tokenizer=TOKENIZER)["pg"])

    def test_answer_equivalent_and_uncertified_absence_are_excluded(self):
        item, private = self.graph()
        item["question"] = "Is there an edge between node 0 and node 1?"
        self.assertEqual(rows.observation_eligibility_v18(item)["reason"], "OBS_TARGET_ANSWER_EQUIVALENT")
        item["question"] = "Find the shortest path."
        source = {k: v for k, v in private.items() if k != "sha256"}
        source["closed_world_kinds"] = []
        item["v17_private_verifier"] = seal(source)
        self.assertEqual(rows.observation_eligibility_v18(item)["reason"], "OBS_ABSENCE_UNCERTIFIED")

    def test_private_schema_never_crosses_request_boundary(self):
        item, private = self.graph()
        with self.assertRaisesRegex(ValueError, "PRIVATE_OR_CHANGED"):
            rows.build_obs_request_v18(item, public_schema={"schema": rows.QUERY_SCHEMA, "layer": "graph", "value": True})


class V18Notes(unittest.TestCase):
    def test_valid_unique_repair_trains_only_corrected_content_even_if_c_fails(self):
        item = source_record()
        private = verifier(item)
        receipt, _ = repair(item, private)
        self.assertTrue(receipt["n_accepted"], receipt)
        row = rows.n_row_v18(item, tokenizer=TOKENIZER, receipt=receipt)
        selected = bytes(t for t, selected in zip(row["generation"]["completion_token_ids"], row["loss_token_mask"]) if selected).decode()
        self.assertEqual(selected, "7")
        item["resolve"] = output("Final answer: 99", status="INCORRECT")
        self.assertIsNotNone(rows.n_row_v18(item, tokenizer=TOKENIZER, receipt=receipt))

    def test_fenced_note_keeps_byte_exact_repair_tokens(self):
        item = source_record()
        private = verifier(item)
        ledgers = [rows.audit_t1_v18(item, verifier=private, tokenizer=TOKENIZER)]
        request = rows.build_note_request_v18(item, ledgers=ledgers, gold="14", tokenizer=TOKENIZER)
        body = json.dumps({"repairs": [{"stage": "T1", "quote": "AB = 9", "corrected_value": "7", "unit": "length"}],
                           "lesson": "Inspect each printed label before computing."})
        item["note"] = output("```json\n" + body + "\n```", prompt=request["prompt"], status="NOTE")
        receipt = rows.verify_note_v18(item, ledgers=ledgers, verifier=private, choices=[], gold="14", tokenizer=TOKENIZER)
        self.assertTrue(receipt["n_accepted"], receipt)
        row = rows.n_row_v18(item, tokenizer=TOKENIZER, receipt=receipt)
        selected = bytes(t for t, s in zip(row["generation"]["completion_token_ids"], row["loss_token_mask"]) if s).decode()
        self.assertEqual(selected, "7")

    def test_self_found_input_invariant_to_private_verdicts_and_location_condition(self):
        for condition in ("self_found", "location_given"):
            item = source_record()
            item["problem_id"] = next("fixture" + str(n) for n in range(100) if rows.note_condition_v18("fixture" + str(n)) == condition)
            private = verifier(item)
            ledger = rows.audit_t1_v18(item, verifier=private, tokenizer=TOKENIZER)
            request = rows.build_note_request_v18(item, ledgers=[ledger], gold="14")
            changed = {k: deepcopy(v) for k, v in ledger.items() if k != "sha256"}
            changed["claims"][0]["verdict"] = "VERIFIED"
            other = rows.build_note_request_v18(item, ledgers=[seal(changed)], gold="14")
            body = json.loads(request["prompt"].split("\n\n", 1)[1])
            self.assertNotIn("claims", body)
            self.assertNotIn("verdict", request["prompt"])
            if condition == "self_found":
                self.assertEqual(request["prompt"], other["prompt"])
                self.assertNotIn("inspect_passage", body)
            else:
                self.assertEqual(body["inspect_passage"], {"stage": "T1", "quote": "AB = 9"})
                self.assertNotIn("7", json.dumps(body["inspect_passage"]))

    def test_duplicate_quote_fragment_doubt_negation_retraction_and_units_rejected(self):
        for text, quote, unit in [("AB = 9\nAB = 9\nFinal answer: B", "AB = 9", "length"),
                                 ("Maybe AB = 9\nFinal answer: B", "AB = 9", "length"),
                                 ("AB = 9 is not true\nFinal answer: B", "AB = 9", "length"),
                                 ("AB = 9\nActually AB = 7\nFinal answer: B", "AB = 9", "length"),
                                 ("AB = 9\nFinal answer: B", "AB =", "length"),
                                 ("AB = 9\nFinal answer: B", "AB = 9", "m")]:
            with self.subTest(text=text, quote=quote, unit=unit):
                item = source_record(text)
                receipt, _ = repair(item, verifier(item), quote=quote, unit=unit)
                self.assertFalse(receipt["n_accepted"], receipt)

    def test_wrong_entity_value_answer_copy_and_leaked_lesson_rejected(self):
        for corrected in ("99", "14"):
            item = source_record()
            receipt, _ = repair(item, verifier(item), corrected=corrected)
            self.assertFalse(receipt["n_accepted"], receipt)
        item = source_record()
        receipt, _ = repair(item, verifier(item), lesson="Use the printed 7 for segment AB.")
        self.assertTrue(receipt["n_accepted"], receipt)
        self.assertFalse(receipt["resolve_allowed"])

    def test_abstract_article_is_not_a_node_name_but_explicit_names_are_blocked(self):
        item = source_record()
        receipt, _ = repair(item, verifier(item), lesson="Read a printed label before calculating.")
        self.assertTrue(receipt["resolve_allowed"], receipt)
        for lesson in ("Check node a before calculating.", "Read segment AB before calculating."):
            item = source_record()
            receipt, _ = repair(item, verifier(item), lesson=lesson)
            self.assertFalse(receipt["resolve_allowed"], receipt)

    def test_attached_unit_requires_complete_quote_and_same_unit(self):
        for quote, expected in (("AB = 9cm", True), ("AB = 9", False)):
            item = source_record("AB = 9cm\nFinal answer: B")
            private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "7", "unit": "cm"}])
            receipt, _ = repair(item, private, quote=quote, unit="cm")
            self.assertEqual(receipt["n_accepted"], expected, receipt)

    def test_answer_equal_arithmetic_repair_requires_original_false_equation(self):
        item = source_record("2 + 2 = 5\nFinal answer: B")
        ledger = rows.audit_t1_v18(item, verifier=None, tokenizer=TOKENIZER)
        prompt = rows.build_note_request_v18(item, ledgers=[ledger], gold="4")["prompt"]
        item["note"] = output(json.dumps({"repairs": [{"stage": "T1", "quote": "2 + 2 = 5",
            "corrected_value": "4", "unit": ""}], "lesson": "Recompute each elementary arithmetic operation."}),
            prompt=prompt, status="NOTE")
        receipt = rows.verify_note_v18(item, ledgers=[ledger], verifier=None, choices=[], gold="4", tokenizer=TOKENIZER)
        self.assertTrue(receipt["n_accepted"], receipt)
        self.assertEqual(receipt["repairs"][0]["repair_scope"], "VERIFIED_ARITHMETIC")
        self.assertFalse(receipt["repairs"][0]["intermediate_visual"])
        # A source label equal to the final answer still cannot be relabeled as intermediate repair.
        item = source_record()
        private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "14"}])
        receipt, _ = repair(item, private, corrected="14")
        self.assertFalse(receipt["n_accepted"])
        self.assertIn("FINAL_ANSWER_COPY_NOT_INTERMEDIATE_REPAIR", receipt["reasons"])

    def test_empty_historical_note_never_fabricates_repair(self):
        item = source_record()
        private = verifier(item)
        ledgers = [rows.audit_t1_v18(item, verifier=private, tokenizer=TOKENIZER)]
        prompt = rows.build_note_request_v18(item, ledgers=ledgers, gold="14")["prompt"]
        item["note"] = output(json.dumps({"repairs": [], "lesson": "Read each printed label carefully."}), prompt=prompt, status="NOTE")
        receipt = rows.verify_note_v18(item, ledgers=ledgers, verifier=private, choices=[], gold="14", tokenizer=TOKENIZER)
        self.assertFalse(receipt["n_accepted"])
        self.assertEqual(receipt["verified_repair_count"], 0)
        self.assertTrue(receipt["resolve_allowed"])


class V18Admission(unittest.TestCase):
    def test_verified_fact_must_be_used_and_all_operands_grounded(self):
        cases = [("AB = 7\n7 * 2 = 14\nFinal answer: 14", "SOURCE_GROUNDED"),
                 ("AB = 7\nUse the overall shape to decide.\nFinal answer: 14", "RESULT_ONLY"),
                 ("AB = 7\n7 * 3 = 21\nFinal answer: 14", "RESULT_ONLY"),
                 ("2 * 7 = 14\nFinal answer: 14", "RESULT_ONLY")]
        for text, grade in cases:
            item = source_record(text, status="CORRECT")
            item["t1"]["extracted"] = "14"
            receipt = rows.verify_solution_v18(item, tokenizer=TOKENIZER, gold="14", verifier=verifier(item))
            self.assertTrue(receipt["accepted"], receipt)
            self.assertEqual(receipt["evidence_grade"], grade)
            row = rows.a1_row_v18(item, tokenizer=TOKENIZER, receipt=receipt)
            self.assertEqual(any(row["verified_process_token_mask"]), grade == "SOURCE_GROUNDED")

    def test_unit_mismatch_is_not_grounded_or_supervised(self):
        item = source_record("AB = 7m\n7 * 2 = 14\nFinal answer: 14", status="CORRECT")
        item["t1"]["extracted"] = "14"
        private = verifier(item, [{"kind": "SEGMENT_LABEL", "entity": ["A", "B"], "value": "7", "unit": "cm"}])
        receipt = rows.verify_solution_v18(item, tokenizer=TOKENIZER, gold="14", verifier=private)
        self.assertFalse(receipt["accepted"])
        self.assertIn("PROCESS_UNIT_UNVERIFIED", receipt["reasons"])
        self.assertEqual(receipt["evidence_grade"], "RESULT_ONLY")

    def test_entity_word_overlap_alone_is_not_process_use(self):
        item = source_record("AB = 7\nTherefore AB is mentioned when selecting the answer.\nFinal answer: 14", status="CORRECT")
        item["t1"]["extracted"] = "14"
        receipt = rows.verify_solution_v18(item, tokenizer=TOKENIZER, gold="14", verifier=verifier(item))
        self.assertEqual(receipt["evidence_grade"], "RESULT_ONLY")

    def test_verified_graph_path_is_a_grounded_use(self):
        item = source_record("edge (0,1)\nPath: 0 -> 1 -> 2\nFinal answer: A", status="CORRECT")
        item.update(layer="graph", question="Find a path across the graph.")
        facts = [{"kind": "VISIBLE_NODE", "entity": i, "value": True} for i in range(3)]
        facts += [{"kind": "EDGE_PRESENT", "entity": edge, "value": True} for edge in ([0, 1], [1, 2])]
        private = verifier(item, facts, supported_claim_kinds=["VISIBLE_NODE", "EDGE_PRESENT"], closed_world_kinds=["EDGE_PRESENT"])
        receipt = rows.verify_solution_v18(item, tokenizer=TOKENIZER, gold="A", verifier=private)
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["evidence_grade"], "SOURCE_GROUNDED")

    def test_invalid_note_lesson_returns_rejection_receipt(self):
        item = source_record()
        item["note"] = output("{}", status="NOTE")
        item["resolve"] = output("Final answer: 14", extracted="14")
        note_receipt = seal({"lesson": None, "resolve_allowed": False})
        receipt = rows.verify_solution_v18(item, stage="resolve", tokenizer=TOKENIZER, gold="14", note_receipt=note_receipt)
        self.assertFalse(receipt["accepted"])
        self.assertIn("INVALID_ABSTRACT_LESSON", receipt["reasons"])

    def test_correction_use_bonus_only_for_same_repaired_fact(self):
        item = source_record()
        private = verifier(item)
        note_receipt, _ = repair(item, private)
        prompt = rows.previous.build_resolve_prompt_v16b(item["p0_text"], note_receipt["lesson"])
        item["resolve"] = output("AB = 7\n7 * 2 = 14\nFinal answer: 14", prompt=prompt, extracted="14")
        receipt = rows.verify_solution_v18(item, stage="resolve", tokenizer=TOKENIZER, gold="14", note_receipt=note_receipt, verifier=private)
        self.assertTrue(receipt["accepted"], receipt)
        self.assertTrue(receipt["corrected_use_byte_spans"])
        row = rows.c_row_v18(item, tokenizer=TOKENIZER, receipt=receipt)
        self.assertTrue(any(row["corrected_use_token_mask"]))
        self.assertEqual(row["prompt"], item["p0_text"])
        self.assertNotEqual(row["generation_prompt_sha256"], row["training_prompt_sha256"])

    def test_repeated_same_answer_with_units_is_not_conflicting(self):
        item = source_record("Compute the displayed product carefully.\nFinal answer: 14\nFinal answer: 14", status="CORRECT")
        item["t1"]["extracted"] = "14"
        receipt = rows.verify_solution_v18(item, tokenizer=TOKENIZER, gold="14")
        self.assertNotIn("MULTIPLE_FINAL_ANSWERS", receipt["reasons"])
        item = source_record("Compute the displayed product carefully.\nFinal answer: 14\nFinal answer: 15", status="CORRECT")
        item["t1"]["extracted"] = "14"
        self.assertIn("MULTIPLE_FINAL_ANSWERS", rows.verify_solution_v18(item, tokenizer=TOKENIZER, gold="14")["reasons"])

    def test_repeat_mask_does_not_penalize_unrelated_content_or_length(self):
        line = "Recheck the same already completed calculation."
        text = "A unique correct opening calculation.\n" + (line + "\n") * 4
        item = source_record(text)
        policy = rows.t1_policy_receipt_v18(item, tokenizer=TOKENIZER, gold="14")
        self.assertEqual(policy["reason"], "LOOP_CONFIRMED")
        self.assertFalse(any(policy["loss_token_mask"][:len("A unique correct opening calculation.\n".encode())]))
        self.assertTrue(any(policy["loss_token_mask"]))
        text = "A long but distinct derivation with no verified repetition."
        item = source_record()
        item["t1"] = output(text, prompt=item["p0_text"], status="TRUNCATED", finish="length")
        policy = rows.t1_policy_receipt_v18(item, tokenizer=TOKENIZER, gold="14")
        self.assertFalse(policy["pg"])
        self.assertFalse(any(policy["loss_token_mask"]))

    def test_regular_policy_keeps_original_mask_and_changed_receipt_rejected(self):
        item = source_record("Check the image and compare each printed label.\nFinal answer: 14", status="CORRECT")
        item["t1"]["extracted"] = "14"
        policy = rows.t1_policy_receipt_v18(item, tokenizer=TOKENIZER, gold="14")
        self.assertTrue(all(policy["loss_token_mask"]))
        receipt = rows.verify_solution_v18(item, tokenizer=TOKENIZER, gold="14")
        item["t1"]["text"] += " altered"
        with self.assertRaisesRegex(ValueError, "OUTPUT_CHANGED"):
            rows.a1_row_v18(item, tokenizer=TOKENIZER, receipt=receipt)


if __name__ == "__main__":
    unittest.main()
