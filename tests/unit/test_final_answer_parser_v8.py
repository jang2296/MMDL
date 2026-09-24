from __future__ import annotations

import unittest

from mmdl.evaluation import final_answer_parser_v8 as v8
from mmdl.evaluation.parsers import score_with_parser


COLORS = {"A": "red", "B": "blue", "C": "green", "D": "yellow"}


class V8ContractTests(unittest.TestCase):
    def test_version_and_comparison_policy_are_pinned(self) -> None:
        self.assertEqual(v8.VERSION, "team-final-answer-v8")
        self.assertEqual(v8.COMPARISON_POLICY, "single-final-mmmu-round2-v1")

    def test_extractors_return_the_legacy_result_shape(self) -> None:
        for result in (
            v8.extract_mcq("Final answer: B.", COLORS, question="Choose one option."),
            v8.extract_open("Final answer: 2.", question="Which step is blocked?"),
        ):
            self.assertEqual(set(result), {"answer", "status", "rule", "evidence"})
            self.assertEqual(result["status"], v8.PARSED)
            self.assertTrue(result["rule"])
            self.assertTrue(result["evidence"])

        scored = v8.score_response("Final answer: B.", "mcq", COLORS, "B")
        self.assertTrue(
            {"parsed_answer", "parse_status", "correct", "extraction_rule",
             "extraction_evidence", "extraction_status"}.issubset(scored)
        )

    def test_empty_and_invalid_question_type(self) -> None:
        self.assertEqual(v8.extract_open("  ")["status"], v8.EMPTY)
        with self.assertRaises(ValueError):
            v8.score_response("Answer: B", "pair", COLORS, "B")

    def test_repeated_stated_clauses_do_not_recurse_unbounded(self) -> None:
        raws = (
            "Final answer: " + "value is " * 1_500 + "10.",
            "Final answer: " + "x = " * 1_500 + r"\frac{1}{2}.",
        )
        for raw in raws:
            with self.subTest(terminal=raw[-30:]):
                result = v8.extract_open(raw, question="Give one numeric value.")
                self.assertIn(result["status"], {v8.PARSED, v8.NO_PARSE})
                self.assertEqual(set(result), {"answer", "status", "rule", "evidence"})


class SpecificationRegressionTests(unittest.TestCase):
    """The 33 diagnosis cases, with the approved V8 policy adaptations."""

    def assert_case(
        self,
        raw: str,
        question_type: str,
        options: object,
        gold: object,
        expected_correct: bool,
        expected_status: str,
        *,
        question: str,
        finish_reason: str = "stop",
    ) -> None:
        result = v8.score_response(
            raw,
            question_type,
            options,  # type: ignore[arg-type]
            gold,
            finish_reason,
            question=question,
        )
        self.assertEqual(result["correct"], expected_correct, raw)
        self.assertEqual(result["parse_status"], expected_status, raw)

    def test_mcq_minimal_pairs(self) -> None:
        cases = (
            ("Final answer: B is correct.", COLORS, "B", True, v8.PARSED),       # P01
            ("Final answer: B is not correct.", COLORS, "B", False, v8.NO_PARSE), # N01
            ("Final answer: B.", COLORS, "B", True, v8.PARSED),                  # P02
            ("Final answer: B or C.", COLORS, "B", False, v8.NO_PARSE),          # N02
            ("Final answer: B. Not enough information.",
             {"A": "Yes", "B": "Not enough information", "C": "No"},
             "B", True, v8.PARSED),                                               # P03
            ("Final answer: B. Not enough information. I retract that choice; "
             "B is not correct.",
             {"A": "Yes", "B": "Not enough information", "C": "No"},
             "B", False, v8.NO_PARSE),                                            # N03
            (r"Final answer: A. $L \sin \theta$",
             {"A": "L sin θ", "B": "L cos θ", "C": "L", "D": "L tan θ"},
             "A", True, v8.PARSED),                                               # P04
            (r"Final answer: \boxed{\text{C. }2\pi \times 10^{-5}\,\text{A}}",
             {"A": "3π×10^-5 A", "B": "π×10^-5 A", "C": "2π×10^-5 A",
              "D": "π×10^-4 A"},
             "C", True, v8.PARSED),                                               # P05
            ("Final answer: C or A.", COLORS, "C", False, v8.NO_PARSE),          # N05
            ("> **Final answer: B.**", COLORS, "B", True, v8.PARSED),            # P10
            ('A textbook example says: "Final answer: B." I am only quoting it '
             "and have not selected an answer.", COLORS, "B", False, v8.NO_PARSE), # N10
            ("I previously chose B. I retract B. Final answer: C.",
             COLORS, "C", True, v8.PARSED),                                       # P11
            ("Final answer: B.\n\nFinal answer: C.",
             COLORS, "C", False, v8.NO_PARSE),                                    # N11
            ("I cannot determine a unique answer.", COLORS, "B", False, v8.NO_PARSE), # N18
        )
        for raw, options, gold, correct, status in cases:
            with self.subTest(raw=raw):
                self.assert_case(raw, "mcq", options, gold, correct, status,
                                 question="Choose one option.")

    def test_length_minimal_pair(self) -> None:
        self.assert_case(
            "Final answer: B.\nThe supporting calculation starts with",
            "mcq", COLORS, "B", True, v8.PARSED,
            question="Choose one option.", finish_reason="length",              # P12
        )
        self.assert_case(
            "Final answer: B.\nHowever, I withdraw that answer. The correct choice may be",
            "mcq", COLORS, "B", False, v8.NO_PARSE,
            question="Choose one option.", finish_reason="length",              # N12
        )

    def test_open_minimal_pairs(self) -> None:
        cases = (
            ("Final answer: 10 V.", "What is the voltage in V?", "10", True,
             v8.PARSED),                                                           # P06
            ("Final answer: 10 A.", "What is the voltage in V?", "10", False,
             v8.PARSED),                                                           # N06
            ("Final answer: Step 2.", "Which step is blocked?", "2", True,
             v8.PARSED),                                                           # P07
            ("Final answer: Step 2 or Step 3.", "Which step is blocked?", "2",
             False, v8.NO_PARSE),                                                  # N07
            ("Final answer: cat.", "Name the object.", "cat", True, v8.PARSED), # P08
            ("Final answer: caterpillar.", "Name the object.", "cat", False,
             v8.PARSED),                                                           # N08
            ("Final answer: 50.", "Give one numeric value.", "50", True,
             v8.PARSED),                                                           # P09
            ("Final answer: 50 or 100.", "Give one numeric value.", "50", False,
             v8.NO_PARSE),                                                         # N09
            ("An intermediate value was 50. Final answer: 75.",
             "Give one numeric value.", "75", True, v8.PARSED),                 # P13
            ("An intermediate value was 50. Final answer: 75.",
             "Give one numeric value.", "50", False, v8.PARSED),                # N13
            ("Final answer: 1/2.", "Give the ratio.", "0.5", True, v8.PARSED), # P14
            ("Final answer: -10.", "Give the signed value.", "10", False,
             v8.PARSED),                                                           # N14
            ("Final answer: (50, 100).", "Give the ordered pair (x, y).",
             ["50", "100"], False, v8.NO_PARSE),                                 # P15 adapted
            ("Final answer: (50, 100).", "Give the ordered pair (x, y).",
             ["50", "200"], False, v8.NO_PARSE),                                 # N15 adapted
            ("Final answer: 0.9632.", "Give the numeric value.", "0.963", True,
             v8.PARSED),                                                           # P17 adapted
            ("Final answer: Point A.", "Which point is marked?", "A", True,
             v8.PARSED),                                                           # P18
        )
        for raw, question, gold, correct, status in cases:
            with self.subTest(raw=raw, question=question):
                self.assert_case(raw, "open", [], gold, correct, status,
                                 question=question)

    def test_named_point_commitment_and_revocation(self) -> None:
        cases = (
            ("Final answer: Point A.", True, v8.PARSED),
            ("Final answer: Point A shows the marked location.", True, v8.PARSED),
            ("Final answer: Point A isn't correct.", False, v8.NO_PARSE),
            ("Final answer: Point A might be correct.", False, v8.NO_PARSE),
            ("Final answer: Point A.\n\nPoint A isn't correct.", False, v8.NO_PARSE),
            ("Final answer: Point A.\n\nPoint A might be correct.", False, v8.NO_PARSE),
        )
        for raw, correct, status in cases:
            with self.subTest(raw=raw):
                self.assert_case(raw, "open", [], "A", correct, status,
                                 question="Which point is marked?")

    def test_revoked_mcq_candidate_allows_a_new_final_answer(self) -> None:
        options = {"A": "a palace", "B": "a necropolis", "C": "a market", "D": "a temple"}
        revoked = v8.score_response(
            "The correct answer is D. a temple? No — temples are for worship.\n\n"
            "Final answer: B. a necropolis",
            "mcq", options, "B", question="Choose one option.",
        )
        unresolved = v8.score_response(
            "The correct answer is D. a temple.\n\nFinal answer: B. a necropolis",
            "mcq", options, "B", question="Choose one option.",
        )
        self.assertEqual((revoked["parsed_answer"], revoked["parse_status"]),
                         ("B", v8.PARSED))
        self.assertTrue(revoked["correct"])
        self.assertEqual(unresolved["parse_status"], v8.NO_PARSE)
        self.assertFalse(unresolved["correct"])

        literal_question_mark = v8.score_response(
            "Final answer: A. Maybe? No",
            "mcq", {"A": "Maybe? No", "B": "Always"}, "A",
            question="Choose one option.",
        )
        self.assertTrue(literal_question_mark["correct"])

    def test_explicit_self_correction_replaces_an_earlier_choice(self) -> None:
        corrected = v8.score_response(
            "Final answer: B. I need to correct myself. Final answer: C.",
            "mcq", COLORS, "C", question="Choose one option.",
        )
        unresolved = v8.score_response(
            "Final answer: B. Final answer: C.",
            "mcq", COLORS, "C", question="Choose one option.",
        )
        self.assertEqual((corrected["parsed_answer"], corrected["parse_status"]),
                         ("C", v8.PARSED))
        self.assertTrue(corrected["correct"])
        self.assertEqual(unresolved["parse_status"], v8.NO_PARSE)
        self.assertFalse(unresolved["correct"])

    def test_label_text_conflict_is_unresolved(self) -> None:
        self.assert_case("Final answer: B. green.", "mcq", COLORS, "B", False,
                         v8.NO_PARSE, question="Choose one option.")               # N16

    def test_later_withdrawal_or_unresolved_alternative_blocks_fallback(self) -> None:
        cases = (
            (r"\boxed{B}. I retract B.", "boxed answer later withdrawn"),
            ("Final answer: B. Correction: C. I withdraw C.",
             "replacement later withdrawn"),
            ("Final answer: B and C.", "joined alternatives"),
            ("Final answer: B, C.", "comma-separated alternatives"),
            ("Final answer: B nor C.", "nor-separated alternatives"),
            ("Final answer: B. C is also correct.", "later competing assertion"),
            ("Final answer: B isn't correct.", "contracted negation"),
            ("Final answer: B cannot be correct.", "modal negation"),
            ("Final answer: B. Correction:", "malformed latest correction"),
        )
        for raw, description in cases:
            with self.subTest(description=description):
                result = v8.score_response(
                    raw, "mcq", COLORS, "B", question="Choose one option."
                )
                self.assertEqual(result["parse_status"], v8.NO_PARSE)
                self.assertFalse(result["correct"])

    def test_attributed_answer_does_not_hide_later_owned_answer(self) -> None:
        result = v8.score_response(
            "The source says: Final answer: B. After checking, my final answer: C.",
            "mcq", COLORS, "C", question="Choose one option.",
        )
        self.assertEqual((result["parsed_answer"], result["parse_status"]),
                         ("C", v8.PARSED))
        self.assertTrue(result["correct"])


class MetamorphicAndRecoveredCaseTests(unittest.TestCase):
    def test_gold_changes_only_correctness(self) -> None:
        raw = "An intermediate value was 50. Final answer: 75."
        first = v8.score_response(raw, "open", [], "75", question="Give one value.")
        second = v8.score_response(raw, "open", [], "50", question="Give one value.")
        for key in ("parsed_answer", "parse_status", "extraction_rule",
                    "extraction_evidence", "extraction_status"):
            self.assertEqual(first[key], second[key])
        self.assertTrue(first["correct"])
        self.assertFalse(second["correct"])

    def test_units_come_only_from_the_question_and_are_not_converted(self) -> None:
        cases = (
            ("Final answer: 10 V.", "What is the voltage in V?"),
            ("Final answer: 1000 mV.", "What is the voltage in V?"),
            ("Final answer: 10 V.", "What is the value?"),
            ("Final answer: 10 A.", "What occurs in a circuit?"),
        )
        expected = (True, False, False, False)
        for (raw, question), correct in zip(cases, expected, strict=True):
            with self.subTest(raw=raw, question=question):
                result = v8.score_response(raw, "open", [], "10", question=question)
                self.assertEqual(result["correct"], correct)

    def test_implied_unit_is_equivalent_but_explicit_mismatch_conflicts(self) -> None:
        equivalent = v8.score_response(
            "Final answer: $0.286 per share.\nAnswer: $0.286",
            "open", [], "0.286", question="What is FCFE per share?",
        )
        mismatch = v8.score_response(
            "Final answer: $0.286 per share.\nAnswer: 0.286 V",
            "open", [], "0.286", question="What is FCFE per share?",
        )
        self.assertEqual((equivalent["parsed_answer"], equivalent["parse_status"]),
                         ("0.286", v8.PARSED))
        self.assertTrue(equivalent["correct"])
        self.assertEqual(mismatch["parse_status"], v8.NO_PARSE)
        self.assertFalse(mismatch["correct"])

    def test_option_permutation_maps_content_to_the_new_label(self) -> None:
        raw = "Final answer: green."
        original = v8.score_response(raw, "mcq", COLORS, "C")
        permuted = v8.score_response(
            raw, "mcq", {"A": "green", "B": "red", "C": "yellow", "D": "blue"}, "A"
        )
        self.assertEqual((original["parsed_answer"], permuted["parsed_answer"]), ("C", "A"))
        self.assertTrue(original["correct"] and permuted["correct"])

    def test_unknown_label_can_be_exact_option_content(self) -> None:
        valid_label = v8.score_response(
            "Final answer: B.", "mcq", {"A": "B", "B": "C", "C": "D"}, "B",
            question="Choose one option.",
        )
        unknown_label = v8.score_response(
            "Final answer: E.", "mcq", ["E", "B", "C", "A"], "A",
            question="Which diagram region contains the hydrophobic tails?",
        )
        self.assertEqual((valid_label["parsed_answer"], valid_label["parse_status"]),
                         ("B", v8.PARSED))
        self.assertEqual((unknown_label["parsed_answer"], unknown_label["parse_status"]),
                         ("A", v8.PARSED))
        self.assertTrue(valid_label["correct"] and unknown_label["correct"])

    def test_recovered_terminal_conclusion_and_wrappers(self) -> None:
        embryonic = v8.score_response(
            "Therefore, the red arrow belongs to the embryonic stage.",
            "open", [], "embryonic", question="Which prenatal stage is marked?",
        )
        boxed = v8.score_response(
            r"Final Answer: $65$. Final Answer: $\boxed{65}$",
            "open", [], "65", question="What is the value?",
        )
        gallbladder = v8.score_response(
            "C. Gallbladder", "mcq",
            {"A": "Liver", "B": "Pancreas", "C": "Gallbladder", "D": "Spleen"},
            "C", question="Which organ is indicated?",
        )
        self.assertEqual(embryonic["parsed_answer"], "embryonic")
        self.assertEqual(boxed["parsed_answer"], "65")
        self.assertEqual(gallbladder["parsed_answer"], "C")
        self.assertTrue(embryonic["correct"] and boxed["correct"] and gallbladder["correct"])

    def test_wrapper_dispatch_matches_direct_v8_scoring(self) -> None:
        args = ("Final answer: B.", "mcq", COLORS, "B")
        direct = v8.score_response(*args, question="Choose one option.")
        wrapped = score_with_parser(
            *args, parser_id=v8.VERSION, question="Choose one option."
        )
        self.assertEqual(wrapped, direct)

        quantity = score_with_parser(
            "Final answer: 10 V.", "open", [], "10", parser_id=v8.VERSION,
            question="What is the voltage in V?",
        )
        self.assertTrue(quantity["correct"])


if __name__ == "__main__":
    unittest.main()
