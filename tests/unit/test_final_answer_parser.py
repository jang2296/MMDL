from __future__ import annotations

import unittest

from mmdl.evaluation.final_answer_parser import (
    EMPTY,
    NO_PARSE,
    PARSED,
    VERSION,
    extract_mcq,
    extract_open,
    score_response,
)


OPTIONS = {
    "A": "Conservative treatment",
    "B": "A ruptured abdominal aortic aneurysm",
    "C": "1",
    "D": "14/6",
}


class McqExtractionTests(unittest.TestCase):
    def test_version_is_distinct_team_policy(self) -> None:
        self.assertEqual(VERSION, "team-final-answer-v4")

    def test_final_assertion_beats_article_a(self) -> None:
        result = extract_mcq(
            "Correct Answer: **B. A ruptured abdominal aortic aneurysm.**", OPTIONS
        )
        self.assertEqual((result["answer"], result["status"]), ("B", PARSED))

    def test_final_assertion_does_not_treat_article_a_as_label(self) -> None:
        options = ["red", "blue"]
        result = extract_mcq("Final answer: a common misconception.", options)
        self.assertEqual((result["status"], result["rule"]), (NO_PARSE, "invalid_final"))
        for raw in ("Final answer: A", "Final answer: A.", "Final answer: A red"):
            with self.subTest(raw=raw):
                self.assertEqual(extract_mcq(raw, options)["answer"], "A")

    def test_final_assertion_beats_excluded_parenthesized_choice(self) -> None:
        raw = "Localized ileus (D) is excluded.\nFinal answer: B."
        self.assertEqual(extract_mcq(raw, OPTIONS)["answer"], "B")

    def test_explicit_label_avoids_numeric_option_prefix(self) -> None:
        result = extract_mcq("Final Answer: **D. 14/6**", OPTIONS)
        self.assertEqual(result["answer"], "D")

    def test_conflicting_final_assertions_abstain(self) -> None:
        result = extract_mcq("Final answer: A.\nFinal answer: B.", OPTIONS)
        self.assertEqual(result["status"], NO_PARSE)
        self.assertEqual(result["rule"], "conflicting_final")

    def test_invalid_box_abstains(self) -> None:
        result = extract_mcq(r"The result is \boxed{Z}.", OPTIONS)
        self.assertEqual((result["answer"], result["rule"]), (None, "invalid_box"))

    def test_valid_boxed_label_is_explicit(self) -> None:
        result = extract_mcq(r"After checking, \boxed{D}", OPTIONS)
        self.assertEqual((result["answer"], result["rule"]), ("D", "boxed_answer"))

    def test_final_correction_beats_stale_box(self) -> None:
        result = extract_mcq(r"An earlier guess was \boxed{A}. Final answer: B.", OPTIONS)
        self.assertEqual((result["answer"], result["rule"]), ("B", "final_assertion"))

    def test_short_label_and_exact_label_option(self) -> None:
        self.assertEqual(extract_mcq("(C)", OPTIONS)["answer"], "C")
        self.assertEqual(extract_mcq("D. 14/6", OPTIONS)["answer"], "D")
        self.assertEqual(extract_mcq("C. 14/6", OPTIONS)["status"], NO_PARSE)

    def test_rationale_without_final_does_not_fall_back(self) -> None:
        result = extract_mcq("A possibility is (D), but more work is required.", OPTIONS)
        self.assertEqual((result["answer"], result["status"]), (None, NO_PARSE))

    def test_truncated_response_requires_complete_final(self) -> None:
        result = extract_mcq("Work continues. Final answer:", OPTIONS, finish_reason="length")
        self.assertEqual((result["status"], result["rule"]),
                         (NO_PARSE, "length_without_complete_final"))

    def test_length_response_accepts_a_complete_explicit_final(self) -> None:
        result = extract_mcq("Work repeats. Final answer: B", OPTIONS, finish_reason="length")
        self.assertEqual((result["answer"], result["status"]), ("B", PARSED))

    def test_v2_format_coverage_without_v2_fallback_policy(self) -> None:
        options = ["red", "blue", "green", "yellow"]
        cases = [
            ("C. green", "C"),
            ("(C)", "C"),
            ("C", "C"),
            ("**C. green**", "C"),
            ("B. blue", "B"),
            ("The correct answer is: **B. blue**", "B"),
            ("Correct Answer:\n> **A. red**", "A"),
            ("Final Answer: **D. yellow**", "D"),
            ("Final Answer\n\n\\boxed{C}", "C"),
            ("Final Answer: \\boxed{\\text{C}}", "C"),
            ("### Final Answer:\n\\[\n\\boxed{D. yellow}\n\\]", "D"),
            ("Final Answer:\n> **C. green**", "C"),
            ("Final decision: **B. blue**", "B"),
            ("Answer: **C**", "C"),
            ("The answer is C.", "C"),
            ("Correct option: A", "A"),
            ("B is tempting. So \\boxed{C}", "C"),
            ("So \\boxed{C}. Even though uncertain, perhaps it is intended.", "C"),
            ("blue", "B"),
            ("We reject (D). Final Answer: B.", "B"),
            ("We reject (A). Final Answer: C.", "C"),
            ("Answer is not B. Answer: C.", "C"),
            (r"\boxed{C}. However, the answer is B.", "B"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(extract_mcq(raw, options)["answer"], expected)

    def test_actual_final_outranks_earlier_generic_assertion(self) -> None:
        options = ["red", "blue", "green", "yellow"]
        result = extract_mcq(
            "Correct answer is D. Later reconsidered. Final Answer: A", options
        )
        self.assertEqual(result["answer"], "A")

    def test_rationale_mentions_do_not_poison_final_section(self) -> None:
        cases = [
            "The answer choices need checking. The correct decision depends on the graph.\nFinal Answer: B.",
            "The answer is an intermediate estimate.\nFinal Answer:\nAn estimate is 6.\nCorrect option: B.",
            "Reasoning with (A) and (C) did not resolve it.\nB. blue",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(extract_mcq(raw, ["red", "blue", "green"])["answer"], "B")
        self.assertEqual(extract_mcq("Answer: B.\nFinal Answer: Z", ["red", "blue"])["status"], NO_PARSE)

    def test_conclusion_prose_is_not_an_answer_declaration(self) -> None:
        raw = "Correct Answer: B.\nConclusion: this is a descriptive conclusion.\nCorrect Answer: B."
        self.assertEqual(extract_mcq(raw, ["red", "blue"])["answer"], "B")
        self.assertEqual(extract_mcq("The answer is 78 but not an option.\nAnswer: B.",
                                     ["red", "blue"])["answer"], "B")
        self.assertEqual(extract_mcq("Final Answer: A — red", ["red", "blue"])["answer"], "A")

    def test_incomplete_earlier_final_does_not_override_terminal_answer(self) -> None:
        self.assertEqual(extract_mcq("Final answer: not yet known.\nFinal Answer: B.",
                                     ["red", "blue"])["answer"], "B")

    def test_hypothetical_answer_markers_do_not_conflict_with_final(self) -> None:
        options = ["red", "blue", "green", "yellow"]
        cases = [
            "Perhaps the correct answer is B. Final Answer: C",
            "If the correct answer is B, one implication follows. Final Answer: C",
            "Whether the answer is B remains unclear. Final Answer: C",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(extract_mcq(raw, options)["answer"], "C")

    def test_explicit_answer_connectors(self) -> None:
        options = ["red", "blue", "green", "yellow"]
        for raw in ("The answer should be C", "Answer would be C", "Correct answer would be C"):
            with self.subTest(raw=raw):
                self.assertEqual(extract_mcq(raw, options)["answer"], "C")

    def test_nested_tex_box_option_text_and_duplicate_content(self) -> None:
        options = ["58.4 kip", "125 kip", "50.4 kip", "100 kip"]
        self.assertEqual(
            extract_mcq(r"Final Answer: \boxed{50.4 \text{ kip}}", options)["answer"], "C"
        )
        result = extract_mcq(r"Final Answer: \boxed{42}", ["42", "42"])
        self.assertEqual((result["status"], result["rule"]), (NO_PARSE, "invalid_final"))

    def test_nfkc_normalization(self) -> None:
        self.assertEqual(extract_mcq("Ｆｉｎａｌ Ａｎｓｗｅｒ： Ｂ", ["red", "blue"])["answer"], "B")


class OpenExtractionTests(unittest.TestCase):
    def test_currency_commas_scientific_and_exact_fraction(self) -> None:
        self.assertTrue(score_response("Answer: $2,960", "open", [], "2960")["correct"])
        self.assertTrue(score_response("Final answer: 1e3", "open", [], "1000")["correct"])
        self.assertTrue(score_response(r"Answer: \frac{24}{7}", "open", [], "24/7")["correct"])
        self.assertTrue(score_response("Answer: 1/2", "open", [], "0.5")["correct"])

    def test_final_open_answer_overrides_intermediate_number(self) -> None:
        raw = "An intermediate estimate is 0.3.\nFinal answer: -1.83"
        result = score_response(raw, "open", [], "0.3")
        self.assertEqual(result["parsed_answer"], "-1.83")
        self.assertFalse(result["correct"])

    def test_open_conflict_abstains(self) -> None:
        result = extract_open("Final answer: 8\nFinal answer: 9")
        self.assertEqual((result["status"], result["rule"]), (NO_PARSE, "conflicting_final"))

    def test_truncated_open_final_abstains(self) -> None:
        result = extract_open("Calculation...\nFinal answer:", finish_reason="length")
        self.assertEqual((result["status"], result["rule"]),
                         (NO_PARSE, "length_without_complete_final"))

    def test_nonfinite_and_invalid_numeric_answers_abstain(self) -> None:
        for raw in ("Final answer: NaN", "Final answer: inf", "Final answer: 1/0"):
            with self.subTest(raw=raw):
                self.assertEqual(extract_open(raw)["status"], NO_PARSE)

    def test_numeric_policy_reuses_official_rounding_not_a_team_tolerance(self) -> None:
        self.assertFalse(score_response("Answer: 24.3", "open", [], "24.32")["correct"])
        self.assertTrue(score_response("Answer: 24.324", "open", [], "24.32")["correct"])


class ScoringContractTests(unittest.TestCase):
    def test_gold_changes_correctness_not_extraction(self) -> None:
        first = score_response("Final answer: B", "multiple-choice", OPTIONS, "A")
        second = score_response("Final answer: B", "multiple-choice", OPTIONS, "B")
        for key in ("parsed_answer", "parse_status", "extraction_rule",
                    "extraction_evidence", "extraction_status"):
            self.assertEqual(first[key], second[key])
        self.assertFalse(first["correct"])
        self.assertTrue(second["correct"])

    def test_empty_response_keeps_writer_status_contract(self) -> None:
        result = score_response("  ", "open", [], "anything")
        self.assertEqual(result["parse_status"], EMPTY)
        self.assertFalse(result["correct"])


if __name__ == "__main__":
    unittest.main()
