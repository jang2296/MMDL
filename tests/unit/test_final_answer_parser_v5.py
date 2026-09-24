from __future__ import annotations

import unittest

from mmdl.evaluation.final_answer_parser import (
    NO_PARSE,
    PARSED,
    extract_mcq,
    extract_open,
    score_response,
)
from mmdl.evaluation.parsers import score_with_parser


V4 = "team-final-answer-v4"
V5 = "team-final-answer-v5"
V6 = "team-final-answer-v6"
OPTIONS = ["Real", "Larger", "Other", "None"]


class ChronologicalDeclarationTests(unittest.TestCase):
    def test_later_answer_beats_earlier_correct_answer_for_mcq_and_open(self) -> None:
        cases = [
            ("Correct Answer: A. Real\nAnswer: D", "D"),
            ("Correct Answer:\n---\nAnswer: D", "D"),
            ("Correct Answer: Larger\nAnswer: D", "D"),
            ("Correct Answer: B. Larger\nAnswer: D (correction)", "D"),
            ("Suppose the correct answer is A. Real. After review, Answer: D", "D"),
            ("Answer: D\nCorrect Answer: A. Real", "A"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                result = extract_mcq(raw, OPTIONS, parser_id=V5)
                self.assertEqual((result["answer"], result["status"]), (expected, PARSED))

        opened = extract_open("Correct Answer: 3\nAnswer: 8", parser_id=V5)
        self.assertEqual((opened["answer"], opened["status"]), ("8", PARSED))

    def test_final_section_uses_latest_declaration_inside_section(self) -> None:
        raw = "Correct Answer: A. Real\nFinal Answer: A. Real\nCorrect Answer: B. Larger\nAnswer: D"
        self.assertEqual(extract_mcq(raw, OPTIONS, parser_id=V5)["answer"], "D")

    def test_later_hypothetical_declarations_do_not_replace_valid_answer(self) -> None:
        for hypothetical in (
            "If the answer is B, that would be plausible.",
            "Whether the answer is B remains uncertain.",
            "Perhaps the answer is B, but this is only a possibility.",
        ):
            with self.subTest(hypothetical=hypothetical):
                raw = f"Correct Answer: A. Real\n{hypothetical}"
                self.assertEqual(extract_mcq(raw, OPTIONS, parser_id=V5)["answer"], "A")

    def test_later_hypothetical_inside_final_section_does_not_replace_answer(self) -> None:
        raw = "Final Answer: A. Real\nPerhaps the answer is B, but this is only a possibility."
        self.assertEqual(extract_mcq(raw, OPTIONS, parser_id=V5)["answer"], "A")

    def test_v4_default_and_explicit_version_keep_historical_preference(self) -> None:
        raw = "Correct Answer: A. Real\nAnswer: D"
        self.assertEqual(extract_mcq(raw, OPTIONS)["answer"], "A")
        self.assertEqual(extract_mcq(raw, OPTIONS, parser_id=V4)["answer"], "A")


class AbstentionTests(unittest.TestCase):
    def test_invalid_latest_answer_does_not_fall_back(self) -> None:
        result = extract_mcq("Correct Answer: A. Real\nAnswer: Z", OPTIONS, parser_id=V5)
        self.assertEqual((result["answer"], result["status"]), (None, NO_PARSE))

    def test_conflicting_final_sections_abstain(self) -> None:
        result = extract_open("Final Answer: 8\nFinal Answer: 9", parser_id=V5)
        self.assertEqual((result["answer"], result["status"], result["rule"]),
                         (None, NO_PARSE, "conflicting_final"))

    def test_truncated_final_abstains(self) -> None:
        result = extract_mcq("Correct Answer: A. Real\nAnswer:", OPTIONS,
                             finish_reason="length", parser_id=V5)
        self.assertEqual((result["answer"], result["status"]), (None, NO_PARSE))

    def test_rationale_only_choice_does_not_use_random_or_legacy_fallback(self) -> None:
        for raw in ("Reasoning mentions (B), but no answer is selected.",
                    "A possibility is D; more work is required."):
            with self.subTest(raw=raw):
                result = extract_mcq(raw, OPTIONS, parser_id=V5)
                self.assertEqual((result["answer"], result["status"]), (None, NO_PARSE))

    def test_hypothetical_only_declaration_abstains(self) -> None:
        for raw in ("If the answer is B, that would be plausible.",
                    "Whether the answer is B remains uncertain.",
                    "Perhaps the answer is B, but this is only a possibility."):
            with self.subTest(raw=raw):
                result = extract_mcq(raw, OPTIONS, parser_id=V5)
                self.assertEqual((result["answer"], result["status"]), (None, NO_PARSE))


class VersionedScoringTests(unittest.TestCase):
    def test_open_intermediate_matching_gold_is_not_scored_as_answer(self) -> None:
        raw = "Intermediate estimate: 0.3\nAnswer: 1.2"
        result = score_response(raw, "open", [], "0.3", parser_id=V5)
        self.assertEqual(result["parsed_answer"], "1.2")
        self.assertFalse(result["correct"])

    def test_open_numeric_answer_survives_later_hypothetical(self) -> None:
        raw = "Answer: 8\nIf the answer is 9, that would follow from another assumption."
        result = extract_open(raw, parser_id=V5)
        self.assertEqual((result["answer"], result["status"]), ("8", PARSED))

    def test_gold_changes_scoring_but_not_v5_extraction(self) -> None:
        raw = "Correct Answer: A. Real\nAnswer: D"
        wrong_gold = score_response(raw, "multiple-choice", OPTIONS, "A", parser_id=V5)
        matching_gold = score_response(raw, "multiple-choice", OPTIONS, "D", parser_id=V5)
        self.assertEqual(wrong_gold["parsed_answer"], matching_gold["parsed_answer"])
        self.assertFalse(wrong_gold["correct"])
        self.assertTrue(matching_gold["correct"])

    def test_score_dispatch_uses_v5_extraction_not_score_only_adjustment(self) -> None:
        raw = "Correct Answer: A. Real\nAnswer: D"
        extracted = extract_mcq(raw, OPTIONS, parser_id=V5)
        scored = score_with_parser(raw, "multiple-choice", OPTIONS, "D", parser_id=V5)
        self.assertEqual(scored["parsed_answer"], extracted["answer"])
        self.assertTrue(scored["correct"])

    def test_unknown_parser_id_is_rejected_even_for_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            extract_mcq("", OPTIONS, parser_id="unknown")
        with self.assertRaises(ValueError):
            extract_open("", parser_id="unknown")
        with self.assertRaises(ValueError):
            score_response("", "open", [], "x", parser_id="unknown")


class LongHypotheticalContextTests(unittest.TestCase):
    LONG_HYPOTHETICAL = (
        "Perhaps, after reviewing the complete case and all the available evidence, "
        "the answer is B"
    )

    def test_v6_uses_full_sentence_context_while_v5_keeps_historical_result(self) -> None:
        raw = self.LONG_HYPOTHETICAL
        self.assertEqual(extract_mcq(raw, OPTIONS, parser_id=V5)["answer"], "B")
        result = extract_mcq(raw, OPTIONS, parser_id=V6)
        self.assertEqual((result["answer"], result["status"]), (None, NO_PARSE))

    def test_v6_keeps_prior_answer_before_long_hypothetical(self) -> None:
        raw = f"Answer: A\n{self.LONG_HYPOTHETICAL}"
        result = extract_mcq(raw, OPTIONS, parser_id=V6)
        self.assertEqual((result["answer"], result["status"]), ("A", PARSED))

    def test_v6_hypothetical_filter_applies_inside_final_sections(self) -> None:
        raw = f"Final Answer: A. Real\n{self.LONG_HYPOTHETICAL}"
        result = extract_mcq(raw, OPTIONS, parser_id=V6)
        self.assertEqual((result["answer"], result["status"]), ("A", PARSED))

    def test_v6_hypothetical_filter_applies_to_open_numeric_answers(self) -> None:
        raw = "Answer: 8\nPerhaps, after reviewing all the assumptions in detail, the answer is 9"
        result = extract_open(raw, parser_id=V6)
        self.assertEqual((result["answer"], result["status"]), ("8", PARSED))

    def test_v6_accepts_answer_after_hypothetical_sentence_or_newline(self) -> None:
        cases = (
            f"{self.LONG_HYPOTHETICAL}. Answer: D",
            f"{self.LONG_HYPOTHETICAL}\nAnswer: D",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(extract_mcq(raw, OPTIONS, parser_id=V6)["answer"], "D")

    def test_v6_extraction_is_gold_blind(self) -> None:
        raw = f"Answer: A\n{self.LONG_HYPOTHETICAL}"
        wrong_gold = score_response(raw, "multiple-choice", OPTIONS, "D", parser_id=V6)
        matching_gold = score_response(raw, "multiple-choice", OPTIONS, "A", parser_id=V6)
        self.assertEqual(wrong_gold["parsed_answer"], matching_gold["parsed_answer"])
        self.assertFalse(wrong_gold["correct"])
        self.assertTrue(matching_gold["correct"])


if __name__ == "__main__":
    unittest.main()
