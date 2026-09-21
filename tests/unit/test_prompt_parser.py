from __future__ import annotations

import random
import unittest
from pathlib import Path

from mmdl.evaluation.parsers import EMPTY, NO_PARSE, PARSED, score_response
from mmdl.evaluation.prompt import build_messages, build_prompt


ROOT = Path(__file__).resolve().parents[2]


class PromptTests(unittest.TestCase):
    def test_mcq_hint_options_and_no_gold(self) -> None:
        sample = {
            "question": "Which label?",
            "question_type": "multiple-choice",
            "options": ["first", "second"],
            "hint": "Read the axis.",
            "answer": "hidden-gold-token",
            "gold_explanation": "hidden explanation",
        }
        prompt = build_prompt(sample, ROOT)
        self.assertEqual(
            prompt,
            "Hint: Read the axis.\nQuestion: Which label?\nOptions:\n"
            "A. first\nB. second\nPlease select the correct answer from the options above.",
        )
        self.assertNotIn("hidden-gold-token", prompt)
        self.assertNotIn("hidden explanation", prompt)

    def test_open_prompt_omits_options_and_preserves_hint(self) -> None:
        sample = {
            "question": "What is the result?",
            "question_type": "open",
            "options": ["not shown"],
            "hint": "Use the chart.",
            "answer": "gold",
        }
        self.assertEqual(build_prompt(sample, ROOT), "Hint: Use the chart.\nQuestion: What is the result?")

    def test_messages_preserve_image_order_and_references(self) -> None:
        first, second = object(), object()
        sample = {
            "question": "Compare <image 2> with <image 1>.",
            "question_type": "open",
            "options": [],
        }
        messages = build_messages(sample, [first, second], ROOT)
        content = messages[0]["content"]
        self.assertIs(content[0]["image"], first)
        self.assertIs(content[1]["image"], second)
        self.assertEqual(content[-1]["type"], "text")
        self.assertIn("<image 2>", content[-1]["text"])

    def test_messages_reject_invalid_image_slot(self) -> None:
        sample = {"question": "See <image 2>.", "question_type": "open", "options": []}
        with self.assertRaises(ValueError):
            build_messages(sample, [object()], ROOT)


class ParserTests(unittest.TestCase):
    def test_mcq_label_with_extra_text(self) -> None:
        result = score_response("The final answer is (B).", "multiple-choice", ["red", "blue"], "B")
        self.assertEqual(result, {"parsed_answer": "B", "parse_status": PARSED, "correct": True})

    def test_mcq_unparseable_does_not_guess(self) -> None:
        result = score_response("I cannot determine this.", "multiple-choice", ["red", "blue"], "A")
        self.assertEqual(result["parse_status"], NO_PARSE)
        self.assertIsNone(result["parsed_answer"])
        self.assertFalse(result["correct"])

    def test_empty_response(self) -> None:
        self.assertEqual(
            score_response("  ", "multiple-choice", ["red", "blue"], "A"),
            {"parsed_answer": None, "parse_status": EMPTY, "correct": False},
        )

    def test_open_evaluator_normalizes_numeric_gold(self) -> None:
        state = random.getstate()
        result = score_response("The answer is 42,000.", "open", [], "42000")
        self.assertEqual(random.getstate(), state)
        self.assertEqual(result["parse_status"], PARSED)
        self.assertTrue(result["correct"])


if __name__ == "__main__":
    unittest.main()
