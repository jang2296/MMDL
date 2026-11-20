"""Synthetic CPU cases for the fixed V8 target selection contract."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mmdl.evaluation.CompassVerifier_checker import CHECKER_VERSION
from mmdl.evaluation.CompassVerifier_checker import selection
from mmdl.evaluation.CompassVerifier_checker.prompt import CV_PROMPT_SHA256
from mmdl.evaluation.parsers import score_with_parser
from mmdl.runtime.artifacts import digest


def prediction(identifier="synthetic-1", raw="I need more information.", **changes):
    row = {
        "id": identifier, "subject": "Synthetic arithmetic", "question_type": "multiple-choice",
        "prompt": "Question: What is two plus two?\nOptions:\nA. three\nB. four",
        "question": "What is two plus two?", "options": ["three", "four"], "answer": "B",
        "raw_response": raw, "finish_reason": "stop", "generated_tokens": len(raw),
    }
    row.update(changes)
    row.update(score_with_parser(row["raw_response"], row["question_type"], row["options"],
                                 row["answer"], "team-final-answer-v8", row["finish_reason"],
                                 question=row["question"]))
    return seal(row)


def seal(row):
    row["_record_sha256"] = digest({key: value for key, value in row.items()
                                     if key != "_record_sha256"})
    return row


def characters(text):
    return list(map(ord, text))


def decode_characters(ids):
    return "".join(map(chr, ids))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_rows(self, rows, name="predictions.jsonl"):
        path = self.directory / name
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                        encoding="utf-8")
        return path

    def test_six_categories_and_precedence_with_real_v8_records(self):
        rows = [
            prediction("correct", "Final answer: B."),
            prediction("length", "Still reasoning.", finish_reason="length"),
            prediction("empty", " \n "),
            prediction("no-parse"),
            prediction("open-wrong", "Final answer: 7.", question_type="open", options=[], answer="4"),
            prediction("mcq-wrong", "Final answer: A."),
        ]
        loaded = selection.load_rows([self.write_rows(rows)])
        selection.replay_v8(loaded)
        self.assertEqual(tuple(map(selection.categorize, loaded)), selection.CATEGORIES)
        self.assertEqual(selection.categorize({**rows[0], "finish_reason": "length"}), "V8_CORRECT")
        self.assertEqual(selection.categorize({**rows[2], "finish_reason": "length"}), "LENGTH")
        self.assertEqual(selection.categorize({**rows[3], "raw_response": "  "}), "EMPTY")

    def test_integrity_duplicate_ids_and_empty_input_are_rejected(self):
        row = prediction()
        changed = {**row, "raw_response": "tampered"}
        with self.assertRaisesRegex(ValueError, "integrity"):
            selection.load_rows([self.write_rows([changed])])
        first = self.write_rows([row], "first.jsonl")
        second = self.write_rows([row], "second.jsonl")
        with self.assertRaisesRegex(ValueError, "Duplicate prediction id"):
            selection.load_rows([first, second])
        with self.assertRaises(ValueError):
            selection.load_rows([])
        with self.assertRaises(ValueError):
            selection.load_rows([self.write_rows([])])

    def test_multiple_files_keep_source_order(self):
        paths = [self.write_rows([prediction("second")], "a.jsonl"),
                 self.write_rows([prediction("first")], "b.jsonl")]
        self.assertEqual([row["id"] for row in selection.load_rows(paths)], ["second", "first"])

    def test_malformed_records_are_rejected_even_with_recomputed_digest(self):
        for key, bad in (
            ("id", ""), ("subject", None), ("correct", 0), ("generated_tokens", True),
            ("generated_tokens", -1), ("raw_response", []), ("parsed_answer", 1),
            ("question_type", "mcq"), ("finish_reason", "abort"), ("parse_status", "unknown"),
            ("options", "three, four"), ("options", ["three", 4]), ("options", []),
            ("answer", []), ("answer", ["B", 1]), ("answer", "C"),
            ("extraction_rule", None), ("_record_sha256", "not-a-hash"),
        ):
            with self.subTest(key=key, bad=bad):
                row = seal({**prediction(), key: bad})
                if key == "_record_sha256":
                    row[key] = bad
                with self.assertRaises(ValueError):
                    selection.load_rows([self.write_rows([row])])
        row = prediction()
        del row["parsed_answer"]
        with self.assertRaises(ValueError):
            selection.load_rows([self.write_rows([seal(row)])])
        with self.assertRaises(ValueError):
            selection.load_rows([self.write_rows([[]])])

    def test_duplicate_json_fields_are_rejected(self):
        path = self.directory / "duplicate.jsonl"
        path.write_text('{"id":"one","id":"two"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            selection.load_rows([path])

    def test_replay_checks_each_of_the_three_saved_fields(self):
        for key, value in (("correct", True), ("parse_status", "PARSED"), ("parsed_answer", "A")):
            with self.subTest(key=key):
                row = seal({**prediction(), key: value})
                loaded = selection.load_rows([self.write_rows([row])])
                with self.assertRaisesRegex(ValueError, "V8 replay"):
                    selection.replay_v8(loaded)

    def test_gold_formats_and_aliases_are_preserved(self):
        self.assertEqual(selection.gold_texts(prediction()), ["B. four"])
        self.assertEqual(selection.gold_texts(prediction(answer=["B", "A"])), ["B. four", "A. three"])
        opened = prediction(question_type="open", options=[], answer=["4", "four"])
        self.assertEqual(selection.gold_texts(opened), ["4", "four"])

    def test_targets_use_p0_alias_order_and_exact_input_digest(self):
        config = selection.load_config()
        row = prediction(answer=["B", "A"])
        targets = selection.build_targets([prediction("skip", "Final answer: B."), row], config,
                                          characters, decode_characters)
        self.assertEqual([target["alias_index"] for target in targets], [0, 1])
        self.assertEqual([target["gold_answer"] for target in targets], ["B. four", "A. three"])
        for target in targets:
            self.assertEqual(set(target), {
                "id", "subject", "question_type", "category", "alias_index", "question",
                "gold_answer", "llm_response", "response_truncated", "response_tokens", "input_sha256",
            })
            self.assertEqual(target["question"], row["prompt"])
            self.assertFalse(target["response_truncated"])
            self.assertEqual(target["response_tokens"], len(row["raw_response"]))
            self.assertEqual(target["input_sha256"], digest([
                CHECKER_VERSION, CV_PROMPT_SHA256, row["prompt"], target["gold_answer"], row["raw_response"],
            ]))

    def test_tail_only_truncates_over_limit_and_aligns_to_source_lines(self):
        cases = (
            ("a\nbc", 4, ("a\nbc", False)),
            ("abcdef\nlast", 8, ("last", True)),
            ("abcdef\nlast", 4, ("last", True)),
            ("abcdef", 3, ("def", True)),
        )
        for text, limit, expected in cases:
            with self.subTest(text=text, limit=limit):
                self.assertEqual(selection.tail(text, limit, characters, decode_characters), expected)
        for bad in (0, -1, True):
            with self.assertRaises(ValueError):
                selection.tail("abc", bad, characters, decode_characters)

    def test_tail_does_not_erase_single_line_answer_with_trailing_newline(self):
        self.assertEqual(selection.tail("abcdef\n", 3, characters, decode_characters), ("ef\n", True))
        with self.assertRaises(ValueError):
            selection.tail("answer     ", 3, characters, decode_characters)

    def test_tail_avoids_partial_utf8_and_replacement_characters(self):
        def encode(text):
            return list(text.encode("utf-8"))
        for errors in ("strict", "replace"):
            with self.subTest(errors=errors):
                def decode(ids):
                    return bytes(ids).decode("utf-8", errors=errors)
                self.assertEqual(selection.tail("aé끝", 4, encode, decode), ("끝", True))
                self.assertEqual(selection.tail("aé\n끝", 5, encode, decode), ("끝", True))
        with self.assertRaisesRegex(ValueError, "valid response suffix"):
            selection.tail("끝", 1, encode, lambda ids: bytes(ids).decode("utf-8", errors="replace"))

    def test_config_and_base_parser_hashes_are_frozen(self):
        config = selection.load_config()
        root = Path(selection.__file__).resolve().parents[4]
        selection.verify_base_parser(config, root)
        for section, key, value in (
            ("decision", "accept_threshold", 0.89), ("input", "response_tail_tokens", 3073),
            ("selection", "finish_reason", "length"),
            ("judge", "labels", {"A": "INCORRECT", "B": "CORRECT", "C": "INVALID"}),
        ):
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(config)
                changed[section][key] = value
                path = self.directory / "changed.yaml"
                path.write_text(yaml.safe_dump(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    selection.load_config(path)
        with patch.object(selection, "CV_PROMPT_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "prompt SHA256"):
                selection.load_config()
        with patch.object(selection, "sha256_file", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "parser SHA256"):
                selection.verify_base_parser(config, root)
        changed = copy.deepcopy(config)
        changed["base_parser_files"] = {}
        with self.assertRaisesRegex(ValueError, "file manifest"):
            selection.verify_base_parser(changed, root)


if __name__ == "__main__":
    unittest.main()
