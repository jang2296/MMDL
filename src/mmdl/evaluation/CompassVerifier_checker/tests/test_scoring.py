"""Synthetic scoring checks; no benchmark questions or responses."""

from copy import deepcopy
from pathlib import Path
import unittest

import yaml

from mmdl.evaluation.parsers import score_with_parser
from mmdl.runtime.artifacts import digest

from ..prompt import CV_PROMPT_SHA256
from ..scoring import item_verdict, merge, review_markdown, summarize, verdict
from ..selection import build_targets, load_config

LABEL_IDS = {"A": 10, "B": 11, "C": 12}


def probabilities(p_a=0.9, p_b=0.06, p_c=0.03, top="A"):
    return {"p_a": p_a, "p_b": p_b, "p_c": p_c, "top_token_id": LABEL_IDS.get(top, 99),
            "top_token": top, "input_tokens": 20}


def prediction(item_id, raw="7", *, answer="7", question_type="open", finish_reason="stop"):
    row = {"id": item_id, "subject": "Synthetic arithmetic", "question_type": question_type,
           "question": "What is three plus four?", "prompt": "Question: What is three plus four?",
           "options": ["six", "seven"], "answer": answer, "raw_response": raw,
           "finish_reason": finish_reason, "generated_tokens": len(raw.split())}
    row.update(score_with_parser(raw, question_type, row["options"], answer,
                                 "team-final-answer-v8", finish_reason, question=row["question"]))
    row["_record_sha256"] = digest(row)
    return row


def identity(mode="official"):
    return {"mode": mode, "reportable": mode == "official",
            "source_predictions_sha256": ["synthetic-source"], "v8_scoring_sha256": "synthetic-v8",
            "config_sha256": "synthetic-config", "prompt_sha256": CV_PROMPT_SHA256,
            "judge": {"model": "synthetic", "label_token_ids": LABEL_IDS}}


class ScoringTests(unittest.TestCase):
    def test_float32_rounding_does_not_reject_valid_confident_softmax(self):
        result = {"p_a": 1.0, "p_b": 2e-9, "p_c": 1e-9,
                  "top_token_id": 0, "top_token": "A"}
        self.assertEqual(verdict(result, {"A": 0, "B": 1, "C": 2}, 0.90), "ACCEPT")

    def setUp(self):
        self.config = {**load_config(), "label_token_ids": LABEL_IDS}
        self.rows = [prediction("correct"), prediction("length", "unfinished", finish_reason="length"),
                     prediction("empty", " "),
                     prediction("no-parse", "I cannot determine the result.", answer="B",
                                question_type="multiple-choice"),
                     prediction("open-wrong", "8"),
                     prediction("explicit-wrong", "A", answer="B", question_type="multiple-choice")]
        self.targets = build_targets(self.rows, self.config, str.split, " ".join)
        self.verdicts = [{"id": target["id"], "alias_index": target["alias_index"],
                          "input_sha256": target["input_sha256"], **probabilities()}
                         for target in self.targets]

    def test_threshold_and_top_tokens(self):
        for record, expected in [
            (probabilities(), "ACCEPT"),
            (probabilities(p_a=0.8999), "UNSURE"),
            (probabilities(0.02, 0.9, 0.03, "B"), "REJECT"),
            (probabilities(0.02, 0.03, 0.9, "C"), "INVALID"),
            (probabilities(0.1, 0.1, 0.1, "D"), "UNSURE"),
        ]:
            with self.subTest(expected=expected):
                self.assertEqual(verdict(record, LABEL_IDS, 0.9), expected)

    def test_invalid_probabilities_fail(self):
        for value in (float("nan"), float("inf"), -0.01, 1.01, True, "0.95", None):
            for field in ("p_a", "p_b", "p_c"):
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    verdict({**probabilities(), field: value}, LABEL_IDS, 0.9)
        with self.assertRaises(ValueError):
            verdict(probabilities(0.95, 0.1, 0.01), LABEL_IDS, 0.9)

    def test_label_id_and_top_token_consistency(self):
        for ids in ({"A": 10, "B": 10, "C": 12}, {"A": True, "B": 11, "C": 12},
                    {"A": 10, "B": 11}, {"A": -1, "B": 11, "C": 12}):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                verdict(probabilities(), ids, 0.9)
        for changes in ({"top_token_id": 11}, {"top_token_id": 99}, {"top_token": "B"},
                        {"top_token_id": True}, {"top_token_id": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                verdict({**probabilities(), **changes}, LABEL_IDS, 0.9)
        with self.assertRaises(ValueError):
            verdict(probabilities(0.1, 0.7, 0.1), LABEL_IDS, 0.9)

    def test_invalid_threshold_and_missing_probability_fail(self):
        for threshold in (float("nan"), float("inf"), -0.1, 1.1, True, "0.9"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                verdict(probabilities(), LABEL_IDS, threshold)
        result = probabilities()
        del result["p_a"]
        with self.assertRaises(ValueError):
            verdict(result, LABEL_IDS, 0.9)

    def test_alias_acceptance_and_deterministic_ties(self):
        aliases = [{"alias_index": 1, "verdict": "REJECT", "p_a": 0.2},
                   {"alias_index": 0, "verdict": "ACCEPT", "p_a": 0.91}]
        self.assertEqual(item_verdict(aliases)["verdict"], "ACCEPT")
        self.assertEqual(item_verdict([{"alias_index": 2, "verdict": "INVALID", "p_a": 0.2},
                                       aliases[0]])["verdict"], "REJECT")
        for invalid in ([], [aliases[0], aliases[0]], [{**aliases[0], "verdict": "UNKNOWN"}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                item_verdict(invalid)

    def test_merge_preserves_v8_and_rederives_verdicts(self):
        for result in self.verdicts:
            result["verdict"] = "REJECT"
        merged = merge(self.rows, self.targets, self.verdicts, self.config)
        self.assertEqual([row["category"] for row in merged],
                         ["V8_CORRECT", "LENGTH", "EMPTY", "NO_PARSE_STOP", "OPEN_MISMATCH_STOP",
                          "MCQ_EXPLICIT_WRONG"])
        self.assertEqual([row["final_correct"] for row in merged], [True, False, False, True, True, False])
        self.assertIsNone(merged[0]["checker_verdict"])
        self.verdicts[0].update(probabilities(0.02, 0.9, 0.03, "B"), verdict="ACCEPT")
        self.assertEqual(merge(self.rows, self.targets, self.verdicts, self.config)[3]["checker_verdict"],
                         "REJECT")

    def test_non_target_verdict_fails(self):
        extra = {**self.verdicts[0], "id": "correct"}
        with self.assertRaises(ValueError):
            merge(self.rows, self.targets, self.verdicts + [extra], self.config)

    def test_missing_verdict_fails(self):
        with self.assertRaises(ValueError):
            merge(self.rows, self.targets, self.verdicts[:-1], self.config)

    def test_input_sha_mismatch_fails(self):
        self.verdicts[0]["input_sha256"] = "different-run"
        with self.assertRaises(ValueError):
            merge(self.rows, self.targets, self.verdicts, self.config)

    def test_target_self_hash_mismatch_fails(self):
        self.targets[0]["llm_response"] += " injected"
        with self.assertRaises(ValueError):
            merge(self.rows, self.targets, self.verdicts, self.config)

    def test_duplicate_verdict_or_target_fails(self):
        with self.assertRaises(ValueError):
            merge(self.rows, self.targets, self.verdicts + self.verdicts[:1], self.config)
        with self.assertRaises(ValueError):
            merge(self.rows, self.targets + self.targets[:1], self.verdicts, self.config)

    def test_missing_or_unexpected_target_fails(self):
        with self.assertRaises(ValueError):
            merge(self.rows, self.targets[:-1], self.verdicts, self.config)
        with self.assertRaises(ValueError):
            merge(self.rows, self.targets + [{**self.targets[0], "id": "correct"}],
                  self.verdicts, self.config)

    def test_duplicate_or_invalid_prediction_id_fails(self):
        with self.assertRaises(ValueError):
            merge(self.rows + self.rows[:1], self.targets, self.verdicts, self.config)
        with self.assertRaises(ValueError):
            merge([{**self.rows[0], "id": ""}], [], [], self.config)

    def test_alias_set_is_exact_and_one_accepts(self):
        rows = [prediction("aliases", "8", answer=["7", "seven"])]
        targets = build_targets(rows, self.config, str.split, " ".join)
        results = [{"id": target["id"], "alias_index": target["alias_index"],
                    "input_sha256": target["input_sha256"],
                    **probabilities(0.02, 0.9, 0.03, "B")}
                   for target in targets]
        results[1].update(probabilities())
        self.assertTrue(merge(rows, targets, results, self.config)[0]["final_correct"])
        with self.assertRaises(ValueError):
            merge(rows, targets, results[:1], self.config)
        results[1]["alias_index"] = 3
        with self.assertRaises(ValueError):
            merge(rows, targets, results, self.config)

    def test_summary_counts_and_partial_denominator(self):
        self.targets[0]["response_truncated"] = True
        merged = merge(self.rows, self.targets, self.verdicts, self.config)
        summary = summarize(merged, identity())
        self.assertEqual(summary["denominator"], 6)
        self.assertEqual(summary["v8_correct"], 1)
        self.assertEqual(summary["final_correct"], 3)
        self.assertEqual(summary["transitions"], {"0->0": 3, "0->1": 2, "1->1": 1, "1->0": 0})
        self.assertEqual(sum(summary["categories"].values()), 6)
        self.assertEqual(sum(summary["verdicts"].values()), 2)
        self.assertEqual(summary["truncated_inputs"], 1)
        self.assertEqual(summary["by_subject"]["Synthetic arithmetic"],
                         {"n": 6, "v8_correct": 1, "final_correct": 3})
        self.assertNotIn("%", str(summary))

    def test_summary_refuses_correct_to_incorrect_transition(self):
        merged = merge(self.rows, self.targets, self.verdicts, self.config)
        merged[0]["final_correct"] = False
        with self.assertRaises(ValueError):
            summarize(merged, identity())

    def test_smoke_is_diagnostic_and_subset_scope_is_preserved(self):
        merged = merge(self.rows, self.targets, self.verdicts, self.config)
        info = {**identity("smoke"), "scope": "smoke-target-subset", "source_denominator": 900}
        summary = summarize(merged, info)
        self.assertFalse(summary["reportable"])
        self.assertEqual(summary["source_denominator"], 900)
        self.assertEqual(summary["scope"], "smoke-target-subset")
        self.assertIn("diagnostics only", " ".join(summary["notes"]))
        info["reportable"] = True
        with self.assertRaises(ValueError):
            summarize(merged, info)

    def test_review_only_includes_accept_and_unsure(self):
        self.verdicts[1].update(probabilities(p_a=0.85))
        merged = merge(self.rows, self.targets, self.verdicts, self.config)
        review = review_markdown(merged, self.rows, self.verdicts)
        self.assertIn('"id": "no-parse"', review)
        self.assertIn('"id": "open-wrong"', review)
        self.assertNotIn('"id": "correct"', review)
        self.assertIn("Gold answer", review)
        self.assertIn("I cannot determine the result.", review)

    def test_review_fences_untrusted_response_text(self):
        rows = deepcopy(self.rows)
        rows[3]["raw_response"] = "```\n# pretend heading\n```"
        merged = merge(self.rows, self.targets, self.verdicts, self.config)
        review = review_markdown(merged, rows, self.verdicts)
        self.assertIn("````\n```\n# pretend heading\n```\n````", review)

    def test_checkset_is_synthetic_complete_and_has_a_long_case(self):
        path = Path(__file__).parents[1] / "checkset_v1.yaml"
        checkset = yaml.safe_load(path.read_text(encoding="utf-8"))
        cases = checkset["cases"]
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({case["id"] for case in cases}), 30)
        self.assertEqual({case["expect"] for case in cases}, {"accept", "not_accept"})
        for case in cases:
            self.assertTrue(all(isinstance(case[key], str) and case[key]
                                for key in ("question", "gold_answer", "llm_response")))
        self.assertGreater(max(len(case["llm_response"].split()) for case in cases), 3072)


if __name__ == "__main__":
    unittest.main()
