"""Synthetic Pro contracts; no benchmark text or images are included."""

import io
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from mmdl.data.download import prepare
from mmdl.evaluation.datasets import mmmu_pro
from mmdl.evaluation.engine import prepare_request, publish
from mmdl.evaluation.parsers import score_with_parser
from mmdl.evaluation.writer import RunWriter, finalize, render
from mmdl.runtime.artifacts import digest, read_json, sha256_file, write_json
from mmdl.runtime.contracts import MODEL_ID, PRO_ID, SUBJECTS, sample_seed
from scripts.validate_run import validate

ROOT = Path(__file__).resolve().parents[2]


def row(slots=(1,), answer="A", options=2):
    item = dict(id="test_Accounting_1", question="Synthetic " + " ".join(f"<image {n}>" for n in slots),
                options=[f"synthetic choice {n}" for n in range(options)], answer=answer,
                explanation="PRIVATE ANSWER SIDE", hint="PRIVATE HINT")
    for n in slots:
        stream = io.BytesIO()
        Image.new("RGB", (n + 1, n + 2), (n, 0, 0)).save(stream, format="PNG")
        item[f"image_{n}"] = {"bytes": stream.getvalue()}
    return item


def result(sid, subject, correct=False):
    return dict(id=sid, subject=subject, correct=correct, parse_status="PARSED",
                generated_tokens=1, generation_seconds=0.01, finish_reason="stop")


class ProDatasetTests(unittest.TestCase):
    def test_seven_images_in_slot_order_repeated_reversed_references(self):
        item = row(range(1, 8), options=12, answer="L")
        item["question"] = "Order <image 7> then <image 1> and <image 7>"
        item["options"][0] += " " + " ".join(f"<image {n}>" for n in range(6, 1, -1))
        sample, images, gold = mmmu_pro.separate_sample(item, "Accounting")
        messages = mmmu_pro.build_messages(sample, images, ROOT)
        self.assertEqual([image.getpixel((0, 0))[0] for image in images], list(range(1, 8)))
        self.assertEqual(sample["used_image_slots"], list(range(1, 8)))
        self.assertEqual(len(messages[0]["content"]), 8)
        text = messages[0]["content"][-1]["text"]
        self.assertIn(item["question"], text)
        self.assertIn("L. synthetic choice 11", text)
        self.assertNotIn("PRIVATE", text)
        self.assertNotIn("answer", sample)
        self.assertEqual(gold["answer"], "L")
        for answer in ("K", "L"):
            self.assertTrue(score_with_parser(f"Final answer: {answer}", "multiple-choice", sample["options"],
                                             answer, "team-final-answer-v8", "stop")["correct"])

    def test_sparse_slots_and_explanation_only_images(self):
        item = row((1, 3))
        item["image_2"] = row((2,))["image_2"]
        sample, images, _ = mmmu_pro.separate_sample(item, "Accounting")
        self.assertEqual(sample["original_image_slots"], [1, 2, 3])
        self.assertEqual(sample["used_image_slots"], [1, 3])
        messages = mmmu_pro.build_messages(sample, images, ROOT)
        self.assertIn("<image 3>", messages[0]["content"][-1]["text"])
        self.assertEqual(images[1].getpixel((0, 0))[0], 3)
        item["question"] = "Only <image 1>"
        sample, images, _ = mmmu_pro.separate_sample(item, "Accounting")
        self.assertEqual(sample["used_image_slots"], [1])
        self.assertEqual(len(images), 1)

    def test_missing_malformed_references_gold_and_subject_fail_closed(self):
        for changes in ({"question": "Missing <image 2>"}, {"question": "Bad <image eight>"},
                        {"question": "Bad <image 8>"}, {"answer": "L"},
                        {"id": "test_Unknown_1"}, {"id": "../outside"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                mmmu_pro.separate_sample(row() | changes, "Accounting")

    def test_request_preserves_input_identity_and_blocks_eight_images(self):
        cfg = {"dataset": {"id": PRO_ID}, "generation": {"seed": 3407}}
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary)
            request, record = prepare_request(row(range(1, 8)), "Accounting", artifacts, ROOT, cfg)
            self.assertEqual(len(request["images"]), 7)
            self.assertNotIn("answer", request)
            self.assertEqual(len(record["image_references"]), 7)
            self.assertEqual(len({ref["pixels_sha256"] for ref in record["image_references"]}), 7)
            with self.assertRaises(ValueError):
                prepare_request(row(range(1, 9)), "Accounting", artifacts, ROOT, cfg)

    def test_validation_still_rejects_six_images(self):
        item = row(range(1, 7)) | {"id": "validation_Accounting_1", "question_type": "multiple-choice"}
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "suite limit"):
            prepare_request(item, "Accounting", Path(temporary), ROOT,
                            {"generation": {"seed": 3407}})

    def test_pinned_source_contract_and_file_hash_are_required(self):
        pinned = read_json(ROOT / "manifests/datasets/mmmu_pro.json")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / "manifest.json", pinned | {"revision": "0" * 40})
            with self.assertRaisesRegex(ValueError, "fixed MMMU-Pro"):
                mmmu_pro.load_test(directory)
            write_json(directory / "manifest.json", pinned | {"files": []})
            with self.assertRaisesRegex(ValueError, "pinned manifest"):
                mmmu_pro.load_test(directory)
            write_json(directory / "manifest.json", pinned)
            (directory / "README.md").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                mmmu_pro.load_test(directory)

    def test_selected_prepare_never_acquires_validation_and_preserves_receipts(self):
        calls = []
        def acquire(repo, revision, kind, filenames, cache, destination):
            calls.append(repo)
            return dict(repo_id=repo, revision=revision, files=[], kind=kind), cache / "hub/snapshot"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            old = directory / "artifacts/setup/locations.json"
            write_json(old, {"archival": "unchanged"})
            with patch("mmdl.data.download.acquire", side_effect=acquire):
                prepare(directory, directory / "data", directory / "cache", directory / "artifacts", suite="mmmu-pro")
            self.assertEqual(calls, [MODEL_ID, PRO_ID])
            self.assertEqual(read_json(directory / "data/evaluation/mmmu_pro/manifest.json")["usage"], "evaluation_only")
            self.assertEqual(read_json(old), {"archival": "unchanged"})


class ProWriterTests(unittest.TestCase):
    def test_independent_full_artifact_audit_and_missing_record_negative(self):
        mapping = {f"test_{subject}_{n}": subject for index, subject in enumerate(SUBJECTS)
                   for n in range(58 if index < 20 else 57)}
        expected = sorted(mapping)
        identity = dict(suite="mmmu-pro", id_subjects=mapping, ids_sha256=digest(expected),
                        submission_order_sha256=digest(expected),
                        seed_order_sha256=digest([sample_seed(3407, sid) for sid in expected]))
        cfg = {"parser": "team-final-answer-v8"}
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary) / "artifacts"
            directory = artifacts / "runs/synthetic-pro"
            image = artifacts / "assets/images/synthetic.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(image)
            renamed = image.with_name(sha256_file(image) + ".png")
            image.rename(renamed)
            image = renamed
            reference = f"../../assets/images/{image.name}"
            with RunWriter(directory, identity, expected) as writer:
                for sid in expected:
                    prompt = "Question: Synthetic <image 1>\nOptions:\nA. Yes\nB. No\nPlease select the correct answer from the options above."
                    tensors = {"synthetic": {"sha256": "0" * 64}}
                    raw = "Final answer: A"
                    item = result(sid, mapping[sid]) | dict(question_type="multiple-choice", question="Synthetic <image 1>",
                        options=["Yes", "No"], image_references=[{"field": "image_1", "marker": "<image 1>", "content_index": 0, "original_size": [2, 2], "pixels_sha256": hashlib.sha256(bytes(12)).hexdigest()}],
                        original_image_slots=[1], used_image_slots=[1], images=[reference],
                        messages=[{"role": "user", "content": [{"type": "image", "image": reference, "sha256": sha256_file(image)}, {"type": "text", "text": prompt}]}],
                        prompt=prompt, seed=sample_seed(3407, sid), input_tensors=tensors,
                        input_sha256=digest(tensors), image_grid_thw=[[1, 2, 2]],
                        raw_response=raw, answer="A", generated_token_ids=[1],
                        model_explanation=None, gold_explanation=None, total_sample_seconds=0.01,
                        status="COMPLETED")
                    writer.save(item | score_with_parser(raw, "multiple-choice", item["options"], "A", cfg["parser"], "stop"))
            summary = finalize(directory, expected, "full")
            import yaml
            (directory / "resolved_eval_config.yaml").write_text(yaml.safe_dump(cfg))
            (directory / "resolved_hardware_config.yaml").write_text("{}")
            for name in ("environment", "backend", "coverage", "command", "code_files"):
                write_json(directory / f"{name}.json", {})
            render(directory)
            public = Path(temporary) / "public"
            publish(public, directory, summary, {}, cfg, {}, identity)
            receipt = validate(directory, public / directory.name)
            self.assertEqual(receipt["completed_count"], 1730)
            self.assertEqual(receipt["correct"], 1730)
            (directory / "samples" / f"{expected[-1]}.json").unlink()
            with self.assertRaises(AssertionError):
                validate(directory, public / directory.name)

    def test_full_pro_actual_denominators_missing_resume_and_immutable_answers(self):
        mapping = {f"test_{subject}_{n}": subject for index, subject in enumerate(SUBJECTS)
                   for n in range(58 if index < 20 else 57)}
        expected = sorted(mapping)
        identity = dict(suite="mmmu-pro", id_subjects=mapping, model_sha256="base")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            with RunWriter(directory, identity, expected) as writer:
                for sid in expected[:-1]:
                    writer.save(result(sid, mapping[sid], mapping[sid] == SUBJECTS[0]))
            incomplete = finalize(directory, expected, "full")
            self.assertEqual(incomplete["status"], "INCOMPLETE")
            self.assertEqual(incomplete["denominator"], 1730)
            first = directory / "samples" / f"{expected[0]}.json"
            original = first.read_bytes()
            with self.assertRaisesRegex(ValueError, "Resume identity"):
                RunWriter(directory, identity | {"model_sha256": "ft"}, expected, resume=True)
            with RunWriter(directory, identity, expected, resume=True) as writer:
                self.assertEqual(set(writer.completed), set(expected[:-1]))
                with self.assertRaises(FileExistsError):
                    writer.save(result(expected[0], mapping[expected[0]]))
                writer.save(result(expected[-1], mapping[expected[-1]]))
            self.assertEqual(first.read_bytes(), original)
            summary = finalize(directory, expected, "full")
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertAlmostEqual(summary["macro_average"], 1 / 30)
            self.assertAlmostEqual(summary["accuracy"], 58 / 1730)
            self.assertNotEqual(summary["macro_average"], summary["accuracy"])
            self.assertEqual([item["total"] for item in summary["subject_scores"]].count(58), 20)
            with self.assertRaisesRegex(ValueError, "registered"):
                finalize(directory, expected[:-1], "full")
            self.assertEqual(len((directory / "predictions.jsonl").read_text().splitlines()), 1730)

    def test_pro_subject_swap_rejected_before_reporting(self):
        sid = "test_Accounting_1"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            with RunWriter(directory, {"suite": "mmmu-pro", "id_subjects": {sid: "Accounting"}}, [sid]) as writer:
                writer.save(result(sid, "Art"))
            with self.assertRaisesRegex(ValueError, "mapping mismatch"):
                finalize(directory, [sid], "partial")


if __name__ == "__main__":
    unittest.main()
