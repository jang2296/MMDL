"""Independent final artifact audit, without loading a model or generating answers."""

import argparse
import hashlib
import math
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image
import yaml

from mmdl.evaluation.parsers import score_with_parser
from mmdl.runtime.artifacts import digest, read_json, resolve_image, sha256_file
from mmdl.runtime.contracts import SUBJECTS, SUITE_COUNTS, sample_seed


class ViewerImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs).get("src"))


def validate(run_dir, public_dir):
    if not __debug__:
        raise RuntimeError("Artifact audit must not run with Python assertion checks disabled")
    run_dir, public_dir = Path(run_dir), Path(public_dir)
    for name in ("summary.json", "run_manifest.json", "predictions.jsonl", "failures.jsonl",
                 "subject_scores.csv", "environment.json", "resolved_eval_config.yaml",
                 "resolved_hardware_config.yaml", "backend.json", "coverage.json",
                 "code_files.json", "command.json", "review.html"):
        assert (run_dir / name).is_file(), f"Required artifact missing: {name}"
    summary = read_json(run_dir / "summary.json")
    manifest = read_json(run_dir / "run_manifest.json")
    public = read_json(public_dir / "run_manifest.json")
    identity = manifest["identity"]
    managed = identity.get("run_role") in {"assignment", "analysis", "evaluation"}
    if managed:
        assert re.fullmatch(r"[0-9a-f]{40}", identity.get("git_commit", "")), "Missing fixed commit"
        assert run_dir.name == f"{identity['job_id']}-{identity['run_role']}", "Wrong job/run role"
    eval_config = yaml.safe_load((run_dir / "resolved_eval_config.yaml").read_text())
    invocations = {p.stem: read_json(p) for p in (run_dir / "inference_invocations").glob("*.json")}
    inference_ids, invocation_ids, image_paths, checked_images = set(), set(), set(), set()
    pixel_identities = {}
    rows = [read_json(p) for p in sorted((run_dir / "samples").glob("*.json"))]
    suite = identity.get("suite", "mmmu-val")
    pro, mapped = suite == "mmmu-pro", suite != "mmmu-val"
    count = SUITE_COUNTS[suite]
    id_subjects = identity.get("id_subjects", {})
    if mapped:
        from mmdl.evaluation.datasets.mmmu_pro import subject_from_id
        assert set(id_subjects) == set(manifest["expected_ids"]), "Missing registered Pro subjects"
        assert all(subject_from_id(sid) == subject for sid, subject in id_subjects.items())
    subject_counts = Counter(id_subjects.values()) if mapped else Counter({subject: 30 for subject in SUBJECTS})
    assert set(subject_counts) == set(SUBJECTS)
    if pro:
        assert all(50 <= n <= 60 for n in subject_counts.values())
    if suite == "mmmu-test":
        assert all(sid.startswith("test_") for sid in id_subjects), "Non-test ID in the test suite"
        assert dict(subject_counts) == identity["subject_counts"], "Registered test subject sizes differ"
    assert summary["status"] == "COMPLETE", "Not a complete run"
    assert summary["mode"] == "full"
    assert summary["denominator"] == summary["expected_count"] == summary["completed_count"] == count
    assert summary["missing_count"] == summary["unresolved_failure_count"] == 0
    assert len(rows) == len({r["id"] for r in rows}) == count
    assert set(r["id"] for r in rows) == set(manifest["expected_ids"])
    assert len(manifest["expected_ids"]) == count
    assert identity["ids_sha256"] == digest(manifest["expected_ids"])
    assert Counter(r["subject"] for r in rows) == subject_counts
    if mapped:
        assert identity["submission_order_sha256"] == digest(manifest["expected_ids"])
        assert identity["seed_order_sha256"] == digest([sample_seed(3407, sid) for sid in manifest["expected_ids"]])
    correct = 0
    for row in rows:
        required = {"id", "subject", "question_type", "question", "options", "image_references", "images",
                    "messages", "prompt", "seed", "input_sha256", "raw_response", "parsed_answer", "answer",
                    "correct", "parse_status", "finish_reason", "generated_tokens", "generation_seconds",
                    "model_explanation", "gold_explanation", "total_sample_seconds"}
        assert required <= row.keys(), "Required analysis field missing"
        assert row["_record_sha256"] == digest({k: v for k, v in row.items() if k != "_record_sha256"})
        assert row["status"] == "COMPLETED"
        if mapped:
            assert row["subject"] == id_subjects[row["id"]]
            assert 1 <= len(row["images"]) <= 7
        if pro:
            assert row["question_type"] == "multiple-choice"
            assert row["answer"] in [chr(65 + n) for n in range(len(row["options"]))]
            assert 1 <= len(row["images"]) <= 7
            slots = row["used_image_slots"]
            assert slots == sorted(set(slots)) and set(slots) <= set(row["original_image_slots"])
            assert len(slots) == len(row["images"])
            assert [ref["field"] for ref in row["image_references"]] == [f"image_{n}" for n in slots]
        assert row["seed"] == sample_seed(3407, row["id"])
        assert row["input_sha256"] == digest(row["input_tensors"])
        assert len(row["generated_token_ids"]) == row["generated_tokens"]
        assert row["model_explanation"] is None or row["model_explanation"] in row["raw_response"]
        assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user"
        content = row["messages"][0]["content"]
        assert content[-1] == {"type": "text", "text": row["prompt"]}
        assert len(content) - 1 == len(row["image_grid_thw"]) == len(row["images"])
        assert [item["image"] for item in content[:-1]] == row["images"]
        for image_index, image in enumerate(content[:-1]):
            assert image["type"] == "image"
            image_path = resolve_image(run_dir, image["image"])
            assert sha256_file(image_path) == image["sha256"]
            image_paths.add(image["image"])
            if image_path not in checked_images:
                with Image.open(image_path) as decoded:
                    decoded.verify()
                checked_images.add(image_path)
            if pro:
                if image_path not in pixel_identities:
                    with Image.open(image_path) as decoded:
                        rgb = decoded.convert("RGB")
                        pixel_identities[image_path] = (list(rgb.size), hashlib.sha256(rgb.tobytes()).hexdigest())
                reference = row["image_references"][image_index]
                assert reference["content_index"] == image_index
                assert (reference["original_size"], reference["pixels_sha256"]) == pixel_identities[image_path]
        if managed:
            iid, vid = row["inference_id"], row["inference_invocation_id"]
            assert re.fullmatch(r"[0-9a-f]{32}", iid) and iid not in inference_ids
            assert row["run_id"] == run_dir.name and row["inference_started_at"]
            invocation = invocations[vid]
            assert invocation["id"] == vid and invocation["run_id"] == run_dir.name
            assert invocation["git_commit"] == identity["git_commit"]
            # The atomic sample is completion evidence even if a crash preceded telemetry update.
            assert invocation["completed_ids"].count(row["id"]) <= 1
            assert invocation["attempted_ids"].count(row["id"]) == 1
            inference_ids.add(iid)
            invocation_ids.add(vid)
        score = score_with_parser(row["raw_response"], row["question_type"], row["options"], row["answer"],
                                  eval_config["parser"], row.get("finish_reason"), question=row.get("question", ""))
        assert all(row[k] == v for k, v in score.items()), row["id"]
        correct += row["correct"]
    assert summary["correct"] == correct
    macro = sum(sum(r["correct"] for r in rows if r["subject"] == subject) / total
                for subject, total in subject_counts.items()) / 30
    assert math.isclose(summary["macro_average"], macro, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(summary["accuracy"], correct / count, rel_tol=0, abs_tol=1e-12)
    assert len(summary["subject_scores"]) == 30
    import csv
    with (run_dir / "subject_scores.csv").open() as stream:
        table = {item["subject"]: item for item in csv.DictReader(stream)}
    assert set(table) == set(SUBJECTS)
    for item in summary["subject_scores"]:
        selected = [row for row in rows if row["subject"] == item["subject"]]
        assert item["total"] == subject_counts[item["subject"]] and item["correct"] == sum(row["correct"] for row in selected)
        assert int(table[item["subject"]]["correct"]) == item["correct"]
        assert int(table[item["subject"]]["total"]) == item["total"]
        assert math.isclose(float(table[item["subject"]]["accuracy"]), item["correct"] / item["total"])
        for field, source in (("generation_seconds", "generation_seconds"), ("sample_seconds", "total_sample_seconds")):
            assert math.isclose(item[field], math.fsum(row[source] for row in selected), rel_tol=1e-12)
            assert math.isclose(float(table[item["subject"]][field]), item[field], rel_tol=1e-12)
    assert read_json(public_dir / "summary.json") == summary
    assert public["identity"] == manifest["identity"]
    for entry in public["external_files"]:
        path = run_dir / entry["path"]
        assert not Path(entry["path"]).is_absolute() and ".." not in Path(entry["path"]).parts
        assert not path.is_symlink() and path.resolve().is_relative_to(run_dir.resolve())
        assert path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"]
    import json
    references = [json.loads(line) for line in (public_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(references) == count and {r["id"] for r in references} == {r["id"] for r in rows}
    for reference in references:
        assert not Path(reference["external_record"]).is_absolute() and ".." not in Path(reference["external_record"]).parts
        assert sha256_file(run_dir / reference["external_record"]) == reference["sha256"]
    jsonl_rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(jsonl_rows) == count and {r["id"]: r for r in jsonl_rows} == {r["id"]: r for r in rows}
    viewer = ViewerImages()
    viewer.feed((run_dir / "review.html").read_text())
    assert image_paths <= set(viewer.images), "Viewer is missing viewable images"
    print(f"VERIFIED: {count} IDs, 30 subjects, {correct}/{count}, hashes, timings, viewer images and raw rescore")
    return {"run_id": run_dir.name, "completed_count": count, "correct": correct, "identity": identity,
            "inference_ids": sorted(inference_ids), "inference_invocation_ids": sorted(invocation_ids),
            "viewable_images": len(checked_images)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--public-dir", required=True, type=Path)
    args = parser.parse_args()
    validate(args.run_dir, args.public_dir)
