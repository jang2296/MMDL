import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from mmdl.evaluation.backends.vllm_backend import (
    MAX_MODEL_LEN, KVAdmission, VLLMBackend, allocated_kv_capacity, observe_reservation_engine)
from mmdl.evaluation.datasets import mmmu_test
from mmdl.evaluation.writer import RunWriter, finalize
from mmdl.runtime import bundle, contracts, reproduce
from mmdl.runtime.artifacts import digest, sha256_file, write_json
import test_fast_backends as fast
from test_submission_backends import scheduler

ROOT = Path(__file__).resolve().parents[2]
PID = contracts.MMMU_TEST_PROFILE
# Frozen before the test suite was added (HEAD 44da907): existing tables and protocol bytes.
EXISTING_TABLES_SHA256 = "e7692a3c5579fd852dfdea3a103b563b0758f4ea9f6ccd96914fd88bc67d1c04"
EXISTING_PROTOCOLS = {
    "mmmu_pro_standard10_v8_4090.yaml": "7383a10de6bdc2865f393d729b167106ade34ca45ff38cd4871f78c310ec3d07",
    "mmmu_pro_standard10_v8_a100.yaml": "1dd45bd63dc830a625223d55481eaf1794bd81ac2b22ccc5e0b0a3a647a0bb66",
    "mmmu_val_continuous_vllm_v1.yaml": "07b7b23d38bb5911ea12c670bd0e4cb8e2e58f2827243bc4c67bb870e3c8f4b5",
    "mmmu_val_control_a_v2.yaml": "fc4b2e8cf708082fd68d22cea76d04ec9ab2e05c61ba6a50a0c162ddf300f51f",
    "mmmu_val_control_b_v2.yaml": "357e22048facc9700c31ef8fc1af9e9b882d7a1a81a653ef70b8bebe167926cc",
    "mmmu_val_control_c_v2.yaml": "e3ed1bd70c56dc1d162a7896e24019e1b0d856b7efb605b595b2cfb42524d06b",
    "mmmu_val_fast_transformers_v1.yaml": "efaaa9ba6cda4ab4cf87976cb252250a60c69b2b73ee842bf4c4780a2c02cb13",
    "mmmu_val_fast_vllm_v1.yaml": "ddf46f594cc60d7a5ed1647d2fb0a7e407cf058e7a90b4e66042835af4c72d2d",
    "mmmu_val_official_vllm_b2_v2.yaml": "4dc0acc7f046626fe346848b1e764c4bcabaca81c2917a414c028b5e52303527",
    "mmmu_val_v1.yaml": "57850c23923cd08162a103eba43bb5734c7e7f23dfa1ee6d4b6fa576008f2c70",
    "mmmu_val_v18b_ft_4090.yaml": "0d563e76b14652fc6d60b298d882c13bc899d5a6ba9cf8e64bb6882668674ed8",
    "mmmu_val_v18b_ft_a100.yaml": "03038a0f659e24f093a54ca127de47ea410fc11b6e2699920bca683acb02c8c9",
    "mmmu_val_v8.yaml": "96c0948813d9b379798e0ec1029bceb6a1953defa251c5a07a316ed0e125e200",
    "v18b_reference_00fffac.json": "bf6580580982f24980705fbec2876e05107869b85e05a349ae5b8734b6e3bfce",
}
IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def _png():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _fixture(root):
    """One item per subject; Math is an open item whose official gold is a list."""
    official, files = {}, []
    for subject in contracts.SUBJECTS:
        sid, open_item = f"test_{subject}_1", subject == "Math"
        answer = "['1.5', '1.50']" if open_item else "B"
        official[sid] = {"question_type": "open" if open_item else "multiple-choice",
                         "ground_truth": ["1.5", "1.50"] if open_item else "B"}
        columns = {"id": [sid], "question": ["Q <image 1>"], "options": ["[]" if open_item else "['x', 'y']"],
                   "explanation": [""], "answer": [answer], "question_type": [official[sid]["question_type"]]}
        table = pa.table(columns | {f"image_{n}": pa.array([{"bytes": _png(), "path": None} if n == 1 else None],
                                                            type=IMAGE_TYPE) for n in range(1, 8)})
        name = f"{subject}/test-00000-of-00001.parquet"
        (root / subject).mkdir(parents=True)
        pq.write_table(table, root / name)
        files.append({"path": name, "bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)})
    return official, files


def _pin(root, official, files):
    (root / mmmu_test.ANSWER_FILE).write_text(json.dumps(official))
    source = dict(contracts.TEST_ANSWER_SOURCE, sha256=sha256_file(root / mmmu_test.ANSWER_FILE))
    answer = dict(path=mmmu_test.ANSWER_FILE, bytes=(root / mmmu_test.ANSWER_FILE).stat().st_size, **source)
    manifest = {"repo_id": contracts.DATA_ID, "revision": contracts.DATA_REVISION, "files": files,
                "answer_file": answer}
    write_json(root / "manifest.json", manifest)
    write_json(root / "pinned.json", manifest)
    return source


class MMMUTestSuiteTests(unittest.TestCase):
    def test_profile_registration_leaves_existing_profiles_and_protocols_unchanged(self):
        tables = (contracts.EXECUTION_PROFILES, contracts.SUBMISSION_PROFILES, contracts.CONTROL_IMAGES)
        self.assertEqual(digest([{k: v for k, v in t.items() if k != PID} for t in tables]), EXISTING_TABLES_SHA256)
        for name, checksum in EXISTING_PROTOCOLS.items():
            self.assertEqual(sha256_file(ROOT / "configs/eval" / name), checksum, name)
        for hardware in ("a100_sxm_80gb", "rtx4090_24gb"):
            cfg, hw = contracts.load_configs(ROOT / "configs/eval/mmmu_test_lowres2k_v8.yaml",
                                             ROOT / f"configs/hardware/{hardware}.yaml")
        self.assertEqual(contracts.suite_of(cfg), "mmmu-test")
        self.assertEqual((cfg["generation"]["max_new_tokens"], cfg["execution"]["max_model_len"]), (2048, 16384))
        self.assertEqual(cfg["image"], contracts.CONTROL_IMAGES["mmmu-val-control-b-v2"])
        for kind in ("base", "unmerged-lora"):
            contracts.validate_profile_model(cfg, kind)
        for section, key, value in (("generation", "max_new_tokens", 32768), ("execution", "batch_size", 9),
                                    ("execution", "admission", "fixed")):
            changed = json.loads(json.dumps(cfg))
            changed[section][key] = value
            with self.assertRaises(ValueError):
                contracts.validate_configs(changed, hw)

    def test_load_test_checks_integrity_and_every_gold_against_the_answer_file(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(mmmu_test, "TOTAL", 30):
            root = Path(temp)
            official, files = _fixture(root)
            source = _pin(root, official, files)
            with patch.object(mmmu_test, "TEST_ANSWER_SOURCE", source):
                datasets, coverage = mmmu_test.load_test(root, root / "pinned.json")
                self.assertEqual(len(coverage["ordered_ids"]), 30)
                self.assertEqual(coverage["question_types"], {"multiple-choice": 29, "open": 1})
                self.assertEqual(set(coverage["subject_counts"].values()), {1})
                official["test_Art_1"]["ground_truth"] = "A"
                _pin(root, official, files)
                with self.assertRaises(ValueError):  # answer bytes no longer match the pinned source hash
                    mmmu_test.load_test(root, root / "pinned.json")
            with patch.object(mmmu_test, "TEST_ANSWER_SOURCE", _pin(root, official, files)), \
                    self.assertRaisesRegex(ValueError, "official answer file: test_Art_1"):
                mmmu_test.load_test(root, root / "pinned.json")
            (root / files[0]["path"]).write_bytes(b"changed")
            with patch.object(mmmu_test, "TEST_ANSWER_SOURCE", _pin(root, official, files)), \
                    self.assertRaisesRegex(ValueError, "integrity"):
                mmmu_test.load_test(root, root / "pinned.json")

    def test_test_ids_are_accepted_only_by_the_test_splitter(self):
        row = {"id": "test_Math_5", "question": "Q <image 1>", "options": "[]", "question_type": "open",
               "answer": "40", "explanation": "", "image_1": {"bytes": _png(), "path": None}}
        sample, images, gold = mmmu_test.separate_sample(row, "Math")
        self.assertEqual((sample["id"], len(images), gold["answer"]), ("test_Math_5", 1, "40"))
        self.assertNotIn("answer", sample)
        for bad, subject in (("validation_Math_5", "Math"), ("test_Math_5", "Physics")):
            with self.assertRaises(ValueError):
                mmmu_test.separate_sample(row | {"id": bad}, subject)
        _, _, listed = mmmu_test.separate_sample(row | {"answer": "['1/12', '0.083']"}, "Math")
        self.assertEqual(listed["answer"], ["1/12", "0.083"])
        from mmdl.evaluation.final_answer_parser_v8 import score_response
        self.assertTrue(score_response("Final answer: 0.083", "open", [], listed["answer"])["correct"])

    def test_kv_admission_reserves_rounded_contexts_and_refuses_impossible_requests(self):
        admission = KVAdmission(usable_tokens=100, block_size=10, slots=2, max_new_tokens=10)
        self.assertTrue(admission.admit("a", 25))  # ceil(35/10)*10 = 40
        self.assertFalse(admission.admit("b", 55))  # 40 + 70 > 100
        self.assertTrue(admission.admit("c", 45))  # 40 + 60 = 100
        self.assertFalse(admission.admit("d", 1))  # slot cap
        admission.release("a")
        self.assertTrue(admission.admit("d", 1))
        with self.assertRaisesRegex(ValueError, "can never fit"):
            admission.admit("e", 91)
        self.assertEqual(admission.summary()["max_concurrent_requests"], 2)

    def test_stream_holds_the_head_request_and_never_skips_ahead(self):
        ids = [7, 99, 99, 99, 99]
        engine = fast._ContinuousEngine([
            [], [fast._Output(ids, (1,), "b")], [fast._Output(ids, (1,), "a")],
            [fast._Output(ids, (1,), "c"), fast._Output(ids, (1,), "d")],
        ])
        backend = self._sized_backend(engine, {"a": 30, "b": 50, "c": 70, "d": 5}, slots=64)
        backend.kv_admission_limits = dict(usable_tokens=120, block_size=10, slots=64, max_new_tokens=10)
        rows = list(backend.generate_stream([fast._request(name, n) for n, name in enumerate("abcd")]))
        adds = [event[1] for event in engine.events if event[0] == "add"]
        self.assertEqual(adds, ["a", "b", "c", "d"])
        # c (80) waits for b to finish; the small d (20) would fit beside a and b but never jumps ahead of c.
        events = engine.events
        self.assertLess(events.index(("step", ("b",))), events.index(("add", "c")))
        self.assertLess(events.index(("step", ("a",))), events.index(("add", "d")))
        self.assertEqual([row["sample_id"] for row in rows], ["b", "a", "c", "d"])
        self.assertEqual(backend.last_admission.summary()["max_concurrent_requests"], 2)
        self.assertEqual(backend.last_admission.reserved, {})

    def test_stream_aborts_active_requests_when_a_request_can_never_fit(self):
        engine = fast._ContinuousEngine([[]])
        backend = self._sized_backend(engine, {"first": 3, "second": 3, "huge": 9}, slots=3)
        backend.kv_admission_limits = dict(usable_tokens=10, block_size=1, slots=3, max_new_tokens=2)
        with self.assertRaisesRegex(ValueError, "can never fit"):
            list(backend.generate_stream([fast._request(sid, n) for n, sid in enumerate(("first", "second", "huge"))]))
        self.assertEqual(engine.aborted, [["first", "second"]])

    @staticmethod
    def _sized_backend(engine, sizes, *, slots):
        backend = fast.FastBackendTests._stream_backend(engine)
        backend.batch_size, backend.max_model_len = slots, MAX_MODEL_LEN
        prepare = backend._prepare_request

        def sized(request, **kwargs):
            item, prompt, params = prepare(request, **kwargs)
            return item | {"input_tokens": sizes[request["sample_id"]]}, prompt, params

        backend._prepare_request = sized
        return backend

    def test_reservation_capacity_and_engine_observation(self):
        value = allocated_kv_capacity(scheduler(blocks=2000), "0.11.0", max_model_len=16384, slots=64,
                                      admission="kv_reservation")
        self.assertTrue(value["pass"])
        self.assertEqual((value["usable_tokens"], value["required_tokens"]), (28800, 16384))
        with self.assertRaises(ValueError):
            allocated_kv_capacity(scheduler(blocks=1000), "0.11.0", max_model_len=16384, slots=64,
                                  admission="kv_reservation")
        self.assertEqual(allocated_kv_capacity(scheduler(), "0.11.0", max_model_len=40960, slots=9)["version"],
                         "submission-kv-v1")
        engine = scheduler(blocks=2000)
        sampler = NS(logprobs_mode="raw_logprobs")
        engine.engine_core.engine_core.model_executor = NS(driver_worker=NS(worker=NS(model_runner=NS(sampler=sampler))))
        engine.vllm_config = NS(
            model_config=NS(enforce_eager=False, dtype="torch.bfloat16", quantization=None,
                            logprobs_mode="raw_logprobs", max_model_len=16384),
            scheduler_config=NS(max_num_seqs=64),
            compilation_config=NS(level=3, cudagraph_mode=NS(name="FULL_AND_PIECEWISE"), cudagraph_capture_sizes=[1]))
        backend = NS(llm=NS(llm_engine=engine), vllm_version="0.11.0")
        self.assertEqual(observe_reservation_engine(backend, "cuda_graph", 64, 16384)["slots"], 64)
        with self.assertRaises(ValueError):
            observe_reservation_engine(backend, "cuda_graph", 64, 40960)

    def test_constructor_uses_measured_admission_for_the_test_profile(self):
        cfg, hw = contracts.load_configs(ROOT / "configs/eval/mmmu_test_lowres2k_v8.yaml",
                                         ROOT / "configs/hardware/a100_sxm_80gb.yaml")
        captured = {}

        class LLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.llm_engine = scheduler(blocks=27775)

            def get_default_sampling_params(self):
                return "defaults"

        processor = NS(image_processor=NS(size={}), tokenizer=NS(eos_token_id=2, pad_token_id=0))
        prefix = "mmdl.evaluation.backends.vllm_backend."
        with patch(prefix + "torch.cuda.is_available", return_value=True), \
                patch(prefix + "torch.cuda.is_bf16_supported", return_value=True), \
                patch(prefix + "AutoProcessor.from_pretrained", return_value=processor), \
                patch(prefix + "AutoConfig.from_pretrained", return_value=NS()), \
                patch(prefix + "_vllm", return_value=(LLM, dict, "0.11.0")), \
                patch(prefix + "_final_only_output_kind", return_value="FINAL_ONLY"), \
                patch(prefix + "observe_reservation_engine", return_value={"pass": True}) as observed, \
                patch("fine_tuning.error_note.v17_eval_engine.observe_engine", side_effect=AssertionError), \
                patch.dict("os.environ", {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}):
            backend = VLLMBackend("/synthetic", cfg, hw)
        self.assertEqual((captured["max_num_seqs"], captured["max_model_len"]), (64, 16384))
        self.assertEqual(captured["limit_mm_per_prompt"], {"image": 7, "video": 0})
        self.assertEqual(observed.call_args.args[1:], ("cuda_graph", 64, 16384))
        self.assertEqual(backend.capacity_receipt["usable_tokens"], 399960)  # 444,400 measured A100 tokens
        self.assertEqual(backend.kv_admission_limits, dict(usable_tokens=399960, block_size=16, slots=64,
                                                           max_new_tokens=2048))

    def test_suite_counts_roles_and_finalization(self):
        self.assertEqual(reproduce.SUITE_COUNTS, contracts.SUITE_COUNTS)
        self.assertEqual(reproduce.SUITE_DATA_DIRS, contracts.SUITE_DATA_DIRS)
        self.assertEqual(bundle._SUITE_ROLES["mmmu-pro"], bundle._SUITE_ROLES["mmmu-test"])
        self.assertEqual([bundle._suite_count(s) for s in ("mmmu-val-two-runs", "mmmu-pro", "mmmu-test")],
                         [900, 1730, 10500])
        ids = ["test_Math_2", "test_Art_1"]
        identity = {"suite": "mmmu-test", "id_subjects": {"test_Math_2": "Math", "test_Art_1": "Art"}}
        with tempfile.TemporaryDirectory() as temp:
            with RunWriter(Path(temp) / "run", identity, ids) as writer:
                for sid, subject in identity["id_subjects"].items():
                    writer.save({"id": sid, "subject": subject, "correct": sid == "test_Art_1",
                                 "parse_status": "PARSED", "generated_tokens": 1,
                                 "generation_seconds": 1.0, "finish_reason": "stop"})
            summary = finalize(Path(temp) / "run", ids, "smoke")
            self.assertEqual((summary["suite"], summary["correct"], summary["denominator"]), ("mmmu-test", 1, 2))
            with self.assertRaisesRegex(ValueError, "10500"):
                finalize(Path(temp) / "run", ids, "full")
            wrong = {"suite": "mmmu-test", "id_subjects": {"validation_Math_2": "Math"}}
            RunWriter(Path(temp) / "wrong", wrong, ["validation_Math_2"]).close()
            with self.assertRaisesRegex(ValueError, "mapping mismatch"):
                finalize(Path(temp) / "wrong", ["validation_Math_2"], "smoke")

    def test_cli_and_operator_defaults_select_the_test_protocol_and_supplemental_scoring(self):
        from mmdl.evaluation import cli

        _, args = cli.arguments(["--suite", "mmmu-test", "--model", "finetuned", "--run-id", "t"])
        self.assertEqual((args.protocol.name, args.score), ("mmmu_test_lowres2k_v8.yaml", "v8"))
        self.assertEqual(args.data_root.parts[-2:], ("evaluation", "mmmu_test"))
        _, args = cli.arguments(["--suite", "mmmu-test", "--score", "both", "--run-id", "t"])
        self.assertEqual(args.score, "both")
        hardware = ROOT / "configs/hardware/a100_pcie_80gb.yaml"
        plan = reproduce.arguments(["--suite", "mmmu-test", "--model", "finetuned", "--score", "both",
                                    "--hardware", str(hardware), "--job-id", "t", "--commit", "0" * 40])
        self.assertEqual((plan.protocol.name, plan.model_ref.name), ("mmmu_test_lowres2k_v8.yaml", "v18b.json"))
        operator = reproduce._operator_plan(plan, ROOT)
        self.assertIn("CompassVerifier-7B", operator["scoring"]["supplemental"])
        evaluation = next(c["argv"] for c in operator["commands"] if c["stage"] == "evaluation")
        self.assertEqual(evaluation[evaluation.index("--score") + 1], "both")

    def test_a100_pcie_profile_matches_real_device_names(self):
        import re

        pattern = reproduce.HARDWARE_PROFILES["a100_pcie_80gb.yaml"][0]
        for name in ("NVIDIA A100 80GB PCIe", "NVIDIA A100-PCIE-80GB"):
            self.assertTrue(re.fullmatch(pattern, name, re.IGNORECASE), name)
        self.assertIsNone(re.fullmatch(reproduce.HARDWARE_PROFILES["a100_sxm_80gb.yaml"][0],
                                       "NVIDIA A100 80GB PCIe", re.IGNORECASE))
        cfg, hw = contracts.load_configs(ROOT / "configs/eval/mmmu_test_lowres2k_v8.yaml",
                                         ROOT / "configs/hardware/a100_pcie_80gb.yaml")
        self.assertEqual(hw["name"], "a100_pcie_80gb")
        contracts.load_configs(ROOT / "configs/eval/mmmu_pro_standard10_v8_a100.yaml",
                               ROOT / "configs/hardware/a100_pcie_80gb.yaml")

    def test_bounded_rehearsal_canonicalizes_to_the_registered_profile(self):
        import yaml
        from mmdl.evaluation.cli import load_evaluation_configs

        cfg = yaml.safe_load((ROOT / "configs/eval/mmmu_test_lowres2k_v8.yaml").read_text())
        cfg["generation"]["max_new_tokens"] = 16
        cfg["execution"].update(batch_size=2, gpu_memory_utilization=.62)
        hw = yaml.safe_load((ROOT / "configs/hardware/rtx5060_8gb.yaml").read_text()) | {"placement": "gpu_only"}
        with tempfile.TemporaryDirectory() as temp:
            protocol, hardware = Path(temp) / "p.yaml", Path(temp) / "h.yaml"
            protocol.write_text(yaml.safe_dump(cfg))
            hardware.write_text(yaml.safe_dump(hw))
            loaded, _ = load_evaluation_configs(argparse.Namespace(rehearsal=True, protocol=protocol, hardware=hardware))
            self.assertTrue(loaded["rehearsal"])
            cfg["execution"]["max_model_len"] = 8192
            protocol.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):
                load_evaluation_configs(argparse.Namespace(rehearsal=True, protocol=protocol, hardware=hardware))


if __name__ == "__main__":
    unittest.main()
