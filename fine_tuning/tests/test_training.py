"""CPU checks with a tiny real Qwen3-VL architecture, no pretrained weights."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from fine_tuning.modules import (ARMS, configure_arm, inspect_modules, load_checkpoint,
                                 parameter_group, save_checkpoint)
from fine_tuning.train import (AssistantCollator, approved_records, file_hash, optimizer_update,
                               parameter_probes, probe_indices, validate_training_input_contract)


def tiny_model():
    torch.manual_seed(17)
    config = Qwen3VLConfig(
        text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 3, "num_attention_heads": 2,
                     "num_key_value_heads": 2, "head_dim": 8,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 3, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "out_hidden_size": 16,
                       "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1, 2]},
        image_token_id=3, video_token_id=6, vision_start_token_id=4, vision_end_token_id=5,
        eos_token_id=2, pad_token_id=0,
    )
    return Qwen3VLForConditionalGeneration(config)


def check_real_partition_gradient_update_and_reload(arm, tmp_path):
    model, inventory, groups = configure_arm(tiny_model(), arm)
    assert len(inventory["mergers"]) == 4
    assert {r["group"] for r in inventory["trainable"]} == set(ARMS[arm])
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == inventory["trainable_count"]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    probes = parameter_probes(model, inventory)
    # A real image path crosses frozen L during FV/FM; all DeepStack branches participate.
    batch = {"input_ids": torch.tensor([[4, 3, 3, 3, 3, 5, 9, 10, 11, 2]]),
             "attention_mask": torch.ones(1, 10, dtype=torch.long),
             "pixel_values": torch.randn(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]]),
             "labels": torch.tensor([[-100] * 7 + [10, 11, 2]])}
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    (model(**batch).loss * 3).backward()
    metrics = optimizer_update(model, optimizer, 3, inventory, probes)
    assert all(value > 0 for value in metrics["gradient_l2"].values())
    assert all(value > 0 for value in metrics["sampled_update_l1"].values())
    destination = tmp_path / "checkpoint"
    save_checkpoint(model, destination, {"arm": arm}, inventory)
    restored = load_checkpoint(tiny_model(), destination)
    assert set(dict(restored.named_parameters())) == set(dict(model.named_parameters()))
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert torch.equal(param, dict(restored.named_parameters())[name])
    if "M" in ARMS[arm]:
        saved = torch.load(destination / "mergers.pt", weights_only=True)
        assert len(saved) == sum(1 for n, p in model.named_parameters()
                                 if p.requires_grad and parameter_group(n, inventory) == "M")


def check_missing_deepstack_rejected():
    model = tiny_model()
    model.model.visual.deepstack_merger_list = torch.nn.ModuleList([])
    with unittest.TestCase().assertRaisesRegex(ValueError, "four mergers"):
        inspect_modules(model)


class FakeProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(patch_size=2, merge_size=2)
        self.tokenizer = SimpleNamespace(eos_token_id=6, convert_tokens_to_ids=lambda _: 99,
                                         all_special_tokens=["<|im_end|>"])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "prefix" if add_generation_prompt else "prefixanswer"

    def __call__(self, text, **kwargs):
        ids = [2, 99, 3, 4, 5] if text == ["prefix"] else [2, 99, 3, 4, 5, 7, 8, 6, 0]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([[int(i != 0) for i in ids]]),
                "image_grid_thw": torch.tensor([[1, 2, 2]])}


def check_assistant_mask_eos_padding_and_budgets(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "image.png")
    record = {"image_paths": ["image.png"], "user_text": "question", "target_text": "answer"}
    collator = AssistantCollator(FakeProcessor(), tmp_path, 9, 16)
    assert collator(record)["labels"].tolist() == [[-100] * 5 + [7, 8, 6, -100]]
    with unittest.TestCase().assertRaisesRegex(ValueError, "overflow"):
        AssistantCollator(FakeProcessor(), tmp_path, 8, 16)(record)
    with unittest.TestCase().assertRaisesRegex(ValueError, "image budget"):
        AssistantCollator(FakeProcessor(), tmp_path, 9, 15)(record)


def check_manifest_permissions_lineage_and_byte_hashes(tmp_path):
    import fine_tuning.train as training

    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(image)
    record = {"id": "nvw_1", "split": "train", "training_allowed": True, "benchmark_derived": False,
              "lineage": ["clean_generator"], "image_paths": ["image.png"], "user_text": "question",
              "target_text": "answer"}
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(record) + "\n")
    manifest = {"status": "APPROVED", "training_allowed": True, "benchmark_derived": False,
                "contamination_audit": {"status": "PASS"}, "records_sha256": file_hash(records),
                "record_count": 1, "known_parent_ids": ["clean_generator"],
                "files": {"image.png": file_hash(image)},
                "source_sha256": {n: file_hash(Path(training.__file__).with_name(n)) for n in ("train.py", "modules.py")}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert approved_records(records, manifest_path, tmp_path)[0] == [record]
    for name in ("train.py", "modules.py"):
        original = manifest["source_sha256"][name]
        manifest["source_sha256"][name] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with unittest.TestCase().assertRaisesRegex(ValueError, "source differs"):
            approved_records(records, manifest_path, tmp_path)
        manifest["source_sha256"][name] = original
    manifest["training_allowed"] = False
    manifest_path.write_text(json.dumps(manifest))
    with unittest.TestCase().assertRaisesRegex(ValueError, "approved clean"):
        approved_records(records, manifest_path, tmp_path)
    manifest["training_allowed"] = True
    manifest["known_parent_ids"] = []
    manifest_path.write_text(json.dumps(manifest))
    with unittest.TestCase().assertRaisesRegex(ValueError, "lineage"):
        approved_records(records, manifest_path, tmp_path)
    manifest["known_parent_ids"] = ["clean_generator"]
    manifest["benchmark_derived"] = True
    manifest_path.write_text(json.dumps(manifest))
    with unittest.TestCase().assertRaisesRegex(ValueError, "approved clean"):
        approved_records(records, manifest_path, tmp_path)


def check_checkpoint_corruption_rejected(tmp_path):
    model, inventory, _ = configure_arm(tiny_model(), "FM")
    save_checkpoint(model, tmp_path / "checkpoint", {}, inventory)
    path = tmp_path / "checkpoint" / "mergers.pt"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with unittest.TestCase().assertRaisesRegex(ValueError, "file mismatch"):
        load_checkpoint(tiny_model(), tmp_path / "checkpoint")


class TrainingTests(unittest.TestCase):
    def test_runtime_input_contract_parent_child_budgets_and_hash(self):
        from fine_tuning import audit

        parent = audit.training_input_contract()
        child_path = Path(audit.__file__).with_name("bridge_training_protocol_v1.json")
        for protocol in (None, child_path):
            contract = audit.training_input_contract(protocol)
            budgets = {key: contract[key] for key in
                       ("min_pixels", "max_pixels", "episode_pixels", "sequence_budget")}
            proof = {"status": "PASS", "processor_revision": contract["processor_revision"],
                     "record_count": 1, **budgets}
            if protocol is not None:
                proof["training_input_contract"] = contract
            manifest = {"record_count": 1, "model_input_audit": proof}
            self.assertEqual(validate_training_input_contract(manifest, **budgets,
                             training_input_protocol=protocol), contract)
            for key in budgets:
                with self.subTest(protocol=protocol, budget=key), self.assertRaises(ValueError):
                    validate_training_input_contract(manifest, **{**budgets, key: budgets[key] * 2},
                                                     training_input_protocol=protocol)
            for key, value in (("status", "PENDING"), ("processor_revision", "0" * 40),
                               ("record_count", 2), ("sequence_budget", 8192),
                               ("episode_pixels", 1), ("min_pixels", 131072),
                               ("max_pixels", 524288)):
                bad = copy.deepcopy(manifest)
                bad["model_input_audit"][key] = value
                with self.subTest(protocol=protocol, proof=key), self.assertRaises(ValueError):
                    validate_training_input_contract(bad, **budgets, training_input_protocol=protocol)
            if protocol is None:
                legacy = copy.deepcopy(manifest)
                for key in ("min_pixels", "max_pixels"):
                    del legacy["model_input_audit"][key]
                self.assertEqual(validate_training_input_contract(legacy, **budgets), parent)
                with self.assertRaises(ValueError):
                    validate_training_input_contract(manifest, **{**budgets, "episode_pixels": 3145728})
            else:
                for key in ("min_pixels", "max_pixels", "training_input_contract"):
                    bad = copy.deepcopy(manifest)
                    del bad["model_input_audit"][key]
                    with self.subTest(missing=key), self.assertRaises(ValueError):
                        validate_training_input_contract(bad, **budgets, training_input_protocol=protocol)
                bad = copy.deepcopy(manifest)
                bad["model_input_audit"]["training_input_contract"]["protocol_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    validate_training_input_contract(bad, **budgets, training_input_protocol=protocol)
                with self.assertRaises(ValueError):
                    validate_training_input_contract(manifest, **budgets)
                with self.assertRaises(ValueError):
                    validate_training_input_contract(manifest, **budgets,
                        training_input_protocol=child_path.with_name("bridge_input_protocol_v1.json"))

    def test_main_rejects_input_budget_mismatch_before_cuda_or_model_load(self):
        import peft
        import transformers
        from fine_tuning import audit, train

        contract = audit.training_input_contract()
        manifest = {"record_count": 1, "model_input_audit": {
            "status": "PASS", "record_count": 1, "processor_revision": contract["processor_revision"],
            "sequence_budget": 4096, "episode_pixels": 1048576}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["train", "--model-path", str(root/"unused-model"), "--records", str(root/"unused-records"),
                    "--manifest", str(root/"unused-manifest"), "--image-root", str(root),
                    "--output", str(root/"output"), "--arm", "FL", "--min-pixels", "262144",
                    "--max-pixels", "1048576", "--episode-pixels", "3145728"]
            with patch("sys.argv", args), patch.object(torch, "__version__", "2.8.0+cu128"), \
                    patch.object(transformers, "__version__", "4.57.1"), \
                    patch.object(peft, "__version__", "0.17.1"), \
                    patch.object(train, "approved_records", return_value=([{"id": "fixture"}], manifest)), \
                    patch.object(torch.cuda, "is_available") as cuda, \
                    patch.object(train, "load_base") as load:
                with self.assertRaisesRegex(ValueError, "input contract"):
                    train.main()
                cuda.assert_not_called()
                load.assert_not_called()
            self.assertFalse((root/"output").exists())

    def test_probe_indices_large_parameters_never_round_out_of_bounds(self):
        for numel in (1, 17, 1024, 151936 * 2560, 2**31 + 17):
            with self.subTest(numel=numel):
                indices = probe_indices(numel)
                self.assertEqual(indices.dtype, torch.long)
                self.assertEqual(len(indices), min(1024, numel))
                self.assertEqual(int(indices[0]), 0)
                self.assertEqual(int(indices[-1]), numel - 1)
                self.assertEqual(len(indices.unique()), len(indices))
                self.assertTrue(bool(((indices >= 0) & (indices < numel)).all()))
        with self.assertRaisesRegex(ValueError, "empty parameter"):
            probe_indices(0)

    def test_real_architecture_all_four_arms(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                check_real_partition_gradient_update_and_reload(arm, Path(directory))

    def test_missing_deepstack_rejected(self):
        check_missing_deepstack_rejected()

    def test_assistant_mask_eos_padding_and_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            check_assistant_mask_eos_padding_and_budgets(Path(directory))

    def test_manifest_permissions_lineage_and_byte_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            check_manifest_permissions_lineage_and_byte_hashes(Path(directory))

    def test_checkpoint_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            check_checkpoint_corruption_rejected(Path(directory))


if __name__ == "__main__":
    unittest.main()
