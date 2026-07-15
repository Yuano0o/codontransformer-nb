import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ProjectPortabilityTests(unittest.TestCase):
    def test_colab_notebook_contains_required_workflow(self):
        path = ROOT / "notebooks" / "codontransformer_finetune_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "!nvidia-smi",
            "git\", \"clone",
            "drive.mount",
            "requirements-colab.txt",
            "download_pretrained.py",
            "configs/smoke_test_cuda_csi_top10.yaml",
            "SMOKE_SAMPLE_COUNT = 8",
            "--limit-train-batches",
            "last.ckpt",
            "validate_checkpoint_inference.py",
            "translation_verified",
        ):
            self.assertIn(required, source)

    def test_scripts_have_no_mac_user_absolute_paths(self):
        marker = "/" + "Users" + "/"
        for path in (ROOT / "scripts").glob("*.py"):
            self.assertNotIn(marker, path.read_text(encoding="utf-8"), str(path))

    def test_formal_top10_config_is_bounded_and_leak_resistant(self):
        path = ROOT / "configs" / "finetune_cuda_csi_top10.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["expected_records"],
            {"train": 4524, "validation": 531, "test": 594},
        )
        training = config["training"]
        self.assertEqual(training["accelerator"], "gpu")
        self.assertEqual(training["precision"], "16-mixed")
        self.assertEqual(training["batch_size"], 1)
        self.assertEqual(training["accumulate_grad_batches"], 8)
        self.assertGreaterEqual(training["max_epochs"], 3)
        self.assertLessEqual(training["max_epochs"], 5)
        self.assertEqual(training["save_top_k"], 1)
        self.assertGreaterEqual(training["early_stopping_patience"], 1)
        self.assertIn("validation_dataset_path", config["paths"])
        self.assertIn("test_dataset_path", config["paths"])

    def test_formal_top10_colab_entry_is_persistent_and_resumable(self):
        path = (
            ROOT
            / "notebooks"
            / "codontransformer_finetune_csi_top10_colab.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "!nvidia-smi",
            'EXPECTED_COUNTS = {"train": 4524, "validation": 531, "test": 594}',
            "finetune_cuda_csi_top10.yaml",
            "--validation-dataset-path",
            "--test-dataset-path",
            "--resume-from-checkpoint",
            "checkpoints\" / \"last.ckpt",
            "test_baseline_vs_finetuned.json",
            "--expected-records\", \"594",
            "translation_verified",
        ):
            self.assertIn(required, source)
        code_source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertNotIn("csi_top25_hc", code_source)
        self.assertNotIn("all_clean_hc", code_source)

    def test_biological_evaluation_colab_is_test_only_and_resumable(self):
        path = (
            ROOT
            / "notebooks"
            / "codontransformer_biological_evaluation_colab.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "test.jsonl",
            "best-epoch04-val_loss0.842573.ckpt",
            "codon_reference.json",
            "evaluate_biological_fidelity.py",
            "--expected-records\", \"594",
            "prediction caches",
            "biological_evaluation_summary.json",
        ):
            self.assertIn(required, source)
        self.assertNotIn("finetune_codontransformer.py", source)
        self.assertNotIn("trainer.fit", source)

    def test_refined_biological_evaluation_is_analysis_only(self):
        script = (
            ROOT / "scripts" / "refine_biological_evaluation.py"
        ).read_text(encoding="utf-8")
        for required in (
            "--per-sequence-csv",
            "--reference-json",
            "--bootstrap-samples",
            "synonymous_family_jsd_to_true",
            "length_stratified_paired_statistics",
            "required_stable_improvement_categories",
            "Refusing to overwrite existing refined outputs",
            "test_reuse_warning",
        ):
            self.assertIn(required, script)
        for forbidden in (
            "import torch",
            "from torch",
            "AutoModel",
            "trainer.fit",
        ):
            self.assertNotIn(forbidden, script)

    def test_validation_evaluation_freezes_refined_v2_inputs_and_boundaries(self):
        config = yaml.safe_load(
            (
                ROOT / "configs" / "evaluate_validation_refined_v2.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["evaluation_version"], "refined_v2")
        self.assertEqual(config["dataset_role"], "validation")
        self.assertEqual(config["inputs"]["expected_records"], 531)
        self.assertEqual(
            config["inputs"]["dataset_sha256"],
            "8e37ce0eff5684b1e42d6772fd3b0b6549a57d734fbb5e8fbc0ba4ec40058b49",
        )
        self.assertEqual(config["length_boundaries"]["short_max_aa"], 140.0)
        self.assertEqual(
            config["length_boundaries"]["medium_max_aa"],
            280.3333333333333,
        )
        self.assertEqual(config["statistics"]["bootstrap_samples"], 10000)
        self.assertEqual(config["statistics"]["seed"], 23)

    def test_validation_evaluation_colab_is_read_only_and_resumable(self):
        path = (
            ROOT
            / "notebooks"
            / "codontransformer_validation_evaluation_colab.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "evaluate_validation_refined_v2.yaml",
            'CONFIG["paths"]["validation_dataset"]',
            'CONFIG["inputs"]["dataset_sha256"]',
            "expected_records",
            "--dataset-role\", \"validation",
            "--expected-dataset-sha256",
            "--length-short-max",
            "--length-medium-max",
            "refine_biological_evaluation.py",
            'CONFIG["paths"]["output_directory"]',
            "Prediction caches",
            "--force",
        ):
            self.assertIn(required, source)
        self.assertNotIn("finetune_codontransformer.py", source)
        self.assertNotIn("trainer.fit", source)

    def test_validation_decoding_config_freezes_safe_strategies(self):
        config = yaml.safe_load(
            (
                ROOT / "configs" / "diagnose_validation_decoding.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["dataset_role"], "validation")
        self.assertEqual(config["inputs"]["expected_records"], 531)
        self.assertEqual(config["target_families"], ["E", "K", "L", "S"])
        self.assertEqual(config["statistics"]["seed"], 23)
        self.assertEqual(
            config["length_boundaries"],
            {
                "source": "frozen_refined_v2_test_boundaries",
                "short_max_aa": 140.0,
                "medium_max_aa": 280.3333333333333,
            },
        )
        strategies = config["strategies"]
        self.assertEqual(set(strategies), {
            "greedy",
            "temperature_sampling",
            "synonymous_family_sampling",
        })
        self.assertEqual(strategies["greedy"]["mode"], "argmax")
        self.assertEqual(strategies["temperature_sampling"]["temperature"], 0.5)
        self.assertEqual(strategies["temperature_sampling"]["top_p"], 0.95)
        self.assertEqual(
            strategies["synonymous_family_sampling"]["temperature"], 1.0
        )
        self.assertEqual(strategies["synonymous_family_sampling"]["top_p"], 1.0)

    def test_validation_decoding_script_enforces_translation_and_no_test_input(self):
        script = (
            ROOT / "scripts" / "diagnose_validation_decoding.py"
        ).read_text(encoding="utf-8")
        for required in (
            "--validation-dataset",
            'validation_dataset.name != "validation.jsonl"',
            "Refusing any dataset other than validation.jsonl",
            "AMINO_ACID_TO_INDEX",
            "translation_correct",
            "sequence_valid",
            "stable_seed",
            '"checkpoint_sha256": checkpoint_sha256',
            "argmax_concentration",
            "mean_position_normalized_entropy",
            "paired_strategy_comparisons_vs_greedy",
            "paired_strategy_comparisons_vs_greedy_by_length",
            "probability_diagnostics_by_length",
            "collapse_diagnosis_by_length",
            "test_access_prohibited",
        ):
            self.assertIn(required, script)
        for forbidden in (
            'add_argument("--test-dataset"',
            "trainer.fit",
            "finetune_codontransformer.py",
        ):
            self.assertNotIn(forbidden, script)

    def test_validation_decoding_colab_is_read_only_and_resumable(self):
        path = (
            ROOT
            / "notebooks"
            / "codontransformer_validation_decoding_colab.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "diagnose_validation_decoding.yaml",
            "diagnose_validation_decoding.py",
            "--validation-dataset",
            "record cache",
            "collapse_diagnosis",
            "test_access_prohibited",
            "--force-analysis",
        ):
            self.assertIn(required, source)
        self.assertNotIn("test.jsonl", source)
        self.assertNotIn("TEST_PATH", source)
        self.assertNotIn("trainer.fit", source)
        self.assertNotIn("finetune_codontransformer.py", source)

    def test_hybrid_decoding_config_is_a_strict_pre_v2_gate(self):
        config = yaml.safe_load(
            (
                ROOT / "configs" / "check_validation_hybrid_decoding.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["dataset_role"], "validation")
        self.assertEqual(config["inputs"]["expected_records"], 531)
        self.assertEqual(
            config["inputs"]["source_checkpoint_sha256"],
            "89e16c13e8b0bc0004d9552b254c30fe847374537a2de0c3854c1ba69f7c5c82",
        )
        self.assertEqual(
            set(config["candidates"]),
            {
                "s_temperature_only",
                "s_family_only",
                "high_entropy_temperature",
                "s_family_ekl_entropy_temperature",
            },
        )
        gate = config["selection_gate"]
        self.assertEqual(gate["translation_requirement"], 1.0)
        self.assertEqual(gate["validity_requirement"], 1.0)
        self.assertTrue(gate["require_stable_jsd_improvement"])
        self.assertTrue(gate["require_stable_rscu_improvement"])
        self.assertIn("csi", gate["protected_metrics"])
        self.assertIn("cai", gate["protected_metrics"])
        self.assertIn("gc3_absolute_error", gate["protected_metrics"])

    def test_hybrid_decoding_script_is_cache_only_and_validation_only(self):
        script = (
            ROOT / "scripts" / "check_validation_hybrid_decoding.py"
        ).read_text(encoding="utf-8")
        for required in (
            "--validation-dataset",
            "--source-decoding-dir",
            "record_cache",
            "finetuned.jsonl",
            "decoder_only_gate_passed",
            "proceed_to_v2_finetuning",
            '"test_access_prohibited": True',
            '"model_forward_performed": False',
            '"training_performed": False',
            "paired_comparisons_vs_v1_greedy_by_length",
        ):
            self.assertIn(required, script)
        for forbidden in (
            'add_argument("--test-dataset"',
            "BigBirdForMaskedLM",
            "AutoModel",
            "load_checkpoint",
            "trainer.fit",
        ):
            self.assertNotIn(forbidden, script)

    def test_hybrid_decoding_colab_does_not_load_a_model_or_test(self):
        path = (
            ROOT
            / "notebooks"
            / "codontransformer_validation_hybrid_decoding_colab.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "check_validation_hybrid_decoding.yaml",
            "check_validation_hybrid_decoding.py",
            "record_cache/finetuned.jsonl",
            "--source-decoding-dir",
            "proceed_to_v2_finetuning",
            "model_forward_performed",
            "training_performed",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "test.jsonl",
            "TEST_PATH",
            "download_pretrained.py",
            "BigBirdForMaskedLM",
            "load_checkpoint",
            "trainer.fit",
        ):
            self.assertNotIn(forbidden, source)

    def test_v2_configs_freeze_inputs_and_regularized_objective(self):
        smoke = yaml.safe_load(
            (
                ROOT / "configs" / "smoke_test_cuda_csi_top10_v2.yaml"
            ).read_text(encoding="utf-8")
        )
        formal = yaml.safe_load(
            (
                ROOT / "configs" / "finetune_cuda_csi_top10_v2.yaml"
            ).read_text(encoding="utf-8")
        )
        for config in (smoke, formal):
            self.assertEqual(
                config["expected_records"], {"train": 4524, "validation": 531}
            )
            self.assertNotIn("test_dataset_path", config["paths"])
            self.assertEqual(
                config["expected_sha256"]["train"],
                "bce089949fbbdee0c5e2403c22d6a1cba16a6cc851fdf731954af18f6b9df16d",
            )
            self.assertEqual(
                config["expected_sha256"]["validation"],
                "8e37ce0eff5684b1e42d6772fd3b0b6549a57d734fbb5e8fbc0ba4ec40058b49",
            )
            objective = config["objective"]
            self.assertEqual(objective["target_families"], ["E", "K", "L", "S"])
            self.assertEqual(objective["mlm_weight"], 1.0)
            self.assertEqual(objective["synonymous_distribution_weight"], 0.2)
            self.assertEqual(objective["csi_reference_weight"], 0.5)
            self.assertEqual(objective["cai_reference_weight"], 0.5)
        self.assertEqual(smoke["training"]["limit_train_batches"], 8)
        self.assertEqual(smoke["training"]["limit_val_batches"], 8)
        self.assertEqual(formal["training"]["precision"], "16-mixed")
        self.assertEqual(formal["training"]["accumulate_grad_batches"], 8)
        self.assertEqual(formal["training"]["save_every_n_steps"], 200)
        self.assertGreaterEqual(formal["training"]["max_epochs"], 3)
        self.assertLessEqual(formal["training"]["max_epochs"], 5)

    def test_v2_trainer_is_separate_resumable_and_validation_selected(self):
        source = (
            ROOT / "scripts" / "finetune_codontransformer_v2.py"
        ).read_text(encoding="utf-8")
        for required in (
            "synonymous_distribution_loss",
            "torch.isin",
            "val_total_loss",
            "synonymous_reference_targets.json",
            "every_n_train_steps",
            "resume_from_checkpoint",
            '"test_access_prohibited": True',
            "trainer.save_checkpoint(last_checkpoint)",
        ):
            self.assertIn(required, source)
        for forbidden in (
            'add_argument("--test-dataset',
            "test_dataset_path",
            "evaluate_codontransformer_test.py",
        ):
            self.assertNotIn(forbidden, source)

    def test_v2_colab_defaults_to_smoke_and_prohibits_test_access(self):
        path = (
            ROOT
            / "notebooks"
            / "codontransformer_finetune_csi_top10_v2_colab.ipynb"
        )
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for required in (
            "!nvidia-smi",
            'CODONTRANSFORMER_V2_RUN_MODE\", \"smoke',
            "smoke_test_cuda_csi_top10_v2.yaml",
            "finetune_cuda_csi_top10_v2.yaml",
            "finetune_codontransformer_v2.py",
            "--resume-from-checkpoint",
            "synonymous_jsd_loss",
            "parameter_changed",
            "translation_verified",
            "checkpoints/last.ckpt",
            "Binary stack:",
            "BINARY_STACK_SENTINEL",
            "Restart session",
        ):
            self.assertIn(required, source)
        for forbidden in (
            "test.jsonl",
            "TEST_PATH",
            "--test-dataset",
            "test_dataset_path",
            "finetune_csi_top10_hc_formal_v1/checkpoints",
            "pd.read_csv",
            "from scripts.validate_checkpoint_inference import load_checkpoint",
        ):
            self.assertNotIn(forbidden, source)

    def test_checkpoint_loader_ignores_non_model_v2_buffers(self):
        source = (
            ROOT / "scripts" / "validate_checkpoint_inference.py"
        ).read_text(encoding="utf-8")
        self.assertIn('if key.startswith("model.")', source)
        self.assertIn("return model_state or state_dict", source)

    def test_large_local_artifacts_are_ignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for required in (
            ".venv/",
            "upstream/",
            "models/pretrained/",
            "data/raw/*",
            "data/processed/*",
            "*.ckpt",
            "*.safetensors",
        ):
            self.assertIn(required, gitignore)

    def test_hygiene_check_includes_untracked_candidates(self):
        script = (ROOT / "scripts" / "check_repository_hygiene.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--others"', script)
        self.assertIn('"--exclude-standard"', script)

    def test_ambiguous_csi_cohort_name_is_not_used(self):
        forbidden = "high" + "_csi_hc"
        candidates = [ROOT / "README.md"]
        for directory in ("configs", "scripts", "tests", "notebooks"):
            candidates.extend(
                path
                for path in (ROOT / directory).rglob("*")
                if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".ipynb"}
            )
        for path in candidates:
            self.assertNotIn(forbidden, path.read_text(encoding="utf-8"), str(path))


if __name__ == "__main__":
    unittest.main()
