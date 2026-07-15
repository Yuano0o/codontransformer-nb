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
