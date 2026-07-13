import json
import unittest
from pathlib import Path


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
            "configs/smoke_test.yaml",
            "last.ckpt",
            "validate_checkpoint_inference.py",
            "translation_verified",
        ):
            self.assertIn(required, source)

    def test_scripts_have_no_mac_user_absolute_paths(self):
        marker = "/" + "Users" + "/"
        for path in (ROOT / "scripts").glob("*.py"):
            self.assertNotIn(marker, path.read_text(encoding="utf-8"), str(path))

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


if __name__ == "__main__":
    unittest.main()
