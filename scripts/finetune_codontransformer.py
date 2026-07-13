#!/usr/bin/env python3
"""Config-driven CodonTransformer fine-tuning for CUDA/Linux and Colab."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BigBirdForMaskedLM

from CodonTransformer.CodonUtils import MAX_LEN, TOKEN2MASK


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATH_KEYS = ("model_dir", "dataset_path", "output_dir")
REQUIRED_TRAINING_KEYS = (
    "accelerator",
    "devices",
    "precision",
    "batch_size",
    "max_epochs",
    "num_workers",
    "accumulate_grad_batches",
    "learning_rate",
    "warmup_fraction",
    "mask_probability",
    "save_every_n_steps",
)


def resolve_path(value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (PROJECT_ROOT / expanded).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("paths", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing YAML mapping: {section}")
    missing_paths = [key for key in REQUIRED_PATH_KEYS if key not in config["paths"]]
    missing_training = [
        key for key in REQUIRED_TRAINING_KEYS if key not in config["training"]
    ]
    if missing_paths or missing_training or "seed" not in config:
        raise ValueError(
            f"Missing configuration keys: paths={missing_paths}, "
            f"training={missing_training}, seed={'seed' not in config}"
        )
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    for key in REQUIRED_PATH_KEYS:
        value = getattr(args, key)
        if value is not None:
            config["paths"][key] = value
    for key in ("accelerator", "devices", "max_epochs", "limit_train_batches"):
        value = getattr(args, key)
        if value is not None:
            config["training"][key] = value
    return config


def open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


class JSONLinesDataset(Dataset):
    """In-memory JSONL dataset with deterministic map-style indexing."""

    def __init__(self, path: Path):
        self.records: list[dict[str, Any]] = []
        with open_jsonl(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record.get("codons"), str):
                    raise ValueError(f"Missing string codons at line {line_number}")
                if not isinstance(record.get("organism"), int):
                    raise ValueError(f"Missing integer organism at line {line_number}")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"No training records found in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class MaskedTokenizerCollator:
    def __init__(self, tokenizer, mask_probability: float):
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

    def __call__(self, examples: list[dict[str, Any]]):
        tokenized = self.tokenizer(
            [example["codons"] for example in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        sequence_length = tokenized["input_ids"].shape[-1]
        species = torch.tensor([[example["organism"]] for example in examples])
        tokenized["token_type_ids"] = species.repeat(1, sequence_length)

        inputs = tokenized["input_ids"]
        targets = inputs.clone()
        probability = torch.full(inputs.shape, self.mask_probability)
        probability[inputs < 5] = 0.0
        selected = torch.bernoulli(probability).bool()

        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        replacement_ids = [TOKEN2MASK[int(value)] for value in inputs[replaced].tolist()]
        if replacement_ids:
            inputs[replaced] = torch.tensor(replacement_ids, dtype=torch.long)

        randomized = (
            torch.bernoulli(torch.full(selected.shape, 0.1)).bool()
            & selected
            & ~replaced
        )
        random_ids = torch.randint(26, 90, probability.shape, dtype=torch.long)
        inputs[randomized] = random_ids[randomized]
        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)
        return tokenized


class TrainingHarness(pl.LightningModule):
    def __init__(self, model, learning_rate: float, warmup_fraction: float):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=max(total_steps, 1),
            pct_start=self.warmup_fraction,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_index):
        del batch_index
        self.model.bert.set_attention_type("block_sparse")
        output = self.model(**batch)
        self.log("loss", output.loss, on_step=True, prog_bar=True)
        return output.loss


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("codontransformer_finetune")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_runtime(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    model_dir = resolve_path(config["paths"]["model_dir"])
    dataset_path = resolve_path(config["paths"]["dataset_path"])
    output_dir = resolve_path(config["paths"]["output_dir"])
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise ValueError("output_dir must not be the pretrained model directory")
    accelerator = str(config["training"]["accelerator"])
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return model_dir, dataset_path, output_dir


def save_runtime_metadata(output_dir: Path, config: dict[str, Any]) -> None:
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    with (output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def train(config: dict[str, Any]) -> Path:
    model_dir, dataset_path, output_dir = validate_runtime(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    save_runtime_metadata(output_dir, config)
    training = config["training"]
    seed = int(config["seed"])
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    logger.info("Loading local tokenizer and model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BigBirdForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    dataset = JSONLinesDataset(dataset_path)
    logger.info("Loaded %d training records from %s", len(dataset), dataset_path)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        persistent_workers=int(training["num_workers"]) > 0,
        collate_fn=MaskedTokenizerCollator(
            tokenizer, mask_probability=float(training["mask_probability"])
        ),
    )

    checkpoint_dir = output_dir / "checkpoints"
    save_every_n_steps = int(training["save_every_n_steps"])
    checkpoint_options: dict[str, Any] = {
        "dirpath": checkpoint_dir,
        "filename": "step-{step:06d}",
        "save_last": True,
        "save_top_k": -1 if save_every_n_steps > 0 else 0,
    }
    if save_every_n_steps > 0:
        checkpoint_options["every_n_train_steps"] = save_every_n_steps
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **checkpoint_options,
    )
    csv_logger = CSVLogger(save_dir=output_dir / "logs", name="lightning")
    harness = TrainingHarness(
        model,
        learning_rate=float(training["learning_rate"]),
        warmup_fraction=float(training["warmup_fraction"]),
    )
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=training["accelerator"],
        devices=training["devices"],
        strategy=training.get("strategy", "auto"),
        precision=training["precision"],
        max_epochs=int(training["max_epochs"]),
        limit_train_batches=training.get("limit_train_batches", 1.0),
        deterministic=bool(training.get("deterministic", True)),
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        accumulate_grad_batches=int(training["accumulate_grad_batches"]),
        log_every_n_steps=int(training.get("log_every_n_steps", 10)),
        num_sanity_val_steps=0,
    )
    trainer.fit(harness, loader)
    last_checkpoint = Path(checkpoint_callback.last_model_path)
    if not last_checkpoint.is_file():
        raise RuntimeError("Training finished without a last checkpoint")
    logger.info("Last checkpoint: %s", last_checkpoint)
    return last_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", dest="model_dir")
    parser.add_argument("--dataset-path", dest="dataset_path")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--max-epochs", dest="max_epochs", type=int)
    parser.add_argument("--limit-train-batches", dest="limit_train_batches", type=int)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate YAML structure and overrides without loading data or training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config.resolve()), args)
    if args.check_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    checkpoint = train(config)
    print(checkpoint)


if __name__ == "__main__":
    main()
