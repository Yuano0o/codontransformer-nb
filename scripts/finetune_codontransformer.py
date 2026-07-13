#!/usr/bin/env python3
"""Config-driven CodonTransformer fine-tuning for CUDA/Linux and Colab."""

from __future__ import annotations

import argparse
import gzip
import hashlib
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
OPTIONAL_PATH_KEYS = (
    "validation_dataset_path",
    "test_dataset_path",
    "resume_from_checkpoint",
)
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
    for key in REQUIRED_PATH_KEYS + OPTIONAL_PATH_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            config["paths"][key] = value
    for key in (
        "accelerator",
        "devices",
        "max_epochs",
        "limit_train_batches",
        "limit_val_batches",
    ):
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
    def __init__(
        self,
        tokenizer,
        mask_probability: float,
        deterministic_seed: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.deterministic_seed = deterministic_seed

    def _generator(self, example: dict[str, Any], row: int) -> torch.Generator | None:
        if self.deterministic_seed is None:
            return None
        identifier = int(example.get("idx", row))
        seed = (self.deterministic_seed * 1_000_003 + identifier) % (2**63 - 1)
        return torch.Generator().manual_seed(seed)

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
        selected = torch.zeros_like(inputs, dtype=torch.bool)
        replaced = torch.zeros_like(inputs, dtype=torch.bool)
        randomized = torch.zeros_like(inputs, dtype=torch.bool)
        random_ids = torch.empty_like(inputs)
        for row, example in enumerate(examples):
            generator = self._generator(example, row)
            valid = inputs[row] >= 5
            selected_row = (
                torch.rand(inputs.shape[1], generator=generator)
                < self.mask_probability
            ) & valid
            if not selected_row.any():
                valid_positions = torch.where(valid)[0]
                if not len(valid_positions):
                    raise ValueError("No maskable codon tokens in training example")
                choice = torch.randint(
                    len(valid_positions), (1,), generator=generator
                ).item()
                selected_row[valid_positions[choice]] = True
            replaced_row = (
                torch.rand(inputs.shape[1], generator=generator) < 0.8
            ) & selected_row
            # Preserve the upstream fine-tuning collator's random-token rule.
            randomized_row = (
                torch.rand(inputs.shape[1], generator=generator) < 0.1
            ) & selected_row & ~replaced_row
            selected[row] = selected_row
            replaced[row] = replaced_row
            randomized[row] = randomized_row
            random_ids[row] = torch.randint(
                26, 90, (inputs.shape[1],), generator=generator
            )

        replacement_ids = [TOKEN2MASK[int(value)] for value in inputs[replaced].tolist()]
        if replacement_ids:
            inputs[replaced] = torch.tensor(replacement_ids, dtype=torch.long)
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
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite training loss")
        batch_size = int(batch["input_ids"].shape[0])
        self.log("loss", output.loss, on_step=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "train_loss",
            output.loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        return output.loss

    def validation_step(self, batch, batch_index):
        del batch_index
        self.model.bert.set_attention_type("block_sparse")
        output = self.model(**batch)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite validation loss")
        self.log(
            "val_loss",
            output.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(batch["input_ids"].shape[0]),
        )
        return output.loss


def setup_logging(output_dir: Path, append: bool = False) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("codontransformer_finetune")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(
            output_dir / "training.log",
            mode="a" if append else "w",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def optional_path(config: dict[str, Any], key: str) -> Path | None:
    value = config["paths"].get(key)
    return resolve_path(value) if value else None


def validate_runtime(config: dict[str, Any]) -> dict[str, Path | None]:
    model_dir = resolve_path(config["paths"]["model_dir"])
    dataset_path = resolve_path(config["paths"]["dataset_path"])
    validation_dataset_path = optional_path(config, "validation_dataset_path")
    test_dataset_path = optional_path(config, "test_dataset_path")
    output_dir = resolve_path(config["paths"]["output_dir"])
    resume_from_checkpoint = optional_path(config, "resume_from_checkpoint")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    for label, path in (
        ("Validation dataset", validation_dataset_path),
        ("Test dataset", test_dataset_path),
        ("Resume checkpoint", resume_from_checkpoint),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise ValueError("output_dir must not be the pretrained model directory")
    accelerator = str(config["training"]["accelerator"])
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return {
        "model_dir": model_dir,
        "dataset_path": dataset_path,
        "validation_dataset_path": validation_dataset_path,
        "test_dataset_path": test_dataset_path,
        "output_dir": output_dir,
        "resume_from_checkpoint": resume_from_checkpoint,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_runtime_metadata(
    output_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path | None],
    dataset_counts: dict[str, int],
) -> None:
    resolved_config = json.loads(json.dumps(config))
    for key, path in paths.items():
        if key == "dataset_path" or key in OPTIONAL_PATH_KEYS or key in REQUIRED_PATH_KEYS:
            resolved_config["paths"][key] = str(path) if path is not None else None
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)
    model_weights = paths["model_dir"] / "model.safetensors"
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "dataset_counts": dataset_counts,
        "dataset_sha256": {
            key: sha256(path)
            for key, path in (
                ("train", paths["dataset_path"]),
                ("validation", paths["validation_dataset_path"]),
                ("test", paths["test_dataset_path"]),
            )
            if path is not None
        },
        "pretrained_model_safetensors_sha256": (
            sha256(model_weights) if model_weights.is_file() else None
        ),
    }
    with (output_dir / "runtime.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def validate_dataset_counts(
    config: dict[str, Any], dataset_counts: dict[str, int]
) -> None:
    expected = config.get("expected_records", {})
    if not isinstance(expected, dict):
        raise ValueError("expected_records must be a YAML mapping")
    for split, expected_count in expected.items():
        if split not in dataset_counts:
            raise ValueError(f"Expected record count configured for missing {split} split")
        actual_count = dataset_counts[split]
        if actual_count != int(expected_count):
            raise ValueError(
                f"{split} record count mismatch: expected {expected_count}, "
                f"found {actual_count}"
            )


def train(config: dict[str, Any]) -> dict[str, Any]:
    paths = validate_runtime(config)
    model_dir = paths["model_dir"]
    dataset_path = paths["dataset_path"]
    validation_dataset_path = paths["validation_dataset_path"]
    output_dir = paths["output_dir"]
    resume_from_checkpoint = paths["resume_from_checkpoint"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, append=resume_from_checkpoint is not None)
    training = config["training"]
    seed = int(config["seed"])
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    logger.info("Loading local tokenizer and model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BigBirdForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    dataset = JSONLinesDataset(dataset_path)
    validation_dataset = (
        JSONLinesDataset(validation_dataset_path)
        if validation_dataset_path is not None
        else None
    )
    dataset_counts = {"train": len(dataset)}
    if validation_dataset is not None:
        dataset_counts["validation"] = len(validation_dataset)
    if paths["test_dataset_path"] is not None:
        dataset_counts["test"] = len(JSONLinesDataset(paths["test_dataset_path"]))
    validate_dataset_counts(config, dataset_counts)
    save_runtime_metadata(output_dir, config, paths, dataset_counts)
    logger.info("Loaded %d training records from %s", len(dataset), dataset_path)
    if validation_dataset is not None:
        logger.info(
            "Loaded %d validation records from %s",
            len(validation_dataset),
            validation_dataset_path,
        )
    if resume_from_checkpoint is not None:
        logger.info("Resuming from checkpoint: %s", resume_from_checkpoint)
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
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(training.get("validation_batch_size", training["batch_size"])),
            shuffle=False,
            num_workers=int(training["num_workers"]),
            persistent_workers=int(training["num_workers"]) > 0,
            collate_fn=MaskedTokenizerCollator(
                tokenizer,
                mask_probability=float(training["mask_probability"]),
                deterministic_seed=int(training.get("validation_mask_seed", seed + 1)),
            ),
        )

    checkpoint_dir = output_dir / "checkpoints"
    save_every_n_steps = int(training["save_every_n_steps"])
    if validation_loader is not None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-epoch{epoch:02d}-val_loss{val_loss:.6f}",
            auto_insert_metric_name=False,
            monitor="val_loss",
            mode="min",
            save_top_k=int(training.get("save_top_k", 1)),
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
    else:
        checkpoint_options: dict[str, Any] = {
            "dirpath": checkpoint_dir,
            "filename": "step-{step:06d}",
            "save_last": True,
            "save_top_k": -1 if save_every_n_steps > 0 else 0,
        }
        if save_every_n_steps > 0:
            checkpoint_options["every_n_train_steps"] = save_every_n_steps
        checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_options)
    callbacks: list[pl.Callback] = [checkpoint_callback]
    early_stopping = None
    if validation_loader is not None:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(training.get("early_stopping_patience", 2)),
            min_delta=float(training.get("early_stopping_min_delta", 0.0)),
            check_finite=True,
            strict=True,
        )
        callbacks.append(early_stopping)
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
        callbacks=callbacks,
        logger=csv_logger,
        accumulate_grad_batches=int(training["accumulate_grad_batches"]),
        log_every_n_steps=int(training.get("log_every_n_steps", 10)),
        num_sanity_val_steps=int(training.get("num_sanity_val_steps", 0)),
        limit_val_batches=training.get("limit_val_batches", 1.0),
        check_val_every_n_epoch=int(training.get("check_val_every_n_epoch", 1)),
        gradient_clip_val=float(training.get("gradient_clip_val", 0.0)),
        gradient_clip_algorithm=str(training.get("gradient_clip_algorithm", "norm")),
    )
    trainer.fit(
        harness,
        train_dataloaders=loader,
        val_dataloaders=validation_loader,
        ckpt_path=str(resume_from_checkpoint) if resume_from_checkpoint else None,
    )
    last_checkpoint = Path(checkpoint_callback.last_model_path)
    if not last_checkpoint.is_file():
        raise RuntimeError("Training finished without a last checkpoint")
    best_checkpoint = (
        Path(checkpoint_callback.best_model_path)
        if checkpoint_callback.best_model_path
        else last_checkpoint
    )
    if validation_loader is not None and not best_checkpoint.is_file():
        raise RuntimeError("Training finished without a best validation checkpoint")
    result = {
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "best_val_loss": (
            float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None
        ),
        "global_step": int(trainer.global_step),
        "current_epoch": int(trainer.current_epoch),
        "early_stopped": bool(early_stopping and early_stopping.stopped_epoch > 0),
        "stopped_epoch": int(early_stopping.stopped_epoch) if early_stopping else 0,
        "resumed_from_checkpoint": (
            str(resume_from_checkpoint) if resume_from_checkpoint else None
        ),
    }
    result_path = output_dir / "training_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info("Best checkpoint: %s", best_checkpoint)
    logger.info("Last checkpoint: %s", last_checkpoint)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", dest="model_dir")
    parser.add_argument("--dataset-path", dest="dataset_path")
    parser.add_argument("--validation-dataset-path", dest="validation_dataset_path")
    parser.add_argument("--test-dataset-path", dest="test_dataset_path")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint")
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--max-epochs", dest="max_epochs", type=int)
    parser.add_argument("--limit-train-batches", dest="limit_train_batches", type=int)
    parser.add_argument("--limit-val-batches", dest="limit_val_batches", type=int)
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
    result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
