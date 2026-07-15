#!/usr/bin/env python3
"""Validation-selected v2 fine-tuning with synonymous-family calibration."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BigBirdForMaskedLM

try:
    from finetune_codontransformer import (
        JSONLinesDataset,
        MaskedTokenizerCollator,
        resolve_path,
        sha256,
    )
except ModuleNotFoundError:  # Imported as scripts.finetune_codontransformer_v2.
    from scripts.finetune_codontransformer import (
        JSONLinesDataset,
        MaskedTokenizerCollator,
        resolve_path,
        sha256,
    )


REQUIRED_PATH_KEYS = (
    "model_dir",
    "train_dataset_path",
    "validation_dataset_path",
    "reference_json",
    "output_dir",
)
REQUIRED_TRAINING_KEYS = (
    "accelerator",
    "devices",
    "precision",
    "batch_size",
    "validation_batch_size",
    "max_epochs",
    "num_workers",
    "accumulate_grad_batches",
    "learning_rate",
    "warmup_fraction",
    "mask_probability",
    "validation_mask_seed",
    "save_every_n_steps",
)
REQUIRED_OBJECTIVE_KEYS = (
    "mlm_weight",
    "synonymous_distribution_weight",
    "target_families",
    "csi_reference_weight",
    "cai_reference_weight",
    "epsilon",
)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("paths", "training", "objective"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing YAML mapping: {section}")
    missing_paths = [key for key in REQUIRED_PATH_KEYS if key not in config["paths"]]
    missing_training = [
        key for key in REQUIRED_TRAINING_KEYS if key not in config["training"]
    ]
    missing_objective = [
        key for key in REQUIRED_OBJECTIVE_KEYS if key not in config["objective"]
    ]
    if missing_paths or missing_training or missing_objective or "seed" not in config:
        raise ValueError(
            f"Missing keys: paths={missing_paths}, training={missing_training}, "
            f"objective={missing_objective}, seed={'seed' not in config}"
        )
    objective = config["objective"]
    families = list(objective["target_families"])
    if not families or len(families) != len(set(families)):
        raise ValueError("objective.target_families must be non-empty and unique")
    if float(objective["mlm_weight"]) <= 0:
        raise ValueError("objective.mlm_weight must be positive")
    if float(objective["synonymous_distribution_weight"]) < 0:
        raise ValueError("objective.synonymous_distribution_weight cannot be negative")
    reference_weight_sum = float(objective["csi_reference_weight"]) + float(
        objective["cai_reference_weight"]
    )
    if abs(reference_weight_sum - 1.0) > 1e-9:
        raise ValueError("CSI and CAI reference weights must sum to 1")
    if float(objective["epsilon"]) <= 0:
        raise ValueError("objective.epsilon must be positive")
    if int(config["training"]["save_every_n_steps"]) <= 0:
        raise ValueError("v2 requires positive optimizer-step recovery checkpointing")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    for key in REQUIRED_PATH_KEYS + ("resume_from_checkpoint",):
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
        value = getattr(args, key, None)
        if value is not None:
            config["training"][key] = value
    return config


def resolved_paths(config: dict[str, Any]) -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {
        key: resolve_path(config["paths"][key]) for key in REQUIRED_PATH_KEYS
    }
    resume = config["paths"].get("resume_from_checkpoint")
    paths["resume_from_checkpoint"] = resolve_path(resume) if resume else None
    for key in ("model_dir", "train_dataset_path", "validation_dataset_path", "reference_json"):
        path = paths[key]
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing required v2 input {key}: {path}")
    if not paths["model_dir"].is_dir():
        raise NotADirectoryError(paths["model_dir"])
    for key in ("train_dataset_path", "validation_dataset_path", "reference_json"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    resume_path = paths["resume_from_checkpoint"]
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    output_dir = paths["output_dir"]
    model_dir = paths["model_dir"]
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise ValueError("v2 output_dir must not overwrite the pretrained model")
    if str(config["training"]["accelerator"]) == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return paths


def validate_frozen_inputs(
    config: dict[str, Any], paths: dict[str, Path | None], counts: dict[str, int]
) -> None:
    expected_records = config.get("expected_records", {})
    expected_sha = config.get("expected_sha256", {})
    for split in ("train", "validation"):
        expected_count = int(expected_records[split])
        if counts[split] != expected_count:
            raise ValueError(
                f"{split} record count mismatch: expected {expected_count}, "
                f"found {counts[split]}"
            )
        actual_sha = sha256(paths[f"{split}_dataset_path"])
        if actual_sha != expected_sha[split]:
            raise ValueError(f"Unexpected {split} SHA256: {actual_sha}")
    reference_sha = sha256(paths["reference_json"])
    if reference_sha != expected_sha["reference"]:
        raise ValueError(f"Unexpected reference SHA256: {reference_sha}")


def normalized_family_distribution(
    reference: dict[str, Any], reference_name: str, codons: list[str]
) -> torch.Tensor:
    frequencies = reference[reference_name]["frequencies"]
    values = torch.tensor([float(frequencies[codon]) for codon in codons])
    if not torch.isfinite(values).all() or float(values.sum()) <= 0:
        raise ValueError(f"Invalid {reference_name} distribution for {codons}")
    return values / values.sum()


def build_synonymous_targets(
    reference: dict[str, Any], tokenizer, objective: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if int(reference["genetic_code"]) != 1:
        raise ValueError("v2 currently requires NCBI genetic code 1")
    csi_families = reference["csi_reference"]["synonymous_families"]
    cai_families = reference["cai_reference"]["synonymous_families"]
    csi_weight = float(objective["csi_reference_weight"])
    cai_weight = float(objective["cai_reference_weight"])
    targets: dict[str, dict[str, Any]] = {}
    for family in objective["target_families"]:
        codons = list(csi_families[family])
        if codons != list(cai_families[family]) or len(codons) < 2:
            raise ValueError(f"Invalid synonymous family definition for {family}")
        token_ids = [
            int(tokenizer.convert_tokens_to_ids(f"{family.lower()}_{codon.lower()}"))
            for codon in codons
        ]
        if len(set(token_ids)) != len(token_ids) or tokenizer.unk_token_id in token_ids:
            raise ValueError(f"Tokenizer cannot represent family {family}: {codons}")
        csi = normalized_family_distribution(reference, "csi_reference", codons)
        cai = normalized_family_distribution(reference, "cai_reference", codons)
        target = csi_weight * csi + cai_weight * cai
        target = target / target.sum()
        targets[family] = {
            "codons": codons,
            "token_ids": token_ids,
            "csi_distribution": csi.tolist(),
            "cai_distribution": cai.tolist(),
            "target_distribution": target.tolist(),
        }
    return targets


class V2TrainingHarness(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate: float,
        warmup_fraction: float,
        objective: dict[str, Any],
        family_targets: dict[str, dict[str, Any]],
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction
        self.mlm_weight = float(objective["mlm_weight"])
        self.distribution_weight = float(objective["synonymous_distribution_weight"])
        self.epsilon = float(objective["epsilon"])
        self.family_buffer_names: dict[str, tuple[str, str]] = {}
        for family, target in family_targets.items():
            ids_name = f"family_ids_{family}"
            target_name = f"family_target_{family}"
            self.register_buffer(ids_name, torch.tensor(target["token_ids"], dtype=torch.long))
            self.register_buffer(
                target_name,
                torch.tensor(target["target_distribution"], dtype=torch.float32),
            )
            self.family_buffer_names[family] = (ids_name, target_name)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=self.warmup_fraction,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def synonymous_distribution_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        losses: list[torch.Tensor] = []
        calibrated_positions = 0
        for ids_name, target_name in self.family_buffer_names.values():
            token_ids = getattr(self, ids_name)
            target = getattr(self, target_name).float()
            selected = torch.isin(labels, token_ids)
            count = int(selected.sum().item())
            if not count:
                continue
            family_logits = logits[selected][:, token_ids].float()
            predicted = torch.softmax(family_logits, dim=-1).mean(dim=0)
            predicted = predicted.clamp_min(self.epsilon)
            predicted = predicted / predicted.sum()
            target = target.clamp_min(self.epsilon)
            target = target / target.sum()
            midpoint = 0.5 * (predicted + target)
            jsd = 0.5 * torch.sum(predicted * (predicted.log() - midpoint.log()))
            jsd = jsd + 0.5 * torch.sum(target * (target.log() - midpoint.log()))
            # Roundoff can produce a tiny negative value at an exact match.
            losses.append(jsd.clamp_min(0.0))
            calibrated_positions += count
        if not losses:
            return logits.float().sum() * 0.0, 0, 0
        return torch.stack(losses).mean(), len(losses), calibrated_positions

    def losses(self, batch: dict[str, torch.Tensor]):
        self.model.bert.set_attention_type("block_sparse")
        output = self.model(**batch)
        mlm_loss = output.loss
        distribution_loss, active_families, calibrated_positions = (
            self.synonymous_distribution_loss(output.logits, batch["labels"])
        )
        total_loss = self.mlm_weight * mlm_loss + self.distribution_weight * distribution_loss
        if not all(torch.isfinite(value) for value in (mlm_loss, distribution_loss, total_loss)):
            raise FloatingPointError("Non-finite v2 training objective")
        return total_loss, mlm_loss, distribution_loss, active_families, calibrated_positions

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str):
        total, mlm, distribution, active_families, positions = self.losses(batch)
        batch_size = int(batch["input_ids"].shape[0])
        on_step = stage == "train"
        self.log(
            f"{stage}_total_loss",
            total,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_mlm_loss",
            mlm,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_synonymous_jsd_loss",
            distribution,
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_active_families",
            float(active_families),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}_calibrated_positions",
            float(positions),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if stage == "train":
            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )
        return total

    def training_step(self, batch, batch_index):
        del batch_index
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_index):
        del batch_index
        return self._shared_step(batch, "val")


def setup_logging(output_dir: Path, append: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("codontransformer_finetune_v2")
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


def save_metadata(
    output_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path | None],
    counts: dict[str, int],
    family_targets: dict[str, dict[str, Any]],
) -> None:
    resolved = json.loads(json.dumps(config))
    for key, path in paths.items():
        resolved["paths"][key] = str(path) if path is not None else None
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "synonymous_reference_targets.json").write_text(
        json.dumps(family_targets, indent=2) + "\n", encoding="utf-8"
    )
    model_weights = paths["model_dir"] / "model.safetensors"
    runtime = {
        "experiment_version": config["experiment_version"],
        "dataset_role": "train_and_validation_only",
        "test_access_prohibited": True,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "dataset_counts": counts,
        "sha256": {
            "train": sha256(paths["train_dataset_path"]),
            "validation": sha256(paths["validation_dataset_path"]),
            "reference": sha256(paths["reference_json"]),
            "pretrained_model": sha256(model_weights) if model_weights.is_file() else None,
        },
    }
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )


def train(config: dict[str, Any]) -> dict[str, Any]:
    paths = resolved_paths(config)
    training = config["training"]
    output_dir = paths["output_dir"]
    resume_checkpoint = paths["resume_from_checkpoint"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, append=resume_checkpoint is not None)
    seed = int(config["seed"])
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    logger.info("Loading official local tokenizer/model from %s", paths["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(paths["model_dir"], local_files_only=True)
    model = BigBirdForMaskedLM.from_pretrained(paths["model_dir"], local_files_only=True)
    train_dataset = JSONLinesDataset(paths["train_dataset_path"])
    validation_dataset = JSONLinesDataset(paths["validation_dataset_path"])
    counts = {"train": len(train_dataset), "validation": len(validation_dataset)}
    validate_frozen_inputs(config, paths, counts)
    reference = json.loads(paths["reference_json"].read_text(encoding="utf-8"))
    family_targets = build_synonymous_targets(reference, tokenizer, config["objective"])
    save_metadata(output_dir, config, paths, counts, family_targets)
    logger.info("Loaded frozen train=%d validation=%d; test remains out of scope", counts["train"], counts["validation"])
    logger.info("v2 target families: %s", ",".join(family_targets))
    if resume_checkpoint is not None:
        logger.info("Resuming the same v2 run from %s", resume_checkpoint)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=int(training["num_workers"]),
        persistent_workers=int(training["num_workers"]) > 0,
        collate_fn=MaskedTokenizerCollator(
            tokenizer, mask_probability=float(training["mask_probability"])
        ),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["validation_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        persistent_workers=int(training["num_workers"]) > 0,
        collate_fn=MaskedTokenizerCollator(
            tokenizer,
            mask_probability=float(training["mask_probability"]),
            deterministic_seed=int(training["validation_mask_seed"]),
        ),
    )

    checkpoint_dir = output_dir / "checkpoints"
    best_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-epoch{epoch:02d}-val_total_loss{val_total_loss:.6f}",
        auto_insert_metric_name=False,
        monitor="val_total_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )
    recovery_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="recovery-step{step:06d}",
        auto_insert_metric_name=False,
        save_top_k=0,
        save_last=True,
        every_n_train_steps=int(training["save_every_n_steps"]),
        every_n_epochs=None,
        save_on_train_epoch_end=False,
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_total_loss",
        mode="min",
        patience=int(training.get("early_stopping_patience", 2)),
        min_delta=float(training.get("early_stopping_min_delta", 0.0)),
        check_finite=True,
        strict=True,
    )
    csv_logger = CSVLogger(save_dir=output_dir / "logs", name="lightning")
    harness = V2TrainingHarness(
        model,
        learning_rate=float(training["learning_rate"]),
        warmup_fraction=float(training["warmup_fraction"]),
        objective=config["objective"],
        family_targets=family_targets,
    )
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=training["accelerator"],
        devices=training["devices"],
        strategy=training.get("strategy", "auto"),
        precision=training["precision"],
        max_epochs=int(training["max_epochs"]),
        limit_train_batches=training.get("limit_train_batches", 1.0),
        limit_val_batches=training.get("limit_val_batches", 1.0),
        deterministic=bool(training.get("deterministic", True)),
        callbacks=[best_callback, recovery_callback, early_stopping],
        logger=csv_logger,
        accumulate_grad_batches=int(training["accumulate_grad_batches"]),
        log_every_n_steps=int(training.get("log_every_n_steps", 25)),
        num_sanity_val_steps=int(training.get("num_sanity_val_steps", 0)),
        check_val_every_n_epoch=int(training.get("check_val_every_n_epoch", 1)),
        gradient_clip_val=float(training.get("gradient_clip_val", 0.0)),
        gradient_clip_algorithm=str(training.get("gradient_clip_algorithm", "norm")),
    )
    trainer.fit(
        harness,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint else None,
    )
    last_checkpoint = checkpoint_dir / "last.ckpt"
    trainer.save_checkpoint(last_checkpoint)
    best_checkpoint = Path(best_callback.best_model_path)
    if not best_checkpoint.is_file() or not last_checkpoint.is_file():
        raise RuntimeError("v2 training finished without best and last checkpoints")
    result = {
        "experiment_version": config["experiment_version"],
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "best_val_total_loss": float(best_callback.best_model_score),
        "global_step": int(trainer.global_step),
        "current_epoch": int(trainer.current_epoch),
        "early_stopped": bool(early_stopping.stopped_epoch > 0),
        "stopped_epoch": int(early_stopping.stopped_epoch),
        "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "test_access_prohibited": True,
    }
    (output_dir / "training_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Best v2 checkpoint: %s", best_checkpoint)
    logger.info("Final resumable checkpoint: %s", last_checkpoint)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--validation-dataset-path")
    parser.add_argument("--reference-json")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate the v2 YAML without loading data, a model, or a checkpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config.resolve()), args)
    if args.check_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    print(json.dumps(train(config), indent=2))


if __name__ == "__main__":
    main()
