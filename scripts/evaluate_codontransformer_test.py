#!/usr/bin/env python3
"""Compare pretrained and fine-tuned CodonTransformer models on a fixed test set."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BigBirdForMaskedLM

from finetune_codontransformer import (
    JSONLinesDataset,
    MaskedTokenizerCollator,
    sha256,
)
from validate_checkpoint_inference import load_checkpoint, select_device


def evaluate_model(
    model: BigBirdForMaskedLM,
    tokenizer,
    dataset: JSONLinesDataset,
    device: torch.device,
    batch_size: int,
    mask_probability: float,
    mask_seed: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=MaskedTokenizerCollator(
            tokenizer,
            mask_probability=mask_probability,
            deterministic_seed=mask_seed,
        ),
    )
    model.to(device).eval()
    total_nll = 0.0
    masked_tokens = 0
    top1_correct = 0
    top3_correct = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            model.bert.set_attention_type("block_sparse")
            logits = model(**batch, return_dict=True).logits
            selected = labels != -100
            count = int(selected.sum())
            if not count:
                raise RuntimeError(f"No masked tokens in test batch {batch_index}")
            selected_logits = logits[selected].float()
            selected_labels = labels[selected]
            total_nll += float(
                F.cross_entropy(
                    selected_logits,
                    selected_labels,
                    reduction="sum",
                )
            )
            masked_tokens += count
            top1_correct += int(
                (selected_logits.argmax(dim=-1) == selected_labels).sum()
            )
            top3 = selected_logits.topk(k=3, dim=-1).indices
            top3_correct += int((top3 == selected_labels.unsqueeze(-1)).any(dim=-1).sum())
            if batch_index % 100 == 0 or batch_index == len(loader):
                logger.info("Evaluated %d/%d batches", batch_index, len(loader))
    mean_nll = total_nll / masked_tokens
    metrics = {
        "records": len(dataset),
        "masked_tokens": masked_tokens,
        "mean_masked_token_nll": mean_nll,
        "masked_token_perplexity": math.exp(mean_nll),
        "masked_token_top1_accuracy": top1_correct / masked_tokens,
        "masked_token_top3_accuracy": top3_correct / masked_tokens,
    }
    if not all(math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
        raise FloatingPointError("Non-finite test metric")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--mask-seed", type=int, default=2025)
    parser.add_argument("--expected-records", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("codontransformer_test_comparison")
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    test_dataset = args.test_dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for label, path in (
        ("Model directory", model_dir),
        ("Checkpoint", checkpoint),
        ("Test dataset", test_dataset),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    device = select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    dataset = JSONLinesDataset(test_dataset)
    if args.expected_records is not None and len(dataset) != args.expected_records:
        raise ValueError(
            f"Expected {args.expected_records} test records, found {len(dataset)}"
        )

    logger.info("Evaluating official pretrained baseline on %d records", len(dataset))
    baseline_model = BigBirdForMaskedLM.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    baseline = evaluate_model(
        baseline_model,
        tokenizer,
        dataset,
        device,
        args.batch_size,
        args.mask_probability,
        args.mask_seed,
        logger,
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Evaluating fine-tuned checkpoint: %s", checkpoint)
    finetuned_model = BigBirdForMaskedLM.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    finetuned_model.load_state_dict(load_checkpoint(checkpoint), strict=True)
    finetuned = evaluate_model(
        finetuned_model,
        tokenizer,
        dataset,
        device,
        args.batch_size,
        args.mask_probability,
        args.mask_seed,
        logger,
    )
    if baseline["masked_tokens"] != finetuned["masked_tokens"]:
        raise RuntimeError("Baseline and fine-tuned evaluations used different masks")

    report = {
        "device": str(device),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "test_dataset": str(test_dataset),
        "test_dataset_sha256": sha256(test_dataset),
        "mask_probability": args.mask_probability,
        "mask_seed": args.mask_seed,
        "baseline": baseline,
        "finetuned": finetuned,
        "finetuned_minus_baseline": {
            "mean_masked_token_nll": (
                finetuned["mean_masked_token_nll"]
                - baseline["mean_masked_token_nll"]
            ),
            "masked_token_perplexity": (
                finetuned["masked_token_perplexity"]
                - baseline["masked_token_perplexity"]
            ),
            "masked_token_top1_accuracy": (
                finetuned["masked_token_top1_accuracy"]
                - baseline["masked_token_top1_accuracy"]
            ),
            "masked_token_top3_accuracy": (
                finetuned["masked_token_top3_accuracy"]
                - baseline["masked_token_top3_accuracy"]
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
