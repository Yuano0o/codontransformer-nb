#!/usr/bin/env python3
"""Run an offline CodonTransformer baseline and save a machine-readable result."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from CodonTransformer.CodonPrediction import predict_dna_sequence
from transformers import AutoTokenizer, BigBirdForMaskedLM


ROOT = Path(__file__).resolve().parents[1]


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", default="MALWMRLLPLLALLALWGPDPAAA")
    parser.add_argument("--organism", default="Nicotiana tabacum")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/pretrained")
    parser.add_argument("--output", type=Path, default=ROOT / "results/nicotiana_tabacum_baseline.json")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = BigBirdForMaskedLM.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()
    with torch.inference_mode():
        prediction = predict_dna_sequence(
            protein=args.protein,
            organism=args.organism,
            device=device,
            tokenizer=tokenizer,
            model=model,
            attention_type="original_full",
            deterministic=True,
        )

    payload = {"device": str(device), "model_dir": str(args.model_dir.resolve()), **asdict(prediction)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
