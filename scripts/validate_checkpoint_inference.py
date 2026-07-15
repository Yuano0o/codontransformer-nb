#!/usr/bin/env python3
"""Reload a Lightning checkpoint and verify one CodonTransformer prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from Bio.Seq import Seq
from CodonTransformer.CodonPrediction import predict_dna_sequence
from transformers import AutoTokenizer, BigBirdForMaskedLM


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    # Lightning harnesses may contain objective buffers in addition to model.*.
    # Older raw model state dictionaries have no model. prefix and remain valid.
    return model_state or state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protein", default="MALWMRLLPLLALLALWGPDPAAA")
    parser.add_argument("--organism", default="Nicotiana tabacum")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BigBirdForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    model.load_state_dict(load_checkpoint(checkpoint), strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        prediction = predict_dna_sequence(
            protein=args.protein,
            organism=args.organism,
            device=device,
            tokenizer=tokenizer,
            model=model,
            attention_type="original_full",
            deterministic=True,
            match_protein=True,
        )

    translated = str(Seq(prediction.predicted_dna).translate(to_stop=True))
    expected = args.protein.rstrip("*_")
    verified = translated == expected
    payload = {
        "device": str(device),
        "checkpoint": str(checkpoint),
        "organism": prediction.organism,
        "protein": prediction.protein,
        "predicted_dna": prediction.predicted_dna,
        "translated_protein": translated,
        "translation_verified": verified,
    }
    if not verified:
        raise RuntimeError(f"Translation mismatch: {translated} != {expected}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
