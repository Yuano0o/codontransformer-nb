#!/usr/bin/env python3
"""Paired biological evaluation of baseline and fine-tuned CodonTransformer DNA."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import torch
from Bio.Data import CodonTable
from Bio.Seq import Seq
from scipy.stats import wilcoxon
from transformers import AutoTokenizer, BigBirdForMaskedLM

from CodonTransformer.CodonPrediction import predict_dna_sequence

try:
    from finetune_codontransformer import JSONLinesDataset, sha256
    from validate_checkpoint_inference import load_checkpoint, select_device
except ModuleNotFoundError:  # Imported as scripts.evaluate_biological_fidelity.
    from scripts.finetune_codontransformer import JSONLinesDataset, sha256
    from scripts.validate_checkpoint_inference import load_checkpoint, select_device


MODEL_LABELS = ("baseline", "finetuned")
HIGHER_IS_BETTER = {
    "translation_correct_rate": True,
    "sequence_valid_rate": True,
    "csi": True,
    "cai": True,
    "gc_absolute_error": False,
    "gc3_absolute_error": False,
    "codon_jsd_to_true": False,
    "rare_codon_fraction": False,
    "rare_codon_fraction_absolute_error": False,
    "codon_match_rate": True,
}


def parse_test_record(record: dict[str, Any]) -> dict[str, Any]:
    tokens = record["codons"].split()
    if len(tokens) < 2 or not tokens[-1].startswith("__"):
        raise ValueError(f"Malformed codon tokens for idx={record.get('idx')}")
    sense_tokens = tokens[:-1]
    protein = "".join(token.split("_", 1)[0] for token in sense_tokens)
    dna = "".join(token.rsplit("_", 1)[-1] for token in tokens).upper()
    return {
        "idx": int(record["idx"]),
        "protein": protein,
        "true_dna": dna,
        "organism": int(record["organism"]),
        "protein_length": len(protein),
    }


def load_test_records(path: Path, expected_records: int | None) -> list[dict[str, Any]]:
    dataset = JSONLinesDataset(path)
    records = [parse_test_record(record) for record in dataset.records]
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} test records, found {len(records)}")
    indices = [record["idx"] for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("Test idx values are not unique")
    return records


def load_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("genetic_code", "csi_reference", "cai_reference"):
        if key not in payload:
            raise ValueError(f"Reference JSON is missing {key}")
    for reference_name in ("csi_reference", "cai_reference"):
        reference = payload[reference_name]
        for key in ("frequencies", "relative_adaptiveness"):
            if key not in reference:
                raise ValueError(f"{reference_name} is missing {key}")
    return payload


def geometric_score(codons: Iterable[str], weights: dict[str, float], informative: set[str]) -> float:
    logs = []
    for codon in codons:
        if codon not in informative:
            continue
        weight = float(weights[codon])
        if weight <= 0:
            return 0.0
        logs.append(math.log(weight))
    return math.exp(sum(logs) / len(logs)) if logs else math.nan


def gc_fraction(sequence: str) -> float:
    return (sequence.count("G") + sequence.count("C")) / len(sequence) if sequence else math.nan


def codon_frequencies(codons: list[str], sense_codons: tuple[str, ...]) -> list[float]:
    counts = Counter(codons)
    total = sum(counts[codon] for codon in sense_codons)
    if not total:
        return [0.0] * len(sense_codons)
    return [counts[codon] / total for codon in sense_codons]


def jensen_shannon_distance(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Jensen-Shannon vectors must have equal lengths")
    total_first = sum(first)
    total_second = sum(second)
    if total_first <= 0 or total_second <= 0:
        return math.nan
    p = [value / total_first for value in first]
    q = [value / total_second for value in second]
    midpoint = [(left + right) / 2 for left, right in zip(p, q)]

    def kl(values: list[float], reference: list[float]) -> float:
        return sum(
            value * math.log2(value / target)
            for value, target in zip(values, reference)
            if value > 0
        )

    return math.sqrt((kl(p, midpoint) + kl(q, midpoint)) / 2)


def sequence_metrics(
    dna: str,
    protein: str,
    reference: dict[str, Any],
    rare_threshold: float,
) -> dict[str, Any]:
    dna = dna.upper()
    genetic_code = int(reference["genetic_code"])
    table = CodonTable.unambiguous_dna_by_id[genetic_code]
    stop_codons = set(table.stop_codons)
    sense_codons = tuple(sorted(table.forward_table))
    families: dict[str, list[str]] = {}
    for codon, amino_acid in table.forward_table.items():
        families.setdefault(amino_acid, []).append(codon)
    informative = {
        codon for codons in families.values() if len(codons) > 1 for codon in codons
    }
    atcg_only = bool(dna) and set(dna) <= set("ATCG")
    length_multiple_of_three = bool(dna) and len(dna) % 3 == 0
    codons = (
        [dna[index : index + 3] for index in range(0, len(dna), 3)]
        if length_multiple_of_three
        else []
    )
    start_atg = bool(codons) and codons[0] == "ATG"
    standard_stop = bool(codons) and codons[-1] in stop_codons
    internal_stop_absent = bool(codons) and not any(codon in stop_codons for codon in codons[:-1])
    length_matches_protein = len(dna) == (len(protein) + 1) * 3
    translated = ""
    if atcg_only and length_multiple_of_three:
        translated = str(Seq(dna).translate(table=genetic_code, to_stop=False))
    translation_correct = translated == protein + "*"
    sequence_valid = all(
        (
            atcg_only,
            length_multiple_of_three,
            length_matches_protein,
            start_atg,
            standard_stop,
            internal_stop_absent,
            translation_correct,
        )
    )
    sense = codons[:-1] if standard_stop else codons
    valid_sense = all(codon in table.forward_table for codon in sense)
    csi_weights = reference["csi_reference"]["relative_adaptiveness"]
    cai_weights = reference["cai_reference"]["relative_adaptiveness"]
    informative_codons = [codon for codon in sense if codon in informative]
    if valid_sense and sense:
        csi = geometric_score(sense, csi_weights, informative)
        cai = geometric_score(sense, cai_weights, informative)
        gc = gc_fraction(dna)
        gc3 = gc_fraction("".join(codon[2] for codon in sense))
        rare = (
            sum(float(csi_weights[codon]) < rare_threshold for codon in informative_codons)
            / len(informative_codons)
            if informative_codons
            else math.nan
        )
        frequencies = codon_frequencies(sense, sense_codons)
    else:
        csi = cai = gc = gc3 = rare = math.nan
        frequencies = [math.nan] * len(sense_codons)
    return {
        "atcg_only": atcg_only,
        "length_multiple_of_three": length_multiple_of_three,
        "length_matches_protein": length_matches_protein,
        "start_atg": start_atg,
        "standard_stop": standard_stop,
        "internal_stop_absent": internal_stop_absent,
        "translation_correct": translation_correct,
        "sequence_valid": sequence_valid,
        "translated_protein": translated,
        "csi": csi,
        "cai": cai,
        "gc": gc,
        "gc3": gc3,
        "rare_codon_fraction": rare,
        "sense_codons": sense,
        "codon_frequencies": frequencies,
        "terminal_stop_codon": codons[-1] if codons else "",
    }


def prediction_cache_path(output_dir: Path, label: str) -> Path:
    return output_dir / "prediction_cache" / f"{label}_predictions.jsonl"


def read_prediction_cache(path: Path, records: list[dict[str, Any]]) -> dict[int, str]:
    if not path.exists():
        return {}
    expected = {record["idx"] for record in records}
    predictions: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            if idx not in expected or idx in predictions:
                raise ValueError(f"Invalid cached idx={idx} at {path}:{line_number}")
            predictions[idx] = str(row["dna"]).upper()
    return predictions


def generate_predictions(
    label: str,
    model: BigBirdForMaskedLM,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    organism: str,
    output_dir: Path,
    flush_every: int,
    logger: logging.Logger,
) -> dict[int, str]:
    cache_path = prediction_cache_path(output_dir, label)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = read_prediction_cache(cache_path, records)
    if len(predictions) == len(records):
        logger.info("Using complete %s prediction cache", label)
        return predictions
    logger.info("Generating %s DNA; resuming at %d/%d", label, len(predictions), len(records))
    model.to(device).eval()
    with cache_path.open("a", encoding="utf-8") as handle:
        generated_since_flush = 0
        for position, record in enumerate(records, start=1):
            idx = record["idx"]
            if idx in predictions:
                continue
            prediction = predict_dna_sequence(
                protein=record["protein"],
                organism=organism,
                device=device,
                tokenizer=tokenizer,
                model=model,
                attention_type="original_full",
                deterministic=True,
                match_protein=True,
            )
            dna = prediction.predicted_dna.upper()
            predictions[idx] = dna
            handle.write(json.dumps({"idx": idx, "dna": dna}, separators=(",", ":")) + "\n")
            generated_since_flush += 1
            if generated_since_flush >= flush_every:
                handle.flush()
                generated_since_flush = 0
                logger.info("Generated %s %d/%d", label, position, len(records))
        handle.flush()
    if len(predictions) != len(records):
        raise RuntimeError(f"Incomplete {label} prediction cache")
    return predictions


def tercile_boundaries(lengths: list[int]) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(lengths, [1 / 3, 2 / 3], method="linear"))


def length_bin(length: int, boundaries: tuple[float, float]) -> str:
    if length <= boundaries[0]:
        return "short"
    if length <= boundaries[1]:
        return "medium"
    return "long"


def finite_mean(values: Iterable[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return mean(valid) if valid else math.nan


def summarize_rows(rows: list[dict[str, Any]], label: str, reference_frequencies: list[float]) -> dict[str, Any]:
    count = len(rows)
    metrics = {
        "records": count,
        "atcg_only_rate": mean(float(row[f"{label}_atcg_only"]) for row in rows),
        "length_multiple_of_three_rate": mean(
            float(row[f"{label}_length_multiple_of_three"]) for row in rows
        ),
        "length_matches_protein_rate": mean(
            float(row[f"{label}_length_matches_protein"]) for row in rows
        ),
        "start_atg_rate": mean(float(row[f"{label}_start_atg"]) for row in rows),
        "standard_stop_rate": mean(float(row[f"{label}_standard_stop"]) for row in rows),
        "internal_stop_absent_rate": mean(
            float(row[f"{label}_internal_stop_absent"]) for row in rows
        ),
        "translation_correct_rate": mean(float(row[f"{label}_translation_correct"]) for row in rows),
        "sequence_valid_rate": mean(float(row[f"{label}_sequence_valid"]) for row in rows),
        "csi": finite_mean(row[f"{label}_csi"] for row in rows),
        "cai": finite_mean(row[f"{label}_cai"] for row in rows),
        "gc": finite_mean(row[f"{label}_gc"] for row in rows),
        "gc3": finite_mean(row[f"{label}_gc3"] for row in rows),
        "gc_absolute_error": finite_mean(row[f"{label}_gc_absolute_error"] for row in rows),
        "gc3_absolute_error": finite_mean(row[f"{label}_gc3_absolute_error"] for row in rows),
        "rare_codon_fraction": finite_mean(row[f"{label}_rare_codon_fraction"] for row in rows),
        "rare_codon_fraction_absolute_error": finite_mean(
            row[f"{label}_rare_codon_fraction_absolute_error"] for row in rows
        ),
        "codon_jsd_to_true": finite_mean(row[f"{label}_codon_jsd_to_true"] for row in rows),
        "codon_match_rate": finite_mean(row[f"{label}_codon_match_rate"] for row in rows),
        "codon_match_rate_micro": (
            sum(row[f"{label}_codon_matches"] for row in rows)
            / sum(row["protein_length"] for row in rows)
        ),
    }
    aggregate_counts: Counter[str] = Counter()
    for row in rows:
        aggregate_counts.update(row[f"_{label}_sense_codons"])
    # Reference vectors are stored in canonical codon order separately in each row.
    if rows:
        canonical = rows[0]["_sense_codon_order"]
        aggregate = [aggregate_counts[codon] for codon in canonical]
        metrics["aggregate_codon_jsd_to_n_benthamiana_reference"] = jensen_shannon_distance(
            aggregate, reference_frequencies
        )
    else:
        metrics["aggregate_codon_jsd_to_n_benthamiana_reference"] = math.nan
    return metrics


def summarize_true_rows(
    rows: list[dict[str, Any]], reference_frequencies: list[float]
) -> dict[str, Any]:
    aggregate_counts: Counter[str] = Counter()
    for row in rows:
        aggregate_counts.update(row["_true_sense_codons"])
    canonical = rows[0]["_sense_codon_order"]
    aggregate = [aggregate_counts[codon] for codon in canonical]
    return {
        "records": len(rows),
        "csi": finite_mean(row["true_csi"] for row in rows),
        "cai": finite_mean(row["true_cai"] for row in rows),
        "gc": finite_mean(row["true_gc"] for row in rows),
        "gc3": finite_mean(row["true_gc3"] for row in rows),
        "rare_codon_fraction": finite_mean(
            row["true_rare_codon_fraction"] for row in rows
        ),
        "aggregate_codon_jsd_to_n_benthamiana_reference": jensen_shannon_distance(
            aggregate, reference_frequencies
        ),
    }


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(values)
    estimates = [mean(values[rng.randrange(size)] for _ in range(size)) for _ in range(samples)]
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


def paired_test(
    rows: list[dict[str, Any]], metric: str, higher_is_better: bool, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    baseline = [float(row[f"baseline_{metric}"]) for row in rows]
    finetuned = [float(row[f"finetuned_{metric}"]) for row in rows]
    pairs = [(left, right) for left, right in zip(baseline, finetuned) if math.isfinite(left) and math.isfinite(right)]
    baseline = [pair[0] for pair in pairs]
    finetuned = [pair[1] for pair in pairs]
    raw_deltas = [right - left for left, right in pairs]
    improvements = raw_deltas if higher_is_better else [-value for value in raw_deltas]
    tolerance = 1e-12
    wins = sum(value > tolerance for value in improvements)
    ties = sum(abs(value) <= tolerance for value in improvements)
    losses = len(improvements) - wins - ties
    if all(abs(value) <= tolerance for value in raw_deltas):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(finetuned, baseline, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    lower, upper = bootstrap_mean_ci(improvements, bootstrap_samples, seed)
    return {
        "paired_records": len(improvements),
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "baseline_mean": mean(baseline),
        "finetuned_mean": mean(finetuned),
        "finetuned_minus_baseline": mean(raw_deltas),
        "mean_improvement": mean(improvements),
        "bootstrap_95_ci_mean_improvement": [lower, upper],
        "wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses},
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value_two_sided": p_value,
    }


def benjamini_hochberg(tests: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(tests, key=lambda name: tests[name]["wilcoxon_p_value_two_sided"])
    total = len(ordered)
    adjusted = [0.0] * total
    running = 1.0
    for reverse_index in range(total - 1, -1, -1):
        name = ordered[reverse_index]
        rank = reverse_index + 1
        value = min(1.0, tests[name]["wilcoxon_p_value_two_sided"] * total / rank)
        running = min(running, value)
        adjusted[reverse_index] = running
    for name, q_value in zip(ordered, adjusted):
        tests[name]["wilcoxon_q_value_bh"] = q_value


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public = [public_row(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    dataset_role = str(summary.get("dataset_role", "test"))
    lines = [
        "# csi_top10_hc biological fidelity evaluation",
        "",
        f"{dataset_role.capitalize()} records: {summary['records']}",
        "",
        "| Metric | Baseline | Fine-tuned | Mean improvement | 95% bootstrap CI | Wilcoxon BH q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, result in summary["paired_statistics"].items():
        interval = result["bootstrap_95_ci_mean_improvement"]
        lines.append(
            f"| {metric} | {result['baseline_mean']:.6f} | {result['finetuned_mean']:.6f} | "
            f"{result['mean_improvement']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{result['wilcoxon_q_value_bh']:.6g} |"
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            f"Translation requirement met: {decision['translation_requirement_met']}",
            f"Stable target-feature improvements: {', '.join(decision['stable_target_feature_improvements']) or 'none'}",
            f"Supports improved N. benthamiana codon adaptation: {decision['supports_claim']}",
            "",
            "Positive mean improvement always means the fine-tuned model is better after accounting for metric direction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dataset", "--dataset", dest="test_dataset", type=Path, required=True)
    parser.add_argument("--dataset-role", choices=("test", "validation"), default="test")
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--organism", default="Nicotiana tabacum")
    parser.add_argument("--expected-records", type=int, default=594)
    parser.add_argument("--expected-test-sha256")
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-reference-sha256")
    parser.add_argument("--expected-pretrained-sha256")
    parser.add_argument("--length-short-max", type=float)
    parser.add_argument("--length-medium-max", type=float)
    parser.add_argument("--rare-threshold", type=float, default=0.3)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--flush-every", type=int, default=25)
    args = parser.parse_args()

    fixed_boundaries_requested = (
        args.length_short_max is not None or args.length_medium_max is not None
    )
    if fixed_boundaries_requested and (
        args.length_short_max is None or args.length_medium_max is None
    ):
        raise ValueError("Both --length-short-max and --length-medium-max are required")
    if fixed_boundaries_requested and args.length_short_max >= args.length_medium_max:
        raise ValueError("length-short-max must be smaller than length-medium-max")
    if args.dataset_role == "validation" and not fixed_boundaries_requested:
        raise ValueError("Validation evaluation requires frozen explicit length boundaries")
    if (
        args.expected_test_sha256 is not None
        and args.expected_dataset_sha256 is not None
        and args.expected_test_sha256 != args.expected_dataset_sha256
    ):
        raise ValueError("Conflicting expected dataset SHA256 values")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("codontransformer_biological_evaluation")
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    test_dataset = args.test_dataset.expanduser().resolve()
    reference_json = args.reference_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path in (
        ("Model directory", model_dir),
        ("Checkpoint", checkpoint),
        ("Test dataset", test_dataset),
        ("Codon reference", reference_json),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise ValueError("output_dir must not be inside the pretrained model directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_sha256 = sha256(test_dataset)
    reference_sha256 = sha256(reference_json)
    pretrained_weights = model_dir / "model.safetensors"
    if not pretrained_weights.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_weights}")
    pretrained_sha256 = sha256(pretrained_weights)
    expected_dataset_sha256 = (
        args.expected_dataset_sha256 or args.expected_test_sha256
    )
    for label, expected, actual in (
        (f"{args.dataset_role} dataset", expected_dataset_sha256, test_sha256),
        ("codon reference", args.expected_reference_sha256, reference_sha256),
        ("pretrained weights", args.expected_pretrained_sha256, pretrained_sha256),
    ):
        if expected is not None and expected != actual:
            raise ValueError(f"Unexpected {label} SHA256: {actual}; expected {expected}")
    device = select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = load_test_records(test_dataset, args.expected_records)
    reference = load_reference(reference_json)
    lengths = [record["protein_length"] for record in records]
    boundaries = (
        (float(args.length_short_max), float(args.length_medium_max))
        if fixed_boundaries_requested
        else tercile_boundaries(lengths)
    )
    dataset_hash_key = f"{args.dataset_role}_dataset_sha256"
    manifest = {
        dataset_hash_key: test_sha256,
        "reference_json_sha256": reference_sha256,
        "pretrained_model_safetensors_sha256": pretrained_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "model_dir": str(model_dir),
        "records": len(records),
        "organism": args.organism,
        "deterministic": True,
        "synonymous_decoding_constrained": True,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    if args.dataset_role != "test" or fixed_boundaries_requested:
        manifest.update(
            {
                "dataset_role": args.dataset_role,
                "length_boundaries": {
                    "source": (
                        "frozen_refined_v2_test_boundaries"
                        if fixed_boundaries_requested
                        else "dataset_terciles"
                    ),
                    "short_max": boundaries[0],
                    "medium_max": boundaries[1],
                },
            }
        )
    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing prediction cache belongs to a different evaluation manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    predictions: dict[str, dict[int, str]] = {}
    for label in MODEL_LABELS:
        cached = read_prediction_cache(prediction_cache_path(output_dir, label), records)
        if len(cached) == len(records):
            predictions[label] = cached
            continue
        model = BigBirdForMaskedLM.from_pretrained(model_dir, local_files_only=True)
        if label == "finetuned":
            model.load_state_dict(load_checkpoint(checkpoint), strict=True)
        predictions[label] = generate_predictions(
            label,
            model,
            tokenizer,
            records,
            device,
            args.organism,
            output_dir,
            args.flush_every,
            logger,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = CodonTable.unambiguous_dna_by_id[int(reference["genetic_code"])]
    sense_codon_order = tuple(sorted(table.forward_table))
    reference_frequencies = [
        float(reference["csi_reference"]["frequencies"][codon]) for codon in sense_codon_order
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        true_metrics = sequence_metrics(record["true_dna"], record["protein"], reference, args.rare_threshold)
        row: dict[str, Any] = {
            "idx": record["idx"],
            "protein": record["protein"],
            "protein_length": record["protein_length"],
            "length_bin": length_bin(record["protein_length"], boundaries),
            "true_dna": record["true_dna"],
            "true_csi": true_metrics["csi"],
            "true_cai": true_metrics["cai"],
            "true_gc": true_metrics["gc"],
            "true_gc3": true_metrics["gc3"],
            "true_rare_codon_fraction": true_metrics["rare_codon_fraction"],
            "_sense_codon_order": sense_codon_order,
            "_true_sense_codons": true_metrics["sense_codons"],
        }
        true_frequencies = true_metrics["codon_frequencies"]
        for label in MODEL_LABELS:
            dna = predictions[label][record["idx"]]
            metrics = sequence_metrics(dna, record["protein"], reference, args.rare_threshold)
            row[f"{label}_dna"] = dna
            for validity_key in (
                "atcg_only",
                "length_multiple_of_three",
                "length_matches_protein",
                "start_atg",
                "standard_stop",
                "internal_stop_absent",
                "translation_correct",
                "sequence_valid",
            ):
                row[f"{label}_{validity_key}"] = metrics[validity_key]
            for metric in ("csi", "cai", "gc", "gc3", "rare_codon_fraction"):
                row[f"{label}_{metric}"] = metrics[metric]
            row[f"{label}_gc_absolute_error"] = abs(metrics["gc"] - true_metrics["gc"])
            row[f"{label}_gc3_absolute_error"] = abs(metrics["gc3"] - true_metrics["gc3"])
            row[f"{label}_rare_codon_fraction_absolute_error"] = abs(
                metrics["rare_codon_fraction"] - true_metrics["rare_codon_fraction"]
            )
            row[f"{label}_codon_jsd_to_true"] = jensen_shannon_distance(
                metrics["codon_frequencies"], true_frequencies
            )
            matches = sum(
                predicted == actual
                for predicted, actual in zip(metrics["sense_codons"], true_metrics["sense_codons"])
            )
            row[f"{label}_codon_matches"] = matches
            row[f"{label}_codon_match_rate"] = matches / record["protein_length"]
            row[f"{label}_stop_codon_match"] = (
                metrics["terminal_stop_codon"] == true_metrics["terminal_stop_codon"]
            )
            row[f"_{label}_sense_codons"] = metrics["sense_codons"]
        rows.append(row)

    for label in MODEL_LABELS:
        for row in rows:
            row[f"{label}_translation_correct_rate"] = float(row[f"{label}_translation_correct"])
            row[f"{label}_sequence_valid_rate"] = float(row[f"{label}_sequence_valid"])
    overall = {
        label: summarize_rows(rows, label, reference_frequencies) for label in MODEL_LABELS
    }
    by_length: dict[str, dict[str, Any]] = {}
    for bin_name in ("short", "medium", "long"):
        subset = [row for row in rows if row["length_bin"] == bin_name]
        by_length[bin_name] = {
            "real_cds": summarize_true_rows(subset, reference_frequencies),
            **{
                label: summarize_rows(subset, label, reference_frequencies)
                for label in MODEL_LABELS
            },
        }
    paired_statistics = {
        metric: paired_test(
            rows,
            metric,
            higher_is_better,
            args.bootstrap_samples,
            args.seed + position,
        )
        for position, (metric, higher_is_better) in enumerate(HIGHER_IS_BETTER.items())
    }
    benjamini_hochberg(paired_statistics)
    target_metrics = ("csi", "cai", "gc3_absolute_error", "codon_jsd_to_true")
    stable_improvements = [
        metric
        for metric in target_metrics
        if paired_statistics[metric]["bootstrap_95_ci_mean_improvement"][0] > 0
        and paired_statistics[metric]["wilcoxon_q_value_bh"] < 0.05
    ]
    translation_requirement_met = overall["finetuned"]["translation_correct_rate"] >= 0.999
    summary = {
        **manifest,
        "dataset_role": args.dataset_role,
        "length_bins": {
            "method": (
                "frozen refined-v2 test-set protein-length boundaries"
                if fixed_boundaries_requested
                else "test-set protein-length terciles"
            ),
            "short_max": boundaries[0],
            "medium_max": boundaries[1],
            "counts": Counter(row["length_bin"] for row in rows),
        },
        "real_cds": summarize_true_rows(rows, reference_frequencies),
        "models": overall,
        "by_length": by_length,
        "paired_statistics": paired_statistics,
        "decision": {
            "translation_requirement": ">= 0.999",
            "translation_requirement_met": translation_requirement_met,
            "stable_improvement_rule": "bootstrap 95% CI lower bound > 0 and BH-adjusted Wilcoxon q < 0.05",
            "stable_target_feature_improvements": stable_improvements,
            "supports_claim": translation_requirement_met and bool(stable_improvements),
        },
    }
    write_csv(output_dir / "per_sequence_metrics.csv", rows)
    (output_dir / "biological_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "biological_evaluation_report.md", summary)
    logger.info("Wrote biological evaluation to %s", output_dir)


if __name__ == "__main__":
    main()
