#!/usr/bin/env python3
"""Diagnose synonymous-family probability concentration on validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
import yaml
from Bio.Seq import Seq
from scipy.stats import wilcoxon
from transformers import AutoTokenizer, BigBirdForMaskedLM

from CodonTransformer.CodonData import get_merged_seq
from CodonTransformer.CodonPrediction import tokenize, validate_and_convert_organism
from CodonTransformer.CodonUtils import AMINO_ACID_TO_INDEX, INDEX2TOKEN

try:
    from evaluate_biological_fidelity import (
        length_bin,
        load_reference,
        load_test_records,
        sequence_metrics,
        sha256,
    )
    from refine_biological_evaluation import (
        adjust_bh,
        bootstrap_ci,
        family_counts,
        family_definitions,
        js_distance,
        probability_l1,
        reference_family_counts,
        weighted_family_distances,
    )
    from validate_checkpoint_inference import load_checkpoint, select_device
except ModuleNotFoundError:  # Imported as scripts.diagnose_validation_decoding.
    from scripts.evaluate_biological_fidelity import (
        length_bin,
        load_reference,
        load_test_records,
        sequence_metrics,
        sha256,
    )
    from scripts.refine_biological_evaluation import (
        adjust_bh,
        bootstrap_ci,
        family_counts,
        family_definitions,
        js_distance,
        probability_l1,
        reference_family_counts,
        weighted_family_distances,
    )
    from scripts.validate_checkpoint_inference import load_checkpoint, select_device


MODEL_LABELS = ("baseline", "finetuned")
COMPARISON_METRICS = {
    "csi": True,
    "cai": True,
    "gc_absolute_error": False,
    "gc3_absolute_error": False,
    "target_family_jsd_to_true": False,
    "target_family_jsd_to_reference": False,
    "codon_match_rate": True,
}


def stable_seed(base_seed: int, model_label: str, strategy: str, idx: int) -> int:
    payload = f"{base_seed}|{model_label}|{strategy}|{idx}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


def normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    return entropy(probabilities) / math.log2(len(probabilities))


def normalized_counter(counter: Counter[str], codons: tuple[str, ...]) -> list[float]:
    total = sum(counter[codon] for codon in codons)
    if not total:
        return [0.0] * len(codons)
    return [counter[codon] / total for codon in codons]


def sample_allowed_token(
    logits: torch.Tensor,
    allowed_indices: list[int],
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> int:
    selected = logits[allowed_indices].float() / temperature
    probabilities = torch.softmax(selected, dim=-1)
    sorted_probabilities, order = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative - sorted_probabilities > top_p
    sorted_probabilities[remove] = 0.0
    sorted_probabilities /= sorted_probabilities.sum()
    sampled_rank = int(
        torch.multinomial(sorted_probabilities, 1, generator=generator).item()
    )
    return int(allowed_indices[int(order[sampled_rank])])


def decode_record(
    logits: torch.Tensor,
    merged_tokens: list[str],
    strategies: dict[str, dict[str, Any]],
    base_seed: int,
    model_label: str,
    idx: int,
    target_families: set[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if logits.shape[0] != len(merged_tokens):
        raise ValueError(
            f"Model positions {logits.shape[0]} do not match merged tokens "
            f"{len(merged_tokens)} for idx={idx}"
        )
    decoded: dict[str, list[int]] = {name: [] for name in strategies}
    diagnostics: list[dict[str, Any]] = []
    generators = {
        name: torch.Generator(device="cpu").manual_seed(
            stable_seed(base_seed, model_label, name, idx)
        )
        for name in strategies
    }
    for position, token in enumerate(merged_tokens):
        amino_acid = token[0]
        allowed = [int(value) for value in AMINO_ACID_TO_INDEX[amino_acid]]
        family_logits = logits[position, allowed].float()
        family_probabilities = torch.softmax(family_logits, dim=-1).tolist()
        codons = [INDEX2TOKEN[index][-3:].upper() for index in allowed]
        if amino_acid in target_families:
            argmax_rank = int(torch.argmax(family_logits).item())
            diagnostics.append(
                {
                    "position": position,
                    "amino_acid": amino_acid,
                    "codon_probabilities": {
                        codon: float(probability)
                        for codon, probability in zip(codons, family_probabilities)
                    },
                    "entropy_bits": entropy(family_probabilities),
                    "normalized_entropy": normalized_entropy(family_probabilities),
                    "max_probability": max(family_probabilities),
                    "argmax_codon": codons[argmax_rank],
                }
            )
        for name, strategy in strategies.items():
            if strategy["mode"] == "argmax":
                rank = int(torch.argmax(family_logits).item())
                token_index = allowed[rank]
            else:
                token_index = sample_allowed_token(
                    logits[position],
                    allowed,
                    float(strategy["temperature"]),
                    float(strategy["top_p"]),
                    generators[name],
                )
            decoded[name].append(token_index)
    dna = {
        name: "".join(INDEX2TOKEN[index][-3:] for index in indices).upper()
        for name, indices in decoded.items()
    }
    return dna, diagnostics


def read_cache(path: Path, expected_indices: set[int]) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    output: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            if idx not in expected_indices or idx in output:
                raise ValueError(f"Invalid cached idx={idx} at {path}:{line_number}")
            output[idx] = row
    return output


def generate_model_cache(
    model_label: str,
    model: BigBirdForMaskedLM,
    tokenizer,
    records: list[dict[str, Any]],
    organism_id: int,
    device: torch.device,
    strategies: dict[str, dict[str, Any]],
    seed: int,
    target_families: set[str],
    cache_path: Path,
    flush_every: int,
    logger: logging.Logger,
) -> dict[int, dict[str, Any]]:
    expected_indices = {record["idx"] for record in records}
    cached = read_cache(cache_path, expected_indices)
    if len(cached) == len(records):
        logger.info("Using complete %s diagnostic cache", model_label)
        return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    model.bert.set_attention_type("original_full")
    generated_since_flush = 0
    with cache_path.open("a", encoding="utf-8") as handle, torch.no_grad():
        for position, record in enumerate(records, start=1):
            idx = record["idx"]
            if idx in cached:
                continue
            merged = get_merged_seq(protein=record["protein"], dna="")
            merged_tokens = merged.split()
            batch = tokenize(
                [{"idx": idx, "codons": merged, "organism": organism_id}],
                tokenizer=tokenizer,
            ).to(device)
            logits = model(**batch, return_dict=True).logits[0, 1:-1, :].float().cpu()
            predictions, diagnostics = decode_record(
                logits,
                merged_tokens,
                strategies,
                seed,
                model_label,
                idx,
                target_families,
            )
            payload = {
                "idx": idx,
                "predictions": predictions,
                "target_family_probabilities": diagnostics,
            }
            cached[idx] = payload
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            generated_since_flush += 1
            if generated_since_flush >= flush_every:
                handle.flush()
                generated_since_flush = 0
                logger.info("Generated %s %d/%d", model_label, position, len(records))
        handle.flush()
    if len(cached) != len(records):
        raise RuntimeError(f"Incomplete {model_label} diagnostic cache")
    return cached


def aggregate_probability_diagnostics(
    cache: dict[int, dict[str, Any]],
    target_codons: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    accumulators: dict[str, dict[str, Any]] = {
        amino_acid: {
            "positions": 0,
            "probability_sums": Counter(),
            "entropy": [],
            "normalized_entropy": [],
            "max_probability": [],
            "argmax": Counter(),
        }
        for amino_acid in target_codons
    }
    for record in cache.values():
        for item in record["target_family_probabilities"]:
            amino_acid = item["amino_acid"]
            accumulator = accumulators[amino_acid]
            accumulator["positions"] += 1
            accumulator["probability_sums"].update(item["codon_probabilities"])
            accumulator["entropy"].append(float(item["entropy_bits"]))
            accumulator["normalized_entropy"].append(
                float(item["normalized_entropy"])
            )
            accumulator["max_probability"].append(float(item["max_probability"]))
            accumulator["argmax"][item["argmax_codon"]] += 1
    output: dict[str, Any] = {}
    for amino_acid, codons in target_codons.items():
        accumulator = accumulators[amino_acid]
        positions = accumulator["positions"]
        if not positions:
            raise ValueError(f"No validation positions found for family {amino_acid}")
        argmax_frequencies = {
            codon: accumulator["argmax"][codon] / positions for codon in codons
        }
        mean_probabilities = {
            codon: accumulator["probability_sums"][codon] / positions
            for codon in codons
        }
        output[amino_acid] = {
            "codons": list(codons),
            "positions": positions,
            "mean_conditional_probabilities": mean_probabilities,
            "entropy_of_mean_conditional_probabilities_bits": entropy(
                list(mean_probabilities.values())
            ),
            "normalized_entropy_of_mean_conditional_probabilities": normalized_entropy(
                list(mean_probabilities.values())
            ),
            "mean_position_entropy_bits": mean(accumulator["entropy"]),
            "mean_position_normalized_entropy": mean(
                accumulator["normalized_entropy"]
            ),
            "mean_position_max_probability": mean(accumulator["max_probability"]),
            "argmax_frequencies": argmax_frequencies,
            "argmax_concentration": max(argmax_frequencies.values()),
            "argmax_hhi": sum(value * value for value in argmax_frequencies.values()),
        }
    return output


def target_family_metrics(
    query_dna: str,
    true_dna: str,
    target_families: dict[str, tuple[str, ...]],
    reference_counts: dict[str, Counter[str]],
    stop_codons: set[str],
) -> tuple[float, float]:
    query_codons = [
        query_dna[index : index + 3]
        for index in range(0, len(query_dna) - 3, 3)
    ]
    true_codons = [
        true_dna[index : index + 3] for index in range(0, len(true_dna) - 3, 3)
    ]
    if query_dna[-3:] not in stop_codons or true_dna[-3:] not in stop_codons:
        raise ValueError("Expected standard terminal stop codons")
    query_counts = family_counts(query_codons, target_families)
    true_counts = family_counts(true_codons, target_families)
    jsd_true, _ = weighted_family_distances(
        query_counts, true_counts, target_families
    )
    jsd_reference, _ = weighted_family_distances(
        query_counts, reference_counts, target_families
    )
    return jsd_true, jsd_reference


def build_per_sequence_metrics(
    records: list[dict[str, Any]],
    caches: dict[str, dict[int, dict[str, Any]]],
    strategies: dict[str, dict[str, Any]],
    reference: dict[str, Any],
    target_families: dict[str, tuple[str, ...]],
    boundaries: tuple[float, float],
    rare_threshold: float,
) -> list[dict[str, Any]]:
    reference_counts = reference_family_counts(reference, target_families)
    _, _, stop_codons = family_definitions(int(reference["genetic_code"]))
    rows: list[dict[str, Any]] = []
    for record in records:
        true_metrics = sequence_metrics(
            record["true_dna"], record["protein"], reference, rare_threshold
        )
        row: dict[str, Any] = {
            "idx": record["idx"],
            "protein_length": record["protein_length"],
            "length_bin": length_bin(record["protein_length"], boundaries),
        }
        for model_label in MODEL_LABELS:
            for strategy in strategies:
                prefix = f"{model_label}_{strategy}"
                dna = caches[model_label][record["idx"]]["predictions"][strategy]
                metrics = sequence_metrics(
                    dna, record["protein"], reference, rare_threshold
                )
                if not metrics["translation_correct"] or not metrics["sequence_valid"]:
                    translated = str(
                        Seq(dna).translate(table=int(reference["genetic_code"]))
                    )
                    raise RuntimeError(
                        f"Translation constraint failed for {prefix}, idx={record['idx']}: "
                        f"{translated}"
                    )
                jsd_true, jsd_reference = target_family_metrics(
                    dna,
                    record["true_dna"],
                    target_families,
                    reference_counts,
                    stop_codons,
                )
                predicted_codons = [
                    dna[index : index + 3]
                    for index in range(0, len(dna) - 3, 3)
                ]
                true_codons = true_metrics["sense_codons"]
                row[f"{prefix}_dna"] = dna
                row[f"{prefix}_translation_correct"] = True
                row[f"{prefix}_sequence_valid"] = True
                row[f"{prefix}_csi"] = metrics["csi"]
                row[f"{prefix}_cai"] = metrics["cai"]
                row[f"{prefix}_gc_absolute_error"] = abs(
                    metrics["gc"] - true_metrics["gc"]
                )
                row[f"{prefix}_gc3_absolute_error"] = abs(
                    metrics["gc3"] - true_metrics["gc3"]
                )
                row[f"{prefix}_target_family_jsd_to_true"] = jsd_true
                row[f"{prefix}_target_family_jsd_to_reference"] = jsd_reference
                row[f"{prefix}_codon_match_rate"] = sum(
                    predicted == actual
                    for predicted, actual in zip(predicted_codons, true_codons)
                ) / len(true_codons)
        rows.append(row)
    return rows


def aggregate_strategy_families(
    records: list[dict[str, Any]],
    caches: dict[str, dict[int, dict[str, Any]]],
    strategies: dict[str, dict[str, Any]],
    target_families: dict[str, tuple[str, ...]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    reference_counts = reference_family_counts(reference, target_families)
    codon_to_family = {
        codon: amino_acid
        for amino_acid, codons in target_families.items()
        for codon in codons
    }
    true_counts = {family: Counter() for family in target_families}
    generated = {
        model: {
            strategy: {family: Counter() for family in target_families}
            for strategy in strategies
        }
        for model in MODEL_LABELS
    }
    for record in records:
        true_codons = [
            record["true_dna"][index : index + 3]
            for index in range(0, len(record["true_dna"]) - 3, 3)
        ]
        for codon in true_codons:
            family = codon_to_family.get(codon)
            if family:
                true_counts[family][codon] += 1
        for model in MODEL_LABELS:
            for strategy in strategies:
                dna = caches[model][record["idx"]]["predictions"][strategy]
                codons = [
                    dna[index : index + 3] for index in range(0, len(dna) - 3, 3)
                ]
                for codon in codons:
                    family = codon_to_family.get(codon)
                    if family:
                        generated[model][strategy][family][codon] += 1
    output: dict[str, Any] = {}
    for model in MODEL_LABELS:
        output[model] = {}
        for strategy in strategies:
            output[model][strategy] = {}
            for family, codons in target_families.items():
                query = generated[model][strategy][family]
                query_vector = [query[codon] for codon in codons]
                true_vector = [true_counts[family][codon] for codon in codons]
                reference_vector = [reference_counts[family][codon] for codon in codons]
                frequencies = normalized_counter(query, codons)
                output[model][strategy][family] = {
                    "codons": list(codons),
                    "positions": sum(query_vector),
                    "codon_frequencies": dict(zip(codons, frequencies)),
                    "normalized_entropy": normalized_entropy(frequencies),
                    "maximum_codon_frequency": max(frequencies),
                    "jsd_to_true": js_distance(query_vector, true_vector),
                    "jsd_to_reference": js_distance(query_vector, reference_vector),
                    "rscu_l1_to_true": probability_l1(query_vector, true_vector),
                    "rscu_l1_to_reference": probability_l1(
                        query_vector, reference_vector
                    ),
                }
    return output


def paired_strategy_test(
    rows: list[dict[str, Any]],
    candidate_prefix: str,
    greedy_prefix: str,
    metric: str,
    higher_is_better: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate = np.asarray(
        [float(row[f"{candidate_prefix}_{metric}"]) for row in rows]
    )
    greedy = np.asarray([float(row[f"{greedy_prefix}_{metric}"]) for row in rows])
    selected = np.isfinite(candidate) & np.isfinite(greedy)
    candidate = candidate[selected]
    greedy = greedy[selected]
    raw_delta = candidate - greedy
    improvement = raw_delta if higher_is_better else -raw_delta
    if np.all(np.abs(raw_delta) <= 1e-12):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(candidate, greedy, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(test.statistic), float(test.pvalue)
    return {
        "records": int(len(candidate)),
        "candidate_mean": float(candidate.mean()),
        "greedy_mean": float(greedy.mean()),
        "mean_improvement_vs_greedy": float(improvement.mean()),
        "bootstrap_95_ci_mean_improvement": list(
            bootstrap_ci(improvement, bootstrap_samples, seed)
        ),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value_two_sided": p_value,
    }


def strategy_comparisons(
    rows: list[dict[str, Any]],
    strategies: dict[str, dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_index, model in enumerate(MODEL_LABELS):
        output[model] = {}
        greedy_prefix = f"{model}_greedy"
        for strategy_index, strategy in enumerate(strategies):
            if strategy == "greedy":
                continue
            candidate_prefix = f"{model}_{strategy}"
            tests = {
                metric: paired_strategy_test(
                    rows,
                    candidate_prefix,
                    greedy_prefix,
                    metric,
                    higher_is_better,
                    bootstrap_samples,
                    seed + model_index * 100 + strategy_index * 20 + metric_index,
                )
                for metric_index, (metric, higher_is_better) in enumerate(
                    COMPARISON_METRICS.items()
                )
            }
            adjust_bh(tests)
            output[model][strategy] = tests
    return output


def collapse_diagnosis(
    probability: dict[str, Any],
    family_outputs: dict[str, Any],
    thresholds: dict[str, float],
    target_families: list[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in target_families:
        baseline = probability["baseline"][family]
        finetuned = probability["finetuned"][family]
        entropy_delta = (
            finetuned["mean_position_normalized_entropy"]
            - baseline["mean_position_normalized_entropy"]
        )
        max_delta = (
            finetuned["mean_position_max_probability"]
            - baseline["mean_position_max_probability"]
        )
        argmax_delta = (
            finetuned["argmax_concentration"] - baseline["argmax_concentration"]
        )
        probability_collapse = (
            entropy_delta <= -float(thresholds["normalized_entropy_drop"])
            and (
                max_delta
                >= float(thresholds["mean_max_probability_increase"])
                or argmax_delta
                >= float(thresholds["argmax_concentration_increase"])
            )
        )
        greedy_entropy = family_outputs["finetuned"]["greedy"][family][
            "normalized_entropy"
        ]
        probability_mass_entropy = finetuned[
            "normalized_entropy_of_mean_conditional_probabilities"
        ]
        greedy_gap = probability_mass_entropy - greedy_entropy
        greedy_amplification = greedy_gap >= float(
            thresholds["greedy_entropy_gap"]
        )
        family_sampling_jsd = family_outputs["finetuned"][
            "synonymous_family_sampling"
        ][family]["jsd_to_true"]
        greedy_jsd = family_outputs["finetuned"]["greedy"][family]["jsd_to_true"]
        sampling_improves_jsd = family_sampling_jsd < greedy_jsd
        if probability_collapse:
            mechanism = "model_probability_concentration"
        elif greedy_amplification and sampling_improves_jsd:
            mechanism = "greedy_decoding_amplification"
        elif greedy_amplification:
            mechanism = "greedy_amplification_without_jsd_rescue"
        else:
            mechanism = "mixed_or_no_collapse_signal"
        output[family] = {
            "finetuned_minus_baseline_mean_normalized_entropy": entropy_delta,
            "finetuned_minus_baseline_mean_max_probability": max_delta,
            "finetuned_minus_baseline_argmax_concentration": argmax_delta,
            "finetuned_probability_mass_minus_greedy_output_normalized_entropy": greedy_gap,
            "family_sampling_minus_greedy_jsd_to_true": family_sampling_jsd
            - greedy_jsd,
            "probability_collapse_signal": probability_collapse,
            "greedy_amplification_signal": greedy_amplification,
            "family_sampling_improves_jsd_to_true": sampling_improves_jsd,
            "provisional_mechanism": mechanism,
        }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Validation synonymous-family decoding diagnosis",
        "",
        "Validation-only analysis; the test split was neither read nor referenced.",
        "",
        "## Probability diagnostics",
        "",
        "| Family | Model | Mean normalized entropy | Mean max probability | Argmax concentration |",
        "|---|---|---:|---:|---:|",
    ]
    for family in summary["target_families"]:
        for model in MODEL_LABELS:
            result = summary["probability_diagnostics"][model][family]
            lines.append(
                f"| {family} | {model} | {result['mean_position_normalized_entropy']:.6f} | "
                f"{result['mean_position_max_probability']:.6f} | "
                f"{result['argmax_concentration']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Provisional collapse diagnosis",
            "",
            "| Family | Probability collapse | Greedy amplification | Family sampling improves JSD | Mechanism |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for family, result in summary["collapse_diagnosis"].items():
        lines.append(
            f"| {family} | {result['probability_collapse_signal']} | "
            f"{result['greedy_amplification_signal']} | "
            f"{result['family_sampling_improves_jsd_to_true']} | "
            f"{result['provisional_mechanism']} |"
        )
    lines.extend(
        [
            "",
            "## Length-stratified provisional mechanisms",
            "",
            "| Stratum | Family | Mechanism |",
            "|---|---|---|",
        ]
    )
    for stratum, families in summary["collapse_diagnosis_by_length"].items():
        for family, result in families.items():
            lines.append(
                f"| {stratum} | {family} | {result['provisional_mechanism']} |"
            )
    lines.extend(
        [
            "",
            "All generated DNA sequences passed strict translation and sequence-validity checks. Strategy selection is validation-only; do not read or tune against test outputs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-analysis", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    validation_dataset = args.validation_dataset.expanduser().resolve()
    reference_json = args.reference_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["dataset_role"] != "validation":
        raise ValueError("This experiment is validation-only")
    if validation_dataset.name != "validation.jsonl" or "test" in {
        part.lower() for part in validation_dataset.parts
    }:
        raise ValueError("Refusing any dataset other than validation.jsonl")
    if any(part.lower() == "biological_evaluation_v1" for part in output_dir.parts):
        raise ValueError("Refusing to write into the frozen test evaluation directory")
    for path in (config_path, model_dir, checkpoint, validation_dataset, reference_json):
        if not path.exists():
            raise FileNotFoundError(path)
    expected = config["inputs"]
    for label, expected_hash, path in (
        ("validation", expected["validation_sha256"], validation_dataset),
        ("reference", expected["reference_sha256"], reference_json),
        (
            "pretrained",
            expected["pretrained_model_safetensors_sha256"],
            model_dir / "model.safetensors",
        ),
    ):
        actual = sha256(path)
        if actual != expected_hash:
            raise ValueError(f"Unexpected {label} SHA256: {actual}")
    if checkpoint.name != expected["checkpoint_filename"]:
        raise ValueError("Unexpected checkpoint filename")
    if checkpoint.stat().st_size != int(expected["checkpoint_size_bytes"]):
        raise ValueError("Unexpected checkpoint size")
    checkpoint_sha256 = sha256(checkpoint)
    if expected.get("checkpoint_sha256") not in (None, checkpoint_sha256):
        raise ValueError("Unexpected checkpoint SHA256")

    records = load_test_records(validation_dataset, int(expected["expected_records"]))
    reference = load_reference(reference_json)
    target_families_list = list(config["target_families"])
    all_families, _, _ = family_definitions(int(reference["genetic_code"]))
    target_families = {
        family: all_families[family] for family in target_families_list
    }
    boundaries = (
        float(config["length_boundaries"]["short_max_aa"]),
        float(config["length_boundaries"]["medium_max_aa"]),
    )
    strategies = config["strategies"]
    for name, strategy in strategies.items():
        if strategy["mode"] not in {"argmax", "sample"}:
            raise ValueError(f"Invalid strategy mode for {name}")
        if float(strategy["temperature"]) <= 0:
            raise ValueError(f"Invalid temperature for {name}")
        if not 0 < float(strategy["top_p"]) <= 1:
            raise ValueError(f"Invalid top_p for {name}")

    manifest = {
        "experiment_version": config["experiment_version"],
        "dataset_role": "validation",
        "validation_sha256": expected["validation_sha256"],
        "reference_sha256": expected["reference_sha256"],
        "pretrained_sha256": expected["pretrained_model_safetensors_sha256"],
        "checkpoint_filename": checkpoint.name,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "records": len(records),
        "target_families": target_families_list,
        "strategies": strategies,
        "length_boundaries": config["length_boundaries"],
        "seed": int(config["statistics"]["seed"]),
        "test_access_prohibited": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing cache belongs to a different manifest")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("validation_decoding_diagnosis")
    device = select_device(config["model"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    organism_id, _ = validate_and_convert_organism(config["model"]["organism"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    caches: dict[str, dict[int, dict[str, Any]]] = {}
    for model_label in MODEL_LABELS:
        model = BigBirdForMaskedLM.from_pretrained(model_dir, local_files_only=True)
        if model_label == "finetuned":
            model.load_state_dict(load_checkpoint(checkpoint), strict=True)
        caches[model_label] = generate_model_cache(
            model_label,
            model,
            tokenizer,
            records,
            organism_id,
            device,
            strategies,
            int(config["statistics"]["seed"]),
            set(target_families_list),
            output_dir / "record_cache" / f"{model_label}.jsonl",
            int(config["runtime"]["cache_flush_every"]),
            logger,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    analysis_outputs = (
        output_dir / "family_probability_summary.csv",
        output_dir / "per_sequence_strategy_metrics.csv",
        output_dir / "decoding_diagnostic_summary.json",
        output_dir / "decoding_diagnostic_report.md",
    )
    if any(path.exists() for path in analysis_outputs) and not args.force_analysis:
        raise FileExistsError(
            "Analysis outputs already exist; caches are complete. Use --force-analysis "
            "to regenerate only the small summary files."
        )
    probability = {
        model: aggregate_probability_diagnostics(caches[model], target_families)
        for model in MODEL_LABELS
    }
    per_sequence = build_per_sequence_metrics(
        records,
        caches,
        strategies,
        reference,
        target_families,
        boundaries,
        float(config["statistics"]["rare_codon_threshold"]),
    )
    family_outputs = aggregate_strategy_families(
        records, caches, strategies, target_families, reference
    )
    comparisons = strategy_comparisons(
        per_sequence,
        strategies,
        int(config["statistics"]["bootstrap_samples"]),
        int(config["statistics"]["seed"]),
    )
    diagnosis = collapse_diagnosis(
        probability,
        family_outputs,
        config["statistics"]["collapse_thresholds"],
        target_families_list,
    )
    probability_by_length: dict[str, Any] = {}
    family_outputs_by_length: dict[str, Any] = {}
    comparisons_by_length: dict[str, Any] = {}
    diagnosis_by_length: dict[str, Any] = {}
    for stratum in ("short", "medium", "long"):
        stratum_records = [
            record
            for record in records
            if length_bin(record["protein_length"], boundaries) == stratum
        ]
        stratum_indices = {record["idx"] for record in stratum_records}
        stratum_caches = {
            model: {
                idx: payload
                for idx, payload in caches[model].items()
                if idx in stratum_indices
            }
            for model in MODEL_LABELS
        }
        probability_by_length[stratum] = {
            model: aggregate_probability_diagnostics(
                stratum_caches[model], target_families
            )
            for model in MODEL_LABELS
        }
        family_outputs_by_length[stratum] = aggregate_strategy_families(
            stratum_records,
            stratum_caches,
            strategies,
            target_families,
            reference,
        )
        stratum_rows = [row for row in per_sequence if row["length_bin"] == stratum]
        comparisons_by_length[stratum] = strategy_comparisons(
            stratum_rows,
            strategies,
            int(config["statistics"]["bootstrap_samples"]),
            int(config["statistics"]["seed"]) + 1000 * (len(comparisons_by_length) + 1),
        )
        diagnosis_by_length[stratum] = collapse_diagnosis(
            probability_by_length[stratum],
            family_outputs_by_length[stratum],
            config["statistics"]["collapse_thresholds"],
            target_families_list,
        )
    summary = {
        **manifest,
        "translation_constraint": "hard synonymous-family mask at every position",
        "probability_definition": "T=1 softmax conditional on the synonymous codon family",
        "probability_diagnostics": probability,
        "probability_diagnostics_by_length": probability_by_length,
        "strategy_family_outputs": family_outputs,
        "strategy_family_outputs_by_length": family_outputs_by_length,
        "paired_strategy_comparisons_vs_greedy": comparisons,
        "paired_strategy_comparisons_vs_greedy_by_length": comparisons_by_length,
        "collapse_diagnosis": diagnosis,
        "collapse_diagnosis_by_length": diagnosis_by_length,
        "selection_policy": (
            "Select decoding strategy using validation only. Test prediction caches and "
            "reports must not be read by this experiment."
        ),
    }
    probability_rows = []
    probability_scopes = {"overall": probability, **probability_by_length}
    for scope, scoped_probability in probability_scopes.items():
        for model in MODEL_LABELS:
            for family in target_families_list:
                result = scoped_probability[model][family]
                probability_rows.append(
                    {
                        "length_scope": scope,
                        "model": model,
                        "amino_acid": family,
                        "codons": " ".join(result["codons"]),
                        "positions": result["positions"],
                        "mean_position_normalized_entropy": result[
                            "mean_position_normalized_entropy"
                        ],
                        "mean_position_max_probability": result[
                            "mean_position_max_probability"
                        ],
                        "argmax_concentration": result["argmax_concentration"],
                        "argmax_hhi": result["argmax_hhi"],
                        "mean_conditional_probabilities": json.dumps(
                            result["mean_conditional_probabilities"], sort_keys=True
                        ),
                        "argmax_frequencies": json.dumps(
                            result["argmax_frequencies"], sort_keys=True
                        ),
                    }
                )
    write_csv(output_dir / "family_probability_summary.csv", probability_rows)
    write_csv(output_dir / "per_sequence_strategy_metrics.csv", per_sequence)
    (output_dir / "decoding_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "decoding_diagnostic_report.md", summary)
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
