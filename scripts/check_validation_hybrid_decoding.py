#!/usr/bin/env python3
"""Final cache-only decoder check before v2 fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml
from scipy.stats import wilcoxon

try:
    from evaluate_biological_fidelity import (
        codon_frequencies,
        jensen_shannon_distance,
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
        reference_family_counts,
        significant_improvement,
        significant_regression,
        weighted_family_distances,
    )
except ModuleNotFoundError:  # Imported as scripts.check_validation_hybrid_decoding.
    from scripts.evaluate_biological_fidelity import (
        codon_frequencies,
        jensen_shannon_distance,
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
        reference_family_counts,
        significant_improvement,
        significant_regression,
        weighted_family_distances,
    )


GREEDY = "greedy"
SOURCE_STRATEGIES = {
    "greedy",
    "temperature_sampling",
    "synonymous_family_sampling",
}
PAIRED_METRICS = {
    "translation_correct_rate": True,
    "sequence_valid_rate": True,
    "csi": True,
    "cai": True,
    "gc_absolute_error": False,
    "gc3_absolute_error": False,
    "codon_jsd_to_true": False,
    "rare_codon_fraction_absolute_error": False,
    "codon_match_rate": True,
    "synonymous_family_jsd_to_true": False,
    "synonymous_family_jsd_to_reference": False,
    "synonymous_family_rscu_l1_to_true": False,
    "synonymous_family_rscu_l1_to_reference": False,
    "target_family_jsd_to_true": False,
    "target_family_jsd_to_reference": False,
    "target_family_rscu_l1_to_true": False,
    "target_family_rscu_l1_to_reference": False,
}
TARGET_JSD_METRICS = {
    "target_family_jsd_to_true",
    "target_family_jsd_to_reference",
}
TARGET_RSCU_METRICS = {
    "target_family_rscu_l1_to_true",
    "target_family_rscu_l1_to_reference",
}
LENGTH_PAIRED_METRICS = {
    metric: direction
    for metric, direction in PAIRED_METRICS.items()
    if metric
    in {
        "csi",
        "cai",
        "gc_absolute_error",
        "gc3_absolute_error",
        "codon_match_rate",
        "synonymous_family_jsd_to_true",
        "synonymous_family_jsd_to_reference",
        "target_family_jsd_to_true",
        "target_family_jsd_to_reference",
        "target_family_rscu_l1_to_true",
        "target_family_rscu_l1_to_reference",
    }
}


def read_cache(path: Path, expected_indices: set[int]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["idx"])
            if idx not in expected_indices or idx in output:
                raise ValueError(f"Invalid cached idx={idx} at {path}:{line_number}")
            if set(row["predictions"]) != SOURCE_STRATEGIES:
                raise ValueError(f"Unexpected source strategies at idx={idx}")
            output[idx] = row
    if set(output) != expected_indices:
        raise ValueError(
            f"Expected {len(expected_indices)} cached records, found {len(output)}"
        )
    return output


def codons(dna: str) -> list[str]:
    if not dna or len(dna) % 3:
        raise ValueError("DNA must be non-empty and divisible by three")
    return [dna[index : index + 3] for index in range(0, len(dna), 3)]


def rule_matches(item: dict[str, Any], rule: dict[str, Any]) -> bool:
    if item["amino_acid"] not in set(rule["families"]):
        return False
    if (
        "minimum_normalized_entropy" in rule
        and float(item["normalized_entropy"])
        < float(rule["minimum_normalized_entropy"])
    ):
        return False
    if (
        "maximum_probability" in rule
        and float(item["max_probability"]) > float(rule["maximum_probability"])
    ):
        return False
    return True


def build_candidate_dna(
    cache_record: dict[str, Any], candidate: dict[str, Any]
) -> tuple[str, dict[str, int]]:
    default_source = candidate["default_source"]
    if default_source not in SOURCE_STRATEGIES:
        raise ValueError(f"Unknown default source: {default_source}")
    sources = {
        name: codons(dna) for name, dna in cache_record["predictions"].items()
    }
    lengths = {len(values) for values in sources.values()}
    if len(lengths) != 1:
        raise ValueError(f"Source prediction lengths disagree for idx={cache_record['idx']}")
    output = list(sources[default_source])
    selected = Counter({default_source: len(output)})
    for item in cache_record["target_family_probabilities"]:
        position = int(item["position"])
        for rule in candidate["rules"]:
            if not rule_matches(item, rule):
                continue
            source = rule["source"]
            if source not in SOURCE_STRATEGIES:
                raise ValueError(f"Unknown candidate source: {source}")
            output[position] = sources[source][position]
            selected[default_source] -= 1
            selected[source] += 1
            break
    return "".join(output), dict(selected)


def sense_codons(dna: str, stop_codons: set[str]) -> list[str]:
    values = codons(dna)
    if values[-1] not in stop_codons:
        raise ValueError("Expected a standard terminal stop codon")
    return values[:-1]


def conditional_distances(
    query_codons: list[str],
    true_codons: list[str],
    families: dict[str, tuple[str, ...]],
    reference_counts: dict[str, Counter[str]],
) -> dict[str, float]:
    query_counts = family_counts(query_codons, families)
    true_counts = family_counts(true_codons, families)
    jsd_true, rscu_true = weighted_family_distances(
        query_counts, true_counts, families
    )
    jsd_reference, rscu_reference = weighted_family_distances(
        query_counts, reference_counts, families
    )
    return {
        "jsd_to_true": jsd_true,
        "jsd_to_reference": jsd_reference,
        "rscu_l1_to_true": rscu_true,
        "rscu_l1_to_reference": rscu_reference,
    }


def calculate_metrics(
    dna: str,
    record: dict[str, Any],
    reference: dict[str, Any],
    rare_threshold: float,
    all_families: dict[str, tuple[str, ...]],
    target_families: dict[str, tuple[str, ...]],
    all_reference_counts: dict[str, Counter[str]],
    target_reference_counts: dict[str, Counter[str]],
    stop_codons: set[str],
    sense_codon_order: tuple[str, ...],
) -> dict[str, Any]:
    true_metrics = sequence_metrics(
        record["true_dna"], record["protein"], reference, rare_threshold
    )
    metrics = sequence_metrics(dna, record["protein"], reference, rare_threshold)
    query_sense = sense_codons(dna, stop_codons)
    true_sense = sense_codons(record["true_dna"], stop_codons)
    all_distances = conditional_distances(
        query_sense, true_sense, all_families, all_reference_counts
    )
    target_distances = conditional_distances(
        query_sense, true_sense, target_families, target_reference_counts
    )
    query_frequencies = codon_frequencies(query_sense, sense_codon_order)
    true_frequencies = codon_frequencies(true_sense, sense_codon_order)
    return {
        "translation_correct_rate": float(metrics["translation_correct"]),
        "sequence_valid_rate": float(metrics["sequence_valid"]),
        "csi": metrics["csi"],
        "cai": metrics["cai"],
        "gc_absolute_error": abs(metrics["gc"] - true_metrics["gc"]),
        "gc3_absolute_error": abs(metrics["gc3"] - true_metrics["gc3"]),
        "codon_jsd_to_true": jensen_shannon_distance(
            query_frequencies, true_frequencies
        ),
        "rare_codon_fraction_absolute_error": abs(
            metrics["rare_codon_fraction"]
            - true_metrics["rare_codon_fraction"]
        ),
        "codon_match_rate": sum(
            query == actual for query, actual in zip(query_sense, true_sense)
        )
        / len(true_sense),
        "synonymous_family_jsd_to_true": all_distances["jsd_to_true"],
        "synonymous_family_jsd_to_reference": all_distances[
            "jsd_to_reference"
        ],
        "synonymous_family_rscu_l1_to_true": all_distances["rscu_l1_to_true"],
        "synonymous_family_rscu_l1_to_reference": all_distances[
            "rscu_l1_to_reference"
        ],
        "target_family_jsd_to_true": target_distances["jsd_to_true"],
        "target_family_jsd_to_reference": target_distances["jsd_to_reference"],
        "target_family_rscu_l1_to_true": target_distances["rscu_l1_to_true"],
        "target_family_rscu_l1_to_reference": target_distances[
            "rscu_l1_to_reference"
        ],
    }


def build_rows(
    records: list[dict[str, Any]],
    cache: dict[int, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    reference: dict[str, Any],
    target_family_names: list[str],
    boundaries: tuple[float, float],
    rare_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_families, _, stop_codons = family_definitions(int(reference["genetic_code"]))
    target_families = {
        family: all_families[family] for family in target_family_names
    }
    all_reference_counts = reference_family_counts(reference, all_families)
    target_reference_counts = reference_family_counts(reference, target_families)
    sense_codon_order = tuple(
        sorted(reference["csi_reference"]["frequencies"])
    )
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for record in records:
        base = {
            "idx": record["idx"],
            "protein_length": record["protein_length"],
            "length_bin": length_bin(record["protein_length"], boundaries),
        }
        greedy_dna = cache[record["idx"]]["predictions"][GREEDY]
        greedy_metrics = calculate_metrics(
            greedy_dna,
            record,
            reference,
            rare_threshold,
            all_families,
            target_families,
            all_reference_counts,
            target_reference_counts,
            stop_codons,
            sense_codon_order,
        )
        for metric, value in greedy_metrics.items():
            base[f"greedy_{metric}"] = value
        prediction_row: dict[str, Any] = {
            "idx": record["idx"],
            "greedy_dna": greedy_dna,
        }
        for candidate_name, candidate in candidates.items():
            dna, source_counts = build_candidate_dna(cache[record["idx"]], candidate)
            metrics = calculate_metrics(
                dna,
                record,
                reference,
                rare_threshold,
                all_families,
                target_families,
                all_reference_counts,
                target_reference_counts,
                stop_codons,
                sense_codon_order,
            )
            if metrics["translation_correct_rate"] != 1.0 or metrics[
                "sequence_valid_rate"
            ] != 1.0:
                raise RuntimeError(
                    f"Hard translation constraint failed for {candidate_name}, "
                    f"idx={record['idx']}"
                )
            for metric, value in metrics.items():
                base[f"{candidate_name}_{metric}"] = value
            prediction_row[f"{candidate_name}_dna"] = dna
            prediction_row[f"{candidate_name}_source_counts"] = json.dumps(
                source_counts, sort_keys=True
            )
        rows.append(base)
        predictions.append(prediction_row)
    return rows, predictions


def paired_test(
    rows: list[dict[str, Any]],
    candidate: str,
    metric: str,
    higher_is_better: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    greedy = np.asarray([float(row[f"greedy_{metric}"]) for row in rows])
    proposed = np.asarray([float(row[f"{candidate}_{metric}"]) for row in rows])
    selected = np.isfinite(greedy) & np.isfinite(proposed)
    greedy = greedy[selected]
    proposed = proposed[selected]
    raw_delta = proposed - greedy
    improvement = raw_delta if higher_is_better else -raw_delta
    if np.all(np.abs(raw_delta) <= 1e-12):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(proposed, greedy, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "records": int(len(proposed)),
        "greedy_mean": float(greedy.mean()),
        "candidate_mean": float(proposed.mean()),
        "mean_improvement_vs_greedy": float(improvement.mean()),
        "bootstrap_95_ci_mean_improvement": list(
            bootstrap_ci(improvement, bootstrap_samples, seed)
        ),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value_two_sided": p_value,
    }


def compare_candidates(
    rows: list[dict[str, Any]],
    candidate_names: list[str],
    metrics: dict[str, bool],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    nested: dict[str, dict[str, Any]] = {}
    flat: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidate_names):
        nested[candidate] = {}
        for metric_index, (metric, higher_is_better) in enumerate(metrics.items()):
            result = paired_test(
                rows,
                candidate,
                metric,
                higher_is_better,
                bootstrap_samples,
                seed + candidate_index * 100 + metric_index,
            )
            nested[candidate][metric] = result
            flat[f"{candidate}:{metric}"] = result
    adjust_bh(flat)
    return nested


def evaluate_gate(
    rows: list[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    by_length: dict[str, dict[str, dict[str, Any]]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    protected = set(gate["protected_metrics"])
    output: dict[str, Any] = {}
    for candidate, tests in comparisons.items():
        stable = [metric for metric, result in tests.items() if significant_improvement(result)]
        regressions = [
            metric
            for metric, result in tests.items()
            if metric in protected | TARGET_JSD_METRICS | TARGET_RSCU_METRICS
            and significant_regression(result)
        ]
        length_regressions = [
            f"{stratum}:{metric}"
            for stratum, candidate_results in by_length.items()
            for metric, result in candidate_results[candidate].items()
            if metric in protected | TARGET_JSD_METRICS | TARGET_RSCU_METRICS
            and significant_regression(result)
        ]
        translation_rate = mean(
            float(row[f"{candidate}_translation_correct_rate"]) for row in rows
        )
        validity_rate = mean(
            float(row[f"{candidate}_sequence_valid_rate"]) for row in rows
        )
        stable_jsd = sorted(TARGET_JSD_METRICS & set(stable))
        stable_rscu = sorted(TARGET_RSCU_METRICS & set(stable))
        eligible = (
            translation_rate >= float(gate["translation_requirement"])
            and validity_rate >= float(gate["validity_requirement"])
            and (bool(stable_jsd) or not gate["require_stable_jsd_improvement"])
            and (bool(stable_rscu) or not gate["require_stable_rscu_improvement"])
            and not regressions
            and not length_regressions
        )
        output[candidate] = {
            "translation_correct_rate": translation_rate,
            "sequence_valid_rate": validity_rate,
            "stable_target_jsd_improvements": stable_jsd,
            "stable_target_rscu_improvements": stable_rscu,
            "overall_significant_regressions": regressions,
            "length_stratum_significant_regressions": length_regressions,
            "decoder_only_gate_passed": eligible,
        }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Validation hybrid decoding final gate",
        "",
        "Cache-only validation analysis. No model, checkpoint, logits, or test data were loaded.",
        "",
        "| Candidate | Stable target JSD | Stable target RSCU | Overall regressions | Length regressions | Gate passed |",
        "|---|---|---|---|---|---:|",
    ]
    for candidate, decision in summary["candidate_decisions"].items():
        lines.append(
            f"| {candidate} | "
            f"{', '.join(decision['stable_target_jsd_improvements']) or 'none'} | "
            f"{', '.join(decision['stable_target_rscu_improvements']) or 'none'} | "
            f"{', '.join(decision['overall_significant_regressions']) or 'none'} | "
            f"{', '.join(decision['length_stratum_significant_regressions']) or 'none'} | "
            f"{decision['decoder_only_gate_passed']} |"
        )
    lines.extend(
        [
            "",
            f"Decoder-only resolution found: {summary['decision']['decoder_only_resolution_found']}",
            f"Proceed to v2 fine-tuning: {summary['decision']['proceed_to_v2_finetuning']}",
            "",
            "The test split remains out of scope. Any future final assessment requires a new external holdout.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--source-decoding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    validation_dataset = args.validation_dataset.expanduser().resolve()
    reference_json = args.reference_json.expanduser().resolve()
    source_dir = args.source_decoding_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["dataset_role"] != "validation":
        raise ValueError("Hybrid decoder check is validation-only")
    if validation_dataset.name != "validation.jsonl":
        raise ValueError("Refusing any dataset other than validation.jsonl")
    if any(part.lower() == "test" for part in validation_dataset.parts):
        raise ValueError("Test access is prohibited")
    for path in (validation_dataset, reference_json, source_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    inputs = config["inputs"]
    if sha256(validation_dataset) != inputs["validation_sha256"]:
        raise ValueError("Unexpected validation SHA256")
    if sha256(reference_json) != inputs["reference_sha256"]:
        raise ValueError("Unexpected reference SHA256")
    source_manifest_path = source_dir / "evaluation_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    for key, expected in (
        ("experiment_version", inputs["source_experiment_version"]),
        ("dataset_role", "validation"),
        ("validation_sha256", inputs["validation_sha256"]),
        ("checkpoint_sha256", inputs["source_checkpoint_sha256"]),
        ("records", int(inputs["expected_records"])),
    ):
        if source_manifest.get(key) != expected:
            raise ValueError(f"Unexpected source manifest {key}")
    if source_manifest.get("target_families") != list(config["target_families"]):
        raise ValueError("Unexpected source manifest target_families")
    if set(source_manifest.get("strategies", {})) != SOURCE_STRATEGIES:
        raise ValueError("Unexpected source manifest strategies")

    records = load_test_records(validation_dataset, int(inputs["expected_records"]))
    expected_indices = {record["idx"] for record in records}
    cache_path = source_dir / "record_cache" / "finetuned.jsonl"
    cache = read_cache(cache_path, expected_indices)
    reference = load_reference(reference_json)
    candidates = config["candidates"]
    target_families = list(config["target_families"])
    for candidate_name, candidate in candidates.items():
        if candidate["default_source"] not in SOURCE_STRATEGIES:
            raise ValueError(f"Invalid default source for {candidate_name}")
        for rule in candidate["rules"]:
            if not set(rule["families"]) <= set(target_families):
                raise ValueError(f"Invalid target family for {candidate_name}")
            if rule["source"] not in SOURCE_STRATEGIES:
                raise ValueError(f"Invalid source for {candidate_name}")
    boundaries = (
        float(config["length_boundaries"]["short_max_aa"]),
        float(config["length_boundaries"]["medium_max_aa"]),
    )
    output_files = (
        output_dir / "evaluation_manifest.json",
        output_dir / "hybrid_decoding_summary.json",
        output_dir / "hybrid_decoding_report.md",
        output_dir / "per_sequence_hybrid_metrics.csv",
        output_dir / "candidate_predictions.jsonl",
    )
    if not args.force and any(path.exists() for path in output_files):
        raise FileExistsError("Refusing to overwrite existing hybrid decoder outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, predictions = build_rows(
        records,
        cache,
        candidates,
        reference,
        target_families,
        boundaries,
        float(config["statistics"]["rare_codon_threshold"]),
    )
    candidate_names = list(candidates)
    comparisons = compare_candidates(
        rows,
        candidate_names,
        PAIRED_METRICS,
        int(config["statistics"]["bootstrap_samples"]),
        int(config["statistics"]["seed"]),
    )
    comparisons_by_length: dict[str, Any] = {}
    for stratum_index, stratum in enumerate(("short", "medium", "long")):
        subset = [row for row in rows if row["length_bin"] == stratum]
        comparisons_by_length[stratum] = compare_candidates(
            subset,
            candidate_names,
            LENGTH_PAIRED_METRICS,
            int(config["statistics"]["bootstrap_samples"]),
            int(config["statistics"]["seed"]) + 1000 * (stratum_index + 1),
        )
    decisions = evaluate_gate(
        rows,
        comparisons,
        comparisons_by_length,
        config["selection_gate"],
    )
    eligible = [
        name for name, decision in decisions.items() if decision["decoder_only_gate_passed"]
    ]
    manifest = {
        "experiment_version": config["experiment_version"],
        "dataset_role": "validation",
        "records": len(records),
        "validation_sha256": inputs["validation_sha256"],
        "reference_sha256": inputs["reference_sha256"],
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_finetuned_cache_sha256": sha256(cache_path),
        "source_checkpoint_sha256": inputs["source_checkpoint_sha256"],
        "candidates": candidates,
        "length_boundaries": config["length_boundaries"],
        "seed": int(config["statistics"]["seed"]),
        "bootstrap_samples": int(config["statistics"]["bootstrap_samples"]),
        "test_access_prohibited": True,
        "model_forward_performed": False,
        "training_performed": False,
    }
    summary = {
        **manifest,
        "multiple_testing": config["statistics"]["multiple_testing"],
        "paired_comparisons_vs_v1_greedy": comparisons,
        "paired_comparisons_vs_v1_greedy_by_length": comparisons_by_length,
        "candidate_decisions": decisions,
        "decision": {
            "eligible_decoder_only_candidates": eligible,
            "decoder_only_resolution_found": bool(eligible),
            "proceed_to_v2_finetuning": not bool(eligible),
        },
    }
    write_csv(output_dir / "per_sequence_hybrid_metrics.csv", rows)
    write_jsonl(output_dir / "candidate_predictions.jsonl", predictions)
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "hybrid_decoding_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "hybrid_decoding_report.md", summary)
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
