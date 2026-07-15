#!/usr/bin/env python3
"""Refine an existing biological evaluation without rerunning model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from Bio.Data import CodonTable
from scipy.stats import wilcoxon


LABELS = ("baseline", "finetuned")
PAIRED_METRICS = {
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
    "synonymous_family_jsd_to_true": False,
    "synonymous_family_jsd_to_reference": False,
    "synonymous_family_rscu_l1_to_true": False,
    "synonymous_family_rscu_l1_to_reference": False,
}
TARGET_CATEGORIES = {
    "preference": ("csi", "cai"),
    "composition": ("gc_absolute_error", "gc3_absolute_error"),
    "distribution": (
        "synonymous_family_jsd_to_true",
        "synonymous_family_jsd_to_reference",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_value(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def js_distance(first: Iterable[float], second: Iterable[float]) -> float:
    left = np.asarray(list(first), dtype=float)
    right = np.asarray(list(second), dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Jensen-Shannon vectors must be equal one-dimensional arrays")
    if left.sum() <= 0 or right.sum() <= 0:
        return math.nan
    left = left / left.sum()
    right = right / right.sum()
    midpoint = (left + right) / 2

    def kl(values: np.ndarray) -> float:
        selected = values > 0
        return float(np.sum(values[selected] * np.log2(values[selected] / midpoint[selected])))

    return math.sqrt((kl(left) + kl(right)) / 2)


def probability_l1(first: Iterable[float], second: Iterable[float]) -> float:
    left = np.asarray(list(first), dtype=float)
    right = np.asarray(list(second), dtype=float)
    if left.sum() <= 0 or right.sum() <= 0:
        return math.nan
    return float(np.abs(left / left.sum() - right / right.sum()).sum())


def sense_codons(dna: str, stop_codons: set[str]) -> list[str]:
    dna = dna.upper()
    if not dna or len(dna) % 3:
        raise ValueError("DNA must be non-empty and divisible by three")
    codons = [dna[index : index + 3] for index in range(0, len(dna), 3)]
    if codons[-1] not in stop_codons:
        raise ValueError("DNA must end in a standard stop codon")
    return codons[:-1]


def family_definitions(genetic_code: int) -> tuple[dict[str, tuple[str, ...]], dict[str, str], set[str]]:
    table = CodonTable.unambiguous_dna_by_id[genetic_code]
    families: dict[str, list[str]] = {}
    for codon, amino_acid in table.forward_table.items():
        families.setdefault(amino_acid, []).append(codon)
    synonymous = {
        amino_acid: tuple(sorted(codons))
        for amino_acid, codons in sorted(families.items())
        if len(codons) > 1
    }
    codon_to_family = {
        codon: amino_acid for amino_acid, codons in synonymous.items() for codon in codons
    }
    return synonymous, codon_to_family, set(table.stop_codons)


def family_counts(codons: Iterable[str], families: dict[str, tuple[str, ...]]) -> dict[str, Counter[str]]:
    lookup = {codon: amino_acid for amino_acid, values in families.items() for codon in values}
    counts = {amino_acid: Counter() for amino_acid in families}
    for codon in codons:
        if codon in lookup:
            counts[lookup[codon]][codon] += 1
    return counts


def weighted_family_distances(
    query: dict[str, Counter[str]],
    target: dict[str, Counter[str]],
    families: dict[str, tuple[str, ...]],
) -> tuple[float, float]:
    family_totals = {
        amino_acid: sum(query[amino_acid][codon] for codon in codons)
        for amino_acid, codons in families.items()
    }
    total = sum(family_totals.values())
    if not total:
        return math.nan, math.nan
    jsd = 0.0
    rscu_l1 = 0.0
    for amino_acid, codons in families.items():
        family_total = family_totals[amino_acid]
        if not family_total:
            continue
        query_vector = [query[amino_acid][codon] for codon in codons]
        target_vector = [target[amino_acid][codon] for codon in codons]
        if not sum(target_vector):
            continue
        weight = family_total / total
        jsd += weight * js_distance(query_vector, target_vector)
        # For one synonymous family, mean absolute RSCU difference equals
        # probability L1 because RSCU is degeneracy * conditional frequency.
        rscu_l1 += weight * probability_l1(query_vector, target_vector)
    return jsd, rscu_l1


def reference_family_counts(
    reference: dict[str, Any], families: dict[str, tuple[str, ...]]
) -> dict[str, Counter[str]]:
    counts = reference["csi_reference"]["counts"]
    return {
        amino_acid: Counter({codon: int(counts[codon]) for codon in codons})
        for amino_acid, codons in families.items()
    }


def bootstrap_ci(
    values: np.ndarray, samples: int, seed: int, chunk_size: int = 500
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for start in range(0, samples, chunk_size):
        end = min(samples, start + chunk_size)
        indices = rng.integers(0, len(values), size=(end - start, len(values)))
        estimates[start:end] = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def paired_statistic(
    rows: list[dict[str, Any]],
    metric: str,
    higher_is_better: bool,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = np.asarray([float(row[f"baseline_{metric}"]) for row in rows])
    finetuned = np.asarray([float(row[f"finetuned_{metric}"]) for row in rows])
    selected = np.isfinite(baseline) & np.isfinite(finetuned)
    baseline = baseline[selected]
    finetuned = finetuned[selected]
    raw_delta = finetuned - baseline
    improvement = raw_delta if higher_is_better else -raw_delta
    tolerance = 1e-12
    if np.all(np.abs(raw_delta) <= tolerance):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(finetuned, baseline, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    interval = bootstrap_ci(improvement, bootstrap_samples, seed)
    return {
        "records": int(len(improvement)),
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "baseline_mean": float(baseline.mean()),
        "finetuned_mean": float(finetuned.mean()),
        "finetuned_minus_baseline": float(raw_delta.mean()),
        "mean_improvement": float(improvement.mean()),
        "bootstrap_95_ci_mean_improvement": list(interval),
        "wins_ties_losses": {
            "wins": int(np.sum(improvement > tolerance)),
            "ties": int(np.sum(np.abs(improvement) <= tolerance)),
            "losses": int(np.sum(improvement < -tolerance)),
        },
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value_two_sided": p_value,
    }


def adjust_bh(tests: dict[str, dict[str, Any]]) -> None:
    names = sorted(tests, key=lambda name: tests[name]["wilcoxon_p_value_two_sided"])
    total = len(names)
    running = 1.0
    adjusted = [1.0] * total
    for index in range(total - 1, -1, -1):
        rank = index + 1
        candidate = min(
            1.0,
            tests[names[index]]["wilcoxon_p_value_two_sided"] * total / rank,
        )
        running = min(running, candidate)
        adjusted[index] = running
    for name, q_value in zip(names, adjusted):
        tests[name]["wilcoxon_q_value_bh"] = q_value


def significant_improvement(result: dict[str, Any]) -> bool:
    return (
        result["bootstrap_95_ci_mean_improvement"][0] > 0
        and result["wilcoxon_q_value_bh"] < 0.05
    )


def significant_regression(result: dict[str, Any]) -> bool:
    return (
        result["bootstrap_95_ci_mean_improvement"][1] < 0
        and result["wilcoxon_q_value_bh"] < 0.05
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No per-sequence records found in {path}")
    required = {
        "idx",
        "protein",
        "protein_length",
        "length_bin",
        "true_dna",
        "baseline_dna",
        "finetuned_dna",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing per-sequence columns: {sorted(missing)}")
    return rows


def enrich_rows(
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]], set[str]]:
    genetic_code = int(reference["genetic_code"])
    families, _, stop_codons = family_definitions(genetic_code)
    reference_counts = reference_family_counts(reference, families)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        result: dict[str, Any] = dict(row)
        true_codons = sense_codons(row["true_dna"], stop_codons)
        true_counts = family_counts(true_codons, families)
        true_ref_jsd, true_ref_rscu = weighted_family_distances(
            true_counts, reference_counts, families
        )
        result["true_synonymous_family_jsd_to_reference"] = true_ref_jsd
        result["true_synonymous_family_rscu_l1_to_reference"] = true_ref_rscu
        for label in LABELS:
            query_codons = sense_codons(row[f"{label}_dna"], stop_codons)
            query_counts = family_counts(query_codons, families)
            true_jsd, true_rscu = weighted_family_distances(
                query_counts, true_counts, families
            )
            ref_jsd, ref_rscu = weighted_family_distances(
                query_counts, reference_counts, families
            )
            result[f"{label}_synonymous_family_jsd_to_true"] = true_jsd
            result[f"{label}_synonymous_family_jsd_to_reference"] = ref_jsd
            result[f"{label}_synonymous_family_rscu_l1_to_true"] = true_rscu
            result[f"{label}_synonymous_family_rscu_l1_to_reference"] = ref_rscu
            result[f"{label}_translation_correct_rate"] = float(
                bool_value(row[f"{label}_translation_correct"])
            )
            result[f"{label}_sequence_valid_rate"] = float(
                bool_value(row[f"{label}_sequence_valid"])
            )
        enriched.append(result)
    return enriched, families, stop_codons


def aggregate_family_attribution(
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    families: dict[str, tuple[str, ...]],
    stop_codons: set[str],
) -> list[dict[str, Any]]:
    reference_counts = reference_family_counts(reference, families)
    aggregate = {
        label: {amino_acid: Counter() for amino_acid in families}
        for label in ("true", *LABELS)
    }
    matches = {
        label: Counter({amino_acid: 0 for amino_acid in families})
        for label in LABELS
    }
    positions: Counter[str] = Counter()
    codon_to_family = {
        codon: amino_acid for amino_acid, codons in families.items() for codon in codons
    }
    for row in rows:
        true_codons = sense_codons(row["true_dna"], stop_codons)
        generated = {
            label: sense_codons(row[f"{label}_dna"], stop_codons) for label in LABELS
        }
        for index, true_codon in enumerate(true_codons):
            amino_acid = codon_to_family.get(true_codon)
            if amino_acid is None:
                continue
            positions[amino_acid] += 1
            aggregate["true"][amino_acid][true_codon] += 1
            for label in LABELS:
                predicted = generated[label][index]
                aggregate[label][amino_acid][predicted] += 1
                matches[label][amino_acid] += int(predicted == true_codon)
    total_positions = sum(positions.values())
    output = []
    for amino_acid, codons in families.items():
        true_vector = [aggregate["true"][amino_acid][codon] for codon in codons]
        reference_vector = [reference_counts[amino_acid][codon] for codon in codons]
        row: dict[str, Any] = {
            "amino_acid": amino_acid,
            "codons": " ".join(codons),
            "positions": positions[amino_acid],
            "position_fraction": positions[amino_acid] / total_positions,
            "true_jsd_to_reference": js_distance(true_vector, reference_vector),
        }
        for label in LABELS:
            vector = [aggregate[label][amino_acid][codon] for codon in codons]
            row[f"{label}_jsd_to_true"] = js_distance(vector, true_vector)
            row[f"{label}_jsd_to_reference"] = js_distance(vector, reference_vector)
            row[f"{label}_rscu_l1_to_true"] = probability_l1(vector, true_vector)
            row[f"{label}_rscu_l1_to_reference"] = probability_l1(
                vector, reference_vector
            )
            row[f"{label}_codon_match_rate"] = (
                matches[label][amino_acid] / positions[amino_acid]
            )
        row["finetuned_minus_baseline_jsd_to_true"] = (
            row["finetuned_jsd_to_true"] - row["baseline_jsd_to_true"]
        )
        row["weighted_jsd_to_true_regression_contribution"] = (
            row["position_fraction"] * row["finetuned_minus_baseline_jsd_to_true"]
        )
        output.append(row)
    return sorted(
        output,
        key=lambda row: row["weighted_jsd_to_true_regression_contribution"],
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict[str, Any], families: list[dict[str, Any]]) -> None:
    lines = [
        "# Refined csi_top10_hc biological evaluation",
        "",
        "This is an analysis-only refinement of the frozen 594-record v1 test predictions; no model inference or training was rerun.",
        "",
        "## Strict decision",
        "",
        f"- Translation requirement met: {summary['decision']['translation_requirement_met']}",
        f"- Validity requirement met: {summary['decision']['validity_requirement_met']}",
        f"- Stable improvement categories: {', '.join(summary['decision']['stable_improvement_categories']) or 'none'}",
        f"- Overall significant regressions: {', '.join(summary['decision']['overall_significant_regressions']) or 'none'}",
        f"- Length-stratum significant regressions: {', '.join(summary['decision']['length_stratum_significant_regressions']) or 'none'}",
        f"- Supports broad N. benthamiana codon-adaptation claim: {summary['decision']['supports_claim']}",
        "",
        "## Overall paired statistics",
        "",
        "| Metric | Baseline | Fine-tuned | Improvement | 95% bootstrap CI | BH q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, result in summary["overall_paired_statistics"].items():
        interval = result["bootstrap_95_ci_mean_improvement"]
        lines.append(
            f"| {metric} | {result['baseline_mean']:.6f} | {result['finetuned_mean']:.6f} | "
            f"{result['mean_improvement']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] | "
            f"{result['wilcoxon_q_value_bh']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Length-stratified target metrics",
            "",
            "Positive improvement means fine-tuned is better after accounting for metric direction.",
            "",
            "| Stratum | Metric | Improvement | 95% bootstrap CI | BH q |",
            "|---|---|---:|---:|---:|",
        ]
    )
    target_metrics = {
        metric for metrics in TARGET_CATEGORIES.values() for metric in metrics
    }
    for stratum, tests in summary["length_stratified_paired_statistics"].items():
        for metric, result in tests.items():
            if metric not in target_metrics:
                continue
            interval = result["bootstrap_95_ci_mean_improvement"]
            lines.append(
                f"| {stratum} | {metric} | {result['mean_improvement']:.6f} | "
                f"[{interval[0]:.6f}, {interval[1]:.6f}] | {result['wilcoxon_q_value_bh']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Largest synonymous-family JSD regression contributors",
            "",
            "| Amino acid | Positions | Baseline JSD to true | Fine-tuned JSD to true | Weighted regression contribution |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in families[:10]:
        lines.append(
            f"| {row['amino_acid']} | {row['positions']} | {row['baseline_jsd_to_true']:.6f} | "
            f"{row['finetuned_jsd_to_true']:.6f} | {row['weighted_jsd_to_true_regression_contribution']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The test set has already been inspected. Use validation, not this test report, for any subsequent hyperparameter or checkpoint selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sequence-csv", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    per_sequence_csv = args.per_sequence_csv.expanduser().resolve()
    reference_json = args.reference_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (per_sequence_csv, reference_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = (
        output_dir / "refined_biological_evaluation_summary.json",
        output_dir / "refined_biological_evaluation_report.md",
        output_dir / "per_sequence_refined_metrics.csv",
        output_dir / "synonymous_codon_family_attribution.csv",
    )
    if not args.force:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing refined outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads(reference_json.read_text(encoding="utf-8"))
    original_rows = read_rows(per_sequence_csv)
    rows, families, stop_codons = enrich_rows(original_rows, reference)
    if len(rows) != 594:
        raise ValueError(f"Expected the frozen 594-record test evaluation, found {len(rows)}")

    overall = {
        metric: paired_statistic(
            rows,
            metric,
            higher_is_better,
            args.bootstrap_samples,
            args.seed + index,
        )
        for index, (metric, higher_is_better) in enumerate(PAIRED_METRICS.items())
    }
    adjust_bh(overall)
    by_length: dict[str, dict[str, Any]] = {}
    for stratum_index, stratum in enumerate(("short", "medium", "long")):
        subset = [row for row in rows if row["length_bin"] == stratum]
        tests = {
            metric: paired_statistic(
                subset,
                metric,
                higher_is_better,
                args.bootstrap_samples,
                args.seed + 100 + stratum_index * 100 + metric_index,
            )
            for metric_index, (metric, higher_is_better) in enumerate(PAIRED_METRICS.items())
        }
        adjust_bh(tests)
        by_length[stratum] = tests

    family_attribution = aggregate_family_attribution(
        rows, reference, families, stop_codons
    )
    target_metrics = {
        metric for metrics in TARGET_CATEGORIES.values() for metric in metrics
    }
    stable_metrics = [
        metric
        for metric, result in overall.items()
        if metric in target_metrics and significant_improvement(result)
    ]
    stable_categories = [
        category
        for category, metrics in TARGET_CATEGORIES.items()
        if any(metric in stable_metrics for metric in metrics)
    ]
    overall_regressions = [
        metric
        for metric, result in overall.items()
        if metric in target_metrics and significant_regression(result)
    ]
    length_regressions = [
        f"{stratum}:{metric}"
        for stratum, tests in by_length.items()
        for metric, result in tests.items()
        if metric in target_metrics and significant_regression(result)
    ]
    translation_met = overall["translation_correct_rate"]["finetuned_mean"] >= 0.999
    validity_met = overall["sequence_valid_rate"]["finetuned_mean"] >= 0.999
    supports_claim = (
        translation_met
        and validity_met
        and len(stable_categories) >= 2
        and not overall_regressions
        and not length_regressions
    )
    summary = {
        "analysis_version": "2.0.0",
        "analysis_only": True,
        "records": len(rows),
        "inputs": {
            "per_sequence_csv": str(per_sequence_csv),
            "per_sequence_csv_sha256": sha256(per_sequence_csv),
            "reference_json": str(reference_json),
            "reference_json_sha256": sha256(reference_json),
        },
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "multiple_testing": "Benjamini-Hochberg within overall and separately within each length stratum",
        "synonymous_distribution_metric": {
            "jsd": "amino-acid-family conditional Jensen-Shannon distance, weighted by synonymous positions",
            "rscu_l1": "amino-acid-family conditional RSCU L1 distance, weighted by synonymous positions",
        },
        "overall_paired_statistics": overall,
        "length_stratified_paired_statistics": by_length,
        "decision": {
            "translation_requirement": ">= 0.999",
            "translation_requirement_met": translation_met,
            "validity_requirement": ">= 0.999",
            "validity_requirement_met": validity_met,
            "required_stable_improvement_categories": 2,
            "stable_improvement_categories": stable_categories,
            "stable_improvement_metrics": stable_metrics,
            "overall_significant_regressions": overall_regressions,
            "length_stratum_significant_regressions": length_regressions,
            "supports_claim": supports_claim,
            "interpretation": (
                "broad_support"
                if supports_claim
                else "mixed_or_length_dependent_effect; broad claim not supported"
            ),
        },
        "test_reuse_warning": (
            "The 594-record test set has been inspected. Use validation or a new external holdout, "
            "not this test set, for subsequent tuning and checkpoint selection."
        ),
    }

    compact_fields = [
        "idx",
        "protein_length",
        "length_bin",
        "true_synonymous_family_jsd_to_reference",
        "true_synonymous_family_rscu_l1_to_reference",
        *[
            f"{label}_{metric}"
            for label in LABELS
            for metric in (
                "synonymous_family_jsd_to_true",
                "synonymous_family_jsd_to_reference",
                "synonymous_family_rscu_l1_to_true",
                "synonymous_family_rscu_l1_to_reference",
            )
        ],
    ]
    write_csv(output_dir / "per_sequence_refined_metrics.csv", rows, compact_fields)
    write_csv(output_dir / "synonymous_codon_family_attribution.csv", family_attribution)
    (output_dir / "refined_biological_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output_dir / "refined_biological_evaluation_report.md",
        summary,
        family_attribution,
    )
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
