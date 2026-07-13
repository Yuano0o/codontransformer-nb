#!/usr/bin/env python3
"""Compute reproducible CSI/CAI, GC, and codon-use metrics for clean CDSs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from Bio.Data import CodonTable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "source_id",
    "dna",
    "protein",
    "organism",
    "source_organism",
    "cds_length",
    "protein_length",
    "gc_content",
}
METRIC_COLUMNS = (
    "csi",
    "cai",
    "gc1_content",
    "gc2_content",
    "gc3_content",
    "gc3s_content",
    "sense_codon_count",
    "rare_codon_fraction",
    "optimal_codon_fraction",
    "mean_codon_weight",
    "synonymous_codon_entropy",
    "codon_usage_l1",
    "terminal_stop_codon",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("paths", "metrics", "cohorts"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing YAML mapping: {section}")
    return config


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("n_benthamiana_metrics")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing accepted CSV columns: {sorted(missing)}")
        yield from reader


def genetic_code_families(
    genetic_code: int,
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], frozenset[str]]:
    table = CodonTable.unambiguous_dna_by_id[genetic_code]
    families: dict[str, list[str]] = {}
    for codon, amino_acid in table.forward_table.items():
        families.setdefault(amino_acid, []).append(codon)
    normalized = {
        amino_acid: tuple(sorted(codons))
        for amino_acid, codons in sorted(families.items())
    }
    sense_codons = tuple(sorted(table.forward_table))
    return normalized, sense_codons, frozenset(table.stop_codons)


def coding_codons(dna: str, stop_codons: frozenset[str]) -> list[str]:
    dna = dna.upper()
    codons = [dna[index : index + 3] for index in range(0, len(dna), 3)]
    if not codons or codons[-1] not in stop_codons:
        raise ValueError("CDS must end with a standard stop codon")
    return codons[:-1]


def count_reference_codons(
    rows: Iterable[dict[str, str]], sense_codons: tuple[str, ...], stop_codons: frozenset[str]
) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter({codon: 0 for codon in sense_codons})
    record_count = 0
    for row in rows:
        counts.update(coding_codons(row["dna"], stop_codons))
        record_count += 1
    return counts, record_count


def relative_adaptiveness(
    counts: Counter[str],
    families: dict[str, tuple[str, ...]],
    pseudocount: float,
) -> dict[str, float]:
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")
    weights: dict[str, float] = {}
    for codons in families.values():
        adjusted = {codon: counts[codon] + pseudocount for codon in codons}
        maximum = max(adjusted.values())
        for codon in codons:
            weights[codon] = adjusted[codon] / maximum if maximum else 1.0
    return weights


def geometric_codon_score(
    codons: Iterable[str],
    weights: dict[str, float],
    families: dict[str, tuple[str, ...]],
) -> float:
    informative = {
        codon for family in families.values() if len(family) > 1 for codon in family
    }
    log_weights: list[float] = []
    for codon in codons:
        if codon not in informative:
            continue
        weight = weights[codon]
        if weight <= 0:
            return 0.0
        log_weights.append(math.log(weight))
    return math.exp(sum(log_weights) / len(log_weights)) if log_weights else math.nan


def quantile(values: Iterable[float], probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must be between zero and one")
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        raise ValueError("Cannot calculate a quantile from no finite values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def gc_fraction(sequence: str) -> float:
    if not sequence:
        return math.nan
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def synonymous_entropy(
    codon_counts: Counter[str], families: dict[str, tuple[str, ...]]
) -> float:
    weighted_entropy = 0.0
    observations = 0
    for codons in families.values():
        if len(codons) == 1:
            continue
        family_total = sum(codon_counts[codon] for codon in codons)
        if family_total == 0:
            continue
        entropy = -sum(
            (count / family_total) * math.log(count / family_total)
            for codon in codons
            if (count := codon_counts[codon]) > 0
        )
        weighted_entropy += (entropy / math.log(len(codons))) * family_total
        observations += family_total
    return weighted_entropy / observations if observations else math.nan


def row_metrics(
    dna: str,
    csi_weights: dict[str, float],
    cai_weights: dict[str, float],
    global_frequencies: dict[str, float],
    families: dict[str, tuple[str, ...]],
    stop_codons: frozenset[str],
    rare_threshold: float,
) -> dict[str, str]:
    dna = dna.upper()
    codons = coding_codons(dna, stop_codons)
    counts = Counter(codons)
    total = len(codons)
    synonymous_positions = [
        codon
        for family in families.values()
        if len(family) > 1
        for codon in family
    ]
    synonymous_set = set(synonymous_positions)
    informative = [codon for codon in codons if codon in synonymous_set]
    frequencies = {codon: counts[codon] / total for codon in global_frequencies}
    gc3s_codons = [codon for codon in codons if codon in synonymous_set]
    informative_count = len(informative)
    rare_fraction = (
        sum(csi_weights[codon] < rare_threshold for codon in informative)
        / informative_count
        if informative_count
        else math.nan
    )
    optimal_fraction = (
        sum(math.isclose(csi_weights[codon], 1.0) for codon in informative)
        / informative_count
        if informative_count
        else math.nan
    )
    mean_weight = (
        sum(csi_weights[codon] for codon in informative) / informative_count
        if informative_count
        else math.nan
    )
    result = {
        "csi": f"{geometric_codon_score(codons, csi_weights, families):.8f}",
        "cai": f"{geometric_codon_score(codons, cai_weights, families):.8f}",
        "gc1_content": f"{gc_fraction(''.join(codon[0] for codon in codons)):.8f}",
        "gc2_content": f"{gc_fraction(''.join(codon[1] for codon in codons)):.8f}",
        "gc3_content": f"{gc_fraction(''.join(codon[2] for codon in codons)):.8f}",
        "gc3s_content": f"{gc_fraction(''.join(codon[2] for codon in gc3s_codons)):.8f}",
        "sense_codon_count": str(total),
        "rare_codon_fraction": f"{rare_fraction:.8f}",
        "optimal_codon_fraction": f"{optimal_fraction:.8f}",
        "mean_codon_weight": f"{mean_weight:.8f}",
        "synonymous_codon_entropy": f"{synonymous_entropy(counts, families):.8f}",
        "codon_usage_l1": f"{sum(abs(frequencies[c] - global_frequencies[c]) for c in global_frequencies):.8f}",
        "terminal_stop_codon": dna[-3:],
    }
    return result


def reference_payload(
    counts: Counter[str],
    weights: dict[str, float],
    families: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    total = sum(counts.values())
    return {
        "sense_codon_count": total,
        "counts": {codon: counts[codon] for codon in sorted(counts)},
        "frequencies": {
            codon: counts[codon] / total for codon in sorted(counts)
        },
        "relative_adaptiveness": {
            codon: weights[codon] for codon in sorted(weights)
        },
        "synonymous_families": {
            amino_acid: list(codons) for amino_acid, codons in families.items()
        },
    }


def compute(config: dict[str, Any], force: bool = False) -> None:
    paths = config["paths"]
    settings = config["metrics"]
    input_csv = resolve_path(paths["accepted_csv"])
    output_csv = resolve_path(paths["metrics_csv"])
    reference_json = resolve_path(paths["codon_reference_json"])
    summary_json = resolve_path(paths["metrics_summary_json"])
    logger = setup_logging(resolve_path(paths["metrics_log"]))
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    for generated in (output_csv, reference_json, summary_json):
        if generated.exists() and not force:
            raise FileExistsError(
                f"Refusing to overwrite generated output without --force: {generated}"
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    genetic_code = int(settings.get("genetic_code", 1))
    pseudocount = float(settings.get("pseudocount", 0.0))
    rare_threshold = float(settings.get("rare_codon_weight_threshold", 0.3))
    families, sense_codons, stop_codons = genetic_code_families(genetic_code)

    logger.info("Counting corpus codons in %s", input_csv)
    corpus_counts, record_count = count_reference_codons(
        iter_csv(input_csv), sense_codons, stop_codons
    )
    csi_weights = relative_adaptiveness(corpus_counts, families, pseudocount)
    csi_by_id: dict[str, float] = {}
    for row in iter_csv(input_csv):
        csi_by_id[row["source_id"]] = geometric_codon_score(
            coding_codons(row["dna"], stop_codons), csi_weights, families
        )
    if len(csi_by_id) != record_count:
        raise ValueError("source_id values are not unique")

    cai_settings = settings.get("cai_reference", {})
    if cai_settings.get("mode") != "top_csi_quantile":
        raise ValueError("Only cai_reference.mode=top_csi_quantile is currently supported")
    cai_quantile = float(cai_settings.get("quantile", 0.9))
    cai_threshold = quantile(csi_by_id.values(), cai_quantile)
    logger.info(
        "Building CAI proxy weights from CSI quantile %.3f (threshold %.8f)",
        cai_quantile,
        cai_threshold,
    )
    cai_counts: Counter[str] = Counter({codon: 0 for codon in sense_codons})
    cai_reference_records = 0
    for row in iter_csv(input_csv):
        if csi_by_id[row["source_id"]] >= cai_threshold:
            cai_counts.update(coding_codons(row["dna"], stop_codons))
            cai_reference_records += 1
    cai_weights = relative_adaptiveness(cai_counts, families, pseudocount)
    cohort_thresholds = {
        cohort: {
            "quantile": float(definition["quantile"]),
            "csi_threshold": quantile(
                csi_by_id.values(), float(definition["quantile"])
            ),
        }
        for cohort, definition in config["cohorts"].items()
        if definition.get("selection") == "csi_quantile"
    }
    total_codons = sum(corpus_counts.values())
    global_frequencies = {
        codon: corpus_counts[codon] / total_codons for codon in sense_codons
    }

    logger.info("Writing metrics for %d records to %s", record_count, output_csv)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    csi_values: list[float] = []
    cai_values: list[float] = []
    with input_csv.open(newline="", encoding="utf-8") as source, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=[*(reader.fieldnames or []), *METRIC_COLUMNS])
        writer.writeheader()
        for index, row in enumerate(reader, start=1):
            metrics = row_metrics(
                row["dna"],
                csi_weights,
                cai_weights,
                global_frequencies,
                families,
                stop_codons,
                rare_threshold,
            )
            csi_values.append(float(metrics["csi"]))
            cai_values.append(float(metrics["cai"]))
            writer.writerow({**row, **metrics})
            if index % 10000 == 0:
                logger.info("Computed metrics for %d/%d records", index, record_count)
    os.replace(temporary, output_csv)

    references = {
        "pipeline_version": str(config.get("pipeline_version", "unknown")),
        "genetic_code": genetic_code,
        "pseudocount": pseudocount,
        "csi_reference": {
            "description": "all accepted high-confidence N. benthamiana CDS",
            "records": record_count,
            **reference_payload(corpus_counts, csi_weights, families),
        },
        "cai_reference": {
            "description": "top-CSI proxy reference; not expression-grounded CAI",
            "selection_quantile": cai_quantile,
            "selection_threshold": cai_threshold,
            "records": cai_reference_records,
            **reference_payload(cai_counts, cai_weights, families),
        },
    }
    reference_json.write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")
    summary = {
        "pipeline_version": str(config.get("pipeline_version", "unknown")),
        "seed": int(config.get("seed", 0)),
        "input": {
            "path": str(input_csv.relative_to(PROJECT_ROOT)),
            "sha256": sha256(input_csv),
            "records": record_count,
        },
        "output": {
            "path": str(output_csv.relative_to(PROJECT_ROOT)),
            "sha256": sha256(output_csv),
        },
        "definitions": {
            "csi": "CAI-form geometric mean using all accepted CDS as reference",
            "cai": "CAI-form geometric mean using the configured top-CSI proxy reference",
            "gc_positions": "sense codons only; terminal stop excluded",
            "gc3s": "third-position GC among synonymous codon families; Met and Trp excluded",
            "codon_usage_l1": "L1 distance from corpus-wide sense-codon frequencies",
        },
        "thresholds": {
            "cai_reference_quantile": cai_quantile,
            "cai_reference_csi_threshold": cai_threshold,
            "cohort_csi_thresholds": cohort_thresholds,
            "rare_codon_weight_threshold": rare_threshold,
        },
        "summaries": {
            "csi": {"min": min(csi_values), "max": max(csi_values), "mean": sum(csi_values) / len(csi_values)},
            "cai": {"min": min(cai_values), "max": max(cai_values), "mean": sum(cai_values) / len(cai_values)},
        },
        "cai_reference_records": cai_reference_records,
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Metrics stage complete; cohort thresholds: %s", cohort_thresholds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "n_benthamiana_dataset.yaml",
    )
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite only this stage's generated metrics outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    if args.check_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    compute(config, force=args.force)


if __name__ == "__main__":
    main()
