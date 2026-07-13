#!/usr/bin/env python3
"""Create leak-resistant cluster splits and upstream-compatible training JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ORDER = ("train", "validation", "test")
REQUIRED_COHORTS = ("all_clean_hc", "csi_top10_hc", "csi_top25_hc")


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("paths", "split", "training_export", "cohorts"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing YAML mapping: {section}")
    return config


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("n_benthamiana_dataset_build")
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


def read_membership(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    source_to_cluster: dict[str, str] = {}
    declared_sizes: dict[str, int] = {}
    actual_sizes: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_id", "cluster_id", "cluster_size"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing cluster membership columns: {sorted(missing)}")
        for row in reader:
            source_id = row["source_id"]
            if source_id in source_to_cluster:
                raise ValueError(f"Duplicate cluster member: {source_id}")
            cluster_id = row["cluster_id"]
            source_to_cluster[source_id] = cluster_id
            actual_sizes[cluster_id] += 1
            size = int(row["cluster_size"])
            if cluster_id in declared_sizes and declared_sizes[cluster_id] != size:
                raise ValueError(f"Inconsistent declared size for {cluster_id}")
            declared_sizes[cluster_id] = size
    if dict(actual_sizes) != declared_sizes:
        raise ValueError("Declared cluster sizes do not match membership counts")
    return source_to_cluster, declared_sizes


def validate_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != set(SPLIT_ORDER):
        raise ValueError(f"Split keys must be exactly {SPLIT_ORDER}")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to one")


def cohort_definitions(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = config["cohorts"]
    if tuple(definitions) != REQUIRED_COHORTS:
        raise ValueError(
            f"Cohorts must be defined in this order: {REQUIRED_COHORTS}"
        )
    for cohort, definition in definitions.items():
        selection = definition.get("selection")
        if cohort == "all_clean_hc" and selection != "all":
            raise ValueError("all_clean_hc must use selection=all")
        if cohort != "all_clean_hc":
            if selection != "csi_quantile":
                raise ValueError(f"{cohort} must use selection=csi_quantile")
            probability = float(definition.get("quantile", -1))
            if not 0 < probability < 1:
                raise ValueError(f"Invalid CSI quantile for {cohort}: {probability}")
    return definitions


def linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate cohort threshold from no CSI values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def calculate_cohort_thresholds(
    metrics_csv: Path, definitions: dict[str, dict[str, Any]]
) -> dict[str, float | None]:
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "csi" not in (reader.fieldnames or []):
            raise ValueError("Metrics CSV is missing csi")
        csi_values = [float(row["csi"]) for row in reader]
    return {
        cohort: (
            None
            if definition["selection"] == "all"
            else linear_quantile(csi_values, float(definition["quantile"]))
        )
        for cohort, definition in definitions.items()
    }


def assign_clusters(
    cluster_sizes: dict[str, int], ratios: dict[str, float], seed: int
) -> tuple[dict[str, str], dict[str, int]]:
    validate_ratios(ratios)
    total = sum(cluster_sizes.values())
    targets = {split: ratios[split] * total for split in SPLIT_ORDER}
    counts = {split: 0 for split in SPLIT_ORDER}
    rng = random.Random(seed)
    ordered_clusters = sorted(cluster_sizes)
    rng.shuffle(ordered_clusters)
    assignments: dict[str, str] = {}
    for cluster_id in ordered_clusters:
        size = cluster_sizes[cluster_id]
        split = max(
            SPLIT_ORDER,
            key=lambda name: (
                targets[name] - counts[name],
                -SPLIT_ORDER.index(name),
            ),
        )
        assignments[cluster_id] = split
        counts[split] += size
    return assignments, counts


def merged_codon_tokens(protein: str, dna: str) -> str:
    protein = protein.upper()
    if protein.endswith(("*", "_")):
        protein = protein[:-1]
    dna = dna.upper()
    sense_codons = [dna[index : index + 3] for index in range(0, len(dna) - 3, 3)]
    if len(protein) != len(sense_codons):
        raise ValueError(
            f"Protein/CDS length mismatch during export: {len(protein)} != {len(sense_codons)}"
        )
    tokens = [f"{amino_acid}_{codon}" for amino_acid, codon in zip(protein, sense_codons)]
    tokens.append(f"__{dna[-3:]}")
    return " ".join(tokens)


def open_outputs(
    stack: ExitStack,
    output_dir: Path,
    csv_fieldnames: list[str],
    cohorts: tuple[str, ...],
) -> tuple[
    dict[tuple[str, str], csv.DictWriter],
    dict[tuple[str, str], TextIO],
]:
    csv_writers: dict[tuple[str, str], csv.DictWriter] = {}
    json_handles: dict[tuple[str, str], TextIO] = {}
    for cohort in cohorts:
        cohort_dir = output_dir / cohort
        cohort_dir.mkdir(parents=True, exist_ok=False)
        for split in SPLIT_ORDER:
            csv_handle = stack.enter_context(
                (cohort_dir / f"{split}.csv").open("w", newline="", encoding="utf-8")
            )
            writer = csv.DictWriter(csv_handle, fieldnames=csv_fieldnames)
            writer.writeheader()
            csv_writers[(cohort, split)] = writer
            json_handles[(cohort, split)] = stack.enter_context(
                (cohort_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            )
    return csv_writers, json_handles


def build(config: dict[str, Any]) -> None:
    paths = config["paths"]
    metrics_csv = resolve_path(paths["metrics_csv"])
    membership_csv = resolve_path(paths["cluster_membership_csv"])
    experiments_dir = resolve_path(paths["experiments_dir"])
    manifest_csv = resolve_path(paths["split_manifest_csv"])
    summary_json = resolve_path(paths["dataset_summary_json"])
    logger = setup_logging(resolve_path(paths["dataset_log"]))
    for required in (metrics_csv, membership_csv):
        if not required.is_file():
            raise FileNotFoundError(required)
    for generated in (experiments_dir, manifest_csv, summary_json):
        if generated.exists():
            raise FileExistsError(f"Refusing to overwrite generated output: {generated}")
    temporary_experiments = experiments_dir.with_name(experiments_dir.name + ".tmp")
    temporary_manifest = manifest_csv.with_suffix(manifest_csv.suffix + ".tmp")
    temporary_summary = summary_json.with_suffix(summary_json.suffix + ".tmp")
    for temporary in (temporary_experiments, temporary_manifest, temporary_summary):
        if temporary.exists():
            raise FileExistsError(f"Stale temporary output must be removed: {temporary}")

    source_to_cluster, cluster_sizes = read_membership(membership_csv)
    definitions = cohort_definitions(config)
    cohorts = tuple(definitions)
    cohort_thresholds = calculate_cohort_thresholds(metrics_csv, definitions)
    ratios = {split: float(config["split"][split]) for split in SPLIT_ORDER}
    seed = int(config.get("seed", 0))
    cluster_assignments, expected_split_counts = assign_clusters(
        cluster_sizes, ratios, seed
    )
    organism_id = int(config["training_export"]["organism_id"])
    organism_name = str(config["training_export"]["organism_name"])

    experiments_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    record_counts: Counter[tuple[str, str]] = Counter()
    split_clusters: dict[str, set[str]] = {split: set() for split in SPLIT_ORDER}
    seen_sources: set[str] = set()
    with metrics_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required_columns = {"source_id", "dna", "protein", "organism", "csi", "cai"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing metrics columns: {sorted(missing)}")
        csv_fieldnames = [*(reader.fieldnames or []), "cluster_id", "cluster_size", "split"]
        with ExitStack() as stack:
            csv_writers, json_handles = open_outputs(
                stack, temporary_experiments, csv_fieldnames, cohorts
            )
            manifest_handle = stack.enter_context(
                temporary_manifest.open("w", newline="", encoding="utf-8")
            )
            manifest_writer = csv.DictWriter(
                manifest_handle,
                fieldnames=(
                    "source_id",
                    "cluster_id",
                    "cluster_size",
                    "split",
                    "csi",
                    "cai",
                    *cohorts,
                ),
            )
            manifest_writer.writeheader()
            for index, row in enumerate(reader, start=1):
                source_id = row["source_id"]
                if source_id in seen_sources:
                    raise ValueError(f"Duplicate metrics source_id: {source_id}")
                seen_sources.add(source_id)
                if source_id not in source_to_cluster:
                    raise ValueError(f"No cluster for source_id: {source_id}")
                if row["organism"] != organism_name:
                    raise ValueError(
                        f"Unexpected organism for {source_id}: {row['organism']}"
                    )
                cluster_id = source_to_cluster[source_id]
                split = cluster_assignments[cluster_id]
                split_clusters[split].add(cluster_id)
                csi = float(row["csi"])
                cohort_membership = {
                    cohort: (
                        threshold is None or csi >= threshold
                    )
                    for cohort, threshold in cohort_thresholds.items()
                }
                augmented = {
                    **row,
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_sizes[cluster_id],
                    "split": split,
                }
                tokens = merged_codon_tokens(row["protein"], row["dna"])
                for cohort in (
                    name for name in cohorts if cohort_membership[name]
                ):
                    csv_writers[(cohort, split)].writerow(augmented)
                    output_index = record_counts[(cohort, split)]
                    json_handles[(cohort, split)].write(
                        json.dumps(
                            {"idx": output_index, "codons": tokens, "organism": organism_id},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    record_counts[(cohort, split)] += 1
                manifest_writer.writerow(
                    {
                        "source_id": source_id,
                        "cluster_id": cluster_id,
                        "cluster_size": cluster_sizes[cluster_id],
                        "split": split,
                        "csi": row["csi"],
                        "cai": row["cai"],
                        **{
                            cohort: str(cohort_membership[cohort]).lower()
                            for cohort in cohorts
                        },
                    }
                )
                if index % 10000 == 0:
                    logger.info("Exported %d records", index)

    if seen_sources != set(source_to_cluster):
        raise ValueError("Metrics and cluster membership source IDs differ")
    if sum(expected_split_counts.values()) != len(seen_sources):
        raise ValueError("Split assignment count mismatch")
    cluster_intersections = {
        f"{left}__{right}": len(split_clusters[left] & split_clusters[right])
        for left_index, left in enumerate(SPLIT_ORDER)
        for right in SPLIT_ORDER[left_index + 1 :]
    }
    if any(cluster_intersections.values()):
        raise RuntimeError(f"Cluster leakage detected: {cluster_intersections}")
    summary = {
        "pipeline_version": str(config.get("pipeline_version", "unknown")),
        "seed": seed,
        "inputs": {
            "metrics_csv_sha256": sha256(metrics_csv),
            "cluster_membership_sha256": sha256(membership_csv),
        },
        "split_ratios": ratios,
        "split_algorithm": "seeded cluster shuffle with largest-record-deficit assignment",
        "cluster_parameters": config.get("clustering", {}),
        "cohorts": {
            cohort: {
                **definitions[cohort],
                "csi_threshold": cohort_thresholds[cohort],
            }
            for cohort in cohorts
        },
        "organism": {"name": organism_name, "id": organism_id},
        "records": {
            cohort: {split: record_counts[(cohort, split)] for split in SPLIT_ORDER}
            for cohort in cohorts
        },
        "clusters": {
            "total": len(cluster_sizes),
            "by_split": {split: len(split_clusters[split]) for split in SPLIT_ORDER},
            "cross_split_intersections": cluster_intersections,
            "leakage_check_passed": not any(cluster_intersections.values()),
        },
        "outputs": {
            "split_manifest_sha256": sha256(temporary_manifest),
            "training_format": {
                "fields": ["idx", "codons", "organism"],
                "description": "one compact JSON object per line; matches upstream finetune.py",
            },
        },
    }
    temporary_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_experiments, experiments_dir)
    os.replace(temporary_manifest, manifest_csv)
    os.replace(temporary_summary, summary_json)
    logger.info("Dataset build complete; no cross-split cluster leakage detected")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "n_benthamiana_dataset.yaml",
    )
    parser.add_argument("--check-config", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    if args.check_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    build(config)


if __name__ == "__main__":
    main()
