#!/usr/bin/env python3
"""Independently validate cluster isolation and upstream training exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

import yaml
from Bio.Data import CodonTable

try:
    from build_cluster_splits import (
        SPLIT_ORDER,
        cohort_definitions,
        merged_codon_tokens,
    )
except ModuleNotFoundError:  # Imported as scripts.validate_cluster_splits in tests.
    from scripts.build_cluster_splits import (
        SPLIT_ORDER,
        cohort_definitions,
        merged_codon_tokens,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("n_benthamiana_dataset_validation")
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


def validate_tokens(codons: str, genetic_code: int = 1) -> int:
    table = CodonTable.unambiguous_dna_by_id[genetic_code]
    tokens = codons.split()
    if len(tokens) < 2:
        raise ValueError("Training sequence must include a sense codon and stop token")
    if tokens[0] != "M_ATG":
        raise ValueError(f"Training sequence does not start with M_ATG: {tokens[0]}")
    stop_token = tokens[-1]
    if not stop_token.startswith("__") or stop_token[2:] not in table.stop_codons:
        raise ValueError(f"Invalid stop token: {stop_token}")
    for token in tokens[:-1]:
        if len(token) != 5 or token[1] != "_":
            raise ValueError(f"Invalid sense token: {token}")
        amino_acid, codon = token[0], token[2:]
        if table.forward_table.get(codon) != amino_acid:
            raise ValueError(f"Translation mismatch in token: {token}")
    return len(tokens)


def validate(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    paths = config["paths"]
    experiments_dir = resolve_path(paths["experiments_dir"])
    manifest_csv = resolve_path(paths["split_manifest_csv"])
    summary_json = resolve_path(paths["dataset_summary_json"])
    report_json = resolve_path(paths["validation_report_json"])
    logger = setup_logging(resolve_path(paths["validation_log"]))
    if report_json.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite validation report without --force: {report_json}"
        )
    for required in (experiments_dir, manifest_csv, summary_json):
        if not required.exists():
            raise FileNotFoundError(required)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    organism_id = int(config["training_export"]["organism_id"])
    cohorts = tuple(cohort_definitions(config))
    if tuple(summary["cohorts"]) != cohorts:
        raise ValueError("Configured cohorts do not match dataset summary")

    cluster_splits: dict[str, set[str]] = defaultdict(set)
    expected_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cluster_splits[row["cluster_id"]].add(row["split"])
            for cohort in cohorts:
                if row[cohort] == "true":
                    expected_sources[(cohort, row["split"])].add(row["source_id"])
            if row["all_clean_hc"] != "true":
                raise ValueError("Every manifest row must belong to all_clean_hc")
    leaking_clusters = {
        cluster_id: sorted(splits)
        for cluster_id, splits in cluster_splits.items()
        if len(splits) != 1
    }
    if leaking_clusters:
        raise ValueError(f"Clusters occur in multiple splits: {len(leaking_clusters)}")

    file_reports: dict[str, Any] = {}
    observed_counts: Counter[tuple[str, str]] = Counter()
    maximum_tokens = 0
    for cohort in cohorts:
        for split in SPLIT_ORDER:
            csv_path = experiments_dir / cohort / f"{split}.csv"
            jsonl_path = experiments_dir / cohort / f"{split}.jsonl"
            observed_sources: set[str] = set()
            with csv_path.open(newline="", encoding="utf-8") as csv_handle, jsonl_path.open(
                encoding="utf-8"
            ) as json_handle:
                reader = csv.DictReader(csv_handle)
                for expected_index, pair in enumerate(
                    zip_longest(reader, json_handle), start=0
                ):
                    row, line = pair
                    if row is None or line is None:
                        raise ValueError(f"CSV/JSONL length mismatch: {cohort}/{split}")
                    record = json.loads(line)
                    if set(record) != {"idx", "codons", "organism"}:
                        raise ValueError(f"Unexpected JSONL fields: {cohort}/{split}")
                    if record["idx"] != expected_index:
                        raise ValueError(f"Non-contiguous idx: {cohort}/{split}")
                    if record["organism"] != organism_id:
                        raise ValueError(f"Wrong organism id: {cohort}/{split}")
                    expected_codons = merged_codon_tokens(row["protein"], row["dna"])
                    if record["codons"] != expected_codons:
                        raise ValueError(f"CSV/JSONL token mismatch: {row['source_id']}")
                    maximum_tokens = max(maximum_tokens, validate_tokens(record["codons"]))
                    if row["source_id"] in observed_sources:
                        raise ValueError(f"Duplicate source in export: {row['source_id']}")
                    observed_sources.add(row["source_id"])
                    observed_counts[(cohort, split)] += 1
            if observed_sources != expected_sources[(cohort, split)]:
                raise ValueError(f"Manifest/export source mismatch: {cohort}/{split}")
            expected_count = int(summary["records"][cohort][split])
            if observed_counts[(cohort, split)] != expected_count:
                raise ValueError(f"Summary/export count mismatch: {cohort}/{split}")
            file_reports[f"{cohort}/{split}"] = {
                "records": observed_counts[(cohort, split)],
                "csv_sha256": sha256(csv_path),
                "jsonl_sha256": sha256(jsonl_path),
            }
            logger.info(
                "Validated %s/%s: %d records",
                cohort,
                split,
                observed_counts[(cohort, split)],
            )

    report = {
        "pipeline_version": str(config.get("pipeline_version", "unknown")),
        "seed": int(config.get("seed", 0)),
        "passed": True,
        "checks": {
            "clusters_present_in_exactly_one_split": True,
            "manifest_matches_all_exports": True,
            "csv_jsonl_rows_match": True,
            "jsonl_fields_and_indices_valid": True,
            "codon_translation_valid": True,
            "organism_id_valid": True,
        },
        "clusters": len(cluster_splits),
        "cohorts": list(cohorts),
        "maximum_training_tokens": maximum_tokens,
        "files": file_reports,
        "inputs": {
            "manifest_sha256": sha256(manifest_csv),
            "dataset_summary_sha256": sha256(summary_json),
        },
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Independent validation passed")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "n_benthamiana_dataset.yaml",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    if args.check_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    report = validate(config, force=args.force)
    print(json.dumps({"passed": report["passed"], "files": len(report["files"])}))


if __name__ == "__main__":
    main()
