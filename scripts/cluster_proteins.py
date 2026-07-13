#!/usr/bin/env python3
"""Cluster accepted proteins with MMseqs2 and emit stable cluster membership."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("paths", "clustering"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing YAML mapping: {section}")
    return config


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("n_benthamiana_clustering")
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


def write_protein_fasta(metrics_csv: Path, output_fasta: Path) -> set[str]:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    identifiers: set[str] = set()
    temporary = output_fasta.with_suffix(output_fasta.suffix + ".tmp")
    with metrics_csv.open(newline="", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        required = {"source_id", "protein"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing metrics CSV columns: {sorted(missing)}")
        for row in reader:
            identifier = row["source_id"]
            if not identifier or any(character.isspace() for character in identifier):
                raise ValueError(f"Invalid FASTA identifier: {identifier!r}")
            if identifier in identifiers:
                raise ValueError(f"Duplicate source_id: {identifier}")
            identifiers.add(identifier)
            protein = row["protein"].upper()
            if protein.endswith(("*", "_")):
                protein = protein[:-1]
            if not protein:
                raise ValueError(f"Empty protein sequence: {identifier}")
            destination.write(f">{identifier}\n")
            for offset in range(0, len(protein), 80):
                destination.write(protein[offset : offset + 80] + "\n")
    os.replace(temporary, output_fasta)
    return identifiers


def run_logged(command: list[str], logger: logging.Logger) -> None:
    logger.info("Running: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("MMseqs2: %s", line.rstrip())
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def mmseqs_command(
    executable: str,
    fasta: Path,
    prefix: Path,
    temporary_dir: Path,
    settings: dict[str, Any],
) -> list[str]:
    return [
        executable,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(temporary_dir),
        "--min-seq-id",
        str(settings["min_sequence_identity"]),
        "-c",
        str(settings["coverage"]),
        "--cov-mode",
        str(settings["coverage_mode"]),
        "--cluster-mode",
        str(settings["cluster_mode"]),
        "--max-seqs",
        str(settings["max_sequences"]),
        "-s",
        str(settings["sensitivity"]),
        "--threads",
        str(settings["threads"]),
    ]


def parse_mmseqs_clusters(
    cluster_tsv: Path, expected_ids: set[str]
) -> tuple[list[dict[str, str | int]], dict[str, list[str]]]:
    representative_to_members: dict[str, list[str]] = defaultdict(list)
    seen_members: set[str] = set()
    with cluster_tsv.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid MMseqs2 TSV line {line_number}")
            representative, member = parts
            if member in seen_members:
                raise ValueError(f"MMseqs2 member appears more than once: {member}")
            representative_to_members[representative].append(member)
            seen_members.add(member)
    if seen_members != expected_ids:
        missing = sorted(expected_ids - seen_members)[:10]
        unexpected = sorted(seen_members - expected_ids)[:10]
        raise ValueError(
            f"MMseqs2 membership mismatch; missing={missing}, unexpected={unexpected}"
        )

    rows: list[dict[str, str | int]] = []
    clusters_by_id: dict[str, list[str]] = {}
    for representative, members in representative_to_members.items():
        members = sorted(members)
        fingerprint = hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        cluster_id = f"cluster_{fingerprint}"
        if cluster_id in clusters_by_id:
            raise ValueError(f"Cluster hash collision: {cluster_id}")
        clusters_by_id[cluster_id] = members
        for member in members:
            rows.append(
                {
                    "source_id": member,
                    "cluster_id": cluster_id,
                    "representative_id": representative,
                    "cluster_size": len(members),
                }
            )
    rows.sort(key=lambda row: str(row["source_id"]))
    return rows, clusters_by_id


def cluster(config: dict[str, Any], external_tsv: Path | None = None) -> None:
    paths = config["paths"]
    settings = config["clustering"]
    metrics_csv = resolve_path(paths["metrics_csv"])
    protein_fasta = resolve_path(paths["protein_fasta"])
    prefix = resolve_path(paths["mmseqs_prefix"])
    temporary_dir = resolve_path(paths["mmseqs_tmp_dir"])
    membership_csv = resolve_path(paths["cluster_membership_csv"])
    summary_json = resolve_path(paths["cluster_summary_json"])
    logger = setup_logging(resolve_path(paths["clustering_log"]))
    if not metrics_csv.is_file():
        raise FileNotFoundError(metrics_csv)
    for generated in (protein_fasta, membership_csv, summary_json):
        if generated.exists():
            raise FileExistsError(f"Refusing to overwrite generated output: {generated}")

    logger.info("Exporting protein FASTA from %s", metrics_csv)
    expected_ids = write_protein_fasta(metrics_csv, protein_fasta)
    if external_tsv is None:
        executable_name = str(settings.get("executable", "mmseqs"))
        executable = shutil.which(executable_name)
        if executable is None:
            raise FileNotFoundError(
                f"MMseqs2 executable not found: {executable_name}. Install MMseqs2 first."
            )
        prefix.parent.mkdir(parents=True, exist_ok=True)
        command = mmseqs_command(
            executable, protein_fasta, prefix, temporary_dir, settings
        )
        run_logged(command, logger)
        cluster_tsv = Path(f"{prefix}_cluster.tsv")
        version = subprocess.run(
            [executable, "version"], capture_output=True, text=True, check=True
        ).stdout.strip()
    else:
        cluster_tsv = external_tsv.resolve()
        command = ["external-cluster-tsv", str(cluster_tsv)]
        version = "external"
    if not cluster_tsv.is_file():
        raise FileNotFoundError(cluster_tsv)

    rows, clusters = parse_mmseqs_clusters(cluster_tsv, expected_ids)
    membership_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_membership = membership_csv.with_suffix(membership_csv.suffix + ".tmp")
    with temporary_membership.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_id", "cluster_id", "representative_id", "cluster_size"),
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_membership, membership_csv)
    sizes = sorted((len(members) for members in clusters.values()))
    summary = {
        "pipeline_version": str(config.get("pipeline_version", "unknown")),
        "seed": int(config.get("seed", 0)),
        "input_metrics_sha256": sha256(metrics_csv),
        "protein_fasta_sha256": sha256(protein_fasta),
        "membership_sha256": sha256(membership_csv),
        "mmseqs_version": version,
        "command": command,
        "parameters": settings,
        "records": len(rows),
        "clusters": len(clusters),
        "singleton_clusters": sum(size == 1 for size in sizes),
        "cluster_size": {
            "min": sizes[0],
            "median": median(sizes),
            "max": sizes[-1],
        },
        "all_records_assigned_once": len(rows) == len(expected_ids),
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Clustering complete: %d proteins in %d clusters", len(rows), len(clusters))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "n_benthamiana_dataset.yaml",
    )
    parser.add_argument(
        "--clusters-tsv",
        type=Path,
        help="Use an existing representative/member TSV instead of running MMseqs2.",
    )
    parser.add_argument("--check-config", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config.resolve())
    if args.check_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    cluster(config, args.clusters_tsv)


if __name__ == "__main__":
    main()
