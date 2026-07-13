#!/usr/bin/env python3
"""Create deterministic, auditable N. benthamiana stage-one QC outputs."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import logging
import os
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "n_benthamiana_preprocess.json"

ACCEPTED_COLUMNS = [
    "source_id",
    "dna",
    "protein",
    "organism",
    "source_organism",
    "cds_length",
    "protein_length",
    "gc_content",
]
REJECTED_COLUMNS = ACCEPTED_COLUMNS + ["rejection_reasons"]

STANDARD_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass(frozen=True)
class FastaRecord:
    source_id: str
    sequence: str


@dataclass(frozen=True)
class FilterRules:
    allowed_dna_alphabet: frozenset[str]
    required_start_codon: str
    standard_stop_codons: frozenset[str]
    translation_table: int
    max_protein_length: int


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def iter_fasta(path: Path) -> Iterator[FastaRecord]:
    """Yield FASTA records without changing sequence characters."""
    source_id: str | None = None
    sequence_parts: list[str] = []
    with open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                if source_id is not None:
                    yield FastaRecord(source_id, "".join(sequence_parts))
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header at {path}:{line_number}")
                source_id = header.split()[0]
                sequence_parts = []
            elif line:
                if source_id is None:
                    raise ValueError(
                        f"Sequence encountered before first header at {path}:{line_number}"
                    )
                sequence_parts.append(line.strip())
    if source_id is not None:
        yield FastaRecord(source_id, "".join(sequence_parts))


def load_fasta(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    records: dict[str, str] = {}
    for record in iter_fasta(path):
        if record.source_id in records:
            raise ValueError(f"Duplicate FASTA ID in {path}: {record.source_id}")
        order.append(record.source_id)
        records[record.source_id] = record.sequence
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return order, records


def translate_standard(dna: str) -> str:
    """Translate complete codons; ambiguous/invalid codons become X."""
    return "".join(
        STANDARD_CODE.get(dna[index:index + 3], "X")
        for index in range(0, len(dna) - 2, 3)
    )


def remove_one_terminal_stop(sequence: str) -> str:
    return sequence[:-1] if sequence.endswith("*") else sequence


def gc_content(dna: str) -> float:
    if not dna:
        return 0.0
    normalized = dna.upper()
    return (normalized.count("G") + normalized.count("C")) / len(normalized)


def assess_pair(dna: str, protein: str, rules: FilterRules) -> list[str]:
    normalized_dna = dna.upper()
    normalized_protein = protein.upper()
    reasons: list[str] = []
    if set(normalized_dna) - rules.allowed_dna_alphabet:
        reasons.append("non_atcg")
    if len(normalized_dna) % 3 != 0:
        reasons.append("cds_length_not_multiple_of_3")

    translated = remove_one_terminal_stop(translate_standard(normalized_dna))
    expected = remove_one_terminal_stop(normalized_protein)
    if translated != expected:
        reasons.append("translation_mismatch")
    if not normalized_dna.startswith(rules.required_start_codon):
        reasons.append("missing_atg_start")
    if (
        len(normalized_dna) < 3
        or normalized_dna[-3:] not in rules.standard_stop_codons
    ):
        reasons.append("missing_standard_stop")
    if len(expected) > rules.max_protein_length:
        reasons.append("protein_too_long")
    return reasons


def make_row(
    source_id: str,
    dna: str,
    protein: str,
    organism: str,
    source_organism: str,
) -> dict[str, str | int]:
    return {
        "source_id": source_id,
        "dna": dna,
        "protein": protein,
        "organism": organism,
        "source_organism": source_organism,
        "cds_length": len(dna),
        "protein_length": len(remove_one_terminal_stop(protein)),
        "gc_content": f"{gc_content(dna):.6f}",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(paths: dict[str, Path]) -> dict[str, dict[str, int | str]]:
    return {
        name: {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def summarize_lengths(values: Iterable[int]) -> dict[str, int | float | None]:
    lengths = list(values)
    if not lengths:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(lengths),
        "median": statistics.median(lengths),
        "max": max(lengths),
    }


def atomic_json_dump(payload: dict, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("n_benthamiana_preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def validate_config(config: dict) -> FilterRules:
    filters = config["filters"]
    if filters["translation_table"] != 1:
        raise ValueError("Only standard genetic code translation_table=1 is supported")
    return FilterRules(
        allowed_dna_alphabet=frozenset(filters["allowed_dna_alphabet"]),
        required_start_codon=filters["required_start_codon"],
        standard_stop_codons=frozenset(filters["standard_stop_codons"]),
        translation_table=filters["translation_table"],
        max_protein_length=int(filters["max_protein_length"]),
    )


def verify_outputs(
    accepted_path: Path,
    rejected_path: Path,
    cds_records: dict[str, str],
    protein_records: dict[str, str],
    organism: str,
    source_organism: str,
    rules: FilterRules,
) -> dict[str, int | bool]:
    accepted_ids: set[str] = set()
    rejected_ids: set[str] = set()

    with accepted_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ACCEPTED_COLUMNS:
            raise RuntimeError(f"Unexpected accepted columns: {reader.fieldnames}")
        for row in reader:
            source_id = row["source_id"]
            if source_id in accepted_ids:
                raise RuntimeError(f"Duplicate accepted source_id: {source_id}")
            accepted_ids.add(source_id)
            if row["dna"] != cds_records.get(source_id):
                raise RuntimeError(f"Accepted DNA changed for {source_id}")
            if row["protein"] != protein_records.get(source_id):
                raise RuntimeError(f"Accepted protein changed for {source_id}")
            if row["organism"] != organism or row["source_organism"] != source_organism:
                raise RuntimeError(f"Incorrect organism labels for {source_id}")
            reasons = assess_pair(row["dna"], row["protein"], rules)
            if reasons:
                raise RuntimeError(f"Accepted record fails QC: {source_id}: {reasons}")

    with rejected_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REJECTED_COLUMNS:
            raise RuntimeError(f"Unexpected rejected columns: {reader.fieldnames}")
        for row in reader:
            source_id = row["source_id"]
            if source_id in rejected_ids:
                raise RuntimeError(f"Duplicate rejected source_id: {source_id}")
            rejected_ids.add(source_id)
            expected_dna = cds_records.get(source_id, "")
            expected_protein = protein_records.get(source_id, "")
            if row["dna"] != expected_dna or row["protein"] != expected_protein:
                raise RuntimeError(f"Rejected sequence changed for {source_id}")
            if not expected_dna:
                expected_reasons = ["missing_cds"]
            elif not expected_protein:
                expected_reasons = ["missing_protein"]
            else:
                expected_reasons = assess_pair(expected_dna, expected_protein, rules)
            actual_reasons = row["rejection_reasons"].split(";")
            if actual_reasons != expected_reasons:
                raise RuntimeError(
                    f"Rejected reasons changed for {source_id}: "
                    f"{actual_reasons} != {expected_reasons}"
                )

    overlap = accepted_ids & rejected_ids
    expected_ids = set(cds_records) | set(protein_records)
    observed_ids = accepted_ids | rejected_ids
    if overlap:
        raise RuntimeError(f"Accepted/rejected overlap: {sorted(overlap)[:10]}")
    if observed_ids != expected_ids:
        raise RuntimeError(
            "Output IDs do not cover input IDs: "
            f"missing={sorted(expected_ids - observed_ids)[:10]}, "
            f"extra={sorted(observed_ids - expected_ids)[:10]}"
        )
    return {
        "passed": True,
        "accepted_ids": len(accepted_ids),
        "rejected_ids": len(rejected_ids),
        "overlap_ids": 0,
        "covered_input_ids": len(observed_ids),
    }


def run(config_path: Path) -> dict:
    config = read_json(config_path)
    rules = validate_config(config)
    seed = int(config["random_seed"])
    random.seed(seed)

    input_paths = {name: resolve_path(value) for name, value in config["inputs"].items()}
    missing_inputs = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing input files: {missing_inputs}")

    output_config = config["outputs"]
    output_dir = resolve_path(output_config["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / output_config["accepted_csv"]
    rejected_path = output_dir / output_config["rejected_csv"]
    summary_path = output_dir / output_config["qc_summary_json"]
    log_path = resolve_path(config["log_path"])
    logger = setup_logging(log_path)

    logger.info("Starting pipeline version %s with seed %d", config["pipeline_version"], seed)
    logger.info("Loading CDS FASTA: %s", input_paths["cds_fasta"])
    cds_order, cds_records = load_fasta(input_paths["cds_fasta"])
    logger.info("Loading protein FASTA: %s", input_paths["protein_fasta"])
    protein_order, protein_records = load_fasta(input_paths["protein_fasta"])

    organism = config["labels"]["organism"]
    source_organism = config["labels"]["source_organism"]
    reason_counts: collections.Counter[str] = collections.Counter()
    combination_counts: collections.Counter[str] = collections.Counter()
    accepted_count = 0
    rejected_count = 0
    accepted_cds_lengths: list[int] = []
    accepted_protein_lengths: list[int] = []

    accepted_temp = accepted_path.with_suffix(accepted_path.suffix + ".tmp")
    rejected_temp = rejected_path.with_suffix(rejected_path.suffix + ".tmp")
    with (
        accepted_temp.open("w", encoding="utf-8", newline="") as accepted_handle,
        rejected_temp.open("w", encoding="utf-8", newline="") as rejected_handle,
    ):
        accepted_writer = csv.DictWriter(accepted_handle, fieldnames=ACCEPTED_COLUMNS)
        rejected_writer = csv.DictWriter(rejected_handle, fieldnames=REJECTED_COLUMNS)
        accepted_writer.writeheader()
        rejected_writer.writeheader()

        for source_id in cds_order:
            dna = cds_records[source_id]
            protein = protein_records.get(source_id, "")
            row = make_row(source_id, dna, protein, organism, source_organism)
            if not protein:
                reasons = ["missing_protein"]
            else:
                reasons = assess_pair(dna, protein, rules)

            if reasons:
                rejected_count += 1
                reason_counts.update(reasons)
                combination_counts[";".join(reasons)] += 1
                rejected_writer.writerow({**row, "rejection_reasons": ";".join(reasons)})
            else:
                accepted_count += 1
                accepted_cds_lengths.append(len(dna))
                accepted_protein_lengths.append(len(remove_one_terminal_stop(protein)))
                accepted_writer.writerow(row)

        for source_id in protein_order:
            if source_id in cds_records:
                continue
            protein = protein_records[source_id]
            row = make_row(source_id, "", protein, organism, source_organism)
            reasons = ["missing_cds"]
            rejected_count += 1
            reason_counts.update(reasons)
            combination_counts["missing_cds"] += 1
            rejected_writer.writerow({**row, "rejection_reasons": "missing_cds"})

    os.replace(accepted_temp, accepted_path)
    os.replace(rejected_temp, rejected_path)

    logger.info("Verifying output rows against original FASTA records and QC rules")
    verification = verify_outputs(
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        cds_records=cds_records,
        protein_records=protein_records,
        organism=organism,
        source_organism=source_organism,
        rules=rules,
    )

    manifest = file_manifest(input_paths)
    output_manifest = file_manifest({
        "accepted_csv": accepted_path,
        "rejected_csv": rejected_path,
    })
    summary = {
        "pipeline_version": config["pipeline_version"],
        "random_seed": seed,
        "labels": config["labels"],
        "filters": config["filters"],
        "translation_comparison": (
            "Standard genetic code; one terminal '*' marker is ignored on each side "
            "for FASTA representation equivalence; no sequence is modified in output."
        ),
        "inputs": manifest,
        "counts": {
            "cds_records": len(cds_records),
            "protein_records": len(protein_records),
            "paired_ids": len(set(cds_records) & set(protein_records)),
            "cds_only_ids": len(set(cds_records) - set(protein_records)),
            "protein_only_ids": len(set(protein_records) - set(cds_records)),
            "accepted_records": accepted_count,
            "rejected_records": rejected_count,
        },
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "rejection_reason_combination_counts": dict(sorted(combination_counts.items())),
        "accepted_length_summary": {
            "cds_length": summarize_lengths(accepted_cds_lengths),
            "protein_length": summarize_lengths(accepted_protein_lengths),
        },
        "outputs": output_manifest,
        "verification": verification,
        "final_splits_generated": False,
        "next_stage": [
            "compute CSI/CAI, GC, and codon-usage metrics",
            "cluster protein sequences by similarity",
            "split by similarity cluster to prevent homology leakage",
            "prepare all_clean_hc, csi_top10_hc, and csi_top25_hc cohorts",
            "export CodonTransformer finetune JSONL only after cluster-aware splitting",
        ],
    }
    atomic_json_dump(summary, summary_path)
    logger.info("Accepted %d records; rejected %d records", accepted_count, rejected_count)
    logger.info("Rejection reason counts: %s", dict(sorted(reason_counts.items())))
    logger.info("Wrote %s", accepted_path)
    logger.info("Wrote %s", rejected_path)
    logger.info("Wrote %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic stage-one N. benthamiana QC outputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON configuration file (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()
    try:
        run(args.config.resolve())
    except Exception:
        logging.exception("Preprocessing failed")
        raise


if __name__ == "__main__":
    main()
