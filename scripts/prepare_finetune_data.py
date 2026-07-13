#!/usr/bin/env python3
"""Convert a CDS CSV (dna, protein, organism) to CodonTransformer JSON."""

import argparse
from pathlib import Path

from CodonTransformer.CodonData import prepare_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "n_benthamiana_training.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepare_training_data(str(args.input_csv), str(args.output))


if __name__ == "__main__":
    main()
