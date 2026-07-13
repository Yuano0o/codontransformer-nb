#!/usr/bin/env python3
"""Download a Hugging Face model snapshot to an explicit local directory."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="adibvafa/CodonTransformer")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output_dir,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
