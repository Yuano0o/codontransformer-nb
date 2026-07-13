#!/usr/bin/env python3
"""Fail when a proposed repository contains local paths, secrets, or large artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "upstream", "models", "data", "results", "outputs"}
FORBIDDEN_TEXT = (
    "/" + "Users" + "/",
    "C:" + "\\" + "Users" + "\\",
)
FORBIDDEN_SUFFIXES = {
    ".ckpt", ".safetensors", ".pt", ".pth", ".onnx", ".pem", ".key",
    ".fa", ".fasta", ".gff3", ".jsonl",
}
MAX_FILE_BYTES = 10 * 1024 * 1024


def candidate_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if result.returncode == 0 and result.stdout.strip():
        return [ROOT / line for line in result.stdout.splitlines()]
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_PARTS)
    ]


def main() -> None:
    failures: list[str] = []
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"large file: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact type: {relative}")
        if path.name.endswith((".jsonl.gz", ".fa.gz", ".fasta.gz", ".gff3.gz")):
            failures.append(f"forbidden compressed data artifact: {relative}")
        if path.name.startswith("Nbe_v"):
            failures.append(f"raw NbeBase file: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for marker in FORBIDDEN_TEXT:
            if marker in text:
                failures.append(f"local absolute path in {relative}")
    if failures:
        raise SystemExit("Repository hygiene check failed:\n" + "\n".join(failures))
    print(f"Repository hygiene check passed for {len(candidate_files())} files")


if __name__ == "__main__":
    main()
