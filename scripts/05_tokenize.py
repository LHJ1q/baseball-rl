"""CLI entry point for the tokenization phase."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tokenize import process_all_splits  # noqa: E402

SPLITS_DIR = REPO_ROOT / "data" / "splits"
TOKENS_DIR = REPO_ROOT / "data" / "tokens"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize the filtered Statcast splits.")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--tokens-dir", type=Path, default=TOKENS_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    written = process_all_splits(args.splits_dir, args.tokens_dir)
    for name, path in written.items():
        print(f"{name:<18} {path}")


if __name__ == "__main__":
    main()
