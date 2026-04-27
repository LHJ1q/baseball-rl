"""CLI entry point for the filter + derived-column phase."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.filter import process_season  # noqa: E402


RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter raw Statcast and add derived columns.")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out = process_season(RAW_DIR, PROCESSED_DIR, args.year)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
