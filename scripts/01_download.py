"""CLI entry point for the Statcast download phase."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.download import pull_range, pull_season  # noqa: E402


RAW_DIR = REPO_ROOT / "data" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Statcast pitch-by-pitch data.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pull a single week (April 1–7 of --year) and print summary; write nothing.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-pull months whose parquet already exists.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.dry_run:
        start, end = f"{args.year}-04-01", f"{args.year}-04-07"
        df = pull_range(start, end)
        print(f"shape: {df.shape}")
        print("columns:")
        for c in df.columns:
            print(" ", c)
        if "delta_run_exp" in df.columns:
            cov = df["delta_run_exp"].notna().mean()
            print(f"delta_run_exp coverage: {cov:.4f}")
        return

    written = pull_season(args.year, RAW_DIR, overwrite=args.overwrite)
    print(f"wrote {len(written)} month files to {RAW_DIR}")
    for p in written:
        print(" ", p.name)


if __name__ == "__main__":
    main()
