"""CLI entry point for the temporal splits phase.

Two schemes:

* ``--scheme within_season`` (default for single-year data): split one year's
  data by date into Apr-Aug / Sep 1-15 / Sep 16-end.
* ``--scheme year_level``: train on full earlier seasons, hold out the final
  season for val (first half) and test (second half).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.splits import make_splits_within_season, make_splits_year_level  # noqa: E402


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write temporal train/val/test splits.")
    parser.add_argument(
        "--scheme", choices=("within_season", "year_level"), default="within_season",
        help="Single-year date split or multi-year year-level split.",
    )
    # within_season args
    parser.add_argument("--year", type=int, default=2024,
                        help="Year for within_season scheme.")
    # year_level args
    parser.add_argument("--train-years", type=int, nargs="+", default=None,
                        help="Years to use as train (year_level scheme).")
    parser.add_argument("--val-test-year", type=int, default=None,
                        help="Year to split into val (first half) + test (second half) (year_level scheme).")
    parser.add_argument("--val-end", type=str, default="07-15",
                        help="MM-DD boundary inside the val/test year (default 07-15 — val ≤ this date, test > this date).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.scheme == "within_season":
        processed = PROCESSED_DIR / f"statcast_{args.year}.parquet"
        written = make_splits_within_season(processed, SPLITS_DIR, year=args.year)
    else:
        if not args.train_years or args.val_test_year is None:
            parser.error("year_level scheme requires --train-years and --val-test-year")
        written = make_splits_year_level(
            PROCESSED_DIR,
            SPLITS_DIR,
            train_years=args.train_years,
            val_test_year=args.val_test_year,
            val_end=args.val_end,
        )

    for name, p in written.items():
        print(f"{name:<5} {p}")


if __name__ == "__main__":
    main()
