"""Temporal train/val/test splits.

Strictly date-based — never random. Random splitting leaks future player-form
information across PAs from the same week.

Two schemes:

* ``within_season``: within a single year, split by date (Apr–Aug train, Sep
  1–15 val, Sep 16–end test). Original phase-3 scheme; used when only one
  season of data is available.
* ``year_level``: train on full earlier seasons, hold out one final season
  for val/test. Used once we have multiple years of processed data.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

WITHIN_SEASON_BOUNDS: dict[str, tuple[str, str]] = {
    "train": ("{year}-04-01", "{year}-08-31"),
    "val":   ("{year}-09-01", "{year}-09-15"),
    "test":  ("{year}-09-16", "{year}-10-31"),
}


def make_splits_within_season(processed_path: Path, splits_dir: Path, year: int) -> dict[str, Path]:
    """Read one season's processed parquet, slice by date into train/val/test."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(processed_path, engine="pyarrow")
    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info("loaded %s rows=%d", processed_path.name, len(df))

    written: dict[str, Path] = {}
    for name, (start_tmpl, end_tmpl) in WITHIN_SEASON_BOUNDS.items():
        start_ts = pd.Timestamp(start_tmpl.format(year=year))
        end_ts = pd.Timestamp(end_tmpl.format(year=year))
        mask = (df["game_date"] >= start_ts) & (df["game_date"] <= end_ts)
        sub = df.loc[mask].reset_index(drop=True)
        out = splits_dir / f"{name}.parquet"
        sub.to_parquet(out, engine="pyarrow", index=False)
        logger.info("wrote %s rows=%d (%s..%s)", out.name, len(sub), start_ts.date(), end_ts.date())
        written[name] = out
    return written


def make_splits_year_level(
    processed_dir: Path,
    splits_dir: Path,
    train_years: list[int],
    val_test_year: int,
    val_end: str = "07-15",
) -> dict[str, Path]:
    """Train on full earlier seasons; split the final season into val (first
    half) and test (second half) by date.

    Default boundary: ``val_end='07-15'`` → val = Apr-1 to Jul-15 of the val/test
    year, test = Jul-16 to end-of-Oct of that year.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_frames = []
    for y in train_years:
        path = processed_dir / f"statcast_{y}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing processed parquet for train year {y}: {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        df["game_date"] = pd.to_datetime(df["game_date"])
        logger.info("train year %d: loaded rows=%d", y, len(df))
        train_frames.append(df)
    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = train_df.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"], kind="mergesort"
    ).reset_index(drop=True)
    train_path = splits_dir / "train.parquet"
    train_df.to_parquet(train_path, engine="pyarrow", index=False)
    logger.info("wrote train.parquet rows=%d (years %s)", len(train_df), train_years)

    holdout_path = processed_dir / f"statcast_{val_test_year}.parquet"
    if not holdout_path.exists():
        raise FileNotFoundError(f"missing processed parquet for val/test year {val_test_year}: {holdout_path}")
    holdout_df = pd.read_parquet(holdout_path, engine="pyarrow")
    holdout_df["game_date"] = pd.to_datetime(holdout_df["game_date"])

    val_end_ts = pd.Timestamp(f"{val_test_year}-{val_end}")
    val_mask = holdout_df["game_date"] <= val_end_ts
    val_df = holdout_df.loc[val_mask].reset_index(drop=True)
    test_df = holdout_df.loc[~val_mask].reset_index(drop=True)

    val_path = splits_dir / "val.parquet"
    test_path = splits_dir / "test.parquet"
    val_df.to_parquet(val_path, engine="pyarrow", index=False)
    test_df.to_parquet(test_path, engine="pyarrow", index=False)
    logger.info(
        "wrote val.parquet rows=%d (%d-04-01..%d-%s)",
        len(val_df), val_test_year, val_test_year, val_end,
    )
    logger.info(
        "wrote test.parquet rows=%d (%d-%s..%d-10-31)",
        len(test_df), val_test_year, val_end, val_test_year,
    )

    return {"train": train_path, "val": val_path, "test": test_path}


# Backward-compat alias used by phase-3 code paths.
def make_splits(processed_path: Path, splits_dir: Path, year: int = 2024) -> dict[str, Path]:
    return make_splits_within_season(processed_path, splits_dir, year=year)
