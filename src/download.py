"""Statcast pitch-by-pitch download via pybaseball.

One responsibility: pull raw Statcast data and write per-month parquet to
``data/raw/statcast_{year}_{month:02d}.parquet``. No filtering, no derived
columns — those live in :mod:`src.filter`.
"""
from __future__ import annotations

import calendar
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import pybaseball
from pybaseball import statcast

logger = logging.getLogger(__name__)

SORT_KEYS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]


def _ensure_cache() -> None:
    pybaseball.cache.enable()


def pull_range(start: str, end: str) -> pd.DataFrame:
    """Pull a single date range and re-sort ascending. No disk I/O."""
    _ensure_cache()
    logger.info("statcast pull %s..%s", start, end)
    df = statcast(start_dt=start, end_dt=end)
    if df is None or len(df) == 0:
        logger.warning("empty pull for %s..%s", start, end)
        return pd.DataFrame()
    return df.sort_values(SORT_KEYS, ascending=True, kind="mergesort").reset_index(drop=True)


def pull_month(year: int, month: int, out_dir: Path, overwrite: bool = False) -> Path | None:
    """Pull one calendar month and write parquet. Returns the output path, or None on empty/skip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"statcast_{year}_{month:02d}.parquet"
    if out_path.exists() and not overwrite:
        logger.info("skip existing %s", out_path.name)
        return out_path

    last_day = calendar.monthrange(year, month)[1]
    start, end = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"
    df = pull_range(start, end)
    if df.empty:
        logger.warning("no rows for %s — not writing", out_path.name)
        return None
    df.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("wrote %s rows=%d", out_path.name, len(df))
    return out_path


def pull_season(
    year: int,
    out_dir: Path,
    months: Iterable[int] = range(4, 11),
    overwrite: bool = False,
) -> list[Path]:
    """Loop pull_month over the given months. Logs and skips months that fail; returns written paths."""
    written: list[Path] = []
    skipped: list[tuple[int, str]] = []
    for m in months:
        try:
            p = pull_month(year, m, out_dir, overwrite=overwrite)
            if p is not None:
                written.append(p)
        except Exception as e:  # noqa: BLE001 — log and continue per CLAUDE.md hard rule #5
            logger.exception("pull failed for %d-%02d: %s", year, m, e)
            skipped.append((m, str(e)))
    if skipped:
        logger.warning("skipped months: %s", skipped)
    return written
