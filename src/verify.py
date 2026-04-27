"""Six-check verification report for the processed splits.

Per CLAUDE.md §Verification — must print PASS / WARN / FAIL for every check
and exit non-zero if any check fails.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PA_KEYS = ["game_pk", "at_bat_number"]
SORT_KEYS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

MAINSTREAM_PITCH_TYPES = {"FF", "SL", "CH", "SI", "CU", "FC", "FS", "KC", "ST", "SV"}
# Per-season band based on 2024 actuals (~700K pitches/season). The check
# auto-scales by the number of distinct seasons in the data so the same
# threshold works for single-year (within_season scheme) and multi-year
# (year_level scheme) splits.
EXPECTED_PER_SEASON_LOW = 600_000
EXPECTED_PER_SEASON_HIGH = 800_000
DELTA_RUN_EXP_MIN_COVERAGE = 0.95
PITCH_TYPE_MIN_COVERAGE = 0.95


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    summary: str
    detail: str = ""


def _load_concat(splits_dir: Path) -> pd.DataFrame:
    parts = []
    for name in ("train", "val", "test"):
        p = splits_dir / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"missing split file: {p}")
        parts.append(pd.read_parquet(p, engine="pyarrow"))
    df = pd.concat(parts, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(SORT_KEYS, ascending=True, kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def check_delta_run_exp_coverage(df: pd.DataFrame) -> CheckResult:
    cov = df["delta_run_exp"].notna().mean()
    status = "PASS" if cov > DELTA_RUN_EXP_MIN_COVERAGE else "FAIL"
    return CheckResult(
        name="1. delta_run_exp coverage",
        status=status,
        summary=f"{cov:.4f}",
        detail=f"non-null {df['delta_run_exp'].notna().sum()} / {len(df)}; threshold > {DELTA_RUN_EXP_MIN_COVERAGE}",
    )


def check_row_count(df: pd.DataFrame) -> CheckResult:
    n = len(df)
    n_seasons = max(1, df["game_date"].dt.year.nunique())
    lo, hi = EXPECTED_PER_SEASON_LOW * n_seasons, EXPECTED_PER_SEASON_HIGH * n_seasons
    status = "PASS" if lo <= n <= hi else "WARN"
    return CheckResult(
        name="2. total row count",
        status=status,
        summary=f"{n:,} ({n_seasons} season{'s' if n_seasons > 1 else ''})",
        detail=f"expected band {lo:,}..{hi:,} ({EXPECTED_PER_SEASON_LOW:,}..{EXPECTED_PER_SEASON_HIGH:,} per season)",
    )


def check_pitch_type_distribution(df: pd.DataFrame) -> CheckResult:
    counts = df["pitch_type"].value_counts(dropna=False)
    mainstream = counts[counts.index.isin(MAINSTREAM_PITCH_TYPES)].sum()
    cov = mainstream / counts.sum()
    status = "PASS" if cov > PITCH_TYPE_MIN_COVERAGE else "FAIL"
    top = counts.head(12).to_dict()
    return CheckResult(
        name="3. pitch-type coverage (FF/SL/CH/SI/CU/FC/FS/KC/ST/SV)",
        status=status,
        summary=f"{cov:.4f}",
        detail=f"top12 counts: {top}",
    )


def check_reward_sanity(df: pd.DataFrame, n_pas: int = 10, seed: int = 0) -> CheckResult:
    """Spot-check 10 random PAs: sum of reward_pitcher should be a reasonable RE24-scale number."""
    rng = np.random.default_rng(seed)
    pa_index = df[PA_KEYS].drop_duplicates().reset_index(drop=True)
    pick = pa_index.sample(n=min(n_pas, len(pa_index)), random_state=int(rng.integers(1 << 31)))

    lines: list[str] = []
    abs_sums: list[float] = []
    for _, row in pick.iterrows():
        pa = df[(df["game_pk"] == row["game_pk"]) & (df["at_bat_number"] == row["at_bat_number"])]
        terminal = pa[pa["is_terminal"]]
        event = terminal["events"].iloc[0] if len(terminal) else "?"
        s = pa["reward_pitcher"].sum()
        abs_sums.append(abs(s))
        lines.append(
            f"  game_pk={row['game_pk']} ab={row['at_bat_number']:>3} "
            f"pitches={len(pa)} event={event!s:<22} sum_reward_pitcher={s:+.3f}"
        )

    # Sanity band: a single PA's net delta_run_exp should almost never exceed ~3 runs.
    out_of_band = sum(1 for s in abs_sums if s > 3.0)
    status = "PASS" if out_of_band == 0 else "WARN"
    return CheckResult(
        name="4. reward sanity (10 random PAs)",
        status=status,
        summary=f"{out_of_band} of {n_pas} PAs with |sum| > 3.0",
        detail="\n" + "\n".join(lines),
    )


def check_pa_within_game(df: pd.DataFrame) -> CheckResult:
    max_dates_per_pa = df.groupby(PA_KEYS)["game_date"].nunique().max()
    status = "PASS" if max_dates_per_pa == 1 else "FAIL"
    return CheckResult(
        name="5. no PA crosses game boundaries",
        status=status,
        summary=f"max distinct game_dates per (game_pk, at_bat_number) = {int(max_dates_per_pa)}",
    )


def check_pitch_number_monotonic(df: pd.DataFrame) -> CheckResult:
    """pitch_number must be strictly increasing within every PA."""
    g = df.groupby(PA_KEYS, sort=False)["pitch_number"]
    is_strict = g.apply(lambda s: s.is_monotonic_increasing and s.is_unique)
    bad = is_strict[~is_strict]
    status = "PASS" if len(bad) == 0 else "FAIL"
    detail = "" if len(bad) == 0 else f"first 5 bad PAs: {bad.head(5).index.tolist()}"
    return CheckResult(
        name="6. pitch_number strictly increasing within PA",
        status=status,
        summary=f"{len(bad)} bad PAs",
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# Driver + formatting
# --------------------------------------------------------------------------- #


def run_all_checks(splits_dir: Path) -> list[CheckResult]:
    df = _load_concat(splits_dir)
    return [
        check_delta_run_exp_coverage(df),
        check_row_count(df),
        check_pitch_type_distribution(df),
        check_reward_sanity(df),
        check_pa_within_game(df),
        check_pitch_number_monotonic(df),
    ]


def format_report(results: list[CheckResult]) -> str:
    lines = ["=" * 80, "Statcast pipeline verification report", "=" * 80]
    for r in results:
        lines.append(f"[{r.status:<4}] {r.name}: {r.summary}")
        if r.detail:
            for d in r.detail.splitlines():
                if d:
                    lines.append(f"        {d}")
    lines.append("=" * 80)
    failed = [r for r in results if r.status == "FAIL"]
    warned = [r for r in results if r.status == "WARN"]
    lines.append(f"Summary: {len(results) - len(failed) - len(warned)} pass / {len(warned)} warn / {len(failed)} fail")
    lines.append("=" * 80)
    return "\n".join(lines)
