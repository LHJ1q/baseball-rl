"""Verification report for the processed splits.

Two phases of checks:

1. **Statistical / structural checks** (1-6): coverage, row counts, distributions,
   PA boundary correctness — the original CLAUDE.md §Verification spec.
2. **Drop-rule compliance checks** (7-14): explicitly verify every filter rule
   we documented in ``docs/FILTER_RULES.md`` actually fired and the post-filter
   data complies. Belt-and-suspenders against rule regressions.

Print PASS / WARN / FAIL for every check; exit non-zero if any FAIL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.filter import (
    DROP_PITCH_TYPES,
    DROP_EVENTS,
    FASTBALL_TYPES,
    POSITION_PLAYER_FB_MAX_MPH,
    REQUIRED_NONNULL_COLS,
)

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
# Drop-rule compliance checks (7-14)
# --------------------------------------------------------------------------- #


def check_required_nonnull(df: pd.DataFrame) -> CheckResult:
    """Every row must have non-null values for every column in
    ``REQUIRED_NONNULL_COLS``. This is the strictest data-cleanliness check —
    if any row has a null in any required column, the PA-atomic filter rule
    failed somewhere upstream."""
    bad = {}
    for col in REQUIRED_NONNULL_COLS:
        if col not in df.columns:
            bad[col] = "MISSING_COLUMN"
            continue
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            bad[col] = n_null
    status = "PASS" if not bad else "FAIL"
    return CheckResult(
        name="7. all REQUIRED_NONNULL_COLS are non-null on every row",
        status=status,
        summary=f"{len(REQUIRED_NONNULL_COLS)} required cols checked, {len(bad)} have nulls",
        detail="" if not bad else f"violations: {bad}",
    )


def check_pitch_type_compliance(df: pd.DataFrame) -> CheckResult:
    """No row should have ``pitch_type`` in {PO, UN} or null."""
    n_drop = int(df["pitch_type"].isin(DROP_PITCH_TYPES).sum())
    n_null = int(df["pitch_type"].isna().sum())
    bad = n_drop + n_null
    status = "PASS" if bad == 0 else "FAIL"
    return CheckResult(
        name=f"8. no pitch_type in {{{', '.join(sorted(DROP_PITCH_TYPES))}}} or null",
        status=status,
        summary=f"{bad} non-compliant rows (PO/UN: {n_drop}, null: {n_null})",
    )


def check_intent_ball_compliance(df: pd.DataFrame) -> CheckResult:
    """No row should have ``description == 'intent_ball'`` (pre-2017 IBB)."""
    n_bad = int((df["description"] == "intent_ball").sum())
    status = "PASS" if n_bad == 0 else "FAIL"
    return CheckResult(
        name="9. no description == 'intent_ball'",
        status=status,
        summary=f"{n_bad} non-compliant rows",
    )


def check_drop_events_compliance(df: pd.DataFrame) -> CheckResult:
    """No row should have ``events in DROP_EVENTS`` (intent_walk, truncated_pa)."""
    n_bad = int(df["events"].isin(DROP_EVENTS).sum())
    bad_breakdown = df.loc[df["events"].isin(DROP_EVENTS), "events"].value_counts().to_dict()
    status = "PASS" if n_bad == 0 else "FAIL"
    return CheckResult(
        name=f"10. no events in {sorted(DROP_EVENTS)}",
        status=status,
        summary=f"{n_bad} non-compliant rows",
        detail="" if not bad_breakdown else f"breakdown: {bad_breakdown}",
    )


def check_terminal_per_pa(df: pd.DataFrame) -> CheckResult:
    """Every PA must have exactly one terminal pitch (events.notna().sum() == 1).
    PAs with 0 terminal pitches (inning ended on basepaths) and >1 (impossible)
    are both data-integrity violations after filtering."""
    terms = df.groupby(PA_KEYS, sort=False)["events"].apply(lambda s: s.notna().sum())
    bad = terms[terms != 1]
    n_zero = int((bad == 0).sum())
    n_multi = int((bad > 1).sum())
    status = "PASS" if len(bad) == 0 else "FAIL"
    return CheckResult(
        name="11. every PA has exactly one terminal pitch",
        status=status,
        summary=f"PAs with !=1 terminals: {len(bad)} (zero={n_zero}, multi={n_multi})",
        detail="" if len(bad) == 0 else f"first 5 bad PAs: {bad.head(5).to_dict()}",
    )


def check_pitch_idx_contiguous(df: pd.DataFrame) -> CheckResult:
    """Within every PA, ``pitch_idx_in_pa`` must run 0..len-1 with no gaps.
    Stronger than monotonic check 6 — also requires zero-indexed start."""
    if "pitch_idx_in_pa" not in df.columns:
        return CheckResult(
            name="12. pitch_idx_in_pa runs 0..len-1 within each PA",
            status="WARN",
            summary="column not present (pre-derived data?)",
        )
    g = df.groupby(PA_KEYS, sort=False)["pitch_idx_in_pa"]
    bad = g.apply(lambda s: list(s) != list(range(len(s))))
    n_bad = int(bad.sum())
    status = "PASS" if n_bad == 0 else "FAIL"
    return CheckResult(
        name="12. pitch_idx_in_pa runs 0..len-1 within each PA",
        status=status,
        summary=f"{n_bad} bad PAs",
    )


def check_game_type_regular(df: pd.DataFrame) -> CheckResult:
    """All rows must come from regular-season games (game_type == 'R')."""
    counts = df["game_type"].value_counts(dropna=False).to_dict()
    n_bad = int((df["game_type"] != "R").sum())
    status = "PASS" if n_bad == 0 else "FAIL"
    return CheckResult(
        name="13. all rows have game_type == 'R'",
        status=status,
        summary=f"non-'R' rows: {n_bad}",
        detail="" if n_bad == 0 else f"game_type counts: {counts}",
    )


def check_no_position_player_pitchers(df: pd.DataFrame) -> CheckResult:
    """After filtering, no remaining pitcher should have a season-max
    ``release_speed`` across fastball types {FF, SI, FC} below the
    ``POSITION_PLAYER_FB_MAX_MPH`` threshold. If any are found, the
    position-player rule didn't fire correctly."""
    fb = df[df["pitch_type"].isin(FASTBALL_TYPES) & df["release_speed"].notna()]
    if fb.empty:
        return CheckResult(
            name="14. no surviving position-player pitchers (max FB < threshold)",
            status="WARN",
            summary="no fastball rows to evaluate",
        )
    max_fb = fb.groupby("pitcher")["release_speed"].max()
    bad = max_fb[max_fb < POSITION_PLAYER_FB_MAX_MPH]
    n_bad = len(bad)
    status = "PASS" if n_bad == 0 else "FAIL"
    return CheckResult(
        name=f"14. no surviving pitchers with max FB < {POSITION_PLAYER_FB_MAX_MPH:.0f} mph",
        status=status,
        summary=f"{n_bad} pitchers below threshold",
        detail="" if n_bad == 0 else f"first 5: {bad.head(5).to_dict()}",
    )


# --------------------------------------------------------------------------- #
# Driver + formatting
# --------------------------------------------------------------------------- #


def run_all_checks(splits_dir: Path) -> list[CheckResult]:
    df = _load_concat(splits_dir)
    return [
        # Statistical / structural (1-6)
        check_delta_run_exp_coverage(df),
        check_row_count(df),
        check_pitch_type_distribution(df),
        check_reward_sanity(df),
        check_pa_within_game(df),
        check_pitch_number_monotonic(df),
        # Drop-rule compliance (7-14)
        check_required_nonnull(df),
        check_pitch_type_compliance(df),
        check_intent_ball_compliance(df),
        check_drop_events_compliance(df),
        check_terminal_per_pa(df),
        check_pitch_idx_contiguous(df),
        check_game_type_regular(df),
        check_no_position_player_pitchers(df),
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
