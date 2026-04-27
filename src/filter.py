"""Filter raw Statcast pulls and add the columns the Q-Transformer needs.

One responsibility: read ``data/raw/`` for one season, drop disqualified rows
(PA-atomically), add derived columns, write ``data/processed/statcast_{year}.parquet``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SORT_KEYS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
PA_KEYS = ["game_pk", "at_bat_number"]

DROP_PITCH_TYPES = {"PO", "UN"}
FASTBALL_TYPES = {"FF", "SI", "FC"}
POSITION_PLAYER_FB_MAX_MPH = 80.0

# Events that mark a PA as "didn't really resolve via the pitcher" — drop the
# whole PA if any pitch in it has events in this set.
DROP_EVENTS = {"intent_walk", "truncated_pa"}

# Columns the Q-Transformer needs as state / action / reward / pitch features.
# Any pitch with a null in any of these triggers a PA-atomic drop. Informative
# nulls (events on non-terminal pitches, on_1b/2b/3b when no runner, batted-ball
# fields like launch_speed) are deliberately excluded — they aren't "missing
# data," the absence is the signal.
REQUIRED_NONNULL_COLS = [
    # action
    "pitch_type", "plate_x", "plate_z",
    # reward
    "delta_run_exp",
    # release / pitch physics
    "release_speed", "release_pos_x", "release_pos_y", "release_pos_z",
    "release_spin_rate", "spin_axis",
    "release_extension",
    "pfx_x", "pfx_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "sz_top", "sz_bot",
    # handedness + IDs
    "p_throws", "stand", "batter", "pitcher",
    # count / game state
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "description", "type", "zone",
    "home_score", "away_score", "bat_score", "fld_score",
    "n_thruorder_pitcher",
]


def load_raw(raw_dir: Path, year: int) -> pd.DataFrame:
    """Concatenate all data/raw/statcast_{year}_*.parquet into one ascending-sorted frame."""
    paths = sorted(raw_dir.glob(f"statcast_{year}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no raw parquet files matching statcast_{year}_*.parquet in {raw_dir}")
    logger.info("loading %d raw files for %d", len(paths), year)
    df = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in paths], ignore_index=True)
    df = df.sort_values(SORT_KEYS, ascending=True, kind="mergesort").reset_index(drop=True)
    logger.info("loaded raw frame: rows=%d cols=%d", len(df), df.shape[1])
    return df


def _identify_position_player_pitchers(df: pd.DataFrame) -> set[int]:
    """Pitchers whose season-level max release_speed over FB/SI/FC stays below 80 mph."""
    fb = df[df["pitch_type"].isin(FASTBALL_TYPES) & df["release_speed"].notna()]
    if fb.empty:
        return set()
    max_fb = fb.groupby("pitcher")["release_speed"].max()
    pos_player_pitchers = set(max_fb[max_fb < POSITION_PLAYER_FB_MAX_MPH].index.tolist())
    logger.info(
        "position-player pitchers identified: %d (max FB <%.0f mph)",
        len(pos_player_pitchers),
        POSITION_PLAYER_FB_MAX_MPH,
    )
    return pos_player_pitchers


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply CLAUDE.md drop rules PA-atomically. Logs row + PA counts at each step."""
    n0 = len(df)
    pa0 = df.groupby(PA_KEYS, sort=False).ngroups
    logger.info("filter start: rows=%d pas=%d", n0, pa0)

    # Step 1: drop entire non-regular-season games (PA-atomic by construction).
    df = df[df["game_type"] == "R"].copy()
    logger.info("after game_type=='R': rows=%d (-%d)", len(df), n0 - len(df))

    # Step 2: identify all "bad" rows. Any PA containing a bad row is dropped wholesale.
    pos_player_pitchers = _identify_position_player_pitchers(df)

    # Generic required-column null mask. Per-column counts are logged for
    # interpretability — the drop is the union.
    missing_required = [c for c in REQUIRED_NONNULL_COLS if c not in df.columns]
    if missing_required:
        raise KeyError(f"REQUIRED_NONNULL_COLS missing from frame: {missing_required}")
    null_mask = pd.Series(False, index=df.index)
    for col in REQUIRED_NONNULL_COLS:
        m = df[col].isna()
        if m.any():
            logger.info("  bad-row reason 'null %s': %d rows", col, int(m.sum()))
        null_mask |= m

    bad_pitch_type_value = df["pitch_type"].isin(DROP_PITCH_TYPES)
    bad_intent_pitch = df["description"] == "intent_ball"
    bad_pitcher = df["pitcher"].isin(pos_player_pitchers)
    bad_event = df["events"].isin(DROP_EVENTS)

    other_masks = {
        "pitch_type in {PO, UN}": bad_pitch_type_value,
        "description == 'intent_ball'": bad_intent_pitch,
        "position-player pitcher": bad_pitcher,
        "events in DROP_EVENTS (intent_walk, truncated_pa)": bad_event,
    }
    for name, m in other_masks.items():
        logger.info("  bad-row reason '%s': %d rows", name, int(m.sum()))

    bad_any = null_mask | bad_pitch_type_value | bad_intent_pitch | bad_pitcher | bad_event
    bad_pas = set(map(tuple, df.loc[bad_any, PA_KEYS].drop_duplicates().to_numpy().tolist()))

    # PA-level rule: drop PAs that never produced a terminal pitch (i.e. inning
    # ended on a non-pitch event such as caught-stealing or pickoff for the 3rd
    # out). These have events.notna().sum() == 0 across the whole PA and would
    # otherwise trip the is_terminal == 1 invariant in add_derived_columns.
    has_terminal = df.groupby(PA_KEYS, sort=False)["events"].transform(lambda s: s.notna().any())
    no_terminal_mask = ~has_terminal
    no_terminal_pas = set(
        map(tuple, df.loc[no_terminal_mask, PA_KEYS].drop_duplicates().to_numpy().tolist())
    )
    logger.info("  bad-PA reason 'no terminal pitch (inning-end on basepaths)': %d PAs", len(no_terminal_pas))
    bad_pas |= no_terminal_pas

    logger.info("PAs containing >=1 bad row OR no terminal pitch: %d", len(bad_pas))

    pa_index = pd.MultiIndex.from_arrays([df["game_pk"], df["at_bat_number"]])
    keep_mask = ~pa_index.isin(bad_pas)
    df = df.loc[keep_mask].copy()

    n1, pa1 = len(df), df.groupby(PA_KEYS, sort=False).ngroups
    logger.info("filter done: rows=%d (-%d) pas=%d (-%d)", n1, n0 - n1, pa1, pa0 - pa1)
    return df.reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add reward_pitcher, prev_pitch_type, pitch_idx_in_pa, is_terminal, plate_x_mirrored.

    Assumes ``df`` is already ascending-sorted by SORT_KEYS and contains only
    PAs that survived filtering (so each PA has a complete pitch sequence).
    """
    df = df.sort_values(SORT_KEYS, ascending=True, kind="mergesort").reset_index(drop=True)

    df["reward_pitcher"] = -df["delta_run_exp"]

    pa_groups = df.groupby(PA_KEYS, sort=False)
    df["prev_pitch_type"] = pa_groups["pitch_type"].shift(1)
    df["pitch_idx_in_pa"] = pa_groups.cumcount()

    df["is_terminal"] = df["events"].notna()

    # is_terminal sanity: exactly one terminal pitch per PA.
    terminals_per_pa = df.groupby(PA_KEYS, sort=False)["is_terminal"].sum()
    bad = terminals_per_pa[terminals_per_pa != 1]
    if len(bad) > 0:
        examples = bad.head(5).to_dict()
        raise AssertionError(
            f"is_terminal sanity failed: {len(bad)} PAs have != 1 terminal pitch. "
            f"Examples (pa -> count): {examples}"
        )

    sign = np.where(df["p_throws"].eq("L"), -1.0, 1.0)
    df["plate_x_mirrored"] = df["plate_x"] * sign

    return df


def process_season(raw_dir: Path, processed_dir: Path, year: int) -> Path:
    """End-to-end: load_raw → apply_filters → add_derived_columns → write parquet. Returns output path."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw(raw_dir, year)
    df = apply_filters(df)
    df = add_derived_columns(df)
    out_path = processed_dir / f"statcast_{year}.parquet"
    df.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("wrote %s rows=%d cols=%d", out_path.name, len(df), df.shape[1])
    return out_path
