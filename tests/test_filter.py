"""Tests for src/filter.py — drop rules and derived columns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.filter import (
    PA_KEYS,
    POSITION_PLAYER_FB_MAX_MPH,
    add_derived_columns,
    apply_filters,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _row(
    *,
    game_pk: int,
    at_bat_number: int,
    pitch_number: int,
    pitch_type: str | float = "FF",
    plate_x: float = 0.0,
    plate_z: float = 2.5,
    description: str = "ball",
    events: float | str = np.nan,
    game_type: str = "R",
    p_throws: str = "R",
    pitcher: int = 100,
    release_speed: float = 95.0,
    delta_run_exp: float = 0.0,
    game_date: str = "2024-04-01",
    stand: str = "R",
    batter: int = 200,
) -> dict:
    """Synthetic Statcast row with every REQUIRED_NONNULL_COLS field populated."""
    return dict(
        game_date=pd.Timestamp(game_date),
        game_pk=game_pk,
        at_bat_number=at_bat_number,
        pitch_number=pitch_number,
        pitch_type=pitch_type,
        plate_x=plate_x,
        plate_z=plate_z,
        description=description,
        events=events,
        game_type=game_type,
        p_throws=p_throws,
        pitcher=pitcher,
        release_speed=release_speed,
        delta_run_exp=delta_run_exp,
        stand=stand,
        batter=batter,
        # Required physics / state columns — defaults are physically plausible.
        release_pos_x=-1.5, release_pos_y=54.0, release_pos_z=6.0,
        release_spin_rate=2300.0, spin_axis=200.0,
        release_extension=6.5, effective_speed=95.0,
        pfx_x=-0.5, pfx_z=1.5,
        vx0=5.0, vy0=-135.0, vz0=-5.0,
        ax=-10.0, ay=25.0, az=-20.0,
        sz_top=3.5, sz_bot=1.6,
        balls=0, strikes=0, outs_when_up=0, inning=1, inning_topbot="Top",
        type="B", zone=5,
        home_score=0, away_score=0, bat_score=0, fld_score=0,
        n_thruorder_pitcher=1,
    )


def _pa_terminal(rows: list[dict]) -> list[dict]:
    """Mark the last row of the PA as terminal by setting events to a non-null string."""
    rows[-1]["events"] = "field_out"
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Filter rule tests
# --------------------------------------------------------------------------- #


def test_drops_non_regular_season_games():
    rows = _pa_terminal([_row(game_pk=1, at_bat_number=1, pitch_number=1, game_type="S")])  # spring
    rows += _pa_terminal([_row(game_pk=2, at_bat_number=1, pitch_number=1)])  # regular
    out = apply_filters(_frame(rows))
    assert set(out["game_pk"].unique()) == {2}


def test_drops_pa_with_null_pitch_type():
    rows = _pa_terminal(
        [
            _row(game_pk=1, at_bat_number=1, pitch_number=1),
            _row(game_pk=1, at_bat_number=1, pitch_number=2, pitch_type=np.nan),
            _row(game_pk=1, at_bat_number=1, pitch_number=3),
        ]
    )
    rows += _pa_terminal([_row(game_pk=1, at_bat_number=2, pitch_number=1)])
    out = apply_filters(_frame(rows))
    pas = set(map(tuple, out[PA_KEYS].drop_duplicates().to_numpy().tolist()))
    assert pas == {(1, 2)}


@pytest.mark.parametrize("bad_pitch_type", ["PO", "UN"])
def test_drops_pa_with_pitchout_or_unknown(bad_pitch_type):
    rows = _pa_terminal(
        [
            _row(game_pk=1, at_bat_number=1, pitch_number=1, pitch_type=bad_pitch_type),
        ]
    )
    rows += _pa_terminal([_row(game_pk=1, at_bat_number=2, pitch_number=1)])
    out = apply_filters(_frame(rows))
    assert (out["at_bat_number"] == 1).sum() == 0


def test_drops_pa_with_null_plate_coords():
    rows = _pa_terminal(
        [
            _row(game_pk=1, at_bat_number=1, pitch_number=1),
            _row(game_pk=1, at_bat_number=1, pitch_number=2, plate_x=np.nan),
        ]
    )
    rows += _pa_terminal(
        [
            _row(game_pk=1, at_bat_number=2, pitch_number=1, plate_z=np.nan),
        ]
    )
    rows += _pa_terminal([_row(game_pk=1, at_bat_number=3, pitch_number=1)])
    out = apply_filters(_frame(rows))
    assert set(out["at_bat_number"].unique()) == {3}


def test_drops_pa_with_intent_ball():
    rows = _pa_terminal(
        [
            _row(game_pk=1, at_bat_number=1, pitch_number=1),
            _row(game_pk=1, at_bat_number=1, pitch_number=2, description="intent_ball"),
            _row(game_pk=1, at_bat_number=1, pitch_number=3),
        ]
    )
    rows += _pa_terminal([_row(game_pk=1, at_bat_number=2, pitch_number=1)])
    out = apply_filters(_frame(rows))
    assert set(out["at_bat_number"].unique()) == {2}


def test_drops_position_player_pitcher_all_pitches():
    """Pitcher 999 throws only ~70 mph fastballs all season → drop every PA they appear in."""
    rows: list[dict] = []
    # Position-player pitcher: only pitcher 999, max FB ~72 mph.
    rows += _pa_terminal(
        [
            _row(game_pk=10, at_bat_number=1, pitch_number=1, pitcher=999, pitch_type="FF", release_speed=72.0),
            _row(game_pk=10, at_bat_number=1, pitch_number=2, pitcher=999, pitch_type="CH", release_speed=68.0),
        ]
    )
    # Real pitcher 100, throws a slow changeup at 73 mph but FB sits 95 — must be kept.
    rows += _pa_terminal(
        [
            _row(game_pk=10, at_bat_number=2, pitch_number=1, pitcher=100, pitch_type="FF", release_speed=95.0),
            _row(game_pk=10, at_bat_number=2, pitch_number=2, pitcher=100, pitch_type="CH", release_speed=73.0),
        ]
    )
    out = apply_filters(_frame(rows))
    assert set(out["pitcher"].unique()) == {100}
    assert (out["at_bat_number"] == 2).sum() == 2


def test_drops_pa_with_null_required_column():
    """A null in any REQUIRED_NONNULL_COLS field (here: release_spin_rate) must drop the whole PA."""
    rows = _pa_terminal(
        [
            _row(game_pk=30, at_bat_number=1, pitch_number=1),
            _row(game_pk=30, at_bat_number=1, pitch_number=2),
        ]
    )
    rows[1]["release_spin_rate"] = np.nan  # mid-PA spin tracking failure
    rows += _pa_terminal([_row(game_pk=30, at_bat_number=2, pitch_number=1)])
    out = apply_filters(_frame(rows))
    assert set(out["at_bat_number"].unique()) == {2}


def test_drops_pa_with_intent_walk_event():
    rows = _pa_terminal(
        [
            _row(game_pk=31, at_bat_number=1, pitch_number=1),
        ]
    )
    rows[-1]["events"] = "intent_walk"
    rows += _pa_terminal([_row(game_pk=31, at_bat_number=2, pitch_number=1)])
    out = apply_filters(_frame(rows))
    assert set(out["at_bat_number"].unique()) == {2}


def test_drops_pa_with_truncated_pa_event():
    rows = _pa_terminal(
        [
            _row(game_pk=32, at_bat_number=1, pitch_number=1),
        ]
    )
    rows[-1]["events"] = "truncated_pa"
    rows += _pa_terminal([_row(game_pk=32, at_bat_number=2, pitch_number=1)])
    out = apply_filters(_frame(rows))
    assert set(out["at_bat_number"].unique()) == {2}


def test_drops_pa_with_no_terminal_pitch():
    """A PA that never gets an events value (inning ended on basepaths) must be dropped."""
    # PA with no terminal — every events is NaN — and last pitch had outs_when_up=2
    rows = [
        _row(game_pk=20, at_bat_number=1, pitch_number=1),
        _row(game_pk=20, at_bat_number=1, pitch_number=2),
    ]
    rows += _pa_terminal([_row(game_pk=20, at_bat_number=2, pitch_number=1)])
    out = apply_filters(_frame(rows))
    assert set(out["at_bat_number"].unique()) == {2}


def test_position_player_threshold_uses_only_fastballs():
    """A pitcher whose only fastball sits at 81 must be kept even with slow off-speed."""
    rows = _pa_terminal(
        [
            _row(game_pk=11, at_bat_number=1, pitch_number=1, pitcher=200, pitch_type="FF", release_speed=81.0),
            _row(game_pk=11, at_bat_number=1, pitch_number=2, pitcher=200, pitch_type="CU", release_speed=65.0),
        ]
    )
    out = apply_filters(_frame(rows))
    assert set(out["pitcher"].unique()) == {200}
    assert POSITION_PLAYER_FB_MAX_MPH == 80.0  # guard against silent threshold change


# --------------------------------------------------------------------------- #
# Derived-column tests
# --------------------------------------------------------------------------- #


def _clean_two_pa_frame() -> pd.DataFrame:
    rows = _pa_terminal(
        [
            _row(game_pk=1, at_bat_number=1, pitch_number=1, pitch_type="FF", delta_run_exp=-0.05, p_throws="R", plate_x=0.5),
            _row(game_pk=1, at_bat_number=1, pitch_number=2, pitch_type="SL", delta_run_exp=0.10, p_throws="R", plate_x=-0.3),
            _row(game_pk=1, at_bat_number=1, pitch_number=3, pitch_type="CH", delta_run_exp=-0.20, p_throws="R", plate_x=0.1),
        ]
    )
    rows += _pa_terminal(
        [
            _row(game_pk=1, at_bat_number=2, pitch_number=1, pitch_type="FF", delta_run_exp=0.0, p_throws="L", plate_x=0.4),
            _row(game_pk=1, at_bat_number=2, pitch_number=2, pitch_type="SI", delta_run_exp=-0.15, p_throws="L", plate_x=-0.6),
        ]
    )
    return _frame(rows)


def test_reward_pitcher_is_negated_delta_run_exp():
    df = add_derived_columns(_clean_two_pa_frame())
    np.testing.assert_array_almost_equal(df["reward_pitcher"].to_numpy(), -df["delta_run_exp"].to_numpy())


def test_prev_pitch_type_is_nan_on_first_pitch_and_does_not_leak_across_pas():
    df = add_derived_columns(_clean_two_pa_frame())
    first_pitches = df[df["pitch_number"] == 1]
    assert first_pitches["prev_pitch_type"].isna().all()

    pa1 = df[(df["game_pk"] == 1) & (df["at_bat_number"] == 1)].sort_values("pitch_number")
    assert pa1.iloc[1]["prev_pitch_type"] == "FF"
    assert pa1.iloc[2]["prev_pitch_type"] == "SL"


def test_pitch_idx_in_pa_is_zero_indexed_and_per_pa():
    df = add_derived_columns(_clean_two_pa_frame())
    pa1 = df[(df["game_pk"] == 1) & (df["at_bat_number"] == 1)].sort_values("pitch_number")
    pa2 = df[(df["game_pk"] == 1) & (df["at_bat_number"] == 2)].sort_values("pitch_number")
    assert list(pa1["pitch_idx_in_pa"]) == [0, 1, 2]
    assert list(pa2["pitch_idx_in_pa"]) == [0, 1]


def test_is_terminal_true_only_on_last_pitch_of_pa():
    df = add_derived_columns(_clean_two_pa_frame())
    for (gpk, abn), pa in df.groupby(PA_KEYS, sort=False):
        pa = pa.sort_values("pitch_number")
        assert pa["is_terminal"].sum() == 1
        assert bool(pa.iloc[-1]["is_terminal"]) is True
        if len(pa) > 1:
            assert not pa.iloc[:-1]["is_terminal"].any()


def test_is_terminal_assertion_fires_when_pa_missing_terminal_event():
    """If a PA has no events on any pitch, add_derived_columns must raise."""
    rows = [
        _row(game_pk=1, at_bat_number=1, pitch_number=1),  # no events on any row
        _row(game_pk=1, at_bat_number=1, pitch_number=2),
    ]
    with pytest.raises(AssertionError, match="is_terminal sanity"):
        add_derived_columns(_frame(rows))


def test_plate_x_mirrored_flips_sign_for_lhp_only():
    df = add_derived_columns(_clean_two_pa_frame())
    rhp = df[df["p_throws"] == "R"]
    lhp = df[df["p_throws"] == "L"]
    np.testing.assert_array_almost_equal(rhp["plate_x_mirrored"].to_numpy(), rhp["plate_x"].to_numpy())
    np.testing.assert_array_almost_equal(lhp["plate_x_mirrored"].to_numpy(), -lhp["plate_x"].to_numpy())
