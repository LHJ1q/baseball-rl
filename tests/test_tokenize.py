"""Tests for src/tokenize.py — mirroring, vocab construction, action binning, arsenal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tokenize import (
    N_MIN_BATTER_PROFILE_PAS,
    N_X_BINS,
    N_Z_BINS,
    UNK_ID,
    X_BIN_HI,
    X_BIN_LO,
    Z_BIN_HI,
    Z_BIN_LO,
    _bin_index,
    add_mirrored_columns,
    build_vocabs,
    compute_batter_profile,
    compute_pitcher_arsenal,
    tokenize_split,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _row(
    *,
    game_pk: int = 1,
    at_bat_number: int = 1,
    pitch_number: int = 1,
    pitch_idx_in_pa: int = 0,
    pitch_type: str = "FF",
    description: str = "ball",
    p_throws: str = "R",
    stand: str = "R",
    pitcher: int = 100,
    batter: int = 200,
    plate_x: float = 0.0,
    plate_z: float = 2.5,
    pfx_x: float = -0.5,
    pfx_z: float = 1.4,
    spin_axis: float = 200.0,
    release_pos_x: float = -1.5,
    vx0: float = 5.0,
    ax: float = -10.0,
    release_speed: float = 95.0,
    release_spin_rate: float = 2300.0,
    release_extension: float = 6.5,
    sz_top: float = 3.5,
    sz_bot: float = 1.6,
    zone: int = 5,
    events: float | str = np.nan,
    estimated_woba_using_speedangle: float = np.nan,
    game_pk_required: int = 0,  # ignored — kept for backward compat
    balls: int = 0,
    strikes: int = 0,
    outs_when_up: int = 0,
    inning: int = 1,
    inning_topbot: str = "Top",
    home_score: int = 0,
    away_score: int = 0,
    bat_score: int = 0,
    fld_score: int = 0,
    on_1b: float = np.nan,
    on_2b: float = np.nan,
    on_3b: float = np.nan,
    n_thruorder_pitcher: int = 1,
    reward_pitcher: float = 0.05,
    is_terminal: bool = True,
    game_date: str = "2024-04-01",
) -> dict:
    return dict(
        game_date=pd.Timestamp(game_date),
        game_pk=game_pk,
        at_bat_number=at_bat_number,
        pitch_number=pitch_number,
        pitch_idx_in_pa=pitch_idx_in_pa,
        pitch_type=pitch_type,
        description=description,
        p_throws=p_throws,
        stand=stand,
        pitcher=pitcher,
        batter=batter,
        plate_x=plate_x,
        plate_x_mirrored=plate_x * (-1.0 if p_throws == "L" else 1.0),
        plate_z=plate_z,
        pfx_x=pfx_x,
        pfx_z=pfx_z,
        spin_axis=spin_axis,
        release_pos_x=release_pos_x,
        vx0=vx0,
        ax=ax,
        release_speed=release_speed,
        release_spin_rate=release_spin_rate,
        release_extension=release_extension,
        sz_top=sz_top,
        sz_bot=sz_bot,
        zone=zone,
        events=events,
        estimated_woba_using_speedangle=estimated_woba_using_speedangle,
        balls=balls,
        strikes=strikes,
        outs_when_up=outs_when_up,
        inning=inning,
        inning_topbot=inning_topbot,
        home_score=home_score,
        away_score=away_score,
        bat_score=bat_score,
        fld_score=fld_score,
        on_1b=on_1b,
        on_2b=on_2b,
        on_3b=on_3b,
        n_thruorder_pitcher=n_thruorder_pitcher,
        reward_pitcher=reward_pitcher,
        is_terminal=is_terminal,
    )


def _train_frame() -> pd.DataFrame:
    """Two PAs with multiple pitches each, mixing handedness and pitch types."""
    rows = [
        _row(game_pk=1, at_bat_number=1, pitch_number=1, pitch_idx_in_pa=0, pitch_type="FF",
             p_throws="R", pitcher=100, plate_x=0.5, pfx_x=-0.6, spin_axis=200.0, is_terminal=False),
        _row(game_pk=1, at_bat_number=1, pitch_number=2, pitch_idx_in_pa=1, pitch_type="SL",
             p_throws="R", pitcher=100, plate_x=-0.4, pfx_x=0.7, spin_axis=80.0, is_terminal=True,
             description="hit_into_play"),
        _row(game_pk=1, at_bat_number=2, pitch_number=1, pitch_idx_in_pa=0, pitch_type="FF",
             p_throws="L", pitcher=101, plate_x=0.6, pfx_x=-0.5, spin_axis=200.0, is_terminal=False),
        _row(game_pk=1, at_bat_number=2, pitch_number=2, pitch_idx_in_pa=1, pitch_type="CH",
             p_throws="L", pitcher=101, plate_x=-0.6, pfx_x=0.6, spin_axis=180.0, is_terminal=True,
             description="swinging_strike"),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Mirror tests
# --------------------------------------------------------------------------- #


def test_mirror_flips_signs_for_lhp_only():
    df = add_mirrored_columns(_train_frame())
    rhp = df[df["p_throws"] == "R"]
    lhp = df[df["p_throws"] == "L"]

    np.testing.assert_array_almost_equal(rhp["pfx_x_mirrored"], rhp["pfx_x"])
    np.testing.assert_array_almost_equal(lhp["pfx_x_mirrored"], -lhp["pfx_x"])

    np.testing.assert_array_almost_equal(rhp["release_pos_x_mirrored"], rhp["release_pos_x"])
    np.testing.assert_array_almost_equal(lhp["release_pos_x_mirrored"], -lhp["release_pos_x"])

    np.testing.assert_array_almost_equal(rhp["vx0_mirrored"], rhp["vx0"])
    np.testing.assert_array_almost_equal(lhp["vx0_mirrored"], -lhp["vx0"])

    np.testing.assert_array_almost_equal(rhp["ax_mirrored"], rhp["ax"])
    np.testing.assert_array_almost_equal(lhp["ax_mirrored"], -lhp["ax"])


def test_spin_axis_clock_face_mirror_for_lhp():
    """For LHP a spin_axis of 200 should map to (360 - 200) = 160."""
    df = add_mirrored_columns(_train_frame())
    rhp_200 = df[(df["p_throws"] == "R") & (df["spin_axis"] == 200.0)]
    lhp_200 = df[(df["p_throws"] == "L") & (df["spin_axis"] == 200.0)]
    assert (rhp_200["spin_axis_mirrored"] == 200.0).all()
    assert (lhp_200["spin_axis_mirrored"] == 160.0).all()


def test_spin_axis_mirror_handles_360_wraparound():
    df = pd.DataFrame([
        _row(p_throws="L", spin_axis=0.0),
        _row(p_throws="L", spin_axis=360.0),
        _row(p_throws="R", spin_axis=359.999),
    ])
    out = add_mirrored_columns(df)
    # (360 - 0) % 360 == 0; (360 - 360) % 360 == 0; RHP unchanged.
    np.testing.assert_array_almost_equal(out["spin_axis_mirrored"], [0.0, 0.0, 359.999])


# --------------------------------------------------------------------------- #
# Bin tests
# --------------------------------------------------------------------------- #


def test_bin_index_inside_range():
    values = np.array([X_BIN_LO + 1e-6, 0.0, X_BIN_HI - 1e-6])
    idx, n_clamp = _bin_index(values, X_BIN_LO, X_BIN_HI, N_X_BINS)
    assert n_clamp == 0
    assert idx[0] == 0
    assert idx[-1] == N_X_BINS - 1


def test_bin_index_clamps_out_of_range_and_counts_them():
    values = np.array([X_BIN_LO - 5.0, X_BIN_HI + 5.0, 0.0])
    idx, n_clamp = _bin_index(values, X_BIN_LO, X_BIN_HI, N_X_BINS)
    assert n_clamp == 2
    assert idx[0] == 0
    assert idx[1] == N_X_BINS - 1


# --------------------------------------------------------------------------- #
# Vocab tests
# --------------------------------------------------------------------------- #


def test_unk_reserved_for_player_and_pitch_vocabs():
    train_df = add_mirrored_columns(_train_frame())
    vocabs = build_vocabs(train_df)
    for key in ("pitch_type", "description", "batter", "pitcher"):
        assert vocabs[key]["<UNK>"] == UNK_ID
        assert min(vocabs[key].values()) == UNK_ID
    # Closed sets shouldn't reserve UNK.
    for key in ("p_throws", "stand", "inning_topbot"):
        assert "<UNK>" not in vocabs[key]


def test_bin_edges_are_uniform_and_have_correct_count():
    train_df = add_mirrored_columns(_train_frame())
    vocabs = build_vocabs(train_df)
    assert len(vocabs["x_bin_edges"]) == N_X_BINS + 1
    assert len(vocabs["z_bin_edges"]) == N_Z_BINS + 1
    np.testing.assert_array_almost_equal(np.diff(vocabs["x_bin_edges"]), [(X_BIN_HI - X_BIN_LO) / N_X_BINS] * N_X_BINS)
    np.testing.assert_array_almost_equal(np.diff(vocabs["z_bin_edges"]), [(Z_BIN_HI - Z_BIN_LO) / N_Z_BINS] * N_Z_BINS)


# --------------------------------------------------------------------------- #
# tokenize_split tests
# --------------------------------------------------------------------------- #


def test_unseen_player_in_val_maps_to_unk_id():
    train_df = add_mirrored_columns(_train_frame())
    vocabs = build_vocabs(train_df)

    val_rows = pd.DataFrame([
        _row(pitcher=999_999, batter=888_888, pitch_type="FF"),  # all unseen
    ])
    val_rows = add_mirrored_columns(val_rows)
    out = tokenize_split(val_rows, vocabs, split_name="val")
    assert int(out["pitcher_id"].iloc[0]) == UNK_ID
    assert int(out["batter_id"].iloc[0]) == UNK_ID
    # Pitch type "FF" was in train, so it should *not* be UNK.
    assert int(out["pitch_type_id"].iloc[0]) != UNK_ID


def test_action_bins_are_in_range_and_consistent_with_edges():
    train_df = add_mirrored_columns(_train_frame())
    vocabs = build_vocabs(train_df)
    out = tokenize_split(train_df, vocabs, split_name="train")
    assert (out["x_bin"] >= 0).all() and (out["x_bin"] < N_X_BINS).all()
    assert (out["z_bin"] >= 0).all() and (out["z_bin"] < N_Z_BINS).all()


def test_runner_on_base_booleans():
    rows = pd.DataFrame([
        _row(on_1b=12345.0, on_2b=np.nan, on_3b=np.nan),
        _row(on_1b=np.nan, on_2b=99.0, on_3b=12.0),
    ])
    rows = add_mirrored_columns(rows)
    vocabs = build_vocabs(rows)
    out = tokenize_split(rows, vocabs, split_name="train")
    assert list(out["on_1b"]) == [True, False]
    assert list(out["on_2b"]) == [False, True]
    assert list(out["on_3b"]) == [False, True]


def test_token_split_preserves_per_pa_reward_sums():
    """Tokenization must not change reward — it's a pure cast to float32."""
    df = add_mirrored_columns(_train_frame())
    vocabs = build_vocabs(df)
    out = tokenize_split(df, vocabs, split_name="train")
    pre = df.groupby(["game_pk", "at_bat_number"])["reward_pitcher"].sum()
    post = out.groupby(["game_pk", "at_bat_number"])["reward_pitcher"].sum()
    np.testing.assert_array_almost_equal(pre.to_numpy(), post.to_numpy(), decimal=5)


def test_token_split_has_no_nulls_anywhere():
    df = add_mirrored_columns(_train_frame())
    vocabs = build_vocabs(df)
    out = tokenize_split(df, vocabs, split_name="train")
    nulls = out.isna().sum()
    assert nulls.sum() == 0, f"unexpected nulls: {nulls[nulls > 0].to_dict()}"


# --------------------------------------------------------------------------- #
# Arsenal tests
# --------------------------------------------------------------------------- #


def test_pitcher_arsenal_groups_and_flags_low_sample():
    df = add_mirrored_columns(_train_frame())
    vocabs = build_vocabs(df)
    train_tokens = tokenize_split(df, vocabs, split_name="train")
    arsenal = compute_pitcher_arsenal(train_tokens, df)
    # Each (pitcher_id, pitch_type_id) appears at most twice in the synthetic frame,
    # so every group is below N_MIN_ARSENAL_SAMPLES (30) and should be low_sample=True.
    assert arsenal["low_sample"].all()
    # Mean release_speed should equal raw mean per group.
    for _, row in arsenal.iterrows():
        sub = train_tokens[
            (train_tokens["pitcher_id"] == row["pitcher_id"])
            & (train_tokens["pitch_type_id"] == row["pitch_type_id"])
        ]
        sub_with_features = df.loc[sub.index]
        assert pytest.approx(row["release_speed_mean"], rel=1e-5) == sub_with_features["release_speed"].mean()


# --------------------------------------------------------------------------- #
# Batter profile tests
# --------------------------------------------------------------------------- #


def _batter_profile_frame() -> pd.DataFrame:
    """Synthetic frame: 1 batter facing many pitches with hand-set descriptions."""
    rows = []
    # Batter 700 sees 4 PAs vs FF and 2 PAs vs CU. Outcomes engineered to test rates.
    # PA1: FF ball, FF foul, FF hit_into_play single (2 swings, 1 contact, 0 whiff, no chase)
    rows += [
        _row(batter=700, pitch_type="FF", description="ball", events=np.nan, zone=5,
             game_pk=1, at_bat_number=1, pitch_number=1, pitch_idx_in_pa=0, is_terminal=False),
        _row(batter=700, pitch_type="FF", description="foul", events=np.nan, zone=5,
             game_pk=1, at_bat_number=1, pitch_number=2, pitch_idx_in_pa=1, is_terminal=False),
        _row(batter=700, pitch_type="FF", description="hit_into_play", events="single",
             estimated_woba_using_speedangle=0.5, zone=5,
             game_pk=1, at_bat_number=1, pitch_number=3, pitch_idx_in_pa=2, is_terminal=True),
    ]
    # PA2: FF strikeout (3 pitches, 2 swing-strikes = 2 whiffs, no chase)
    rows += [
        _row(batter=700, pitch_type="FF", description="called_strike", events=np.nan, zone=5,
             game_pk=1, at_bat_number=2, pitch_number=1, pitch_idx_in_pa=0, is_terminal=False),
        _row(batter=700, pitch_type="FF", description="swinging_strike", events=np.nan, zone=5,
             game_pk=1, at_bat_number=2, pitch_number=2, pitch_idx_in_pa=1, is_terminal=False),
        _row(batter=700, pitch_type="FF", description="swinging_strike", events="strikeout", zone=5,
             game_pk=1, at_bat_number=2, pitch_number=3, pitch_idx_in_pa=2, is_terminal=True),
    ]
    # PA3: CU walk (4 balls, no swings)
    rows += [
        _row(batter=700, pitch_type="CU", description="ball", events=np.nan, zone=11,
             game_pk=1, at_bat_number=3, pitch_number=1, pitch_idx_in_pa=0, is_terminal=False),
        _row(batter=700, pitch_type="CU", description="ball", events=np.nan, zone=11,
             game_pk=1, at_bat_number=3, pitch_number=2, pitch_idx_in_pa=1, is_terminal=False),
        _row(batter=700, pitch_type="CU", description="ball", events=np.nan, zone=11,
             game_pk=1, at_bat_number=3, pitch_number=3, pitch_idx_in_pa=2, is_terminal=False),
        _row(batter=700, pitch_type="CU", description="ball", events="walk", zone=11,
             game_pk=1, at_bat_number=3, pitch_number=4, pitch_idx_in_pa=3, is_terminal=True),
    ]
    # PA4: CU chase whiff (1 pitch out-of-zone, swing-and-miss)
    rows += [
        _row(batter=700, pitch_type="CU", description="swinging_strike", events="strikeout", zone=12,
             game_pk=1, at_bat_number=4, pitch_number=1, pitch_idx_in_pa=0, is_terminal=True),
    ]
    return pd.DataFrame(rows)


def test_batter_profile_overall_rates():
    df = add_mirrored_columns(_batter_profile_frame())
    vocabs = build_vocabs(df)
    tokens = tokenize_split(df, vocabs, split_name="train")
    profile = compute_batter_profile(tokens, df)

    overall_row = profile[profile["batter_id"] == vocabs["batter"]["700"]].iloc[0]
    # 4 PAs total: 2 strikeouts, 1 walk, 1 single → k_rate=0.5, bb_rate=0.25
    assert overall_row["pa_count"] == 4
    assert pytest.approx(overall_row["k_rate"]) == 0.5
    assert pytest.approx(overall_row["bb_rate"]) == 0.25
    # 11 pitches; swings = {FF foul, FF hit_into_play, FF ss, FF ss, CU ss} = 5
    assert overall_row["pitch_count"] == 11
    assert pytest.approx(overall_row["swing_rate"]) == 5 / 11
    # whiffs / swings = 3 / 5 = 0.6
    assert pytest.approx(overall_row["whiff_rate"]) == 0.6
    # contacts / swings = 2 / 5 = 0.4
    assert pytest.approx(overall_row["contact_rate"]) == 0.4
    # out_of_zone = 5 (CU PA3 4 balls + CU PA4 1 swing); chases = 1
    # chase_rate = 1 / 5 = 0.2
    assert pytest.approx(overall_row["chase_rate"]) == 0.2


def test_batter_profile_per_pitch_type_rates():
    df = add_mirrored_columns(_batter_profile_frame())
    vocabs = build_vocabs(df)
    tokens = tokenize_split(df, vocabs, split_name="train")
    profile = compute_batter_profile(tokens, df)

    bid = vocabs["batter"]["700"]
    ff_id = vocabs["pitch_type"]["FF"]
    cu_id = vocabs["pitch_type"]["CU"]

    ff_row = profile[(profile["batter_id"] == bid) & (profile["pitch_type_id"] == ff_id)].iloc[0]
    cu_row = profile[(profile["batter_id"] == bid) & (profile["pitch_type_id"] == cu_id)].iloc[0]

    # 6 FF pitches: 4 swings (foul, hip, ss, ss). swing_rate = 4/6
    assert ff_row["count"] == 6
    assert pytest.approx(ff_row["swing_rate_vs_type"]) == 4 / 6
    # 2 whiffs out of 4 swings on FF
    assert pytest.approx(ff_row["whiff_rate_vs_type"]) == 0.5

    # 5 CU pitches: 1 swing (the chase). swing_rate = 1/5
    assert cu_row["count"] == 5
    assert pytest.approx(cu_row["swing_rate_vs_type"]) == 1 / 5
    # 1 whiff / 1 swing = 1.0
    assert pytest.approx(cu_row["whiff_rate_vs_type"]) == 1.0


def test_batter_profile_low_sample_flag():
    df = add_mirrored_columns(_batter_profile_frame())
    vocabs = build_vocabs(df)
    tokens = tokenize_split(df, vocabs, split_name="train")
    profile = compute_batter_profile(tokens, df)
    # Synthetic frame is tiny — every (batter, pitch_type) has count < N_MIN_BATTER_PROFILE_PAS
    assert N_MIN_BATTER_PROFILE_PAS == 30
    assert profile["low_sample"].all()
