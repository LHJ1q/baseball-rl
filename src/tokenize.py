"""Discretize the action space and emit per-pitch token rows + a static
pitcher arsenal lookup.

Phase 5 of the data pipeline. See CLAUDE.md § Phase 5 and
``docs/FILTER_RULES.md`` for the upstream contract.

Inputs:  ``data/splits/{train,val,test}.parquet``
Outputs: ``data/tokens/{train,val,test}.parquet``,
         ``data/tokens/pitcher_arsenal.parquet``,
         ``data/tokens/vocab.json``

Vocabularies and the pitcher arsenal are built **on the train split only** —
val/test must never feed back into the discrete index space or the per-pitcher
physics priors.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

X_BIN_LO, X_BIN_HI, N_X_BINS = -2.5, 2.5, 20
Z_BIN_LO, Z_BIN_HI, N_Z_BINS = -1.0, 6.0, 20

# Reserved index for out-of-train-vocab IDs in val/test (pitchers, batters,
# pitch_types we've never seen). The trainer attaches an "unknown" embedding
# row at this index.
UNK_ID = 0

# Arsenal entries with fewer than this many training-set pitches are flagged
# so the trainer can fall back to a global mean / pitcher-only prior.
N_MIN_ARSENAL_SAMPLES = 30

ARSENAL_RAW_COLUMNS = [
    "release_speed",
    "release_spin_rate",
    "spin_axis_mirrored",
    "pfx_x_mirrored",
    "pfx_z",
    "release_extension",
]

# Statcast description vocabulary for swing / whiff / contact accounting.
SWING_DESCRIPTIONS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
    "foul", "foul_tip", "foul_bunt", "bunt_foul_tip",
    "hit_into_play",
})
WHIFF_DESCRIPTIONS = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
CONTACT_DESCRIPTIONS = frozenset({"foul", "foul_tip", "foul_bunt", "bunt_foul_tip", "hit_into_play"})

# Statcast `zone` field: 1-9 strike zone, 11-14 out of zone (corners of the
# expanded box around the zone). Anything else (rare/null) we skip.
OUT_OF_ZONE_CODES = frozenset({11, 12, 13, 14})

# PA-terminating events that count as a strikeout / walk for batter rate stats.
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
BB_EVENTS = frozenset({"walk"})

N_MIN_BATTER_PROFILE_PAS = 30

# Categorical columns whose vocab we build from train and reuse on val/test.
CATEGORICAL_VOCAB_COLUMNS = (
    "pitch_type",
    "description",
    "inning_topbot",
    "p_throws",
    "stand",
    "batter",
    "pitcher",
)


# --------------------------------------------------------------------------- #
# Mirror-for-LHP
# --------------------------------------------------------------------------- #


def add_mirrored_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flip every left-right-signed column when ``p_throws == 'L'``.

    ``plate_x_mirrored`` already exists from the filter step. We add the rest
    here so the processed parquet stays in raw catcher's-perspective coordinates
    and mirroring is concentrated in one place.
    """
    sign = np.where(df["p_throws"].eq("L"), -1.0, 1.0)
    df = df.copy()
    df["pfx_x_mirrored"] = df["pfx_x"] * sign
    df["release_pos_x_mirrored"] = df["release_pos_x"] * sign
    df["vx0_mirrored"] = df["vx0"] * sign
    df["ax_mirrored"] = df["ax"] * sign
    # spin_axis is in degrees, clock-face convention → mirror via 360 - axis.
    df["spin_axis_mirrored"] = np.where(
        df["p_throws"].eq("L"),
        (360.0 - df["spin_axis"].to_numpy()) % 360.0,
        df["spin_axis"].to_numpy(),
    )
    return df


# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #


def _build_categorical_vocab(values: Iterable, *, reserve_unk: bool) -> dict:
    """Build a ``{value: int}`` vocab. If ``reserve_unk``, index 0 is reserved
    for the UNK token and real values start at 1; otherwise values are 0-indexed
    by descending frequency."""
    counts = pd.Series(list(values)).value_counts()
    sorted_values = counts.index.tolist()
    if reserve_unk:
        return {"<UNK>": UNK_ID, **{str(v): i + 1 for i, v in enumerate(sorted_values)}}
    return {str(v): i for i, v in enumerate(sorted_values)}


def build_vocabs(train_df: pd.DataFrame) -> dict:
    """Build every discrete vocabulary from the train split.

    Player IDs (batter/pitcher) reserve UNK_ID for unseen players in val/test.
    Categorical columns with closed sets (pitch_type, description, etc.) also
    reserve UNK_ID so future seasons with new categories degrade gracefully.
    """
    vocabs = {}
    vocabs["pitch_type"] = _build_categorical_vocab(train_df["pitch_type"], reserve_unk=True)
    vocabs["description"] = _build_categorical_vocab(train_df["description"], reserve_unk=True)
    vocabs["inning_topbot"] = _build_categorical_vocab(train_df["inning_topbot"], reserve_unk=False)
    vocabs["p_throws"] = _build_categorical_vocab(train_df["p_throws"], reserve_unk=False)
    vocabs["stand"] = _build_categorical_vocab(train_df["stand"], reserve_unk=False)
    vocabs["batter"] = _build_categorical_vocab(train_df["batter"], reserve_unk=True)
    vocabs["pitcher"] = _build_categorical_vocab(train_df["pitcher"], reserve_unk=True)
    vocabs["x_bin_edges"] = np.linspace(X_BIN_LO, X_BIN_HI, N_X_BINS + 1).tolist()
    vocabs["z_bin_edges"] = np.linspace(Z_BIN_LO, Z_BIN_HI, N_Z_BINS + 1).tolist()
    return vocabs


def _map_with_unk(series: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    """Map a series of values through a string-keyed vocab, falling back to UNK."""
    keys = series.astype(str).to_numpy()
    out = np.full(len(keys), UNK_ID, dtype=np.int32)
    for i, k in enumerate(keys):
        if k in vocab:
            out[i] = vocab[k]
    return out


# --------------------------------------------------------------------------- #
# Action discretization
# --------------------------------------------------------------------------- #


def _bin_index(values: np.ndarray, lo: float, hi: float, n_bins: int) -> tuple[np.ndarray, int]:
    """Clamp values to [lo, hi] and return integer bin index in [0, n_bins-1]
    plus the count of values that were clamped (for logging)."""
    clipped = np.clip(values, lo, hi)
    n_clamped = int(((values < lo) | (values > hi)).sum())
    width = (hi - lo) / n_bins
    idx = np.minimum(((clipped - lo) / width).astype(np.int32), n_bins - 1)
    return idx, n_clamped


# --------------------------------------------------------------------------- #
# Pitcher arsenal
# --------------------------------------------------------------------------- #


def compute_batter_profile(
    train_tokens: pd.DataFrame, train_with_features: pd.DataFrame
) -> pd.DataFrame:
    """Per-batter scouting profile, computed on the train split only.

    Two layers of stats per batter:

    * **Overall**: PA count, K%, BB%, swing%, whiff%-per-swing, contact%-per-swing,
      chase%-on-out-of-zone, mean xwOBA on batted balls.
    * **Per pitch type**: pitch count, swing%, whiff%-per-swing — picks up
      "this batter chases curves but lays off changeups" type tendencies.

    Returns one row per (``batter_id``, ``pitch_type_id``) with the per-type
    stats AND the batter-level overall stats denormalized in. Rows with
    ``count < N_MIN_BATTER_PROFILE_PAS`` (per-pitch-type) get a ``low_sample``
    flag for the trainer's fallback path.
    """
    df = train_with_features.copy()
    df["batter_id"] = train_tokens["batter_id"].to_numpy()
    df["pitch_type_id"] = train_tokens["pitch_type_id"].to_numpy()

    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["is_contact"] = df["description"].isin(CONTACT_DESCRIPTIONS)
    df["is_out_of_zone"] = df["zone"].isin(OUT_OF_ZONE_CODES)
    df["is_chase"] = df["is_swing"] & df["is_out_of_zone"]

    # PA-level aggregates: K%, BB%, xwOBA. One row per PA.
    pa = (
        df[df["events"].notna()]
        .drop_duplicates(["game_pk", "at_bat_number"])
        .copy()
    )
    pa["is_k"] = pa["events"].isin(K_EVENTS)
    pa["is_bb"] = pa["events"].isin(BB_EVENTS)
    pa_overall = pa.groupby("batter_id").agg(
        pa_count=("events", "size"),
        k_rate=("is_k", "mean"),
        bb_rate=("is_bb", "mean"),
        xwoba_mean=("estimated_woba_using_speedangle", "mean"),
    )
    # xwoba_mean is null for batters with no batted balls (all K/BB/HBP). Fill with 0.0.
    pa_overall["xwoba_mean"] = pa_overall["xwoba_mean"].astype("float64").fillna(0.0)

    # Pitch-level aggregates: swing/whiff/contact/chase rates.
    pitch_overall = df.groupby("batter_id").agg(
        pitch_count=("description", "size"),
        n_swings=("is_swing", "sum"),
        n_whiffs=("is_whiff", "sum"),
        n_contacts=("is_contact", "sum"),
        n_out_of_zone=("is_out_of_zone", "sum"),
        n_chases=("is_chase", "sum"),
    )
    def _safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
        # Cast to plain float64 first so 0/0 → np.nan (numpy), then fillna(0.0)
        # Pandas Float64 nullable dtype can mix <NA> and np.nan inconsistently.
        return (num.astype("float64") / den.astype("float64")).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pitch_overall["swing_rate"] = _safe_rate(pitch_overall["n_swings"], pitch_overall["pitch_count"])
    pitch_overall["whiff_rate"] = _safe_rate(pitch_overall["n_whiffs"], pitch_overall["n_swings"])
    pitch_overall["contact_rate"] = _safe_rate(pitch_overall["n_contacts"], pitch_overall["n_swings"])
    pitch_overall["chase_rate"] = _safe_rate(pitch_overall["n_chases"], pitch_overall["n_out_of_zone"])
    pitch_overall = pitch_overall.drop(columns=["n_swings", "n_whiffs", "n_contacts", "n_out_of_zone", "n_chases"])

    overall = pa_overall.join(pitch_overall, how="outer")

    # Per-(batter, pitch_type) stats.
    pp = df.groupby(["batter_id", "pitch_type_id"]).agg(
        count=("description", "size"),
        n_swings=("is_swing", "sum"),
        n_whiffs=("is_whiff", "sum"),
    )
    pp["swing_rate_vs_type"] = _safe_rate(pp["n_swings"], pp["count"])
    pp["whiff_rate_vs_type"] = _safe_rate(pp["n_whiffs"], pp["n_swings"])
    pp = pp.drop(columns=["n_swings", "n_whiffs"])

    profile = pp.join(overall, on="batter_id", how="left").reset_index()
    profile["low_sample"] = profile["count"] < N_MIN_BATTER_PROFILE_PAS
    return profile


def compute_pitcher_arsenal(train_tokens: pd.DataFrame, train_with_features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(pitcher_id, pitch_type_id) physics on the train split only."""
    df = train_with_features[["pitcher", "pitch_type", *ARSENAL_RAW_COLUMNS]].copy()
    # Use the same int IDs as in the token file so the trainer can join cleanly.
    df["pitcher_id"] = train_tokens["pitcher_id"].to_numpy()
    df["pitch_type_id"] = train_tokens["pitch_type_id"].to_numpy()
    grouped = df.groupby(["pitcher_id", "pitch_type_id"], sort=True)
    agg = grouped[ARSENAL_RAW_COLUMNS].agg(["mean", "std", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    # Every column shares the same count; collapse to a single 'count' field.
    agg = agg.rename(columns={f"{ARSENAL_RAW_COLUMNS[0]}_count": "count"})
    agg = agg.drop(columns=[c for c in agg.columns if c.endswith("_count")])
    # std() on a single sample is NaN — fill with 0.0 so the arsenal table is finite.
    std_cols = [c for c in agg.columns if c.endswith("_std")]
    agg[std_cols] = agg[std_cols].astype("float64").fillna(0.0)
    agg["low_sample"] = agg["count"] < N_MIN_ARSENAL_SAMPLES
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# Per-pitch tokenization
# --------------------------------------------------------------------------- #

TOKEN_COLUMNS_ORDER = [
    # PA static (denormalized onto every pitch row)
    "game_date", "game_pk", "at_bat_number",
    "batter_id", "pitcher_id", "p_throws_id", "stand_id",
    "inning", "inning_topbot_id",
    "home_score", "away_score", "bat_score", "fld_score",
    "on_1b", "on_2b", "on_3b",
    "n_thruorder_pitcher",
    # Decision-time state
    "pitch_idx_in_pa",
    "balls", "strikes", "outs_when_up",
    "sz_top", "sz_bot",
    # Action target (what the policy predicts)
    "pitch_type_id", "x_bin", "z_bin",
    # Execution outcome (visible to the *next* timestep, not the current one)
    "plate_x_mirrored", "plate_z",
    "description_id",
    "reward_pitcher",
    "is_terminal",
]


def tokenize_split(df: pd.DataFrame, vocabs: dict, *, split_name: str) -> pd.DataFrame:
    """Convert one filtered + mirrored split into the token row format."""
    n0 = len(df)
    out = pd.DataFrame(index=df.index)

    out["game_date"] = pd.to_datetime(df["game_date"])
    out["game_pk"] = df["game_pk"].astype(np.int64)
    out["at_bat_number"] = df["at_bat_number"].astype(np.int32)

    out["batter_id"] = _map_with_unk(df["batter"], vocabs["batter"])
    out["pitcher_id"] = _map_with_unk(df["pitcher"], vocabs["pitcher"])
    out["p_throws_id"] = _map_with_unk(df["p_throws"], vocabs["p_throws"])
    out["stand_id"] = _map_with_unk(df["stand"], vocabs["stand"])

    out["inning"] = df["inning"].astype(np.int16)
    out["inning_topbot_id"] = _map_with_unk(df["inning_topbot"], vocabs["inning_topbot"])

    for col in ("home_score", "away_score", "bat_score", "fld_score"):
        out[col] = df[col].astype(np.int16)

    out["on_1b"] = df["on_1b"].notna().to_numpy()
    out["on_2b"] = df["on_2b"].notna().to_numpy()
    out["on_3b"] = df["on_3b"].notna().to_numpy()

    out["n_thruorder_pitcher"] = df["n_thruorder_pitcher"].astype(np.int16)

    out["pitch_idx_in_pa"] = df["pitch_idx_in_pa"].astype(np.int16)
    out["balls"] = df["balls"].astype(np.int8)
    out["strikes"] = df["strikes"].astype(np.int8)
    out["outs_when_up"] = df["outs_when_up"].astype(np.int8)
    out["sz_top"] = df["sz_top"].astype(np.float32)
    out["sz_bot"] = df["sz_bot"].astype(np.float32)

    out["pitch_type_id"] = _map_with_unk(df["pitch_type"], vocabs["pitch_type"])
    x_idx, n_x_clamp = _bin_index(df["plate_x_mirrored"].to_numpy(), X_BIN_LO, X_BIN_HI, N_X_BINS)
    z_idx, n_z_clamp = _bin_index(df["plate_z"].to_numpy(), Z_BIN_LO, Z_BIN_HI, N_Z_BINS)
    out["x_bin"] = x_idx
    out["z_bin"] = z_idx

    out["plate_x_mirrored"] = df["plate_x_mirrored"].astype(np.float32)
    out["plate_z"] = df["plate_z"].astype(np.float32)
    out["description_id"] = _map_with_unk(df["description"], vocabs["description"])
    out["reward_pitcher"] = df["reward_pitcher"].astype(np.float32)
    out["is_terminal"] = df["is_terminal"].astype(bool)

    logger.info(
        "[%s] tokenized rows=%d | x_bin clamps=%d (%.3f%%) | z_bin clamps=%d (%.3f%%)",
        split_name, n0,
        n_x_clamp, 100 * n_x_clamp / n0,
        n_z_clamp, 100 * n_z_clamp / n0,
    )

    out_of_vocab = {
        "batter": int((out["batter_id"] == UNK_ID).sum()),
        "pitcher": int((out["pitcher_id"] == UNK_ID).sum()),
        "pitch_type": int((out["pitch_type_id"] == UNK_ID).sum()),
        "description": int((out["description_id"] == UNK_ID).sum()),
    }
    if any(v > 0 for v in out_of_vocab.values()):
        logger.info("[%s] out-of-train-vocab rows: %s", split_name, out_of_vocab)

    return out[TOKEN_COLUMNS_ORDER]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


ARSENAL_HEAD_FIELDS = (
    "count", "low_sample",
    "release_speed_mean", "release_speed_std",
    "release_spin_rate_mean", "release_spin_rate_std",
    "spin_axis_mirrored_mean", "spin_axis_mirrored_std",
    "pfx_x_mirrored_mean", "pfx_x_mirrored_std",
    "pfx_z_mean", "pfx_z_std",
    "release_extension_mean", "release_extension_std",
)

BATTER_PER_TYPE_HEAD_FIELDS = (
    "count", "low_sample",
    "swing_rate_vs_type", "whiff_rate_vs_type",
)


def compute_feature_stats(
    train_tokens: pd.DataFrame, batter_profile: pd.DataFrame, pitcher_arsenal: pd.DataFrame | None = None
) -> dict[str, dict[str, list[float]]]:
    """Per-column mean/std for the model's continuous inputs, computed on train only.

    Used by the encoder to standardize inputs to roughly zero mean / unit variance
    before they hit the MLP. Saved alongside ``vocab.json`` as ``feature_stats.json``.
    Booleans (``on_1b/2b/3b``, ``low_sample`` flags) are kept in the stats so they get
    centered too, but the encoder can also leave them raw — both work.
    """
    pre_cont_fields = [
        "balls", "strikes", "outs_when_up", "inning",
        "n_thruorder_pitcher", "pitch_idx_in_pa",
        "score_diff", "sz_top", "sz_bot",
        "on_1b", "on_2b", "on_3b",
    ]
    post_cont_fields = [
        "plate_x_mirrored", "plate_z",
        "reward_pitcher", "is_terminal",
    ]
    profile_overall_fields = [
        "pa_count", "k_rate", "bb_rate", "xwoba_mean",
        "swing_rate", "whiff_rate", "contact_rate", "chase_rate",
        "low_sample",
    ]

    train = train_tokens.copy()
    train["score_diff"] = (train["bat_score"] - train["fld_score"]).astype("int16")

    def _stats(df: pd.DataFrame, cols: list[str]) -> dict[str, list[float]]:
        # Defensive: nanmean/nanstd in case any rate column has a sneaky NaN.
        # Also assert no NaN-fraction > 1% — if it is, the upstream computation
        # has a real bug and we shouldn't paper over it with nanmean.
        means, stds = [], []
        for c in cols:
            arr = df[c].astype("float64").to_numpy()
            nan_frac = float(np.isnan(arr).mean())
            if nan_frac > 0.01:
                raise ValueError(f"feature_stats: column {c!r} has {nan_frac:.2%} NaN — fix upstream")
            m = float(np.nanmean(arr)) if len(arr) else 0.0
            s = float(np.nanstd(arr)) if len(arr) else 1.0
            if not np.isfinite(m):
                m = 0.0
            means.append(m)
            stds.append(s if s > 1e-6 else 1.0)
        return {"mean": means, "std": stds, "fields": cols}

    profile_overall = batter_profile.drop_duplicates("batter_id").rename(columns={"low_sample": "batter_low_sample"})
    # Profile stats use the unique-batter table, not denormalized per-pitch (correct — one obs per batter).
    profile_overall_renamed_fields = ["pa_count", "k_rate", "bb_rate", "xwoba_mean",
                                       "swing_rate", "whiff_rate", "contact_rate", "chase_rate",
                                       "batter_low_sample"]
    out = {
        "pre_cont": _stats(train, pre_cont_fields),
        "post_cont": _stats(train, post_cont_fields),
        "profile": _stats(profile_overall, profile_overall_renamed_fields),
    }
    if pitcher_arsenal is not None:
        # Arsenal has count + low_sample + 6 means + 6 stds = 14 fields per (pitcher, pitch_type).
        arsenal_df = pitcher_arsenal.copy()
        arsenal_df["low_sample"] = arsenal_df["low_sample"].astype("float64")
        out["arsenal_head"] = _stats(arsenal_df, list(ARSENAL_HEAD_FIELDS))

    # Batter per-(batter, pitch_type) features (count + low_sample + 2 rates).
    batter_pt_df = batter_profile.copy()
    batter_pt_df["low_sample"] = batter_pt_df["low_sample"].astype("float64")
    out["batter_per_type_head"] = _stats(batter_pt_df, list(BATTER_PER_TYPE_HEAD_FIELDS))

    return out


def process_all_splits(splits_dir: Path, tokens_dir: Path) -> dict[str, Path]:
    """End-to-end Phase 5: build vocabs from train, tokenize all splits, write artifacts."""
    tokens_dir.mkdir(parents=True, exist_ok=True)

    raw = {name: pd.read_parquet(splits_dir / f"{name}.parquet", engine="pyarrow") for name in ("train", "val", "test")}
    mirrored = {name: add_mirrored_columns(df) for name, df in raw.items()}

    logger.info("building vocabs from train split (rows=%d)", len(mirrored["train"]))
    vocabs = build_vocabs(mirrored["train"])

    written: dict[str, Path] = {}
    train_tokens: pd.DataFrame | None = None
    for name, df in mirrored.items():
        toks = tokenize_split(df, vocabs, split_name=name)
        path = tokens_dir / f"{name}.parquet"
        toks.to_parquet(path, engine="pyarrow", index=False)
        logger.info("wrote %s rows=%d cols=%d", path.name, len(toks), toks.shape[1])
        written[name] = path
        if name == "train":
            train_tokens = toks

    assert train_tokens is not None
    arsenal = compute_pitcher_arsenal(train_tokens, mirrored["train"])
    arsenal_path = tokens_dir / "pitcher_arsenal.parquet"
    arsenal.to_parquet(arsenal_path, engine="pyarrow", index=False)
    logger.info(
        "wrote %s rows=%d (pitcher,pitch_type) groups, low_sample=%d",
        arsenal_path.name, len(arsenal), int(arsenal["low_sample"].sum()),
    )
    written["pitcher_arsenal"] = arsenal_path

    batter_profile = compute_batter_profile(train_tokens, mirrored["train"])
    batter_profile_path = tokens_dir / "batter_profile.parquet"
    batter_profile.to_parquet(batter_profile_path, engine="pyarrow", index=False)
    logger.info(
        "wrote %s rows=%d (batter,pitch_type) groups, low_sample=%d, batters=%d",
        batter_profile_path.name,
        len(batter_profile),
        int(batter_profile["low_sample"].sum()),
        batter_profile["batter_id"].nunique(),
    )
    written["batter_profile"] = batter_profile_path

    vocab_path = tokens_dir / "vocab.json"
    with vocab_path.open("w") as f:
        json.dump(vocabs, f, indent=2, sort_keys=False)
    logger.info("wrote %s", vocab_path.name)
    written["vocab"] = vocab_path

    feature_stats = compute_feature_stats(train_tokens, batter_profile, pitcher_arsenal=arsenal)
    stats_path = tokens_dir / "feature_stats.json"
    with stats_path.open("w") as f:
        json.dump(feature_stats, f, indent=2, sort_keys=False)
    logger.info("wrote %s", stats_path.name)
    written["feature_stats"] = stats_path

    return written
