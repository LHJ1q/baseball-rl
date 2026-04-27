"""PA-grouped dataloader for the Q-Transformer.

Loads ``data/tokens/{split}.parquet``, joins ``batter_profile.parquet`` (overall
batter stats only — per-pitch-type stats join at Q-head time later), groups
rows by ``(game_pk, at_bat_number)``, sorts within PA by ``pitch_idx_in_pa``,
and produces fixed-length-padded batches with attention/PA-length masks.

This is the dataloader used by both the Macbook smoke test and the eventual
Linux/Colab trainer (phase 8).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.encoder import (
    BATTER_PROFILE_OVERALL_FIELDS,
    POST_ACTION_CATEGORICAL_FIELDS,
    POST_ACTION_CONTINUOUS_FIELDS,
    PRE_ACTION_CATEGORICAL_FIELDS,
    PRE_ACTION_CONTINUOUS_FIELDS,
)
from src.tokenize import ARSENAL_HEAD_FIELDS, BATTER_PER_TYPE_HEAD_FIELDS


# --------------------------------------------------------------------------- #
# PA-level batch
# --------------------------------------------------------------------------- #


@dataclass
class PABatch:
    """One mini-batch of plate appearances. All tensors have a leading batch dim
    and a time dim padded to ``max(pa_lengths)``."""

    pre_cat: dict[str, torch.Tensor]    # {field: (B, T) int64}
    pre_cont: torch.Tensor              # (B, T, len(PRE_ACTION_CONTINUOUS_FIELDS))
    profile: torch.Tensor               # (B, T, len(BATTER_PROFILE_OVERALL_FIELDS))
    post_cat: dict[str, torch.Tensor]   # {field: (B, T) int64}
    post_cont: torch.Tensor             # (B, T, len(POST_ACTION_CONTINUOUS_FIELDS))
    reward: torch.Tensor                # (B, T) float — reward_pitcher per pitch
    is_terminal: torch.Tensor           # (B, T) bool — last pitch of PA
    pa_lengths: torch.Tensor            # (B,) int — true length per PA
    valid_mask: torch.Tensor            # (B, T) bool — True where t < pa_lengths[b]
    # Per-(pitcher, pitch_type) and per-(batter, pitch_type) lookup features
    # for the type/x/z heads. Layout: (B, T, n_pitch_types, k_features).
    arsenal_per_type: torch.Tensor      # (B, T, n_pitch_types, len(ARSENAL_HEAD_FIELDS))
    batter_per_type: torch.Tensor       # (B, T, n_pitch_types, len(BATTER_PER_TYPE_HEAD_FIELDS))

    def to(self, device: torch.device | str) -> "PABatch":
        def _move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return {k: v.to(device) for k, v in x.items()}
        return PABatch(
            pre_cat=_move(self.pre_cat),
            pre_cont=_move(self.pre_cont),
            profile=_move(self.profile),
            post_cat=_move(self.post_cat),
            post_cont=_move(self.post_cont),
            reward=_move(self.reward),
            is_terminal=_move(self.is_terminal),
            pa_lengths=_move(self.pa_lengths),
            valid_mask=_move(self.valid_mask),
            arsenal_per_type=_move(self.arsenal_per_type),
            batter_per_type=_move(self.batter_per_type),
        )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class PitchPADataset(Dataset):
    """One PA = one item. ``__getitem__`` returns a dict of per-pitch tensors of
    shape ``(T,)`` or ``(T, D)`` where ``T`` is this PA's length."""

    def __init__(
        self,
        tokens_path: Path,
        batter_profile_path: Path,
        pitcher_arsenal_path: Path | None = None,
        n_pitch_types: int | None = None,
    ):
        df = pd.read_parquet(tokens_path, engine="pyarrow")
        # score_diff is a derived field used by the encoder (see Phase 5 § "Future
        # tokenization improvements" — collapsing scores). Compute on the fly here
        # so the parquet stays minimal.
        df["score_diff"] = (df["bat_score"] - df["fld_score"]).astype(np.int16)
        df = df.sort_values(
            ["game_pk", "at_bat_number", "pitch_idx_in_pa"], kind="mergesort"
        ).reset_index(drop=True)

        # Pull batter overall stats from batter_profile (one row per batter is enough
        # since the overall fields are denormalized across pitch_types).
        profile = pd.read_parquet(batter_profile_path, engine="pyarrow")
        profile_overall = (
            profile.drop_duplicates("batter_id")
            .set_index("batter_id")
        )
        profile_overall = profile_overall.rename(columns={"low_sample": "batter_low_sample"})

        # Reindex against every batter_id in tokens; missing batters (UNK or new in val/test)
        # get zeros + low_sample = True so the model knows to fall back.
        unique_batters = df["batter_id"].unique()
        cols = [c for c in BATTER_PROFILE_OVERALL_FIELDS if c in profile_overall.columns]
        # batter_low_sample may not be a column directly — handle gracefully.
        missing_cols = [c for c in BATTER_PROFILE_OVERALL_FIELDS if c not in profile_overall.columns]
        for c in missing_cols:
            profile_overall[c] = 0.0
        joined = profile_overall.reindex(unique_batters)[list(BATTER_PROFILE_OVERALL_FIELDS)]
        # Mark imputed (missing-in-train) batters as low-sample.
        was_missing = joined.isna().any(axis=1)
        joined = joined.fillna(0.0)
        joined.loc[was_missing, "batter_low_sample"] = True
        joined["batter_low_sample"] = joined["batter_low_sample"].astype(float)
        self._batter_profile_lookup = joined.astype(np.float32)

        # ------------------------------------------------------------------ #
        # Per-(pitcher, pitch_type) arsenal lookup table — built into a dense
        # (n_pitchers, n_pitch_types, k_arsenal) numpy array for fast gather.
        # Missing entries (pitcher never threw that pitch type in train) get
        # zeros + low_sample=1.0 so the model can detect "unknown" via the flag.
        # ------------------------------------------------------------------ #
        self._n_pitch_types = (
            n_pitch_types if n_pitch_types is not None else int(df["pitch_type_id"].max()) + 1
        )
        n_pitchers = int(df["pitcher_id"].max()) + 1
        n_batters = int(df["batter_id"].max()) + 1
        k_arsenal = len(ARSENAL_HEAD_FIELDS)
        k_batter_pt = len(BATTER_PER_TYPE_HEAD_FIELDS)

        self._arsenal_lookup = np.zeros(
            (n_pitchers, self._n_pitch_types, k_arsenal), dtype=np.float32
        )
        # Default low_sample = True (1.0) for every missing entry — index 1 in ARSENAL_HEAD_FIELDS.
        low_sample_idx = ARSENAL_HEAD_FIELDS.index("low_sample")
        self._arsenal_lookup[:, :, low_sample_idx] = 1.0
        if pitcher_arsenal_path is not None:
            arsenal = pd.read_parquet(pitcher_arsenal_path, engine="pyarrow")
            arsenal_arr = arsenal[list(ARSENAL_HEAD_FIELDS)].to_numpy(dtype=np.float32)
            arsenal_pid = arsenal["pitcher_id"].to_numpy(dtype=np.int64)
            arsenal_pt = arsenal["pitch_type_id"].to_numpy(dtype=np.int64)
            valid = (arsenal_pid < n_pitchers) & (arsenal_pt < self._n_pitch_types)
            self._arsenal_lookup[arsenal_pid[valid], arsenal_pt[valid]] = arsenal_arr[valid]

        # Per-(batter, pitch_type) profile lookup, same pattern.
        self._batter_pt_lookup = np.zeros(
            (n_batters, self._n_pitch_types, k_batter_pt), dtype=np.float32
        )
        bpt_low_idx = BATTER_PER_TYPE_HEAD_FIELDS.index("low_sample")
        self._batter_pt_lookup[:, :, bpt_low_idx] = 1.0
        bpt_arr = profile[list(BATTER_PER_TYPE_HEAD_FIELDS)].to_numpy(dtype=np.float32)
        bpt_bid = profile["batter_id"].to_numpy(dtype=np.int64)
        bpt_pt = profile["pitch_type_id"].to_numpy(dtype=np.int64)
        valid_b = (bpt_bid < n_batters) & (bpt_pt < self._n_pitch_types)
        self._batter_pt_lookup[bpt_bid[valid_b], bpt_pt[valid_b]] = bpt_arr[valid_b]

        # Group by PA, store row indices per group for fast __getitem__.
        groups = df.groupby(["game_pk", "at_bat_number"], sort=False)
        self._pa_keys = list(groups.groups.keys())
        self._pa_indices: list[np.ndarray] = [groups.groups[k].to_numpy() for k in self._pa_keys]
        self._df = df

    def __len__(self) -> int:
        return len(self._pa_keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rows = self._df.iloc[self._pa_indices[idx]]
        T = len(rows)
        out: dict[str, torch.Tensor] = {}

        for f in PRE_ACTION_CATEGORICAL_FIELDS:
            out[f"pre_cat__{f}"] = torch.from_numpy(rows[f].to_numpy(dtype=np.int64))
        out["pre_cont"] = torch.from_numpy(
            np.stack([rows[f].to_numpy(dtype=np.float32) for f in PRE_ACTION_CONTINUOUS_FIELDS], axis=1)
        )

        # Batter profile join — index into the precomputed lookup table.
        batter_ids = rows["batter_id"].to_numpy()
        profile_arr = self._batter_profile_lookup.loc[batter_ids].to_numpy(dtype=np.float32)
        out["profile"] = torch.from_numpy(profile_arr)

        for f in POST_ACTION_CATEGORICAL_FIELDS:
            out[f"post_cat__{f}"] = torch.from_numpy(rows[f].to_numpy(dtype=np.int64))
        out["post_cont"] = torch.from_numpy(
            np.stack([rows[f].to_numpy(dtype=np.float32) for f in POST_ACTION_CONTINUOUS_FIELDS], axis=1)
        )

        out["reward"] = torch.from_numpy(rows["reward_pitcher"].to_numpy(dtype=np.float32))
        out["is_terminal"] = torch.from_numpy(rows["is_terminal"].to_numpy(dtype=bool))
        out["pa_length"] = torch.tensor(T, dtype=torch.int32)

        # Gather per-type lookup features for this PA.
        pitcher_ids = rows["pitcher_id"].to_numpy(dtype=np.int64)
        out["arsenal_per_type"] = torch.from_numpy(self._arsenal_lookup[pitcher_ids])
        out["batter_per_type"] = torch.from_numpy(self._batter_pt_lookup[batter_ids])

        return out


# --------------------------------------------------------------------------- #
# Collate function — pads to max length in batch and builds masks
# --------------------------------------------------------------------------- #


def pa_collate(items: list[dict[str, torch.Tensor]]) -> PABatch:
    """Pad variable-length PAs to the batch max length and stack into a PABatch."""
    B = len(items)
    pa_lengths = torch.stack([it["pa_length"] for it in items])
    T_max = int(pa_lengths.max().item())

    n_pre_cont = len(PRE_ACTION_CONTINUOUS_FIELDS)
    n_post_cont = len(POST_ACTION_CONTINUOUS_FIELDS)
    n_profile = len(BATTER_PROFILE_OVERALL_FIELDS)
    n_pitch_types = items[0]["arsenal_per_type"].shape[1]
    k_arsenal = items[0]["arsenal_per_type"].shape[2]
    k_batter_pt = items[0]["batter_per_type"].shape[2]

    pre_cat: dict[str, torch.Tensor] = {
        f: torch.zeros(B, T_max, dtype=torch.int64) for f in PRE_ACTION_CATEGORICAL_FIELDS
    }
    pre_cont = torch.zeros(B, T_max, n_pre_cont, dtype=torch.float32)
    profile = torch.zeros(B, T_max, n_profile, dtype=torch.float32)
    post_cat: dict[str, torch.Tensor] = {
        f: torch.zeros(B, T_max, dtype=torch.int64) for f in POST_ACTION_CATEGORICAL_FIELDS
    }
    post_cont = torch.zeros(B, T_max, n_post_cont, dtype=torch.float32)
    reward = torch.zeros(B, T_max, dtype=torch.float32)
    is_terminal = torch.zeros(B, T_max, dtype=torch.bool)
    valid_mask = torch.zeros(B, T_max, dtype=torch.bool)
    arsenal_per_type = torch.zeros(B, T_max, n_pitch_types, k_arsenal, dtype=torch.float32)
    batter_per_type = torch.zeros(B, T_max, n_pitch_types, k_batter_pt, dtype=torch.float32)

    for b, it in enumerate(items):
        T = int(it["pa_length"].item())
        valid_mask[b, :T] = True
        for f in PRE_ACTION_CATEGORICAL_FIELDS:
            pre_cat[f][b, :T] = it[f"pre_cat__{f}"]
        pre_cont[b, :T] = it["pre_cont"]
        profile[b, :T] = it["profile"]
        for f in POST_ACTION_CATEGORICAL_FIELDS:
            post_cat[f][b, :T] = it[f"post_cat__{f}"]
        post_cont[b, :T] = it["post_cont"]
        reward[b, :T] = it["reward"]
        is_terminal[b, :T] = it["is_terminal"]
        arsenal_per_type[b, :T] = it["arsenal_per_type"]
        batter_per_type[b, :T] = it["batter_per_type"]

    return PABatch(
        pre_cat=pre_cat,
        pre_cont=pre_cont,
        profile=profile,
        post_cat=post_cat,
        post_cont=post_cont,
        reward=reward,
        is_terminal=is_terminal,
        pa_lengths=pa_lengths.to(torch.int64),
        valid_mask=valid_mask,
        arsenal_per_type=arsenal_per_type,
        batter_per_type=batter_per_type,
    )


# --------------------------------------------------------------------------- #
# Vocab loader
# --------------------------------------------------------------------------- #


def load_vocab_sizes(vocab_path: Path) -> dict[str, int]:
    """Return ``{field: vocab_size}`` from ``data/tokens/vocab.json``."""
    with vocab_path.open() as f:
        vocab = json.load(f)
    return {
        "pitch_type": len(vocab["pitch_type"]),
        "description": len(vocab["description"]),
        "inning_topbot": len(vocab["inning_topbot"]),
        "p_throws": len(vocab["p_throws"]),
        "stand": len(vocab["stand"]),
        "batter": len(vocab["batter"]),
        "pitcher": len(vocab["pitcher"]),
    }
