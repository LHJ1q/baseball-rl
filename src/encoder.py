"""State encoder modules for the Q-Transformer.

Two encoders, each mapping per-pitch features to a fixed-dim vector:

* :class:`PreActionEncoder` — encodes everything visible to the pitcher
  *before* throwing pitch ``i``: the decision-time state (count, outs,
  runners, score, batter, pitcher, strike-zone bounds, batter scouting
  profile). No action or outcome info — that would leak the target.

* :class:`PostActionEncoder` — encodes what was thrown and what happened on
  pitch ``i`` (action target, actual landing point, description, reward,
  terminal flag). Used as past-pitch context for future decisions.

The Q-Transformer (phase 7) interleaves these into a sequence
``[pre_0, post_0, pre_1, post_1, ...]`` so a causal mask correctly hides
each pitch's own action/outcome from its own decision step while still
exposing past pitches' actions and outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    """Hyperparameters shared by both encoders. Defaults match the Phase 6+7 spec."""
    d_model: int = 384
    d_player_emb: int = 96
    d_pitch_type_emb: int = 32
    d_description_emb: int = 16
    d_action_loc_emb: int = 24
    d_small_emb: int = 4
    dropout: float = 0.1


# --------------------------------------------------------------------------- #
# Field schemas — kept here so the dataset module can build matching tensors.
# --------------------------------------------------------------------------- #

# Pre-action: visible at decision time. Continuous tensor will hold these in
# this exact order (the dataset module enforces the order).
PRE_ACTION_CATEGORICAL_FIELDS = (
    "pitcher_id", "batter_id", "p_throws_id", "stand_id", "inning_topbot_id",
)

PRE_ACTION_CONTINUOUS_FIELDS = (
    # Count + state ints (cast to float, standardized in __init__)
    "balls", "strikes", "outs_when_up", "inning",
    "n_thruorder_pitcher", "pitch_idx_in_pa",
    # Score (collapsed) + zone bounds
    "score_diff", "sz_top", "sz_bot",
    # Runner booleans (already 0/1)
    "on_1b", "on_2b", "on_3b",
)

# Batter overall scouting features pulled from batter_profile (constant per batter
# regardless of pitch_type — joined once per token via batter_id).
BATTER_PROFILE_OVERALL_FIELDS = (
    "pa_count", "k_rate", "bb_rate", "xwoba_mean",
    "swing_rate", "whiff_rate", "contact_rate", "chase_rate",
    "batter_low_sample",
)

# Post-action: action target + execution outcome.
POST_ACTION_CATEGORICAL_FIELDS = (
    "pitch_type_id", "x_bin", "z_bin", "description_id",
)

POST_ACTION_CONTINUOUS_FIELDS = (
    "plate_x_mirrored", "plate_z",
    "reward_pitcher",
    "is_terminal",  # bool → float
)


# --------------------------------------------------------------------------- #
# Helper: standardization that's frozen at construction
# --------------------------------------------------------------------------- #


class _StandardizeFloat(nn.Module):
    """Centers and scales a (..., D) float tensor by frozen per-feature mean/std.

    Stats are stored as buffers (no gradient). If a stat std is zero, falls back
    to 1 to avoid division by zero.
    """

    def __init__(self, means: torch.Tensor, stds: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", means.float())
        self.register_buffer("std", stds.float().clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def _zero_stats(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity standardization (mean=0, std=1) — used when stats aren't supplied."""
    return torch.zeros(n), torch.ones(n)


# --------------------------------------------------------------------------- #
# Pre-action encoder
# --------------------------------------------------------------------------- #


class PreActionEncoder(nn.Module):
    """Encodes the decision-time state at pitch ``i`` to a ``d_model`` vector."""

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        cfg: EncoderConfig | None = None,
        cont_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        profile_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        c = self.cfg

        self.emb_pitcher = nn.Embedding(vocab_sizes["pitcher"], c.d_player_emb)
        self.emb_batter = nn.Embedding(vocab_sizes["batter"], c.d_player_emb)
        self.emb_p_throws = nn.Embedding(vocab_sizes["p_throws"], c.d_small_emb)
        self.emb_stand = nn.Embedding(vocab_sizes["stand"], c.d_small_emb)
        self.emb_inning_topbot = nn.Embedding(vocab_sizes["inning_topbot"], c.d_small_emb)

        n_cont = len(PRE_ACTION_CONTINUOUS_FIELDS)
        n_profile = len(BATTER_PROFILE_OVERALL_FIELDS)
        self.cont_norm = _StandardizeFloat(*(cont_stats or _zero_stats(n_cont)))
        self.profile_norm = _StandardizeFloat(*(profile_stats or _zero_stats(n_profile)))

        in_dim = (
            2 * c.d_player_emb
            + 3 * c.d_small_emb
            + n_cont
            + n_profile
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model, c.d_model),
        )

    def forward(
        self,
        cat: dict[str, torch.Tensor],   # each tensor: (B, T) int64
        cont: torch.Tensor,             # (B, T, len(PRE_ACTION_CONTINUOUS_FIELDS)) float32
        profile: torch.Tensor,          # (B, T, len(BATTER_PROFILE_OVERALL_FIELDS)) float32
    ) -> torch.Tensor:
        e_p = self.emb_pitcher(cat["pitcher_id"])
        e_b = self.emb_batter(cat["batter_id"])
        e_pt = self.emb_p_throws(cat["p_throws_id"])
        e_st = self.emb_stand(cat["stand_id"])
        e_ti = self.emb_inning_topbot(cat["inning_topbot_id"])

        cont_n = self.cont_norm(cont)
        profile_n = self.profile_norm(profile)

        h = torch.cat([e_p, e_b, e_pt, e_st, e_ti, cont_n, profile_n], dim=-1)
        return self.proj(h)


# --------------------------------------------------------------------------- #
# Post-action encoder
# --------------------------------------------------------------------------- #


class PostActionEncoder(nn.Module):
    """Encodes the action + execution outcome at pitch ``i`` to a ``d_model`` vector.

    The Q-Transformer feeds these as the in-between tokens so future decisions
    can attend to past pitches' actions and outcomes via causal self-attention.
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        n_x_bins: int,
        n_z_bins: int,
        cfg: EncoderConfig | None = None,
        cont_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        action_emb_modules: dict[str, nn.Module] | None = None,
    ):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        c = self.cfg

        # Action embeddings can be SHARED with the Q heads (so the Q heads use the
        # same action vocab as the past-action context). The QTransformer passes them
        # in via ``action_emb_modules`` so there's a single source of truth.
        if action_emb_modules is None:
            self.emb_pitch_type = nn.Embedding(vocab_sizes["pitch_type"], c.d_pitch_type_emb)
            self.emb_x_bin = nn.Embedding(n_x_bins, c.d_action_loc_emb)
            self.emb_z_bin = nn.Embedding(n_z_bins, c.d_action_loc_emb)
        else:
            self.emb_pitch_type = action_emb_modules["pitch_type"]
            self.emb_x_bin = action_emb_modules["x_bin"]
            self.emb_z_bin = action_emb_modules["z_bin"]

        self.emb_description = nn.Embedding(vocab_sizes["description"], c.d_description_emb)

        n_cont = len(POST_ACTION_CONTINUOUS_FIELDS)
        self.cont_norm = _StandardizeFloat(*(cont_stats or _zero_stats(n_cont)))

        in_dim = (
            c.d_pitch_type_emb
            + 2 * c.d_action_loc_emb
            + c.d_description_emb
            + n_cont
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, c.d_model),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.d_model, c.d_model),
        )

    def forward(
        self,
        cat: dict[str, torch.Tensor],   # int64 (B, T) for each of POST_ACTION_CATEGORICAL_FIELDS
        cont: torch.Tensor,             # (B, T, len(POST_ACTION_CONTINUOUS_FIELDS)) float32
    ) -> torch.Tensor:
        e_pt = self.emb_pitch_type(cat["pitch_type_id"])
        e_x = self.emb_x_bin(cat["x_bin"])
        e_z = self.emb_z_bin(cat["z_bin"])
        e_d = self.emb_description(cat["description_id"])
        cont_n = self.cont_norm(cont)
        h = torch.cat([e_pt, e_x, e_z, e_d, cont_n], dim=-1)
        return self.proj(h)
