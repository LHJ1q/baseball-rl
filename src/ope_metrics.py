"""Phase 9 Part A — Behavioral / distributional analysis of the learned policy.

Compares the learned policy's actions against what real MLB pitchers actually
threw, on the same states. Outputs a portrait of where the two agree, where
they diverge, and along which dimensions the divergence concentrates.

This is a *descriptive* analysis — it does NOT estimate the value of the
learned policy. For that, see :mod:`src.fqe` (Part B).

Aggregate metrics computed:

* **Pitch-type top-1 / top-3 agreement** — fraction of pitches where the
  policy's argmax (or top-3) pitch_type matches what was actually thrown.
* **Coarse-zone agreement** — bin both policy's and actual location into a
  4×4 macro grid (vs the model's 20×20), measure (type, macro-zone) match.
* **Spatial distance** — Euclidean distance in feet between the policy's
  predicted bin center and the actual landing point. Mean / median / p75 /
  fraction within 6 inches.
* **Pitch-type marginal KL** — KL[π_learned ∥ behavior] over the marginal
  pitch_type distribution.
* **Distribution histograms** — counts per pitch_type / x_bin / z_bin under
  each policy.
* **Pitcher-blind variants** — same metrics with pitcher embedding zeroed,
  to measure how much the policy depends on player identity.

Per-segment breakdowns by ``(balls, strikes)`` count, batter handedness ×
pitcher handedness, and PA length are also exposed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.dataset import PABatch
from src.eval import _zeroed_pitcher_embedding
from src.qtransformer import QTransformer
from src.tokenize import N_X_BINS, N_Z_BINS, X_BIN_HI, X_BIN_LO, Z_BIN_HI, Z_BIN_LO

# Coarse macro-zone grid for "did the model anticipate roughly where the
# pitch went" agreement (less stringent than the 20×20 model bins).
N_MACRO_X = 4
N_MACRO_Y = 4
WITHIN_THRESHOLD_FT = 0.5  # 6 inches


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def bin_center(bin_idx: torch.Tensor, lo: float, hi: float, n_bins: int) -> torch.Tensor:
    """Center coordinate (in feet) of each bin index."""
    width = (hi - lo) / n_bins
    return lo + (bin_idx.float() + 0.5) * width


def to_macro_zone(x_bin: torch.Tensor, z_bin: torch.Tensor) -> torch.Tensor:
    """Map fine 20×20 bins to coarse 4×4 macro zones (returns flat ID 0..15)."""
    macro_x = (x_bin.long() * N_MACRO_X) // N_X_BINS
    macro_z = (z_bin.long() * N_MACRO_Y) // N_Z_BINS
    return macro_x * N_MACRO_Y + macro_z


# --------------------------------------------------------------------------- #
# Per-batch inference
# --------------------------------------------------------------------------- #


@dataclass
class PerPitchPredictions:
    """Stacked predictions for one batch — flattened to (N_valid_pitches, ...)."""
    # Policy predictions
    policy_pitch_type: torch.Tensor              # (N,)
    policy_x_bin: torch.Tensor                   # (N,)
    policy_z_bin: torch.Tensor                   # (N,)
    policy_top3_pitch_type: torch.Tensor         # (N, 3)
    # Actual (behavior) actions
    actual_pitch_type: torch.Tensor              # (N,)
    actual_x_bin: torch.Tensor                   # (N,)
    actual_z_bin: torch.Tensor                   # (N,)
    actual_plate_x_mirrored: torch.Tensor        # (N,) continuous landing
    actual_plate_z: torch.Tensor                 # (N,) continuous landing
    # State context (for segment breakdowns)
    balls: torch.Tensor                          # (N,)
    strikes: torch.Tensor                        # (N,)
    p_throws_id: torch.Tensor                    # (N,)
    stand_id: torch.Tensor                       # (N,)
    pa_length: torch.Tensor                      # (N,) — length of the PA each pitch belongs to
    pitch_idx_in_pa: torch.Tensor                # (N,)


@torch.no_grad()
def predict_batch(
    model: QTransformer, batch: PABatch, *, top_k: int = 3
) -> PerPitchPredictions:
    """Run the policy + extract actuals on one batch. Returns flattened
    predictions over only valid (non-padded) pitches."""
    out = model.policy(batch, return_logits=True)
    valid = batch.valid_mask  # (B, T)

    # Top-K pitch types. Mask invalid logits to -inf first.
    q_type_logits = out["q_type_logits"]
    top3_idx = q_type_logits.topk(min(top_k, q_type_logits.shape[-1]), dim=-1).indices  # (B, T, k)

    # Compute per-PA length (broadcast to per-pitch).
    pa_len = batch.pa_lengths.unsqueeze(1).expand_as(valid)

    flat = lambda x: x[valid]  # noqa: E731
    return PerPitchPredictions(
        policy_pitch_type=flat(out["pitch_type"]),
        policy_x_bin=flat(out["x_bin"]),
        policy_z_bin=flat(out["z_bin"]),
        policy_top3_pitch_type=top3_idx[valid],
        actual_pitch_type=flat(batch.post_cat["pitch_type_id"]),
        actual_x_bin=flat(batch.post_cat["x_bin"]),
        actual_z_bin=flat(batch.post_cat["z_bin"]),
        actual_plate_x_mirrored=flat(batch.post_cont[..., 0]),
        actual_plate_z=flat(batch.post_cont[..., 1]),
        balls=flat(batch.pre_cont[..., 0]).long(),
        strikes=flat(batch.pre_cont[..., 1]).long(),
        p_throws_id=flat(batch.pre_cat["p_throws_id"]),
        stand_id=flat(batch.pre_cat["stand_id"]),
        pa_length=flat(pa_len),
        pitch_idx_in_pa=flat(batch.pre_cont[..., 5]).long(),
    )


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #


@dataclass
class BehavioralMetrics:
    n_pitches: int
    pitch_type_top1: float
    pitch_type_top3: float
    coarse_zone_top1: float                     # (type, 4×4 macro) match
    spatial_distance_mean_ft: float
    spatial_distance_median_ft: float
    spatial_distance_p75_ft: float
    spatial_within_6in_frac: float
    pitch_type_kl_learned_to_behavior: float
    pitch_type_dist_learned: dict               # pitch_type_id -> probability
    pitch_type_dist_behavior: dict
    # Pitcher-blind variants (filled when computed)
    pitch_type_top1_blind: float | None = None
    pitch_type_top3_blind: float | None = None
    spatial_distance_median_ft_blind: float | None = None


def _kl_divergence(p_counts: dict[int, int], q_counts: dict[int, int], eps: float = 1e-9) -> float:
    """KL[p ∥ q] over a discrete vocabulary defined by the union of keys."""
    keys = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    if p_total == 0 or q_total == 0:
        return float("nan")
    kl = 0.0
    for k in keys:
        p = (p_counts.get(k, 0) + eps) / (p_total + eps * len(keys))
        q = (q_counts.get(k, 0) + eps) / (q_total + eps * len(keys))
        kl += p * np.log(p / q)
    return float(kl)


def metrics_from_predictions(preds_list: list[PerPitchPredictions]) -> BehavioralMetrics:
    """Aggregate per-batch predictions into one BehavioralMetrics."""
    # Concatenate
    cat = lambda attr: torch.cat([getattr(p, attr) for p in preds_list], dim=0)
    policy_type = cat("policy_pitch_type")
    policy_x = cat("policy_x_bin")
    policy_z = cat("policy_z_bin")
    policy_top3 = cat("policy_top3_pitch_type")
    actual_type = cat("actual_pitch_type")
    actual_x = cat("actual_x_bin")
    actual_z = cat("actual_z_bin")
    actual_plate_x = cat("actual_plate_x_mirrored")
    actual_plate_z = cat("actual_plate_z")
    n = int(policy_type.shape[0])
    if n == 0:
        return BehavioralMetrics(
            n_pitches=0,
            pitch_type_top1=float("nan"), pitch_type_top3=float("nan"),
            coarse_zone_top1=float("nan"),
            spatial_distance_mean_ft=float("nan"),
            spatial_distance_median_ft=float("nan"),
            spatial_distance_p75_ft=float("nan"),
            spatial_within_6in_frac=float("nan"),
            pitch_type_kl_learned_to_behavior=float("nan"),
            pitch_type_dist_learned={}, pitch_type_dist_behavior={},
        )

    # Top-1 / top-3 pitch type
    top1 = (policy_type == actual_type).float().mean().item()
    top3 = (policy_top3 == actual_type.unsqueeze(-1)).any(dim=-1).float().mean().item()

    # Coarse-zone (type + 4x4 zone) agreement
    policy_macro = to_macro_zone(policy_x, policy_z)
    actual_macro = to_macro_zone(actual_x, actual_z)
    coarse = ((policy_type == actual_type) & (policy_macro == actual_macro)).float().mean().item()

    # Spatial distance from policy bin center to actual landing point
    pred_x_ft = bin_center(policy_x, X_BIN_LO, X_BIN_HI, N_X_BINS)
    pred_z_ft = bin_center(policy_z, Z_BIN_LO, Z_BIN_HI, N_Z_BINS)
    dx = pred_x_ft - actual_plate_x
    dz = pred_z_ft - actual_plate_z
    dist = torch.sqrt(dx ** 2 + dz ** 2)
    dist_np = dist.cpu().numpy()

    # Marginal KL on pitch_type (learned vs behavior)
    learned_counts = dict(Counter(policy_type.cpu().tolist()))
    behavior_counts = dict(Counter(actual_type.cpu().tolist()))
    kl = _kl_divergence(learned_counts, behavior_counts)

    learned_total = sum(learned_counts.values())
    behavior_total = sum(behavior_counts.values())
    learned_dist = {int(k): v / learned_total for k, v in learned_counts.items()}
    behavior_dist = {int(k): v / behavior_total for k, v in behavior_counts.items()}

    return BehavioralMetrics(
        n_pitches=n,
        pitch_type_top1=top1,
        pitch_type_top3=top3,
        coarse_zone_top1=coarse,
        spatial_distance_mean_ft=float(np.mean(dist_np)),
        spatial_distance_median_ft=float(np.median(dist_np)),
        spatial_distance_p75_ft=float(np.percentile(dist_np, 75)),
        spatial_within_6in_frac=float(np.mean(dist_np <= WITHIN_THRESHOLD_FT)),
        pitch_type_kl_learned_to_behavior=kl,
        pitch_type_dist_learned=learned_dist,
        pitch_type_dist_behavior=behavior_dist,
    )


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate_behavioral(
    model: QTransformer,
    loader: Iterable,
    *,
    device,
    include_pitcher_blind: bool = True,
) -> BehavioralMetrics:
    """Run the full Part-A behavioral evaluation across a DataLoader."""
    model.eval()
    preds: list[PerPitchPredictions] = []
    for batch in loader:
        batch = batch.to(device)
        preds.append(predict_batch(model, batch))
    metrics = metrics_from_predictions(preds)

    if include_pitcher_blind and metrics.n_pitches > 0:
        with _zeroed_pitcher_embedding(model):
            blind_preds: list[PerPitchPredictions] = []
            for batch in loader:
                batch = batch.to(device)
                blind_preds.append(predict_batch(model, batch))
        blind = metrics_from_predictions(blind_preds)
        metrics.pitch_type_top1_blind = blind.pitch_type_top1
        metrics.pitch_type_top3_blind = blind.pitch_type_top3
        metrics.spatial_distance_median_ft_blind = blind.spatial_distance_median_ft

    return metrics


# --------------------------------------------------------------------------- #
# Segment breakdowns
# --------------------------------------------------------------------------- #


def _segment_metrics(preds: PerPitchPredictions, mask: torch.Tensor) -> dict:
    """Compute lite metrics on a subset of pitches."""
    if mask.sum() == 0:
        return {"n": 0, "top1": float("nan"), "top3": float("nan"), "median_dist_ft": float("nan")}
    pt_match = (preds.policy_pitch_type[mask] == preds.actual_pitch_type[mask]).float().mean().item()
    top3_match = (preds.policy_top3_pitch_type[mask] == preds.actual_pitch_type[mask].unsqueeze(-1)).any(-1).float().mean().item()
    pred_x_ft = bin_center(preds.policy_x_bin[mask], X_BIN_LO, X_BIN_HI, N_X_BINS)
    pred_z_ft = bin_center(preds.policy_z_bin[mask], Z_BIN_LO, Z_BIN_HI, N_Z_BINS)
    dx = pred_x_ft - preds.actual_plate_x_mirrored[mask]
    dz = pred_z_ft - preds.actual_plate_z[mask]
    dist = torch.sqrt(dx ** 2 + dz ** 2)
    return {
        "n": int(mask.sum().item()),
        "top1": pt_match,
        "top3": top3_match,
        "median_dist_ft": float(torch.median(dist).item()),
    }


@torch.no_grad()
def segment_breakdowns(
    model: QTransformer,
    loader: Iterable,
    *,
    device,
) -> dict[str, pd.DataFrame]:
    """Returns ``{segment_name: DataFrame}`` for each segmenting axis."""
    model.eval()
    preds_list: list[PerPitchPredictions] = []
    for batch in loader:
        batch = batch.to(device)
        preds_list.append(predict_batch(model, batch))
    if not preds_list:
        return {}

    # Concat all predictions into one PerPitchPredictions
    cat = lambda attr: torch.cat([getattr(p, attr) for p in preds_list], dim=0)
    P = PerPitchPredictions(
        policy_pitch_type=cat("policy_pitch_type"),
        policy_x_bin=cat("policy_x_bin"),
        policy_z_bin=cat("policy_z_bin"),
        policy_top3_pitch_type=cat("policy_top3_pitch_type"),
        actual_pitch_type=cat("actual_pitch_type"),
        actual_x_bin=cat("actual_x_bin"),
        actual_z_bin=cat("actual_z_bin"),
        actual_plate_x_mirrored=cat("actual_plate_x_mirrored"),
        actual_plate_z=cat("actual_plate_z"),
        balls=cat("balls"),
        strikes=cat("strikes"),
        p_throws_id=cat("p_throws_id"),
        stand_id=cat("stand_id"),
        pa_length=cat("pa_length"),
        pitch_idx_in_pa=cat("pitch_idx_in_pa"),
    )

    out: dict[str, pd.DataFrame] = {}

    # By count (balls, strikes)
    rows = []
    for b in range(4):
        for s in range(3):
            mask = (P.balls == b) & (P.strikes == s)
            m = _segment_metrics(P, mask)
            m["count"] = f"{b}-{s}"
            rows.append(m)
    out["count"] = pd.DataFrame(rows).set_index("count")

    # By matchup (p_throws × stand)
    rows = []
    for pt_id in P.p_throws_id.unique().tolist():
        for st_id in P.stand_id.unique().tolist():
            mask = (P.p_throws_id == pt_id) & (P.stand_id == st_id)
            m = _segment_metrics(P, mask)
            m["matchup"] = f"p_throws={int(pt_id)}, stand={int(st_id)}"
            rows.append(m)
    out["matchup"] = pd.DataFrame(rows).set_index("matchup")

    # By PA length (1-2 short, 3-4 medium, 5+ long)
    rows = []
    for label, lo, hi in [("short (1-2)", 1, 2), ("medium (3-4)", 3, 4), ("long (5+)", 5, 99)]:
        mask = (P.pa_length >= lo) & (P.pa_length <= hi)
        m = _segment_metrics(P, mask)
        m["pa_length"] = label
        rows.append(m)
    out["pa_length"] = pd.DataFrame(rows).set_index("pa_length")

    return out
