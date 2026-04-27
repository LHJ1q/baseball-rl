"""Unit tests for src/ope_metrics.py."""
from __future__ import annotations

import torch

from src.configs import load_preset
from src.dataset import PABatch, pa_collate
from src.encoder import (
    BATTER_PROFILE_OVERALL_FIELDS,
    POST_ACTION_CATEGORICAL_FIELDS,
    POST_ACTION_CONTINUOUS_FIELDS,
    PRE_ACTION_CATEGORICAL_FIELDS,
    PRE_ACTION_CONTINUOUS_FIELDS,
)
from src.ope_metrics import (
    N_MACRO_X,
    N_MACRO_Y,
    bin_center,
    metrics_from_predictions,
    PerPitchPredictions,
    to_macro_zone,
    _kl_divergence,
)
from src.qtransformer import QTransformer
from src.tokenize import N_X_BINS, N_Z_BINS, X_BIN_LO


def test_bin_center_first_bin_is_low_plus_half_width():
    centers = bin_center(torch.tensor([0]), -2.5, 2.5, 20)
    # bin 0 covers [-2.5, -2.25]; center = -2.375
    assert centers[0].item() == -2.375


def test_to_macro_zone_collapses_20x20_to_4x4():
    # bin 0 of x and 0 of z → macro 0
    z = to_macro_zone(torch.tensor([0]), torch.tensor([0]))
    assert z.item() == 0
    # bin 19 of x and 19 of z → macro 15 (last)
    z = to_macro_zone(torch.tensor([19]), torch.tensor([19]))
    assert z.item() == N_MACRO_X * N_MACRO_Y - 1


def test_kl_divergence_equal_distributions_is_zero():
    p = {0: 100, 1: 200, 2: 300}
    q = {0: 100, 1: 200, 2: 300}
    assert abs(_kl_divergence(p, q)) < 1e-6


def test_kl_divergence_disjoint_supports_is_finite_via_smoothing():
    """Empty key in one distribution should still yield finite KL via the eps smoothing."""
    p = {0: 10}
    q = {1: 10}
    kl = _kl_divergence(p, q)
    assert torch.tensor(kl).isfinite().item()
    assert kl > 0


def test_metrics_from_predictions_perfect_agreement():
    """When policy == actual everywhere, top-1 == 100%, distance = 0."""
    n = 8
    pp = PerPitchPredictions(
        policy_pitch_type=torch.tensor([1, 2, 3, 1, 2, 3, 1, 2]),
        policy_x_bin=torch.tensor([10] * n),
        policy_z_bin=torch.tensor([10] * n),
        policy_top3_pitch_type=torch.zeros(n, 3, dtype=torch.long),
        actual_pitch_type=torch.tensor([1, 2, 3, 1, 2, 3, 1, 2]),
        actual_x_bin=torch.tensor([10] * n),
        actual_z_bin=torch.tensor([10] * n),
        actual_plate_x_mirrored=bin_center(torch.tensor([10] * n), -2.5, 2.5, N_X_BINS),
        actual_plate_z=bin_center(torch.tensor([10] * n), -1.0, 6.0, N_Z_BINS),
        balls=torch.zeros(n, dtype=torch.long),
        strikes=torch.zeros(n, dtype=torch.long),
        p_throws_id=torch.zeros(n, dtype=torch.long),
        stand_id=torch.zeros(n, dtype=torch.long),
        pa_length=torch.full((n,), 4, dtype=torch.long),
        pitch_idx_in_pa=torch.zeros(n, dtype=torch.long),
    )
    m = metrics_from_predictions([pp])
    assert m.n_pitches == n
    assert m.pitch_type_top1 == 1.0
    assert m.coarse_zone_top1 == 1.0
    assert abs(m.spatial_distance_mean_ft) < 1e-6


def test_metrics_from_predictions_total_disagreement_on_pitch_type():
    n = 4
    pp = PerPitchPredictions(
        policy_pitch_type=torch.tensor([0, 0, 0, 0]),
        policy_x_bin=torch.tensor([10] * n),
        policy_z_bin=torch.tensor([10] * n),
        policy_top3_pitch_type=torch.tensor([[0, 4, 5]] * n),
        actual_pitch_type=torch.tensor([1, 2, 3, 1]),
        actual_x_bin=torch.tensor([10] * n),
        actual_z_bin=torch.tensor([10] * n),
        actual_plate_x_mirrored=torch.zeros(n),
        actual_plate_z=torch.zeros(n),
        balls=torch.zeros(n, dtype=torch.long),
        strikes=torch.zeros(n, dtype=torch.long),
        p_throws_id=torch.zeros(n, dtype=torch.long),
        stand_id=torch.zeros(n, dtype=torch.long),
        pa_length=torch.full((n,), 4, dtype=torch.long),
        pitch_idx_in_pa=torch.zeros(n, dtype=torch.long),
    )
    m = metrics_from_predictions([pp])
    assert m.pitch_type_top1 == 0.0
    assert m.pitch_type_top3 == 0.0  # actual {1,2,3} not in top3 {0,4,5}


def test_metrics_from_predictions_partial_top3_agreement():
    """top1 = 0 but top3 captures actual when actual is in {0, 4, 5}."""
    n = 3
    pp = PerPitchPredictions(
        policy_pitch_type=torch.tensor([2, 2, 2]),
        policy_x_bin=torch.tensor([10] * n),
        policy_z_bin=torch.tensor([10] * n),
        policy_top3_pitch_type=torch.tensor([[2, 4, 5]] * n),
        actual_pitch_type=torch.tensor([4, 5, 4]),  # in top3 but not top1
        actual_x_bin=torch.tensor([10] * n),
        actual_z_bin=torch.tensor([10] * n),
        actual_plate_x_mirrored=torch.zeros(n),
        actual_plate_z=torch.zeros(n),
        balls=torch.zeros(n, dtype=torch.long),
        strikes=torch.zeros(n, dtype=torch.long),
        p_throws_id=torch.zeros(n, dtype=torch.long),
        stand_id=torch.zeros(n, dtype=torch.long),
        pa_length=torch.full((n,), 4, dtype=torch.long),
        pitch_idx_in_pa=torch.zeros(n, dtype=torch.long),
    )
    m = metrics_from_predictions([pp])
    assert m.pitch_type_top1 == 0.0
    assert m.pitch_type_top3 == 1.0
