"""Unit tests for src/qtransformer.py."""
from __future__ import annotations

import torch

from src.dataset import PABatch
from src.encoder import (
    BATTER_PROFILE_OVERALL_FIELDS,
    POST_ACTION_CATEGORICAL_FIELDS,
    POST_ACTION_CONTINUOUS_FIELDS,
    PRE_ACTION_CATEGORICAL_FIELDS,
    PRE_ACTION_CONTINUOUS_FIELDS,
    EncoderConfig,
)
from src.qtransformer import (
    QTransformer,
    QTransformerConfig,
    build_repertoire_mask,
    expectile_loss,
    iql_losses,
    shift_v_for_next_state,
)

VOCAB_SIZES = {
    "pitch_type": 6, "description": 4,
    "p_throws": 2, "stand": 2, "inning_topbot": 2,
    "batter": 20, "pitcher": 20,
}
N_X = 8
N_Z = 8


def _make_batch(B=2, T=3) -> PABatch:
    pre_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in PRE_ACTION_CATEGORICAL_FIELDS}
    pre_cat["pitcher_id"] = torch.tensor([1, 2]).unsqueeze(1).expand(B, T).contiguous()
    pre_cont = torch.randn(B, T, len(PRE_ACTION_CONTINUOUS_FIELDS))
    profile = torch.randn(B, T, len(BATTER_PROFILE_OVERALL_FIELDS))
    post_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in POST_ACTION_CATEGORICAL_FIELDS}
    post_cont = torch.randn(B, T, len(POST_ACTION_CONTINUOUS_FIELDS))
    reward = torch.randn(B, T)
    is_terminal = torch.zeros(B, T, dtype=torch.bool)
    is_terminal[:, -1] = True
    pa_lengths = torch.tensor([T, T], dtype=torch.int64)
    valid_mask = torch.ones(B, T, dtype=torch.bool)
    n_pitch_types = VOCAB_SIZES["pitch_type"]
    arsenal_per_type = torch.randn(B, T, n_pitch_types, 14)
    batter_per_type = torch.randn(B, T, n_pitch_types, 4)
    return PABatch(pre_cat, pre_cont, profile, post_cat, post_cont, reward, is_terminal, pa_lengths, valid_mask,
                   arsenal_per_type, batter_per_type)


def test_forward_returns_expected_keys_and_shapes():
    cfg = QTransformerConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64,
                             n_x_bins=N_X, n_z_bins=N_Z)
    model = QTransformer(VOCAB_SIZES, cfg=cfg, encoder_cfg=EncoderConfig(d_model=32, d_player_emb=8))
    batch = _make_batch(B=2, T=4)
    out = model(batch)
    assert out["q_chosen"].shape == (2, 4)
    assert out["v"].shape == (2, 4)
    assert out["q_type_logits"].shape == (2, 4, VOCAB_SIZES["pitch_type"])
    assert out["q_x_logits"].shape == (2, 4, N_X)
    assert out["q_z_logits"].shape == (2, 4, N_Z)


def test_repertoire_mask_blocks_disallowed_pitch_types():
    pitcher = torch.tensor([[1, 1], [2, 2]])
    arsenal = {(1, 0): 100, (1, 3): 100, (2, 5): 100}
    mask = build_repertoire_mask(pitcher, arsenal, n_pitch_types=6, n_min=10)
    # Pitcher 1 should allow only types 0 and 3
    assert mask[0, 0].tolist() == [True, False, False, True, False, False]
    # Pitcher 2 should allow only type 5
    assert mask[1, 0].tolist() == [False, False, False, False, False, True]


def test_repertoire_mask_falls_back_for_unseen_pitcher():
    pitcher = torch.tensor([[99]])
    mask = build_repertoire_mask(pitcher, {}, n_pitch_types=6, n_min=10)
    # All-True fallback so the model isn't blocked from acting at all.
    assert mask[0, 0].all().item()


def test_expectile_loss_asymmetry():
    diff_pos = torch.tensor([1.0])
    diff_neg = torch.tensor([-1.0])
    loss_pos = expectile_loss(diff_pos, tau=0.7)
    loss_neg = expectile_loss(diff_neg, tau=0.7)
    # tau > 0.5 weights positive residuals more heavily.
    assert loss_pos.item() > loss_neg.item()
    import pytest
    assert loss_pos.item() == pytest.approx(0.7, rel=1e-5)
    assert loss_neg.item() == pytest.approx(0.3, rel=1e-5)


def test_iql_loss_terminal_drops_bootstrap():
    """At a terminal pitch, target should equal reward (no γV(s')) regardless of v_next."""
    q = torch.tensor([[0.0]])
    v = torch.tensor([[0.0]])
    v_next = torch.tensor([[100.0]])  # huge — should be ignored
    r = torch.tensor([[0.5]])
    term = torch.tensor([[True]])
    mask = torch.tensor([[True]])
    losses = iql_losses(q, v, v_next, r, term, mask, gamma=1.0, tau=0.7)
    # target = r = 0.5, q = 0 → q_loss = 0.25
    assert abs(losses["q_loss"].item() - 0.25) < 1e-6


def test_shift_v_for_next_state_pads_last_with_zero():
    v = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    valid = torch.ones(2, 3, dtype=torch.bool)
    v_next = shift_v_for_next_state(v, valid)
    assert torch.equal(v_next, torch.tensor([[2.0, 3.0, 0.0], [5.0, 6.0, 0.0]]))


def test_policy_returns_indices_within_action_vocab():
    cfg = QTransformerConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64,
                             n_x_bins=N_X, n_z_bins=N_Z)
    model = QTransformer(VOCAB_SIZES, cfg=cfg, encoder_cfg=EncoderConfig(d_model=32, d_player_emb=8))
    batch = _make_batch(B=2, T=3)
    out = model.policy(batch)
    assert (out["pitch_type"] >= 0).all() and (out["pitch_type"] < VOCAB_SIZES["pitch_type"]).all()
    assert (out["x_bin"] >= 0).all() and (out["x_bin"] < N_X).all()
    assert (out["z_bin"] >= 0).all() and (out["z_bin"] < N_Z).all()
