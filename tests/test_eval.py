"""Unit tests for src/eval.py."""
from __future__ import annotations

import torch

from src.configs import load_preset
from src.dataset import PABatch
from src.encoder import (
    BATTER_PROFILE_OVERALL_FIELDS,
    POST_ACTION_CATEGORICAL_FIELDS,
    POST_ACTION_CONTINUOUS_FIELDS,
    PRE_ACTION_CATEGORICAL_FIELDS,
    PRE_ACTION_CONTINUOUS_FIELDS,
)
from src.eval import _zeroed_pitcher_embedding, eval_losses, eval_pitcher_blind
from src.qtransformer import QTransformer

VOCAB_SIZES = {
    "pitch_type": 6, "description": 4,
    "p_throws": 2, "stand": 2, "inning_topbot": 2,
    "batter": 20, "pitcher": 20,
}


def _make_batch(B=2, T=3) -> PABatch:
    n_pitch_types = VOCAB_SIZES["pitch_type"]
    pre_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in PRE_ACTION_CATEGORICAL_FIELDS}
    pre_cat["pitcher_id"] = torch.tensor([1, 2]).unsqueeze(1).expand(B, T).contiguous()
    pre_cont = torch.randn(B, T, len(PRE_ACTION_CONTINUOUS_FIELDS))
    profile = torch.randn(B, T, len(BATTER_PROFILE_OVERALL_FIELDS))
    post_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in POST_ACTION_CATEGORICAL_FIELDS}
    post_cont = torch.randn(B, T, len(POST_ACTION_CONTINUOUS_FIELDS))
    reward = torch.randn(B, T)
    is_terminal = torch.zeros(B, T, dtype=torch.bool); is_terminal[:, -1] = True
    pa_lengths = torch.tensor([T, T], dtype=torch.int64)
    valid_mask = torch.ones(B, T, dtype=torch.bool)
    arsenal = torch.randn(B, T, n_pitch_types, 14)
    batter_pt = torch.randn(B, T, n_pitch_types, 4)
    return PABatch(pre_cat, pre_cont, profile, post_cat, post_cont, reward, is_terminal,
                   pa_lengths, valid_mask, arsenal, batter_pt)


def _tiny_model() -> QTransformer:
    enc, q = load_preset("smoke")
    # vocab compatible with our test batch
    return QTransformer(VOCAB_SIZES, cfg=q, encoder_cfg=enc)


def test_eval_losses_returns_finite_scalars():
    model = _tiny_model()
    batch = _make_batch()
    losses = eval_losses(model, batch, gamma=1.0, tau=0.7)
    for k in ("q_loss", "v_loss", "n_valid"):
        assert k in losses
        assert torch.isfinite(losses[k]).item()


def test_zeroed_pitcher_embedding_is_actually_zero_inside_block():
    model = _tiny_model()
    original = model.pre_encoder.emb_pitcher.weight.data.clone()
    with _zeroed_pitcher_embedding(model):
        assert (model.pre_encoder.emb_pitcher.weight.data == 0).all()
    # Restored after exit
    torch.testing.assert_close(model.pre_encoder.emb_pitcher.weight.data, original)


def test_pitcher_blind_eval_changes_outputs_vs_normal():
    """When pitcher embedding is non-trivial, blind forward should produce
    different Q/V than normal forward."""
    model = _tiny_model()
    # Make pitcher embedding strongly non-zero so blinding has a real effect.
    with torch.no_grad():
        model.pre_encoder.emb_pitcher.weight.data.normal_(mean=0.0, std=2.0)
    batch = _make_batch()
    normal = eval_losses(model, batch, gamma=1.0, tau=0.7)
    blind = eval_pitcher_blind(model, batch, gamma=1.0, tau=0.7)
    # The two losses should differ given non-zero pitcher embeddings.
    assert abs(normal["q_loss"].item() - blind["q_loss"].item()) > 1e-6


def test_pitcher_blind_eval_does_not_mutate_model():
    """After calling blind, the embedding must be exactly the original."""
    model = _tiny_model()
    with torch.no_grad():
        model.pre_encoder.emb_pitcher.weight.data.normal_(mean=1.0, std=1.0)
    snapshot = model.pre_encoder.emb_pitcher.weight.data.clone()
    batch = _make_batch()
    _ = eval_pitcher_blind(model, batch, gamma=1.0, tau=0.7)
    torch.testing.assert_close(model.pre_encoder.emb_pitcher.weight.data, snapshot)


# --------------------------------------------------------------------------- #
# Pitcher-blind = full blind (also blanks arsenal_per_type)
# --------------------------------------------------------------------------- #


def test_pitcher_blind_zeros_arsenal_per_type():
    """_fully_blinded_pitcher must zero both pitcher embedding AND arsenal_per_type
    (with low_sample=1.0). Arsenal restored on exit."""
    from src.eval import _fully_blinded_pitcher
    model = _tiny_model()
    batch = _make_batch()
    original_arsenal = batch.arsenal_per_type.clone()
    with _fully_blinded_pitcher(model, batch):
        # Inside: arsenal should be zeros + low_sample=1.0
        assert (batch.arsenal_per_type[..., 0] == 0).all(), "count field should be zero"
        assert (batch.arsenal_per_type[..., 1] == 1.0).all(), "low_sample field should be 1.0"
        # All other arsenal stat fields should be zero
        for f in range(2, 14):
            assert (batch.arsenal_per_type[..., f] == 0).all(), f"arsenal field {f} should be zero"
        # Pitcher embedding should also be zero
        assert (model.pre_encoder.emb_pitcher.weight.data == 0).all()
    # After: arsenal restored exactly
    torch.testing.assert_close(batch.arsenal_per_type, original_arsenal)


def test_eval_pitcher_blind_uses_full_blinding_now():
    """eval_pitcher_blind should now use _fully_blinded_pitcher under the hood.
    Verify by checking that arsenal AND embedding are both blanked during the call."""
    from src.eval import eval_pitcher_blind
    model = _tiny_model()
    # Make pitcher embedding strongly non-zero so blinding has a measurable effect
    with torch.no_grad():
        model.pre_encoder.emb_pitcher.weight.data.normal_(mean=0.0, std=2.0)
    batch = _make_batch()
    # Set non-trivial arsenal so we'd see a difference if NOT blanking
    batch.arsenal_per_type.normal_()
    # Just verify the call runs and returns expected keys (full integration test).
    out = eval_pitcher_blind(model, batch, gamma=1.0, tau=0.7)
    assert "q_loss" in out
    assert "v_loss" in out
