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
