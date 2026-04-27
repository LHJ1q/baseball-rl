"""Unit tests for src/encoder.py."""
from __future__ import annotations

import torch

from src.encoder import (
    BATTER_PROFILE_OVERALL_FIELDS,
    POST_ACTION_CATEGORICAL_FIELDS,
    POST_ACTION_CONTINUOUS_FIELDS,
    PRE_ACTION_CATEGORICAL_FIELDS,
    PRE_ACTION_CONTINUOUS_FIELDS,
    EncoderConfig,
    PostActionEncoder,
    PreActionEncoder,
)

VOCAB_SIZES = {
    "pitch_type": 10, "description": 8,
    "p_throws": 2, "stand": 2, "inning_topbot": 2,
    "batter": 50, "pitcher": 50,
}


def _pre_inputs(B=2, T=3):
    cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in PRE_ACTION_CATEGORICAL_FIELDS}
    cont = torch.randn(B, T, len(PRE_ACTION_CONTINUOUS_FIELDS))
    profile = torch.randn(B, T, len(BATTER_PROFILE_OVERALL_FIELDS))
    return cat, cont, profile


def _post_inputs(B=2, T=3):
    cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in POST_ACTION_CATEGORICAL_FIELDS}
    cont = torch.randn(B, T, len(POST_ACTION_CONTINUOUS_FIELDS))
    return cat, cont


def test_pre_encoder_output_shape():
    enc = PreActionEncoder(VOCAB_SIZES, cfg=EncoderConfig(d_model=64, d_player_emb=16))
    cat, cont, profile = _pre_inputs()
    out = enc(cat, cont, profile)
    assert out.shape == (2, 3, 64)


def test_post_encoder_output_shape():
    enc = PostActionEncoder(VOCAB_SIZES, n_x_bins=20, n_z_bins=20,
                            cfg=EncoderConfig(d_model=64))
    cat, cont = _post_inputs()
    out = enc(cat, cont)
    assert out.shape == (2, 3, 64)


def test_encoder_no_nan_on_zero_inputs():
    enc = PreActionEncoder(VOCAB_SIZES, cfg=EncoderConfig(d_model=64))
    cat = {f: torch.zeros(2, 3, dtype=torch.int64) for f in PRE_ACTION_CATEGORICAL_FIELDS}
    cont = torch.zeros(2, 3, len(PRE_ACTION_CONTINUOUS_FIELDS))
    profile = torch.zeros(2, 3, len(BATTER_PROFILE_OVERALL_FIELDS))
    out = enc(cat, cont, profile)
    assert torch.isfinite(out).all()


def test_post_encoder_uses_shared_action_embeddings():
    """When QTransformer passes shared action embedding modules, the encoder
    must use them — not allocate new ones."""
    shared_pt = torch.nn.Embedding(VOCAB_SIZES["pitch_type"], 32)
    shared_x = torch.nn.Embedding(20, 24)
    shared_z = torch.nn.Embedding(20, 24)
    enc = PostActionEncoder(
        VOCAB_SIZES, n_x_bins=20, n_z_bins=20,
        cfg=EncoderConfig(d_model=64, d_pitch_type_emb=32, d_action_loc_emb=24),
        action_emb_modules={"pitch_type": shared_pt, "x_bin": shared_x, "z_bin": shared_z},
    )
    assert enc.emb_pitch_type is shared_pt
    assert enc.emb_x_bin is shared_x
    assert enc.emb_z_bin is shared_z
