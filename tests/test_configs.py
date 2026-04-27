"""Unit tests for src/configs.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.configs import (
    PRESETS,
    apply_overrides,
    load_from_json,
    load_preset,
    save_to_json,
)
from src.encoder import EncoderConfig
from src.qtransformer import QTransformerConfig


def test_presets_v1_dimensions():
    enc, q = load_preset("v1")
    assert enc.d_model == 384
    assert enc.d_player_emb == 96
    assert q.n_layers == 6
    assert q.n_heads == 8
    assert q.d_ff == 1536
    assert enc.d_model == q.d_model  # consistency


def test_presets_smoke_is_tiny():
    enc, q = load_preset("smoke")
    assert enc.d_model == 64
    assert q.n_layers == 2


def test_load_preset_returns_independent_copies():
    enc1, q1 = load_preset("v1")
    enc1.d_model = 999
    enc2, q2 = load_preset("v1")
    assert enc2.d_model == 384  # mutation of enc1 didn't pollute the registry


def test_load_preset_unknown_name_raises():
    with pytest.raises(KeyError, match="unknown preset"):
        load_preset("this_does_not_exist")


def test_json_save_and_load_round_trip():
    enc_in, q_in = load_preset("v1")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        save_to_json(path, enc_in, q_in)
        enc_out, q_out = load_from_json(path)
        assert enc_in == enc_out
        assert q_in == q_out
    finally:
        path.unlink(missing_ok=True)


def test_apply_overrides_routes_to_correct_config():
    enc, q = load_preset("smoke")
    enc2, q2 = apply_overrides(enc, q, {"n_layers": 8, "d_player_emb": 32})
    assert q2.n_layers == 8
    assert enc2.d_player_emb == 32
    # Original unchanged
    assert q.n_layers == 2
    assert enc.d_player_emb == 16


def test_apply_overrides_d_model_propagates_to_both_configs():
    enc, q = load_preset("smoke")
    enc2, q2 = apply_overrides(enc, q, {"d_model": 128})
    assert enc2.d_model == 128
    assert q2.d_model == 128


def test_apply_overrides_unknown_field_raises():
    enc, q = load_preset("smoke")
    with pytest.raises(KeyError, match="not in EncoderConfig"):
        apply_overrides(enc, q, {"this_field_does_not_exist": 42})


def test_all_presets_construct_valid_configs():
    """Every preset should at least be instantiable as the dataclasses."""
    for name in PRESETS:
        enc, q = load_preset(name)
        assert isinstance(enc, EncoderConfig)
        assert isinstance(q, QTransformerConfig)
        # Sanity: n_heads divides d_model
        assert q.d_model % q.n_heads == 0
