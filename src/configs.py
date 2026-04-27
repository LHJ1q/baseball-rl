"""Named hyperparameter presets and JSON config I/O.

A single source of truth for the encoder + Q-Transformer configuration. The
smoke test and (future) phase-8 trainer both pull from here so v1 vs smoke
isn't redefined in two places.

Conventions:

* ``PRESETS`` maps a preset name → ``(EncoderConfig, QTransformerConfig)``.
* ``load_preset(name)`` returns a fresh copy (so callers can mutate without
  polluting the registry).
* ``load_from_json(path)`` reads a JSON file with ``encoder`` and ``qtransformer``
  sub-dicts; ``save_to_json(path, ...)`` writes the same format.
* ``build_qtransformer(tokens_dir, preset_or_configs)`` is the recommended
  one-liner to construct a fully-wired model from disk artifacts (vocab +
  feature stats).
"""
from __future__ import annotations

import dataclasses
import json
from copy import deepcopy
from pathlib import Path

import torch

from src.dataset import load_vocab_sizes
from src.encoder import EncoderConfig
from src.qtransformer import QTransformer, QTransformerConfig


# --------------------------------------------------------------------------- #
# Named presets
# --------------------------------------------------------------------------- #

# Tiny — for the Macbook smoke test. Identical code path, drastically smaller.
SMOKE_ENCODER = EncoderConfig(
    d_model=64, d_player_emb=16, d_pitch_type_emb=16, d_description_emb=8,
    d_action_loc_emb=12, d_small_emb=4, dropout=0.1,
)
SMOKE_QTRANSFORMER = QTransformerConfig(
    d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.1,
    n_x_bins=20, n_z_bins=20, max_seq_len=64,
)

# v1 — recommended preset for the Linux/Colab GPU run on 2021-2025 data.
# ~25M params, BF16 fits easily in 24 GB at batch 512 PAs.
V1_ENCODER = EncoderConfig(
    d_model=384, d_player_emb=96, d_pitch_type_emb=32, d_description_emb=16,
    d_action_loc_emb=24, d_small_emb=4, dropout=0.1,
)
V1_QTRANSFORMER = QTransformerConfig(
    d_model=384, n_layers=6, n_heads=8, d_ff=1536, dropout=0.1,
    n_x_bins=20, n_z_bins=20, max_seq_len=64,
)

PRESETS: dict[str, tuple[EncoderConfig, QTransformerConfig]] = {
    "smoke": (SMOKE_ENCODER, SMOKE_QTRANSFORMER),
    "v1": (V1_ENCODER, V1_QTRANSFORMER),
}


def load_preset(name: str) -> tuple[EncoderConfig, QTransformerConfig]:
    """Return a fresh copy of the named preset. Caller can safely mutate."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {list(PRESETS)}")
    enc, q = PRESETS[name]
    return deepcopy(enc), deepcopy(q)


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #


def save_to_json(path: Path, encoder_cfg: EncoderConfig, q_cfg: QTransformerConfig) -> None:
    payload = {
        "encoder": dataclasses.asdict(encoder_cfg),
        "qtransformer": dataclasses.asdict(q_cfg),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_from_json(path: Path) -> tuple[EncoderConfig, QTransformerConfig]:
    payload = json.loads(path.read_text())
    return EncoderConfig(**payload["encoder"]), QTransformerConfig(**payload["qtransformer"])


# --------------------------------------------------------------------------- #
# Ad-hoc CLI overrides
# --------------------------------------------------------------------------- #


def apply_overrides(
    encoder_cfg: EncoderConfig,
    q_cfg: QTransformerConfig,
    overrides: dict[str, object],
) -> tuple[EncoderConfig, QTransformerConfig]:
    """Apply ``{field: value}`` overrides. Routes each field to whichever config
    declares it (``d_model`` is in both — set on both for consistency)."""
    enc_fields = {f.name for f in dataclasses.fields(EncoderConfig)}
    q_fields = {f.name for f in dataclasses.fields(QTransformerConfig)}
    enc_kwargs = dataclasses.asdict(encoder_cfg)
    q_kwargs = dataclasses.asdict(q_cfg)
    for k, v in overrides.items():
        applied = False
        if k in enc_fields:
            enc_kwargs[k] = v
            applied = True
        if k in q_fields:
            q_kwargs[k] = v
            applied = True
        if not applied:
            raise KeyError(f"override field {k!r} is not in EncoderConfig or QTransformerConfig")
    return EncoderConfig(**enc_kwargs), QTransformerConfig(**q_kwargs)


# --------------------------------------------------------------------------- #
# Stats → tensor helpers (consumed by the encoder constructor)
# --------------------------------------------------------------------------- #


def _stats_to_tensors(stats_entry: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(stats_entry["mean"], dtype=torch.float32), torch.tensor(stats_entry["std"], dtype=torch.float32)


def load_feature_stats(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return ``{group: (mean_tensor, std_tensor)}`` for every continuous-feature
    group: ``pre_cont``, ``profile``, ``post_cont``, plus ``arsenal_head`` and
    ``batter_per_type_head`` (the per-type Q-head joins)."""
    raw = json.loads(path.read_text())
    out = {
        "pre_cont": _stats_to_tensors(raw["pre_cont"]),
        "profile": _stats_to_tensors(raw["profile"]),
        "post_cont": _stats_to_tensors(raw["post_cont"]),
    }
    if "arsenal_head" in raw:
        out["arsenal_head"] = _stats_to_tensors(raw["arsenal_head"])
    if "batter_per_type_head" in raw:
        out["batter_per_type_head"] = _stats_to_tensors(raw["batter_per_type_head"])
    return out


# --------------------------------------------------------------------------- #
# One-liner model builder from disk artifacts
# --------------------------------------------------------------------------- #


def build_qtransformer(
    tokens_dir: Path,
    preset: str | tuple[EncoderConfig, QTransformerConfig] = "v1",
    overrides: dict[str, object] | None = None,
) -> QTransformer:
    """Build a fully-wired :class:`QTransformer` from disk artifacts.

    Reads ``vocab.json`` and ``feature_stats.json`` from ``tokens_dir``, applies
    the named preset (or a passed tuple), applies any CLI-style overrides, and
    returns the constructed model with frozen standardization buffers.
    """
    if isinstance(preset, str):
        encoder_cfg, q_cfg = load_preset(preset)
    else:
        encoder_cfg, q_cfg = preset
    if overrides:
        encoder_cfg, q_cfg = apply_overrides(encoder_cfg, q_cfg, overrides)

    vocab_sizes = load_vocab_sizes(tokens_dir / "vocab.json")
    stats_path = tokens_dir / "feature_stats.json"
    if stats_path.exists():
        stats = load_feature_stats(stats_path)
    else:
        stats = {}

    return QTransformer(
        vocab_sizes,
        cfg=q_cfg,
        encoder_cfg=encoder_cfg,
        pre_cont_stats=stats.get("pre_cont"),
        profile_stats=stats.get("profile"),
        post_cont_stats=stats.get("post_cont"),
        arsenal_head_stats=stats.get("arsenal_head"),
        batter_per_type_head_stats=stats.get("batter_per_type_head"),
    )
