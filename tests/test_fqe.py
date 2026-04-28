"""Unit tests for src/fqe.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
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
from src.fqe import FQETrainer, FQETrainerConfig, estimate_pa_values, fqe_loss
from src.qtransformer import QTransformer

VOCAB_SIZES = {
    "pitch_type": 6, "description": 4,
    "p_throws": 2, "stand": 2, "inning_topbot": 2,
    "batter": 20, "pitcher": 20,
}


def _make_batch(B=2, T=3) -> PABatch:
    n_pt = VOCAB_SIZES["pitch_type"]
    pre_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in PRE_ACTION_CATEGORICAL_FIELDS}
    pre_cat["pitcher_id"] = torch.tensor([1, 2]).unsqueeze(1).expand(B, T).contiguous()
    pre_cont = torch.randn(B, T, len(PRE_ACTION_CONTINUOUS_FIELDS))
    profile = torch.randn(B, T, len(BATTER_PROFILE_OVERALL_FIELDS))
    post_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in POST_ACTION_CATEGORICAL_FIELDS}
    post_cont = torch.randn(B, T, len(POST_ACTION_CONTINUOUS_FIELDS))
    reward = torch.tensor([[0.1, -0.05, 0.3], [0.0, 0.2, -0.1]])
    is_terminal = torch.zeros(B, T, dtype=torch.bool); is_terminal[:, -1] = True
    pa_lengths = torch.tensor([T, T], dtype=torch.int64)
    valid_mask = torch.ones(B, T, dtype=torch.bool)
    arsenal = torch.randn(B, T, n_pt, 14)
    batter_pt = torch.randn(B, T, n_pt, 4)
    return PABatch(pre_cat, pre_cont, profile, post_cat, post_cont, reward, is_terminal,
                   pa_lengths, valid_mask, arsenal, batter_pt)


def _tiny_model() -> QTransformer:
    enc, q = load_preset("smoke")
    return QTransformer(VOCAB_SIZES, cfg=q, encoder_cfg=enc)


def test_fqe_loss_returns_finite_scalar():
    fqe = _tiny_model()
    policy = _tiny_model()
    batch = _make_batch()
    out = fqe_loss(fqe, policy, batch, gamma=1.0)
    assert torch.isfinite(out["fqe_loss"]).item()
    assert torch.isfinite(out["q_mean"]).item()
    assert torch.isfinite(out["target_mean"]).item()
    # FQE returns ONLY fqe_loss (q_z), not per-axis variants — shallow heads are deliberately untrained.
    assert "fqe_loss_type" not in out
    assert "fqe_loss_x" not in out


def test_fqe_loss_terminal_target_equals_reward():
    """For a 1-pitch PA where is_terminal=True, target = reward (no bootstrap)."""
    fqe = _tiny_model()
    policy = _tiny_model()
    # All-terminal batch: target should be just reward
    n_pt = VOCAB_SIZES["pitch_type"]
    B, T = 1, 1
    pre_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in PRE_ACTION_CATEGORICAL_FIELDS}
    pre_cat["pitcher_id"] = torch.tensor([[1]])
    batch = PABatch(
        pre_cat=pre_cat,
        pre_cont=torch.randn(B, T, len(PRE_ACTION_CONTINUOUS_FIELDS)),
        profile=torch.randn(B, T, len(BATTER_PROFILE_OVERALL_FIELDS)),
        post_cat={f: torch.zeros(B, T, dtype=torch.int64) for f in POST_ACTION_CATEGORICAL_FIELDS},
        post_cont=torch.randn(B, T, len(POST_ACTION_CONTINUOUS_FIELDS)),
        reward=torch.tensor([[0.5]]),
        is_terminal=torch.tensor([[True]]),
        pa_lengths=torch.tensor([T], dtype=torch.int64),
        valid_mask=torch.ones(B, T, dtype=torch.bool),
        arsenal_per_type=torch.randn(B, T, n_pt, 14),
        batter_per_type=torch.randn(B, T, n_pt, 4),
    )
    out = fqe_loss(fqe, policy, batch, gamma=1.0)
    # target_mean should equal reward_mean = 0.5
    assert out["target_mean"].item() == pytest.approx(0.5, rel=1e-5)


def test_fqe_loss_does_not_train_shallow_heads():
    """FQE trains ONLY q_head_z. The shallow heads (q_head_type, q_head_x) and
    v_head must receive zero gradient from fqe_loss — using them would either be
    wasted compute (v_head) or actively harmful (max over q_x/q_z logits computes
    Q*, biasing FQE upward). Catches a future regression where someone adds
    shallow losses back to fqe_loss."""
    fqe = _tiny_model()
    policy = _tiny_model()
    for p in policy.parameters():
        p.requires_grad_(False)
    batch = _make_batch()
    fqe.zero_grad()
    out = fqe_loss(fqe, policy, batch, gamma=1.0)
    out["fqe_loss"].backward()

    EXPECTED_NO_GRAD_PREFIXES = ("q_head_type.", "q_head_x.", "v_head.")
    bad: list[str] = []
    for name, p in fqe.named_parameters():
        if name.startswith(EXPECTED_NO_GRAD_PREFIXES):
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                bad.append(name)
    assert not bad, (
        f"FQE should not train shallow heads or v_head, but these got gradient: {bad[:5]}. "
        "Someone may have re-added shallow losses to fqe_loss — that biases the FQE "
        "estimate upward (computes Q* instead of Q^π)."
    )


def test_estimate_pa_values_returns_finite_estimates():
    fqe = _tiny_model()
    policy = _tiny_model()
    batches = [_make_batch() for _ in range(3)]
    # Build a fake "loader" — list of batches
    out = estimate_pa_values(fqe, policy, batches, device=torch.device("cpu"))
    assert "learned_per_pa" in out
    assert "behavior_per_pa" in out
    assert "advantage" in out
    assert torch.tensor(out["learned_per_pa"]).isfinite().item()


class _SyntheticPADataset:
    def __init__(self, n_items: int = 16):
        torch.manual_seed(0)
        n_pt = VOCAB_SIZES["pitch_type"]
        T = 3
        self._data = []
        for i in range(n_items):
            d = {}
            for f in PRE_ACTION_CATEGORICAL_FIELDS:
                d[f"pre_cat__{f}"] = torch.tensor([1, 1, 1], dtype=torch.int64)
            d["pre_cont"] = torch.randn(T, len(PRE_ACTION_CONTINUOUS_FIELDS))
            d["profile"] = torch.randn(T, len(BATTER_PROFILE_OVERALL_FIELDS))
            for f in POST_ACTION_CATEGORICAL_FIELDS:
                d[f"post_cat__{f}"] = torch.zeros(T, dtype=torch.int64)
            d["post_cont"] = torch.randn(T, len(POST_ACTION_CONTINUOUS_FIELDS))
            d["reward"] = torch.tensor([0.1, -0.05, 0.3])
            d["is_terminal"] = torch.tensor([False, False, True])
            d["pa_length"] = torch.tensor(T, dtype=torch.int32)
            d["arsenal_per_type"] = torch.randn(T, n_pt, 14)
            d["batter_per_type"] = torch.randn(T, n_pt, 4)
            self._data.append(d)

    def __len__(self): return len(self._data)
    def __getitem__(self, idx): return self._data[idx]


def test_fqe_trainer_loss_decreases_on_synthetic_data():
    train_ds = _SyntheticPADataset(n_items=16)
    val_ds = _SyntheticPADataset(n_items=4)
    fqe = _tiny_model()
    policy = _tiny_model()
    cfg = FQETrainerConfig(
        lr=1e-3, warmup_steps=2, batch_size=4, num_workers=0,
        epochs=2, bf16=False, pin_memory=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = FQETrainer(fqe, policy, train_ds, val_ds, cfg, torch.device("cpu"), Path(tmp))
        first_batch = next(iter(trainer.train_loader))
        first = trainer.step(first_batch)
        trainer.fit()
        final = trainer.evaluate()
        assert final["fqe_loss"] < first["fqe_loss"], \
            f"FQE loss did not decrease: first={first['fqe_loss']} final={final['fqe_loss']}"


def test_fqe_policy_model_remains_frozen():
    """π_learned weights must not change during FQE training."""
    train_ds = _SyntheticPADataset(n_items=8)
    val_ds = _SyntheticPADataset(n_items=4)
    fqe = _tiny_model()
    policy = _tiny_model()
    snapshot = {n: p.data.clone() for n, p in policy.named_parameters()}

    cfg = FQETrainerConfig(
        lr=1e-3, warmup_steps=1, batch_size=4, num_workers=0,
        epochs=1, bf16=False, pin_memory=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = FQETrainer(fqe, policy, train_ds, val_ds, cfg, torch.device("cpu"), Path(tmp))
        trainer.fit()

    for n, p in policy.named_parameters():
        assert torch.equal(p.data, snapshot[n]), f"policy parameter {n} changed during FQE training"


# --------------------------------------------------------------------------- #
# Repertoire mask hyperparameter (default OFF)
# --------------------------------------------------------------------------- #


def test_fqe_loss_passes_no_mask_by_default():
    """With repertoire_mask_min_count=0 (default), policy_model.policy should
    be called WITHOUT a mask (None). Captured via monkey-patch."""
    from src.qtransformer import QTransformer as _QT
    fqe = _tiny_model()
    policy = _tiny_model()
    for p in policy.parameters():
        p.requires_grad_(False)
    batch = _make_batch()

    captured = {}
    original_policy = _QT.policy
    def _spy(self, batch, repertoire_mask=None, return_logits=False):
        captured["mask"] = repertoire_mask
        return original_policy(self, batch, repertoire_mask=repertoire_mask, return_logits=return_logits)
    _QT.policy = _spy
    try:
        fqe_loss(fqe, policy, batch, gamma=1.0)  # default repertoire_mask_min_count=0
    finally:
        _QT.policy = original_policy
    assert captured["mask"] is None, f"expected None mask by default, got {type(captured['mask'])}"


def test_fqe_loss_passes_mask_when_enabled():
    """With repertoire_mask_min_count > 0, policy_model.policy should be called
    with a (B, T, n_pitch_types) bool tensor."""
    from src.qtransformer import QTransformer as _QT
    fqe = _tiny_model()
    policy = _tiny_model()
    for p in policy.parameters():
        p.requires_grad_(False)
    batch = _make_batch()
    # Set non-trivial arsenal counts so the mask isn't trivial
    batch.arsenal_per_type[..., 0] = 100  # all types past threshold

    captured = {}
    original_policy = _QT.policy
    def _spy(self, batch, repertoire_mask=None, return_logits=False):
        captured["mask"] = repertoire_mask
        return original_policy(self, batch, repertoire_mask=repertoire_mask, return_logits=return_logits)
    _QT.policy = _spy
    try:
        fqe_loss(fqe, policy, batch, gamma=1.0, repertoire_mask_min_count=10)
    finally:
        _QT.policy = original_policy
    assert captured["mask"] is not None, "mask should be passed when min_count > 0"
    assert captured["mask"].dtype == torch.bool
    n_pt = VOCAB_SIZES["pitch_type"]
    assert captured["mask"].shape == (batch.valid_mask.shape[0], batch.valid_mask.shape[1], n_pt)


def test_fqe_trainer_config_default_mask_disabled():
    """Default config must have mask disabled — locks in the design choice."""
    cfg = FQETrainerConfig()
    assert cfg.repertoire_mask_min_count == 0, \
        "default repertoire_mask_min_count must be 0 (disabled); see CLAUDE.md design rationale"


def test_fqe_target_dropout_silenced_in_target_forward():
    """fqe_loss should put fqe_model in eval mode for the target-side forward
    so the q_head_z MLP's dropout doesn't fire on the target. We verify by
    spying on heads_chosen and capturing fqe.training at each call site.

    Note: encoder/transformer h_fqe is shared between prediction and target
    sides (computed once, reused), so encoder dropout is NOT independently
    drawn on the target side. The vulnerability is specifically in the head
    MLPs' dropout layers, which run twice (once per gather call).
    """
    import types
    fqe = _tiny_model()
    policy = _tiny_model()
    for p in policy.parameters():
        p.requires_grad_(False)
    fqe.train()
    assert fqe.training is True
    batch = _make_batch()

    captured_modes: list[bool] = []
    original_heads_chosen = fqe.heads_chosen.__func__

    def spy(self, *args, **kwargs):
        captured_modes.append(self.training)
        return original_heads_chosen(self, *args, **kwargs)

    fqe.heads_chosen = types.MethodType(spy, fqe)
    try:
        fqe_loss(fqe, policy, batch, gamma=1.0)
    finally:
        del fqe.heads_chosen  # restore the bound method to the class one

    # Two heads_chosen calls in fqe_loss: first for chosen actions (prediction),
    # second for policy actions (target). We expect:
    #   - first call: training=True (dropout active for prediction)
    #   - second call: training=False (dropout silenced for target)
    assert captured_modes == [True, False], (
        f"heads_chosen training modes were {captured_modes}, expected [True, False] "
        "(prediction in train mode, target in eval mode)"
    )


def test_fqe_loss_restores_train_mode_after_call():
    """The target-forward eval-mode wrapper must restore the prior mode."""
    fqe = _tiny_model()
    policy = _tiny_model()
    batch = _make_batch()
    fqe.train()
    fqe_loss(fqe, policy, batch, gamma=1.0)
    assert fqe.training is True, "fqe_model should be back in train mode after fqe_loss"
