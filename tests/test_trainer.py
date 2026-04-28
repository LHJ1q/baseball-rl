"""Unit tests for src/trainer.py — LR schedule, checkpoint round-trip, loss decreasing."""
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
from src.qtransformer import QTransformer
from src.trainer import Trainer, TrainerConfig, cosine_warmup_lr


VOCAB_SIZES = {
    "pitch_type": 6, "description": 4,
    "p_throws": 2, "stand": 2, "inning_topbot": 2,
    "batter": 20, "pitcher": 20,
}


# --------------------------------------------------------------------------- #
# Cosine warmup schedule
# --------------------------------------------------------------------------- #


def test_cosine_warmup_zero_at_first_step_after_warmup_start():
    lr = cosine_warmup_lr(step=0, base_lr=1e-3, warmup_steps=100, total_steps=1000)
    # First step: linear warmup, lr ≈ base_lr * 1/100
    assert lr == pytest.approx(1e-5, rel=1e-3)


def test_cosine_warmup_reaches_base_lr_at_warmup_end():
    lr = cosine_warmup_lr(step=99, base_lr=1e-3, warmup_steps=100, total_steps=1000)
    assert lr == pytest.approx(1e-3, rel=1e-3)


def test_cosine_warmup_reaches_zero_at_end():
    lr = cosine_warmup_lr(step=1000, base_lr=1e-3, warmup_steps=100, total_steps=1000)
    assert lr == pytest.approx(0.0, abs=1e-9)


def test_cosine_warmup_is_monotone_decreasing_after_warmup():
    base = 1e-3
    points = [cosine_warmup_lr(s, base_lr=base, warmup_steps=100, total_steps=1000)
              for s in (100, 300, 500, 700, 900)]
    for a, b in zip(points, points[1:]):
        assert a > b


# --------------------------------------------------------------------------- #
# Trainer end-to-end smoke (synthetic dataset, no disk)
# --------------------------------------------------------------------------- #


class _SyntheticPADataset:
    """Fixed-content dataset for trainer tests — yields the same PA every time
    so the loss should drop quickly."""

    def __init__(self, n_items: int = 32):
        self.n_items = n_items
        torch.manual_seed(0)
        self._data = [self._make_one(i) for i in range(n_items)]

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]

    def _make_one(self, idx: int) -> dict:
        T = 3
        n_pt = VOCAB_SIZES["pitch_type"]
        out = {}
        for f in PRE_ACTION_CATEGORICAL_FIELDS:
            out[f"pre_cat__{f}"] = torch.tensor([1, 1, 1], dtype=torch.int64)
        out["pre_cont"] = torch.randn(T, len(PRE_ACTION_CONTINUOUS_FIELDS))
        out["profile"] = torch.randn(T, len(BATTER_PROFILE_OVERALL_FIELDS))
        for f in POST_ACTION_CATEGORICAL_FIELDS:
            out[f"post_cat__{f}"] = torch.zeros(T, dtype=torch.int64)
        out["post_cont"] = torch.randn(T, len(POST_ACTION_CONTINUOUS_FIELDS))
        out["reward"] = torch.tensor([0.1, -0.05, 0.3])
        out["is_terminal"] = torch.tensor([False, False, True])
        out["pa_length"] = torch.tensor(T, dtype=torch.int32)
        out["arsenal_per_type"] = torch.randn(T, n_pt, 14)
        out["batter_per_type"] = torch.randn(T, n_pt, 4)
        return out


def _tiny_model() -> QTransformer:
    enc, q = load_preset("smoke")
    return QTransformer(VOCAB_SIZES, cfg=q, encoder_cfg=enc)


def test_trainer_loss_decreases_on_synthetic_data():
    model = _tiny_model()
    train_ds = _SyntheticPADataset(n_items=32)
    val_ds = _SyntheticPADataset(n_items=8)

    cfg = TrainerConfig(
        lr=1e-3, warmup_steps=2, batch_size=8, num_workers=0,
        epochs=2, bf16=False, pin_memory=False, include_pitcher_blind_eval=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(model, train_ds, val_ds, cfg, torch.device("cpu"), Path(tmp))
        # Capture the very first step loss
        first_batch = next(iter(trainer.train_loader))
        first_metrics = trainer.step(first_batch)
        # Run remaining training
        trainer.fit()
        eval_metrics = trainer.evaluate()
        assert eval_metrics["q_loss"] < first_metrics["q_loss"], \
            f"Q-loss did not decrease: first={first_metrics['q_loss']} final={eval_metrics['q_loss']}"


def test_trainer_checkpoint_roundtrip():
    model = _tiny_model()
    train_ds = _SyntheticPADataset(n_items=16)
    val_ds = _SyntheticPADataset(n_items=4)

    cfg = TrainerConfig(
        lr=1e-3, warmup_steps=2, batch_size=4, num_workers=0,
        epochs=1, bf16=False, pin_memory=False, include_pitcher_blind_eval=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(model, train_ds, val_ds, cfg, torch.device("cpu"), Path(tmp))
        trainer.fit()
        # Save explicitly and reload into a fresh model
        ckpt = Path(tmp) / "test_ckpt.pt"
        trainer.save_checkpoint(ckpt)

        model2 = _tiny_model()
        trainer2 = Trainer(model2, train_ds, val_ds, cfg, torch.device("cpu"), Path(tmp) / "run2")
        trainer2.load_checkpoint(ckpt)

        # Forward outputs should match
        model.eval(); model2.eval()
        batch = pa_collate([train_ds[0], train_ds[1]])
        with torch.no_grad():
            out1 = model(batch)["q_chosen"]
            out2 = model2(batch)["q_chosen"]
        torch.testing.assert_close(out1, out2)


def test_trainer_metrics_csv_is_written():
    model = _tiny_model()
    train_ds = _SyntheticPADataset(n_items=16)
    val_ds = _SyntheticPADataset(n_items=4)
    cfg = TrainerConfig(
        lr=1e-3, warmup_steps=2, batch_size=4, num_workers=0,
        epochs=1, bf16=False, pin_memory=False, include_pitcher_blind_eval=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "test_run"
        trainer = Trainer(model, train_ds, val_ds, cfg, torch.device("cpu"), run_dir)
        trainer.fit()
        metrics_file = run_dir / "metrics.csv"
        assert metrics_file.exists()
        content = metrics_file.read_text()
        assert "q_loss" in content
        assert "phase" in content


# --------------------------------------------------------------------------- #
# Runtime is_terminal invariant assertion in Trainer.step
# --------------------------------------------------------------------------- #


def test_trainer_step_asserts_is_terminal_invariant():
    """Trainer.step must reject batches where any PA has != 1 terminal pitches.
    The invariant is load-bearing for shift_v_for_next_state's zero pad."""
    import pytest as _pytest
    model = _tiny_model()
    train_ds = _SyntheticPADataset(n_items=8)
    val_ds = _SyntheticPADataset(n_items=4)
    cfg = TrainerConfig(
        lr=1e-3, warmup_steps=2, batch_size=4, num_workers=0,
        epochs=1, bf16=False, pin_memory=False, include_pitcher_blind_eval=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(model, train_ds, val_ds, cfg, torch.device("cpu"), Path(tmp))
        bad_batch = next(iter(trainer.train_loader))
        # Corrupt the batch: zero out is_terminal so no PA has a terminal pitch.
        bad_batch.is_terminal.zero_()
        with _pytest.raises(AssertionError, match="is_terminal invariant violated"):
            trainer.step(bad_batch)


def test_trainer_load_checkpoint_warns_on_epochs_mismatch(caplog):
    """Resuming with a different --epochs than the original silently shifts the
    cosine LR schedule — must emit a loud warning so the user knows."""
    import logging
    model = _tiny_model()
    train_ds = _SyntheticPADataset(n_items=8)
    val_ds = _SyntheticPADataset(n_items=4)

    # Original run: epochs=4
    cfg_orig = TrainerConfig(
        lr=1e-3, warmup_steps=2, batch_size=4, num_workers=0,
        epochs=4, bf16=False, pin_memory=False, include_pitcher_blind_eval=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(model, train_ds, val_ds, cfg_orig, torch.device("cpu"), Path(tmp))
        trainer.fit()
        ckpt = Path(tmp) / "test_ckpt.pt"
        trainer.save_checkpoint(ckpt)

        # Resume with epochs=8 (different) — should warn
        model2 = _tiny_model()
        cfg_resume = TrainerConfig(
            lr=1e-3, warmup_steps=2, batch_size=4, num_workers=0,
            epochs=8, bf16=False, pin_memory=False, include_pitcher_blind_eval=False,
        )
        trainer2 = Trainer(model2, train_ds, val_ds, cfg_resume, torch.device("cpu"), Path(tmp) / "run2")
        with caplog.at_level(logging.WARNING, logger="src.trainer"):
            trainer2.load_checkpoint(ckpt)
        assert any("EPOCHS MISMATCH ON RESUME" in r.message for r in caplog.records), \
            "expected warning about epochs mismatch on resume"


def test_trainer_load_checkpoint_silent_when_epochs_match(caplog):
    """Same --epochs on resume = no warning (the standard usage pattern)."""
    import logging
    model = _tiny_model()
    train_ds = _SyntheticPADataset(n_items=8)
    val_ds = _SyntheticPADataset(n_items=4)

    cfg = TrainerConfig(
        lr=1e-3, warmup_steps=2, batch_size=4, num_workers=0,
        epochs=4, bf16=False, pin_memory=False, include_pitcher_blind_eval=False,
    )
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(model, train_ds, val_ds, cfg, torch.device("cpu"), Path(tmp))
        trainer.fit()
        ckpt = Path(tmp) / "test_ckpt.pt"
        trainer.save_checkpoint(ckpt)

        model2 = _tiny_model()
        trainer2 = Trainer(model2, train_ds, val_ds, cfg, torch.device("cpu"), Path(tmp) / "run2")
        with caplog.at_level(logging.WARNING, logger="src.trainer"):
            trainer2.load_checkpoint(ckpt)
        assert not any("EPOCHS MISMATCH" in r.message for r in caplog.records), \
            "should not warn when epochs match"
