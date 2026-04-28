"""Validation metrics for the IQL Q-Transformer.

Two parallel paths:

* ``eval_losses(model, batch, gamma, tau)`` — standard Q+V losses on the val
  data, model forward unchanged.
* ``eval_pitcher_blind(model, batch, ...)`` — same losses with ``pitcher_id``
  zeroed (UNK index) before the encoder forward. Measures how much the model
  relies on pitcher identity vs. shared general rules. The gap between the
  two is the diagnostic signal for whether to enable embedding dropout in
  future training runs.

``evaluate_dataset`` walks a DataLoader, averages per-batch metrics weighted
by the number of valid pitches per batch, returns a single dict.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from src.dataset import PABatch
from src.qtransformer import QTransformer, iql_losses, shift_v_for_next_state


# --------------------------------------------------------------------------- #
# Per-batch loss helpers
# --------------------------------------------------------------------------- #


def eval_losses(model: QTransformer, batch: PABatch, gamma: float, tau: float) -> dict[str, torch.Tensor]:
    """Forward + IQL losses. Returns ``{q_loss, v_loss, n_valid, q_loss_type/x/z}``."""
    out = model(batch)
    v_next = shift_v_for_next_state(out["v"], batch.valid_mask)
    losses = iql_losses(
        q_type=out["q_type"],
        q_x=out["q_x"],
        q_z=out["q_z"],
        q_x_logits=out["q_x_logits"],
        q_z_logits=out["q_z_logits"],
        v_current=out["v"],
        v_next=v_next,
        reward=batch.reward,
        is_terminal=batch.is_terminal,
        valid_mask=batch.valid_mask,
        gamma=gamma,
        tau=tau,
    )
    losses["n_valid"] = batch.valid_mask.sum().to(losses["q_loss"].dtype)
    return losses


@contextmanager
def _zeroed_pitcher_embedding(model: QTransformer):
    """Temporarily zero the pitcher embedding row for every pitcher_id.

    Zeroes ``emb_pitcher.weight`` only — does NOT touch ``arsenal_per_type``.
    For a TRUE pitcher-blind eval (where the model has no pitcher info at all)
    use :func:`_fully_blinded_pitcher` instead — the arsenal is keyed by
    ``(pitcher_id, pitch_type)`` and would otherwise leak the pitcher's
    per-pitch-type physics into the Q-heads even with the embedding zeroed.

    Restore on exit so the model is unchanged after evaluation.
    """
    emb = model.pre_encoder.emb_pitcher
    original = emb.weight.data.clone()
    with torch.no_grad():
        emb.weight.data.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            emb.weight.data.copy_(original)


@contextmanager
def _fully_blinded_pitcher(model: QTransformer, batch: PABatch):
    """Truly blind the model to pitcher identity. Zeroes both:
      1. ``model.pre_encoder.emb_pitcher.weight``, and
      2. ``batch.arsenal_per_type`` to defaults (``count=0, low_sample=1.0``,
         all stats ``= 0``) — matching the dataset's "unknown pitcher" fallback.

    The arsenal lookup is keyed by ``(pitcher_id, pitch_type)`` and feeds the
    pitcher's per-pitch-type physics into the Q-heads via the per-type join.
    Zeroing only the embedding leaves this leak intact — which silently
    understates the diagnostic gap. This context manager closes that leak.

    Restores both on exit.
    """
    emb = model.pre_encoder.emb_pitcher
    original_emb = emb.weight.data.clone()
    original_arsenal = batch.arsenal_per_type.clone()
    with torch.no_grad():
        emb.weight.data.zero_()
        batch.arsenal_per_type.zero_()
        # ARSENAL_HEAD_FIELDS index 1 = 'low_sample'; set to 1.0 = "stranger pitcher"
        batch.arsenal_per_type[..., 1] = 1.0
    try:
        yield
    finally:
        with torch.no_grad():
            emb.weight.data.copy_(original_emb)
            batch.arsenal_per_type.copy_(original_arsenal)


def eval_pitcher_blind(
    model: QTransformer, batch: PABatch, gamma: float, tau: float
) -> dict[str, torch.Tensor]:
    """Same Q+V losses, but with pitcher information FULLY blinded.

    Uses :func:`_fully_blinded_pitcher` — zeroes both the pitcher embedding AND
    ``arsenal_per_type`` (the per-(pitcher, pitch_type) physics lookup, keyed
    by pitcher_id). The "gap" between this and the personalized loss is the
    actual measure of how much the model relies on pitcher identity.

    NOTE on comparing this to the personalized loss: under per-axis IQL, the
    targets for ``q_loss_type`` and ``q_loss_x`` are derived from the model's
    own logits (max over q_x_logits and q_z_logits respectively). Pitcher-
    blinding zeroes the pitcher embedding for ALL forward passes, so the
    blind-pass logits *and* the blind-pass targets shift together. The
    aggregate loss values are therefore not strictly comparable across the two
    runs in absolute units. The intended diagnostic — "did the loss go up
    under blinding" → "is the model relying on pitcher identity" — is still
    valid in *relative* terms. ``q_loss_z``'s target is stationary (depends
    only on r + γV(s')) and IS directly comparable across the two.
    """
    with _fully_blinded_pitcher(model, batch):
        return eval_losses(model, batch, gamma, tau)


# --------------------------------------------------------------------------- #
# Dataset-level aggregator
# --------------------------------------------------------------------------- #


def _move_batch(batch: PABatch, device) -> PABatch:
    return batch.to(device)


@torch.no_grad()
def evaluate_dataset(
    model: QTransformer,
    loader: Iterable,
    *,
    gamma: float,
    tau: float,
    device,
    include_pitcher_blind: bool = True,
) -> dict[str, float]:
    """Run validation across a DataLoader. Per-batch losses are weighted by
    ``n_valid`` (number of real pitches in the batch) so PA-length variation
    doesn't bias the average.

    Returns a dict with keys ``q_loss``, ``v_loss`` and (if requested) the
    pitcher-blind variants ``q_loss_blind``, ``v_loss_blind``, plus the gap
    ``q_loss_blind_gap = q_loss_blind - q_loss``.
    """
    model.eval()

    sum_q = sum_v = 0.0
    sum_n = 0.0
    sum_q_blind = sum_v_blind = 0.0

    for batch in loader:
        batch = _move_batch(batch, device)
        n = float(batch.valid_mask.sum().item())
        if n == 0:
            continue
        sum_n += n

        l = eval_losses(model, batch, gamma, tau)
        sum_q += float(l["q_loss"].item()) * n
        sum_v += float(l["v_loss"].item()) * n

        if include_pitcher_blind:
            lb = eval_pitcher_blind(model, batch, gamma, tau)
            sum_q_blind += float(lb["q_loss"].item()) * n
            sum_v_blind += float(lb["v_loss"].item()) * n

    if sum_n == 0:
        out = {"q_loss": float("nan"), "v_loss": float("nan")}
        if include_pitcher_blind:
            out.update({"q_loss_blind": float("nan"), "v_loss_blind": float("nan"), "q_loss_blind_gap": float("nan")})
        return out

    out = {"q_loss": sum_q / sum_n, "v_loss": sum_v / sum_n}
    if include_pitcher_blind:
        q_blind = sum_q_blind / sum_n
        v_blind = sum_v_blind / sum_n
        out.update({
            "q_loss_blind": q_blind,
            "v_loss_blind": v_blind,
            "q_loss_blind_gap": q_blind - out["q_loss"],
        })
    return out
