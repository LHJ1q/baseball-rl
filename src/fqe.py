"""Phase 9 Part B — Fitted Q Evaluation (FQE).

Trains a *separate* Q-network ``Q^π`` to estimate the value of a fixed,
trained policy ``π_learned``. Mirrors the IQL training infrastructure but uses
the on-policy SARSA target instead of IQL's expectile + V-bootstrap target:

    Q^π(s, a)  ←  r + γ · Q^π(s', π_learned(s'))      for non-terminal
    Q^π(s, a)  ←  r                                    for terminal

Output: an estimate of the per-PA expected return under ``π_learned`` on the
held-out splits, plus per-state Q values for further analysis.

The ``FQEModel`` reuses :class:`src.qtransformer.QTransformer` as the
architecture (same as the IQL preset). What differs:

1. Loaded with ``π_learned``'s checkpoint as initialization (so the value
   surface starts close to π_learned's, then refines via SARSA).
2. Trained with the FQE loss in :func:`fqe_loss`, not the IQL Q+V loss.
3. ``π_learned`` is held frozen during FQE training and used only to compute
   next-state actions for the SARSA target.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import PABatch, PitchPADataset, pa_collate
from src.qtransformer import QTransformer, repertoire_mask_from_batch
from src.trainer import cosine_warmup_lr

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def _gather_qz_at_actions(
    fqe_model: QTransformer,
    h_pre: torch.Tensor,                        # (B, T, d_model)
    arsenal_per_type: torch.Tensor,             # (B, T, n_pt, k_arsenal)
    batter_per_type: torch.Tensor,              # (B, T, n_pt, k_batter_pt)
    chosen_pitch_type: torch.Tensor,            # (B, T) int64
    chosen_x_bin: torch.Tensor,                 # (B, T) int64
    chosen_z_bin: torch.Tensor,                 # (B, T) int64
) -> torch.Tensor:
    """Evaluate fqe_model's deepest Q head at the given (state, action) tuples.
    Returns ``q_z`` of shape ``(B, T)``.

    NOTE: FQE only trains and consumes the deepest head ``q_z``. The shallow
    heads ``q_head_type`` and ``q_head_x`` are intentionally left untrained
    in FQE — using a per-axis max over their logits would compute Q* (optimal
    value), which biases FQE estimates upward and defeats the entire point of
    FQE (estimating Q^π for a fixed policy, not Q*).
    """
    out = fqe_model.heads_chosen(
        h_pre, chosen_pitch_type, chosen_x_bin, chosen_z_bin,
        arsenal_per_type, batter_per_type,
    )
    return out["q_z"]


def fqe_loss(
    fqe_model: QTransformer,
    policy_model: QTransformer,
    batch: PABatch,
    *,
    gamma: float = 1.0,
    repertoire_mask_min_count: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute the FQE TD loss for one batch.

    Trains only the deepest Q head ``q_z`` against the SARSA-style target
    ``r + γ · q_z(s', π_learned(s'))``. The shallow heads ``q_head_type`` and
    ``q_head_x`` are intentionally NOT trained here — see :func:`_gather_qz_at_actions`
    for why (a per-axis max would compute Q*, biasing FQE estimates upward).

    ``repertoire_mask_min_count`` (default 0 = disabled) lets you mask
    ``π_learned`` to a per-pitcher repertoire when computing the bootstrap
    action. When > 0, FQE evaluates the *masked* policy; when 0, it evaluates
    the unmasked argmax. Set to whatever matches your intended deployment.

    Both models share the same architecture but are separate instances —
    ``policy_model`` is ``π_learned`` (frozen) and ``fqe_model`` is the
    Q-network being trained.
    """
    h_fqe = fqe_model.encode(batch)[:, 0::2]                # (B, T, d_model)

    # Q^π(s_t, a_t) at the actions the behavior policy actually took.
    q_z = _gather_qz_at_actions(
        fqe_model, h_fqe,
        batch.arsenal_per_type, batch.batter_per_type,
        batch.post_cat["pitch_type_id"], batch.post_cat["x_bin"], batch.post_cat["z_bin"],
    )

    # π_learned's actions at every state (its argmax greedy policy, optionally
    # constrained by the per-pitcher repertoire mask).
    with torch.no_grad():
        rmask = repertoire_mask_from_batch(batch, repertoire_mask_min_count)
        policy_out = policy_model.policy(batch, repertoire_mask=rmask)
        # Put fqe_model in eval mode for the TARGET-side forward to silence
        # dropout. Prediction (q_z above) and target (q_z_at_policy) would
        # otherwise see independent dropout masks at q_head_z, inflating
        # target variance unnecessarily. Restore prior mode on exit.
        was_training = fqe_model.training
        fqe_model.eval()
        try:
            q_z_at_policy = _gather_qz_at_actions(
                fqe_model, h_fqe,
                batch.arsenal_per_type, batch.batter_per_type,
                policy_out["pitch_type"], policy_out["x_bin"], policy_out["z_bin"],
            )
        finally:
            if was_training:
                fqe_model.train()

    # Shift left by one to get Q^π(s_{t+1}, π_learned(s_{t+1})).
    q_next = torch.zeros_like(q_z_at_policy)
    q_next[:, :-1] = q_z_at_policy[:, 1:].detach()
    target = (batch.reward + gamma * q_next * (~batch.is_terminal).float()).detach()

    mask = batch.valid_mask.float()
    n = mask.sum().clamp_min(1.0)
    fqe_loss_z = (((q_z - target).pow(2)) * mask).sum() / n

    return {
        "fqe_loss": fqe_loss_z,
        "q_mean": (q_z * mask).sum() / n,
        "target_mean": (target * mask).sum() / n,
    }


# --------------------------------------------------------------------------- #
# Trainer config + class
# --------------------------------------------------------------------------- #


@dataclass
class FQETrainerConfig:
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True

    epochs: int = 20            # FQE typically converges faster than IQL since target is simpler
    checkpoint_every_epochs: int = 1

    gamma: float = 1.0
    bf16: bool = True
    seed: int = 0

    # Cosine LR floor (matches TrainerConfig)
    min_lr_factor: float = 0.1

    # torch.compile on CUDA (auto-skipped on MPS/CPU). Compiles fqe_model.encode
    # and .heads_chosen — the methods fqe_loss / estimate_pa_values invoke.
    # policy_model stays uncompiled to keep the IQL-checkpoint load path simple.
    compile: bool = True

    # Repertoire mask threshold (default 0 = mask disabled). Set > 0 to constrain
    # ``π_learned``'s argmax to pitch types this pitcher has thrown at least N times in train.
    # See src/qtransformer.py:repertoire_mask_from_batch for the rationale (default OFF).
    repertoire_mask_min_count: int = 0


class FQETrainer:
    """Mirror of :class:`src.trainer.Trainer` but with the FQE target.

    Both ``policy_model`` (frozen) and ``fqe_model`` (trainable) live on the
    same device; the FQE forward also depends on policy_model's ``policy()``
    which is called under ``no_grad``.
    """

    def __init__(
        self,
        fqe_model: QTransformer,
        policy_model: QTransformer,
        train_ds: PitchPADataset,
        val_ds: PitchPADataset,
        cfg: FQETrainerConfig,
        device: torch.device,
        run_dir: Path,
    ):
        self.fqe_model = fqe_model
        self.policy_model = policy_model
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TF32 for fp32 matmul corners (BF16 autocast covers the rest).
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        self.fqe_model.to(self.device)
        self.policy_model.to(self.device)
        # π_learned is FROZEN during FQE training.
        for p in self.policy_model.parameters():
            p.requires_grad = False
        self.policy_model.eval()

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # Dataloader hot-path: persistent workers + prefetch (matches Trainer).
        _persist = cfg.num_workers > 0
        _prefetch = 4 if cfg.num_workers > 0 else None
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            collate_fn=pa_collate, drop_last=True,
            persistent_workers=_persist, prefetch_factor=_prefetch,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            collate_fn=pa_collate,
            persistent_workers=_persist, prefetch_factor=_prefetch,
        )
        self.steps_per_epoch = max(1, len(self.train_loader))
        self.total_steps = self.steps_per_epoch * cfg.epochs

        self.optimizer = torch.optim.AdamW(
            self.fqe_model.parameters(),
            lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay,
        )

        # torch.compile applied to the methods fqe_loss and estimate_pa_values
        # actually invoke — `.encode(batch)` and `.heads_chosen(...)`. We do NOT
        # compile fqe_model.__call__ (the forward path) because fqe_loss never
        # calls it: it composes encode + heads_chosen at chosen actions and
        # again at policy actions. Wrapping fqe_model itself would route only
        # the unused forward through the compiler, giving zero speedup. Method-
        # level compile is the equivalent path.
        # policy_model stays uncompiled — it's frozen and called only via
        # .policy(), keeping the IQL-checkpoint state_dict load path simple.
        if cfg.compile and self.device.type == "cuda":
            self.fqe_model.encode = torch.compile(self.fqe_model.encode, dynamic=True)
            self.fqe_model.heads_chosen = torch.compile(self.fqe_model.heads_chosen, dynamic=True)
            logger.info("torch.compile applied to fqe_model.encode + .heads_chosen (CUDA)")

        self._use_bf16 = cfg.bf16 and self.device.type == "cuda"

        self.metrics_path = self.run_dir / "fqe_metrics.csv"
        self._csv_file = None
        self._csv_writer = None
        self._metrics_columns: list[str] | None = None

        self.global_step = 0
        self.epoch = 0

    # --------------------------------------------------------------------- #

    def _ensure_csv(self, columns: list[str]) -> None:
        if self._csv_writer is not None:
            return
        self._metrics_columns = columns
        new_file = not self.metrics_path.exists()
        self._csv_file = self.metrics_path.open("a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=columns)
        if new_file:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def _log_row(self, row: dict) -> None:
        if self._metrics_columns is None:
            self._ensure_csv(sorted(row.keys()))
        full = {c: row.get(c, "") for c in self._metrics_columns}
        self._csv_writer.writerow(full)
        self._csv_file.flush()

    def _close_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # --------------------------------------------------------------------- #

    def _set_lr(self) -> float:
        lr = cosine_warmup_lr(
            self.global_step,
            base_lr=self.cfg.lr,
            warmup_steps=self.cfg.warmup_steps,
            total_steps=self.total_steps,
            min_lr_factor=self.cfg.min_lr_factor,
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def step(self, batch: PABatch) -> dict[str, float]:
        self.fqe_model.train()
        batch = batch.to(self.device)
        # Runtime guard: every PA must have exactly one terminal pitch
        # (load-bearing for shift_v_for_next_state's zero pad and FQE's q_next shift).
        assert (batch.is_terminal.sum(dim=1) == 1).all(), \
            "is_terminal invariant violated — check filter pipeline"
        lr = self._set_lr()
        self.optimizer.zero_grad(set_to_none=True)

        autocast_dtype = torch.bfloat16 if self._use_bf16 else None
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self._use_bf16):
            losses = fqe_loss(
                self.fqe_model, self.policy_model, batch,
                gamma=self.cfg.gamma,
                repertoire_mask_min_count=self.cfg.repertoire_mask_min_count,
            )
            loss = losses["fqe_loss"]

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.fqe_model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.global_step += 1

        return {
            "fqe_loss": float(loss.item()),
            "q_mean": float(losses["q_mean"].item()),
            "target_mean": float(losses["target_mean"].item()),
            "lr": lr,
            "grad_norm": float(grad_norm.item()),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.fqe_model.eval()
        sum_loss = sum_q = sum_target = 0.0
        sum_n = 0.0
        for batch in self.val_loader:
            batch = batch.to(self.device)
            n = float(batch.valid_mask.sum().item())
            if n == 0:
                continue
            losses = fqe_loss(
                self.fqe_model, self.policy_model, batch,
                gamma=self.cfg.gamma,
                repertoire_mask_min_count=self.cfg.repertoire_mask_min_count,
            )
            sum_loss += float(losses["fqe_loss"].item()) * n
            sum_q += float(losses["q_mean"].item()) * n
            sum_target += float(losses["target_mean"].item()) * n
            sum_n += n
        if sum_n == 0:
            return {"fqe_loss": float("nan"), "q_mean": float("nan"), "target_mean": float("nan")}
        return {
            "fqe_loss": sum_loss / sum_n,
            "q_mean": sum_q / sum_n,
            "target_mean": sum_target / sum_n,
        }

    # --------------------------------------------------------------------- #

    def save_checkpoint(self, path: Path) -> None:
        # Method-level compile wraps fqe_model.encode / .heads_chosen, NOT
        # fqe_model itself, so state_dict() already has clean keys. The unwrap
        # is a defensive no-op kept for forward-compat in case the wrap shape
        # changes; getattr falls through to fqe_model when _orig_mod is absent.
        underlying = getattr(self.fqe_model, "_orig_mod", self.fqe_model)
        torch.save(
            {
                "model": underlying.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "trainer_cfg": asdict(self.cfg),
            },
            path,
        )

    def fit(self) -> dict[str, float]:
        try:
            final: dict[str, float] = {}
            for epoch_idx in range(self.epoch, self.cfg.epochs):
                self.epoch = epoch_idx
                t0 = time.time()
                run = {"fqe_loss": 0.0, "q_mean": 0.0, "target_mean": 0.0}
                n = 0
                for batch in self.train_loader:
                    m = self.step(batch)
                    self._log_row({"phase": "train", "epoch": epoch_idx, "step": self.global_step, **m})
                    for k in run:
                        run[k] += m[k]
                    n += 1

                avg = {k: v / max(1, n) for k, v in run.items()}
                eval_metrics = self.evaluate()
                final = eval_metrics
                self._log_row({
                    "phase": "eval", "epoch": epoch_idx, "step": self.global_step,
                    **{f"train_avg_{k}": v for k, v in avg.items()},
                    **{f"val_{k}": v for k, v in eval_metrics.items()},
                    "elapsed_s": time.time() - t0,
                })
                logger.info(
                    "epoch %3d | step %6d | train fqe=%.4f q_mean=%.4f | val fqe=%.4f q_mean=%.4f | %.1fs",
                    epoch_idx, self.global_step,
                    avg["fqe_loss"], avg["q_mean"],
                    eval_metrics["fqe_loss"], eval_metrics["q_mean"],
                    time.time() - t0,
                )

                self.save_checkpoint(self.run_dir / "fqe_checkpoint_latest.pt")
                if (epoch_idx + 1) % self.cfg.checkpoint_every_epochs == 0:
                    self.save_checkpoint(self.run_dir / f"fqe_checkpoint_epoch_{epoch_idx}.pt")

            return final
        finally:
            self._close_csv()


# --------------------------------------------------------------------------- #
# Per-PA value estimator — what we actually report after training
# --------------------------------------------------------------------------- #


@torch.no_grad()
def estimate_pa_values(
    fqe_model: QTransformer,
    policy_model: QTransformer,
    loader,
    *,
    device,
    repertoire_mask_min_count: int = 0,
) -> dict[str, float]:
    """Estimate the per-PA expected return under ``π_learned`` using the
    trained FQE model. Returns aggregate stats over the dataset.

    For each PA, the value at the FIRST pitch (state s_0) under π_learned is
    Q^π(s_0, π_learned(s_0)). Aggregating across PAs gives the per-PA mean
    estimated value.

    ``repertoire_mask_min_count`` (default 0 = disabled): if > 0, ``π_learned``
    is constrained to per-pitcher repertoire types via the mask. Use the same
    setting that you would deploy with.
    """
    fqe_model.eval()
    policy_model.eval()
    sum_v = 0.0
    n_pa = 0
    sum_behavior_reward = 0.0  # sum of per-PA behavior reward for comparison
    n_pa_behavior = 0

    for batch in loader:
        batch = batch.to(device)
        h = fqe_model.encode(batch)[:, 0::2]
        rmask = repertoire_mask_from_batch(batch, repertoire_mask_min_count)
        policy_out = policy_model.policy(batch, repertoire_mask=rmask)
        q_at_policy = _gather_qz_at_actions(
            fqe_model, h,
            batch.arsenal_per_type, batch.batter_per_type,
            policy_out["pitch_type"], policy_out["x_bin"], policy_out["z_bin"],
        )
        # First pitch of each PA = pitch_idx 0. Use position 0 of each row.
        first_pitch_q = q_at_policy[:, 0]                       # (B,)
        first_valid = batch.valid_mask[:, 0]
        sum_v += float((first_pitch_q * first_valid.float()).sum().item())
        n_pa += int(first_valid.sum().item())

        # Behavior baseline: actual sum of reward per PA.
        rewards = (batch.reward * batch.valid_mask.float()).sum(dim=1)  # (B,)
        # Count any PA that had at least one valid pitch.
        any_valid = batch.valid_mask.any(dim=1)
        sum_behavior_reward += float((rewards * any_valid.float()).sum().item())
        n_pa_behavior += int(any_valid.sum().item())

    if n_pa == 0:
        return {"learned_per_pa": float("nan"), "behavior_per_pa": float("nan"), "advantage": float("nan")}
    learned = sum_v / n_pa
    behavior = sum_behavior_reward / max(1, n_pa_behavior)
    return {
        "learned_per_pa": learned,
        "behavior_per_pa": behavior,
        "advantage": learned - behavior,
        "n_pa": n_pa,
    }
