"""IQL trainer for the Q-Transformer.

Phase 8 deliverable. Owns the training loop, logging, checkpointing, and
end-of-epoch evaluation. Designed for the user's Blackwell RTX Pro 4500
(BF16 autocast, single-GPU); also runs on CPU/MPS for the Macbook smoke
training run.

Run-directory layout:
    data/runs/{run_name}/
    ├── config.json
    ├── metrics.csv              # one row per train step + per-epoch eval
    ├── checkpoint_latest.pt
    ├── checkpoint_epoch_{N}.pt  # periodic full snapshots
    └── checkpoint_best.pt       # best by val q_loss
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import PABatch, PitchPADataset, pa_collate
from src.eval import eval_pitcher_blind, evaluate_dataset
from src.qtransformer import QTransformer, iql_losses, shift_v_for_next_state

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Trainer config
# --------------------------------------------------------------------------- #


@dataclass
class TrainerConfig:
    # Optimizer
    lr: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    # Batching
    batch_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True

    # Schedule
    epochs: int = 40
    eval_every_steps: int = 0          # 0 = end-of-epoch only
    checkpoint_every_epochs: int = 1

    # IQL
    gamma: float = 1.0
    tau: float = 0.7
    pitcher_dropout: float = 0.0       # default off; toggle for "general rule maker" experiments

    # Mixed precision (BF16 on CUDA only — Macbook MPS/CPU falls back to fp32)
    bf16: bool = True

    # Cosine LR floor — final LR = min_lr_factor * base_lr (was decay to 0)
    min_lr_factor: float = 0.1

    # torch.compile on CUDA (auto-skipped on MPS/CPU). dynamic=True is
    # mandatory because pa_collate produces variable T_max batches.
    compile: bool = True

    # Reproducibility
    seed: int = 0

    # Eval cadence options
    include_pitcher_blind_eval: bool = True


# --------------------------------------------------------------------------- #
# Cosine LR with linear warmup
# --------------------------------------------------------------------------- #


def cosine_warmup_lr(
    step: int, *, base_lr: float, warmup_steps: int, total_steps: int,
    min_lr_factor: float = 0.1,
) -> float:
    """Linear warmup over ``warmup_steps`` then cosine decay from ``base_lr``
    to ``min_lr_factor * base_lr`` over the remaining steps.

    The floor (default 0.1 × base_lr) prevents the last ~10% of training from
    doing nothing, which the previous decay-to-0 schedule effectively was.
    Matches Chinchilla / Llama practice. The clamp also holds past
    ``total_steps`` (a real case when resume sets ``global_step >= total_steps``
    after ``--epochs`` is decreased).
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))   # in [0, 1]
    return base_lr * (min_lr_factor + (1.0 - min_lr_factor) * cosine)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class Trainer:
    """One trainer = one run directory."""

    def __init__(
        self,
        model: QTransformer,
        train_ds: PitchPADataset,
        val_ds: PitchPADataset,
        cfg: TrainerConfig,
        device: torch.device,
        run_dir: Path,
        encoder_q_config_payload: dict | None = None,
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.cfg = cfg
        self.device = device
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Enable TF32 paths for fp32 matmuls (BF16 autocast covers most of
        # forward/backward, but fp32 corners — standardization buffers,
        # attention bias compute — still benefit).
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        self.model.to(self.device)

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # Dataloader hot-path: keep workers warm + prefetch ahead. The
        # bottleneck is parquet I/O + pa_collate, so persistent_workers
        # avoids re-spawning workers each epoch and prefetch_factor=4 keeps
        # the GPU fed.
        _persist = cfg.num_workers > 0
        _prefetch = 4 if cfg.num_workers > 0 else None
        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            collate_fn=pa_collate,
            drop_last=True,
            persistent_workers=_persist,
            prefetch_factor=_prefetch,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            collate_fn=pa_collate,
            persistent_workers=_persist,
            prefetch_factor=_prefetch,
        )

        self.steps_per_epoch = max(1, len(self.train_loader))
        self.total_steps = self.steps_per_epoch * cfg.epochs

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        # torch.compile MUST come AFTER optimizer creation. If we compile first,
        # self.model.parameters() returns _orig_mod-prefixed parameter names and
        # the optimizer holds references through the wrapper — complicates the
        # save/load path. Compiling after lets the optimizer hold the underlying
        # tensors directly while torch.compile only wraps the forward pass.
        # Standard nanoGPT / HuggingFace Trainer pattern.
        if cfg.compile and self.device.type == "cuda":
            self.model = torch.compile(self.model, dynamic=True)
            logger.info("torch.compile(model, dynamic=True) applied (CUDA)")

        self._use_bf16 = cfg.bf16 and self.device.type == "cuda"
        if cfg.bf16 and not self._use_bf16:
            logger.info("BF16 requested but device is %s — falling back to fp32", self.device.type)

        # Checkpoint paths
        self.ckpt_latest = self.run_dir / "checkpoint_latest.pt"
        self.ckpt_best = self.run_dir / "checkpoint_best.pt"

        # CSV logging
        self.metrics_path = self.run_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None
        self._metrics_columns: list[str] | None = None

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_q_loss = float("inf")

        # Save config payload (encoder + qtransformer + trainer) for reproducibility.
        if encoder_q_config_payload is not None:
            cfg_payload = {**encoder_q_config_payload, "trainer": asdict(cfg)}
            (self.run_dir / "config.json").write_text(json.dumps(cfg_payload, indent=2))

    # --------------------------------------------------------------------- #
    # CSV logging
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
        # Stable column set: union of seen columns, padded with empty strings.
        if self._metrics_columns is None:
            self._ensure_csv(sorted(row.keys()))
        # Coerce missing columns to empty strings to keep CSV well-formed.
        full = {c: row.get(c, "") for c in self._metrics_columns}
        self._csv_writer.writerow(full)
        self._csv_file.flush()

    def _close_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # --------------------------------------------------------------------- #
    # Step + epoch
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
        """One optimizer step. Returns ``{q_loss, v_loss, lr, grad_norm}``."""
        self.model.train()
        batch = batch.to(self.device)
        # Runtime guard: every PA must end with exactly one terminal pitch.
        # Load-bearing for shift_v_for_next_state's zero pad — if the filter
        # ever regresses and a PA has 0 or >1 terminals, the V bootstrap on
        # the last position becomes garbage and silently corrupts training.
        assert (batch.is_terminal.sum(dim=1) == 1).all(), \
            "is_terminal invariant violated — check filter pipeline"
        lr = self._set_lr()
        self.optimizer.zero_grad(set_to_none=True)

        autocast_dtype = torch.bfloat16 if self._use_bf16 else None
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self._use_bf16):
            out = self.model(batch)
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
                gamma=self.cfg.gamma,
                tau=self.cfg.tau,
            )
            total = losses["q_loss"] + losses["v_loss"]

        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.global_step += 1

        return {
            "q_loss": float(losses["q_loss"].item()),
            "v_loss": float(losses["v_loss"].item()),
            "lr": lr,
            "grad_norm": float(grad_norm.item()),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        return evaluate_dataset(
            self.model, self.val_loader,
            gamma=self.cfg.gamma, tau=self.cfg.tau, device=self.device,
            include_pitcher_blind=self.cfg.include_pitcher_blind_eval,
        )

    # --------------------------------------------------------------------- #
    # Checkpoint
    # --------------------------------------------------------------------- #

    def save_checkpoint(self, path: Path) -> None:
        # Unwrap if torch.compile wrapped the model — its state_dict adds an
        # `_orig_mod.` prefix to every key. Save the unwrapped state so the
        # checkpoint is loadable by uncompiled models too (e.g., FQE's
        # policy_model.load_state_dict path in scripts/10_fqe.py).
        underlying = getattr(self.model, "_orig_mod", self.model)
        payload = {
            "model": underlying.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_q_loss": self.best_val_q_loss,
            "trainer_cfg": asdict(self.cfg),
        }
        # Atomic save: write to .tmp then rename. torch.save itself is NOT
        # atomic — a crash mid-write (SIGKILL, OOM, power loss) leaves a
        # truncated file at `path`, which breaks --resume on next launch.
        # os.replace is atomic on POSIX (and on Windows since Py 3.3).
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        # Strip _orig_mod. prefix from any incoming state (in case the ckpt was
        # saved by a path that didn't unwrap). Belt-and-suspenders.
        state = payload["model"]
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        underlying = getattr(self.model, "_orig_mod", self.model)
        underlying.load_state_dict(state)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = payload["global_step"]
        self.epoch = payload["epoch"]
        self.best_val_q_loss = payload.get("best_val_q_loss", float("inf"))
        logger.info("resumed from %s: epoch=%d step=%d best_val_q=%.4f",
                    path, self.epoch, self.global_step, self.best_val_q_loss)

        # Defensive guard against silent LR-schedule reinterpretation on resume.
        # cosine_warmup_lr is parameterized by total_steps, which is recomputed
        # from cfg.epochs in __init__. If the user resumes with a different
        # --epochs than the original, the cosine schedule shifts and the LR
        # jumps at the resume point with NO log signal. Warn loudly.
        saved_cfg = payload.get("trainer_cfg") or {}
        saved_epochs = saved_cfg.get("epochs")
        if saved_epochs is not None and saved_epochs != self.cfg.epochs:
            saved_total = self.steps_per_epoch * saved_epochs
            # Use the SAVED min_lr_factor when reconstructing the old schedule
            # (default 0.0 for pre-floor checkpoints which decayed to 0). The new
            # schedule uses the live cfg.min_lr_factor. Without this, the warning
            # silently uses the new floor (0.1) for both, masking the actual
            # schedule change for users resuming from a pre-floor checkpoint.
            old_min_lr = saved_cfg.get("min_lr_factor", 0.0)
            old_lr = cosine_warmup_lr(
                self.global_step, base_lr=self.cfg.lr,
                warmup_steps=self.cfg.warmup_steps, total_steps=saved_total,
                min_lr_factor=old_min_lr,
            )
            new_lr = cosine_warmup_lr(
                self.global_step, base_lr=self.cfg.lr,
                warmup_steps=self.cfg.warmup_steps, total_steps=self.total_steps,
                min_lr_factor=self.cfg.min_lr_factor,
            )
            logger.warning(
                "EPOCHS MISMATCH ON RESUME: checkpoint trained with epochs=%d, "
                "current cfg.epochs=%d. Cosine LR schedule reinterpreted: at "
                "global_step=%d, lr was %.2e under old schedule, now %.2e under "
                "new schedule (delta = %+.2e). If unintended, restart with the "
                "original --epochs to preserve the schedule.",
                saved_epochs, self.cfg.epochs, self.global_step, old_lr, new_lr,
                new_lr - old_lr,
            )

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #

    def fit(self) -> dict[str, float]:
        """Train for ``cfg.epochs`` epochs, evaluating end-of-epoch (or every
        ``eval_every_steps`` if > 0). Returns the final eval metrics."""
        try:
            final_metrics: dict[str, float] = {}
            for epoch_idx in range(self.epoch, self.cfg.epochs):
                self.epoch = epoch_idx
                t0 = time.time()
                step_metrics_running = {"q_loss": 0.0, "v_loss": 0.0, "lr": 0.0, "grad_norm": 0.0}
                n_steps_this_epoch = 0

                for batch in self.train_loader:
                    m = self.step(batch)
                    self._log_row({
                        "phase": "train", "epoch": epoch_idx, "step": self.global_step,
                        **m,
                    })
                    for k in step_metrics_running:
                        step_metrics_running[k] += m[k]
                    n_steps_this_epoch += 1

                    if self.cfg.eval_every_steps > 0 and self.global_step % self.cfg.eval_every_steps == 0:
                        self._mid_epoch_eval(epoch_idx)

                # End-of-epoch eval
                avg = {k: v / max(1, n_steps_this_epoch) for k, v in step_metrics_running.items()}
                eval_metrics = self.evaluate()
                final_metrics = eval_metrics
                self._log_row({
                    "phase": "eval", "epoch": epoch_idx, "step": self.global_step,
                    **{f"train_avg_{k}": v for k, v in avg.items()},
                    **eval_metrics,
                    "elapsed_s": time.time() - t0,
                })
                self._log_eval_summary(epoch_idx, avg, eval_metrics, time.time() - t0)

                # Mark this epoch as completed BEFORE saving so the persisted
                # `self.epoch` is the next epoch to run. Without this bump,
                # resume's `range(self.epoch, cfg.epochs)` re-runs the just-
                # completed epoch (re-trains data the model already saw and
                # advances the LR schedule past where it should be).
                self.epoch = epoch_idx + 1
                self.save_checkpoint(self.ckpt_latest)
                if (epoch_idx + 1) % self.cfg.checkpoint_every_epochs == 0:
                    self.save_checkpoint(self.run_dir / f"checkpoint_epoch_{epoch_idx}.pt")
                if eval_metrics["q_loss"] < self.best_val_q_loss:
                    self.best_val_q_loss = eval_metrics["q_loss"]
                    self.save_checkpoint(self.ckpt_best)
                    logger.info("new best val q_loss=%.4f at epoch %d", self.best_val_q_loss, epoch_idx)

            return final_metrics
        finally:
            self._close_csv()

    def _mid_epoch_eval(self, epoch_idx: int) -> None:
        eval_metrics = self.evaluate()
        self._log_row({
            "phase": "eval_intra", "epoch": epoch_idx, "step": self.global_step,
            **eval_metrics,
        })
        logger.info("step %d intra-epoch eval | q_loss=%.4f v_loss=%.4f",
                    self.global_step, eval_metrics["q_loss"], eval_metrics["v_loss"])

    def _log_eval_summary(self, epoch_idx: int, train_avg: dict, eval_m: dict, elapsed_s: float) -> None:
        line = (f"epoch {epoch_idx:>3} | step {self.global_step:>6} | "
                f"train q={train_avg['q_loss']:.4f} v={train_avg['v_loss']:.4f} | "
                f"val q={eval_m['q_loss']:.4f} v={eval_m['v_loss']:.4f}")
        if "q_loss_blind" in eval_m:
            line += (f" | blind q={eval_m['q_loss_blind']:.4f} "
                     f"gap={eval_m['q_loss_blind_gap']:+.4f}")
        line += f" | {elapsed_s:.1f}s"
        logger.info(line)
