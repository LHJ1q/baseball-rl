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

    # Reproducibility
    seed: int = 0

    # Eval cadence options
    include_pitcher_blind_eval: bool = True


# --------------------------------------------------------------------------- #
# Cosine LR with linear warmup
# --------------------------------------------------------------------------- #


def cosine_warmup_lr(step: int, *, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup over ``warmup_steps`` then cosine decay to 0 over the
    remaining steps. Always returns a positive scalar."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


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

        self.model.to(self.device)

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            collate_fn=pa_collate,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and self.device.type == "cuda",
            collate_fn=pa_collate,
        )

        self.steps_per_epoch = max(1, len(self.train_loader))
        self.total_steps = self.steps_per_epoch * cfg.epochs

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

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
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def step(self, batch: PABatch) -> dict[str, float]:
        """One optimizer step. Returns ``{q_loss, v_loss, lr, grad_norm}``."""
        self.model.train()
        batch = batch.to(self.device)
        lr = self._set_lr()
        self.optimizer.zero_grad(set_to_none=True)

        autocast_dtype = torch.bfloat16 if self._use_bf16 else None
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self._use_bf16):
            out = self.model(batch)
            v_next = shift_v_for_next_state(out["v"], batch.valid_mask)
            losses = iql_losses(
                q_chosen=out["q_chosen"],
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
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_q_loss": self.best_val_q_loss,
            "trainer_cfg": asdict(self.cfg),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = payload["global_step"]
        self.epoch = payload["epoch"]
        self.best_val_q_loss = payload.get("best_val_q_loss", float("inf"))
        logger.info("resumed from %s: epoch=%d step=%d best_val_q=%.4f",
                    path, self.epoch, self.global_step, self.best_val_q_loss)

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

                # Checkpointing
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
