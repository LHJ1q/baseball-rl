"""Phase 8 — IQL training run.

Runs on the user's Blackwell RTX Pro 4500 (default). Macbook smoke is
``--smoke-train``, which uses tiny config + tiny subset to validate the loop
end-to-end in seconds.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.configs import build_qtransformer, load_preset  # noqa: E402
from src.dataset import PitchPADataset, load_vocab_sizes  # noqa: E402
from src.trainer import Trainer, TrainerConfig  # noqa: E402

TOKENS_DIR = REPO_ROOT / "data" / "tokens"
RUNS_DIR = REPO_ROOT / "data" / "runs"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_run_dir(args) -> tuple[Path, bool]:
    """Returns ``(run_dir, should_resume)``. Refuses to overwrite an existing
    run dir unless ``--resume`` is set; auto-resumes if ``--resume`` is set
    and a checkpoint exists."""
    name = args.run_name or f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / name
    ckpt = run_dir / "checkpoint_latest.pt"

    if run_dir.exists():
        if args.no_resume:
            print(
                f"ERROR: run dir already exists at {run_dir} and --no-resume is set. "
                "Pick a new --run-name or remove the directory.",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.resume and ckpt.exists():
            return run_dir, True
        if not args.resume:
            print(
                f"ERROR: run dir already exists at {run_dir}. "
                "Pass --resume to continue from the latest checkpoint, or pick a new --run-name.",
                file=sys.stderr,
            )
            sys.exit(2)
        # --resume but no checkpoint → start fresh in existing dir
        return run_dir, False

    return run_dir, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 IQL training run.")
    parser.add_argument("--preset", choices=("smoke", "v1"), default="v1")
    parser.add_argument("--tokens-dir", type=Path, default=TOKENS_DIR)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run directory under data/runs/. Auto-generated timestamp if omitted.")
    parser.add_argument("--resume", action="store_true",
                        help="If run dir exists, continue from checkpoint_latest.pt.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Refuse to use an existing run dir even if it exists.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--pitcher-dropout", type=float, default=None)
    parser.add_argument("--no-bf16", action="store_true",
                        help="Force fp32 even on CUDA (default uses BF16 on CUDA).")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (default: enabled on CUDA, ~1.2-1.5x speedup; "
                             "auto-skipped on Macbook MPS/CPU).")
    parser.add_argument("--no-pitcher-blind-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--smoke-train", action="store_true",
                        help="Macbook smoke-train: tiny preset, tiny dataset slice, few epochs, validates the loop.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    device = _device()
    print(f"device: {device}")

    encoder_cfg, q_cfg = load_preset("smoke" if args.smoke_train else args.preset)

    trainer_cfg = TrainerConfig()
    if args.smoke_train:
        # Tiny config for end-to-end validation on Macbook in seconds.
        trainer_cfg = TrainerConfig(
            lr=1e-3, warmup_steps=10, batch_size=8, num_workers=0,
            epochs=2, checkpoint_every_epochs=1, bf16=False,
            pin_memory=False, include_pitcher_blind_eval=True,
        )

    # Apply CLI overrides (any explicit value wins).
    overrides = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "warmup_steps": args.warmup_steps, "num_workers": args.num_workers,
        "eval_every_steps": args.eval_every_steps,
        "checkpoint_every_epochs": args.checkpoint_every_epochs,
        "gamma": args.gamma, "tau": args.tau, "pitcher_dropout": args.pitcher_dropout,
        "seed": args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(trainer_cfg, k, v)
    if args.no_bf16:
        trainer_cfg.bf16 = False
    if args.no_compile:
        trainer_cfg.compile = False
    if args.no_pitcher_blind_eval:
        trainer_cfg.include_pitcher_blind_eval = False

    # Resolve run dir + resume
    run_dir, should_resume = _resolve_run_dir(args)

    print("=" * 70)
    print(f"Run: {run_dir.name}")
    if should_resume:
        print(f"Status: RESUMING from {run_dir / 'checkpoint_latest.pt'}")
    else:
        print("Status: STARTING FRESH")
    print(f"Preset: {'smoke' if args.smoke_train else args.preset}  |  device: {device}")
    print("=" * 70)

    # Build model
    print("loading vocab + feature stats...")
    vocab_sizes = load_vocab_sizes(args.tokens_dir / "vocab.json")
    model = build_qtransformer(args.tokens_dir, preset=(encoder_cfg, q_cfg))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # Datasets
    print("loading datasets...")
    train_ds = PitchPADataset(
        args.tokens_dir / "train.parquet",
        args.tokens_dir / "batter_profile.parquet",
        pitcher_arsenal_path=args.tokens_dir / "pitcher_arsenal.parquet",
        n_pitch_types=vocab_sizes["pitch_type"],
    )
    val_ds = PitchPADataset(
        args.tokens_dir / "val.parquet",
        args.tokens_dir / "batter_profile.parquet",
        pitcher_arsenal_path=args.tokens_dir / "pitcher_arsenal.parquet",
        n_pitch_types=vocab_sizes["pitch_type"],
    )
    print(f"train PAs: {len(train_ds)}  val PAs: {len(val_ds)}")

    if args.smoke_train:
        train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(32, len(val_ds)))))
        print(f"SMOKE: trimmed to {len(train_ds)} train / {len(val_ds)} val PAs")

    encoder_q_payload = {"encoder": dataclasses.asdict(encoder_cfg), "qtransformer": dataclasses.asdict(q_cfg)}
    trainer = Trainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=trainer_cfg,
        device=device,
        run_dir=run_dir,
        encoder_q_config_payload=encoder_q_payload,
    )

    if should_resume:
        trainer.load_checkpoint(run_dir / "checkpoint_latest.pt")

    final = trainer.fit()
    print("=" * 70)
    print(f"DONE. final eval metrics: {final}")
    print(f"run dir: {run_dir}")
    print(f"metrics CSV: {run_dir / 'metrics.csv'}")
    print(f"best checkpoint: {run_dir / 'checkpoint_best.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
