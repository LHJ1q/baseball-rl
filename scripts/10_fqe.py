"""Phase 9 Part B — train the FQE Q-network and estimate the policy's value.

Loads a trained policy checkpoint, builds a fresh ``QTransformer`` initialized
from the policy weights, trains it with the FQE on-policy SARSA target on the
same data, then runs :func:`src.fqe.estimate_pa_values` on val and test to
output the policy's estimated per-PA expected return.

Macbook smoke: ``--smoke-train`` runs ~2 epochs on a tiny subset (~1 minute).
GPU production: full epochs on the full splits (~1-2 hours).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.configs import build_qtransformer, load_preset  # noqa: E402
from src.dataset import PitchPADataset, load_vocab_sizes, pa_collate  # noqa: E402
from src.fqe import FQETrainer, FQETrainerConfig, estimate_pa_values  # noqa: E402

TOKENS_DIR = REPO_ROOT / "data" / "tokens"
RUNS_DIR = REPO_ROOT / "data" / "runs"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 Part B — train FQE + estimate policy value.")
    parser.add_argument("--run-name", required=True,
                        help="Existing IQL run dir under data/runs/. FQE outputs go in the same dir.")
    parser.add_argument("--policy-checkpoint", default="checkpoint_best.pt",
                        help="Filename inside the run dir to use as π_learned.")
    parser.add_argument("--preset", choices=("smoke", "v1"), default="v1",
                        help="Architecture preset (must match the policy checkpoint).")
    parser.add_argument("--tokens-dir", type=Path, default=TOKENS_DIR)

    # FQE training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--smoke-train", action="store_true",
                        help="Tiny dataset slice + 2 epochs; validates the loop end-to-end on Macbook.")
    parser.add_argument("--no-init-from-policy", action="store_true",
                        help="Don't initialize FQE weights from the policy checkpoint (start fresh).")
    parser.add_argument(
        "--repertoire-mask-min-count", type=int, default=0,
        help="If >0, FQE evaluates π_learned constrained to per-pitcher repertoire types with "
             "count >= N. Default 0 = mask disabled (matches default deployment policy).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    run_dir = RUNS_DIR / args.run_name
    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        sys.exit(2)
    policy_ckpt = run_dir / args.policy_checkpoint
    if not policy_ckpt.exists():
        print(f"ERROR: policy checkpoint not found: {policy_ckpt}", file=sys.stderr)
        sys.exit(2)

    device = _device()
    print(f"device: {device}")

    encoder_cfg, q_cfg = load_preset("smoke" if args.smoke_train else args.preset)
    fqe_cfg = FQETrainerConfig()
    if args.smoke_train:
        fqe_cfg = FQETrainerConfig(
            lr=1e-3, warmup_steps=10, batch_size=8, num_workers=0,
            epochs=2, checkpoint_every_epochs=1, bf16=False, pin_memory=False,
        )

    overrides = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "num_workers": args.num_workers, "gamma": args.gamma, "seed": args.seed,
        "repertoire_mask_min_count": args.repertoire_mask_min_count,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(fqe_cfg, k, v)
    if args.no_bf16:
        fqe_cfg.bf16 = False

    print("=" * 70)
    print(f"FQE run for: {args.run_name}")
    print(f"Policy checkpoint: {policy_ckpt.name}")
    print(f"Preset: {'smoke' if args.smoke_train else args.preset}  |  device: {device}")
    print("=" * 70)

    # Vocab + datasets
    vocab_sizes = load_vocab_sizes(args.tokens_dir / "vocab.json")
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
    test_ds = PitchPADataset(
        args.tokens_dir / "test.parquet",
        args.tokens_dir / "batter_profile.parquet",
        pitcher_arsenal_path=args.tokens_dir / "pitcher_arsenal.parquet",
        n_pitch_types=vocab_sizes["pitch_type"],
    )
    if args.smoke_train:
        train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(32, len(val_ds)))))
        test_ds = Subset(test_ds, list(range(min(16, len(test_ds)))))
        print(f"SMOKE: trimmed to {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test PAs")

    # Build π_learned (frozen) from checkpoint
    print("loading policy model...")
    policy_model = build_qtransformer(args.tokens_dir, preset=(encoder_cfg, q_cfg))
    payload = torch.load(policy_ckpt, map_location=device, weights_only=False)
    policy_model.load_state_dict(payload["model"])

    # Build FQE model — same architecture, optionally init from policy weights
    print("building FQE model...")
    fqe_model = build_qtransformer(args.tokens_dir, preset=(encoder_cfg, q_cfg))
    if not args.no_init_from_policy:
        fqe_model.load_state_dict(payload["model"])
        print("  initialized from policy checkpoint")
    n_params = sum(p.numel() for p in fqe_model.parameters())
    print(f"FQE params: {n_params:,}")

    # Train
    trainer = FQETrainer(fqe_model, policy_model, train_ds, val_ds, fqe_cfg, device, run_dir)
    print(f"starting FQE training: {fqe_cfg.epochs} epochs, batch={fqe_cfg.batch_size}")
    t0 = time.time()
    final_eval = trainer.fit()
    print(f"FQE training done in {(time.time() - t0) / 60:.1f} min")
    print(f"  final val: {final_eval}")

    # Per-PA value estimates on val + test
    print("estimating per-PA values on val + test...")
    val_loader = DataLoader(val_ds, batch_size=fqe_cfg.batch_size, shuffle=False,
                            num_workers=fqe_cfg.num_workers, collate_fn=pa_collate)
    test_loader = DataLoader(test_ds, batch_size=fqe_cfg.batch_size, shuffle=False,
                             num_workers=fqe_cfg.num_workers, collate_fn=pa_collate)
    val_v = estimate_pa_values(
        fqe_model, policy_model, val_loader, device=device,
        repertoire_mask_min_count=args.repertoire_mask_min_count,
    )
    test_v = estimate_pa_values(
        fqe_model, policy_model, test_loader, device=device,
        repertoire_mask_min_count=args.repertoire_mask_min_count,
    )

    print("=" * 70)
    print("FQE per-PA value estimates")
    print(f"  val:   learned={val_v['learned_per_pa']:+.4f}  behavior={val_v['behavior_per_pa']:+.4f}  "
          f"advantage={val_v['advantage']:+.4f}  (n_pa={val_v['n_pa']})")
    print(f"  test:  learned={test_v['learned_per_pa']:+.4f}  behavior={test_v['behavior_per_pa']:+.4f}  "
          f"advantage={test_v['advantage']:+.4f}  (n_pa={test_v['n_pa']})")
    print("=" * 70)
    print(f"\nFQE checkpoint: {run_dir / 'fqe_checkpoint_latest.pt'}")
    print(f"FQE metrics CSV: {run_dir / 'fqe_metrics.csv'}")


if __name__ == "__main__":
    main()
