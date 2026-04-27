"""Phase 9 Part A — behavioral / distributional eval CLI.

Loads a trained checkpoint, runs the policy on val (or test), computes
behavioral metrics + segment breakdowns + pitcher-blind variants, and writes
a markdown report under the run directory.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.configs import build_qtransformer, load_preset  # noqa: E402
from src.dataset import PitchPADataset, load_vocab_sizes, pa_collate  # noqa: E402
from src.ope_metrics import evaluate_behavioral, segment_breakdowns  # noqa: E402
from src.report import write_behavioral_report  # noqa: E402

TOKENS_DIR = REPO_ROOT / "data" / "tokens"
RUNS_DIR = REPO_ROOT / "data" / "runs"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 Part A — behavioral evaluation.")
    parser.add_argument("--run-name", required=True, help="Existing run directory under data/runs/.")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt",
                        help="Checkpoint file inside the run dir (default: checkpoint_best.pt).")
    parser.add_argument("--preset", choices=("smoke", "v1"), default="v1",
                        help="Architecture preset to instantiate (must match the trained checkpoint).")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--tokens-dir", type=Path, default=TOKENS_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pitcher-blind", action="store_true")
    parser.add_argument("--no-segments", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    run_dir = RUNS_DIR / args.run_name
    ckpt = run_dir / args.checkpoint
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found at {ckpt}", file=sys.stderr)
        sys.exit(2)

    device = _device()
    print(f"device: {device}")

    # Load vocab + tokens + arsenal
    vocab_sizes = load_vocab_sizes(args.tokens_dir / "vocab.json")
    ds = PitchPADataset(
        args.tokens_dir / f"{args.split}.parquet",
        args.tokens_dir / "batter_profile.parquet",
        pitcher_arsenal_path=args.tokens_dir / "pitcher_arsenal.parquet",
        n_pitch_types=vocab_sizes["pitch_type"],
    )
    print(f"loaded {len(ds)} PAs from {args.split}")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=pa_collate,
    )

    # Build model from preset and load checkpoint
    print(f"building model (preset={args.preset})...")
    model = build_qtransformer(args.tokens_dir, preset=args.preset)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # Run metrics
    print("running aggregate metrics + (optionally) pitcher-blind variant...")
    t0 = time.time()
    metrics = evaluate_behavioral(
        model, loader, device=device,
        include_pitcher_blind=not args.no_pitcher_blind,
    )
    print(f"  aggregate done in {time.time() - t0:.1f}s")

    segments = {}
    if not args.no_segments:
        print("running segment breakdowns...")
        t0 = time.time()
        segments = segment_breakdowns(model, loader, device=device)
        print(f"  segments done in {time.time() - t0:.1f}s")

    out_path = run_dir / f"behavioral_report_{args.split}.md"
    write_behavioral_report(
        metrics, segments,
        out_path=out_path,
        run_name=args.run_name, split_name=args.split,
        vocab_path=args.tokens_dir / "vocab.json",
    )

    # Console summary
    print("=" * 60)
    print(f"Behavioral metrics summary  (n={metrics.n_pitches:,} pitches on {args.split})")
    print(f"  pitch_type top-1:        {metrics.pitch_type_top1*100:.2f}%")
    print(f"  pitch_type top-3:        {metrics.pitch_type_top3*100:.2f}%")
    print(f"  coarse-zone top-1:       {metrics.coarse_zone_top1*100:.2f}%")
    print(f"  spatial median (in):     {metrics.spatial_distance_median_ft*12:.1f}")
    print(f"  within 6in:              {metrics.spatial_within_6in_frac*100:.2f}%")
    print(f"  KL(learned ∥ behavior):  {metrics.pitch_type_kl_learned_to_behavior:.4f} nats")
    if metrics.pitch_type_top1_blind is not None:
        gap = metrics.pitch_type_top1 - metrics.pitch_type_top1_blind
        print(f"  pitcher-blind top-1:     {metrics.pitch_type_top1_blind*100:.2f}%  "
              f"(personal lift: {gap*100:+.2f} pp)")
    print(f"\nreport written: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
