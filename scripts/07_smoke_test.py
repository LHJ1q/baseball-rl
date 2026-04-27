"""Strongest end-to-end smoke test for phase 6+7.

Runs on Macbook CPU/MPS in seconds with the default ``smoke`` preset. Pass
``--preset v1`` to validate the production preset on a single batch before
shipping to the GPU.

Ten checks. Exits non-zero if any fail.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.configs import build_qtransformer, load_preset  # noqa: E402
from src.dataset import PitchPADataset, pa_collate  # noqa: E402
from src.qtransformer import (  # noqa: E402
    build_repertoire_mask,
    iql_losses,
    shift_v_for_next_state,
)

TOKENS_DIR = REPO_ROOT / "data" / "tokens"

N_PA_FOR_OVERFIT = 16
N_PA_BATCH = 4


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    detail: str = ""


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_forward_shapes(model, batch) -> CheckResult:
    out = model(batch)
    B, T = batch.valid_mask.shape
    n_x, n_z = model.cfg.n_x_bins, model.cfg.n_z_bins
    assert out["q_chosen"].shape == (B, T), out["q_chosen"].shape
    assert out["v"].shape == (B, T), out["v"].shape
    assert out["q_type_logits"].dim() == 3 and out["q_type_logits"].shape[:2] == (B, T)
    assert out["q_x_logits"].shape == (B, T, n_x)
    assert out["q_z_logits"].shape == (B, T, n_z)
    return CheckResult(
        "1. forward shapes correct",
        "PASS",
        f"B={B} T={T}  q_chosen={tuple(out['q_chosen'].shape)} v={tuple(out['v'].shape)}",
    )


def check_no_nan_inf(model, batch) -> CheckResult:
    out = model(batch)
    bad = []
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            if torch.isnan(v).any():
                bad.append(f"{k} has NaN")
            if torch.isinf(v).any():
                bad.append(f"{k} has Inf")
    return CheckResult(
        "2. no NaN/Inf in any output",
        "PASS" if not bad else "FAIL",
        "all finite" if not bad else "; ".join(bad),
    )


def check_causal_mask(model, batch) -> CheckResult:
    """Perturb post-action features at the last pitch; outputs at *earlier*
    pre-positions must be unchanged."""
    model.eval()
    with torch.no_grad():
        out0 = model(batch)
    # Mutate post_cont at last valid position of the first PA only.
    batch2 = batch
    perturbed_post = batch.post_cont.clone()
    last_t = int(batch.pa_lengths[0].item()) - 1
    perturbed_post[0, last_t] += 100.0
    batch2 = type(batch)(
        pre_cat=batch.pre_cat, pre_cont=batch.pre_cont, profile=batch.profile,
        post_cat=batch.post_cat, post_cont=perturbed_post,
        reward=batch.reward, is_terminal=batch.is_terminal,
        pa_lengths=batch.pa_lengths, valid_mask=batch.valid_mask,
        arsenal_per_type=batch.arsenal_per_type, batter_per_type=batch.batter_per_type,
    )
    with torch.no_grad():
        out1 = model(batch2)

    # h_pre at positions 0..last_t should be unchanged (they only see pre+post of
    # earlier pitches; perturbing pitch last_t's post should not affect them).
    diff_at_earlier = (out0["h_pre"][0, :last_t] - out1["h_pre"][0, :last_t]).abs().max()
    diff_at_last = (out0["h_pre"][0, last_t] - out1["h_pre"][0, last_t]).abs().max()
    ok = bool(diff_at_earlier.item() < 1e-5)
    return CheckResult(
        "3. causal mask: earlier pre-positions unaffected by later post perturbation",
        "PASS" if ok else "FAIL",
        f"max|Δh_pre| earlier={diff_at_earlier:.2e}  at_last={diff_at_last:.2e}  (earlier should be ~0)",
    )


def check_batch_independence(model, batch) -> CheckResult:
    """Different PAs in the same batch must not influence each other."""
    model.eval()
    with torch.no_grad():
        out_full = model(batch)

    # Run each PA in its own batch-of-1 and compare.
    max_diff = 0.0
    for b in range(batch.valid_mask.shape[0]):
        T_b = int(batch.pa_lengths[b].item())
        single = type(batch)(
            pre_cat={k: v[b:b+1] for k, v in batch.pre_cat.items()},
            pre_cont=batch.pre_cont[b:b+1],
            profile=batch.profile[b:b+1],
            post_cat={k: v[b:b+1] for k, v in batch.post_cat.items()},
            post_cont=batch.post_cont[b:b+1],
            reward=batch.reward[b:b+1],
            is_terminal=batch.is_terminal[b:b+1],
            pa_lengths=batch.pa_lengths[b:b+1],
            valid_mask=batch.valid_mask[b:b+1],
            arsenal_per_type=batch.arsenal_per_type[b:b+1],
            batter_per_type=batch.batter_per_type[b:b+1],
        )
        with torch.no_grad():
            out_solo = model(single)
        diff = (out_full["q_chosen"][b, :T_b] - out_solo["q_chosen"][0, :T_b]).abs().max()
        max_diff = max(max_diff, float(diff.item()))
    ok = max_diff < 1e-4
    return CheckResult(
        "4. PA-batch independence (no cross-PA leak via padding)",
        "PASS" if ok else "FAIL",
        f"max |Δq_chosen| between batched and singleton runs = {max_diff:.2e}",
    )


def check_repertoire_mask(model, batch, vocab_sizes) -> CheckResult:
    """Synthetic arsenal: only allow pitch_type=1 for every pitcher. Argmax must
    pick pitch_type=1 everywhere."""
    n_pt = vocab_sizes["pitch_type"]
    pitcher_ids = batch.pre_cat["pitcher_id"]
    arsenal = {}
    for pid in pitcher_ids.unique().tolist():
        if pid == 0:
            continue
        arsenal[(int(pid), 1)] = 100  # only pitch_type 1 has count >= n_min
    mask = build_repertoire_mask(pitcher_ids, arsenal, n_pitch_types=n_pt, n_min=10)

    out = model.policy(batch, repertoire_mask=mask)
    chosen = out["pitch_type"][batch.valid_mask]

    # All chosen pitch types should be either 1 (in repertoire) or correspond
    # to UNK fallback (where mask was all-True). Pitchers != UNK_ID=0 should
    # all pick pitch_type=1.
    non_unk = pitcher_ids[batch.valid_mask] != 0
    chosen_for_known = chosen[non_unk]
    ok = bool((chosen_for_known == 1).all().item())
    return CheckResult(
        "5. repertoire mask restricts argmax to in-repertoire pitch types",
        "PASS" if ok else "FAIL",
        f"chosen_for_known_pitchers unique = {chosen_for_known.unique().tolist()} (expected [1])",
    )


def check_no_nulls_in_batch(batch) -> CheckResult:
    """Sanity: batch tensors shouldn't have NaN — the dataloader should fill all
    positions, padded with zeros."""
    bad = {}
    for f, t in batch.pre_cat.items():
        if (t < 0).any():
            bad[f"pre_cat__{f}"] = "negative ID"
    if torch.isnan(batch.pre_cont).any():
        bad["pre_cont"] = "NaN"
    if torch.isnan(batch.profile).any():
        bad["profile"] = "NaN"
    if torch.isnan(batch.post_cont).any():
        bad["post_cont"] = "NaN"
    return CheckResult(
        "6. no NaN/negative in batched tensors",
        "PASS" if not bad else "FAIL",
        "all clean" if not bad else f"violations: {bad}",
    )


def check_iql_loss_finite(model, batch) -> CheckResult:
    out = model(batch)
    v_next = shift_v_for_next_state(out["v"], batch.valid_mask)
    losses = iql_losses(
        q_chosen=out["q_chosen"],
        v_current=out["v"],
        v_next=v_next,
        reward=batch.reward,
        is_terminal=batch.is_terminal,
        valid_mask=batch.valid_mask,
    )
    finite = all(torch.isfinite(losses[k]).item() for k in ("q_loss", "v_loss"))
    return CheckResult(
        "7. IQL loss expression is finite",
        "PASS" if finite else "FAIL",
        f"q_loss={losses['q_loss'].item():.4f}  v_loss={losses['v_loss'].item():.4f}",
    )


def check_overfit_single_batch(model, batch, n_steps: int = 50) -> CheckResult:
    """Loss on a fixed batch should drop substantially with a small Adam run."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        out = model(batch)
        v_next = shift_v_for_next_state(out["v"], batch.valid_mask)
        loss_dict = iql_losses(
            out["q_chosen"], out["v"], v_next, batch.reward,
            batch.is_terminal, batch.valid_mask,
        )
        total = loss_dict["q_loss"] + loss_dict["v_loss"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(total.item())
    drop = losses[0] - losses[-1]
    rel_drop = drop / max(abs(losses[0]), 1e-6)
    ok = rel_drop > 0.5  # >50% relative decrease
    return CheckResult(
        "8. overfit single batch: loss drops > 50% in 50 steps",
        "PASS" if ok else "FAIL",
        f"loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}  rel_drop={rel_drop*100:.1f}%",
    )


def check_argmax_determinism(model, batch) -> CheckResult:
    model.eval()
    with torch.no_grad():
        a = model.policy(batch)
        b = model.policy(batch)
    same = (
        torch.equal(a["pitch_type"], b["pitch_type"])
        and torch.equal(a["x_bin"], b["x_bin"])
        and torch.equal(a["z_bin"], b["z_bin"])
    )
    return CheckResult(
        "9. argmax policy is deterministic",
        "PASS" if same else "FAIL",
        "two identical calls produce identical actions" if same else "outputs differ across calls",
    )


def check_save_load_roundtrip(model, batch, preset_name) -> CheckResult:
    model.eval()
    with torch.no_grad():
        out_before = model(batch)["q_chosen"].clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)

    model2 = build_qtransformer(TOKENS_DIR, preset=preset_name)
    model2.load_state_dict(torch.load(path, weights_only=True))
    model2.eval()
    with torch.no_grad():
        out_after = model2(batch)["q_chosen"]
    Path(path).unlink(missing_ok=True)

    diff = (out_before - out_after).abs().max().item()
    ok = diff < 1e-5
    return CheckResult(
        "10. save/load round-trip preserves outputs",
        "PASS" if ok else "FAIL",
        f"max|Δq_chosen| = {diff:.2e}",
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def format_report(results: list[CheckResult]) -> str:
    lines = ["=" * 80, "Phase 6+7 smoke test report (tiny config)", "=" * 80]
    for r in results:
        lines.append(f"[{r.status:<4}] {r.name}: {r.summary}")
        if r.detail:
            for d in r.detail.splitlines():
                if d:
                    lines.append(f"        {d}")
    failed = sum(1 for r in results if r.status == "FAIL")
    warned = sum(1 for r in results if r.status == "WARN")
    lines.append("=" * 80)
    lines.append(
        f"Summary: {len(results) - failed - warned} pass / {warned} warn / {failed} fail"
    )
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6+7 smoke test.")
    parser.add_argument(
        "--preset", choices=("smoke", "v1"), default="smoke",
        help="Which preset to instantiate. 'smoke' is tiny (CPU-friendly); 'v1' is the full GPU preset.",
    )
    parser.add_argument("--n-pa-overfit", type=int, default=N_PA_FOR_OVERFIT)
    parser.add_argument("--n-pa-batch", type=int, default=N_PA_BATCH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from src.dataset import load_vocab_sizes
    vocab_sizes = load_vocab_sizes(TOKENS_DIR / "vocab.json")
    ds = PitchPADataset(
        TOKENS_DIR / "train.parquet",
        TOKENS_DIR / "batter_profile.parquet",
        pitcher_arsenal_path=TOKENS_DIR / "pitcher_arsenal.parquet",
        n_pitch_types=vocab_sizes["pitch_type"],
    )
    print(f"preset={args.preset}  loaded {len(ds)} PAs")

    items = [ds[i] for i in range(args.n_pa_overfit)]
    big_batch = pa_collate(items)
    small_batch = pa_collate(items[: args.n_pa_batch])

    torch.manual_seed(0)
    model = build_qtransformer(TOKENS_DIR, preset=args.preset)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    results = [
        check_forward_shapes(model, small_batch),
        check_no_nan_inf(model, small_batch),
        check_causal_mask(model, small_batch),
        check_batch_independence(model, small_batch),
        check_repertoire_mask(model, small_batch, vocab_sizes),
        check_no_nulls_in_batch(small_batch),
        check_iql_loss_finite(model, small_batch),
        check_overfit_single_batch(model, big_batch, n_steps=50),
        check_argmax_determinism(model, small_batch),
        check_save_load_roundtrip(model, small_batch, args.preset),
    ]
    print(format_report(results))
    if any(r.status == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
