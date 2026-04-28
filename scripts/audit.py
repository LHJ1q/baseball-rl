"""End-to-end framework audit for the offline-RL pipeline.

Walks the entire stack — dataset → encoders → Q-Transformer → IQL loss → FQE
loss → policy → save/load — and verifies invariants that, if violated, would
either crash training or (worse) silently degrade it.

Each check returns a CheckResult with PASS / WARN / FAIL. Exit code is non-zero
if any FAIL.

Designed to run on Macbook CPU/MPS with a TINY config in seconds. Same code
paths as the v1 preset, just smaller dimensions.
"""
from __future__ import annotations

import logging
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.configs import build_qtransformer, load_preset  # noqa: E402
from src.dataset import (  # noqa: E402
    BATTER_PROFILE_OVERALL_FIELDS,
    POST_ACTION_CATEGORICAL_FIELDS,
    POST_ACTION_CONTINUOUS_FIELDS,
    PABatch,
    PRE_ACTION_CATEGORICAL_FIELDS,
    PRE_ACTION_CONTINUOUS_FIELDS,
    PitchPADataset,
    load_vocab_sizes,
    pa_collate,
)
from src.encoder import EncoderConfig  # noqa: E402
from src.eval import _zeroed_pitcher_embedding  # noqa: E402
from src.fqe import FQETrainer, FQETrainerConfig, fqe_loss  # noqa: E402
from src.ope_metrics import N_X_BINS as METRICS_N_X, N_Z_BINS as METRICS_N_Z, predict_batch  # noqa: E402
from src.qtransformer import (  # noqa: E402
    QTransformer,
    QTransformerConfig,
    build_repertoire_mask,
    iql_losses,
    shift_v_for_next_state,
)
from src.tokenize import (  # noqa: E402
    ARSENAL_HEAD_FIELDS,
    BATTER_PER_TYPE_HEAD_FIELDS,
    N_X_BINS as TOK_N_X,
    N_Z_BINS as TOK_N_Z,
    UNK_ID,
)

TOKENS_DIR = REPO_ROOT / "data" / "tokens"


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    summary: str
    detail: str = ""


# --------------------------------------------------------------------------- #
# Test fixtures (synthetic, no disk dependency for most checks)
# --------------------------------------------------------------------------- #


VOCAB_SIZES = {
    "pitch_type": 6, "description": 4,
    "p_throws": 2, "stand": 2, "inning_topbot": 2,
    "batter": 20, "pitcher": 20,
}
N_PT = VOCAB_SIZES["pitch_type"]


def _make_batch(B=4, T=3, *, valid_lengths=None) -> PABatch:
    if valid_lengths is None:
        valid_lengths = [T] * B
    pre_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in PRE_ACTION_CATEGORICAL_FIELDS}
    pre_cat["pitcher_id"] = torch.tensor([1, 2, 3, 4]).unsqueeze(1).expand(B, T).contiguous()
    pre_cat["batter_id"] = torch.tensor([5, 6, 7, 8]).unsqueeze(1).expand(B, T).contiguous()
    pre_cont = torch.randn(B, T, len(PRE_ACTION_CONTINUOUS_FIELDS))
    profile = torch.randn(B, T, len(BATTER_PROFILE_OVERALL_FIELDS))
    post_cat = {f: torch.zeros(B, T, dtype=torch.int64) for f in POST_ACTION_CATEGORICAL_FIELDS}
    post_cat["pitch_type_id"] = torch.randint(0, N_PT, (B, T))
    post_cont = torch.randn(B, T, len(POST_ACTION_CONTINUOUS_FIELDS))
    reward = torch.randn(B, T)
    is_terminal = torch.zeros(B, T, dtype=torch.bool)
    valid_mask = torch.zeros(B, T, dtype=torch.bool)
    pa_lengths = torch.tensor(valid_lengths, dtype=torch.int64)
    for b, L in enumerate(valid_lengths):
        valid_mask[b, :L] = True
        is_terminal[b, L - 1] = True
    arsenal = torch.randn(B, T, N_PT, 14)
    batter_pt = torch.randn(B, T, N_PT, 4)
    return PABatch(pre_cat, pre_cont, profile, post_cat, post_cont, reward, is_terminal,
                   pa_lengths, valid_mask, arsenal, batter_pt)


def _tiny_model() -> QTransformer:
    enc, q = load_preset("smoke")
    return QTransformer(VOCAB_SIZES, cfg=q, encoder_cfg=enc)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_action_space_constants_consistent() -> CheckResult:
    """Every module that has its own copy of N_X_BINS / N_Z_BINS should agree."""
    if TOK_N_X != METRICS_N_X or TOK_N_Z != METRICS_N_Z:
        return CheckResult(
            "1. action-space constants consistent across modules", "FAIL",
            f"tokenize x={TOK_N_X} z={TOK_N_Z} vs ope_metrics x={METRICS_N_X} z={METRICS_N_Z}",
        )
    return CheckResult(
        "1. action-space constants consistent across modules", "PASS",
        f"N_X_BINS={TOK_N_X}  N_Z_BINS={TOK_N_Z}",
    )


def check_pre_cont_field_index_alignment() -> CheckResult:
    """``ope_metrics.predict_batch`` reads pre_cont[..., k] for specific k. Verify
    those indices match the documented field positions."""
    expected = {0: "balls", 1: "strikes", 5: "pitch_idx_in_pa"}
    bad = {}
    for idx, name in expected.items():
        if PRE_ACTION_CONTINUOUS_FIELDS[idx] != name:
            bad[idx] = (PRE_ACTION_CONTINUOUS_FIELDS[idx], name)
    if bad:
        return CheckResult(
            "2. ope_metrics positional pre_cont indices match field schema", "FAIL",
            f"mismatches: {bad}",
        )
    return CheckResult(
        "2. ope_metrics positional pre_cont indices match field schema", "PASS",
        f"balls@0, strikes@1, pitch_idx@5 — verified against PRE_ACTION_CONTINUOUS_FIELDS",
    )


def check_post_cont_field_index_alignment() -> CheckResult:
    """``ope_metrics.predict_batch`` reads post_cont[..., 0] for plate_x_mirrored
    and [..., 1] for plate_z."""
    expected = {0: "plate_x_mirrored", 1: "plate_z"}
    bad = {}
    for idx, name in expected.items():
        if POST_ACTION_CONTINUOUS_FIELDS[idx] != name:
            bad[idx] = (POST_ACTION_CONTINUOUS_FIELDS[idx], name)
    if bad:
        return CheckResult(
            "3. ope_metrics positional post_cont indices match field schema", "FAIL",
            f"mismatches: {bad}",
        )
    return CheckResult(
        "3. ope_metrics positional post_cont indices match field schema", "PASS",
        "plate_x_mirrored@0, plate_z@1 — verified",
    )


def check_arsenal_field_count_matches_tensor_shape() -> CheckResult:
    """Dataset builds an arsenal_per_type tensor of width len(ARSENAL_HEAD_FIELDS).
    Trainer's standardization buffer must agree."""
    n = len(ARSENAL_HEAD_FIELDS)
    model = _tiny_model()
    if model.arsenal_norm.mean.shape[0] != n:
        return CheckResult(
            "4. arsenal_per_type feature count matches standardization buffer", "FAIL",
            f"len(ARSENAL_HEAD_FIELDS)={n} but arsenal_norm.mean.shape={model.arsenal_norm.mean.shape}",
        )
    if len(BATTER_PER_TYPE_HEAD_FIELDS) != model.batter_pt_norm.mean.shape[0]:
        return CheckResult(
            "4. batter_per_type feature count matches standardization buffer", "FAIL",
            "BATTER_PER_TYPE_HEAD_FIELDS does not match batter_pt_norm",
        )
    return CheckResult(
        "4. arsenal/batter_per_type feature counts match standardization buffer shapes",
        "PASS",
        f"arsenal={n} batter_per_type={len(BATTER_PER_TYPE_HEAD_FIELDS)}",
    )


def check_gradient_flow_iql() -> CheckResult:
    """Run IQL loss forward+backward; verify every requires_grad=True parameter
    receives a non-zero gradient. Catches dead branches and accidentally frozen
    layers."""
    model = _tiny_model()
    batch = _make_batch()

    # Initialize optimizer to zero grads first
    model.zero_grad()
    out = model(batch)
    v_next = shift_v_for_next_state(out["v"], batch.valid_mask)
    losses = iql_losses(
        q_type=out["q_type"], q_x=out["q_x"], q_z=out["q_z"],
        q_x_logits=out["q_x_logits"], q_z_logits=out["q_z_logits"],
        v_current=out["v"], v_next=v_next,
        reward=batch.reward, is_terminal=batch.is_terminal,
        valid_mask=batch.valid_mask, gamma=1.0, tau=0.7,
    )
    total = losses["q_loss"] + losses["v_loss"]
    total.backward()

    no_grad: list[str] = []
    zero_grad: list[str] = []
    nan_grad: list[str] = []
    n_with_grad = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            no_grad.append(name)
            continue
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            nan_grad.append(name)
            continue
        if p.grad.abs().sum().item() == 0.0:
            zero_grad.append(name)
            continue
        n_with_grad += 1

    issues = []
    if no_grad:
        issues.append(f"{len(no_grad)} params with grad=None: {no_grad[:5]}…")
    if zero_grad:
        issues.append(f"{len(zero_grad)} params with grad=0 everywhere: {zero_grad[:5]}…")
    if nan_grad:
        issues.append(f"{len(nan_grad)} params with NaN/Inf grad: {nan_grad[:5]}…")

    status = "PASS" if not issues else "WARN" if not nan_grad and not no_grad else "FAIL"
    return CheckResult(
        "5. IQL: every requires_grad parameter receives a non-zero finite gradient",
        status,
        f"{n_with_grad}/{n_with_grad + len(no_grad) + len(zero_grad) + len(nan_grad)} params got gradient",
        detail="" if not issues else "\n".join(issues),
    )


def check_gradient_flow_fqe() -> CheckResult:
    """FQE backward pass should propagate gradients to fqe_model only — never policy_model."""
    fqe = _tiny_model()
    policy = _tiny_model()
    batch = _make_batch()

    # Snapshot policy params, freeze gradients, ensure they don't change
    for p in policy.parameters():
        p.requires_grad_(False)

    fqe.zero_grad()
    losses = fqe_loss(fqe, policy, batch, gamma=1.0)
    losses["fqe_loss"].backward()

    # FQE trains ONLY q_head_z (the deepest head). The shallow heads
    # (q_head_type, q_head_x) and v_head are intentionally unused — using a
    # per-axis max over their logits would compute Q* (optimal value),
    # biasing FQE estimates upward and defeating its whole point.
    #
    # Two-part check:
    #   Part A (positive negative): assert NO gradient on shallow heads + v_head
    #   Part B (positive): all other params (encoder/transformer/q_head_z/embeddings) get gradient
    #
    # Part A defends against future regressions where someone re-adds the
    # shallow losses to FQE.
    EXPECTED_NO_GRAD_PREFIXES = ("q_head_type.", "q_head_x.", "v_head.")

    # Part A: positive assertion that shallow heads + v_head have NO gradient
    unexpected_grad: list[str] = []
    for name, p in fqe.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(EXPECTED_NO_GRAD_PREFIXES):
            if p.grad is not None and p.grad.abs().sum().item() > 0.0:
                unexpected_grad.append(name)

    # Part B: the rest should have gradient
    no_grad_unexpected: list[str] = []
    n_with_grad = 0
    for name, p in fqe.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(EXPECTED_NO_GRAD_PREFIXES):
            continue
        if p.grad is None or p.grad.abs().sum().item() == 0.0:
            no_grad_unexpected.append(name)
            continue
        n_with_grad += 1

    # Verify policy got nothing
    policy_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in policy.parameters())

    issues = []
    if policy_has_grad:
        issues.append("policy_model has non-zero grads after fqe backward")
    if unexpected_grad:
        issues.append(
            f"{len(unexpected_grad)} shallow/V-head params unexpectedly have gradient — "
            f"someone may have re-added shallow losses to FQE: {unexpected_grad[:5]}"
        )
    if no_grad_unexpected:
        issues.append(f"{len(no_grad_unexpected)} expected-trainable params got no gradient: {no_grad_unexpected[:5]}")

    if issues:
        return CheckResult(
            "6. FQE: gradient flows to q_head_z + encoder only; shallow heads and v_head untrained; policy frozen",
            "FAIL", "; ".join(issues),
        )
    return CheckResult(
        "6. FQE: gradient flows to q_head_z + encoder only; shallow heads and v_head untrained; policy frozen",
        "PASS",
        f"{n_with_grad} expected params got grad; q_head_type/q_head_x/v_head correctly received none; policy_model unchanged",
    )


def check_iql_loss_components_separable() -> CheckResult:
    """v_loss should depend only on V-head + encoder; q_loss should reach Q-heads.
    A common bug is using V's gradient flow on Q parameters or vice versa.
    Quick signal: zero out q_loss and verify Q-head parameters get no gradient."""
    model = _tiny_model()
    batch = _make_batch()

    # Try v_loss-only backward pass; q-head params should get no gradient
    model.zero_grad()
    out = model(batch)
    v_next = shift_v_for_next_state(out["v"], batch.valid_mask)
    losses = iql_losses(
        q_type=out["q_type"], q_x=out["q_x"], q_z=out["q_z"],
        q_x_logits=out["q_x_logits"], q_z_logits=out["q_z_logits"],
        v_current=out["v"], v_next=v_next, reward=batch.reward,
        is_terminal=batch.is_terminal, valid_mask=batch.valid_mask,
        gamma=1.0, tau=0.7,
    )
    losses["v_loss"].backward(retain_graph=False)

    q_head_params_with_grad = []
    for name, p in model.named_parameters():
        if "q_head" in name and p.grad is not None and p.grad.abs().sum().item() > 0:
            q_head_params_with_grad.append(name)

    # IQL's v_loss uses Q^.detach() in the expectile, so Q-heads should NOT receive gradient.
    # If they do, the loss isn't doing the IQL trick correctly.
    if q_head_params_with_grad:
        return CheckResult(
            "7. IQL v_loss does not flow gradient back into Q-heads (uses Q.detach)", "FAIL",
            f"{len(q_head_params_with_grad)} Q-head params got grad from v_loss",
            detail=f"first 5: {q_head_params_with_grad[:5]}",
        )
    return CheckResult(
        "7. IQL v_loss does not flow gradient back into Q-heads (uses Q.detach)", "PASS",
        "Q-head params correctly receive no gradient from v_loss",
    )


def check_no_nan_inf_in_forward() -> CheckResult:
    """All model outputs should be finite on a typical batch."""
    model = _tiny_model()
    batch = _make_batch()
    model.eval()
    with torch.no_grad():
        out = model(batch)
    bad = []
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            if torch.isnan(v).any():
                bad.append(f"{k}: NaN")
            if torch.isinf(v).any():
                bad.append(f"{k}: Inf")
    return CheckResult(
        "8. forward pass produces no NaN/Inf in any output", "PASS" if not bad else "FAIL",
        "all finite" if not bad else "; ".join(bad),
    )


def check_padding_does_not_affect_valid_outputs() -> CheckResult:
    """A batch with mixed PA lengths shouldn't differ from individually-batched
    PAs at the valid positions. Tests collate-time padding doesn't leak."""
    model = _tiny_model()
    model.eval()
    batch_full = _make_batch(B=4, T=4, valid_lengths=[2, 4, 3, 1])

    with torch.no_grad():
        out_full = model(batch_full)["q_chosen"]

    # Re-run each PA individually (size-1 batch with T=true length)
    max_diff = 0.0
    for b in range(4):
        L = int(batch_full.pa_lengths[b].item())
        single = PABatch(
            pre_cat={k: v[b:b+1, :L] for k, v in batch_full.pre_cat.items()},
            pre_cont=batch_full.pre_cont[b:b+1, :L],
            profile=batch_full.profile[b:b+1, :L],
            post_cat={k: v[b:b+1, :L] for k, v in batch_full.post_cat.items()},
            post_cont=batch_full.post_cont[b:b+1, :L],
            reward=batch_full.reward[b:b+1, :L],
            is_terminal=batch_full.is_terminal[b:b+1, :L],
            pa_lengths=batch_full.pa_lengths[b:b+1],
            valid_mask=batch_full.valid_mask[b:b+1, :L],
            arsenal_per_type=batch_full.arsenal_per_type[b:b+1, :L],
            batter_per_type=batch_full.batter_per_type[b:b+1, :L],
        )
        with torch.no_grad():
            out_solo = model(single)["q_chosen"]
        diff = (out_full[b, :L] - out_solo[0, :L]).abs().max()
        max_diff = max(max_diff, float(diff.item()))

    ok = max_diff < 1e-4
    return CheckResult(
        "9. padding does not affect valid-position outputs (variable PA lengths)",
        "PASS" if ok else "FAIL",
        f"max |Δq_chosen| between batched and singleton runs = {max_diff:.2e}",
    )


def check_repertoire_mask_unk_fallback() -> CheckResult:
    """Pitcher with no arsenal entries should get an all-True mask (fallback to
    'all actions allowed') instead of being blocked from acting entirely."""
    pitcher_ids = torch.tensor([[99, 99]])  # pitcher_id 99 not in arsenal
    mask = build_repertoire_mask(pitcher_ids, {}, n_pitch_types=N_PT, n_min=10)
    if mask.all().item():
        return CheckResult(
            "10. repertoire mask: unseen pitcher gets all-True fallback (not blocked)",
            "PASS", "fallback verified",
        )
    return CheckResult(
        "10. repertoire mask: unseen pitcher gets all-True fallback (not blocked)",
        "FAIL", "mask is not all-True for unseen pitcher",
    )


def check_repertoire_mask_known_pitcher_restricts() -> CheckResult:
    """Pitcher with arsenal entries should be restricted to their repertoire."""
    pitcher_ids = torch.tensor([[1, 1]])
    arsenal = {(1, 0): 100, (1, 3): 100}  # only types 0 and 3
    mask = build_repertoire_mask(pitcher_ids, arsenal, n_pitch_types=N_PT, n_min=10)
    expected = [True, False, False, True, False, False]
    actual = mask[0, 0].tolist()
    if actual == expected:
        return CheckResult(
            "11. repertoire mask: known pitcher restricted to in-arsenal types",
            "PASS", f"mask = {actual}",
        )
    return CheckResult(
        "11. repertoire mask: known pitcher restricted to in-arsenal types",
        "FAIL", f"mask = {actual} expected {expected}",
    )


def check_save_load_bit_exact() -> CheckResult:
    """Save and reload should produce identical forward outputs."""
    model = _tiny_model()
    batch = _make_batch()
    model.eval()
    with torch.no_grad():
        out_before = model(batch)["q_chosen"].clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    torch.save(model.state_dict(), path)
    model2 = _tiny_model()
    model2.load_state_dict(torch.load(path, weights_only=True))
    model2.eval()
    with torch.no_grad():
        out_after = model2(batch)["q_chosen"]
    path.unlink(missing_ok=True)

    diff = (out_before - out_after).abs().max().item()
    return CheckResult(
        "12. save/load bit-exact round-trip",
        "PASS" if diff < 1e-7 else "FAIL",
        f"max |Δ| = {diff:.2e}",
    )


def check_disk_artifacts_present() -> CheckResult:
    """Verify Phase 5 disk artifacts exist with expected keys."""
    needed = ["train.parquet", "val.parquet", "test.parquet",
              "pitcher_arsenal.parquet", "batter_profile.parquet",
              "vocab.json", "feature_stats.json"]
    missing = [n for n in needed if not (TOKENS_DIR / n).exists()]
    if missing:
        return CheckResult(
            "13. data/tokens/ has all Phase 5 artifacts",
            "WARN", f"missing: {missing} (run scripts/05_tokenize.py to regenerate)",
        )
    # Validate vocab.json has expected keys
    import json
    vocab = json.loads((TOKENS_DIR / "vocab.json").read_text())
    expected_keys = {"pitch_type", "description", "inning_topbot", "p_throws", "stand",
                     "batter", "pitcher", "x_bin_edges", "z_bin_edges"}
    missing_vocab = expected_keys - set(vocab.keys())
    if missing_vocab:
        return CheckResult(
            "13. data/tokens/ artifacts complete", "FAIL",
            f"vocab.json missing keys: {missing_vocab}",
        )
    return CheckResult(
        "13. data/tokens/ has all Phase 5 artifacts",
        "PASS", f"all 7 files present; vocab.json has all expected keys",
    )


def check_real_data_forward_runs() -> CheckResult:
    """Build the real model from disk artifacts and run a forward pass on real
    val data. Catches integration issues that synthetic batches can't."""
    if not (TOKENS_DIR / "vocab.json").exists():
        return CheckResult(
            "14. real-data forward pass via build_qtransformer", "WARN",
            "skipped — data/tokens/ not populated",
        )
    try:
        vocab_sizes = load_vocab_sizes(TOKENS_DIR / "vocab.json")
        ds = PitchPADataset(
            TOKENS_DIR / "val.parquet",
            TOKENS_DIR / "batter_profile.parquet",
            pitcher_arsenal_path=TOKENS_DIR / "pitcher_arsenal.parquet",
            n_pitch_types=vocab_sizes["pitch_type"],
        )
        items = [ds[i] for i in range(min(8, len(ds)))]
        batch = pa_collate(items)
        model = build_qtransformer(TOKENS_DIR, preset="smoke")
        model.eval()
        with torch.no_grad():
            out = model(batch)
        finite = all(
            torch.isfinite(v).all().item() for v in out.values()
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
        )
        if not finite:
            return CheckResult(
                "14. real-data forward pass via build_qtransformer", "FAIL",
                "outputs contained NaN/Inf",
            )
        return CheckResult(
            "14. real-data forward pass via build_qtransformer", "PASS",
            f"forward on {len(items)} real PAs from val: outputs finite, "
            f"q_chosen shape={tuple(out['q_chosen'].shape)}",
        )
    except Exception as e:
        return CheckResult(
            "14. real-data forward pass via build_qtransformer", "FAIL",
            f"raised {type(e).__name__}: {e}",
            detail=traceback.format_exc(),
        )


def check_predict_batch_dtypes() -> CheckResult:
    """ope_metrics.predict_batch flattens valid pitches and returns various tensors.
    Verify dtypes are reasonable (long for IDs, float for continuous)."""
    model = _tiny_model()
    batch = _make_batch()
    model.eval()
    preds = predict_batch(model, batch)
    bad = []
    for name, expected_dtype in [
        ("policy_pitch_type", torch.int64),
        ("policy_x_bin", torch.int64),
        ("policy_z_bin", torch.int64),
        ("actual_pitch_type", torch.int64),
        ("actual_x_bin", torch.int64),
        ("actual_z_bin", torch.int64),
        ("balls", torch.int64),
        ("strikes", torch.int64),
    ]:
        t = getattr(preds, name)
        if t.dtype != expected_dtype:
            bad.append(f"{name}: {t.dtype} (expected {expected_dtype})")
    return CheckResult(
        "15. ope_metrics.predict_batch returns expected dtypes",
        "PASS" if not bad else "FAIL",
        "all dtypes correct" if not bad else "; ".join(bad),
    )


def check_repertoire_mask_default_is_disabled() -> CheckResult:
    """Lock in the design choice that the repertoire mask defaults to OFF.
    Future regressions that flip the default would fail this check immediately."""
    from src.fqe import FQETrainerConfig
    from src.qtransformer import repertoire_mask_from_batch

    issues = []
    if FQETrainerConfig().repertoire_mask_min_count != 0:
        issues.append(f"FQETrainerConfig default min_count is {FQETrainerConfig().repertoire_mask_min_count}, expected 0")

    batch = _make_batch()
    if repertoire_mask_from_batch(batch, n_min=0) is not None:
        issues.append("repertoire_mask_from_batch(n_min=0) should return None (disabled)")
    mask = repertoire_mask_from_batch(batch, n_min=1)
    if mask is None or mask.dtype != torch.bool:
        issues.append("repertoire_mask_from_batch(n_min=1) should return a bool tensor")

    return CheckResult(
        "17. repertoire mask defaults to disabled (OFF)",
        "PASS" if not issues else "FAIL",
        f"FQETrainerConfig.repertoire_mask_min_count default = {FQETrainerConfig().repertoire_mask_min_count}",
        detail="" if not issues else "; ".join(issues),
    )


def check_param_count_invariant() -> CheckResult:
    """Sanity: the v1 preset should have ~12M params; the smoke preset should
    have ~130-160K. Catches accidental architectural inflation."""
    if not (TOKENS_DIR / "vocab.json").exists():
        return CheckResult(
            "16. parameter counts in expected ranges per preset", "WARN",
            "skipped — data/tokens/ not populated",
        )
    smoke = build_qtransformer(TOKENS_DIR, preset="smoke")
    v1 = build_qtransformer(TOKENS_DIR, preset="v1")
    smoke_n = sum(p.numel() for p in smoke.parameters())
    v1_n = sum(p.numel() for p in v1.parameters())
    smoke_ok = 100_000 <= smoke_n <= 200_000
    v1_ok = 8_000_000 <= v1_n <= 20_000_000
    status = "PASS" if (smoke_ok and v1_ok) else "WARN"
    return CheckResult(
        "16. parameter counts in expected ranges per preset",
        status,
        f"smoke={smoke_n:,} (expected 100K-200K) | v1={v1_n:,} (expected 8M-20M)",
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def format_report(results: list[CheckResult]) -> str:
    lines = ["=" * 80, "Framework audit report", "=" * 80]
    for r in results:
        lines.append(f"[{r.status:<4}] {r.name}: {r.summary}")
        if r.detail:
            for d in r.detail.splitlines():
                if d:
                    lines.append(f"        {d}")
    failed = sum(1 for r in results if r.status == "FAIL")
    warned = sum(1 for r in results if r.status == "WARN")
    lines.append("=" * 80)
    lines.append(f"Summary: {len(results) - failed - warned} pass / {warned} warn / {failed} fail")
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    torch.manual_seed(0)

    results = [
        check_action_space_constants_consistent(),
        check_pre_cont_field_index_alignment(),
        check_post_cont_field_index_alignment(),
        check_arsenal_field_count_matches_tensor_shape(),
        check_gradient_flow_iql(),
        check_gradient_flow_fqe(),
        check_iql_loss_components_separable(),
        check_no_nan_inf_in_forward(),
        check_padding_does_not_affect_valid_outputs(),
        check_repertoire_mask_unk_fallback(),
        check_repertoire_mask_known_pitcher_restricts(),
        check_save_load_bit_exact(),
        check_disk_artifacts_present(),
        check_real_data_forward_runs(),
        check_predict_batch_dtypes(),
        check_repertoire_mask_default_is_disabled(),
        check_param_count_invariant(),
    ]
    print(format_report(results))
    if any(r.status == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
