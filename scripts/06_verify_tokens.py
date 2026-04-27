"""Token-output verification report. Exits non-zero on any FAIL.

Spec: CLAUDE.md § Phase 5 — seven checks covering action ranges, terminal
invariants, sequence contiguity, reward preservation, mirror symmetry,
no-nulls, and arsenal coverage.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tokenize import N_X_BINS, N_Z_BINS, N_MIN_ARSENAL_SAMPLES, UNK_ID  # noqa: E402

SPLITS_DIR = REPO_ROOT / "data" / "splits"
TOKENS_DIR = REPO_ROOT / "data" / "tokens"

PA_KEYS = ["game_pk", "at_bat_number"]


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    summary: str
    detail: str = ""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_tokens(name: str) -> pd.DataFrame:
    return pd.read_parquet(TOKENS_DIR / f"{name}.parquet", engine="pyarrow")


def _load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS_DIR / f"{name}.parquet", engine="pyarrow")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_action_in_vocab_and_range(tokens: dict[str, pd.DataFrame], vocab: dict) -> CheckResult:
    bad: dict[str, int] = {}
    for name, df in tokens.items():
        n_bad_pt = int((df["pitch_type_id"] == UNK_ID).sum()) if name == "train" else 0
        n_bad_x = int(((df["x_bin"] < 0) | (df["x_bin"] >= N_X_BINS)).sum())
        n_bad_z = int(((df["z_bin"] < 0) | (df["z_bin"] >= N_Z_BINS)).sum())
        if n_bad_pt + n_bad_x + n_bad_z:
            bad[name] = {"pitch_type_unk_in_train": n_bad_pt, "x_oob": n_bad_x, "z_oob": n_bad_z}
    status = "PASS" if not bad else "FAIL"
    pt_vocab_size = len(vocab["pitch_type"])
    return CheckResult(
        name="1. actions in-vocab and in-range",
        status=status,
        summary=f"pitch_type_vocab={pt_vocab_size} | x_bins={N_X_BINS} | z_bins={N_Z_BINS}",
        detail="" if not bad else f"violations: {bad}",
    )


def check_terminal_exactly_once_per_pa(tokens: dict[str, pd.DataFrame]) -> CheckResult:
    summaries = []
    failed = False
    for name, df in tokens.items():
        terms = df.groupby(PA_KEYS, sort=False)["is_terminal"].sum()
        if terms.min() != 1 or terms.max() != 1:
            failed = True
        summaries.append(f"{name}: pas={len(terms)} terminals_min={terms.min()} terminals_max={terms.max()}")
        # Also: terminal must be on the LAST row of the PA (highest pitch_idx).
        last_idx = df.groupby(PA_KEYS, sort=False)["pitch_idx_in_pa"].idxmax()
        if not df.loc[last_idx, "is_terminal"].all():
            failed = True
    status = "FAIL" if failed else "PASS"
    return CheckResult(
        name="2. is_terminal exactly once and on the last pitch per PA",
        status=status,
        summary="; ".join(summaries),
    )


def check_pitch_idx_contiguous(tokens: dict[str, pd.DataFrame]) -> CheckResult:
    failed_splits = []
    for name, df in tokens.items():
        idx = df.groupby(PA_KEYS, sort=False)["pitch_idx_in_pa"]
        bad = idx.apply(lambda s: list(s) != list(range(len(s))))
        if bad.any():
            failed_splits.append((name, int(bad.sum())))
    status = "PASS" if not failed_splits else "FAIL"
    return CheckResult(
        name="3. pitch_idx_in_pa runs 0..len(PA)-1 with no gaps and matches row order",
        status=status,
        summary=f"bad PAs: {failed_splits}" if failed_splits else "all clean",
    )


def check_reward_preserved(splits: dict[str, pd.DataFrame], tokens: dict[str, pd.DataFrame]) -> CheckResult:
    diffs = {}
    for name in tokens:
        pre = splits[name].groupby(PA_KEYS, sort=False)["reward_pitcher"].sum()
        post = tokens[name].groupby(PA_KEYS, sort=False)["reward_pitcher"].sum()
        joined = pre.to_frame("pre").join(post.to_frame("post"), how="outer")
        max_abs_diff = float((joined["pre"] - joined["post"]).abs().max())
        diffs[name] = max_abs_diff
    status = "PASS" if all(v < 1e-3 for v in diffs.values()) else "FAIL"
    return CheckResult(
        name="4. per-PA reward sums match pre-tokenization",
        status=status,
        summary=" | ".join(f"{k} max|Δ|={v:.2e}" for k, v in diffs.items()),
    )


PER_PLATOON_GAP_THRESHOLD = 0.10  # ft, ~1.2 inches
DIRECTION_SANITY_THRESHOLD = 0.30  # ft, ~3.6 inches — a wrong-direction mirror would blow past this


def check_mirror_invariant(splits: dict[str, pd.DataFrame], tokens: dict[str, pd.DataFrame]) -> CheckResult:
    """Mirror correctness — compared like-with-like across pitcher hand.

    Compares the median ``plate_x_mirrored`` between the two same-handed
    matchups (RHP-vs-RHB vs LHP-vs-LHB) and between the two opposite-handed
    matchups (RHP-vs-LHB vs LHP-vs-RHB). Holding the platoon situation
    constant isolates "is the mirror itself working" from "do pitchers
    behave asymmetrically by handedness" (they do, and that's not a bug).

    Also guards against a wrong-sign mirror: each (p_throws, stand) cell's
    median must be within DIRECTION_SANITY_THRESHOLD ft of zero.
    """
    train_split = splits["train"]
    train_tok = tokens["train"].copy()
    train_tok["p_throws"] = train_split["p_throws"].to_numpy()
    train_tok["stand"] = train_split["stand"].to_numpy()

    medians = train_tok.groupby(["p_throws", "stand"])["plate_x_mirrored"].median()

    same_R = float(medians.loc[("R", "R")])
    same_L = float(medians.loc[("L", "L")])
    opp_RL = float(medians.loc[("R", "L")])
    opp_LR = float(medians.loc[("L", "R")])

    same_gap = abs(same_R - same_L)
    opp_gap = abs(opp_RL - opp_LR)

    direction_violations = [
        f"({pt},{st}) median={float(m):+.4f}"
        for (pt, st), m in medians.items()
        if abs(float(m)) >= DIRECTION_SANITY_THRESHOLD
    ]

    if direction_violations:
        status = "FAIL"
    elif same_gap >= PER_PLATOON_GAP_THRESHOLD or opp_gap >= PER_PLATOON_GAP_THRESHOLD:
        status = "WARN"
    else:
        status = "PASS"

    summary = (
        f"same-handed |Δ|={same_gap:.4f}  opposite-handed |Δ|={opp_gap:.4f}  "
        f"(per-platoon threshold {PER_PLATOON_GAP_THRESHOLD})"
    )
    detail_lines = [
        f"same-handed gap:     RHP/RHB median={same_R:+.4f}  LHP/LHB median={same_L:+.4f}  |Δ|={same_gap:.4f}",
        f"opposite-handed gap: RHP/LHB median={opp_RL:+.4f}  LHP/RHB median={opp_LR:+.4f}  |Δ|={opp_gap:.4f}",
        f"direction sanity (all 4 cells |median| < {DIRECTION_SANITY_THRESHOLD} ft): "
        + ("PASS" if not direction_violations else f"FAIL — {direction_violations}"),
    ]
    return CheckResult(
        name="5. mirror invariant: like-with-like medians align across pitcher hand",
        status=status,
        summary=summary,
        detail="\n".join(detail_lines),
    )


def check_no_nulls_in_tokens(tokens: dict[str, pd.DataFrame]) -> CheckResult:
    bad = {}
    for name, df in tokens.items():
        nulls = df.isna().sum()
        nulls = nulls[nulls > 0]
        if len(nulls):
            bad[name] = nulls.to_dict()
    status = "PASS" if not bad else "FAIL"
    return CheckResult(
        name="6. no nulls anywhere in token files",
        status=status,
        summary="all clean" if not bad else f"violations: {bad}",
    )


def check_arsenal_coverage(tokens: dict[str, pd.DataFrame]) -> CheckResult:
    arsenal = pd.read_parquet(TOKENS_DIR / "pitcher_arsenal.parquet", engine="pyarrow")
    train_pairs = tokens["train"][["pitcher_id", "pitch_type_id"]].drop_duplicates()
    arsenal_pairs = arsenal[["pitcher_id", "pitch_type_id"]].drop_duplicates()
    merged = train_pairs.merge(arsenal_pairs, on=["pitcher_id", "pitch_type_id"], how="left", indicator=True)
    coverage = (merged["_merge"] == "both").mean()
    n_low_sample = int(arsenal["low_sample"].sum())
    status = "PASS" if coverage >= 0.95 else "WARN"
    return CheckResult(
        name="7. pitcher_arsenal covers ≥95% of train (pitcher, pitch_type) pairs",
        status=status,
        summary=(
            f"coverage={coverage:.4f}  arsenal_rows={len(arsenal)}  "
            f"low_sample (count<{N_MIN_ARSENAL_SAMPLES})={n_low_sample}"
        ),
    )


def check_batter_profile_coverage(tokens: dict[str, pd.DataFrame]) -> CheckResult:
    profile = pd.read_parquet(TOKENS_DIR / "batter_profile.parquet", engine="pyarrow")
    train_batters = set(tokens["train"]["batter_id"].unique().tolist())
    profile_batters = set(profile["batter_id"].unique().tolist())
    coverage = len(train_batters & profile_batters) / max(1, len(train_batters))
    n_low_sample = int(profile["low_sample"].sum())
    n_unique_batters = profile["batter_id"].nunique()
    status = "PASS" if coverage >= 0.95 else "WARN"
    return CheckResult(
        name="8. batter_profile covers ≥95% of train batters",
        status=status,
        summary=(
            f"coverage={coverage:.4f}  unique_batters={n_unique_batters}  "
            f"rows={len(profile)} ((batter,pitch_type) groups)  "
            f"low_sample (count<30)={n_low_sample}"
        ),
    )


# --------------------------------------------------------------------------- #
# Driver + formatting
# --------------------------------------------------------------------------- #


def run_all_checks() -> list[CheckResult]:
    tokens = {name: _load_tokens(name) for name in ("train", "val", "test")}
    splits = {name: _load_split(name) for name in ("train", "val", "test")}
    with (TOKENS_DIR / "vocab.json").open() as f:
        vocab = json.load(f)
    return [
        check_action_in_vocab_and_range(tokens, vocab),
        check_terminal_exactly_once_per_pa(tokens),
        check_pitch_idx_contiguous(tokens),
        check_reward_preserved(splits, tokens),
        check_mirror_invariant(splits, tokens),
        check_no_nulls_in_tokens(tokens),
        check_arsenal_coverage(tokens),
        check_batter_profile_coverage(tokens),
    ]


def format_report(results: list[CheckResult]) -> str:
    lines = ["=" * 80, "Token verification report (Phase 5)", "=" * 80]
    for r in results:
        lines.append(f"[{r.status:<4}] {r.name}: {r.summary}")
        if r.detail:
            for d in r.detail.splitlines():
                if d:
                    lines.append(f"        {d}")
    lines.append("=" * 80)
    failed = sum(1 for r in results if r.status == "FAIL")
    warned = sum(1 for r in results if r.status == "WARN")
    lines.append(f"Summary: {len(results) - failed - warned} pass / {warned} warn / {failed} fail")
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    results = run_all_checks()
    print(format_report(results))
    if any(r.status == "FAIL" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
