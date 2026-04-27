"""Markdown report generator for Phase 9 evaluation.

Renders :class:`src.ope_metrics.BehavioralMetrics` and segment breakdown
DataFrames into a readable markdown document for inclusion in a run directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ope_metrics import BehavioralMetrics


def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:.2f}%"


def _fmt_ft(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.3f} ft ({x*12:.1f} in)"


def _fmt_count(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{int(x):,}"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Tiny markdown-table renderer (avoids the optional ``tabulate`` dependency)."""
    cols = [df.index.name or ""] + list(df.columns)
    widths = [max(len(str(c)), 3) for c in cols]
    for i, idx in enumerate(df.index):
        widths[0] = max(widths[0], len(str(idx)))
        for j, c in enumerate(df.columns):
            widths[j + 1] = max(widths[j + 1], len(str(df.iloc[i, j])))
    lines = []
    lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for idx, row in df.iterrows():
        cells = [str(idx)] + [str(v) for v in row.values]
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |")
    return "\n".join(lines)


def render_behavioral_report(
    metrics: BehavioralMetrics,
    segments: dict[str, pd.DataFrame],
    *,
    run_name: str,
    split_name: str,
    pitch_type_vocab_inv: dict[int, str] | None = None,
) -> str:
    """Render the Part-A behavioral evaluation report as markdown."""
    lines: list[str] = []
    lines.append(f"# Behavioral evaluation — `{run_name}` on `{split_name}`\n")
    lines.append(f"_n_pitches = {metrics.n_pitches:,}_\n")

    # --------------------------------------------------------------------- #
    # Aggregate metrics
    # --------------------------------------------------------------------- #
    lines.append("## Aggregate metrics\n")
    lines.append("| Metric | Value | Pitcher-blind | Personal lift |")
    lines.append("|---|---:|---:|---:|")
    lift_pt1 = (
        f"{(metrics.pitch_type_top1 - metrics.pitch_type_top1_blind)*100:+.2f} pp"
        if metrics.pitch_type_top1_blind is not None else "—"
    )
    lift_pt3 = (
        f"{(metrics.pitch_type_top3 - metrics.pitch_type_top3_blind)*100:+.2f} pp"
        if metrics.pitch_type_top3_blind is not None else "—"
    )
    lines.append(f"| Pitch-type top-1 agreement | {_fmt_pct(metrics.pitch_type_top1)} | "
                 f"{_fmt_pct(metrics.pitch_type_top1_blind)} | {lift_pt1} |")
    lines.append(f"| Pitch-type top-3 agreement | {_fmt_pct(metrics.pitch_type_top3)} | "
                 f"{_fmt_pct(metrics.pitch_type_top3_blind)} | {lift_pt3} |")
    lines.append(f"| Coarse-zone (type × 4×4) agreement | {_fmt_pct(metrics.coarse_zone_top1)} | — | — |")
    lines.append(f"| Spatial distance — mean | {_fmt_ft(metrics.spatial_distance_mean_ft)} | — | — |")
    lines.append(f"| Spatial distance — median | {_fmt_ft(metrics.spatial_distance_median_ft)} | "
                 f"{_fmt_ft(metrics.spatial_distance_median_ft_blind)} | — |")
    lines.append(f"| Spatial distance — p75 | {_fmt_ft(metrics.spatial_distance_p75_ft)} | — | — |")
    lines.append(f"| Within 6 inches of actual | {_fmt_pct(metrics.spatial_within_6in_frac)} | — | — |")
    lines.append(f"| KL[π_learned ∥ behavior] (pitch_type) | {metrics.pitch_type_kl_learned_to_behavior:.4f} nats | — | — |")
    lines.append("")

    # --------------------------------------------------------------------- #
    # Pitch-type distribution
    # --------------------------------------------------------------------- #
    lines.append("## Pitch-type distribution: learned vs behavior\n")
    all_types = sorted(set(metrics.pitch_type_dist_learned) | set(metrics.pitch_type_dist_behavior))
    lines.append("| Pitch type | Learned % | Behavior % | Δ (pp) |")
    lines.append("|---|---:|---:|---:|")
    for t in all_types:
        name = pitch_type_vocab_inv.get(t, f"#{t}") if pitch_type_vocab_inv else f"#{t}"
        learned = metrics.pitch_type_dist_learned.get(t, 0.0)
        behavior = metrics.pitch_type_dist_behavior.get(t, 0.0)
        delta = (learned - behavior) * 100
        lines.append(f"| {name} | {learned*100:.2f}% | {behavior*100:.2f}% | {delta:+.2f} |")
    lines.append("")

    # --------------------------------------------------------------------- #
    # Segment breakdowns
    # --------------------------------------------------------------------- #
    if segments:
        lines.append("## Per-segment breakdowns\n")
        for seg_name, df in segments.items():
            if df.empty:
                continue
            lines.append(f"### By {seg_name}\n")
            df = df.copy()
            for c in ("top1", "top3"):
                if c in df.columns:
                    df[c] = df[c].apply(lambda v: _fmt_pct(v) if pd.notna(v) else "—")
            for c in ("median_dist_ft",):
                if c in df.columns:
                    df[c] = df[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            if "n" in df.columns:
                df["n"] = df["n"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
            lines.append(_df_to_markdown(df))
            lines.append("")

    return "\n".join(lines)


def write_behavioral_report(
    metrics: BehavioralMetrics,
    segments: dict[str, pd.DataFrame],
    *,
    out_path: Path,
    run_name: str,
    split_name: str,
    vocab_path: Path | None = None,
) -> None:
    """Render and write the report. Loads pitch_type vocab inverse if available."""
    pitch_type_vocab_inv: dict[int, str] | None = None
    if vocab_path is not None and vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
        pitch_type_vocab_inv = {v: k for k, v in vocab["pitch_type"].items()}

    md = render_behavioral_report(
        metrics, segments,
        run_name=run_name, split_name=split_name,
        pitch_type_vocab_inv=pitch_type_vocab_inv,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
