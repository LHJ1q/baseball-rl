# baseball-rl

Offline reinforcement learning for MLB pitch selection. A **Q-Transformer** trained on Statcast data with **Implicit Q-Learning (IQL)** that predicts the best `(pitch_type, plate_x, plate_z)` action for the pitcher given the within-PA pitch sequence and game state.

- **Reward:** `delta_run_exp` (Statcast's per-pitch run-expectancy delta), negated for the pitcher's perspective (`reward_pitcher = -delta_run_exp`).
- **Action space:** discretized pitch type × plate_x bins × plate_z bins, decoded autoregressively (`Q(s, type) → Q(s, type, x) → Q(s, type, x, z)`) so inference is `~57` evaluations instead of `~6,800`.
- **Training:** per-axis Q-Transformer Bellman backup (Chebotar et al. 2023) with IQL's V-bootstrap on the deepest axis. Each shallow head's target is the next axis's max (no reward, no γ within the same timestep); only the deepest head bootstraps from `V(s')`.
- **Conservatism:** IQL — expectile regression on V; no max-over-actions queries on out-of-distribution samples.
- **Repertoire-aware action masking** at inference: the policy can never pick a pitch type a pitcher physically doesn't throw (mask derived from the precomputed `pitcher_arsenal.parquet` lookup).

## Pipeline

| Phase | Script | Output |
|---|---|---|
| 1. Download | `scripts/01_download.py` | `data/raw/statcast_{year}_{month}.parquet` |
| 2. Filter + derive | `scripts/02_filter.py` | `data/processed/statcast_{year}.parquet` |
| 3. Splits | `scripts/03_split.py` | `data/splits/{train,val,test}.parquet` |
| 4. Verify (filter) | `scripts/04_verify.py` | 14-check report (statistical + drop-rule compliance) |
| 5. Tokenize | `scripts/05_tokenize.py` | `data/tokens/{train,val,test}.parquet` + `pitcher_arsenal.parquet` + `batter_profile.parquet` + `vocab.json` + `feature_stats.json` |
| 6. Verify (tokens) | `scripts/06_verify_tokens.py` | 8-check report |
| 7. Smoke test | `scripts/07_smoke_test.py` | 10-check model-architecture battery (CPU/MPS) |
| 8. Train | `scripts/08_train.py` | `data/runs/{run_name}/` (config, metrics, checkpoints) |
| 9. Behavioral eval | `scripts/09_behavioral.py` | `data/runs/{run_name}/behavioral_report_{split}.md` |
| 10. FQE eval | `scripts/10_fqe.py` | FQE checkpoint + per-PA value estimates on val + test |
| Audit | `scripts/audit.py` | 24-check framework audit (gradient flow, field alignment, FQE freeze, save/load round-trip, real-data integrity, modern init) |

The full filter + tokenize spec lives in `docs/FILTER_RULES.md` and `CLAUDE.md`. Read those before changing rules or extending seasons.

## Setup

```bash
git clone https://github.com/LHJ1q/baseball-rl.git
cd baseball-rl

# Option A — conda (recommended; handles CUDA toolchain on GPU boxes)
conda env create -f environment.yml
conda activate baseball-rl
pip install -e .       # editable install if you skipped the `-e .` line in environment.yml

# Option B — pip + system Python (>= 3.10)
pip install -e ".[dev]"
```

**GPU compatibility.** The trainer auto-detects CUDA / MPS / CPU and turns on
BF16 autocast + `torch.compile` + TF32 only on CUDA. Verified GPU targets:

## Reproduce — five-season run (2021–2025)

```bash
# 1. Download all five seasons (~25 min cold; pybaseball cache enabled)
for y in 2021 2022 2023 2024 2025; do python scripts/01_download.py --year $y; done

# 2. Filter each season (PA-atomic drops + derived columns; see docs/FILTER_RULES.md)
for y in 2021 2022 2023 2024 2025; do python scripts/02_filter.py --year $y; done

# 3. Year-level temporal splits: train = 2021-2024, val = 2025 first half, test = 2025 second half
python scripts/03_split.py --scheme year_level \
    --train-years 2021 2022 2023 2024 \
    --val-test-year 2025

# 4. Verify the filtered splits (14 checks)
python scripts/04_verify.py

# 5. Tokenize: discretize actions (17 pitch types × 20 x-bins × 20 z-bins),
#    mirror-for-LHP, build per-(pitcher, pitch_type) arsenal + per-(batter, pitch_type) profile lookups
python scripts/05_tokenize.py

# 6. Verify the tokenized output (eight checks)
python scripts/06_verify_tokens.py

# 7. Architecture smoke test (model + IQL loss + repertoire mask + save/load round-trip)
python scripts/07_smoke_test.py --preset smoke
python scripts/07_smoke_test.py --preset v1   # validates the full GPU preset on a tiny batch

# 7b. Framework audit (gradient flow per parameter, FQE freeze, field-index alignment, ...)
python scripts/audit.py

# 8. Train (Linux/Colab GPU recommended; ~1-2 hours total on Blackwell RTX Pro 4500
#    — model is dataloader-bound, FLOPS aren't the bottleneck at 12M params)
python scripts/08_train.py --preset v1 --run-name iql_5year_v1 --epochs 40 --batch-size 1024

# 9. Behavioral / distributional analysis on val + test (runs in seconds)
python scripts/09_behavioral.py --run-name iql_5year_v1 --preset v1 --split val
python scripts/09_behavioral.py --run-name iql_5year_v1 --preset v1 --split test

# 10. FQE — estimates per-PA expected return under the learned policy
#     (~1-2 hours on GPU; trains a separate Q^π network)
python scripts/10_fqe.py --run-name iql_5year_v1 --preset v1 --epochs 20

# Tests
pytest tests/ -v
```

## Single-season reproduction (2024 only)

```bash
python scripts/01_download.py --year 2024
python scripts/02_filter.py --year 2024
python scripts/03_split.py --scheme within_season --year 2024
python scripts/04_verify.py
python scripts/05_tokenize.py
python scripts/06_verify_tokens.py
python scripts/07_smoke_test.py --preset smoke
python scripts/audit.py
```

## Architecture summary

```
                                     ┌──────────────────────┐
                                     │  pitcher_arsenal     │  per (pitcher, pitch_type) physics
                                     │  batter_profile      │  per (batter, pitch_type) tendencies
                                     └──────────┬───────────┘
                                                │ joined at Q-heads
   token sequence per PA                        │
   [pre_0, post_0, pre_1, post_1, ...]          │
            │                                   │
   PreActionEncoder + PostActionEncoder         │
            │                                   │
   causal Transformer (6 layers × 384-dim)     │
            │                                   │
   ┌────────┼────────────────────┐             │
   │        │                    │             │
 V(s)   Q(s, type) ←─────────────┼─ arsenal[pitcher,type] + profile[batter,type]
        Q(s, type, x_bin) ←─────┘
        Q(s, type, x_bin, z_bin)

IQL training (per-axis Bellman backup):
    target_type = max_x q_x_logits.detach()                        ← within-timestep max
    target_x    = max_z q_z_logits.detach()                        ← within-timestep max
    target_z    = (r + γ V(s') (1 − terminal)).detach()            ← IQL TD with V-bootstrap
    q_loss      = MSE(q_type, target_type) + MSE(q_x, target_x) + MSE(q_z, target_z)
    v_loss      = expectile_loss(q_z.detach() − V(s), τ)
```

`v1` preset: `~12M params`, BF16 autocast on Blackwell, batch 512 PAs (1024 also fits comfortably), AdamW lr=3e-4 with 1k-step warmup + cosine decay, IQL τ=0.7, γ=1.0. See CLAUDE.md § Phase 8 → "Deviations from canonical Q-Transformer + IQL" for what's intentionally non-canonical.

## Project layout

```
baseball-rl/
├── CLAUDE.md                # full project spec
├── docs/FILTER_RULES.md     # authoritative drop rules
├── pyproject.toml
├── data/                    # gitignored
├── src/                     # download / filter / splits / verify / tokenize / encoder / dataset / qtransformer / configs / eval / trainer / ope_metrics / report / fqe
├── scripts/                 # 01_download → 10_fqe + audit.py
└── tests/                   # 100 unit tests across all modules (1 CUDA-gated skip on Macbook)
```

## Five-season dataset (2021–2025) — current state

| Stage | Output |
|---|---|
| Raw pulls | 35 month files, 516 MB |
| Filtered (per season) | 5 × ~700K pitches → **3.46M total pitches across all seasons** |
| Splits (year-level) | train: 2.78M rows / 714K PAs (2021-2024) · val: 397K / 103K PAs (2025 Apr–Jul 15) · test: 287K / 74K PAs (2025 Jul 16–Oct) |
| Vocabularies | 17 pitch types · 13 descriptions · **1,570 pitchers** · **1,448 batters** |
| Lookup tables | `pitcher_arsenal.parquet` 6,985 (pitcher, type) groups · `batter_profile.parquet` 13,290 (batter, type) groups |
| Verify | filter 14/14 PASS · tokens 8/8 PASS · framework audit 24/24 PASS |

## Status

- **Phases 1–9: complete.** Data pipeline (download → filter → splits → verify → tokenize → verify) and model stack (encoder → Q-Transformer → IQL trainer → behavioral eval → FQE) are all built, tested, and pushed.
- Tests: **100 passing / 1 skipped** (CUDA-gated `torch.compile` round-trip). Audit: **24/24 PASS**. Smoke: **9 PASS / 1 WARN** (`smoke` preset) and **8 PASS / 2 WARN** (`v1` preset); WARNs are benign (causal-mask check skipped when first PA is length 1; per-axis IQL's q_loss_type non-monotonic during overfit because its target is `max q_x_logits.detach()`, itself learning — documented behavior).
- **Next step:** run `scripts/08_train.py --preset v1` on a GPU box, then run `scripts/09_behavioral.py` and `scripts/10_fqe.py` against the resulting checkpoint.

## License

MIT (see `LICENSE`).
