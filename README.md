# baseball-rl

Offline reinforcement learning for MLB pitch selection. A **Q-Transformer** trained on Statcast data with **Implicit Q-Learning (IQL)** that predicts the best `(pitch_type, plate_x, plate_z)` action for the pitcher given the within-PA pitch sequence and game state.

- **Reward:** `delta_run_exp` (Statcast's per-pitch run-expectancy delta), negated for the pitcher's perspective (`reward_pitcher = -delta_run_exp`).
- **Action space:** discretized pitch type × plate_x bins × plate_z bins, decoded autoregressively (`Q(s, type) → Q(s, type, x) → Q(s, type, x, z)`) so inference is `~57` evaluations instead of `~6,800`.
- **Conservatism:** IQL — TD with V-bootstrap on Q, expectile regression on V. No max-over-actions queries on out-of-distribution actions.
- **Repertoire-aware action masking** at inference: the policy can never pick a pitch type a pitcher physically doesn't throw (mask derived from the precomputed `pitcher_arsenal.parquet` lookup).

## Pipeline

| Phase | Script | Output |
|---|---|---|
| 1. Download | `scripts/01_download.py` | `data/raw/statcast_{year}_{month}.parquet` |
| 2. Filter + derive | `scripts/02_filter.py` | `data/processed/statcast_{year}.parquet` |
| 3. Splits | `scripts/03_split.py` | `data/splits/{train,val,test}.parquet` |
| 4. Verify (filter) | `scripts/04_verify.py` | 6-check report (must all PASS) |
| 5. Tokenize | `scripts/05_tokenize.py` | `data/tokens/{train,val,test}.parquet` + `pitcher_arsenal.parquet` + `batter_profile.parquet` + `vocab.json` + `feature_stats.json` |
| 6. Verify (tokens) | `scripts/06_verify_tokens.py` | 8-check report |
| 7. Smoke test | `scripts/07_smoke_test.py` | 10-check model-architecture battery (CPU/MPS) |
| 8. Train | `scripts/08_train.py` | `data/runs/{run_name}/` (config, metrics, checkpoints) |

The full filter + tokenize spec lives in `docs/FILTER_RULES.md` and `CLAUDE.md`. Read those before changing rules or extending seasons.

## Reproduce — five-season run (2021–2025)

```bash
# Install (Python >= 3.10)
pip install -e ".[dev]"

# 1. Download all five seasons (~25 min cold; pybaseball cache enabled)
for y in 2021 2022 2023 2024 2025; do python scripts/01_download.py --year $y; done

# 2. Filter each season (PA-atomic drops + derived columns; see docs/FILTER_RULES.md)
for y in 2021 2022 2023 2024 2025; do python scripts/02_filter.py --year $y; done

# 3. Year-level temporal splits: train = 2021-2024, val = 2025 first half, test = 2025 second half
python scripts/03_split.py --scheme year_level \
    --train-years 2021 2022 2023 2024 \
    --val-test-year 2025

# 4. Verify the filtered splits (six checks)
python scripts/04_verify.py

# 5. Tokenize: discretize actions (17 pitch types × 20 x-bins × 20 z-bins),
#    mirror-for-LHP, build per-(pitcher, pitch_type) arsenal + per-(batter, pitch_type) profile lookups
python scripts/05_tokenize.py

# 6. Verify the tokenized output (eight checks)
python scripts/06_verify_tokens.py

# 7. Architecture smoke test (model + IQL loss + repertoire mask + save/load round-trip)
python scripts/07_smoke_test.py --preset smoke
python scripts/07_smoke_test.py --preset v1   # validates the full GPU preset on a tiny batch

# 8. Train (Linux/Colab GPU recommended; ~30 min/epoch on Blackwell RTX Pro 4500 at BF16)
python scripts/08_train.py --preset v1 --run-name iql_5year_v1 --epochs 40

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

IQL training:
    Q-loss = MSE(Q(s,a), r + γ V(s'))     no max over actions on OOD samples
    V-loss = expectile_loss(Q(s,a) − V(s), τ)
```

`v1` preset: `~12M params`, BF16 autocast on Blackwell, batch 512 PAs, AdamW lr=3e-4 with 1k-step warmup + cosine decay, IQL τ=0.7, γ=1.0.

## Project layout

```
baseball-rl/
├── CLAUDE.md                # full project spec
├── docs/FILTER_RULES.md     # authoritative drop rules
├── pyproject.toml
├── data/                    # gitignored
├── src/                     # download / filter / splits / verify / tokenize / encoder / dataset / qtransformer / configs / eval / trainer
├── scripts/                 # 01_download → 08_train
└── tests/                   # 65 unit tests across all modules
```

## Five-season dataset (2021–2025) — current state

| Stage | Output |
|---|---|
| Raw pulls | 35 month files, 516 MB |
| Filtered (per season) | 5 × ~700K pitches → **3.46M total pitches across all seasons** |
| Splits (year-level) | train: 2.78M rows / 714K PAs (2021-2024) · val: 397K / 103K PAs (2025 Apr–Jul 15) · test: 287K / 74K PAs (2025 Jul 16–Oct) |
| Vocabularies | 17 pitch types · 13 descriptions · **1,570 pitchers** · **1,448 batters** |
| Lookup tables | `pitcher_arsenal.parquet` 6,985 (pitcher, type) groups · `batter_profile.parquet` 13,290 (batter, type) groups |
| Verify | filter 6/6 PASS · tokens 8/8 PASS |

## Status

- Phases 1–8: complete
- Phase 9 (off-policy evaluation: FQE, weighted IS, distributional checks): TODO

## License

MIT (see `LICENSE`).
