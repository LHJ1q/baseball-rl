# baseball-rl

Offline reinforcement learning for MLB pitch selection.

## What we're building

A **Q-Transformer** that takes the within-PA pitch sequence + game state as input and predicts the best `(pitch_type, plate_x, plate_z)` action for the pitcher.

- **Reward**: `delta_run_exp` from Baseball Savant, negated for the pitcher's perspective (`reward_pitcher = -delta_run_exp`).
- **Action space**: discretized — pitch type (~8 classes) × plate_x (~16 bins) × plate_z (~16 bins), with autoregressive decomposition `Q(s, type) → Q(s, type, x) → Q(s, type, x, z)`.
- **Approach**: offline RL on historical Statcast data, with conservative regularization (CQL or IQL) to handle out-of-distribution actions.

## Why this isn't just Decision Transformer

DT conditions on return-to-go but doesn't explicitly learn a value function — it struggles with credit assignment over long horizons, which matters here because a strike-now / base-hit-later tradeoff requires actually understanding state value. Q-Transformer's autoregressive Q-learning gives proper credit assignment and matches the "pick pitch type, then refine location" structure of how pitchers actually decide.

## Current scope

- **Phases 1–4 (data pipeline) — DONE.** 2024 season filtered to **684,466 pitches / 175,984 PAs**, written to `data/splits/{train,val,test}.parquet` with all six verification checks passing. See `docs/FILTER_RULES.md` for the authoritative drop spec.
- **Phase 5 (tokenization) — DONE.** 8/8 verification checks PASS. Per-pitch tokens + pitcher-arsenal + batter-profile + vocab.json under `data/tokens/`.
- **Phases 6+7 (state encoder + Q-Transformer architecture) — DONE.** `src/encoder.py`, `src/dataset.py`, `src/qtransformer.py`. Strongest smoke test (10 checks) PASS on Macbook with tiny config (130K params).
- **Phase 8 (offline RL trainer) — DONE.** `src/trainer.py`, `src/eval.py`, `scripts/08_train.py`. End-to-end smoke-train (Macbook MPS, 2 epochs on 64 PAs) verified loop, save/load, resume, and refuse-on-collision behaviors. Real training on the Blackwell RTX Pro 4500 is one `git push` away.
- **Multi-year scaling — pending.** Code is year-agnostic; the data download phase needs to be re-run for years 2021–2025 once the model is validated. See "Multi-year extension" below.

Heavy training runs on a separate Linux server / Colab. Macbook PyTorch is fine for the smoke test and for prototyping; full training runs are NOT to happen on Macbook.

---

## Build sequence

Each phase must be working and verified before starting the next.

### Phase 1 — Download (`src/download.py`, `scripts/01_download.py`)

Pull pitch-by-pitch Statcast data via `pybaseball`, month by month, and save raw parquet to `data/raw/statcast_{year}_{month:02d}.parquet`.

- Always `pybaseball.cache.enable()` before any call.
- Always wrap entry points in `if __name__ == "__main__":` (macOS / `ThreadPoolExecutor`).
- Re-sort ascending by `(game_date, game_pk, at_bat_number, pitch_number)` immediately — pybaseball returns descending.
- Range: **2024 regular season only** (April–October). Multi-year scaling comes later, after the pipeline + first model are working end-to-end on a single season.
- Implement a `--dry-run` mode that pulls a single week. Always run dry-run first.

### Phase 2 — Filter (`src/filter.py`, `scripts/02_filter.py`)

Read `data/raw/`, apply filters and add derived columns, write to `data/processed/statcast_{year}.parquet`. **Authoritative drop spec lives in `docs/FILTER_RULES.md` — read that before editing rules or extending to other seasons.**

Summary of rules (full rationale in the docs file):

- All drops are **PA-atomic**. A bad pitch drops its entire PA — never partial sequences.
- Drop `game_type != 'R'` (spring + postseason).
- Drop any PA whose pitches have nulls in `REQUIRED_NONNULL_COLS` (action + reward + release/physics + handedness/IDs + count/state). Derived columns (e.g. `effective_speed`) are deliberately *not* in this list — gate on inputs, not derived features.
- Drop pitchouts (`pitch_type ∈ {'PO', 'UN'}`).
- Drop intentional walks: `description == 'intent_ball'` (pre-2017 era) or `events ∈ {'intent_walk'}` (post-2017).
- Drop `events == 'truncated_pa'` (inning ended administratively mid-PA).
- Drop position-player pitchers: per-pitcher season max `release_speed` over `{FF, SI, FC}` < 80 mph.
- Drop PAs with no terminal pitch (`events.notna().sum() == 0` over the PA — inning ended on basepaths for the 3rd out).

**Add derived columns:**
- `reward_pitcher = -delta_run_exp`
- `prev_pitch_type` (within PA, NaN for first pitch)
- `pitch_idx_in_pa` (0-indexed pitch number within PA)
- `is_terminal` (boolean: last pitch of PA — `events` is non-null). Invariant: exactly one terminal pitch per PA after filtering — `add_derived_columns` asserts this.
- `plate_x_mirrored` — mirror to RHP perspective: `plate_x * (-1 if p_throws == 'L' else 1)`. Other left-right-signed columns (`pfx_x`, `release_pos_x`, `vx0`, `ax`, `spin_axis`) are *not* mirrored at the filter step — they get mirrored later in tokenization (phase 5) so the raw processed parquet stays untouched.

Per-column null counts and per-rule drop counts are logged at runtime.

### Phase 3 — Splits (`src/splits.py`, `scripts/03_split.py`)

Read `data/processed/`, write temporal splits to `data/splits/`. Since we only have 2024 right now, split **within the season by date**:

- `train.parquet`: 2024-04-01 to 2024-08-31 (April–August)
- `val.parquet`: 2024-09-01 to 2024-09-15
- `test.parquet`: 2024-09-16 to end of season

Strictly temporal — never random. Random splitting leaks future player-form information and is the most common mistake in baseball ML papers. When we add more seasons later, this will switch to year-level splits.

### Phase 4 — Verify (`src/verify.py`, `scripts/04_verify.py`)

Mandatory before declaring the pipeline done. Print a report covering all six checks:

1. `delta_run_exp` coverage for 2024 (must be > 95%; flag loudly if below).
2. Total row count (expect ~700K pitches for the full regular season).
3. Pitch-type coverage — `{FF, SL, CH, SI, CU, FC, FS, KC, ST, SV}` together must cover > 95%. (`ST`/`SV` were added on top of CLAUDE.md's original 8 because sweeper/slurve became common post-2023.)
4. Reward sanity: spot-check 10 random PAs — sum of `reward_pitcher` should be coherent with the PA outcome.
5. No PA crosses game boundaries: `df.groupby(['game_pk', 'at_bat_number'])['game_date'].nunique().max() == 1`.
6. Within-PA sort stability: `pitch_number` is strictly increasing within every PA.

### Phase 5 — Tokenize (`src/tokenize.py`, `scripts/05_tokenize.py`, `scripts/06_verify_tokens.py`)

Discretize the action space and emit two artifacts under `data/tokens/`:

**(a) `data/tokens/{train,val,test}.parquet` — per-pitch token rows.** One row per surviving pitch, ascending order preserved. Columns:
- *Action target* (what the policy predicts): `pitch_type_id`, `x_bin`, `z_bin`
- *Decision-time state* (visible before the pitch is thrown): `balls`, `strikes`, `outs_when_up`, `pitch_idx_in_pa`, `sz_top`, `sz_bot`
- *Execution outcome* (what actually happened — visible to the *next* timestep): `plate_x_mirrored`, `plate_z`, `description_id`, `reward_pitcher`, `is_terminal`
- *PA static fields* (denormalized onto every pitch row for ease of joining): `game_pk`, `at_bat_number`, `game_date`, `batter_id`, `pitcher_id`, `p_throws`, `stand`, `inning`, `inning_topbot`, `home_score`, `away_score`, `bat_score`, `fld_score`, `on_1b` (bool), `on_2b` (bool), `on_3b` (bool), `n_thruorder_pitcher`

**(b) `data/tokens/pitcher_arsenal.parquet` — static `(pitcher_id, pitch_type_id)` lookup**. Computed on **train split only** (no val/test leakage). Columns: `count`, mean and std of `release_speed`, `release_spin_rate`, `spin_axis_mirrored`, `pfx_x_mirrored`, `pfx_z`, `release_extension`. Rows with `count < N_MIN_ARSENAL_SAMPLES` are flagged so the trainer can fall back to a global mean. **The trainer derives the per-pitcher repertoire mask from this table** (see "Repertoire-aware action masking" below).

**(c) `data/tokens/batter_profile.parquet` — per-`(batter_id, pitch_type_id)` scouting table.** Computed on train only. Carries per-(batter, pitch_type) `count`, `swing_rate_vs_type`, `whiff_rate_vs_type`, plus the batter's overall stats denormalized in: `pa_count`, `k_rate`, `bb_rate`, `xwoba_mean`, `swing_rate`, `whiff_rate`, `contact_rate`, `chase_rate`. `low_sample` flag at `count < N_MIN_BATTER_PROFILE_PAS`. Symmetric to pitcher_arsenal — joined at the dataloader stage to give the model batter context without bloating per-pitch tokens.

**(d) `data/tokens/vocab.json`** — single source of truth for every discrete vocabulary: `pitch_type_vocab`, `description_vocab`, `inning_topbot_vocab`, `p_throws_vocab`, `stand_vocab`, `batter_id_vocab`, `pitcher_id_vocab`, plus `x_bin_edges` and `z_bin_edges` (the np.linspace endpoints used to discretize location).

**(e) `data/tokens/feature_stats.json`** — per-column mean/std for the model's continuous inputs (pre-action state, post-action outcome, batter profile overall stats), computed on train only. Loaded by `src.configs.build_qtransformer` and passed to the encoders as frozen standardization buffers — keeps continuous features at roughly zero mean / unit variance before the MLP.

**Action discretization:**
- `pitch_type_vocab` enumerates every `pitch_type` that survived filtering on the train split, sorted by frequency descending. No "OTHER" bucket.
- `x_bin`: 20 uniform bins over `plate_x_mirrored ∈ [-2.5, +2.5]` ft (~3-inch resolution; covers chase / waste pitches and most wild misses).
- `z_bin`: 20 uniform bins over `plate_z ∈ [-1.0, +6.0]` ft (~4.2-inch resolution). Range chosen to capture both bounced splitters/curveballs (which on 2024 land below 0 ft for ~6-12% of CU/FS/KC pitches) and head-high setup fastballs.
- Pitches outside the box are clamped to the edge bin; on 2024 the clamp rate is < 0.5% on each axis.

**Mirror-for-LHP step (lives in tokenizer, not filter):** any column with a left-right sign convention gets flipped when `p_throws == 'L'`. That's `plate_x → plate_x_mirrored` (already done in the filter), plus newly mirrored `pfx_x_mirrored`, `release_pos_x_mirrored`, `vx0_mirrored`, `ax_mirrored`, `spin_axis_mirrored = (360 - spin_axis) % 360`. The processed parquet stays untouched; mirroring is a tokenizer-time transform.

**Why pitch features live in `pitcher_arsenal.parquet`, not in every pitch row:** for a given `(pitcher_id, pitch_type)` the release speed / spin / movement / extension are approximately constant — the per-pitch noise is small relative to between-pitcher variation. So including raw physics on every per-pitch token is mostly redundant with the pitcher embedding (added in a later phase). Storing the per-`(pitcher, pitch_type)` mean+std once gives the model the same signal at a fraction of the storage and zero data-leakage risk. The genuine per-pitch variance — *where it actually landed* (`plate_x_mirrored`, `plate_z`) and *what the batter did* (`description_id`) — stays on the per-pitch token, since those are what reward depends on.

**`scripts/06_verify_tokens.py` — must pass before declaring phase 5 done:**

1. Every `(pitch_type_id, x_bin, z_bin)` is in-vocab and in-range.
2. Every PA has `is_terminal == True` exactly once and only on the last pitch.
3. `pitch_idx_in_pa` runs `0..len(PA)-1` with no gaps and matches token-file row order within each PA.
4. Per-PA sum of `reward_pitcher` matches the pre-tokenization processed parquet (exact equality, sanity on mirroring not corrupting reward).
5. Mirror invariant: marginal distribution of `plate_x_mirrored` for original-LHP rows is statistically indistinguishable from original-RHP rows (KS test or visual side-by-side).
6. No nulls in any token-file column.
7. `pitcher_arsenal.parquet` covers ≥ 95% of `(pitcher_id, pitch_type_id)` pairs that appear in train; the rest are flagged with `count < N_MIN_ARSENAL_SAMPLES` for the trainer's fallback path.
8. `batter_profile.parquet` covers ≥ 95% of train batters; per-`(batter, pitch_type)` rows below `N_MIN_BATTER_PROFILE_PAS` are flagged `low_sample`.

**Repertoire-aware action masking (consumed by phase 6/7, not phase 5).** The Q-Transformer must not assign positive Q to pitches a pitcher physically cannot throw. The trainer derives a per-pitcher repertoire mask from `pitcher_arsenal.parquet`:

```
repertoire(pitcher_id) = { pitch_type_id : arsenal[(pitcher_id, pitch_type_id)].count >= N_MIN_REPERTOIRE }
```

…and masks `Q(s, pitch_type=t)` to `-inf` for `t ∉ repertoire(pitcher_id)` before argmax. This is a hard physical constraint complementary to CQL/IQL conservatism, which handles the softer OOD cases. The mask is computed at dataloader time — no new pipeline artifact needed.

### Phase 6 — State encoders (`src/encoder.py`)

Two encoders, each mapping per-pitch features to a `d_model` vector:

- **`PreActionEncoder`** — encodes the *decision-time state* at pitch `i` (count, outs, runners, score, batter, pitcher, sz_top/sz_bot, batter overall scouting profile). No action or outcome info — would leak the prediction target.
- **`PostActionEncoder`** — encodes the action target + execution outcome at pitch `i` (pitch_type/x_bin/z_bin, actual `plate_x_mirrored`/`plate_z`, description, reward, terminal flag). Used as past-pitch context for future decisions.

Action embedding tables (`pitch_type`, `x_bin`, `z_bin`) are SHARED between the post-action encoder and the Q-heads — single source of truth.

The dataset module (`src/dataset.py`) groups pitches into PAs, joins `batter_profile` overall stats per `batter_id`, and produces `PABatch` objects with attention/PA-length masks ready for the model.

### Phase 7 — Q-Transformer (`src/qtransformer.py`)

`QTransformer` runs a causal transformer over an interleaved sequence per PA:

```
[pre_0, post_0, pre_1, post_1, ..., pre_{T-1}, post_{T-1}]   (length 2T)
```

A causal mask + `role_emb(0=pre, 1=post)` ensures that the encoded vector at position `pre_i` attends to every prior pitch's full `(pre, post)` tokens but to nothing of pitch `i`'s own action/outcome — exactly the state representation needed for `Q(s_i, a_i)`.

**Heads (per-(pitcher, pitch_type) and per-(batter, pitch_type) joins applied):**
- `q_head_type(h_pre, arsenal[pitcher, t], batter_pt[batter, t])` → logits over pitch types. h_pre is broadcast over the type axis; for each candidate type, the head sees that pitcher's stuff for that pitch type and that batter's tendencies vs that pitch type.
- `q_head_x(h_pre, type_emb, arsenal[chosen_type], batter_pt[chosen_type])` → logits over x bins.
- `q_head_z(h_pre, type_emb, x_emb, arsenal[chosen_type], batter_pt[chosen_type])` → logits over z bins.
- `v_head(h_pre)` → scalar V(s) for IQL (state-only — no per-action conditioning).

The joined features come from `data/tokens/pitcher_arsenal.parquet` and `data/tokens/batter_profile.parquet` via `src.dataset.PitchPADataset`, which builds dense `(n_pitchers, n_pitch_types, k)` and `(n_batters, n_pitch_types, k)` lookup tables at startup. Missing entries (UNK or never-seen-in-train) get `low_sample=1.0` and zeros — the model can detect "stranger" via the flag and fall back to the pitcher/batter embedding alone.

`heads_chosen()` evaluates per-head Q on the chosen action triple for training. `policy()` does autoregressive argmax for inference (17 + 20 + 20 = 57 evaluations vs the 17 × 20 × 20 = 6800 joint).

**Repertoire-aware action masking** (called from `policy()`): the trainer/inference code derives `repertoire(pitcher_id) = {pitch_type : pitcher_arsenal.count >= N_MIN_REPERTOIRE}` (helper in `src/qtransformer.py:build_repertoire_mask`) and passes a `(B, T, n_pitch_types)` bool mask to `policy()`. Non-repertoire types are masked to `-inf` before argmax. Pitchers with no arsenal entries fall back to "all allowed" so the model can act.

**IQL loss helper (`iql_losses`)**:
- `q_loss = MSE(Q(s, a), r + γ · V(s') · (1 − terminal))` — TD with V, no max over actions (IQL's key trick).
- `v_loss = expectile_loss(Q(s, a) − V(s), τ)` — asymmetric L2 with weight `τ` for positive residuals.
- The actual training loop / optimizer is phase 8.

### Phase 7b — Smoke test (`scripts/07_smoke_test.py`)

Same code path as the v1 preset — selectable via `--preset {smoke, v1}` (default `smoke`). Tiny config (`d_model=64`, `n_layers=2`, batch of 4 PAs) exercises every code path on Macbook CPU/MPS in seconds. Ten checks: forward shapes, NaN/Inf, causal mask correctness (perturb-and-check), PA-batch independence, repertoire mask correctness, batch tensor sanity, IQL loss finite, overfit single batch (loss drops > 50%), argmax determinism, save/load round-trip. Both presets validated to pass all 10 checks.

### Phase 7c — Configs and presets (`src/configs.py`)

Presets, JSON I/O, ad-hoc overrides, and a one-liner model builder. The training pipeline reads from disk artifacts (`data/tokens/vocab.json` + `feature_stats.json`) and uses a named preset:

```python
from src.configs import build_qtransformer
model = build_qtransformer(tokens_dir, preset="v1")  # or "smoke"
# ad-hoc override: build_qtransformer(tokens_dir, preset="v1", overrides={"n_layers": 8})
# config file:    build_qtransformer(tokens_dir, preset=load_from_json(path))
```

Both encoder and Q-Transformer configs are dataclasses (`EncoderConfig`, `QTransformerConfig`); JSON files dump `{"encoder": {...}, "qtransformer": {...}}` for clean checkpointable run configs.

### Phase 8 — Offline RL trainer (DONE)

`src/trainer.py`, `src/eval.py`, `scripts/08_train.py`. Components:

- PA-grouped dataloader from `src/dataset.py` with worker prefetch (`num_workers`).
- BF16 autocast on CUDA (auto-falls-back to fp32 on CPU/MPS — Macbook smoke uses fp32).
- AdamW + linear-warmup-then-cosine-decay LR schedule (`cosine_warmup_lr`).
- IQL Q+V loss (TD on Q with V bootstrap; expectile regression on V). Loss helper in `src/qtransformer.py:iql_losses`.
- Gradient clipping at `cfg.grad_clip` (default 1.0).
- Checkpointing: `checkpoint_latest.pt` every epoch, `checkpoint_epoch_{N}.pt` periodic, `checkpoint_best.pt` on best val Q-loss.
- Auto-resume: `--resume` finds `checkpoint_latest.pt`; refuses to overwrite an existing run dir without `--resume`; `--no-resume` rejects existing dirs entirely.
- CSV logging of per-step train metrics + per-epoch eval metrics in `data/runs/{run_name}/metrics.csv`. No external logging deps for v1; W&B integration is ~20 LOC if/when needed.
- Eval pass on val split: Q-loss, V-loss, plus **pitcher-blind variants** (zero pitcher embedding before forward) and the gap = blind − personalized — diagnostic for "is the model relying on player identity vs general rules" (the answer informs whether to enable embedding dropout in a follow-up run).
- **OPE / FQE / weighted IS deferred to phase 9.**

CLI:
```
python scripts/08_train.py \
    --preset v1 \
    --run-name iql_baseline_2021_2024 \
    --epochs 40 \
    --batch-size 512

# Macbook smoke (small preset, tiny subset, 2 epochs, no BF16)
python scripts/08_train.py --smoke-train --run-name smoke_test
```

Training output one-liner per epoch:
```
epoch  12 | step  4800 | train q=0.0345 v=0.0118 | val q=0.0398 v=0.0142 | blind q=0.0421 gap=+0.0023 | 28.7s
```

The `gap` column is the pitcher-blind eval signal: bigger = more pitcher-specific behavior; near zero = model leans on general rules.

**v1 preset (recommended for the 2021-2025 scaled training run on Blackwell RTX Pro 4500):**

| Knob | v1 value | Notes |
|---|---|---|
| `d_model` | 384 | |
| `n_layers` | 6 | |
| `n_heads` | 8 | head_dim = 48 |
| `d_ff` | 1536 | 4 × d_model |
| `dropout` | 0.1 | |
| `d_player_emb` | 96 | shared dim for pitcher and batter |
| `d_pitch_type_emb` | 32 | |
| `d_description_emb` | 16 | |
| `d_action_loc_emb` | 24 | x_bin / z_bin |
| Total params | ~12M | dominated by transformer body (~10.5M); embeddings ~150K |
| `max_seq_len` | 64 | covers 2 × max_PA_pitches (16) with margin; learned absolute position embeddings of this size are added to the interleaved sequence |
| Optimizer | AdamW (β=(0.9, 0.95), wd=0.01) | |
| LR | 3e-4 with 1000-step warmup, cosine decay | |
| Batch | 512 PAs / step | ~2K-5K tokens/step; fits in 24 GB at BF16 |
| Mixed precision | BF16 autocast | no GradScaler needed (BF16 has FP32 dynamic range) |
| Gradient clip | 1.0 | |
| γ (discount) | 1.0 | undiscounted; PAs terminate within ~10 steps and rewards are in run units |
| IQL τ (expectile) | 0.7 | scan {0.7, 0.8, 0.9} if eval is poor |
| IQL β (AWR temperature) | 3.0 | scan 3-10 if needed |
| Epochs | 40 (start) | early-stop on val Q-loss + OPE |
| Wall-clock estimate | ~30 min/epoch, ~20 hours total | overnight on Blackwell |

### Multi-year extension (when ready to scale beyond 2024)

Plan: train on 4 full clean seasons **2021–2024**, hold out 2025 split for val/test (val = 2025 first half, test = 2025 second half).

What needs doing when ready:
- `scripts/01_download.py` — call `pull_season(year=Y)` for each Y. The script already supports any year.
- `scripts/02_filter.py` — concatenate raw files across years; filter rules unchanged.
- `scripts/03_split.py` — **add `--scheme {within_season, year_level}` CLI flag** (currently only within-season is implemented). Year-level scheme: train = 2021-2024 full, val = 2025-04-01..2025-07-15, test = 2025-07-16..end. Code change is small.
- `scripts/05_tokenize.py` — same code; vocab sizes grow (more pitchers/batters); arsenal + profile recomputed on the bigger train.
- Phase 6/7 modules are vocab-size-agnostic — `vocab_sizes` from `vocab.json` flows into `QTransformer.__init__`. **No model code change.**

The `--scheme year_level` flag is the only actual code change required; defer until the 2021-2023 + 2025 raw data is pulled.

### Future tokenization improvements (deferred)

Open follow-ups that didn't block phase 5 shipping but are worth doing before training scales beyond v1:

- **Add `re24_at_pitch_start` derived feature.** Compute the 24-row base-out → run-expectancy lookup from train data, join onto each token row as a single float. Compact summary of "how many runs is this PA worth right now" — useful as a fast-path signal alongside the raw `(inning, score, runners, outs)` fields. Costs one float per token and a tiny lookup table.
- **Collapse score representation.** Currently we carry four score fields (`home_score`, `away_score`, `bat_score`, `fld_score`); only `score_diff = bat_score - fld_score` is strategically informative. Drop the redundant three.
- **Per-pitcher within-game degradation features** (only if v1 trainer shows pitcher embeddings can't pick this up): residual = (raw `release_speed` for this pitch) − (arsenal mean `release_speed` for this `(pitcher, pitch_type)`). Captures fatigue / form. Adds storage; only worth it if we observe the embedding missing it.
- **Strike-zone-normalized `z_bin`** (only if uniform bins underperform): map `plate_z` to `(plate_z - sz_bot) / (sz_top - sz_bot)` so each batter's strike zone occupies the same bin range. Trades simplicity for batter-relative resolution.

- **Sweeper (`ST`) / Slider (`SL`) label drift in pre-2023 data.** Statcast introduced the `ST` class around 2022-2023 and *retroactively* relabeled some pre-2023 sliders as sweepers — but the retroactive backfill was **selective, not exhaustive**. A 2021 `SL` row therefore *might* be a pitch that today's classifier would call `ST` but Savant never updated. Empirical effect is small (the model has `pfx_x_mirrored` / `pfx_z` / `release_spin_rate` as physics features and the pitcher arsenal absorbs per-pitcher physics regardless of label), but it's a real noise source for the breaking-ball category in 2021-2022. If a v1 trainer shows ST-specific weirdness or a year-over-year distribution shift on `SL`, the cheapest mitigation is to **collapse `ST` into `SL`** in tokenization (treat them as one class, lose the modern sweeper-vs-slider strategic distinction). Other options: train only on 2023+ (lose ~40% of data) or build a physics-based reclassifier for pre-2023 (high effort, ambiguous ground truth). Don't preempt — train as-is and only intervene if eval shows it matters.
- **Replace (or augment) `pitcher_arsenal.parquet` and `batter_profile.parquet` with learned representations.** The current artifacts are precomputed *statistical* features (mean release_speed, K%, swing rate, etc.) — interpretable, fast, and useful as a v1 baseline. A natural next step is to train a separate encoder that maps `pitcher_id`/`batter_id` to a dense embedding from raw pitch-by-pitch trajectories (autoencoder, contrastive, or BERT-style masked-pitch objective on the same Statcast data). Possible designs:
  - **Replace** the lookup tables with a player encoder, freeze the embeddings, feed them to the Q-Transformer.
  - **Augment**: keep the statistical lookups (provide a strong, interpretable prior) and concatenate the learned embedding alongside.
  - **Bootstrap from public work** like Heaton & Mitra player embeddings if their pretrained vectors are available — saves the encoder-training phase.
  Decide between replace / augment / bootstrap once the v1 Q-Transformer is trained and we can see whether the statistical features are a bottleneck. Until then, the data pipeline doesn't change — switching to embeddings is an additive change in the model phase.

- **Swapping in pretrained pitcher/batter embeddings.** The current `nn.Embedding(n_players, d_player_emb)` tables in `PreActionEncoder` are random-init and trained end-to-end with the Q-loss. They can be **replaced or augmented with pretrained vectors** at any point without touching the rest of the architecture. The arsenal/profile joins at the Q-heads are a **separate channel** and keep working regardless of where the embeddings come from — they're insurance against any embedding failure mode (e.g. pretrained vectors that are mediocre on rare players still get backed up by "his FF is 96 mph" from the arsenal).

  Three regimes, all supported by the existing architecture:

  1. **Frozen pretrained** — load vectors into `model.pre_encoder.emb_pitcher.weight.data`, set `.requires_grad = False`. The rest of the model adapts to a fixed representation space.
  2. **Pretrained init + fine-tune** — same load step, leave `requires_grad = True`. Pretrained vectors are the starting point; the Q-loss refines them for pitch selection.
  3. **Augment** — add a second embedding table loaded from the pretrained tensor (frozen) alongside the existing learned one; concatenate before the projection. Encoder input dim grows by the pretrained dim. ~5 LOC change in `PreActionEncoder.__init__` and `.forward`.

  Three sources, ordered by effort:

  | Source | Effort | Notes |
  |---|---|---|
  | **Heaton & Mitra** (or other published) | Low (~30 LOC) | One-time MLBAM-ID → `vocab.json` integer-ID remap via pandas merge. |
  | **Self-pretrained** (autoencoder / contrastive / masked-pitch on Statcast tokens) | Medium (~300 LOC for a separate training script) | Output dim controllable; can match `d_player_emb` exactly. |
  | **Two-stage on this model** | Trivial | Run phase 8 → save learned embeddings → use as init for next run on more data. |

  Compatibility checks at load time: (a) pretrained tensor shape = `(n_pitchers, d_player_emb)` — if dims differ, edit `EncoderConfig.d_player_emb` in the preset and the architecture rebuilds (arsenal/profile joins unchanged); (b) player IDs must be re-keyed to our `vocab.json` integer-ID space.

  Don't preempt this — switching is purely additive and can happen mid-project once we know whether the random-init embeddings are a bottleneck.

---

## Domain conventions (read once, apply everywhere)

- **Pitch ordering**: `(game_date, game_pk, at_bat_number, pitch_number)`, ascending.
- **PA boundaries**: `(game_pk, at_bat_number)`. Never mix PAs across games.
- **Coordinate frame**: `plate_x` and `plate_z` in feet, batter's perspective, `plate_z=0` at ground level. Use `plate_x_mirrored` for the model so left-handed pitchers don't flip the inside/outside frame.
- **Statcast is mutable** — Savant updates historical rows. Re-pulling produces small diffs. Don't panic.

---

## Project layout

```
baseball-rl/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── docs/
│   └── FILTER_RULES.md   # Authoritative drop spec — read before extending to other seasons
├── data/                 # gitignored
│   ├── raw/              # Per-month parquet, untouched scrape
│   ├── processed/        # Filtered + derived columns, per-season
│   ├── splits/           # train.parquet / val.parquet / test.parquet
│   ├── tokens/           # train/val/test.parquet + pitcher_arsenal.parquet + batter_profile.parquet + vocab.json + feature_stats.json
│   └── runs/             # one subdirectory per training run (config.json, metrics.csv, checkpoints)
├── src/
│   ├── __init__.py
│   ├── download.py
│   ├── filter.py
│   ├── splits.py
│   ├── verify.py
│   ├── tokenize.py
│   ├── encoder.py
│   ├── dataset.py
│   ├── qtransformer.py
│   ├── configs.py
│   ├── eval.py
│   └── trainer.py
├── scripts/
│   ├── 01_download.py
│   ├── 02_filter.py
│   ├── 03_split.py
│   ├── 04_verify.py
│   ├── 05_tokenize.py
│   ├── 06_verify_tokens.py
│   ├── 07_smoke_test.py
│   └── 08_train.py
└── tests/
    ├── test_filter.py
    ├── test_tokenize.py
    ├── test_encoder.py
    ├── test_qtransformer.py
    ├── test_configs.py
    ├── test_eval.py
    └── test_trainer.py
```

`data/` is gitignored. Raw parquet for 2024 alone is ~115 MB.

---

## Code style

- Type hints on all public functions.
- One responsibility per module. `download.py` only downloads. No reaching across.
- `logging` (not `print`) for anything that runs > 10 seconds.
- `pyarrow` engine for parquet.
- No notebooks checked in — scratchpads, not deliverables.
- Boring, readable code. No clever pandas chains.

---

## What is OUT of scope right now

- **Phase 9 — Off-policy evaluation framework** (FQE, weighted IS, distributional analysis vs behavior, reward decomposition by count/handedness/pitcher tier). This is the next phase to build after we have a trained model.
- Pretrained or external player embeddings (Heaton & Mitra, etc.) — see "Future tokenization improvements" for the path.
- CQL implementation (we chose IQL for v1).
- Pitcher-embedding dropout *during training* (the "general rule maker" booster). Plumbed as `TrainerConfig.pitcher_dropout=0.0` so it's available; user opted to try it later.
- Heavy training runs on this Macbook. PyTorch is installed locally for `--smoke-train` only; anything that needs a GPU or > a few minutes of CPU time is out of scope here.

If a task seems to require any of the above, **stop and ask** rather than scope-creep.

---

## Hard rules for the agent

1. **Never start a multi-month download without a dry-run first.** Pull one week, verify columns and `delta_run_exp` coverage, then scale to the full 2024 season.
2. **Always check `delta_run_exp` is present and dense before computing the reward.** If it's sparse, stop and report — do not silently fall back to anything.
3. **Never modify files outside this repo.**
4. **Never `rm -rf data/`** without asking, even on a "clean restart."
5. **If a Savant pull fails repeatedly, log and skip — do not retry indefinitely.** Report skipped ranges at the end.
