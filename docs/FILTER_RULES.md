# Filter rules

Authoritative spec for what `src/filter.py` drops from raw Statcast pulls and why. **Re-read this before extending the pipeline to additional seasons** — every rule here was added because a real anomaly tripped it on 2024 data, and most will recur in other seasons. Implementation lives in `src/filter.py::apply_filters`.

## Order of operations

1. **Game-level drop** — `game_type != 'R'`. Removes spring training, postseason, and All-Star data wholesale. Applied first so per-PA aggregations later (e.g. position-player heuristic, no-terminal detection) operate on the regular-season subset only.
2. **Identify position-player pitchers** (per-pitcher aggregation, see rule 5 below). Done before the row-level masks because it produces a *pitcher set*, then row-level masks reference it.
3. **Compute row-level "bad" masks** (rules 2–5 below). Each is logged with its row count.
4. **Compute the no-terminal-PA set** (rule 6 below).
5. **Union** all bad rows' PAs with the no-terminal-PA set → final drop set.
6. **Drop** every pitch whose `(game_pk, at_bat_number)` is in the drop set. **PA-atomic** — never partial.

After filtering, `add_derived_columns` enforces the strict invariant that **every surviving PA has exactly one terminal pitch** (`events.notna().sum() == 1`). If that ever fails on a new season, a new drop rule is needed — do not loosen the invariant.

---

## Rule catalogue

### 1. Non-regular-season games (`game_type != 'R'`)
- **What:** drop every pitch from spring training (`'S'`), exhibition (`'E'`), All-Star (`'A'`), postseason (`'F'`/`'D'`/`'L'`/`'W'`).
- **Why:** training distribution must match the regulation environment we want the policy to act in. Spring rosters and postseason matchups are not representative.
- **Atomicity:** PA-atomic by construction (game_type is per-game).

### 2. Any null in a model-required column (`REQUIRED_NONNULL_COLS`)
- **What:** drop the entire PA if **any** pitch in it has a null in any column the Q-Transformer consumes — action (`pitch_type`, `plate_x`, `plate_z`), reward (`delta_run_exp`), full release / pitch-physics block (`release_speed`, `release_pos_*`, `release_spin_rate`, `spin_axis`, `release_extension`, `pfx_x/z`, `vx0/vy0/vz0`, `ax/ay/az`, `sz_top/sz_bot`), handedness/IDs (`p_throws`, `stand`, `batter`, `pitcher`), and count/state (`balls`, `strikes`, `outs_when_up`, `inning`, `inning_topbot`, `description`, `type`, `zone`, `home_score`, `away_score`, `bat_score`, `fld_score`, `n_thruorder_pitcher`).
- **Why:** the user mandate is "every PA is fully supported" — no row in a kept PA can have a null in a column the model uses. The biggest contributors on 2024 were `release_spin_rate`/`spin_axis` (~5.5K rows each, mostly Hawk-Eye spin-tracking failures) and missing pitch-physics blocks (~2.5K rows when no tracking at all).
- **Amplifies to PA-atomic drop** by definition.
- **Informative nulls — *deliberately excluded* from `REQUIRED_NONNULL_COLS`:**
  - `events`: null on every non-terminal pitch by design
  - `on_1b/2b/3b`: null = no runner on that base
  - Batted-ball fields (`launch_speed`, `launch_angle`, `hit_distance_sc`, `bat_speed`, `swing_length`, `estimated_*`): null on non-batted / non-swing pitches
  - Score-diff helpers and other derived columns we don't actually consume
- **Derived columns are *not* in `REQUIRED_NONNULL_COLS`** — even if nominally non-null. Example: `effective_speed` is a function of `release_speed` and `release_extension`; gating on it would amplify drops without adding any information that isn't already captured by its inputs. Recompute derived features in the model/tokenizer rather than gating the dataset on them.
- **Re-evaluate per season:** if `release_spin_rate` null rate exceeds ~2% in a future season, investigate Hawk-Eye coverage before silently dropping the volume.

### 3. `pitch_type` value in `{'PO', 'UN'}`
- **What:** rows whose `pitch_type` is `'PO'` (pitchout) or `'UN'` (unknown). Note: null `pitch_type` is already handled by rule 2.
- **Why:** pitchouts are a manager's strategic signal-call, not a stuff-vs-batter decision; unknowns are sensor failures with bogus everything-else.
- **Amplifies to PA-atomic drop**.

### 4. Intentional walks
Two paths into this rule depending on era:
- **`description == 'intent_ball'`** — pre-2017, when intentional walks were actually pitched as four lobs. Each row tagged `intent_ball`.
- **`events ∈ {'intent_walk'}`** — post-2017 auto-IBB rule. The PA is recorded with no actual pitches thrown, or with a placeholder pitch tagged `events == 'intent_walk'` on the (single) terminal row. (Empirically zero in 2024 because Statcast appears to omit the no-pitch IBB PAs from pitch-by-pitch entirely; rule still fires for older seasons.)

**Why** (both paths): not a real pitcher decision under the policy we're training — it's a manager's strategic call. Keeping them would teach the policy that throwing the ball intentionally outside the zone is sometimes optimal.

**Amplifies to PA-atomic drop**.

### 5. Position-player pitching (per-pitcher heuristic, **non-trivial**)
- **What:** drop every pitch thrown by any pitcher whose **season-level max `release_speed` over fastball types `{FF, SI, FC}` is < 80 mph**.
- **Why:** position players brought in to mop up blowouts are not throwing real pitches and would skew speed/spin/location distributions. CLAUDE.md prescribes "release_speed < 75 from non-pitchers" but Statcast doesn't carry a per-row position field, so we use a season aggregate. The threshold is FB-specific so a real reliever's slow changeup doesn't get incorrectly flagged.
- **Implementation note:** the aggregation must be over the post-`game_type=='R'` frame, otherwise spring-training data dilutes the speed signal. Threshold lives in `src/filter.py::POSITION_PLAYER_FB_MAX_MPH`.
- **Amplifies to PA-atomic drop** so we never carry partial PAs from these outings.
- **Re-evaluate per season:** if Statcast ever exposes a clean `pitcher_role` field, switch to that — this heuristic exists only because the data doesn't.

### 6. PAs with `events ∈ {'truncated_pa'}`
- **What:** drop the entire PA if any pitch's `events` is `'truncated_pa'`.
- **Why:** the PA was administratively closed (game/inning ended mid-at-bat, weather-shortened, etc.) — not a real PA outcome the policy can learn from. On 2024: 309 such PAs (807 rows, all with `outs_when_up == 2` on the final pitch).
- **Amplifies to PA-atomic drop** (rule lives in `DROP_EVENTS` alongside `intent_walk`).

### 7. PAs with no terminal pitch (`events.notna().sum() == 0` over the PA)
- **What:** drop the entire PA whenever no row in that PA has a non-null `events` value.
- **Why:** these PAs ended on a **non-pitch event for the 3rd out of an inning** — typically caught-stealing or a pickoff. The plate appearance never resolved on a pitch, so:
  1. The pitcher's terminal `delta_run_exp` (and therefore terminal reward) for that episode is missing.
  2. There's no clean `is_terminal` pitch to bootstrap from in Q-learning.
- Empirically rare on 2024: **103 PAs out of 178,604 (~0.058%)**, and 100% of them had `outs_when_up == 2` on the final recorded pitch (smoking gun for the inning-ended-on-basepaths interpretation).
- **Atomicity:** PA-atomic by definition.
- **Re-evaluate per season:** if the count ever exceeds ~0.5% of PAs in a future season, stop and investigate before silently dropping that volume — it would mean either a data-pull issue or a rule schema change.

---

## Adding a new rule

When extending to additional seasons, if you discover a new anomaly:

1. Reproduce on a small slice in `scratch/` (e.g. `scratch/inspect_*.py`).
2. Confirm whether the rule is **row-level** (drop a pitch) or **PA-level** (drop a PA directly). Row-level rules feed into the PA-atomic union and amplify to whole-PA drops automatically.
3. Add the mask to `apply_filters` with a `logger.info` line so the row count is visible at runtime.
4. Add a test in `tests/test_filter.py` mirroring the rule's intent (positive + negative case).
5. Update this document with: what / why / atomicity / re-evaluation criteria.
6. Re-run the pipeline end-to-end and check the verify report.

## Things that are *not* dropped (intentional)

- Pitches with extreme but plausible `release_speed` values (e.g. a real 105 mph fastball). We don't gate on physical bounds — Statcast already cleans these.
- Pitches with sparse `delta_run_exp` rows within an otherwise-good PA. CLAUDE.md hard rule #2 says "if `delta_run_exp` is sparse, stop and report — do not silently fall back." We rely on the verify report's check #1 to catch any season where coverage drops below 95%.
- Game-state outliers (extra innings, blowouts). The policy needs to learn these contexts.
