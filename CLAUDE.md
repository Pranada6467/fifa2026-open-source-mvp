# CLAUDE.md — FIFA 2026 Live Calibration Engine

Read this first. It is the entry point for any new session.

## What this is
A World Cup 2026 prediction system built as a learning project. It rates teams
from historical international results, turns that into per-match and
full-tournament probabilities, and runs as a **live loop** that re-scores its own
predictions and recalibrates after every result. The product is **honest
calibration** ("when I say 70%, it happens ~70% of the time") plus a **technique
leaderboard** ranking models by out-of-sample log-loss / Brier / RPS.

Full context: `docs/PLAN.md` (the approved engineering plan — source of truth),
`docs/DESIGN.md` (the why), `docs/block-v-findings.md` (verified facts).

## Current status (2026-06-20)
**Block V (verification spike): COMPLETE — all 6 items pass, V4 user-verified.**
**Block 0 (calibration spine): COMPLETE (T1–T8).**
**Block 1 (tournament simulator): COMPLETE (T9–T11).**
**Block 2 (technique ladder): COMPLETE (T12–T14). The build is feature-complete;
from here the project is OPERATING the loop, not building it.**

**Accuracy plan (2026-06-19/20): mechanism + 5/5 gate verdicts shipped.** Plan
file `~/.claude/plans/encapsulated-sparking-dove.md`; consolidated outcomes in
`docs/tuning/PHASE_OUTCOMES.md`. What landed in production:
- **Phase 0 — T1+T2:** `[bayesian]` extras now installed on CI nightly;
  `hierarchical_poisson` joins the live roster; per-model 5-min SIGALRM
  timeout wrapper in `orchestrate._fit_roster` (D4) drops slow entrants
  without killing the loop.
- **Phase 4 — S25 temperature + S20 isotonic calibration:** new
  `src/fifapreds/calibration/` package; LOTO discipline (D6) over the
  backtest; publisher emits `leaderboard.parquet` with a new
  `calibration ∈ {raw, temperature, isotonic}` column (distinct from the
  E2 `track ∈ {backtest, live}` column to avoid collision). Viewer hero
  has a pill toggle (DD2) anchored above the verdict, with an
  "experimental, n≈64 holdout" caption for calibrated tracks.
- **Phase 5a — S22 BMA:** `src/fifapreds/ensemble/{bma,stacking}.py`.
  `BMAEnsemble` + `BMAGoalsEnsemble` both join the live roster when
  `data/backtest.db` has history for their members; LOTO-derived
  softmax weights at construction (`exp(-L_i/T)`, T=1). Per D5/D7 the
  goals ensemble raises at `__init__` on a non-goals member.
- **Phase 5b — S19 stacking:** mechanism shipped (`StackedEnsemble` +
  `scripts/train_stacker.py` + `docs/stacker_refresh.md`) but the gate
  FAILED (gap +0.0076 vs SE 0.0170); `data/stacking_weights.json` is
  absent and the orchestrator's stacking entrant skips with a
  structured note. Closest near-miss; revisit after WC2026 adds a 4th
  tournament.
- **Phase 5c — Item 11 modal scoreline:** `modal_scoreline_from_grid` +
  `modal_scoreline_label` in `publish/board.py`. `upcoming.parquet` /
  `scoreline_topn.parquet` gain `modal_h` + `modal_a` columns. Viewer
  match-odds hero now shows `{home} {modal_h}–{modal_a} {away}` with a
  small-gray explainer referencing the argmax (which stays in the audit
  expander). 71% of WC2026 fixtures get a different headline.
- **Phase 5d — DD4 divergence banner:** `divergence_banner()` helper +
  amber `st.warning` in the calibration hero when live freq escapes
  the Wilson interval around the calibrated `p_mean`. Mid-tournament
  D10 policy: weights stay frozen, drift gets surfaced honestly.

**Five single-knob gates FAILED** at the > 1 SE bar (gap-vs-SE in
parens): S6 ν tune (0.11), S1 DC tournament weighting (0.13), S2 NegBin
(0.20), S9 BivariatePoisson (0.19), S19 stacking (0.45). The wrapper
classes + tuning scripts + gate docs all SHIP; the roster wiring does
NOT. Pattern is consistent with the eng-review outside voice: at ~64
graded matches per LOTO holdout the bootstrap SE is ±0.02–0.04, larger
than any single-knob lift produces. The variants live in
`src/fifapreds/models/{negbin,bivariate}.py`, `roster.py`'s
`DixonColesTournamentWeighted`, and the `scripts/tune_*` scripts —
anyone can instantiate them; the gate decides whether they ship to
`default_roster()`.

**Phase 3 — S13 Transfermarkt: DEFERRED** (`docs/tuning/transfermarkt_deferred.md`).
Needs a manual snapshot scrape from a fragile page + the variant's
"validation is it didn't crash" caveat made the gate unfair. Path
forward documented for when the operational moment is right.

**Empirical retro on graded WC2026 matches (n=33, ~40 model-rows):**
- Modal scoreline: 6/40 = **15%** exact-score hits vs argmax 3/40 = 7.5% (doubled).
- Temperature calibration: neutral (Δlog-loss ±0.002).
- **Isotonic calibration HURTS on this sample (+1.08 for dixon_coles, +0.57 for elo_baseline)** — backtest-fit clipping fails on this high-scoring tournament. Decide whether to flip the viewer default off isotonic; the DD4 banner is the alternative surface.
- BMA on backtest-known members (DC + Elo): worse by +0.013 than DC alone — near-equal LOTO weights pulled the stronger model down.
- Tests: 164 → 248 passing (+84).


**Phase 2 Wave 1 — E1 automated nightly loop: DONE 2026-06-12** (scheduled
Action commits DB+artifacts; violations gate exits 2 and turns the build red)
**+ T2 fit-failure resilience** (`_fit_roster` drops an entrant on
RuntimeError/ValueError/LinAlgError/PyMC SamplingError instead of crashing).
**E2 public narrative board: DONE 2026-06-12** (design-reviewed, 8 locked
decisions in `~/.claude/plans/crispy-singing-kettle.md`): viewer restructured —
calibration hero w/ computed verdict + solid-backtest/hollow-live encoding,
RPS-ranked pooled leaderboard with coin-flip reference row, market-disagreement
list, per-match surprise cards, designed empty/small-n states, stale-artifact
banner. New artifacts: surprises/disagreement parquets, calibration `track`
column, consensus columns on upcoming. `publish/board.py` = unit-tested verdict
helpers. NB `data/backtest.db` is now COMMITTED (the nightly publish needs the
proof track). Wave 1 complete.
**Wave 2 started 2026-06-12: E6a score-grid capture DONE** (`score_grids` table;
`log_prediction` stores every goals-model claim's grid at predict time —
perishable, live grids can't be backfilled; `load_grid()` reads them back)
**+ T3 DONE** (`binary_calibration_table` — one reliability binning for any
binary event, Wilson bands via statsmodels; W/D/L table delegates to it;
calibration.parquet now carries ci_lo/ci_hi).
**E4 DONE 2026-06-12**: `tournament_backtest.py` (pre-tournament group-qual MC
for WC 2014/18/22 — groups derived from fixtures via connected components,
96 binary events, qual-Brier DC 0.151–0.231; `data/qualification_backtest.parquet`
COMMITTED) + `leaderboard.py` (seeded paired bootstrap → CIs + best/tied/behind
badges; **headline: elo_baseline is TIED with dixon_coles at n=192 — the gap is
noise**) + viewer verdicts/CIs + qualification section w/ Wilson bars.
**E3 DONE 2026-06-13**: `models/hierarchical.py` — Bayesian hierarchical Poisson
w/ confederation partial pooling (PyMC/nutpie); posterior-mean grids; `random_seed`
pin. `data/raw/confederations.csv` (336 teams, 6 confederations). 6th roster
entrant, goals-capable. Key: Qatar vs Switzerland 6.6% (vs DC 2.3%, Elo 10%).
**E6b DONE 2026-06-13**: `scores_scoreline` table + `score_scoreline_pending()` —
scoreline-log-loss (tail bucket), exact-score top-1/top-3, O/U 2.5 + BTTS from
grids, ET excluded. Wired into orchestrator + publisher (scoreline leaderboard +
O/U/BTTS calibration artifacts).
**Next session pickup**: (1) decide whether to flip the viewer's calibration
toggle default from `isotonic` back to `raw` given the live-sample harm —
DD2 already exposes the choice, just a `DEFAULT_TRACK_PILL` constant change.
(2) E8 (market score grid from totals odds) remains conditional on Odds API
tier. (3) Backtest the rest of the roster (`elo_decay`, `elo_importance`,
`dixon_coles_slow_xi`, `hierarchical_poisson`, `neg_bin`, `bivariate_poisson`)
so stacking has more features and BMA has weights for all 6+ entrants
instead of just 2. (4) Re-run all five dropped gates after WC2026 ends
(adds a 4th tournament; the bootstrap band narrows ~1/√(n) — stacking is
the closest candidate to flip).

Done:
- `ingest.py` (T1) — `data/processed/matches.parquet` from martj42.
- `asof.py` (T2) — point-in-time no-lookahead data access (CRITICAL invariant).
- `loop/odds.py` (T7) — odds snapshotter; first snapshot already captured.
- `config.py`, `db.py`, `registry.py`.
- `models/base.py` (T3) — split `predict_wdl`/`predict_goals` interface; WDL/ScoreGrid
  validating containers; provenance hooks (`hyperparams_hash`, `trained_through`).
- `models/elo.py` (T3) — BaselineElo (Davidson draw model, ν=0.6; neutral/decay/
  importance knobs; incremental `update()` refuses out-of-order results).
- `models/dixoncoles.py` (T4) — penaltyblog wrapper (4y trailing window anchored to
  the frame's max date, xi time decay, ET rows weight-0 by default, thin-window
  doubling fallback; ~6s fit on the real 4y window).
- `db.py` predictions+scores tables, `loop/predict.py`, `loop/score.py` (T5) —
  append-only provenance log (write-time lookahead guard: trained_through must
  predate kickoff), clamped log-loss/Brier/RPS (cross-checked vs penaltyblog;
  ours is natural-log, penaltyblog's ignorance is log2), calibration table,
  `score_pending` (grades resolved rows once; flags integrity violations —
  live rows predicted after kickoff are never scored; backtest rows exempt
  from the wall-clock rule, their claim rests on training_cutoff).
- `backtest.py` (T6) — match-level replay of WC 2014/18/22 through the same
  predict/score path; per-day Elo updates + per-day DC refits; gates: beat
  uniform ln(3), leak canary at log-loss < 0.60.
  CLI: `.venv/bin/python -m fifapreds.backtest` (~10 min; `--elo-only` is fast).
  First readout (2026-06-10, 384 preds, 0 violations): pooled log-loss
  DC 0.973 < Elo 0.982 < uniform 1.099; DC wins 2014+2018, Elo wins 2022;
  calibration max |freq−p| 0.055 (Elo) / 0.099 (DC) in populated bins.
- `loop/predict.py` CLI (T8) — `python -m fifapreds.loop.predict [--days 8|--all]`:
  fits both models, logs live predictions for upcoming WC fixtures; identical
  claims (same match/model/hyperparams/training_cutoff) dedupe to a no-op.
  First live claims logged 2026-06-10: 24 fixtures x 2 models.
- `publish/artifacts.py` (T8) — exports committed `artifacts/` (upcoming,
  leaderboard, calibration, scored parquets + meta.json) from the live +
  backtest DBs; degrades gracefully when a source is missing.
- `app/streamlit_app.py` (T8) — read-only viewer over `artifacts/` only
  (override dir via env `FIFAPREDS_ARTIFACTS`; tested with streamlit AppTest).
  Run: `.venv/bin/python -m streamlit run app/streamlit_app.py`.
- `sim/groups.py` (T9) — FIFA 2026 group tiebreakers (pts/GD/GF, recursive
  head-to-head, seeded drawing of lots; fair play skipped — no card data) +
  cross-group third-place ranking. Pure-python hot-loop code, no pandas.
- `sim/routing.py` (T10) — R32 slots from `bracket_2026.csv` + the verified
  495-combination third-place table; runtime pool checks + the CRITICAL gate
  test re-audits all 495 combos every run. `knockout_template()` for R16→final.
- `sim/montecarlo.py` (T11) — seeded tournament MC over any GoalsModel:
  conditions on played group results (2026 window only — historical WC
  meetings between same-group teams must NOT leak in; regression-tested),
  samples the rest from score grids, walks the bracket; draws → ET as
  Poisson(grid marginals/3) → 50/50 pens; knockouts all neutral-venue
  (documented approximation). 10k sims ≈ 4s after fit.
- `models/market.py` (T12) — h2h snapshot parser (power-method de-vig per
  bookmaker via `penaltyblog.implied`, median across books) + `MarketBlend`
  (default 0.75 market / 0.25 base; per-fixture fallback to base off-coverage;
  `trained_through` delegates to base — odds are pre-match info, audited via
  `odds_snapshot_id` which `predict_upcoming` now auto-attaches).
- `models/roster.py` (T13) — frozen leaderboard entrants as subclasses with
  distinct model_ids: `EloDecay` (decay 0.1/y), `EloImportance` (martj42-exact
  tournament weights — NB the Gold Cup string is plain "Gold Cup"),
  `DixonColesSlowXi` (xi 0.0005, targets the extremes-overconfidence quirk).
  `default_roster()` is THE source of entrants for the loop.
- `orchestrate.py` (T14) — the one-command live loop:
  `python -m fifapreds.orchestrate` = fetch martj42 (certifi SSL, atomic,
  refuses shrinking files; `--no-fetch` to skip) → ingest → score → refit
  roster + MarketBlend → predict (starts TOMORROW: date-only kickoffs make
  same-day claims unscoreable) → 10k-sim MC per goals model
  (`data/tournament_sim.parquet`, seed = UTC YYYYMMDD) → publish, incl.
  `artifacts/tournament.parquet` + a viewer "Tournament odds" section.
  ~25s end-to-end. E2E-tested (incl. tz-aware/naive scoring fix in score.py).
- 164/164 tests passing.
- First tournament odds published 2026-06-11 (DC, 10k sims): Spain 12.0%,
  Argentina 11.8%, England 6.7%; slow-xi flips Argentina/Spain and raises
  Colombia — the leaderboard adjudicates.

Known model quirk (expected, monitored by the calibration loop): Dixon-Coles is
overconfident at the extremes (backtest 0–0.1 bin: claimed 7.7%, happened 17.4%;
e.g. it gives Qatar 2.3% vs Switzerland where Elo says 10%) — cross-confederation
opponent pools are thin and xi-decay sharpens recent form. T13 variants compete
on exactly this.

Live cadence (E1: the loop now runs ITSELF — Phase 2 Wave 1 started 2026-06-12):
- `.github/workflows/nightly.yml` runs orchestrate at 06:30 UTC daily and
  commits the updated `data/fifa2026.db` + `artifacts/` back to the repo.
  **CI is the SOLE writer of the live DB — do NOT run orchestrate locally**
  (it forks the committed DB → binary merge conflict). Local experiments:
  `--db /tmp/scratch.db --no-fetch`. Force a real run via `workflow_dispatch`.
- The Action fails (orchestrate exits 2, nothing committed) when the scorer
  reports violations — a red nightly build means stop and investigate.
- Every day or two, still manual: `git pull` →
  `.venv/bin/python -m fifapreds.loop.odds` → commit + push the DB right away
  (quota-bound, deliberately NOT part of the orchestrator; it writes the DB,
  so don't let it straddle the nightly window).

## Setup & run
```bash
# from the project root (note: the path contains a space)
python3 -m venv .venv
.venv/bin/python -m pip install -e .            # editable install
.venv/bin/python -m pip install pytest networkx # dev/data deps

.venv/bin/python -m pytest -q                   # run tests
.venv/bin/python -m fifapreds.orchestrate       # THE live loop — CI-only now! locally use --db /tmp/scratch.db
.venv/bin/python -m fifapreds.ingest            # (re)build matches.parquet only
.venv/bin/python -m fifapreds.loop.odds         # capture an odds snapshot (uses quota)

# Reproduce the verified data artifacts:
.venv/bin/python scripts/fetch_bracket_2026.py  # routing_r32.parquet + bracket_2026.csv
.venv/bin/python scripts/build_groups_2026.py   # groups_2026.csv (with guards)
```

## Locked architecture decisions (do not relitigate — see PLAN.md)
- **Storage:** SQLite (`data/fifa2026.db`: predictions, odds_snapshots, rating
  snapshots) + parquet (raw/processed match data). The live DB is COMMITTED
  (E1: survives stateless CI runs; CI is its sole writer); `backtest.db`, raw
  CSVs and processed parquet stay gitignored, except the three frozen 2026
  inputs (`groups_2026.csv`, `bracket_2026.csv`, `routing_r32.parquet`).
- **No lookahead:** all match-history reads go through `asof.MatchStore.before(ts)`,
  which returns only *played* matches strictly before `ts`. Backtest and live loop
  share this path. This is the single most important invariant — never bypass it.
- **Split model interface:** `predict_wdl()` (every model) and `predict_goals()`
  (only goals-capable models: Dixon-Coles). Elo is W/D/L-only and CANNOT drive the
  simulator; the Monte Carlo group sim needs a goals model (for GD/goals tiebreaks).
- **Prediction provenance:** every prediction row stores model_id, model_version,
  code_version (git sha), hyperparams_hash, training_cutoff, odds_snapshot_id, seed,
  predicted_at, kickoff_ts — so the leaderboard is auditable.
- **Leaderboard integrity:** model configs are FROZEN before kickoff. Ratings update
  live; hyperparameters are NOT tuned on 2026 results (that contaminates out-of-sample).
- **Live-loop order:** score old predictions → ingest result → update models → publish.
- **Streamlit is a read-only viewer** over committed precomputed artifacts; never runs
  Monte Carlo on page load.

## Verified facts (from Block V — trust these)
- **penaltyblog 1.11.0** (pinned). `DixonColesGoalModel` has **native `neutral_venue`**
  (constructor per-match + `predict(..., neutral_venue=)`), per-match `weights`, and
  `dixon_coles_weights(dates, xi=0.0018)` for time decay. One `predict()` returns a
  `FootballProbabilityGrid` (W/D/L + `exact_score` grid + goal dists). Metrics:
  `penaltyblog.metrics.rps_array` (primary, football-standard), `ignorance_score`
  (log-loss), `multiclass_brier_score`. Odds de-vig: `penaltyblog.implied`.
  Bonus models: `penaltyblog.ratings.Elo`, `PiRatingSystem` (free leaderboard entries).
- **martj42 data**: scores INCLUDE extra time, EXCLUDE penalties. `ingest.py` flags
  `went_to_et` (via shootouts.csv) so the 90-min goals model can exclude/down-weight
  those. Data is fresh to 2026-06-27 (warm-ups scored; WC group fixtures present as
  unplayed rows). 677 ET matches flagged.
- **Team names**: martj42 is canonical. Only 2 odds-feed aliases (`registry.py`):
  USA→United States, Bosnia & Herzegovina→Bosnia and Herzegovina. All 48 WC teams
  have ≥235 historical matches.
- **Groups A–L**: `data/raw/groups_2026.csv` (triple-validated: official draw ==
  fixture clusters == host anchors A=Mexico/B=Canada/D=United States).
- **Round-of-32 routing**: `data/raw/routing_r32.parquet` (495 combos, permutation-
  checked, user-verified vs official table). 8 winners host thirds: A,B,D,E,G,I,K,L
  → matches 79,85,81,74,82,77,87,80. Bracket skeleton: `data/raw/bracket_2026.csv`.

## Conventions
- Tests are **invariant/golden-first**: no-lookahead audits, known-value metric
  checks, schema/permutation guards — over line-coverage worship. Add tests with the
  code, not after.
- `src/` layout; package is `fifapreds`. `pyproject.toml` sets pytest pythonpath.
- Secrets in `.env` (gitignored); `.env.example` is the committed template.
  `ODDS_API_KEY` is The Odds API (free tier, ~500 req/month; sport keys
  `soccer_fifa_world_cup` h2h + `soccer_fifa_world_cup_winner` outrights).
- Reproducible data builders live in `scripts/`; raw + processed data is gitignored
  (rebuild from source).

## Gotchas
- The project path contains a space (`Vibe_code/fifa preds`) — quote it in shells.
- Odds pulls cost quota; the snapshotter pulls 2 markets/run. Run daily-ish, not
  per-minute.
- `.env` is NOT committed — a fresh clone needs the key re-added to run the odds job.
