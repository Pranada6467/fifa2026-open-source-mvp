# Plan: FIFA 2026 Live Calibration Engine — Architecture & Build

## Context

Greenfield project (empty dir, Python 3.11.9, not yet a git repo). Goal: a
World Cup 2026 prediction system that rates teams, produces per-match and
full-tournament probabilities, and runs as a **live loop** that re-scores its
own predictions and recalibrates after every result. The product is **honest
calibration** ("my 70% calls happen ~70% of the time") plus a **technique
leaderboard** ranking models by out-of-sample log-loss / Brier. Source design
doc: `~/.gstack/projects/fifapreds/pranav-unknown-design-20260610-150552.md`
(APPROVED). The 2026 World Cup opens **June 11, 2026** (tomorrow), so the live
loop and odds-snapshotting start immediately; the historical backtest is the
real calibration proving ground (one tournament's ~104 matches is too small to
prove calibration alone).

Scope: specify the **entire** architecture; execute in independent, shippable
blocks. **A verification spike (Block V) runs first** and gates the decisions
that depend on unverified external facts.

Locked decisions (do not depend on external verification):
- **Storage:** SQLite (logs/snapshots) + parquet (raw data).
- **No lookahead:** explicit point-in-time (as-of) data-access layer.
- **Model interface is split:** `predict_wdl()` (every model) and
  `predict_goals()` (only goals-capable models). Elo is W/D/L-only; the
  simulator requires a goals-capable model.
- **Scorer:** log-loss / Brier / calibration curve with probability clamping.

Provisional decisions (confirmed or revised by Block V): `penaltyblog` as the
scoreline/de-vig dependency; martj42 score-semantics handling; the 2026 routing
table; odds-provider coverage. See Block V.

---

## Architecture

```
martj42 history ─┐
wc2026 fixtures ─┤→ ingest ─→ parquet (raw)        SQLite (fifa2026.db):
manual results  ─┘                │                  predictions, odds_snapshots,
                                  ▼                   rating_snapshots
                          asof(as_of_ts)  ◄────── enforces NO lookahead
                                  │                (one read path; backtest +
              ┌───────────────────┼────────────┐   live loop share it)
              ▼                   ▼             ▼
         BaselineElo        DixonColes      MarketBlend
        (W/D/L only)       (W/D/L+goals)   (de-vig odds)
              └───────────────────┼─────────────┘
            predict_wdl() [all]   │   predict_goals() [goals-capable only]
                                  ▼
                       per-match probs (+ score grid where available)
              ┌───────────────────┼─────────────────────┐
              ▼                                          ▼
     predictions_log (append-only,                MonteCarlo simulator
       full provenance)                           group→best-third→R32→…→final
              │  result lands                     (needs a goals model)
              ▼                                          │ stage + winner probs
   score: log-loss / Brier / calibration                │
              └──────────────────┬───────────────────────┘
                                 ▼
              publish/artifacts (parquet+JSON, committed to repo)
                                 ▼
                  Streamlit Community Cloud (read-only viewer)
```

### Module layout
```
fifa2026/
  pyproject.toml          (pinned deps incl. penaltyblog==<verified version>)
  data/
    raw/                  parquet: results.parquet, fixtures.parquet
    fifa2026.db           SQLite: predictions, odds_snapshots, rating_snapshots
  src/fifapreds/
    registry.py           canonical team registry (source name → team_id)
    ingest.py             load martj42 + fixtures → parquet; manual-result append + correction
    asof.py               POINT-IN-TIME access: matches_before(ts), ratings_as_of(ts)
    models/
      base.py             Model interface: predict_wdl(); GoalsModel adds predict_goals()
      elo.py              BaselineElo (W/D/L; time-decay, importance, neutral)
      dixoncoles.py       penaltyblog wrapper (window refit, neutral, goals)
      market.py           MarketBlend (de-vig odds)
    sim/
      groups.py           group table + FIFA tiebreakers (+ declared lots fallback)
      routing.py          2026 best-third routing matrix  ← P1, unit-tested, gated
      montecarlo.py       tournament Monte Carlo (seeded; needs a GoalsModel)
    loop/
      predict.py          write predictions (full provenance) for upcoming fixtures
      score.py            grade resolved predictions; calibration metrics
      orchestrate.py      on-result: SCORE → INGEST → UPDATE → PUBLISH
    publish/artifacts.py  precompute artifacts the app reads; commit to repo
  app/streamlit_app.py    reads committed artifacts only
  tests/
```

### Key design decisions (locked)
- **As-of layer is the only path models use to read match data.** Takes an
  `as_of_ts`; physically filters to `date < as_of_ts`. Predictions store the
  full state needed to audit them (see provenance below). The scorer only grades
  rows where `predicted_at < kickoff_ts`. The backtest replays matches in time
  order feeding each prediction only prior data — same code path as the live
  loop. **Hardening:** ratings are produced by replaying events in order (or
  stored with `trained_through` + creation metadata), never regenerated
  after-the-fact; fixtures carry explicit `kickoff_ts` (UTC), not date-only, to
  prevent same-day / timezone leaks.
- **Split model interface.** `predict_wdl()` on every model (W/D/L leaderboard);
  `predict_goals()` only on goals-capable models (Dixon-Coles/Poisson) returning
  the score grid + lambdas. Elo competes on W/D/L but cannot drive the simulator;
  **the Monte Carlo group sim requires a GoalsModel** because GD / goals-scored
  tiebreakers need scorelines.
- **Elo and Dixon-Coles are independent families** behind that interface (Elo
  incremental per result; DC batch MLE refit on a trailing window). The registry
  scores both — this is the experiment harness.
- **Prediction provenance (auditable leaderboard).** Each prediction row stores:
  `model_id`, `model_version`, `code_version` (git sha), `hyperparams_hash`,
  `training_cutoff`, `feature_cutoff`, `odds_snapshot_id` (nullable), `seed`,
  `predicted_at`, `kickoff_ts`.
- **Leaderboard integrity / freeze rule.** Model *configs* are frozen before
  kickoff. During the tournament, **ratings update live** (that is the model
  running), but **hyperparameters/calibration params are NOT tuned on 2026
  results** — doing so contaminates out-of-sample claims. Any post-hoc tuned
  variant goes on a separate, clearly-labelled non-out-of-sample track.
- **Explicit live-loop order:** `score old predictions → ingest result → update
  models → publish new predictions`. Never update before scoring.
- **Neutral venue is explicit.** Host-advantage term applies only to genuine home
  sides; DC home-adv zeroed for neutral. **Host edge (USA/CAN/MEX 2026) is a
  fixed prior, not backtest-learned** (no useful precedent for a 3-host event).
- **Data semantics.** The goals model trains on **90-minute / regulation-time
  scores**; extra-time and shootout outcomes are handled/flagged so knockout ET
  aggregates don't poison Dixon-Coles. (Exact handling confirmed in Block V.)
- **Manual results support correction.** Append-only with supersession: a
  correction writes a new row that supersedes the prior, never mutates it.
- **Streamlit is a viewer over committed artifacts.** The update job writes
  precomputed artifacts and commits them; Streamlit Community Cloud redeploys and
  renders them. No Monte Carlo on page load; no separate artifact store to keep
  coherent.
- **Monte Carlo is seeded** (reproducible tests); sim count `N` configurable
  (default 10k). ET ≈ scaled expected goals; penalties ≈ logistic on rating gap
  with a small favourite edge.

---

## Block V — Verification Spike (FIRST; ~half a day; gates provisional decisions)

De-risk the external assumptions before committing code to them. Each item has a
pass criterion and a fallback. **Decisions below are provisional until V passes.**

| # | Verify | Pass criterion | Fallback if it fails |
|---|--------|----------------|----------------------|
| V1 | `penaltyblog` install + APIs | Pin a version; confirm Poisson/Dixon-Coles models, `penaltyblog.implied` de-vig, time-decay weights, neutral/home-adv params all exist with the expected signatures | Implement the missing piece directly (de-vig is trivial; DC via statsmodels/manual MLE) |
| V2 | martj42 score semantics + freshness | Document whether `results.csv` scores are regulation or include ET; confirm `shootouts.csv` exists; confirm latest date covers June 2026 warm-ups | Derive 90-min scores where possible; else flag-and-exclude ET knockout matches from goals-model training; manual-enter recent friendlies |
| V3 | WC2026 fixtures source | A parseable fixture list for all 104 matches with groups, dates, venues, neutral flags | Transcribe from the FIFA site as the authority |
| V4 | **2026 best-third routing table** | Official FIFA Round-of-32 assignment table located, understood, transcribable | Keep the simulator's winner-probs **gated**; ship W/D/L + group-stage only until verified |
| V5 | Team-name reconciliation | Every WC2026 team resolves to a martj42 name via the registry (100%) | Hand-map the mismatches into `registry.py` |
| V6 | Odds provider (time-sensitive) | The Odds API free tier (or chosen source) covers WC2026 h2h + outrights; one real snapshot saved | Pick an alternative source; worst case defer market-blend, but START snapshotting whatever is available now |

Output of Block V: a short findings note that either confirms the provisional
decisions or triggers a targeted revision before Block 0 code lands.

---

## Test plan (greenfield — invariant/golden-first, line coverage secondary)

The high-value tests are **no-leak invariants, golden-data replay, timestamp
audits, schema constraints, and known historical fixtures** — not line-coverage
theatre. Every new path still gets a test; these are the ones that matter most.

```
PLANNED PATH                                   TEST REQUIRED                          KIND
[+] asof.py
  ├── matches_before(ts)                        [GAP] no row dated >= ts returned      unit ★★★
  └── ratings_as_of(ts)                          [GAP] predict for M never touches      unit ★★★  ← CRITICAL (silent leak)
                                                       data >= kickoff_ts (timestamp audit)
[+] registry.py resolve(name)                    [GAP] every WC2026 team resolves;      unit ★★★
                                                       unknown name raises (not drops)
[+] models/elo.py update/neutral                 [GAP] symmetry/decay; home-adv off     unit ★★★
                                                       when neutral
[+] models/dixoncoles.py fit/predict_goals       [GAP] thin-window fallback; grid       unit ★★★
                                                       sums to 1, clamped > 0
[+] models/market.py blend                       [GAP] missing odds → model-only        unit ★★
[+] sim/groups.py table+tiebreakers              [GAP] FIFA order exact; lots-fallback  unit ★★★
                                                       deterministic under seed
[+] sim/routing.py best-third                    [GAP] official table; exactly 8        unit ★★★  ← CRITICAL (silent)
                                                       advance; valid 32-team tree
[+] sim/montecarlo.py run(N,seed)                [GAP] seed reproducible; probs sum=1   unit ★★★
[+] loop/score.py metrics                        [GAP] golden inputs → known log-loss/  unit ★★★
                                                       Brier; clamp prevents log(0)
[+] loop/orchestrate.py order                     [GAP] asserts SCORE→INGEST→UPDATE→     [→E2E]
                                                       PUBLISH order
[+] backtest 2014/18/22 (match-level)            [GAP] integration gate: replay, no     [→E2E]  (Block 0 — no sim needed)
                                                       leak, calibration plausible
[+] ingest correction                            [GAP] correction supersedes, prior     unit ★★
                                                       row preserved
```

### Failure modes (each needs a test + handling)
| Failure | Visibility | Handling | Test |
|---|---|---|---|
| Lookahead leak in as-of reads | Silent (flatters scores) | as-of layer + timestamp audit | CRITICAL unit |
| Wrong routing matrix | Silent (bad winner probs) | official table + gate | CRITICAL unit |
| After-the-fact rating snapshot leaks | Silent | event-replay / `trained_through` | unit |
| ET/shootout scores poison DC | Silent (skews goals) | 90-min cleaning (V2) | unit |
| DC fit non-convergence (thin window) | Loud-ish | fallback to Elo / wider window | unit |
| Team name mismatch | Silent (dropped matches) | registry; unknown → raise | unit |
| Missing odds for a fixture | Silent (blend skewed) | degrade to model-only | unit |
| log(0) in log-loss | Loud (NaN) | clamp p∈[ε,1-ε] | unit |
| Leaderboard rewards post-hoc tuning | Silent (fake skill) | freeze rule + provenance | review/process |

---

## What already exists
Nothing in-repo (greenfield). Reused external building blocks instead of custom:
`penaltyblog` (scoreline + de-vig, pending V1), `martj42/international_results`
(history), `openfootball/worldcup` (fixtures seed). FIFA rules/site are the
authority for format, fixtures, and routing; openfootball is a seed only.

## NOT in scope (deferred, with rationale)
- **xG-adjusted form (rung 6):** needs a free xG source; not required to prove the
  loop. After market-blend.
- **Agentic adjustment panel (rung 7):** only worth it once the harness can prove
  it lowers log-loss; noise before that.
- **Paid live data (Sportmonks / API-Football):** v1 is free/open + manual entry.
- **Scheduled GitHub Action auto-updates:** v1 runs the update job manually before
  match days; cron later.
- **Auth / multi-user / write UI:** Streamlit stays read-only.
- **Worktree/PR parallelization:** premature for a one-person empty repo; revisit
  once the spine exists and lanes are real.

---

## Execution blocks (modular, incremental — each a shippable PR)

**Block V — verification spike** (above). Do first.

**Block 0 — live loop spine + proving ground + odds capture**
- T1 (P1) — `registry.py` + `ingest.py` (load → parquet; manual append +
  correction). Tests: every WC2026 team resolves; correction supersedes.
- T2 (P1) — `asof.py` point-in-time access. Tests: no-lookahead timestamp audit. CRITICAL.
- T3 (P1) — `models/base.py` (split interface) + `models/elo.py`. Tests: update
  symmetry/decay, neutral handling.
- T4 (P1) — `models/dixoncoles.py` (penaltyblog, goals, neutral, thin-window
  fallback). Tests: normalized/clamped grid, fallback path.
- T5 (P1) — SQLite schema (full provenance) + `loop/predict.py` + `loop/score.py`.
  Tests: golden-value metrics, clamp, calibration diagonal.
- T6 (P1) — **match-level backtest harness** on 2014/18/22 (no 2026 sim needed).
  Integration gate: replay, no leak, calibration plausible.
- T7 (P1) — **odds snapshotting** (capture only): `odds_snapshots` + a pull job.
  Start now — pre-match odds can't be backfilled.
- T8 (P2) — `app/streamlit_app.py` + `publish/artifacts.py` (commit artifacts).
  Deploy to Streamlit Community Cloud.

**Block 1 — tournament simulator**
- T9 (P1) — `sim/groups.py`: table + FIFA tiebreakers + declared lots fallback.
- T10 (P1) — `sim/routing.py`: 2026 best-third matrix vs official table. CRITICAL,
  gated until verified.
- T11 (P1) — `sim/montecarlo.py`: seeded MC (needs GoalsModel); ET/penalties approx.

**Block 2 — climb the technique ladder**
- T12 (P2) — `models/market.py` de-vig + MarketBlend over the captured snapshots.
- T13 (P2) — register time-decay + importance Elo variants; harness ranks all
  models by out-of-sample log-loss/Brier (the leaderboard).
- T14 (P3) — `orchestrate.py`: full on-result pipeline as one command. Test: E2E.

## Verification (end-to-end, post-build)
1. `pytest` green, including the two CRITICAL tests (as-of no-lookahead audit;
   routing matrix vs official table).
2. Match-level backtest 2014/18/22: prints log-loss/Brier + calibration curve;
   asserts no lookahead; calibration within plausible band.
3. Live smoke: append a real upcoming fixture, run `predict`, confirm a
   `predictions` row with full provenance and `predicted_at < kickoff_ts`; after
   the result, run the loop (`score → ingest → update → publish`) and confirm a
   calibration entry and a fresh artifact.
4. `streamlit run app/streamlit_app.py` renders today's predictions + the running
   calibration curve from committed artifacts.

## Unresolved decisions that may bite later
- Exact host edge for USA/CAN/MEX (fixed prior; start small).
- Dixon-Coles trailing-window length (start ~4y / importance-weighted; tune
  pre-tournament only, then freeze).
- Monte Carlo `N` vs live-loop runtime (10k; raise if winner-prob noise shows).

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | issues_found | 19 points; 3 real bugs absorbed |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | reviewed | 14 issues, 2 critical risks (each gated by a mandatory test) |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **CODEX:** outside voice ran; caught the Elo `score_grid` bug, backtest/odds mis-sequencing, and the leaderboard-contamination risk — all folded in. A verification spike (Block V) was added per the user's "verify before locking" call.
- **CROSS-MODEL:** review and outside voice agree the historical backtest (not the single live tournament) is the real calibration proof, and that a thin vertical slice ships first. No unresolved cross-model tension — the user accepted the three substantive fixes.
- **UNRESOLVED:** 0 decisions left open (3 parameters noted as tune-pre-tournament-then-freeze).
- **VERDICT:** ENG review complete — cleared to implement. Block V (verification) runs first; the two CRITICAL silent-failure risks (as-of lookahead leak, 2026 routing matrix) are each gated by a mandatory test before their code is trusted.
