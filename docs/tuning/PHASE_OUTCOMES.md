# Accuracy plan — phase outcomes (close-out)

Plan source: `~/.claude/plans/encapsulated-sparking-dove.md`. 11 SHIP items + 2 deferred research items.

## What landed in production

| # | Item | Status | Roster wiring | Doc |
|---|------|--------|---------------|-----|
| T1 | Hierarchical Poisson on CI + per-model fit timeout | **SHIPPED** | `hierarchical_poisson` now joins the live roster | CLAUDE.md current status |
| S6 | EloTunedDraw ν sweep | gate FAILED | (no entrant) | `docs/tuning/elo_nu.md` |
| S1 | DixonColesTournamentWeighted | gate FAILED | (no entrant) | `docs/tuning/dc_tournament_weighted.md` |
| S2 | NegBin wrapper | gate FAILED | (no entrant) | `docs/tuning/phase2_goals.md` |
| S9 | BivariatePoisson wrapper | gate FAILED | (no entrant) | `docs/tuning/phase2_goals.md` |
| S13 | Transfermarkt MarketValueElo | DEFERRED | (no entrant) | `docs/tuning/transfermarkt_deferred.md` |
| S25 | Temperature scaling | **SHIPPED** | calibration track on `leaderboard.parquet` | (in-code) |
| S20 | Isotonic recalibration | **SHIPPED** | calibration track on `leaderboard.parquet` | (in-code) |
| S22 | BMAEnsemble + BMAGoalsEnsemble | **SHIPPED** | both join the roster when backtest history exists | (in-code) |
| S19 | StackedEnsemble | mechanism SHIPPED, gate FAILED | (no entrant — wiring skips on missing weights JSON) | `docs/tuning/stacking.md` |
| 11 | Modal scoreline = E[goals] rounded | **SHIPPED** | new `modal_h` / `modal_a` cols on `upcoming.parquet`; two-line viewer headline | (in-code) |
| DD4 | Mid-tournament divergence banner | **SHIPPED** | viewer hero amber banner when freq escapes Wilson on calibrated p | (in-code) |

## The 5/5 single-knob gate pattern

| Gate | Variant | Gap vs DC | Bootstrap SE | Ratio |
|------|---------|-----------|--------------|-------|
| S6 | ν=0.5 (closest neighbour) | +0.0030 | 0.0265 | 0.11 |
| S1 | DC tournament-weighted | +0.0054 | 0.0429 | 0.13 |
| S2 | NegBin | +0.0079 | 0.0392 | 0.20 |
| S9 | BivariatePoisson | +0.0082 | 0.0427 | 0.19 |
| S19 | Stacked logistic | +0.0076 | 0.0170 | **0.45** |

Every gap is POSITIVE (= worse than DC) and every ratio is BELOW 1.0 (= within bootstrap noise). The eng-review outside voice predicted this exactly: at ~64 graded matches per LOTO holdout, the bootstrap noise floor on log-loss is ±0.02–0.04, larger than any plausible single-knob tuning lift produces. Stacking is the closest near-miss (ratio 0.45) and is the most plausible candidate to flip when WC2026 adds a fourth tournament's worth of evidence.

## What is shipping accuracy

Three interventions actually move the published artifact in a measurable way:

1. **Phase 4 calibration** — isotonic recalibration shows ~0.07 log-loss improvement on the LOTO backtest holdouts (well past the > 1 SE bar that the single-knob gates failed). The calibration toggle in the viewer hero (DD2) lets users switch between raw / temperature / isotonic; isotonic is the default.
2. **Phase 5a BMA** — combining the fitted roster without learning a new parameter sidesteps the gate's noise penalty. Both `bma_ensemble` and `bma_goals_ensemble` enter the live roster automatically when backtest history exists for their members.
3. **Item 11 modal scoreline** — 119 of 168 WC2026 fixtures (71%) now show a different headline scoreline (the rounded posterior mean instead of `argmax(grid)`). This is the only intervention that directly moves the user-visible top-line metric the project's diagnosis was about.

## What ships defensively

4. **T1 — hierarchical Poisson on CI + per-model fit timeout.** Activates the 6th roster entrant; bounds any future slow model's impact on the nightly budget.
5. **DD4 — divergence banner.** Mid-tournament policy is "freeze the calibration weights, surface drift to the user." Banner fires only when the live track's frequency escapes the Wilson interval around the calibrated `p_mean` — most days silent, which is the contract.

## Test footprint

- Started at 164 passing tests.
- Ended at 248 passing tests.
- +84 tests across: orchestrator timeout, DC importance kwarg + regression, NegBin invariants, BivariatePoisson invariants, calibration module (23 tests), BMA (17), stacking (7), board helpers (modal + divergence), DD2 calibration filter.
- All gates are reproducible: `scripts/tune_elo_nu.py`, `scripts/tune_dc_tournament_weighted.py`, `scripts/tune_phase2_goals.py`, `scripts/train_stacker.py`.

## Files added this round

```
src/fifapreds/calibration/{__init__,base,isotonic,temperature,pipeline}.py
src/fifapreds/ensemble/{__init__,bma,stacking}.py
src/fifapreds/models/{negbin,bivariate}.py
scripts/{tune_elo_nu,tune_dc_tournament_weighted,tune_negbin,tune_phase2_goals,train_stacker}.py
tests/{test_negbin,test_bivariate,test_calibration,test_bma,test_stacking}.py
docs/tuning/{elo_nu,dc_tournament_weighted,phase2_goals,stacking,transfermarkt_deferred,PHASE_OUTCOMES}.md
docs/stacker_refresh.md
```

## When to revisit the dropped gates

After WC2026 ends, `data/backtest.db` gains a fourth tournament. The bootstrap SE band scales by ~`1/√(n_folds)`, narrowing from ~±0.04 to ~±0.03 for the 3-tournament gates. Stacking specifically narrows from ±0.0170 to ~±0.0140 — the closest candidate to flip. Re-run all five gates after WC2026 completes; if any clear, ship the corresponding entrant via a focused PR.

The deeper structural fixes the outside voice recommended (per-player ratings, market-implied score grids via paid odds API, neural goal models) remain out of scope for this round — they were always Phase 3 backlog, not Phase 1-2 candidates.
