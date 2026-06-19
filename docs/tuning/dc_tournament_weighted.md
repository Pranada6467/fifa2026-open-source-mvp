# S1 — `DixonColesTournamentWeighted` LOTO gate: DROPPED

**Outcome:** Tournament-stage weighting on the xi-decay does **not** beat plain `DixonColes` on the LOTO backtest. The variant class + tests ship (the mechanism is reusable), but the entrant stays out of `default_roster()` per the D11-A gate.

## Method

`scripts/tune_dc_tournament_weighted.py` replays WC2014/18/22 through `fifapreds.backtest.run_backtest` with two factories: plain `DixonColes` and `DixonColesTournamentWeighted` (the variant initializes with `importance=dict(TOURNAMENT_IMPORTANCE)`, the same frozen ladder `EloImportance` already uses). The DC backtest is per-match-day MLE refits, so this is a slow comparison — ~12 minutes wall-time on a 2024 MacBook.

LOTO: for each holdout WC year, pool the variant's scored predictions for that year and compute the average log-loss. Reported statistic is the mean of the three per-holdout means.

Gate per D11-A: ship only when the LOTO log-loss gap to base DC is **negative AND its absolute value exceeds 1 bootstrap SE**. Seeded paired bootstrap (`n_bootstrap=500`, `seed=7`) cancels per-fixture variance via the same alignment the published `leaderboard_bands.parquet` uses.

## Result

```
                         variant   LOTO log-loss        gap     ± 1 SE
------------------------------------------------------------------------
                     dixon_coles          0.9730       base
 dixon_coles_tournament_weighted          0.9784    +0.0054     0.0429
```

(Run: 2026-06-19, commit `4b3857d`, 384 scored backtest claims across 2 variants × 3 WCs.)

Tournament weighting is **0.0054 worse** than plain DC — positive gap means higher log-loss means worse calibration. The bootstrap SE of 0.0429 is **~8× the absolute gap**, so the signed direction is also noise: we cannot reliably say tournament weighting hurts either. The honest reading is "the two variants are indistinguishable on this validation surface."

## Why this gate fails (consistent with S6)

Two Phase 1 variants attempted, two gates failed. The pattern matches the outside-voice concern from `/plan-eng-review`: with ~64 graded matches per holdout, the bootstrap noise band on log-loss is ~±0.04, swamping any tuning lift below that magnitude. Two reasons the math here is harder than for Elo:

1. **DC already absorbs tournament-context indirectly.** The xi decay weighs recent matches more, and the WC2026 backtest's most-recent matches ARE high-importance qualifiers and the WC itself. Re-weighting by importance partly double-counts that signal.
2. **Most matches in the training window are NOT World Cup matches.** TOURNAMENT_IMPORTANCE gives friendlies 0.5 and WCs 1.75, but the WC pool is small relative to friendlies + qualifiers. The aggregate shift in the weighted likelihood is modest, and the model's freedom to compensate is large — penaltyblog's MLE finds a similar solution either way.

## What ships

- `src/fifapreds/models/dixoncoles.py` — DC accepts optional `importance` kwarg (byte-identical math when None).
- `src/fifapreds/models/roster.py` — `DixonColesTournamentWeighted` class definition (subclass with the frozen importance dict).
- `scripts/tune_dc_tournament_weighted.py` — the gate script.
- `tests/test_dixoncoles.py` — 4 new tests covering the optional importance kwarg.
- This document — the gate record.
- **No new roster entrant.** `default_roster()` remains unchanged.

The mechanism is reusable: anyone who later wants to experiment with a different importance dict (or a learned one) instantiates `DixonColes(importance={...})` directly. The plumbing is there; only the specific "TOURNAMENT_IMPORTANCE" variant is dropped.

## When to revisit

Re-run after WC2026 ends, when n=4 tournaments of out-of-sample evidence narrows the bootstrap band. If a future variant proposes a *learned* importance vector (e.g., importance values fit jointly with attack/defense per team) instead of the hand-set ladder, that's a separate gate, separate doc.
