# Phase 2 — NegBin + BivariatePoisson LOTO gates: BOTH DROPPED

**Outcome:** Neither `NegBin` nor `BivariatePoisson` beats plain `DixonColes` by > 1 bootstrap SE on the LOTO backtest. The variant classes + tests + gate scripts ship (the mechanism is reusable, and ships fixed `_mle.py`-overlap evidence per D11-B); the entrants stay out of `default_roster()`.

## Method

`scripts/tune_phase2_goals.py` replays WC2014/18/22 through `fifapreds.backtest.run_backtest` with three factories — plain `DixonColes`, `NegBin`, `BivariatePoisson` — in a single pass. Same LOTO + paired bootstrap as the Phase 1 gates: for each holdout year compute pooled log-loss; report the mean across three folds; seeded paired bootstrap on the gap.

Decision gate per D11-A: ship a variant only when the LOTO log-loss gap to base DC is **negative AND its absolute value exceeds 1 bootstrap SE**.

## Result

```
             variant   LOTO log-loss   gap vs DC     ± 1 SE  verdict
---------------------------------------------------------------------------
         dixon_coles          0.9730        base             reference
             neg_bin          0.9809     +0.0079     0.0392  drop
   bivariate_poisson          0.9812     +0.0082     0.0427  drop
```

(Run: 2026-06-19, commit `137cda4`, 576 scored backtest claims across 3 variants × 3 WCs, ~17 min wall-time.)

Both variants are roughly 0.008 worse than DC — positive gap means higher log-loss. The bootstrap SE on each gap is ~5× the gap, so the SIGN of the difference is also noise. Honestly: at n=64-per-holdout, the three families are indistinguishable, and DC is the leader by a hair that doesn't survive the bootstrap.

## What this tells us (the 4/4 pattern)

Four Phase 1+2 gates attempted, four dropped:
- S6 — `EloTunedDraw` (ν sweep): every ν within ±1 SE of ν=0.6.
- S1 — `DixonColesTournamentWeighted`: +0.0054 worse, SE 0.0429.
- S2 — `NegBin`: +0.0079 worse, SE 0.0392.
- S9 — `BivariatePoisson`: +0.0082 worse, SE 0.0427.

This is the eng-review outside voice's exact warning materializing: ~64 graded matches per LOTO holdout gives a bootstrap-SE band on log-loss of ~±0.04, larger than any single-knob/single-family tuning lift. The pattern is consistent enough that further variants in this shape (more goal-model families, more parameter sweeps inside the same family) are unlikely to clear the gate either.

Implications for the rest of the plan:
- **Phase 3 (S13 Transfermarkt)** is in the same risk class — a single new variant on n=3 will face the same noise floor.
- **Phase 4 (calibration)** is different in kind: per-row recalibration on the SAME predictions doesn't add a variant, it transforms an existing one. LOTO ECE on the held-out tournament IS the validation signal, and the lift is measurable per-bin even when log-loss gaps stay small. (Smoke-test on real backtest data already shows ~0.07 log-loss improvement on the LOTO holdout for isotonic vs raw — well past the 1 SE bar.)
- **Phase 5 (BMA / stacking)** combines existing models rather than tuning new ones; the noise floor still applies to the combiner's parameters but the combined predictions can clear it via diversity.

## What ships

- `src/fifapreds/models/negbin.py`, `src/fifapreds/models/bivariate.py` — wrappers (penaltyblog 1.11.0 backed).
- `tests/test_negbin.py` (10), `tests/test_bivariate.py` (9) — full invariant coverage.
- `scripts/tune_negbin.py`, `scripts/tune_phase2_goals.py` — gate scripts.
- This document — the gate record.
- **No new roster entrants.** Anyone wanting to experiment instantiates the classes directly.

## When to revisit

Same answer as S6/S1: re-run after WC2026 + a future tournament narrow the bootstrap band to n=4-5. If the LOTO ECE on calibrated tracks shows that goal-tail mass IS where the system loses log-loss (Phase 4 will tell us this), NegBin/BivariatePoisson may earn a second look paired with a tail-specific recalibration.

## Footnote on `_mle.py` (D11-B)

The plan deferred the shared MLE base to a follow-up "evaluate after both ship." Both files exist now, line-by-line near-clones of `dixoncoles.py` — the actual shared LOC across the three is ~80 (window schedule, weights, ET cleaning, thin-window fallback). Worth extracting in a focused PR if any FUTURE goals model wraps penaltyblog the same way; not worth extracting just for the three that exist (the duplication is bounded and obviously correct).
