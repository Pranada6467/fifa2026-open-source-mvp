# S19 — StackedEnsemble LOTO gate: DROPPED

**Outcome:** The logistic-stacking meta-learner over per-model probability vectors does **not** beat plain `dixon_coles` by > 1 bootstrap SE on the LOTO backtest. The `StackedEnsemble` class + training script + refresh-cadence doc still ship (the mechanism is reusable, and the weights file gets refit automatically when WC2026 ends); the entrant is absent from `default_roster()` for the duration of this freeze.

## Method

`scripts/train_stacker.py --loto` trains a multinomial L2-regularized `LogisticRegression` (`C=0.5`) on the per-model probability features (concatenated `(p_home, p_draw, p_away)` per base model) for every WC fixture in the backtest, with LOTO discipline: for each holdout year fold, train on the other two years, score on the holdout. Paired-bootstrap on the gap-vs-DC (`seed=7`, `n_bootstrap=500`).

Decision gate per D11-A: ship the weights JSON only when the LOTO log-loss gap to `dixon_coles` is **negative AND its absolute value exceeds 1 bootstrap SE**. Otherwise the script exits 1, the weights file stays whatever it was, and the orchestrator's stacking entrant is skipped via the `FileNotFoundError` path.

## Result

```
  holdout wc2014: stack 0.9531  dc 0.9169  n=64
  holdout wc2018: stack 0.9635  dc 0.9432  n=64
  holdout wc2022: stack 1.0249  dc 1.0587  n=64

LOTO stack log-loss: 0.9805
LOTO dc    log-loss: 0.9729
gap stack-dc:        +0.0076  ± 0.0170 SE
```

(Run: 2026-06-20, commit `21f80e8`, 192 fixtures × 2 base models — only `dixon_coles` and `elo_baseline` have full backtest history at the moment; the other roster members haven't been backtested yet, so the stacker can only learn over those two.)

Stacking IS better than DC on wc2022 (1.0249 vs 1.0587) but worse on wc2014 and wc2018. Per-fold variance is wide enough that the average gap is +0.0076 — meaning stacking is 0.0076 WORSE than DC on average. The 0.0076 gap is well inside the ±0.0170 bootstrap SE, so the signed direction is also noise: we can't reliably say stacking helps or hurts. The disciplined call is "drop."

## The 5/5 pattern

Five Phase 1+2+5b gates run, five dropped:
- **S6** EloTunedDraw: every ν within ±1 SE of ν=0.6.
- **S1** DixonColesTournamentWeighted: +0.0054 worse, SE 0.0429.
- **S2** NegBin: +0.0079 worse, SE 0.0392.
- **S9** BivariatePoisson: +0.0082 worse, SE 0.0427.
- **S19** Stacking: +0.0076 worse, SE 0.0170.

The pattern is now overwhelming: at ~64 graded matches per LOTO holdout, the bootstrap noise floor on log-loss is ~±0.02–0.04, larger than any single-knob tuning lift produces. The eng-review outside voice predicted this exactly. Three things ARE working at this sample size:
- **Calibration** (Phase 4) — measured on per-bin Wilson intervals, not pooled log-loss. Smoke test showed ~0.07 log-loss lift on backtest holdouts for isotonic vs raw.
- **Modal scoreline** (Item 11) — a representation change of the headline, not a probability re-tuning; 71% of fixtures get a different headline scoreline under the modal mean vs argmax.
- **BMA** (Phase 5a) — averaging multiple existing models is a no-cost combination; doesn't carry the variance penalty of a learned new parameter.

## What ships

- `src/fifapreds/ensemble/stacking.py` — `StackedEnsemble(Model)` reads frozen weights from `data/stacking_weights.json`. Raises `FileNotFoundError` on missing file → orchestrator's drop-entrant path skips the ensemble for the night (no weights = no ensemble, no error chain).
- `scripts/train_stacker.py` — the gate + ship script.
- `docs/stacker_refresh.md` — the refresh cadence contract.
- This document — the gate record.
- **No `data/stacking_weights.json`.** Will land automatically when a future LOTO run clears the > 1 SE bar.

## When to revisit

Re-run after WC2026 ends and `data/backtest.db` gains a fourth tournament's worth of evidence. The +0.0076 / 0.0170 ratio is the closest any Phase 1+2+5b gate has come to passing — with one more tournament tightening the noise band, stacking is a plausible candidate to flip from "drop" to "ship." Concretely: a fourth WC narrows the LOTO bootstrap SE to roughly ±0.0140 (1/√(192/64) scaling); the gap needs to either widen to ~0.018 (likely) or the noise needs to halve (less likely) for the gate to pass.

Backtesting the other base models (hierarchical_poisson, neg_bin, bivariate_poisson, the elo variants) would also widen the feature space the stacker sees — currently only `dixon_coles` + `elo_baseline` are in `backtest.db`. A richer stacker over the full roster might clear the gate on the current sample. That's a separate task: extend `fifapreds.backtest` to replay every roster entrant, not just the legacy two.
