# S6 — `EloTunedDraw` LOTO ν sweep: DROPPED

**Outcome:** No ν beats the BaselineElo default of 0.6 by > 1 bootstrap SE in LOTO. The proposed `EloTunedDraw` roster entrant is **not** shipped. The tuning script lands in the repo as the proof of the gate; the variant class does not.

## Method

`scripts/tune_elo_nu.py` replays WC2014/18/22 through `fifapreds.backtest.run_backtest` for each ν in `{0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00}`. Each variant is a fresh `BaselineElo(draw_nu=ν)`; the rest of the rating dynamics are identical, so the ν parameter alone drives the comparison.

LOTO ("Leave One Tournament Out") is the validation convention chosen in `/plan-eng-review` D6: for each holdout year H, pool the scored predictions for year H and compute the average log-loss. The reported LOTO statistic is the mean of the three per-holdout means. Per-fixture pairing across ν values means the seeded paired bootstrap (`n_bootstrap=500`, `seed=7`) cancels per-match noise — exactly the same machinery as `artifacts/leaderboard_bands.parquet`.

Decision gate per `/plan-eng-review` D11-A: ship the variant only when the LOTO log-loss gap to the ν=0.6 baseline is **negative AND its absolute value exceeds 1 bootstrap SE**. Both conditions are non-negotiable — a positive gap means "worse than baseline" and a within-SE negative gap means "noise."

## Result

```
     ν   LOTO log-loss        gap     ± 1 SE  verdict
-----------------------------------------------------------------
  0.40          0.9953    +0.0131     0.0352  drop
  0.50          0.9852    +0.0030     0.0265  drop
  0.60          0.9822       base             ν=0.6 reference
  0.70          0.9838    +0.0016     0.0212  drop
  0.80          0.9885    +0.0063     0.0250  drop
  0.90          0.9954    +0.0132     0.0287  drop
  1.00          1.0038    +0.0216     0.0320  drop
```

(Run: 2026-06-19, commit `aac5344`, 1344 scored backtest claims across 7 variants × 3 WCs.)

The U-shape is convincing: the loss function peaks at the edges of the grid and bottoms near ν≈0.6–0.7, with 0.6 the empirical minimum and 0.7 the second-best at +0.0016. Both 0.5 and 0.7 are within 1 SE of baseline, so the parameter is well-conditioned around the original heuristic value — there is no hidden ν that the n=3 backtest can defend.

The grid's lack of signal is consistent with the validation-noise concern flagged by the eng-review outside voice: with ~64 graded matches per holdout, the bootstrap-SE band is wider than any plausible parameter-tuning lift on a single scalar. A real win would have to come from a structural change (new family, calibration layer), not from re-tuning an existing knob inside the same family.

## What ships

- `scripts/tune_elo_nu.py` — the script.
- This document — the gate record.
- No new roster entrant. `default_roster()` remains unchanged from this PR.

## When to revisit

Re-run the sweep after WC2026 grades enough claims for n=4 tournaments of out-of-sample evidence. If the live-track draw rate diverges substantially from the historical ~23% baseline, the ν=0.6 default may no longer be empirically supported and the sweep should run with the larger sample. Until then: ν=0.6 stays, no variant is added.
