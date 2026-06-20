# StackedEnsemble refresh cadence (D11-F)

## When to refresh

Regenerate `data/stacking_weights.json` via
`scripts/train_stacker.py --loto` **after each completed tournament**,
then commit the new JSON via a versioned PR. The cadence is the same
contract as the calibration pipeline (D6): hold the most-recent
tournament out for validation, re-fit on the rest, ship only when the
LOTO log-loss gap to base `dixon_coles` exceeds 1 bootstrap SE.

## When NOT to refresh

**During an in-progress tournament, the weights stay frozen.** This is
the same policy D10 locked in for BMA: refitting mid-tournament on
data we're predicting against would violate the LOTO discipline that
makes the calibration claim meaningful in the first place. The DD4
divergence banner surfaces any miscalibration the frozen weights
develop during the tournament — the policy answer is "tell the user,"
not "refit silently."

## Failure mode → action

| Condition | What happens | Why |
|---|---|---|
| LOTO log-loss gap to DC < 1 bootstrap SE | Script exits 1; weights file unchanged | D11-A's vanity-entrant gate. Same contract as the other Phase 1+2 gates that all failed. |
| Backtest DB lacks `dixon_coles` predictions | Script exits 2; can't compute the gate | The baseline is the comparison anchor; no DC, no gate. |
| Backtest DB has fewer than 2 tournament years | Script exits 2 | LOTO needs ≥ 2 folds; otherwise "holdout" collapses to "train" by construction. |
| Member missing from supplied roster at runtime | `StackedEnsemble.__init__` raises | D5 fail-loud. Saved `model_ids` are the contract. |
| Weights file missing | `StackedEnsemble.__init__` raises | Loud → orchestrator's existing drop-entrant path skips the ensemble for the night. |

## What "refresh" actually changes

The JSON ships:
- `model_ids`: the order base-model probabilities are concatenated into the feature vector. Changing this order at train time is fine; runtime alignment uses these saved IDs.
- `coefficients` (3 × 3·N) + `intercepts` (length 3): the multinomial logistic regression's parameters. Re-fit on every refresh.
- `C`: the L2 strength (default 0.5). The plan flagged overfitting on n=192; the regularization is the guardrail. Don't relax it without measuring LOTO log-loss BEFORE and AFTER.
- `holdout_years` + `trained_on`: provenance. `trained_on` is the union of the LOTO folds — the final fit uses ALL years (standard "train on all data after CV" pattern).
- `loto_log_loss`, `dc_log_loss`, `gap_vs_dc`, `gap_se`: the gate's verdict on the day of training, captured for audit.
- `fit_date`: the day the refresh ran, surfaced in the viewer's footer caption alongside the weights' age.

## Manual refresh procedure

```bash
# 1. Pull the latest backtest.db (it's committed; nightly CI may have updated it).
git pull --rebase origin main

# 2. Run the gate. Either exits 0 (ship) or 1 (drop, no overwrite).
.venv/bin/python -m scripts.train_stacker --loto

# 3. If shipped, commit the new JSON.
git diff data/stacking_weights.json   # inspect the coefficient change
git add data/stacking_weights.json docs/stacker_refresh.md
git commit -m "stacking: refresh weights for WC{YYYY} (gap {X.XXXX} ± {Y.YYYY} SE)"
git push origin main

# 4. The next nightly will pick up the new JSON automatically — StackedEnsemble
#    reads it at construction time.
```

## Outside the refresh window

If you find yourself wanting to refresh weights for any reason OTHER than "a tournament just ended," stop. The most likely cause is the DD4 divergence banner firing — that's the system working as designed, not a reason to refit. The right action is to investigate WHY the live track diverged (model drift? a bin-specific bug? an unexpected outcome in a rare matchup?) and either accept the divergence as honest noise or open a focused PR that addresses the root cause. Refitting weights to make the banner go away is the failure mode this doc exists to prevent.
