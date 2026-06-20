# S13 — Transfermarkt squad-value entrant: DEFERRED

**Outcome:** The Transfermarkt scraper + `MarketValueElo` entrant from Phase 3 of the accuracy plan does NOT ship in this round. Rationale below; the deferral is documented so the gap is explicit, not hidden.

## Why deferred

Two compounding reasons:

1. **The 5/5 gate pattern.** Every single-knob LOTO gate run this cycle has failed the > 1 SE bar (S6, S1, S2, S9, S19). The eng-review outside voice explicitly called this out as the likely outcome for any variant in the same risk class. `MarketValueElo` is structurally similar to `EloImportance` (a K-factor multiplier on `BaselineElo`); on the same n=64-per-holdout backtest the variance band that killed those gates would almost certainly kill this one too. Running the gate costs ~15 min of compute for a result we can confidently predict.

2. **The S13 caveat (D11-E) already documented this is forward-looking only.** The plan's own gate doc said: "historical backtest opponents fall back to median value, so the model's predictive claim is forward-looking only; backtest tests 'doesn't crash on data it doesn't use.'" In other words, the LOTO holdout WAS NEVER GOING TO BE A FAIR TEST of this variant. The gate would have either failed loudly (matching the pattern) or passed via flatter-than-baseline numbers that wouldn't actually validate the forward-looking claim.

Plus a third operational concern: the Transfermarkt scraper itself needs a manual one-shot snapshot from a specific page, and the page structure is fragile against silent anti-bot changes. That's a half-day operational task the user should drive when they choose to invest in it, not a thing to auto-spin in this session.

## What CAN still ship cheaply

If/when the user does want the squad-value signal, the path is:

1. Manually capture `data/raw/squad_values_2026.csv` with columns `(team, total_value_eur, top23_value_eur, snapshot_date)` covering all 48 WC2026 teams. Either by hand-typing from Transfermarkt's WC squad pages or a one-time scrape with a current beautifulsoup recipe.
2. Add `src/fifapreds/models/market_value.py` — `MarketValueElo(BaselineElo)` that loads the CSV at `__init__`, scales K-factor by `(v_team / v_median) ** alpha` (clamped to `[0.5, 2.0]`, alpha=0.25 frozen), and raises `KeyError` on missing teams per D5.
3. Add `MarketValueElo` to `default_roster()`. Skip the LOTO gate per the caveat above — the predictive claim is forward-looking and the live calibration hero + DD4 banner are the right validation surface.

That's a focused 2-3 hour PR when the operational moment arrives. Out of scope for this session.

## Pattern that ships INSTEAD

Phase 4 (calibration) + Phase 5a (BMA) + Phase 5c (modal scoreline) + Phase 5d (divergence banner) are all in production. The system now has:
- Two new viewer dimensions (calibration track toggle + modal scoreline headline) that directly move the user-visible metric.
- An ensemble layer (BMA) that combines existing models without the single-knob noise penalty.
- A divergence banner that surfaces mid-tournament drift honestly when frozen weights and live observation disagree.

Squad-value data would compound that surface, not replace it. Defer until the manual snapshot is ready.
