# TODOs

## Model-vs-market agreement scatter (post-group-stage)
- **What:** Scatter of model probability vs de-vigged market probability per
  claim (diagonal = perfect agreement), inside the board's audit expander.
- **Why:** The richest single visual of the project's stated win condition
  ("well-calibrated and close to the market") once enough claims accumulate.
- **Context:** The E2 design review (2026-06-12, decision D6) chose the daily
  "Where we differ from the market" list for launch because a scatter is
  sparse and cryptic at week-1 sample sizes. Build the scatter when odds
  coverage x graded claims is large enough to fill it — roughly after the
  group stage. Source data already exists: `artifacts/disagreement.parquet`
  (pre-match) joined to `artifacts/scored.parquet` (outcomes).
- **Depends on:** a few weeks of graded live claims with `odds_snapshot_id`
  coverage; E4's uncertainty machinery makes the agreement bands honest.
