# FIFA 2026 Live Calibration Engine

Predict the 2026 World Cup, then keep an honest scoreboard of how well-calibrated
the predictions actually are. Team ratings → per-match and tournament probabilities
→ a live loop that re-scores itself and recalibrates after every result, ranking
modelling techniques by out-of-sample log-loss / Brier / RPS.

A learning project: build → check results → build further, using the live tournament
as a continuous feedback loop. The honest claim is "well-calibrated and close to the
market," not "beats the bookmakers."

## Quick start
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install pytest networkx
.venv/bin/python -m pytest -q
.venv/bin/python -m fifapreds.ingest        # build data/processed/matches.parquet
```

Odds capture (optional): copy `.env.example` → `.env`, add a free
[The Odds API](https://the-odds-api.com) key, then `python -m fifapreds.loop.odds`.

## Where things are
- `src/fifapreds/` — the package (ingest, as-of layer, models, sim, loop, publish).
- `scripts/` — reproducible data builders (bracket/routing, groups).
- `docs/` — `PLAN.md` (engineering plan), `DESIGN.md` (rationale),
  `block-v-findings.md` (verified data facts).
- `CLAUDE.md` — full onboarding for AI-assisted work; start there.

## Status
Verification spike complete; calibration spine (ingest + no-lookahead as-of layer +
odds snapshotter) in place with tests. See `CLAUDE.md` for the roadmap.

## Data sources
[martj42/international_results](https://github.com/martj42/international_results)
(history), [openfootball](https://github.com/openfootball/worldcup) / FIFA (fixtures
& rules), [penaltyblog](https://pypi.org/project/penaltyblog/) (scoreline models +
metrics), The Odds API (bookmaker odds). FIFA is the authority for format & routing.
