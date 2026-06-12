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

## Live loop + viewer
```bash
.venv/bin/python -m fifapreds.loop.predict       # log predictions (8-day window)
.venv/bin/python -m fifapreds.publish.artifacts  # export committed artifacts/
.venv/bin/python -m streamlit run app/streamlit_app.py
```
The Streamlit app is a read-only viewer over `artifacts/` — to deploy, push to
GitHub and point [Streamlit Community Cloud](https://share.streamlit.io) at
`app/streamlit_app.py`; it redeploys whenever fresh artifacts are committed.
Backtest gate: `.venv/bin/python -m fifapreds.backtest` (WC 2014/18/22 replay).

## Automated nightly loop
A scheduled GitHub Action (`.github/workflows/nightly.yml`, 06:30 UTC) runs
`python -m fifapreds.orchestrate` and commits the updated `data/fifa2026.db`
plus `artifacts/` back to the repo.

**CI is the sole writer of `data/fifa2026.db`.** The DB is committed so the
append-only predictions log survives stateless CI runs — which means running
`orchestrate` locally on the same branch forks the DB and produces binary merge
conflicts. Rules:
- Don't run `python -m fifapreds.orchestrate` locally anymore; force a run via
  the Action's `workflow_dispatch` instead. For local experiments point it at a
  scratch DB: `--db /tmp/scratch.db --no-fetch`.
- The odds snapshotter (`python -m fifapreds.loop.odds`) stays manual (API
  quota) and also writes the DB: `git pull` first, run it, commit + push the DB
  immediately, and stay clear of the nightly window (~06:30 UTC).
- The Action **fails loudly** (exit 2, nothing committed) when the scorer
  reports integrity violations — a red nightly build means stop and
  investigate; the `violations:` line in the job log lists the offending
  prediction ids.

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
