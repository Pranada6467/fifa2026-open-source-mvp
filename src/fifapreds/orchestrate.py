"""One-command live loop (T14): the whole on-result pipeline.

    .venv/bin/python -m fifapreds.orchestrate            # after each match day

Steps, in the locked order (CLAUDE.md: score -> ingest -> update -> publish):

1. FETCH (CLI only, `--no-fetch` to skip): refresh data/raw/results.csv +
   shootouts.csv from martj42 and rebuild matches.parquet. Failing soft —
   stale data beats no run.
2. SCORE: grade every unscored prediction whose result is now known.
   Scoring before model updates is the spirit of the locked order; doing the
   data refresh first is safe because prediction rows are immutable and the
   graders' integrity guards (training_cutoff and predicted_at vs kickoff)
   hold regardless of when the result arrived.
3. UPDATE: refit the whole frozen roster (`models.roster.default_roster`) on
   the as-of store, plus a MarketBlend over the latest captured odds snapshot
   when one exists. A Dixon-Coles fit failure drops that entrant with a
   warning instead of killing the loop (the documented fallback).
4. PREDICT: log claims for upcoming WC fixtures (idempotent per
   (fixture, config, cutoff) — rerunning the same day is a no-op).
5. SIMULATE: seeded Monte Carlo per goals-capable model; tournament
   probabilities land in data/tournament_sim.parquet for the publisher.
   Odds capture stays a separate manual/cron job (`loop.odds`) — it spends
   API quota and must not run on every loop iteration.
6. PUBLISH: export artifacts/ for the read-only viewer.

The default seed is the UTC date (YYYYMMDD): reruns within a day reproduce
identical simulations, and the seed is recorded in the sim meta either way.
"""
from __future__ import annotations

import signal
import sqlite3
import ssl
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fifapreds.config import PROJECT_ROOT
from fifapreds.models.base import GoalsModel, Model
from fifapreds.models.market import MarketBlend, latest_h2h_probs
from fifapreds.models.roster import default_roster

MARTJ42_BASE = (
    "https://raw.githubusercontent.com/martj42/international_results/master"
)
RAW_FILES = ("results.csv", "shootouts.csv")
TOURNAMENT_SIM_PARQUET = PROJECT_ROOT / "data" / "tournament_sim.parquet"

# Per-model fit timeout (D4). PyMC's hierarchical sampler can hang or run
# long on CI's slower hardware; capping each fit at 5 min guarantees that one
# slow model costs one entrant, never the whole nightly budget.
FIT_TIMEOUT_SECONDS = 300

_HAS_SIGALRM = hasattr(signal, "SIGALRM")  # False on Windows; we don't ship there


def fetch_raw(raw_dir: Path | None = None) -> list[str]:
    """Refresh the martj42 CSVs in place; returns notes for the report.

    Atomic (tmp + rename) and guarded: a download that parses to *fewer* rows
    than the file it replaces is refused — upstream truncation must never eat
    our local history.
    """
    raw_dir = raw_dir or (PROJECT_ROOT / "data" / "raw")
    try:  # macOS venv pythons ship no CA bundle for urllib; certifi fills in
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    notes = []
    for name in RAW_FILES:
        target = raw_dir / name
        tmp = target.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(f"{MARTJ42_BASE}/{name}", context=ctx,
                                        timeout=60) as resp:
                tmp.write_bytes(resp.read())
            new_rows = len(pd.read_csv(tmp))
            old_rows = len(pd.read_csv(target)) if target.exists() else 0
            if new_rows < old_rows:
                notes.append(f"{name}: refused ({new_rows} rows < local {old_rows})")
                tmp.unlink()
                continue
            tmp.replace(target)
            notes.append(f"{name}: {new_rows} rows")
        except Exception as exc:  # network/parse trouble -> keep local copy
            notes.append(f"{name}: fetch failed, keeping local ({exc})")
            tmp.unlink(missing_ok=True)
    return notes


def _fit_failure_exceptions() -> tuple[type[Exception], ...]:
    """Exception types that drop an entrant instead of killing the loop.

    The loop runs unattended (nightly Action): one bad fit must cost one
    entrant for one night, never the whole run. PyMC (optional dep, E3) is
    listed explicitly — its SamplingError currently derives from RuntimeError,
    but the catch is the contract, not that accident. Resolved at call time so
    installing the optional dep widens the net without a code change.
    """
    excs: list[type[Exception]] = [RuntimeError, ValueError, np.linalg.LinAlgError]
    try:
        from pymc.exceptions import SamplingError

        excs.append(SamplingError)
    except ImportError:
        pass
    return tuple(excs)


@contextmanager
def _fit_timeout(seconds: int):
    """Raise TimeoutError if the with-block doesn't return in `seconds`.

    SIGALRM-based, so Unix-only — the orchestrator runs on Linux CI and macOS
    dev. On Windows the alarm is a no-op (the CI job's outer `timeout-minutes`
    is the only guard there). Not reentrant: only one SIGALRM per process.
    """
    if not _HAS_SIGALRM:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"fit exceeded {seconds}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _fit_roster(
    roster: list[Model],
    played: pd.DataFrame,
    *,
    timeout_s: int = FIT_TIMEOUT_SECONDS,
) -> tuple[list[Model], list[str]]:
    fitted, notes = [], []
    failures = _fit_failure_exceptions()
    for model in roster:
        try:
            with _fit_timeout(timeout_s):
                fitted.append(model.fit(played))
        except TimeoutError as exc:
            notes.append(f"{model.model_id}: fit timed out, dropped ({exc})")
        except failures as exc:
            notes.append(f"{model.model_id}: fit failed, dropped ({exc})")
    return fitted, notes


def _market_entrant(conn: sqlite3.Connection, fitted: list[Model]) -> tuple[Model | None, str]:
    """MarketBlend over the newest odds snapshot, based on the best fitted
    goals model (falls back to any fitted model)."""
    if not fitted:
        return None, "market_blend: skipped (no fitted base)"
    try:
        snapshot_id, market = latest_h2h_probs(conn)
    except LookupError as exc:
        return None, f"market_blend: skipped ({exc})"
    base = next((m for m in fitted if isinstance(m, GoalsModel)), fitted[0])
    blend = MarketBlend(base=base, market=market, snapshot_id=snapshot_id)
    return blend, f"market_blend: over snapshot {snapshot_id} (base {base.model_id})"


def run_simulations(
    fitted: list[Model],
    matches: pd.DataFrame,
    *,
    n_sims: int,
    seed: int,
    out_path: Path | str = TOURNAMENT_SIM_PARQUET,
) -> tuple[pd.DataFrame, list[dict]]:
    """One tournament Monte Carlo per goals-capable entrant; the combined
    long-format table is written for the publisher to pick up."""
    from fifapreds.sim.montecarlo import simulate_tournament

    frames, metas = [], []
    for model in fitted:
        if not isinstance(model, GoalsModel):
            continue
        summary, meta = simulate_tournament(
            model, n_sims=n_sims, seed=seed, matches=matches
        )
        summary.insert(0, "model_id", model.model_id)
        frames.append(summary)
        metas.append(meta)
    if not frames:
        return pd.DataFrame(), []
    combined = pd.concat(frames, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    return combined, metas


def run(
    conn: sqlite3.Connection,
    store,
    roster: list[Model] | None = None,
    *,
    days: int | None = None,
    n_sims: int = 10_000,
    seed: int | None = None,
    sim_path: Path | str = TOURNAMENT_SIM_PARQUET,
    artifacts_dir: Path | str | None = None,
    live_db: Path | str | None = None,
) -> dict:
    """SCORE -> UPDATE -> PREDICT -> SIMULATE -> PUBLISH over an open store.

    Injectable everywhere (conn, store, roster, paths) so the E2E test runs
    the identical path against a synthetic world. Returns a step report.
    """
    from fifapreds.loop.predict import predict_upcoming
    from fifapreds.loop.score import score_pending, score_scoreline_pending
    from fifapreds.publish import artifacts

    report: dict = {"notes": []}
    seed = seed if seed is not None else int(
        datetime.now(timezone.utc).strftime("%Y%m%d")
    )

    # SCORE — grade what's now resolvable, before anything else moves.
    scored = score_pending(conn, store)
    report["scored"] = scored["scored"]
    report["score_pending"] = scored["pending"]
    report["violations"] = scored["violations"]

    # E6b: scoreline grading for goals-model predictions with stored grids.
    scoreline = score_scoreline_pending(conn, store)
    report["scoreline_scored"] = scoreline["scored"]
    report["scoreline_skipped_et"] = scoreline["skipped_et"]

    # UPDATE — refit the frozen roster on the as-of store.
    fitted, fit_notes = _fit_roster(roster if roster is not None else default_roster(),
                                    store.played)
    report["notes"] += fit_notes
    market, market_note = _market_entrant(conn, fitted)
    report["notes"].append(market_note)
    entrants = fitted + ([market] if market is not None else [])
    report["models"] = [m.model_id for m in entrants]

    # PREDICT — idempotent per-fixture claims for the window.
    report["predicted"] = predict_upcoming(conn, entrants, store, days=days)

    # SIMULATE — trophy odds per goals model.
    combined, sim_metas = run_simulations(
        fitted, store.all, n_sims=n_sims, seed=seed, out_path=sim_path
    )
    report["simulated"] = {m["model_id"]: m["n_sims"] for m in sim_metas}
    report["sim_metas"] = sim_metas

    # PUBLISH — refresh the viewer's artifacts.
    publish_kwargs = {"store": store, "tournament_src": sim_path}
    if artifacts_dir is not None:
        publish_kwargs["out_dir"] = artifacts_dir
    if live_db is not None:
        publish_kwargs["live_db"] = live_db
    report["artifacts"] = artifacts.build(**publish_kwargs)
    return report


def exit_code(report: dict) -> int:
    """0 = clean run; 2 = integrity violations. The nightly Action gates on
    this — a non-empty violations list must turn the build red, never pass
    silently."""
    return 2 if report["violations"] else 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    from fifapreds import db
    from fifapreds.asof import MatchStore
    from fifapreds.ingest import build_matches

    ap = argparse.ArgumentParser(description="Run the full on-result pipeline.")
    ap.add_argument("--days", type=int, default=None,
                    help="prediction window in days (default: all remaining fixtures)")
    ap.add_argument("--n-sims", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=None,
                    help="Monte Carlo seed (default: UTC date as YYYYMMDD)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip the martj42 refresh (use local data as-is)")
    ap.add_argument("--db", default=None, help="SQLite path (default data/fifa2026.db)")
    args = ap.parse_args(argv)

    if not args.no_fetch:
        for note in fetch_raw():
            print(f"fetch: {note}")
        m = build_matches()
        print(f"ingest: {len(m)} rows, played through "
              f"{m.loc[m['is_played'], 'date'].max().date()}")

    conn = db.connect(args.db)
    store = MatchStore()
    report = run(conn, store, days=args.days, n_sims=args.n_sims, seed=args.seed,
                 live_db=args.db if args.db else None)

    print(f"score: {report['scored']} graded, {report['score_pending']} awaiting results")
    print(f"scoreline: {report['scoreline_scored']} graded, "
          f"{report['scoreline_skipped_et']} skipped (ET)")
    for note in report["notes"]:
        print(f"update: {note}")
    for model_id, n in report["predicted"].items():
        print(f"predict: {model_id}: {n} new claims")
    for model_id, n in report["simulated"].items():
        print(f"simulate: {model_id}: {n} tournaments")
    counts = report["artifacts"]["counts"]
    print("publish: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    # Close before exit so the WAL checkpoints — CI commits the .db file and
    # must never snapshot it with writes still sitting in the sidecar.
    conn.close()
    violations = report["violations"]
    print(f"violations: {len(violations)}"
          + (f" — prediction_ids {violations}" if violations else ""))
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
