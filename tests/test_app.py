"""Streamlit board smoke tests (E2): renders real artifact files without
raising, shows the designed empty states (never a crash) when artifacts are
missing, and raises the stale-data trust banner when the nightly went quiet.
The app is a pure file reader — these tests are the read-only contract.
"""
from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.test_artifacts import build_synthetic_artifacts

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

HEADERS = ["Does 70% mean 70%?", "Which technique is winning?",
           "And at picking the score?",
           "Could it pick the group-stage survivors?",
           "Where we differ from the market", "The system didn't see these coming",
           "Match odds", "Who wins the World Cup?", "Audit trail"]


def _run(artifacts_dir, monkeypatch) -> AppTest:
    monkeypatch.setenv("FIFAPREDS_ARTIFACTS", str(artifacts_dir))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_app_renders_artifacts(tmp_path, monkeypatch):
    art_dir, _ = build_synthetic_artifacts(tmp_path)
    at = _run(art_dir, monkeypatch)
    assert at.title[0].value == "FIFA 2026 — Live Calibration Engine"
    # The D1 page story, in order.
    assert [h.value for h in at.header] == HEADERS
    # leaderboard view + match per-model + slate scan + 2 audit tables
    assert len(at.dataframe) == 5
    assert len(at.selectbox) >= 1
    assert "Data through 2022-01-01" in at.caption[1].value
    assert not any("No data yet" in info.value for info in at.info)
    # No stale banner on fresh artifacts.
    assert len(at.warning) == 0
    # The leaderboard's coin-flip reference row is present.
    lb_view = at.dataframe[0].value
    assert "— coin-flip (uniform) —" in set(lb_view["model_id"])


def test_app_empty_states_instead_of_crash(tmp_path, monkeypatch):
    at = _run(tmp_path / "empty", monkeypatch)
    # Every section shows its designed empty state (D2), none crash:
    # meta, calibration hero, leaderboard, scoreline accuracy, qualification,
    # disagreement, surprises, match odds, tournament, audit scored, audit
    # leaderboard.
    assert len(at.info) == 11
    assert len(at.dataframe) == 0
    # The page story is still fully present.
    assert [h.value for h in at.header] == HEADERS
    # Empty states name the next update, not a dev command.
    assert any("06:30 UTC" in info.value for info in at.info)


def test_app_stale_artifacts_banner(tmp_path, monkeypatch):
    art_dir, meta = build_synthetic_artifacts(tmp_path)
    meta["generated_at"] = "2020-01-01T00:00:00+00:00"
    (art_dir / "meta.json").write_text(json.dumps(meta))
    at = _run(art_dir, monkeypatch)
    assert len(at.warning) == 1
    assert "nightly update appears to have failed" in at.warning[0].value
