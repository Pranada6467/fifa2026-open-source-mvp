"""Streamlit viewer smoke tests: renders real artifact files without raising,
and renders quiet empty states (never a crash) when artifacts are missing.
The app is a pure file reader — these tests are the read-only contract.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.test_artifacts import build_synthetic_artifacts

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


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
    # upcoming + leaderboard + recently-scored tables all rendered
    assert len(at.dataframe) == 3
    # meta caption present, no empty-state hints
    assert "Data through 2022-01-01" in at.caption[0].value
    assert not any("No data yet" in info.value for info in at.info)


def test_app_empty_states_instead_of_crash(tmp_path, monkeypatch):
    at = _run(tmp_path / "empty", monkeypatch)
    # Every section (meta + tournament + 4 data sections) shows its hint.
    assert len(at.info) == 6
    assert len(at.dataframe) == 0
