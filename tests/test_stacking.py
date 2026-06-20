"""StackedEnsemble (S19) — frozen-weights loading, member alignment,
the D5 failure-loud paths, and a basic prediction-shape contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fifapreds.ensemble import StackedEnsemble
from fifapreds.models.base import WDL, Model


class _FakeWDL(Model):
    model_version = "1"

    def __init__(self, model_id: str, wdl: tuple[float, float, float]):
        self.model_id = model_id
        self._wdl = wdl
        self.trained_through = pd.Timestamp("2026-01-01")

    def fit(self, matches): return self
    def hyperparams(self): return {"wdl": list(self._wdl)}
    def predict_wdl(self, home, away, *, neutral=False):
        return WDL(*self._wdl)


def _identity_weights_payload(model_ids: list[str]) -> dict:
    """Coefficients that map directly to the FIRST member's probabilities —
    the simplest sanity payload (weights[c, c+0] = 1 for the first model,
    everything else zero, intercepts zero)."""
    n_features = 3 * len(model_ids)
    coef = np.zeros((3, n_features))
    coef[0, 0] = 1.0  # home class reads first member's p_home
    coef[1, 1] = 1.0  # draw class reads first member's p_draw
    coef[2, 2] = 1.0  # away class reads first member's p_away
    return {
        "weights_version": "1",
        "model_ids": list(model_ids),
        "coefficients": coef.tolist(),
        "intercepts": [0.0, 0.0, 0.0],
        "C": 0.5,
        "holdout_years": [2014, 2018, 2022],
        "trained_on": [2014, 2018, 2022],
    }


@pytest.fixture()
def weights_file(tmp_path: Path) -> Path:
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(_identity_weights_payload(["m_a", "m_b"])))
    return path


def test_stacking_loads_frozen_weights(weights_file):
    members = [_FakeWDL("m_a", (0.6, 0.2, 0.2)),
               _FakeWDL("m_b", (0.2, 0.2, 0.6))]
    ensemble = StackedEnsemble(members, weights_path=weights_file)
    # hyperparams surface the saved order + provenance.
    hp = ensemble.hyperparams()
    assert hp["members"] == ["m_a", "m_b"]
    assert hp["weights_path"] == str(weights_file)
    assert hp["holdout_years"] == [2014, 2018, 2022]


def test_stacking_predict_shape_and_sums_to_one(weights_file):
    members = [_FakeWDL("m_a", (0.6, 0.2, 0.2)),
               _FakeWDL("m_b", (0.2, 0.2, 0.6))]
    ensemble = StackedEnsemble(members, weights_path=weights_file)
    out = ensemble.predict_wdl("H", "A")
    assert out.home + out.draw + out.away == pytest.approx(1.0)
    # With identity-on-first-member weights, softmax over (0.6, 0.2, 0.2)
    # concentrates mass on home — should be > draw and > away.
    assert out.home > out.draw and out.home > out.away


def test_stacking_missing_weights_file_raises(tmp_path):
    members = [_FakeWDL("m_a", (0.6, 0.2, 0.2))]
    with pytest.raises(FileNotFoundError, match="missing"):
        StackedEnsemble(members, weights_path=tmp_path / "ghost.json")


def test_stacking_missing_member_raises(weights_file):
    """Weights expect m_a and m_b; supplying only m_a must raise loudly
    rather than silently zeroing out the missing feature columns."""
    members = [_FakeWDL("m_a", (0.6, 0.2, 0.2))]
    with pytest.raises(ValueError, match="m_b"):
        StackedEnsemble(members, weights_path=weights_file)


def test_stacking_extra_members_are_ignored_with_no_error(weights_file):
    """Saved weights expect m_a + m_b; passing m_a + m_b + m_c is fine
    (the extra goes unused)."""
    members = [_FakeWDL("m_a", (0.6, 0.2, 0.2)),
               _FakeWDL("m_b", (0.2, 0.2, 0.6)),
               _FakeWDL("m_c", (1/3, 1/3, 1/3))]
    ensemble = StackedEnsemble(members, weights_path=weights_file)
    # The ensemble's member list only holds the saved ones.
    assert ensemble.hyperparams()["members"] == ["m_a", "m_b"]


def test_stacking_trained_through_is_min_across_saved_members(weights_file):
    a = _FakeWDL("m_a", (0.6, 0.2, 0.2))
    a.trained_through = pd.Timestamp("2026-06-15")
    b = _FakeWDL("m_b", (0.2, 0.2, 0.6))
    b.trained_through = pd.Timestamp("2026-06-10")
    ensemble = StackedEnsemble([a, b], weights_path=weights_file)
    assert ensemble.trained_through == pd.Timestamp("2026-06-10")


def test_stacking_hyperparams_hash_distinct_per_weights_file(tmp_path):
    payload_a = _identity_weights_payload(["m_a", "m_b"])
    payload_b = _identity_weights_payload(["m_a", "m_b"])
    payload_b["coefficients"][0][0] = 2.0  # alter one coefficient
    pa = tmp_path / "a.json"; pa.write_text(json.dumps(payload_a))
    pb = tmp_path / "b.json"; pb.write_text(json.dumps(payload_b))
    members = [_FakeWDL("m_a", (0.6, 0.2, 0.2)),
               _FakeWDL("m_b", (0.2, 0.2, 0.6))]
    e1 = StackedEnsemble(members, weights_path=pa)
    e2 = StackedEnsemble(members, weights_path=pb)
    # The hash includes weights_path, so two different files always
    # produce two different hashes — auditable provenance for a refit.
    assert e1.hyperparams_hash != e2.hyperparams_hash
