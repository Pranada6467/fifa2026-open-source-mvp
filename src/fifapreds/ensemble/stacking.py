"""Logistic stacking over per-model probabilities (S19).

Stacked ensembles are the standard ML follow-on to BMA: instead of one
scalar weight per base model, learn a small logistic regression on the
member predictions themselves. With n=192 backtest fixtures and only
3·N features (N=number of base models), L2 regularization is the
guardrail against overfitting that the plan called out (D11-F).

The shipped weights live in `data/stacking_weights.json` — frozen,
versioned via git, refit ONLY when a new tournament's worth of data
lands via `scripts/train_stacker.py --loto`. See `docs/stacker_refresh.md`
for the cadence contract.

Per D5: missing weights file → raise at construction; member list
mismatch (saved model_ids not subset of supplied members) → raise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from fifapreds.models.base import WDL, Model

_PROB_COLS = ("p_home", "p_draw", "p_away")


class StackedEnsemble(Model):
    """Logistic-stacked combination of per-model probability vectors.

    Construct by passing the fitted members + the path to a frozen
    weights JSON. The JSON contains: `model_ids` (the order used at
    train time), `coefficients` (3 × 3N for multinomial), `intercepts`
    (length 3), `holdout_years` (the LOTO folds reported), and
    `trained_on` (final fit's training set — all years). At predict
    time we re-pack the members' probabilities in the saved model_id
    order and apply softmax(W·x + b).
    """

    model_id: ClassVar[str] = "stacked_ensemble"
    model_version: ClassVar[str] = "1"

    def __init__(self, members: list[Model], *, weights_path: Path | str):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"stacking weights file missing at {self.weights_path}; "
                "run scripts/train_stacker.py first"
            )
        payload = json.loads(self.weights_path.read_text())
        self._model_ids: list[str] = list(payload["model_ids"])
        self._coef = np.asarray(payload["coefficients"], dtype=float)
        self._intercepts = np.asarray(payload["intercepts"], dtype=float)
        # Member alignment: every saved model_id must be supplied. Extra
        # members are fine — they're ignored at predict time.
        available = {m.model_id: m for m in members}
        missing = [mid for mid in self._model_ids if mid not in available]
        if missing:
            raise ValueError(
                f"stacking weights expect members {missing!r} that were not "
                f"supplied (got: {sorted(available)})"
            )
        self._members = [available[mid] for mid in self._model_ids]
        cutoffs = [m.trained_through for m in self._members
                   if m.trained_through is not None]
        self.trained_through = min(cutoffs) if cutoffs else None
        self._payload_meta = {
            k: payload[k] for k in ("holdout_years", "trained_on", "C", "weights_version")
            if k in payload
        }

    def fit(self, matches: pd.DataFrame) -> "StackedEnsemble":
        """Members come in PRE-FITTED from the orchestrator; fit refits
        each in place (so the ensemble adopts the new state). The
        stacking coefficients themselves are frozen — refresh them via
        scripts/train_stacker.py, not at predict time."""
        for m in self._members:
            m.fit(matches)
        cutoffs = [m.trained_through for m in self._members
                   if m.trained_through is not None]
        self.trained_through = min(cutoffs) if cutoffs else None
        return self

    def hyperparams(self) -> dict[str, Any]:
        return {
            "members": list(self._model_ids),
            "weights_path": str(self.weights_path),
            **self._payload_meta,
        }

    def predict_wdl(self, home: str, away: str, *, neutral: bool = False) -> WDL:
        feats = np.concatenate([
            m.predict_wdl(home, away, neutral=neutral).as_array()
            for m in self._members
        ])
        logits = self._coef @ feats + self._intercepts
        logits = logits - logits.max()
        e = np.exp(logits)
        probs = e / e.sum()
        return WDL(home=float(probs[0]), draw=float(probs[1]), away=float(probs[2]))
