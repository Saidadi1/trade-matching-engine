"""A small trained matcher shared by the dashboard and the API."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .blocking import candidate_pairs
from .features import FEATURES, build_features
from .generate import GenConfig, generate
from .models import explain, make_lgbm

FIELDS = ["trade_ref", "product", "ccy_pair", "direction", "notional", "rate",
          "trade_date", "settle_date", "counterparty"]


@dataclass
class Matcher:
    model: object
    system_a: pd.DataFrame
    system_b: pd.DataFrame

    @classmethod
    def train_demo(cls, n_trades: int = 2000, seed: int = 1) -> "Matcher":
        a, b = generate(GenConfig(n_trades=n_trades, seed=seed))
        pairs = candidate_pairs(a, b)
        model = make_lgbm().fit(build_features(pairs), pairs["is_match"])
        return cls(model, a, b)

    def score_pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        X = build_features(pairs)
        out = pairs.copy()
        out["match_probability"] = self.model.predict_proba(X)[:, 1]
        return out.join(explain(self.model, X).add_prefix("shap_")).join(X.add_prefix("feat_"))

    def score_two_records(self, rec_a: dict, rec_b: dict) -> dict:
        row = {f"{k}_a": rec_a.get(k) for k in FIELDS} | {f"{k}_b": rec_b.get(k) for k in FIELDS}
        pair = pd.DataFrame([row])
        for col in ["trade_date_a", "trade_date_b", "settle_date_a", "settle_date_b"]:
            pair[col] = pd.to_datetime(pair[col])
        for col in ["notional_a", "notional_b", "rate_a", "rate_b"]:
            pair[col] = pd.to_numeric(pair[col])
        scored = self.score_pairs(pair).iloc[0]
        return {
            "match_probability": float(scored["match_probability"]),
            "features": {f: _clean(scored[f"feat_{f}"]) for f in FEATURES},
            "contributions": {f: float(scored[f"shap_{f}"]) for f in FEATURES},
        }

    def top_candidates(self, record_id_b: str, a: pd.DataFrame | None = None,
                       b: pd.DataFrame | None = None, k: int = 5) -> pd.DataFrame:
        a = self.system_a if a is None else a
        b = self.system_b if b is None else b
        pairs = candidate_pairs(a, b[b["record_id"] == record_id_b])
        if pairs.empty:
            return pairs
        return self.score_pairs(pairs).nlargest(k, "match_probability").reset_index(drop=True)


def _clean(v):
    return None if isinstance(v, float) and np.isnan(v) else float(v)
