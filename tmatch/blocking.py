"""Candidate generation ("blocking").

Comparing every A record with every B record is quadratic (6k x 6k = 36M pairs).
Blocking keeps only pairs that are plausible, using cheap rules:

  pass 1: same currency pair AND trade dates at most `window` days apart
  pass 2: same normalised trade reference (catches anything pass 1 missed)

The two numbers that describe a blocking scheme are
  * reduction ratio   - share of the quadratic comparisons we avoided
  * pair completeness - share of the true matches that survived blocking
"""
from __future__ import annotations

import pandas as pd

from .features import norm_id

RECORD_COLS = ["record_id", "true_trade", "trade_ref", "product", "ccy_pair", "direction",
               "notional", "rate", "trade_date", "settle_date", "counterparty"]


def candidate_pairs(a: pd.DataFrame, b: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    ka = a[["record_id", "ccy_pair", "trade_date"]]
    kb = b[["record_id", "ccy_pair", "trade_date"]]
    p1 = ka.merge(kb, on="ccy_pair", suffixes=("_a", "_b"))
    gap = (p1["trade_date_a"] - p1["trade_date_b"]).abs().dt.days
    p1 = p1.loc[gap <= window, ["record_id_a", "record_id_b"]]

    ia = a.assign(_id=a["trade_ref"].map(norm_id)).dropna(subset=["_id"])[["record_id", "_id"]]
    ib = b.assign(_id=b["trade_ref"].map(norm_id)).dropna(subset=["_id"])[["record_id", "_id"]]
    p2 = ia.merge(ib, on="_id", suffixes=("_a", "_b"))[["record_id_a", "record_id_b"]]

    keys = pd.concat([p1, p2]).drop_duplicates()
    pairs = (keys
             .merge(a[RECORD_COLS].add_suffix("_a"), on="record_id_a")
             .merge(b[RECORD_COLS].add_suffix("_b"), on="record_id_b"))
    pairs["is_match"] = (pairs["true_trade_a"] == pairs["true_trade_b"]).astype(int)
    return pairs.reset_index(drop=True)


def blocking_stats(a: pd.DataFrame, b: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    total = len(a) * len(b)
    true_pairs = int(b["true_trade"].isin(a["true_trade"]).sum())
    return {
        "records_a": len(a),
        "records_b": len(b),
        "all_possible_pairs": total,
        "candidate_pairs": len(pairs),
        "reduction_ratio": 1 - len(pairs) / total,
        "true_match_pairs": true_pairs,
        "pair_completeness": float(pairs["is_match"].sum() / true_pairs),
        "match_rate_in_candidates": float(pairs["is_match"].mean()),
    }
