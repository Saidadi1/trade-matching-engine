"""Pairwise features: turn (record A, record B) into numbers a model can learn from."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

LEGAL_TOKENS = {"SA", "S.A.", "SA.", "PLC", "AG", "LTD", "NV", "INC", "LLC", "SE", "CORP"}

FEATURES = [
    "id_exact",            # normalised references are identical
    "id_similarity",       # 0..1 similarity of the references (catches typos)
    "id_missing",          # one side has no reference
    "notional_rel_diff",   # |a-b| / max(a,b)
    "notional_exact",
    "trade_date_diff",     # days
    "settle_date_diff",    # days (NaN when missing)
    "settle_missing",
    "rate_rel_diff",       # |a-b| / a (NaN when missing)
    "rate_missing",
    "cpty_jaro_winkler",   # character-level name similarity
    "cpty_token_set",      # word-level name similarity (ignores order / extra words)
    "direction_match",
    "product_match",
]

FEATURE_LABELS = {
    "id_exact": "Trade reference identical",
    "id_similarity": "Trade reference similarity",
    "id_missing": "Trade reference missing",
    "notional_rel_diff": "Notional relative difference",
    "notional_exact": "Notional identical",
    "trade_date_diff": "Trade date gap (days)",
    "settle_date_diff": "Settlement date gap (days)",
    "settle_missing": "Settlement date missing",
    "rate_rel_diff": "Rate relative difference",
    "rate_missing": "Rate missing",
    "cpty_jaro_winkler": "Counterparty name similarity (Jaro-Winkler)",
    "cpty_token_set": "Counterparty name similarity (token set)",
    "direction_match": "Same direction (buy/sell)",
    "product_match": "Same product",
}


def norm_id(x) -> str | None:
    """'TRD-123456', 'REF/123456', '123456' -> '123456'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    digits = re.sub(r"\D", "", str(x))
    return digits or None


def norm_name(x) -> str:
    """Upper-case, strip punctuation and legal suffixes: 'BNP Paribas S.A.' -> 'BNP PARIBAS'."""
    tokens = re.sub(r"[^A-Z0-9 ]", " ", str(x).upper().replace("S.A.", "SA")).split()
    kept = [t for t in tokens if t not in LEGAL_TOKENS]
    return " ".join(kept or tokens)


def _cached(func, left: list[str], right: list[str]) -> np.ndarray:
    cache = {k: func(*k) for k in set(zip(left, right))}
    return np.array([cache[k] for k in zip(left, right)])


def build_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """`pairs` holds the columns of both records with suffixes _a and _b."""
    f = pd.DataFrame(index=pairs.index)

    ida = pairs["trade_ref_a"].map(norm_id)
    idb = pairs["trade_ref_b"].map(norm_id)
    missing = ida.isna() | idb.isna()
    f["id_exact"] = ((ida == idb) & ~missing).astype(int)
    f["id_similarity"] = [0.0 if m else fuzz.ratio(x, y) / 100
                          for x, y, m in zip(ida, idb, missing)]
    f["id_missing"] = missing.astype(int)

    na, nb = pairs["notional_a"].astype(float), pairs["notional_b"].astype(float)
    f["notional_rel_diff"] = (na - nb).abs() / np.maximum(na, nb)
    f["notional_exact"] = (na == nb).astype(int)

    f["trade_date_diff"] = (pairs["trade_date_a"] - pairs["trade_date_b"]).abs().dt.days
    f["settle_date_diff"] = (pairs["settle_date_a"] - pairs["settle_date_b"]).abs().dt.days
    f["settle_missing"] = f["settle_date_diff"].isna().astype(int)

    ra, rb = pairs["rate_a"].astype(float), pairs["rate_b"].astype(float)
    f["rate_rel_diff"] = (ra - rb).abs() / ra
    f["rate_missing"] = f["rate_rel_diff"].isna().astype(int)

    ca = pairs["counterparty_a"].map(norm_name).tolist()
    cb = pairs["counterparty_b"].map(norm_name).tolist()
    f["cpty_jaro_winkler"] = _cached(JaroWinkler.normalized_similarity, ca, cb)
    f["cpty_token_set"] = _cached(fuzz.token_set_ratio, ca, cb) / 100

    f["direction_match"] = (pairs["direction_a"] == pairs["direction_b"]).astype(int)
    f["product_match"] = (pairs["product_a"] == pairs["product_b"]).astype(int)
    return f[FEATURES].astype(float)
