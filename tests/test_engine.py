import numpy as np
import pandas as pd
import pytest

from tmatch.blocking import blocking_stats, candidate_pairs
from tmatch.evaluate import classification_metrics, recall_at_k, temporal_split
from tmatch.features import FEATURES, build_features, norm_id, norm_name
from tmatch.generate import GenConfig, generate
from tmatch.service import Matcher


@pytest.fixture(scope="module")
def data():
    a, b = generate(GenConfig(n_trades=600, seed=3))
    return a, b, candidate_pairs(a, b)


def test_normalisers():
    assert norm_id("TRD-123456") == norm_id("REF/123456") == "123456"
    assert norm_id(None) is None and norm_id(float("nan")) is None
    assert norm_name("BNP Paribas S.A.") == norm_name("bnp  paribas") == "BNP PARIBAS"


def test_generator_is_reproducible_and_noisy(data):
    a, b, _ = data
    a2, b2 = generate(GenConfig(n_trades=600, seed=3))
    pd.testing.assert_frame_equal(a, a2)
    pd.testing.assert_frame_equal(b, b2)
    assert a["record_id"].is_unique and b["record_id"].is_unique
    assert 0.2 < b["trade_ref"].isna().mean() < 0.4  # ~30% missing references


def test_blocking_keeps_matches_and_cuts_pairs(data):
    a, b, pairs = data
    stats = blocking_stats(a, b, pairs)
    assert stats["pair_completeness"] > 0.99
    assert stats["reduction_ratio"] > 0.95
    assert not pairs.duplicated(["record_id_a", "record_id_b"]).any()


def test_features_have_expected_shape_and_ranges(data):
    _, _, pairs = data
    X = build_features(pairs)
    assert list(X.columns) == FEATURES and len(X) == len(pairs)
    for col in ["id_similarity", "cpty_jaro_winkler", "cpty_token_set"]:
        assert X[col].between(0, 1).all()
    assert (X["notional_rel_diff"] >= 0).all()


def test_temporal_split_has_no_leakage(data):
    _, _, pairs = data
    train, test = temporal_split(pairs)
    assert not (train & test).any()
    for side in ["a", "b"]:
        overlap = set(pairs.loc[train, f"true_trade_{side}"]) & set(pairs.loc[test, f"true_trade_{side}"])
        assert not overlap
    assert pairs.loc[train, "trade_date_a"].max() < pairs.loc[test, "trade_date_a"].min()


def test_metrics_on_known_case():
    y = np.array([1, 1, 0, 0])
    m = classification_metrics(y, np.array([0.9, 0.2, 0.8, 0.1]))
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["tp"] == 1 and m["fp"] == 1
    r = recall_at_k(pd.Series(["b1", "b1", "b2", "b2"]), y=np.array([1, 0, 0, 1]),
                    score=np.array([0.9, 0.1, 0.8, 0.3]), ks=(1, 2))
    assert r == {"recall@1": 0.5, "recall@2": 1.0}


def test_matcher_separates_an_obvious_match_from_a_non_match():
    m = Matcher.train_demo(n_trades=600)
    rec = dict(trade_ref="TRD-555111", product="FX_SPOT", ccy_pair="EUR/USD", direction="BUY",
               notional=1_000_000, rate=1.0850, trade_date="2025-02-03",
               settle_date="2025-02-05", counterparty="BNP PARIBAS")
    same = rec | dict(trade_ref=None, notional=1_000_500, counterparty="BNP Paribas SA")
    other = rec | dict(trade_ref="TRD-999000", notional=5_000_000, rate=1.0990,
                       counterparty="DEUTSCHE BANK", direction="SELL")
    good, bad = m.score_two_records(rec, same), m.score_two_records(rec, other)
    assert good["match_probability"] > 0.9 > 0.1 > bad["match_probability"]
    assert set(good["contributions"]) == set(FEATURES)


def test_api_and_dashboard_smoke():
    from fastapi.testclient import TestClient
    from streamlit.testing.v1 import AppTest

    from api.main import app

    rec = dict(trade_ref="TRD-1", product="FX_SPOT", ccy_pair="EUR/USD", direction="BUY",
               notional=1e6, rate=1.085, trade_date="2025-02-03", settle_date="2025-02-05",
               counterparty="UBS")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        res = client.post("/match", json={"record_a": rec, "record_b": rec})
        assert res.status_code == 200 and res.json()["match_probability"] > 0.9

    at = AppTest.from_file("app/dashboard.py", default_timeout=120).run()
    assert not at.exception
