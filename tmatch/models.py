"""Models compared in the benchmark, plus a rule-based baseline and SHAP-style explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURES


def get_models(seed: int = 0) -> dict:
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                   class_weight="balanced_subsample", random_state=seed)),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                                 eval_metric="logloss", random_state=seed),
        "LightGBM": make_lgbm(seed),
    }


def make_lgbm(seed: int = 0) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                          random_state=seed, verbose=-1)


def rule_baseline(X: pd.DataFrame) -> np.ndarray:
    """What a classic exact-match reconciliation rule would do.

    Match if the references are identical, OR every key field agrees exactly.
    """
    all_keys = ((X["notional_exact"] == 1) & (X["trade_date_diff"] == 0)
                & (X["settle_date_diff"] == 0) & (X["cpty_token_set"] == 1)
                & (X["direction_match"] == 1) & (X["product_match"] == 1))
    return ((X["id_exact"] == 1) | all_keys).astype(float).to_numpy()


def explain(model: LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contribution (TreeSHAP, in log-odds) for each row of X.

    Positive = pushes the pair towards "match", negative = towards "no match".
    """
    contrib = model.booster_.predict(X[FEATURES], pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=X.index)
