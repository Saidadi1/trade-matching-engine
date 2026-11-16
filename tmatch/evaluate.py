"""Evaluation helpers: leak-free split, classification metrics, Recall@K."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def temporal_split(pairs: pd.DataFrame, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Train on the past, test on the future.

    A pair goes to train only if BOTH records are before the cut-off date, and to
    test only if both are after it. Pairs that straddle the cut-off are dropped,
    so no record (and no underlying trade) is ever seen on both sides.
    """
    days = np.sort(pairs["trade_date_a"].unique())
    cutoff = days[int(len(days) * train_frac)]
    a_before, b_before = pairs["trade_date_a"] < cutoff, pairs["trade_date_b"] < cutoff
    return (a_before & b_before).to_numpy(), (~a_before & ~b_before).to_numpy()


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "false_positive_rate": fp / (fp + tn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def recall_at_k(record_b: pd.Series, y: np.ndarray, score: np.ndarray, ks=(1, 3, 5)) -> dict:
    """For each B record that has a true match: is that match in its top-K candidates?"""
    df = pd.DataFrame({"b": record_b.to_numpy(), "y": y, "s": score})
    df["rank"] = df.groupby("b")["s"].rank(method="first", ascending=False)
    hits = df[df["y"] == 1]
    return {f"recall@{k}": float((hits["rank"] <= k).mean()) for k in ks}
