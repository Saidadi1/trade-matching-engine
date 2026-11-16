"""End-to-end experiment: generate -> block -> features -> train -> evaluate -> figures.

Run with:  python -m tmatch.pipeline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import viz
from .blocking import blocking_stats, candidate_pairs
from .evaluate import classification_metrics, recall_at_k, temporal_split
from .features import FEATURE_LABELS, FEATURES, build_features
from .generate import GenConfig, generate
from .models import explain, get_models, make_lgbm, rule_baseline

plt = viz.plt
ROOT = Path(__file__).resolve().parents[1]
RESULTS, DATA = ROOT / "results", ROOT / "data"


def prepare(cfg: GenConfig):
    a, b = generate(cfg)
    pairs = candidate_pairs(a, b)
    X = build_features(pairs)
    y = pairs["is_match"].to_numpy()
    train, test = temporal_split(pairs)
    return a, b, pairs, X, y, train, test


def run_benchmark(cfg: GenConfig) -> dict:
    a, b, pairs, X, y, train, test = prepare(cfg)
    DATA.mkdir(exist_ok=True)
    a.drop(columns="true_trade").head(200).to_csv(DATA / "sample_system_a.csv", index=False)
    b.drop(columns="true_trade").head(200).to_csv(DATA / "sample_system_b.csv", index=False)

    out = {"blocking": blocking_stats(a, b, pairs),
           "split": {"train_pairs": int(train.sum()), "test_pairs": int(test.sum()),
                     "train_matches": int(y[train].sum()), "test_matches": int(y[test].sum())},
           "models": {}}
    scores = {"Exact-match rules": rule_baseline(X[test])}
    fitted = {}
    for name, model in get_models().items():
        t0 = time.time()
        model.fit(X[train], y[train])
        scores[name] = model.predict_proba(X[test])[:, 1]
        fitted[name] = model
        print(f"  trained {name} in {time.time() - t0:.1f}s")
    for name, s in scores.items():
        m = classification_metrics(y[test], s)
        m.update(recall_at_k(pairs.loc[test, "record_id_b"], y[test], s))
        out["models"][name] = m

    _fig_blocking(out["blocking"])
    _fig_model_comparison(out["models"])
    _fig_pr_curves(y[test], scores)
    shap = explain(fitted["LightGBM"], X[test])
    out["shap_importance"] = shap.abs().mean().sort_values(ascending=False).to_dict()
    _fig_shap(out["shap_importance"])
    return out


def run_missing_id_stress(rates=(0.0, 0.1, 0.3, 0.5, 0.8, 1.0), n_trades: int = 3000) -> list[dict]:
    """Retrain and re-test as more and more references go missing."""
    rows = []
    for p in rates:
        _, _, pairs, X, y, train, test = prepare(GenConfig(n_trades=n_trades, p_missing_id=p, seed=7))
        model = make_lgbm().fit(X[train], y[train])
        s = model.predict_proba(X[test])[:, 1]
        m = classification_metrics(y[test], s)
        m.update(recall_at_k(pairs.loc[test, "record_id_b"], y[test], s))
        rules = classification_metrics(y[test], rule_baseline(X[test]))
        rows.append({"p_missing_id": p, "f1": m["f1"], "precision": m["precision"],
                     "recall": m["recall"], "pr_auc": m["pr_auc"], "recall@1": m["recall@1"],
                     "rules_f1": rules["f1"], "rules_recall": rules["recall"]})
        print(f"  missing ids {p:.0%}: model F1 {m['f1']:.3f} | rules F1 {rules['f1']:.3f}")
    _fig_stress(pd.DataFrame(rows))
    return rows


# ---------------------------------------------------------------- figures
def _fig_blocking(s: dict):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    labels = ["All possible pairs", "Candidate pairs after blocking", "True matches"]
    vals = [s["all_possible_pairs"], s["candidate_pairs"], s["true_match_pairs"]]
    ax.barh(labels[::-1], vals[::-1], color=viz.BLUE, height=0.55)
    ax.set_xscale("log")
    for i, v in enumerate(vals[::-1]):
        ax.text(v * 1.15, i, f"{v:,}", va="center", color=viz.INK)
    ax.set_xlim(right=max(vals) * 12)
    ax.set_title(f"Blocking removes {s['reduction_ratio']:.1%} of comparisons, "
                 f"keeps {s['pair_completeness']:.1%} of true matches")
    ax.set_xlabel("number of pairs (log scale)")
    ax.grid(axis="y", visible=False)
    viz.save(fig, RESULTS / "blocking_funnel.png")


def _fig_model_comparison(models: dict):
    names = list(models)
    metrics = ["precision", "recall", "f1", "pr_auc"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, metric in zip(axes, metrics):
        vals = [models[n][metric] for n in names]
        ax.barh(names[::-1], vals[::-1], height=0.6,
                color=[viz.MODEL_COLORS[n] for n in names[::-1]])
        for i, v in enumerate(vals[::-1]):
            ax.text(v + 0.02, i, f"{v:.3f}", va="center", color=viz.INK, fontsize=9)
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(metric.upper().replace("_", "-"))
        ax.grid(axis="y", visible=False)
    fig.suptitle("Test-set performance (threshold 0.5)", x=0.01, ha="left", fontweight="bold")
    viz.save(fig, RESULTS / "model_comparison.png")


def _fig_pr_curves(y, scores: dict):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, s in scores.items():
        if name == "Exact-match rules":
            m = classification_metrics(y, s)
            ax.scatter([m["recall"]], [m["precision"]], color=viz.MUTED, s=60, zorder=3,
                       label="Exact-match rules (single point)")
            continue
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=viz.MODEL_COLORS[name], label=name)
    ax.set_xlabel("recall (share of true matches found)")
    ax.set_ylabel("precision (share of predicted matches that are right)")
    ax.set_xlim(0.5, 1.005)
    ax.set_ylim(0.5, 1.005)
    ax.set_title("Precision-recall curves")
    ax.legend(loc="lower left")
    viz.save(fig, RESULTS / "pr_curves.png")


def _fig_shap(importance: dict):
    items = list(importance.items())[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh([FEATURE_LABELS[k] for k, _ in items], [v for _, v in items],
            color=viz.BLUE, height=0.6)
    ax.set_xlabel("mean |SHAP value| (average impact on the match score, log-odds)")
    ax.set_title("Which features drive the LightGBM matching decision")
    ax.grid(axis="y", visible=False)
    viz.save(fig, RESULTS / "shap_importance.png")


def _fig_stress(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = df["p_missing_id"] * 100
    ax.plot(x, df["f1"], marker="o", color=viz.BLUE, label="LightGBM")
    ax.plot(x, df["rules_f1"], marker="o", color=viz.MUTED, label="Exact-match rules")
    ax.text(x.iloc[-1], df["f1"].iloc[-1] + 0.03, f"{df['f1'].iloc[-1]:.2f}", ha="right")
    ax.text(x.iloc[-1], df["rules_f1"].iloc[-1] + 0.03, f"{df['rules_f1'].iloc[-1]:.2f}", ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("% of back-office records with a missing trade reference")
    ax.set_ylabel("F1 on the test set")
    ax.set_title("Robustness when identifiers disappear")
    ax.legend(loc="lower left")
    viz.save(fig, RESULTS / "missing_id_stress.png")


def main():
    RESULTS.mkdir(exist_ok=True)
    print("Benchmark (6,000 trades)")
    out = run_benchmark(GenConfig())
    print("Missing-identifier stress test")
    out["missing_id_stress"] = run_missing_id_stress()
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    table = pd.DataFrame(out["models"]).T[
        ["precision", "recall", "f1", "pr_auc", "false_positive_rate", "recall@1", "recall@3"]]
    print(table.round(4).to_string())
    print(json.dumps(out["blocking"], indent=2, default=float))


if __name__ == "__main__":
    main()
