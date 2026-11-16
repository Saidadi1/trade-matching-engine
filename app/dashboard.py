"""Analyst dashboard:  streamlit run app/dashboard.py

Pick a back-office record, see its best candidate trades, the match probability
and which features pushed the decision up or down.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from tmatch import viz
from tmatch.features import FEATURE_LABELS, FEATURES
from tmatch.service import FIELDS, Matcher

st.set_page_config(page_title="Trade Matching Engine", layout="wide")


@st.cache_resource(show_spinner="Training the demo model (a few seconds)...")
def load_matcher() -> Matcher:
    return Matcher.train_demo()


def read_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["trade_date", "settle_date"])
    missing = {"record_id", *FIELDS} - set(df.columns)
    if missing:
        st.error(f"Missing columns: {sorted(missing)}")
        st.stop()
    if "true_trade" not in df.columns:
        df["true_trade"] = -1  # unknown ground truth
    return df


matcher = load_matcher()
st.title("Trade Matching Engine")
st.caption("Does this back-office record describe the same trade as a front-office booking?")

with st.sidebar:
    st.header("Data")
    up_a = st.file_uploader("System A (front office) CSV", type="csv")
    up_b = st.file_uploader("System B (back office) CSV", type="csv")
    st.caption("Leave empty to use the built-in synthetic demo data. "
               "Uploaded files need the columns of data/sample_system_a.csv.")
    top_k = st.slider("Candidates to show", 1, 10, 5)
    threshold = st.slider("Match threshold", 0.05, 0.95, 0.50, 0.05)

a = read_upload(up_a) if up_a else matcher.system_a
b = read_upload(up_b) if up_b else matcher.system_b

record_id = st.selectbox("Back-office record to match", b["record_id"].head(500))
cands = matcher.top_candidates(record_id, a, b, k=top_k)

if cands.empty:
    st.warning("No candidate trade survived blocking: this record is unmatched.")
    st.stop()

best = cands.iloc[0]
prob = best["match_probability"]
c1, c2, c3 = st.columns(3)
c1.metric("Best candidate", best["record_id_a"])
c2.metric("Match probability", f"{prob:.1%}")
c3.metric("Decision", "MATCH" if prob >= threshold else "NO MATCH - send to analyst")

st.subheader("Candidates ranked by probability")
st.dataframe(
    cands[["record_id_a", "match_probability", "trade_ref_a", "counterparty_a", "notional_a",
           "rate_a", "trade_date_a", "settle_date_a"]]
    .style.format({"match_probability": "{:.1%}", "notional_a": "{:,.2f}"}),
    use_container_width=True, hide_index=True)

pick = st.selectbox("Inspect candidate", cands["record_id_a"])
row = cands[cands["record_id_a"] == pick].iloc[0]

left, right = st.columns(2)
with left:
    st.subheader("Side by side")
    side = pd.DataFrame({"System A": [row[f"{f}_a"] for f in FIELDS],
                         "System B": [row[f"{f}_b"] for f in FIELDS]}, index=FIELDS).astype(str)
    side["same?"] = ["yes" if x == y else "no" for x, y in zip(side["System A"], side["System B"])]
    st.table(side)
with right:
    st.subheader("Why the model decided this")
    shap = pd.Series({f: row[f"shap_{f}"] for f in FEATURES}).sort_values()
    fig, ax = viz.plt.subplots(figsize=(6, 4.2))
    ax.barh([FEATURE_LABELS[f] for f in shap.index], shap.values,
            color=[viz.BLUE if v >= 0 else viz.ORANGE for v in shap.values], height=0.6)
    ax.axvline(0, color=viz.MUTED, linewidth=1)
    ax.set_xlabel("contribution to the match score  (left = against, right = in favour)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Blue bars push towards MATCH, orange bars push towards NO MATCH (SHAP values).")
