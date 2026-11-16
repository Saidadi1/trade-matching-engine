"""Synthetic trade data for two systems that record the same trades differently.

System A = front-office booking system (clean, one record per trade).
System B = back-office / confirmation system (noisy copy of the same trades).

The noise mimics what happens in real reconciliation: missing references,
re-formatted IDs, small notional differences (fees, rounding), date shifts
(time zones), counterparty name variants, duplicates and unmatched records.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Currency pair -> typical market rate
CCY_PAIRS = {
    "EUR/USD": 1.0850, "GBP/USD": 1.2700, "USD/JPY": 151.00, "EUR/GBP": 0.8550,
    "USD/CHF": 0.8800, "AUD/USD": 0.6600, "EUR/JPY": 164.00, "USD/CAD": 1.3600,
}
PRODUCTS = ["FX_SPOT", "FX_FORWARD", "FX_SWAP"]

# Canonical legal name -> short code. Several names are deliberately close to
# each other (different legal entities of the same group): these are the hard cases.
COUNTERPARTIES = {
    "BNP PARIBAS": "BNPP", "BNP PARIBAS SECURITIES SERVICES": "BP2S",
    "SOCIETE GENERALE": "SOCGEN", "CREDIT AGRICOLE CIB": "CACIB",
    "NATIXIS": "NATX", "DEUTSCHE BANK": "DB", "COMMERZBANK": "COBA",
    "HSBC BANK": "HSBC", "HSBC CONTINENTAL EUROPE": "HSBCCE",
    "BARCLAYS BANK": "BARC", "BARCLAYS CAPITAL SECURITIES": "BARCAP",
    "JP MORGAN CHASE BANK": "JPM", "JP MORGAN SECURITIES": "JPMS",
    "GOLDMAN SACHS INTERNATIONAL": "GSI", "GOLDMAN SACHS BANK EUROPE": "GSBE",
    "MORGAN STANLEY": "MS", "CITIBANK": "CITI", "CITIGROUP GLOBAL MARKETS": "CGML",
    "BANK OF AMERICA": "BOFA", "UBS": "UBS", "CREDIT SUISSE": "CS",
    "NOMURA INTERNATIONAL": "NOMURA", "MIZUHO BANK": "MIZUHO", "ING BANK": "ING",
    "SANTANDER": "SAN", "BBVA": "BBVA", "UNICREDIT": "UCG", "INTESA SANPAOLO": "ISP",
    "STANDARD CHARTERED BANK": "SCB", "ROYAL BANK OF CANADA": "RBC",
}
LEGAL_SUFFIXES = ["SA", "S.A.", "PLC", "AG", "LTD", "NV", "INC"]


@dataclass
class GenConfig:
    n_trades: int = 6000
    n_days: int = 60
    p_missing_id: float = 0.30      # B record has no trade reference
    p_id_typo: float = 0.05         # B reference has one wrong digit
    p_notional_noise: float = 0.15  # B notional differs slightly (fees, rounding)
    p_date_shift: float = 0.10      # B trade date is +/- 1 day (time zones)
    p_name_variation: float = 0.70  # B uses a variant of the counterparty name
    p_field_missing: float = 0.15   # B rate / settlement date is empty
    p_duplicate: float = 0.03       # B record is sent twice
    p_unconfirmed: float = 0.05     # trade exists in A but never reaches B
    p_orphan: float = 0.05          # record exists in B with no trade in A
    seed: int = 42


def name_variant(name: str, code: str, rng: np.random.Generator) -> str:
    """Return a realistic spelling variant of a counterparty name."""
    r = rng.random()
    if r < 0.30:
        out = name.title()
    elif r < 0.60:
        out = f"{name} {rng.choice(LEGAL_SUFFIXES)}"
    elif r < 0.75:
        out = f"{name.title()} {rng.choice(LEGAL_SUFFIXES)}"
    elif r < 0.85:
        out = code
    elif r < 0.93:
        out = name.replace(" ", "  ").lower()
    else:  # typo: swap two neighbouring characters
        i = int(rng.integers(0, max(len(name) - 1, 1)))
        chars = list(name)
        if len(chars) > 1:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        out = "".join(chars)
    return out


def _decimals(pair: str) -> int:
    return 3 if "JPY" in pair else 5


def _make_trades(n: int, start_idx: int, cfg: GenConfig, rng: np.random.Generator) -> pd.DataFrame:
    pairs = list(CCY_PAIRS)
    pair_w = np.array([0.30, 0.15, 0.15, 0.08, 0.08, 0.08, 0.08, 0.08])
    names = list(COUNTERPARTIES)
    cp_w = 1.0 / np.arange(1, len(names) + 1) ** 0.8  # a few counterparties dominate
    cp_w /= cp_w.sum()

    ccy = rng.choice(pairs, n, p=pair_w)
    day = rng.integers(0, cfg.n_days, n)
    trade_date = np.busday_offset("2025-01-06", day).astype("datetime64[D]")
    product = rng.choice(PRODUCTS, n, p=[0.5, 0.35, 0.15])
    lag = np.where(product == "FX_SPOT", 2, rng.choice([30, 60, 90, 180], n))
    settle_date = np.where(
        product == "FX_SPOT",
        np.busday_offset(trade_date, 2),
        np.busday_offset(trade_date + lag.astype("timedelta64[D]"), 0, roll="forward"),
    ).astype("datetime64[D]")

    # Round "market size" notionals are common -> many look-alike trades.
    round_sizes = np.array([0.5, 1, 2, 5, 10, 25]) * 1e6
    notional = np.where(
        rng.random(n) < 0.7,
        rng.choice(round_sizes, n, p=[0.15, 0.35, 0.2, 0.15, 0.1, 0.05]),
        np.round(rng.lognormal(14.5, 1.0, n), -3),
    )
    # Rate = pair level * daily drift * small intraday noise.
    daily = {p: np.cumprod(1 + rng.normal(0, 0.003, cfg.n_days)) for p in pairs}
    rate = np.array([
        CCY_PAIRS[p] * daily[p][d] * (1 + rng.normal(0, 0.00015)) for p, d in zip(ccy, day)
    ])
    rate = np.array([round(r, _decimals(p)) for r, p in zip(rate, ccy)])

    return pd.DataFrame({
        "true_trade": np.arange(start_idx, start_idx + n),
        "trade_ref": [f"TRD-{x}" for x in rng.choice(np.arange(100000, 999999), n, replace=False)],
        "product": product,
        "ccy_pair": ccy,
        "direction": rng.choice(["BUY", "SELL"], n),
        "notional": notional.astype(float),
        "rate": rate,
        "trade_date": pd.to_datetime(trade_date),
        "settle_date": pd.to_datetime(settle_date),
        "counterparty": rng.choice(names, n, p=cp_w),
    })


def _corrupt(row: dict, cfg: GenConfig, rng: np.random.Generator) -> dict:
    """Turn a clean trade into what the back-office system would hold."""
    r = dict(row)
    digits = r["trade_ref"].split("-")[1]
    if rng.random() < cfg.p_missing_id:
        r["trade_ref"] = None
    else:
        if rng.random() < cfg.p_id_typo:
            i = int(rng.integers(0, len(digits)))
            digits = digits[:i] + str((int(digits[i]) + int(rng.integers(1, 9))) % 10) + digits[i + 1:]
        r["trade_ref"] = rng.choice([f"TRD-{digits}", digits, f"TRD{digits}", f"REF/{digits}"],
                                    p=[0.5, 0.25, 0.15, 0.10])
    if rng.random() < cfg.p_notional_noise:
        r["notional"] = round(r["notional"] * (1 + rng.uniform(-0.001, 0.001)), 2)
    if rng.random() < cfg.p_date_shift:
        r["trade_date"] = r["trade_date"] + pd.Timedelta(days=int(rng.choice([-1, 1])))
    if rng.random() < cfg.p_date_shift / 2:
        r["settle_date"] = r["settle_date"] + pd.Timedelta(days=int(rng.choice([-1, 1])))
    if rng.random() < cfg.p_name_variation:
        r["counterparty"] = name_variant(r["counterparty"], COUNTERPARTIES[r["counterparty"]], rng)
    r["rate"] = round(r["rate"], 2 if "JPY" in r["ccy_pair"] else 4)  # B stores fewer decimals
    if rng.random() < cfg.p_field_missing:
        r["rate"] = np.nan
    if rng.random() < cfg.p_field_missing:
        r["settle_date"] = pd.NaT
    return r


def generate(cfg: GenConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (system_a, system_b). `true_trade` is the hidden ground truth."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    trades = _make_trades(cfg.n_trades, 0, cfg, rng)

    a = trades.copy()
    a.insert(0, "record_id", [f"A-{i:06d}" for i in range(len(a))])

    confirmed = trades[rng.random(len(trades)) >= cfg.p_unconfirmed]
    n_orphans = int(cfg.n_trades * cfg.p_orphan)
    orphans = _make_trades(n_orphans, cfg.n_trades, cfg, rng)

    rows = []
    for row in pd.concat([confirmed, orphans]).to_dict("records"):
        rows.append(_corrupt(row, cfg, rng))
        if rng.random() < cfg.p_duplicate:
            rows.append(_corrupt(row, cfg, rng))
    b = pd.DataFrame(rows).sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    b.insert(0, "record_id", [f"B-{i:06d}" for i in range(len(b))])
    return a, b
