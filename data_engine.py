"""
NEPSE Data Engine — fetch, normalize, and analyse market data.
"""

import logging
import os
import sys

# ── Bulletproof import path fix ────────────────────────────────────────────────
# Works on local machine, Streamlit Cloud, and any CWD scenario.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# ──────────────────────────────────────────────────────────────────────────────

from typing import Optional
import numpy as np
import pandas as pd

# Use importlib as the final safety net so a bad __init__ never blocks startup
import importlib

def _safe_import(module_name, func_name):
    """Import a function by dotted module path; return a no-op lambda on failure."""
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, func_name)
    except Exception as exc:
        logging.getLogger(__name__).error(f"Could not import {module_name}.{func_name}: {exc}")
        return lambda: None

fetch_from_api         = _safe_import("fetchers.api",         "fetch_from_api")
fetch_from_sharesansar = _safe_import("fetchers.sharesansar", "fetch_from_sharesansar")
fetch_from_nepsealpha  = _safe_import("fetchers.nepsealpha",  "fetch_from_nepsealpha")

from utils import save_cache, load_cache, safe_to_numeric, is_empty_or_none, to_serializable

logger = logging.getLogger(__name__)

# ── Column name normalisation map ──────────────────────────────────────────────
COLUMN_MAP = {
    # Symbol
    "symbol": "symbol", "ticker": "symbol", "scrip": "symbol",
    "stock": "symbol", "stocksymbol": "symbol",
    # % change
    "pct_change": "pct_change", "percent_change": "pct_change",
    "change_percent": "pct_change", "pctchange": "pct_change",
    "change": "pct_change", "percentchange": "pct_change",
    "changepercent": "pct_change", "diff": "pct_change",
    # LTP / close
    "ltp": "ltp", "lasttradedprice": "ltp", "last_traded_price": "ltp",
    "closeprice": "ltp", "close_price": "ltp", "close": "ltp",
    "price": "ltp", "lastprice": "ltp",
    # Open
    "open": "open", "openprice": "open",
    # High / Low
    "high": "high", "highprice": "high", "dayhigh": "high",
    "low": "low",   "lowprice": "low",   "daylow": "low",
    # Volume
    "volume": "volume", "qty": "volume", "quantity": "volume",
    "tradedquantity": "volume", "total_traded_quantity": "volume",
    "totaltraded": "volume", "sharesqty": "volume",
    # Turnover
    "turnover": "turnover", "totalturnovers": "turnover",
    "total_turnover": "turnover", "tradedvalue": "turnover",
    "traded_value": "turnover", "amount": "turnover",
    # Transactions
    "transactions": "transactions", "transaction": "transactions",
    "nooftransactions": "transactions", "no_of_transactions": "transactions",
    # Prev close
    "previousclose": "prev_close", "prev_close": "prev_close",
    "previouscloseprice": "prev_close", "previous_close": "prev_close",
}

NUMERIC_COLS = ["ltp", "open", "high", "low", "pct_change",
                "volume", "turnover", "transactions", "prev_close"]


# ══════════════════════════════════════════════════════════════════════════════
# FETCH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def get_market_data():
    """
    Try all data sources in order; fall back to cache.
    Returns (DataFrame, source_name, status_string).
    """
    sources = [
        ("NEPSE Official API", fetch_from_api),
        ("ShareSansar",        fetch_from_sharesansar),
        ("NepseAlpha",         fetch_from_nepsealpha),
    ]

    for name, fetcher in sources:
        try:
            logger.info(f"Trying: {name}")
            raw = fetcher()
            if is_empty_or_none(raw):
                continue
            df = normalize_market_data(raw)
            if is_empty_or_none(df) or "symbol" not in df.columns:
                continue
            save_cache(df, name)
            logger.info(f"SUCCESS — {name}: {len(df)} rows")
            return df, name, "live"
        except Exception as e:
            logger.error(f"{name} pipeline error: {e}")

    # Cache fallback
    cached, meta = load_cache()
    if not is_empty_or_none(cached):
        age = meta.get("age_hours", "?") if meta else "?"
        src = meta.get("source",    "cache") if meta else "cache"
        return cached, src, f"cached ({age}h ago)"

    return None, "None", "unavailable"


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map any raw DataFrame to a standard schema.  Never crashes.
    """
    if is_empty_or_none(df):
        return pd.DataFrame()
    try:
        df = df.copy()

        # Flatten column names
        df.columns = (
            df.columns.astype(str)
            .str.strip().str.lower()
            .str.replace(r"[\s\-/]+", "_", regex=True)
            .str.replace(r"[^a-z0-9_%]", "", regex=True)
        )

        # Map to standard names
        df.rename(columns={c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP},
                  inplace=True)

        # Deduplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Infer symbol from first text column if missing
        if "symbol" not in df.columns:
            text_cols = df.select_dtypes(include="object").columns.tolist()
            if text_cols:
                df.rename(columns={text_cols[0]: "symbol"}, inplace=True)

        if "symbol" not in df.columns:
            return pd.DataFrame()

        # Clean symbol
        df["symbol"] = (
            df["symbol"].astype(str).str.strip().str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )
        df.dropna(subset=["symbol"], inplace=True)

        # Convert numerics
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = safe_to_numeric(df[col])

        # Derive pct_change if missing
        if ("pct_change" not in df.columns or df["pct_change"].isna().all()):
            if "ltp" in df.columns and "prev_close" in df.columns:
                df["pct_change"] = (
                    (df["ltp"] - df["prev_close"])
                    / df["prev_close"].replace(0, np.nan) * 100
                ).round(2)

        if "pct_change" not in df.columns:
            df["pct_change"] = 0.0
        else:
            df["pct_change"] = df["pct_change"].fillna(0.0)

        for col in ["volume", "turnover", "transactions"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        logger.error(f"normalize_market_data error: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def market_summary(df: pd.DataFrame) -> dict:
    """Market-wide summary stats. Returns plain Python dict."""
    result = {
        "total_stocks": 0, "advances": 0, "declines": 0, "unchanged": 0,
        "total_volume": 0, "total_turnover": 0.0,
        "avg_pct_change": 0.0, "breadth": 0.0,
        "top_gainer": {}, "top_loser": {}, "most_active": {},
        "market_sentiment": "neutral",
    }
    try:
        if is_empty_or_none(df):
            return result

        result["total_stocks"] = int(len(df))

        if "pct_change" in df.columns:
            pct = df["pct_change"].fillna(0)
            result["advances"]       = int((pct > 0).sum())
            result["declines"]       = int((pct < 0).sum())
            result["unchanged"]      = int((pct == 0).sum())
            result["avg_pct_change"] = round(float(pct.mean()), 3)
            tot = result["advances"] + result["declines"]
            result["breadth"] = round(
                (result["advances"] - result["declines"]) / tot, 4
            ) if tot else 0.0

        if "volume" in df.columns:
            result["total_volume"] = int(df["volume"].fillna(0).sum())

        if "turnover" in df.columns:
            result["total_turnover"] = float(round(df["turnover"].fillna(0).sum(), 2))

        if "pct_change" in df.columns and "symbol" in df.columns:
            g = df.loc[df["pct_change"].idxmax()]
            result["top_gainer"] = {
                "symbol": str(g.get("symbol", "—")),
                "pct_change": float(g.get("pct_change", 0)),
                "ltp": float(g["ltp"]) if "ltp" in df.columns else None,
            }
            lo = df.loc[df["pct_change"].idxmin()]
            result["top_loser"] = {
                "symbol": str(lo.get("symbol", "—")),
                "pct_change": float(lo.get("pct_change", 0)),
                "ltp": float(lo["ltp"]) if "ltp" in df.columns else None,
            }

        if "volume" in df.columns:
            ma = df.loc[df["volume"].idxmax()]
            result["most_active"] = {
                "symbol": str(ma.get("symbol", "—")),
                "volume": float(ma.get("volume", 0)),
                "pct_change": float(ma.get("pct_change", 0)) if "pct_change" in df.columns else None,
            }

        b = result["breadth"]
        result["market_sentiment"] = (
            "bullish" if b > 0.3 else ("bearish" if b < -0.3 else "neutral")
        )

    except Exception as e:
        logger.error(f"market_summary error: {e}")

    return to_serializable(result)


def order_flow_signal(df: pd.DataFrame) -> pd.DataFrame:
    if is_empty_or_none(df):
        return df
    try:
        out = df.copy()
        pct = out.get("pct_change", pd.Series(dtype=float)).fillna(0)
        vol = out.get("volume",     pd.Series(dtype=float)).fillna(0)

        vol_range = vol.max() - vol.min()
        vol_norm  = (vol - vol.min()) / vol_range if vol_range > 0 else pd.Series(0.0, index=vol.index)

        out["buy_pressure"]     = (vol_norm * (pct > 0).astype(float) * pct.clip(lower=0)).round(4)
        out["sell_pressure"]    = (vol_norm * (pct < 0).astype(float) * pct.clip(upper=0).abs()).round(4)
        out["persistence_score"]= (pct.abs() * vol.rank(pct=True)).round(4)
    except Exception as e:
        logger.error(f"order_flow_signal error: {e}")
    return out


def detect_large_activity(df: pd.DataFrame) -> pd.DataFrame:
    if is_empty_or_none(df) or "volume" not in df.columns:
        return df
    try:
        out = df.copy()
        vol = out["volume"].fillna(0)
        med = vol.median()
        std = vol.std()
        out["volume_ratio"]       = (vol / med.clip(min=1)).round(3)
        out["large_activity_flag"]= vol > (med + 2 * std)
    except Exception as e:
        logger.error(f"detect_large_activity error: {e}")
    return out


def liquidity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if is_empty_or_none(df):
        return df
    try:
        out = df.copy()
        if "pct_change" in out.columns:
            out["volatility_rank"] = out["pct_change"].abs().fillna(0).rank(pct=True).round(4)
        if "pct_change" in out.columns and "volume" in out.columns:
            vol = out["volume"].fillna(0).replace(0, np.nan)
            out["price_impact"] = (out["pct_change"].abs() / vol).round(8)
        if "turnover" in out.columns:
            to = out["turnover"].fillna(0)
            out["turnover_spike_flag"] = to > (to.mean() + 2 * to.std())
    except Exception as e:
        logger.error(f"liquidity_metrics error: {e}")
    return out


def smart_money_score(df: pd.DataFrame) -> pd.DataFrame:
    if is_empty_or_none(df):
        return df
    try:
        out = df.copy()

        def _rank(s):
            s = pd.to_numeric(s, errors="coerce").fillna(0)
            r = s.max() - s.min()
            return ((s - s.min()) / r * 100) if r > 0 else pd.Series(50.0, index=s.index)

        components, weights = [], []

        if "pct_change" in out.columns:
            components.append(_rank(out["pct_change"]));   weights.append(0.30)
        if "volume_ratio" in out.columns:
            components.append(_rank(out["volume_ratio"])); weights.append(0.30)
        elif "volume" in out.columns:
            components.append(_rank(out["volume"]));       weights.append(0.30)
        if "persistence_score" in out.columns:
            components.append(_rank(out["persistence_score"])); weights.append(0.25)
        if "price_impact" in out.columns:
            components.append(100 - _rank(out["price_impact"].fillna(0))); weights.append(0.15)

        if components:
            tw = sum(weights)
            score = sum(c * (w / tw) for c, w in zip(components, weights))
            out["smart_money_score"] = score.round(2)
        else:
            out["smart_money_score"] = 0.0

        def _label(s):
            if s >= 75: return "🔥 Strong"
            if s >= 55: return "📈 Moderate"
            if s >= 35: return "➡️ Neutral"
            return "📉 Weak"

        out["smart_money_label"] = out["smart_money_score"].apply(_label)

    except Exception as e:
        logger.error(f"smart_money_score error: {e}")
    return out


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run all analytics in sequence. Always returns a DataFrame."""
    if is_empty_or_none(df):
        return pd.DataFrame()
    for fn in (order_flow_signal, detect_large_activity, liquidity_metrics, smart_money_score):
        try:
            df = fn(df)
        except Exception as e:
            logger.error(f"{fn.__name__} failed: {e}")
    return df
