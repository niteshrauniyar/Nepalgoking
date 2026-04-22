"""Shared utilities — caching, numeric conversion, formatting."""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR  = Path("cache")
CACHE_FILE = CACHE_DIR / "market_data.pkl"
META_FILE  = CACHE_DIR / "meta.json"
MAX_AGE_H  = 24


# ── Cache ──────────────────────────────────────────────────────────────────────

def save_cache(df: pd.DataFrame, source: str) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
        with open(META_FILE, "w") as f:
            json.dump({"source": source, "timestamp": datetime.now().isoformat(),
                       "rows": len(df)}, f)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


def load_cache() -> tuple:
    """Returns (DataFrame, meta_dict) or (None, None)."""
    try:
        if not CACHE_FILE.exists():
            return None, None
        with open(CACHE_FILE, "rb") as f:
            df = pickle.load(f)
        meta = {}
        if META_FILE.exists():
            with open(META_FILE) as f:
                meta = json.load(f)
        if "timestamp" in meta:
            age = (datetime.now() - datetime.fromisoformat(meta["timestamp"])).total_seconds() / 3600
            meta["age_hours"] = round(age, 1)
        return df, meta
    except Exception as e:
        logger.warning(f"Cache load failed: {e}")
        return None, None


# ── Safe numeric conversion ────────────────────────────────────────────────────

def safe_to_numeric(series: pd.Series) -> pd.Series:
    try:
        s = (
            series.astype(str).str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("\u2014", "0", regex=False)
            .str.replace("\u2013", "0", regex=False)
            .str.replace("\u2212", "-", regex=False)
            .str.replace("−", "-", regex=False)
        )
        neg = s.str.startswith("(") & s.str.endswith(")")
        s[neg] = "-" + s[neg].str[1:-1]
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.to_numeric(series, errors="coerce")


# ── Validation ─────────────────────────────────────────────────────────────────

def is_empty_or_none(df) -> bool:
    if df is None:
        return True
    if isinstance(df, pd.DataFrame):
        return df.empty
    return True


# ── Serialisation ──────────────────────────────────────────────────────────────

def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt_number(val, decimals: int = 2, suffix: str = "") -> str:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if abs(val) >= 1_000_000_000:
            return f"{val/1e9:.{decimals}f}B{suffix}"
        if abs(val) >= 1_000_000:
            return f"{val/1e6:.{decimals}f}M{suffix}"
        if abs(val) >= 1_000:
            return f"{val/1e3:.{decimals}f}K{suffix}"
        return f"{val:.{decimals}f}{suffix}"
    except Exception:
        return str(val)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
