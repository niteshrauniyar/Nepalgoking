"""NEPSE Official API fetcher — primary data source."""

import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://nepalstock.com.np/",
    "Origin": "https://nepalstock.com.np",
}
TIMEOUT = 15


def fetch_from_api():
    """Fetch today's price data from NEPSE API. Returns DataFrame or None."""
    try:
        session = requests.Session()
        session.headers.update(HEADERS)

        # Optional auth handshake
        try:
            session.post(
                "https://nepalstock.com.np/api/authenticate/prove",
                json={}, timeout=TIMEOUT
            )
        except Exception:
            pass

        resp = session.get(
            "https://nepalstock.com.np/api/nots/nepse-data/todaysprice",
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()

        records = None
        if isinstance(data, list) and data:
            records = data
        elif isinstance(data, dict):
            for key in ("content", "data", "todayPrice", "items"):
                v = data.get(key)
                if isinstance(v, list) and v:
                    records = v
                    break

        if not records:
            logger.warning("NEPSE API: empty or unrecognised payload.")
            return None

        df = pd.DataFrame(records)
        logger.info(f"NEPSE API: {len(df)} rows, cols={list(df.columns)}")
        return df

    except Exception as e:
        logger.error(f"NEPSE API error: {e}")
        return None
