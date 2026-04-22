"""ShareSansar scraper — secondary data source."""

import requests
import pandas as pd
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

URL = "https://www.sharesansar.com/today-share-price"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://www.sharesansar.com/",
}
TIMEOUT = 20


def fetch_from_sharesansar():
    """Scrape today's price table from ShareSansar. Returns DataFrame or None."""
    try:
        resp = requests.get(URL, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        table = None
        for t in soup.find_all("table"):
            header_text = t.get_text(separator="|").lower()
            if any(k in header_text for k in ("symbol", "ltp", "volume")):
                table = t
                break

        if table is None:
            logger.warning("ShareSansar: no data table found.")
            return None

        headers, rows = [], []
        for i, row in enumerate(table.find_all("tr")):
            cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
            if not cells:
                continue
            if i == 0:
                headers = cells
            else:
                # Align length to headers
                if len(cells) > len(headers):
                    cells = cells[:len(headers)]
                elif len(cells) < len(headers):
                    cells += [""] * (len(headers) - len(cells))
                rows.append(cells)

        if not headers or not rows:
            logger.warning("ShareSansar: empty table after parse.")
            return None

        df = pd.DataFrame(rows, columns=headers)
        logger.info(f"ShareSansar: {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"ShareSansar error: {e}")
        return None
