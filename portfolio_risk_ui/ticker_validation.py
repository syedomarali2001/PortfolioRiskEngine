"""Verify a ticker against usable recent provider prices before accepting it.

No guessed ticker-to-company mappings, prices, or trading currencies. A provider
outage blocks unverified entries but is never misreported as a definite delisting.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import pandas as pd

from .core import normalized_symbol


@dataclass(frozen=True)
class TickerCheck:
    ok: bool
    message: str = ""


def verify_ticker(symbol: str, *, fetch_prices: Callable[[str], object] | None = None) -> TickerCheck:
    """Check recent usable daily closes; injectable fetcher enables offline tests.

    Presence of at least one positive recent closing price is only an entry
    check, not a promise of enough history for every risk metric.
    """
    symbol = normalized_symbol(symbol)
    if fetch_prices is None:
        def fetch_prices(s: str):
            from yfinance import Ticker
            return Ticker(s).history(period="3mo", interval="1d", auto_adjust=True,
                                     actions=False, raise_errors=True, timeout=15)
    try:
        frame = fetch_prices(symbol)
    except Exception:
        return TickerCheck(False, "Market prices could not be verified right now. "
                           "Check the ticker and exchange suffix (for example .TO), "
                           "or try again later.")
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame.columns:
        return TickerCheck(False, "No recent market prices were found. "
                           "Check the ticker and exchange suffix (for example .TO).")
    closes = pd.to_numeric(frame["Close"], errors="coerce")
    if not any(math.isfinite(float(value)) and float(value) > 0 for value in closes):
        return TickerCheck(False, "The market-data provider returned no usable closing prices. "
                           "Check the ticker or try again later.")
    return TickerCheck(True)
