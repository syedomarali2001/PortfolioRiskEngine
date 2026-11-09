"""Optional best-effort Yahoo ETF top-holdings adapter.

Top holdings are often incomplete, and constituent asset types are UNKNOWN. Yahoo generally does not expose a verified
holdings as-of date in this API; retrieval is NOT the disclosure date. Do not
persist or redistribute third-party snapshots without checking use terms.
"""
from __future__ import annotations

from datetime import date
from math import isfinite

from portfolio_risk.exposure import (
    Constituent, ExposureError, FundSnapshot, UNKNOWN, validate_snapshots,
)
from portfolio_risk.input import normalize_symbol


def _top_holding_rows(frame) -> list[Constituent]:
    """Parse common yfinance dataframe shapes; ambiguous weights fail closed."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ExposureError("ETF top holdings were not available")
    cols = {str(x).strip().lower().replace("_", " "): x for x in frame.columns}
    weight_col = next((cols[c] for c in ("holding percent", "holdingpercent", "weight", "holding %")
                       if c in cols), None)
    symbol_col = next((cols[c] for c in ("symbol", "ticker", "holding symbol")
                       if c in cols), None)
    if weight_col is None:
        raise ExposureError("Yahoo top holdings table has no recognized weight column")
    found = []
    for index, row in frame.iterrows():
        raw_symbol = row[symbol_col] if symbol_col is not None else index
        try:
            symbol = normalize_symbol(str(raw_symbol))
        except ValueError as exc:
            raise ExposureError(f"Yahoo returned an invalid holdings symbol {raw_symbol!r}") from exc
        try:
            frac = float(row[weight_col])
        except (ValueError, TypeError) as exc:
            raise ExposureError(f"{symbol}: Yahoo holding weight is missing") from exc
        if not isfinite(frac) or frac <= 0 or frac > 1.000001:
            # yfinance uses fractions (e.g. 0.07, not 7%) in Holding Percent.
            raise ExposureError(f"{symbol}: Yahoo weight must be a fraction 0..1")
        found.append(Constituent(symbol, frac, UNKNOWN))
    return found


def fetch_yahoo_top_holdings(symbol: str, *, fetched_on: date | None = None) -> FundSnapshot:
    """Best-effort snapshot; does NOT claim to be complete or disclosure-dated."""
    try:
        import yfinance as yf
        frame = yf.Ticker(symbol).funds_data.top_holdings
        holdings = _top_holding_rows(frame)
        snap = FundSnapshot(
            symbol=symbol,
            holdings=tuple(holdings),
            source="Yahoo Finance top_holdings via yfinance (unverified disclosure date)",
            as_of=None,
            retrieved_on=fetched_on or date.today(),
        )
        validate_snapshots({symbol: snap})
        return snap
    except ExposureError:
        raise
    except Exception as exc:
        raise ExposureError(f"{symbol}: ETF holdings unavailable from Yahoo: {exc}") from exc
