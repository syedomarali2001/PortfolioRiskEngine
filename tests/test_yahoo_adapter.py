"""Tests Yahoo adapter's interface without making network requests."""

from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from portfolio_risk.providers.base import MarketDataError
from portfolio_risk.providers.yahoo import YahooFinanceProvider
from portfolio_risk.quality import validate_history


def test_yahoo_adapter_normalizes_exchange_timezone(monkeypatch):
    seen = {}
    class Ticker:
        def __init__(self, symbol):
            seen["symbol"] = symbol
        def history(self, **kwargs):
            seen.update(kwargs)
            idx = pd.DatetimeIndex(["2026-01-05", "2026-01-06"], tz="America/New_York")
            return pd.DataFrame({"Close": [100, 105]}, index=idx)
        def get_history_metadata(self):
            return {"currency": "USD"}
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=Ticker))
    result = YahooFinanceProvider().get_history("QQQ", date(2026, 1, 5), date(2026, 1, 8))
    assert result.symbol == "QQQ"
    assert result.quote_currency == "USD"
    assert result.prices.index.tz is None
    assert result.prices.index[0].date() == date(2026, 1, 5)
    assert result.adjusted is True
    assert seen["auto_adjust"] is True
    assert seen["end"] == "2026-01-08"
    assert validate_history(result).passed


def test_yahoo_adapter_empty_data(monkeypatch):
    class Ticker:
        def __init__(self, symbol):
            pass
        def history(self, **kwargs):
            return pd.DataFrame()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=Ticker))
    with pytest.raises(MarketDataError, match="no daily close"):
        YahooFinanceProvider().get_history("INVALID", date(2026, 1, 5), date(2026, 1, 8))
