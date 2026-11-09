"""V4 orchestration tests with deterministic mock histories."""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from portfolio_risk.accounting import value_portfolio
from portfolio_risk.backtest import BacktestError
from portfolio_risk.backtest_service import run_backtest_from_market_data
from portfolio_risk.fx import manual_fx_quote
from portfolio_risk.models import Holding, PortfolioInput, PriceHistory


class Cache:
    def get(self, *args):
        return None

    def put(self, *args):
        pass


class Provider:
    def __init__(self, histories):
        self.histories = histories
        self.called = []

    def get_history(self, symbol, start, end):
        self.called.append((symbol, start, end))
        series, ccy = self.histories[symbol]
        part = series.loc[(series.index.date >= start) & (series.index.date < end)]
        return PriceHistory(symbol, part, start, end, "Mock", datetime.now(timezone.utc), ccy)


def _series(start_value, count=45, *, dates=None):
    dates = dates if dates is not None else pd.date_range("2026-01-05", periods=count, freq="B")
    return pd.Series([start_value + i for i in range(count)], index=dates, dtype=float)


def test_service_mixed_cad_usd_and_benchmark_prices():
    p = PortfolioInput((Holding("XEQT.TO", Decimal(100), "CAD", "CAD"),
                        Holding("QQQ", Decimal(100), "USD", "USD")), "file.csv")
    valuation = value_portfolio(p, base_currency="CAD",
                                fx_quote=manual_fx_quote("1.35", date(2026, 4, 1)))
    ds = pd.date_range("2026-01-05", periods=45, freq="B")
    provider = Provider({
        "XEQT.TO": (_series(100, dates=ds), None),
        "QQQ": (_series(50, dates=ds), None),
        "USDCAD=X": (pd.Series([1.30]*45, index=ds, dtype=float), None),
        "SPY": (_series(80, dates=ds), None),
    })
    result = run_backtest_from_market_data(
        p, valuation, provider, Cache(), start=date(2026, 1, 5),
        end=date(2026, 3, 20), benchmark_symbol="SPY", benchmark_quote_currency="USD",
        fee_bps=5, initial_capital=20000,
    )
    assert result.report.observation_dates == 45
    assert result.report.base_currency == "CAD"
    assert result.report.target_weights["QQQ"] == pytest.approx(135/235)
    assert "USDCAD=X" in result.source_info
    assert len(provider.called) == 4
    assert result.source_info["QQQ"]["quote_currency_source"] == "explicit"
    assert result.report.strategies["buy_hold"].total_fees > 0


def test_unknown_quote_currency_fails_without_guessing():
    p = PortfolioInput((Holding("QQQ", Decimal(100), "USD"),), "file.csv")
    v = value_portfolio(p, base_currency="USD")
    provider = Provider({"QQQ": (_series(100), None)})
    with pytest.raises(BacktestError, match="quote_currency"):
        run_backtest_from_market_data(p, v, provider, Cache(),
                                      start=date(2026, 1, 5), end=date(2026, 3, 20))


def test_conflicting_explicit_currency_is_warned():
    p = PortfolioInput((Holding("QQQ", Decimal(100), "CAD", "CAD"),), "file.csv")
    v = value_portfolio(p, base_currency="CAD")
    provider = Provider({"QQQ": (_series(100), "USD")})
    result = run_backtest_from_market_data(p, v, provider, Cache(),
                                           start=date(2026, 1, 5), end=date(2026, 3, 20))
    assert any("differs from provider" in x for x in result.quality_warnings)


def test_fx_needed_by_quote_currency_even_if_entered_value_is_cad():
    p = PortfolioInput((Holding("QQQ", Decimal(100), "CAD", "USD"),), "file.csv")
    v = value_portfolio(p, base_currency="CAD")
    ds = pd.date_range("2026-01-05", periods=45, freq="B")
    provider = Provider({"QQQ": (_series(100, dates=ds), None),
                         "USDCAD=X": (pd.Series([1.3]*45, index=ds, dtype=float), None)})
    res = run_backtest_from_market_data(p, v, provider, Cache(),
                                        start=date(2026,1,5), end=date(2026,3,20))
    assert res.report.observation_dates == 45
    assert "USDCAD=X" in res.source_info
