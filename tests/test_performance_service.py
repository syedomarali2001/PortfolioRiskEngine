from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from portfolio_risk.accounting import value_portfolio
from portfolio_risk.fx import manual_fx_quote
from portfolio_risk.input import parse_portfolio
from portfolio_risk.models import Holding, PortfolioInput, PriceHistory
from portfolio_risk.performance import PerformanceError
from portfolio_risk.performance_service import run_performance


class MockCache:
    def get(self, *args):
        return None

    def put(self, *args):
        pass


class MockProvider:
    def __init__(self, histories):
        self.histories = histories
        self.called = []

    def get_history(self, symbol, start, end):
        self.called.append((symbol, start, end))
        full, currency = self.histories[symbol]
        sub = full.loc[(full.index.date >= start) & (full.index.date < end)]
        return PriceHistory(symbol, sub, start, end, "Fake", datetime.now(timezone.utc), currency)


def series(xs, dates=None):
    if dates is None:
        dates = pd.date_range("2026-01-05", periods=len(xs), freq="B")
    return pd.Series(xs, index=pd.DatetimeIndex(dates), dtype=float)


def portfolio(*holdings):
    return PortfolioInput(tuple(holdings), "dummy.csv")


def test_mixed_quote_currency_with_historical_fx():
    p = portfolio(Holding("XEQT.TO", Decimal("100"), "CAD", "CAD"),
                  Holding("QQQ", Decimal("100"), "USD", "USD"))
    valuation = value_portfolio(p, base_currency="CAD",
                                fx_quote=manual_fx_quote("1.35", date(2026, 3, 1)))
    dates = pd.date_range("2026-01-05", periods=40, freq="B")
    histories = {
        "XEQT.TO": (series(100 + pd.Series(range(40)).to_numpy(), dates), None),
        "QQQ": (series(50 + pd.Series(range(40)).to_numpy(), dates), None),
        "USDCAD=X": (series([1.30] * 40, dates), None),
    }
    run = run_performance(p, valuation, MockProvider(histories), MockCache(),
                          start=date(2026,1,5), end=date(2026,2,28))
    assert run.report.base_currency == "CAD"
    assert run.report.observation_dates == 40
    assert run.report.weights["QQQ"] == pytest.approx(135/235)
    assert run.source_info["QQQ"]["quote_currency_source"] == "explicit"
    assert "USDCAD=X" in run.source_info


def test_unknown_quote_currency_fails_without_silent_inference():
    p = portfolio(Holding("QQQ", Decimal("100"), "USD"))
    valuation = value_portfolio(p, base_currency="USD")
    provider = MockProvider({"QQQ": (series([100,102,103,104]), None)})
    with pytest.raises(PerformanceError, match="quote_currency"):
        run_performance(p, valuation, provider, MockCache(),
                        start=date(2026,1,5), end=date(2026,1,12), min_observations=3)


def test_provider_currency_used_when_available():
    p = portfolio(Holding("QQQ", Decimal("100"), "USD"))
    valuation = value_portfolio(p, base_currency="USD")
    provider = MockProvider({"QQQ": (series([100,102,103,104]), "USD")})
    run = run_performance(p, valuation, provider, MockCache(),
                          start=date(2026,1,5), end=date(2026,1,12), min_observations=3)
    assert run.source_info["QQQ"]["quote_currency_source"] == "provider_metadata"


def test_benchmark_and_fx_are_aligned():
    p = portfolio(Holding("QQQ", Decimal("100"), "USD", "USD"))
    valuation = value_portfolio(p, base_currency="USD")
    idx = pd.date_range("2026-01-05", periods=40, freq="B")
    provider = MockProvider({"QQQ": (series([100+i for i in range(40)], idx), "USD"),
                             "SPY": (series([100+i*.5 for i in range(40)], idx), None)})
    run = run_performance(p, valuation, provider, MockCache(),
                          start=date(2026,1,5), end=date(2026,2,28),
                          benchmark_symbol="SPY", benchmark_quote_currency="USD")
    assert run.report.benchmark.symbol == "SPY"
    assert len(provider.called) == 2


def test_conflicting_provider_currency_causes_quality_warning():
    p = portfolio(Holding("QQQ", Decimal("100"), "USD", "CAD"))
    valuation = value_portfolio(p, base_currency="CAD",
                                fx_quote=manual_fx_quote("1.35", date(2026, 3, 1)))
    provider = MockProvider({"QQQ": (series([100,101,102,103]), "USD")})
    run = run_performance(p, valuation, provider, MockCache(), start=date(2026,1,5),
                          end=date(2026,1,12), min_observations=3)
    assert any("differs from provider" in x for x in run.quality_warnings)
