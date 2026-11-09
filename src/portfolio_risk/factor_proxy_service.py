"""Fetch proxy ETF prices and convert into the same base currency as V4 NAV."""
from __future__ import annotations

from datetime import date, timedelta

from portfolio_risk.backtest_service import BacktestRun
from portfolio_risk.cache import HistoryCache
from portfolio_risk.factor_proxy import FactorProxyError, analyze_factor_proxies
from portfolio_risk.historical_fx import convert_price_history_to_base
from portfolio_risk.providers.base import MarketDataProvider
from portfolio_risk.service import get_validated_history


def run_factor_proxy_from_backtest(
    backtest: BacktestRun, provider: MarketDataProvider, cache: HistoryCache,
    *, symbols: dict[str, str], start: date, end: date,
    max_fx_age_days: int = 5, min_observations: int = 60, refresh: bool = False,
):
    if not symbols or any(ccy not in {"CAD", "USD"} for ccy in symbols.values()):
        raise FactorProxyError("Each proxy must specify CAD or USD quotation currency")
    base = backtest.report.base_currency
    fx = None
    if any(ccy != base for ccy in symbols.values()):
        fx = get_validated_history(
            provider, cache, "USDCAD=X", start-timedelta(days=max_fx_age_days+9), end,
            refresh=refresh,
        ).history.prices
    converted = {}
    for symbol, currency in symbols.items():
        if symbol in backtest.report.target_weights:
            raise FactorProxyError(f"{symbol}: proxy is already a direct portfolio holding")
        history = get_validated_history(provider, cache, symbol, start, end, refresh=refresh)
        converted[symbol] = convert_price_history_to_base(
            history.history.prices, quote_currency=currency, base_currency=base,
            usd_cad_prices=fx, max_fx_age_days=max_fx_age_days,
        )
    return analyze_factor_proxies(backtest.report.strategies["buy_hold"].values,
                                  converted, min_observations=min_observations)
