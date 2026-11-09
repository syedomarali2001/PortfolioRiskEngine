"""V4 service: historical data retrieval, currency auditing and backtest execution.

Kept separate from the pure V4 simulator so it can later power a GUI or API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from portfolio_risk.accounting import PortfolioValuation
from portfolio_risk.backtest import SCHEDULES, BacktestError, BacktestReport, run_backtest
from portfolio_risk.cache import HistoryCache
from portfolio_risk.fx import FXRateError
from portfolio_risk.historical_fx import convert_price_history_to_base
from portfolio_risk.models import PortfolioInput
from portfolio_risk.performance_service import _quote_currency
from portfolio_risk.providers.base import MarketDataProvider
from portfolio_risk.service import get_validated_history


@dataclass(frozen=True)
class BacktestRun:
    report: BacktestReport
    source_info: dict[str, dict]
    quality_warnings: tuple[str, ...]
    # Validated, FX-adjusted price series reused by V5; no extra API calls.
    prices_in_base: dict[str, pd.Series] | None = None


def run_backtest_from_market_data(
    portfolio: PortfolioInput,
    valuation: PortfolioValuation,
    provider: MarketDataProvider,
    cache: HistoryCache,
    *,
    start: date,
    end: date,
    initial_capital: float = 20000.0,
    fee_bps: float = 0.0,
    schedules: tuple[str, ...] = SCHEDULES,
    annual_risk_free_rate: float = 0.0,
    benchmark_symbol: str | None = None,
    benchmark_quote_currency: str | None = None,
    max_fx_age_days: int = 5,
    min_observations: int = 30,
    refresh: bool = False,
) -> BacktestRun:
    if start >= end:
        raise BacktestError("--start must be before --end (exclusive)")
    if max_fx_age_days < 0:
        raise BacktestError("FX age must not be negative")
    names = [h.symbol for h in portfolio.holdings]
    if len(set(names)) != len(names):
        raise BacktestError("Portfolio symbols must be unique")
    if set(names) != {h.symbol for h in valuation.holdings}:
        raise BacktestError("Portfolio and snapshot weights differ")
    if benchmark_symbol in names:
        raise BacktestError("Benchmark must not be a portfolio holding")
    if benchmark_symbol is None and benchmark_quote_currency is not None:
        raise BacktestError("Benchmark currency supplied without benchmark symbol")

    symbols = names + ([benchmark_symbol] if benchmark_symbol else [])
    quotes = {h.symbol: h.quote_currency for h in portfolio.holdings}
    source_info = {}
    warnings = []
    native_prices = {}
    for symbol in symbols:
        fetched = get_validated_history(provider, cache, symbol, start, end, refresh=refresh)
        chosen_explicit = benchmark_quote_currency if symbol == benchmark_symbol else quotes[symbol]
        try:
            selected_ccy = _quote_currency(symbol, chosen_explicit, fetched.history.quote_currency)
        except ValueError as exc:
            raise BacktestError(str(exc)) from exc
        reported = (fetched.history.quote_currency or "").strip().upper()
        if reported in {"CAD", "USD"} and reported != selected_ccy:
            warnings.append(
                f"{symbol}: quote currency {selected_ccy} differs from provider {reported}"
            )
        native_prices[symbol] = (fetched.history.prices, selected_ccy)
        source_info[symbol] = {
            "source": fetched.history.source,
            "quote_currency": selected_ccy,
            "quote_currency_source": "explicit" if chosen_explicit else "provider_metadata",
            "first_price_date": fetched.quality.first_date.isoformat(),
            "last_price_date": fetched.quality.last_date.isoformat(),
            "rows": fetched.quality.observation_count,
            "cached": fetched.cached,
            "warnings": list(fetched.quality.warnings),
        }
        warnings.extend(f"{symbol}: {w}" for w in fetched.quality.warnings)

    need_fx = any(ccy != valuation.base_currency for _, ccy in native_prices.values())
    fx_prices = None
    if need_fx:
        fx_start = start - timedelta(days=max(max_fx_age_days + 2, 9))
        fetched = get_validated_history(provider, cache, "USDCAD=X", fx_start, end, refresh=refresh)
        fx_prices = fetched.history.prices
        source_info["USDCAD=X"] = {
            "source": fetched.history.source,
            "quote_currency": "CAD per USD",
            "first_price_date": fetched.quality.first_date.isoformat(),
            "last_price_date": fetched.quality.last_date.isoformat(),
            "rows": fetched.quality.observation_count,
            "cached": fetched.cached,
            "warnings": list(fetched.quality.warnings),
        }
        warnings.extend(f"USDCAD=X: {w}" for w in fetched.quality.warnings)

    converted: dict[str, pd.Series] = {}
    for symbol, (series, currency) in native_prices.items():
        try:
            converted[symbol] = convert_price_history_to_base(
                series,
                quote_currency=currency,
                base_currency=valuation.base_currency,
                usd_cad_prices=fx_prices,
                max_fx_age_days=max_fx_age_days,
            )
        except (ValueError, FXRateError) as exc:
            raise BacktestError(f"{symbol}: cannot convert historical prices: {exc}") from exc

    weights = {h.symbol: float(h.weight) for h in valuation.holdings}
    result = run_backtest(
        {s: converted[s] for s in names},
        weights,
        initial_capital=initial_capital,
        base_currency=valuation.base_currency,
        fee_bps=fee_bps,
        annual_risk_free_rate=annual_risk_free_rate,
        schedules=schedules,
        benchmark_prices_in_base=(converted.get(benchmark_symbol) if benchmark_symbol else None),
        benchmark_symbol=benchmark_symbol,
        min_observations=min_observations,
    )
    for symbol in symbols:
        total = source_info[symbol]["rows"]
        if result.observation_dates < total * 0.8:
            warnings.append(
                f"CALENDAR_ALIGNMENT: only {result.observation_dates} shared close dates "
                f"compared with {total} dates for {symbol}"
            )
    return BacktestRun(result, source_info, tuple(dict.fromkeys(warnings)), converted)
