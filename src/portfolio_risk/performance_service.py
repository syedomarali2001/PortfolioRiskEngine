"""V3 orchestration: obtain data and FX before invoking the pure metric engine.

Never interpret the currency of a user-entered market value as a security's
trading currency. This layer respects explicit quote currencies and rejects
unknown ones, preventing plausible but incorrect cross-currency performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Mapping

import pandas as pd

from portfolio_risk.accounting import PortfolioValuation
from portfolio_risk.cache import HistoryCache
from portfolio_risk.fx import FXRateError
from portfolio_risk.historical_fx import convert_price_history_to_base
from portfolio_risk.models import PortfolioInput
from portfolio_risk.performance import PerformanceError, PerformanceReport, analyze_performance
from portfolio_risk.providers.base import MarketDataProvider
from portfolio_risk.service import get_validated_history


@dataclass(frozen=True)
class PerformanceRun:
    report: PerformanceReport
    source_info: dict[str, dict]
    quality_warnings: tuple[str, ...]


def _quote_currency(symbol: str, explicit: str | None, provider_currency: str | None) -> str:
    manual = (explicit or "").strip().upper()
    reported = (provider_currency or "").strip().upper()
    if manual and manual not in {"CAD", "USD"}:
        raise PerformanceError(f"{symbol}: quote currency must be CAD or USD")
    if reported not in {"CAD", "USD"}:
        reported = ""
    if manual:
        return manual
    if reported:
        return reported
    raise PerformanceError(
        f"{symbol}: quote currency unavailable from data provider; "
        "add quote_currency=CAD or USD in the portfolio CSV"
    )


def run_performance(
    portfolio: PortfolioInput,
    valuation: PortfolioValuation,
    provider: MarketDataProvider,
    cache: HistoryCache,
    *,
    start: date,
    end: date,
    benchmark_symbol: str | None = None,
    benchmark_quote_currency: str | None = None,
    annual_risk_free_rate: float = 0.0,
    max_fx_age_days: int = 5,
    min_observations: int = 30,
    refresh: bool = False,
) -> PerformanceRun:
    if start >= end:
        raise PerformanceError("--start must be before --end (exclusive)")
    if max_fx_age_days < 0:
        raise PerformanceError("max_fx_age_days must not be negative")
    if len({h.symbol for h in portfolio.holdings}) != len(portfolio.holdings):
        raise PerformanceError("Portfolio symbols must be unique")
    if {h.symbol for h in portfolio.holdings} != {h.symbol for h in valuation.holdings}:
        raise PerformanceError("Portfolio and snapshot valuation contain different holdings")
    if benchmark_symbol in {h.symbol for h in portfolio.holdings}:
        raise PerformanceError("Benchmark must not also be a portfolio holding")
    if benchmark_symbol is None and benchmark_quote_currency is not None:
        raise PerformanceError("Benchmark quotation currency provided without a benchmark")

    symbols = [h.symbol for h in portfolio.holdings]
    if benchmark_symbol:
        symbols.append(benchmark_symbol)
    explicit: Mapping[str, str | None] = {h.symbol: h.quote_currency for h in portfolio.holdings}
    histories = {}
    source_info = {}
    warning_messages = []
    for symbol in symbols:
        result = get_validated_history(provider, cache, symbol, start, end, refresh=refresh)
        selected = _quote_currency(
            symbol,
            (benchmark_quote_currency if symbol == benchmark_symbol else explicit[symbol]),
            result.history.quote_currency,
        )
        if (result.history.quote_currency or "").upper() in {"USD", "CAD"} and (
            selected != result.history.quote_currency.upper()
        ):
            warning_messages.append(
                f"{symbol}: explicit quotation currency {selected} differs from provider "
                f"metadata {result.history.quote_currency}; verify the quote currency"
            )
        histories[symbol] = (result.history.prices, selected)
        source_info[symbol] = {
            "source": result.history.source,
            "quote_currency": selected,
            "quote_currency_source": "explicit"
            if (benchmark_quote_currency if symbol == benchmark_symbol else explicit[symbol])
            else "provider_metadata",
            "first_price_date": result.quality.first_date.isoformat(),
            "last_price_date": result.quality.last_date.isoformat(),
            "rows": result.quality.observation_count,
            "cached": result.cached,
            "warnings": list(result.quality.warnings),
        }
        for msg in result.quality.warnings:
            warning_messages.append(f"{symbol}: {msg}")

    base = valuation.base_currency
    needs_fx = any(currency != base for _, currency in histories.values())
    usd_cad = None
    if needs_fx:
        fx_start = start - timedelta(days=max(max_fx_age_days + 2, 9))
        fx_result = get_validated_history(
            provider, cache, "USDCAD=X", fx_start, end, refresh=refresh
        )
        usd_cad = fx_result.history.prices
        source_info["USDCAD=X"] = {
            "source": fx_result.history.source,
            "quote_currency": "CAD per USD",
            "first_price_date": fx_result.quality.first_date.isoformat(),
            "last_price_date": fx_result.quality.last_date.isoformat(),
            "rows": fx_result.quality.observation_count,
            "cached": fx_result.cached,
            "warnings": list(fx_result.quality.warnings),
        }
        for msg in fx_result.quality.warnings:
            warning_messages.append(f"USDCAD=X: {msg}")

    base_prices: dict[str, pd.Series] = {}
    for symbol, (prices, currency) in histories.items():
        try:
            base_prices[symbol] = convert_price_history_to_base(
                prices,
                quote_currency=currency,
                base_currency=base,
                usd_cad_prices=usd_cad,
                max_fx_age_days=max_fx_age_days,
            )
        except (ValueError, FXRateError) as exc:
            raise PerformanceError(
                f"{symbol}: base-currency price conversion failed: {exc}"
            ) from exc

    current_weights = {h.symbol: float(h.weight) for h in valuation.holdings}
    output = analyze_performance(
        {symbol: base_prices[symbol] for symbol in current_weights},
        current_weights,
        base_currency=base,
        annual_risk_free_rate=annual_risk_free_rate,
        benchmark_prices_in_base=base_prices.get(benchmark_symbol) if benchmark_symbol else None,
        benchmark_symbol=benchmark_symbol,
        min_observations=min_observations,
    )
    # If the complete input histories are very different, call that out.
    actual = output.observation_dates
    for symbol in symbols:
        total = source_info[symbol]["rows"]
        if actual < total * 0.8:
            warning_messages.append(
                f"CALENDAR_ALIGNMENT: {actual} common close dates vs {total} observed "
                f"dates for {symbol}; common-date sampling can discard non-overlapping sessions"
            )
    return PerformanceRun(output, source_info, tuple(dict.fromkeys(warning_messages)))
