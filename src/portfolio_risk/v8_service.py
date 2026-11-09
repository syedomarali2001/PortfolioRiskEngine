"""V8 orchestration: reuse V4's validated base-currency holdings price pipeline."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from portfolio_risk.accounting import PortfolioValuation
from portfolio_risk.backtest_service import run_backtest_from_market_data
from portfolio_risk.cache import HistoryCache
from portfolio_risk.historical_fx import convert_price_history_to_base
from portfolio_risk.models import PortfolioInput
from portfolio_risk.providers.base import MarketDataProvider
from portfolio_risk.quality import PriceQualityError, validate_history
from portfolio_risk.service import HistoryResult, get_validated_history
from portfolio_risk.v8_currency import compare_reporting_currencies
from portfolio_risk.v8_insights import BENCHMARKS, InsightError, analyze_insights


def _validated_v8_fx_history(provider, cache, start: date, end: date, *, refresh: bool = False):
    """FX-only, bounded tolerance for Yahoo occasionally returning adjacent bars.

    Regular equities/ETFs retain strict out-of-range rejection. We never use the
    adjacent FX bar in analysis and validate the complete trimmed series before
    caching it. A large boundary mismatch still fails closed.
    """
    symbol = "USDCAD=X"
    try:
        return get_validated_history(provider, cache, symbol, start, end, refresh=refresh), None
    except PriceQualityError as exc:
        if "outside requested window" not in str(exc):
            raise
        raw = provider.get_history(symbol, start, end)
        if raw.symbol != symbol or raw.start != start or raw.end != end or not raw.adjusted:
            raise exc
        series = raw.prices
        if (
            series.empty
            or not series.index.is_monotonic_increasing
            or series.index.has_duplicates
            or series.index.tz is not None
        ):
            raise exc
        mask = (series.index.date >= start) & (series.index.date < end)
        outliers = series.loc[~mask]
        inside = series.loc[mask]
        if (
            inside.empty
            or outliers.empty
            or any(
                min(abs((day.date() - start).days), abs((day.date() - end).days)) > 7
                for day in outliers.index
            )
        ):
            raise exc
        fixed = replace(raw, prices=inside)
        quality = validate_history(fixed)
        cache.put(fixed)
        message = (
            f"USDCAD=X: removed {len(outliers)} out-of-window FX bar(s) "
            f"(within 7 calendar days of requested boundaries); retained {quality.observation_count} verified observations"
        )
        return HistoryResult(fixed, quality, cached=False), message


def run_insights_from_market_data(
    portfolio: PortfolioInput,
    valuation: PortfolioValuation,
    provider: MarketDataProvider,
    cache: HistoryCache,
    *,
    start: date,
    end: date,
    rf: float = 0.0,
    scenarios=None,
    priorities=None,
    max_weight: float = 0.7,
    min_observations: int = 60,
    robustness=None,
    three_strategies: bool = False,
    max_fx_age_days: int = 5,
    refresh: bool = False,
    mc_paths: int = 300,
    mc_horizon: int = 252,
    seed: int = 42,
    compare_currencies: bool = False,
) -> dict:
    if start >= end:
        raise InsightError("Start must precede exclusive end")
    if end > date.today() + timedelta(days=1):
        raise InsightError("End cannot be in the future")
    bt = run_backtest_from_market_data(
        portfolio,
        valuation,
        provider,
        cache,
        start=start,
        end=end,
        schedules=("buy_hold",),
        min_observations=min_observations,
        max_fx_age_days=max_fx_age_days,
        annual_risk_free_rate=rf,
        refresh=refresh,
    )
    own = {h.symbol: bt.prices_in_base[h.symbol] for h in valuation.holdings}
    bench = {}
    meta = dict(bt.source_info)
    benchmark_warnings: list[str] = []
    # These two benchmark symbols refer to specific USD-listed U.S. ETFs.
    # Yahoo price history can be valid when optional currency metadata is absent.
    # Keep this narrowly scoped; NEVER infer arbitrary portfolio tickers' currency.
    benchmark_quote_currency = {"SPY": "USD", "RSP": "USD"}
    needs_fx = valuation.base_currency != "USD" and any(s not in own for s in BENCHMARKS)
    fx = None
    if needs_fx:
        fetched, fx_note = _validated_v8_fx_history(
            provider,
            cache,
            start - timedelta(days=max(9, max_fx_age_days + 2)),
            end,
            refresh=refresh,
        )
        fx = fetched.history.prices
        meta["USDCAD=X"] = {"quote_currency": "CAD per USD", "source": fetched.history.source}
        if fx_note:
            benchmark_warnings.append(fx_note)
    for symbol in BENCHMARKS:
        expected_ccy = benchmark_quote_currency[symbol]
        if symbol in own:
            # The backtest has already converted the entered holding. Ensure
            # the user/metadata didn't mark a USD benchmark as CAD by mistake.
            actual_ccy = meta[symbol]["quote_currency"]
            if actual_ccy != expected_ccy:
                raise InsightError(
                    f"{symbol}: benchmark is USD-listed, but portfolio quote_currency "
                    f"is {actual_ccy}; set quote_currency=USD in the portfolio CSV"
                )
            bench[symbol] = own[symbol]
            continue
        fetched = get_validated_history(provider, cache, symbol, start, end, refresh=refresh)
        reported_ccy = (fetched.history.quote_currency or "").strip().upper()
        if reported_ccy in {"CAD", "USD"} and reported_ccy != expected_ccy:
            raise InsightError(
                f"{symbol}: provider reports {reported_ccy} but the USD-listed "
                "benchmark requires USD; check provider data before comparing"
            )
        ccy = expected_ccy
        if reported_ccy != expected_ccy:
            benchmark_warnings.append(
                f"{symbol}: provider quote currency unavailable/unsupported; "
                "used the known USD trading currency of this benchmark ETF"
            )
        bench[symbol] = convert_price_history_to_base(
            fetched.history.prices,
            quote_currency=ccy,
            base_currency=valuation.base_currency,
            usd_cad_prices=fx,
            max_fx_age_days=max_fx_age_days,
        )
        meta[symbol] = {
            "quote_currency": ccy,
            "quote_currency_source": "provider_metadata"
            if reported_ccy == ccy
            else "benchmark_symbol_default",
            "source": fetched.history.source,
            "first_price_date": str(fetched.quality.first_date),
            "last_price_date": str(fetched.quality.last_date),
            "warnings": list(fetched.quality.warnings),
        }
    analysis = analyze_insights(
        own,
        {h.symbol: float(h.weight) for h in valuation.holdings},
        bench,
        base_currency=valuation.base_currency,
        annual_risk_free_rate=rf,
        scenarios=scenarios,
        priorities=priorities,
        max_weight=max_weight,
        min_observations=min_observations,
        mc_paths=mc_paths,
        mc_horizon=mc_horizon,
        seed=seed,
        robustness=robustness,
        three_strategies=three_strategies,
    )
    if compare_currencies:
        try:
            if fx is None:
                fetched, fx_note = _validated_v8_fx_history(
                    provider,
                    cache,
                    start - timedelta(days=max(9, max_fx_age_days + 2)),
                    end,
                    refresh=refresh,
                )
                fx = fetched.history.prices
                if fx_note:
                    benchmark_warnings.append(fx_note)
            analysis["currency_comparison"] = compare_reporting_currencies(
                own,
                {h.symbol: float(h.weight) for h in valuation.holdings},
                bench,
                fx,
                base_currency=valuation.base_currency,
                rf=rf,
                max_fx_age_days=max_fx_age_days,
                min_observations=min_observations,
            )
        except (ValueError, InsightError) as exc:
            analysis["currency_comparison"] = {"status": "unavailable", "reason": str(exc)}
    analysis["market_data"] = meta
    analysis["data_quality_warnings"] = list(
        dict.fromkeys([*bt.quality_warnings, *benchmark_warnings])
    )
    return analysis
