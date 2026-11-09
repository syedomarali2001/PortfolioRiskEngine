"""Pure V3 performance calculations on explicitly aligned, base-currency prices.

These metrics describe a *hypothetical constant-weight daily rebalanced index*
anchored to the current snapshot weights. This is not an investor's actual return
history or V4's execution/cost-aware backtesting engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk.historical_fx import _validated_series

TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25


class PerformanceError(ValueError):
    """Input data are insufficient or invalid for a performance calculation."""


@dataclass(frozen=True)
class Metrics:
    cumulative_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float | None
    sortino: float | None
    maximum_drawdown: float
    best_period_return: float
    worst_period_return: float


@dataclass(frozen=True)
class BenchmarkComparison:
    symbol: str
    metrics: Metrics
    beta: float | None
    tracking_error: float
    information_ratio: float | None


@dataclass(frozen=True)
class PerformanceReport:
    first_date: str
    last_date: str
    observation_dates: int
    return_periods: int
    base_currency: str
    risk_free_annual: float
    weights: dict[str, float]
    portfolio: Metrics
    assets: dict[str, Metrics]
    correlations: dict[str, dict[str, float | None]]
    benchmark: BenchmarkComparison | None
    # User-interface-independent outputs for future plotting and export.
    portfolio_index: pd.Series
    period_returns: pd.Series
    warnings: tuple[str, ...]


def _check_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise PerformanceError("Nonempty price DataFrame required")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise PerformanceError("Prices must have a daily DatetimeIndex")
    idx = prices.index
    if idx.tz is not None or idx.has_duplicates or not idx.is_monotonic_increasing:
        raise PerformanceError("Price dates must be unique, sorted, and timezone-naive")
    if not idx.equals(idx.normalize()):
        raise PerformanceError("Price dates must be normalized to midnight")
    if prices.columns.has_duplicates:
        raise PerformanceError("Price columns must be unique")
    if prices.isna().any().any():
        raise PerformanceError("Price frame cannot contain missing values or inferred closes")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not (matrix > 0).all():
        raise PerformanceError("Prices must be strictly positive and finite")
    return numeric.astype(float)


def _metrics(returns: pd.Series, dates: pd.DatetimeIndex, risk_free_annual: float) -> Metrics:
    """All returns are simple periodic returns, with annualized risk from 252 periods.

    CAGR uses actual elapsed calendar days. Sharpe and Sortino use a constant
    effective daily risk-free return derived from the *user-selected* annual rate.
    """
    observations = returns.to_numpy(dtype=float)
    if len(observations) < 2 or not np.isfinite(observations).all():
        raise PerformanceError("Insufficient clean return periods")
    if np.any(observations <= -1):
        raise PerformanceError("A return of -100% or below is incompatible with positive prices")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + observations)))
    if not np.isfinite(wealth).all() or np.any(wealth <= 0):
        raise PerformanceError("Compounded wealth is invalid")
    elapsed_days = (dates[-1] - dates[0]).days
    if elapsed_days <= 0:
        raise PerformanceError("Portfolio sample must span multiple calendar days")
    cumulative = float(wealth[-1] - 1.0)
    cagr = float(np.expm1(np.log(wealth[-1]) * (CALENDAR_DAYS_PER_YEAR / elapsed_days)))
    vol_per_period = float(np.std(observations, ddof=1))
    vol = vol_per_period * sqrt(TRADING_DAYS_PER_YEAR)
    rf_per_period = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    excess = observations - rf_per_period
    sharpe = (float(np.mean(excess)) / vol_per_period * sqrt(TRADING_DAYS_PER_YEAR)
              if vol_per_period > 1e-14 else None)
    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    sortino = (float(np.mean(excess)) / downside_dev * sqrt(TRADING_DAYS_PER_YEAR)
               if downside_dev > 1e-14 else None)
    peaks = np.maximum.accumulate(wealth)
    max_dd = float(np.min(wealth / peaks - 1.0))
    return Metrics(
        cumulative_return=cumulative,
        cagr=cagr,
        annualized_volatility=vol,
        sharpe=sharpe,
        sortino=sortino,
        maximum_drawdown=max_dd,
        best_period_return=float(np.max(observations)),
        worst_period_return=float(np.min(observations)),
    )


def analyze_performance(
    prices_in_base: Mapping[str, pd.Series],
    weights: Mapping[str, float],
    *,
    base_currency: str = "CAD",
    annual_risk_free_rate: float = 0.0,
    benchmark_prices_in_base: pd.Series | None = None,
    benchmark_symbol: str | None = None,
    min_observations: int = 30,
) -> PerformanceReport:
    """Analyze a synthetic fixed-target-weight daily constant-mix portfolio.

    Only price dates **observed by every asset** are used. Adjacent common
    dates define return periods: weekends/closures are absorbed into the next
    observed common-date return, never forward-filled or assigned zero returns.
    All prices must already be in the same base currency. Market values define
    fixed weights, *not* historical units/transactions.
    """
    if base_currency not in {"CAD", "USD"}:
        raise PerformanceError("Base currency must be CAD or USD")
    if not isinstance(min_observations, int) or min_observations < 3:
        raise PerformanceError("min_observations must be an integer >= 3")
    if not np.isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1:
        raise PerformanceError("Annual risk-free rate must be finite and greater than -100%")
    if not prices_in_base or set(prices_in_base) != set(weights):
        raise PerformanceError("Weights must exactly match all supplied assets")
    normalized_weights = {str(k): float(v) for k, v in weights.items()}
    vals = list(normalized_weights.values())
    if not all(np.isfinite(v) and v > 0 for v in vals):
        raise PerformanceError("Weights must be strictly positive and finite")
    if not np.isclose(sum(vals), 1.0, atol=1e-10, rtol=0):
        raise PerformanceError("Weights must sum to 1; normalize them in V2")
    if (benchmark_prices_in_base is None) != (benchmark_symbol is None):
        raise PerformanceError("Benchmark price series and symbol must be supplied together")
    if benchmark_symbol in prices_in_base:
        raise PerformanceError("Benchmark must use a distinct symbol")

    named = {}
    for symbol, series in prices_in_base.items():
        try:
            named[symbol] = _validated_series(series, symbol)
        except ValueError as exc:
            raise PerformanceError(str(exc)) from exc
    if benchmark_prices_in_base is not None:
        try:
            named[benchmark_symbol] = _validated_series(benchmark_prices_in_base, "benchmark")
        except ValueError as exc:
            raise PerformanceError(str(exc)) from exc

    # Strict intersection. No forward fill of exchange-closed assets and no
    # false zero returns for BTC while stock markets are closed.
    price_frame = pd.concat(named, axis=1, join="inner").sort_index()
    price_frame = _check_price_frame(price_frame)
    if len(price_frame) < min_observations:
        raise PerformanceError(
            f"Only {len(price_frame)} common close dates (need >= {min_observations}); "
            "reduce the window or remove assets with insufficient history"
        )
    returns = price_frame.pct_change(fill_method=None).iloc[1:]
    asset_names = list(weights)
    asset_rets = returns[asset_names]
    w = pd.Series(normalized_weights, dtype=float).reindex(asset_names)
    port_rets = asset_rets.mul(w, axis=1).sum(axis=1)
    port_rets.name = "portfolio_return"
    index = pd.Series(100.0, index=price_frame.index, name="portfolio_index", dtype=float)
    index.iloc[1:] = 100.0 * np.cumprod(1.0 + port_rets.to_numpy())
    dates = price_frame.index
    asset_metrics = {
        symbol: _metrics(asset_rets[symbol], dates, annual_risk_free_rate)
        for symbol in asset_names
    }
    correlations_frame = asset_rets.corr()
    correlations = {
        symbol: {
            other: (float(value) if pd.notna(value) else None)
            for other, value in correlations_frame[symbol].items()
        }
        for symbol in asset_names
    }
    benchmark = None
    if benchmark_symbol is not None:
        bench_returns = returns[benchmark_symbol]
        covariance = float(port_rets.cov(bench_returns))
        benchmark_variance = float(bench_returns.var(ddof=1))
        beta = covariance / benchmark_variance if benchmark_variance > 1e-14 else None
        active = port_rets - bench_returns
        active_vol = float(active.std(ddof=1))
        tracking_error = active_vol * sqrt(TRADING_DAYS_PER_YEAR)
        information_ratio = (float(active.mean()) / active_vol * sqrt(TRADING_DAYS_PER_YEAR)
                             if active_vol > 1e-14 else None)
        benchmark = BenchmarkComparison(
            benchmark_symbol,
            _metrics(bench_returns, dates, annual_risk_free_rate),
            beta,
            tracking_error,
            information_ratio,
        )
    warnings = []
    if len(returns) < TRADING_DAYS_PER_YEAR:
        warnings.append("SHORT_SAMPLE: fewer than 252 return periods; annualized ratios may be unstable")
    # User supplied current holdings weights applied retroactively. Never
    # describe this result as the person's realized account returns.
    warnings.append(
        "HYPOTHETICAL: constant snapshot weights applied to each common-date return "
        "(daily constant mix, no costs); not actual investment history or a tradeable backtest"
    )
    return PerformanceReport(
        first_date=dates[0].date().isoformat(),
        last_date=dates[-1].date().isoformat(),
        observation_dates=len(dates),
        return_periods=len(returns),
        base_currency=base_currency,
        risk_free_annual=annual_risk_free_rate,
        weights=normalized_weights,
        portfolio=_metrics(port_rets, dates, annual_risk_free_rate),
        assets=asset_metrics,
        correlations=correlations,
        benchmark=benchmark,
        portfolio_index=index,
        period_returns=port_rets,
        warnings=tuple(warnings),
    )


def performance_as_dict(report: PerformanceReport, *, include_series: bool = True) -> dict:
    """JSON-ready report; no NaN/Infinity and no network/provider dependencies."""
    from dataclasses import asdict

    result = {
        "methodology": "Hypothetical daily constant-mix portfolio on shared observed close dates",
        "first_date": report.first_date,
        "last_date": report.last_date,
        "observation_dates": report.observation_dates,
        "return_periods": report.return_periods,
        "base_currency": report.base_currency,
        "annual_risk_free_rate": report.risk_free_annual,
        "weights": report.weights,
        "portfolio": asdict(report.portfolio),
        "assets": {k: asdict(v) for k, v in report.assets.items()},
        "correlations": report.correlations,
        "benchmark": asdict(report.benchmark) if report.benchmark else None,
        "warnings": list(report.warnings),
    }
    if include_series:
        result["portfolio_index"] = [
            {"date": d.date().isoformat(), "value": float(v)}
            for d, v in report.portfolio_index.items()
        ]
        result["period_returns"] = [
            {"date": d.date().isoformat(), "return": float(v)}
            for d, v in report.period_returns.items()
        ]
    return result
