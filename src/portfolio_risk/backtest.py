"""V4 pure price-based portfolio simulation: no APIs, file I/O or dashboard.

At the first common adjusted close, deploy the specified initial capital into
fractional shares at target weights. A zero-balance cash account pays trading
costs out of the portfolio's assets. On rebalancing dates, trade at the same
adjusted closing prices *after* measuring that day's pre-trade wealth.

CAUTION: adjusted-price execution is an idealized total-return convention, not
an executable historical price series. Fees are proportional to gross traded
notional (both purchases and sales); taxes, spreads, slippage, and financing
are not modeled. This is a strategy comparison, not a brokerage ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from portfolio_risk.historical_fx import _validated_series
from portfolio_risk.performance import Metrics, PerformanceError, _check_price_frame, _metrics

SCHEDULES = ("buy_hold", "monthly", "quarterly", "semiannual", "annual")


class BacktestError(ValueError):
    """Invalid backtest inputs or market data."""


@dataclass(frozen=True)
class Trade:
    date: str
    symbol: str
    quantity: float  # positive buy / negative sell; fractional units permitted
    notional: float  # signed, base currency
    fee: float  # positive, base currency
    reason: str  # initial or rebalance


@dataclass(frozen=True)
class StrategyResult:
    name: str
    metrics: Metrics
    ending_value: float
    total_fees: float
    traded_notional: float  # sum absolute buys and sells, includes initial purchases
    initial_fees: float
    rebalance_count: int
    trade_count: int
    ending_weights: dict[str, float]
    maximum_weight_drift: float  # largest abs(actual - target) at pre-trade close
    values: pd.Series
    returns: pd.Series
    weights_over_time: pd.DataFrame  # post-trade allocation on each common close
    trades: tuple[Trade, ...]


@dataclass(frozen=True)
class BacktestReport:
    first_date: str
    last_date: str
    observation_dates: int
    return_periods: int
    base_currency: str
    initial_capital: float
    target_weights: dict[str, float]
    fee_bps: float
    risk_free_annual: float
    strategies: dict[str, StrategyResult]
    benchmark_symbol: str | None
    benchmark: Metrics | None
    benchmark_values: pd.Series | None
    warnings: tuple[str, ...]


def _is_period_end(dates: pd.DatetimeIndex, position: int, frequency: str) -> bool:
    """Last available *common observed close* in a completed calendar period.

    Never trade at the initial or terminal observation: there is no benefit to
    paying fees at the end of the backtest and we require a new observation to
    prove that the observed period has ended.
    """
    if position == 0 or position >= len(dates) - 1:
        return False
    a, b = dates[position], dates[position + 1]
    if frequency == "monthly":
        return (a.year, a.month) != (b.year, b.month)
    if frequency == "quarterly":
        return (a.year, (a.month - 1) // 3) != (b.year, (b.month - 1) // 3)
    if frequency == "semiannual":
        return (a.year, (a.month - 1) // 6) != (b.year, (b.month - 1) // 6)
    if frequency == "annual":
        return a.year != b.year
    return False


def _rebalance_at_close(
    positions: np.ndarray,
    prices: np.ndarray,
    weights: np.ndarray,
    fee_rate: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Find post-fee target positions, funding costs by slightly reducing NAV.

    W_post = W_pre - f * sum(abs(w_i*W_post - old_notional_i)).
    Bisection is used rather than a heuristic that might spend more cash than
    available. Returned fee is reconciled with the actual traded notional.
    """
    old_values = positions * prices
    pre_wealth = float(old_values.sum())
    if not isfinite(pre_wealth) or pre_wealth <= 0:
        raise BacktestError("Nonpositive or invalid pre-trade portfolio wealth")
    if fee_rate == 0.0:
        post_wealth = pre_wealth
    else:
        lo, hi = 0.0, pre_wealth
        for _ in range(85):
            mid = (lo + hi) / 2.0
            total_fees = fee_rate * float(np.abs(weights * mid - old_values).sum())
            if mid + total_fees > pre_wealth:
                hi = mid
            else:
                lo = mid
        post_wealth = (lo + hi) / 2.0
    desired_values = post_wealth * weights
    notional_change = desired_values - old_values
    total_fee = float(np.abs(notional_change).sum()) * fee_rate
    if abs(pre_wealth - post_wealth - total_fee) > 1e-7 * max(1.0, pre_wealth):
        raise BacktestError("Rebalance accounting failed to reconcile")
    return desired_values / prices, total_fee, notional_change


def _run_one(
    prices: pd.DataFrame,
    weights: np.ndarray,
    schedule: str,
    initial_capital: float,
    fee_rate: float,
    annual_risk_free_rate: float,
) -> StrategyResult:
    columns = list(prices.columns)
    dates = prices.index
    matrix = prices.to_numpy(dtype=float)
    # Initial entry: pay fees for all purchases, from the initial capital.
    invested = initial_capital / (1.0 + fee_rate)
    initial_fees = initial_capital - invested
    positions = (invested * weights) / matrix[0]
    fees = initial_fees
    notional_turnover = invested
    trades: list[Trade] = []
    for j, symbol in enumerate(columns):
        trades.append(
            Trade(
                dates[0].date().isoformat(),
                symbol,
                float(positions[j]),
                float(invested * weights[j]),
                float(initial_fees * weights[j]),
                "initial",
            )
        )

    values = np.empty(len(dates), dtype=float)
    # First displayed NAV is contributed capital immediately before initial entry.
    values[0] = initial_capital  # next return includes initial fees
    weight_history = np.empty((len(dates), len(columns)), dtype=float)
    weight_history[0, :] = weights
    rebalances = 0
    max_drift = 0.0
    for i in range(1, len(dates)):
        current_values = positions * matrix[i]
        pre_wealth = float(current_values.sum())
        actual_weights = current_values / pre_wealth
        max_drift = max(max_drift, float(np.max(np.abs(actual_weights - weights))))
        if schedule != "buy_hold" and _is_period_end(dates, i, schedule):
            new_positions, trade_fees, trade_notional = _rebalance_at_close(
                positions, matrix[i], weights, fee_rate
            )
            # Record only meaningful orders; avoiding tiny floating point trades.
            for j, symbol in enumerate(columns):
                if abs(trade_notional[j]) > pre_wealth * 1e-11:
                    trades.append(
                        Trade(
                            dates[i].date().isoformat(),
                            symbol,
                            float(new_positions[j] - positions[j]),
                            float(trade_notional[j]),
                            float(abs(trade_notional[j]) * fee_rate),
                            "rebalance",
                        )
                    )
            fees += trade_fees
            notional_turnover += float(np.abs(trade_notional).sum())
            positions = new_positions
            rebalances += 1
        post_values = positions * matrix[i]
        values[i] = float(post_values.sum())
        weight_history[i, :] = post_values / values[i]
        if not np.isfinite(values[i]) or values[i] <= 0:
            raise BacktestError("Invalid strategy wealth during simulation")

    nav = pd.Series(values, index=dates, name=schedule)
    # The first point is contributed capital, just before entry execution;
    # the subsequent return captures the entry fee along with price changes.
    returns = nav.pct_change(fill_method=None).iloc[1:]
    # Final amount and total fees are always based on actual simulated NAV.
    metrics = _metrics(returns, dates, annual_risk_free_rate)
    final_positions_value = positions * matrix[-1]
    final_total = float(final_positions_value.sum())
    final_weights = {s: float(v / final_total) for s, v in zip(columns, final_positions_value, strict=False)}
    return StrategyResult(
        name=schedule,
        metrics=metrics,
        ending_value=float(nav.iloc[-1]),
        total_fees=float(fees),
        traded_notional=float(notional_turnover),
        initial_fees=float(initial_fees),
        rebalance_count=rebalances,
        trade_count=len(trades),
        ending_weights=final_weights,
        maximum_weight_drift=float(max_drift),
        values=nav,
        returns=returns,
        weights_over_time=pd.DataFrame(weight_history, index=dates, columns=columns),
        trades=tuple(trades),
    )


def run_backtest(
    prices_in_base: Mapping[str, pd.Series],
    target_weights: Mapping[str, float],
    *,
    initial_capital: float = 20000.0,
    base_currency: str = "CAD",
    fee_bps: float = 0.0,
    annual_risk_free_rate: float = 0.0,
    schedules: Sequence[str] = SCHEDULES,
    benchmark_prices_in_base: pd.Series | None = None,
    benchmark_symbol: str | None = None,
    min_observations: int = 30,
) -> BacktestReport:
    """Simulate strategies on the intersection of observed adjusted-close dates."""
    if base_currency not in {"CAD", "USD"}:
        raise BacktestError("base currency must be CAD or USD")
    if not isfinite(initial_capital) or initial_capital <= 0:
        raise BacktestError("Initial capital must be a positive finite number")
    if not isfinite(fee_bps) or not 0 <= fee_bps <= 10000:
        raise BacktestError("Fee bps must be finite and between 0 and 10000")
    if not isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1:
        raise BacktestError("Annual risk-free rate must be finite and > -100%")
    if not isinstance(min_observations, int) or min_observations < 3:
        raise BacktestError("At least 3 minimum observations required")
    if not prices_in_base or set(prices_in_base) != set(target_weights):
        raise BacktestError("Price symbols and target weights must match exactly")
    if (benchmark_prices_in_base is None) != (benchmark_symbol is None):
        raise BacktestError("Benchmark price and symbol must be supplied together")
    if benchmark_symbol is not None and benchmark_symbol in prices_in_base:
        raise BacktestError("Benchmark ticker must differ from portfolio holdings")
    if not schedules or len(set(schedules)) != len(schedules) or set(schedules) - set(SCHEDULES):
        raise BacktestError(f"Schedules must be unique entries from: {', '.join(SCHEDULES)}")
    weights = np.array([float(target_weights[s]) for s in prices_in_base], dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise BacktestError("Target weights must be strictly positive and finite")
    if not np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=0):
        raise BacktestError("Target weights must sum to 1")
    named = {}
    for s, series in prices_in_base.items():
        try:
            named[s] = _validated_series(series, s)
        except ValueError as exc:
            raise BacktestError(str(exc)) from exc
    if benchmark_symbol is not None:
        try:
            named[benchmark_symbol] = _validated_series(benchmark_prices_in_base, benchmark_symbol)
        except ValueError as exc:
            raise BacktestError(str(exc)) from exc
    frame = pd.concat(named, axis=1, join="inner").sort_index()
    try:
        frame = _check_price_frame(frame)
    except PerformanceError as exc:
        raise BacktestError(str(exc)) from exc
    if len(frame) < min_observations:
        raise BacktestError(
            f"Only {len(frame)} shared adjusted-close dates (need {min_observations}); "
            "reduce the asset set or extend the history window"
        )
    asset_names = list(prices_in_base)
    strategy_prices = frame[asset_names]
    fee_rate = fee_bps / 10000.0
    results = {
        schedule: _run_one(
            strategy_prices, weights, schedule, initial_capital, fee_rate, annual_risk_free_rate
        )
        for schedule in schedules
    }
    bm_metrics = None
    bm_values = None
    if benchmark_symbol:
        bench = frame[benchmark_symbol]
        bm_values = initial_capital * bench / float(bench.iloc[0])
        bench_returns = bm_values.pct_change(fill_method=None).iloc[1:]
        bm_metrics = _metrics(bench_returns, frame.index, annual_risk_free_rate)
    notes = [
        "Initial weights come from the supplied snapshot and are applied retroactively; "
        "this does not reproduce actual personal account performance.",
        "Adjusted-close prices are total-return proxies, not executable closing prices. "
        "Trades are idealized close fills; actual executions can differ.",
        "The simulation permits fractional units and omits taxes, bid/ask spreads, "
        "market impact, and cash contributions/withdrawals.",
    ]
    if benchmark_symbol:
        notes.append(
            "Benchmark is a fee-free, buy-and-hold total-return index; "
            "strategy entry and turnover fees are included when specified."
        )
    if len(frame) - 1 < 252:
        notes.append("SHORT_SAMPLE: fewer than 252 common-close return periods.")
    return BacktestReport(
        first_date=frame.index[0].date().isoformat(),
        last_date=frame.index[-1].date().isoformat(),
        observation_dates=len(frame),
        return_periods=len(frame) - 1,
        base_currency=base_currency,
        initial_capital=initial_capital,
        target_weights=dict(zip(asset_names, weights.tolist(), strict=False)),
        fee_bps=fee_bps,
        risk_free_annual=annual_risk_free_rate,
        strategies=results,
        benchmark_symbol=benchmark_symbol,
        benchmark=bm_metrics,
        benchmark_values=bm_values,
        warnings=tuple(notes),
    )


def backtest_as_dict(
    report: BacktestReport, *, include_series: bool = True, include_trades: bool = True
) -> dict:
    """Serialize a backtest report for CLI or a future dashboard."""
    from dataclasses import asdict

    output = {
        "methodology": "Historical adjusted-price, fractional-unit close-fill simulation",
        "first_date": report.first_date,
        "last_date": report.last_date,
        "observation_dates": report.observation_dates,
        "return_periods": report.return_periods,
        "base_currency": report.base_currency,
        "initial_capital": report.initial_capital,
        "target_weights": report.target_weights,
        "fee_bps": report.fee_bps,
        "annual_risk_free_rate": report.risk_free_annual,
        "benchmark_symbol": report.benchmark_symbol,
        "benchmark": asdict(report.benchmark) if report.benchmark else None,
        "warnings": list(report.warnings),
        "strategies": {},
    }
    for name, row in report.strategies.items():
        detail = {
            "metrics": asdict(row.metrics),
            "ending_value": row.ending_value,
            "total_fees": row.total_fees,
            "traded_notional": row.traded_notional,
            "initial_fees": row.initial_fees,
            "rebalance_count": row.rebalance_count,
            "trade_count": row.trade_count,
            "ending_weights": row.ending_weights,
            "maximum_weight_drift": row.maximum_weight_drift,
        }
        if include_series:
            detail["portfolio_values"] = [
                {"date": d.date().isoformat(), "value": float(v)} for d, v in row.values.items()
            ]
            detail["period_returns"] = [
                {"date": d.date().isoformat(), "return": float(v)} for d, v in row.returns.items()
            ]
            detail["weights_over_time"] = [
                {
                    "date": d.date().isoformat(),
                    "weights": {symbol: float(w) for symbol, w in values.items()},
                }
                for d, values in row.weights_over_time.iterrows()
            ]
        if include_trades:
            detail["trades"] = [asdict(t) for t in row.trades]
        output["strategies"][name] = detail
    if include_series and report.benchmark_values is not None:
        output["benchmark_values"] = [
            {"date": d.date().isoformat(), "value": float(v)}
            for d, v in report.benchmark_values.items()
        ]
    return output
