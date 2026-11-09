"""V6: transparent, long-only, walk-forward portfolio allocation simulations.

Pure calculation layer. Never downloads prices, optimizes against future outcomes,
or accesses current holdings other than the supplied *reference* weights.

Each rebalance signal uses returns THROUGH THE PREVIOUS COMMON CLOSE and
applies target weights at the CURRENT idealized close. This deliberately
avoids using a closing price to determine an order assumed executed at that
same close. Rebalancing happens at the first common close of a new calendar
period, using the current and previous known dates (no next-date peeking).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from portfolio_risk.backtest import Trade, _rebalance_at_close
from portfolio_risk.historical_fx import _validated_series
from portfolio_risk.performance import Metrics, PerformanceError, _check_price_frame, _metrics

METHODS = ("input_buy_hold", "input_rebalanced", "equal_weight", "inverse_vol", "min_variance", "momentum_top_k")
FREQUENCIES = ("monthly", "quarterly", "semiannual", "annual")


class ConstructionError(ValueError):
    """Invalid model inputs, insufficient price history, or unstable results."""


@dataclass(frozen=True)
class AllocationResult:
    method: str
    metrics: Metrics
    evaluation_metrics: Metrics | None
    ending_value: float
    total_fees: float
    traded_notional: float
    rebalance_count: int
    trade_count: int
    ending_weights: dict[str, float]
    values: pd.Series
    returns: pd.Series
    weights_over_time: pd.DataFrame
    trades: tuple[Trade, ...]
    last_target: dict[str, float]


@dataclass(frozen=True)
class ConstructionReport:
    first_date: str
    last_date: str
    observation_dates: int
    return_periods: int
    historical_lookback: int
    momentum_skip: int
    top_k: int
    max_weight: float
    shrinkage: float
    rebalance_frequency: str
    initial_capital: float
    base_currency: str
    fee_bps: float
    risk_free_annual: float
    reference_weights: dict[str, float]
    results: dict[str, AllocationResult]
    evaluation_start: str | None
    benchmark_symbol: str | None
    benchmark_metrics: Metrics | None
    benchmark_evaluation_metrics: Metrics | None
    warnings: tuple[str, ...]


def _first_close_of_new_period(dates: pd.DatetimeIndex, position: int,
                               frequency: str) -> bool:
    """Decide a rebalance from the current date and immediately previous date."""
    if position <= 0 or position >= len(dates):
        return False
    previous, current = dates[position-1], dates[position]
    if frequency == "monthly":
        return (previous.year, previous.month) != (current.year, current.month)
    if frequency == "quarterly":
        return (previous.year, (previous.month-1)//3) != (current.year, (current.month-1)//3)
    if frequency == "semiannual":
        return (previous.year, (previous.month-1)//6) != (current.year, (current.month-1)//6)
    if frequency == "annual":
        return previous.year != current.year
    raise ConstructionError("Unknown calendar frequency")


def _capped_simplex_projection(vector: np.ndarray, maximum: float) -> np.ndarray:
    """Euclidean projection of vector to sum(w)=1 with 0<=w<=maximum."""
    x = np.asarray(vector, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or x.size == 0:
        raise ConstructionError("Projection requires finite, nonempty weights")
    if not isfinite(maximum) or maximum <= 0 or x.size * maximum < 1 - 1e-12:
        raise ConstructionError("max_weight is too small for the available assets")
    lo = float(np.min(x - maximum)) - 1
    hi = float(np.max(x)) + 1
    for _ in range(100):
        mid = (lo + hi) / 2
        if float(np.clip(x - mid, 0.0, maximum).sum()) > 1:
            lo = mid
        else:
            hi = mid
    w = np.clip(x - (lo + hi) / 2, 0.0, maximum)
    if abs(float(w.sum()) - 1) > 1e-9:
        raise ConstructionError("Capped weight projection failed")
    return w


def _minimum_variance(covariance: np.ndarray, *, max_weight: float,
                      shrinkage: float) -> np.ndarray:
    """Long-only minimum variance via projected gradient, with diagonal shrinkage.

    No expected-return prediction. Convex on PSD sample covariance; small
    diagonal shrinkage makes ill-conditioned samples less brittle.
    """
    cov = np.atleast_2d(np.asarray(covariance, dtype=float))
    n = cov.shape[0]
    if cov.shape != (n, n) or not np.isfinite(cov).all():
        raise ConstructionError("Covariance must be a finite square matrix")
    cov = (cov + cov.T) / 2
    shrunk = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
    scale = float(max(np.max(np.diag(shrunk)), 1e-12))
    normalized = shrunk / scale
    lipschitz = float(max(np.linalg.eigvalsh(normalized)[-1], 1e-12))
    w = np.full(n, 1.0 / n)
    for _ in range(6000):
        candidate = _capped_simplex_projection(w - normalized @ w / lipschitz, max_weight)
        if np.max(np.abs(candidate - w)) < 1e-11:
            w = candidate
            break
        w = candidate
    return w


def _allocation(method: str, samples: np.ndarray, reference: np.ndarray, *,
                max_weight: float, shrinkage: float, top_k: int,
                skip: int) -> np.ndarray:
    """Estimate target from strictly older *asset* returns (rows=dates)."""
    n = samples.shape[1]
    if method in {"input_buy_hold", "input_rebalanced"}:
        return reference.copy()  # reference allocations are an ex-post research scenario
    if method == "equal_weight":
        return np.ones(n) / n
    if method == "momentum_top_k":
        if samples.shape[0] <= skip or top_k > n:
            raise ConstructionError("Insufficient momentum observations or too few assets")
        past = samples[:-skip] if skip else samples
        score = np.prod(1.0 + past, axis=0) - 1.0
        # stable tie-breaking by input asset order
        chosen = np.argsort(-score, kind="stable")[:top_k]
        weights = np.zeros(n)
        weights[chosen] = 1.0 / top_k
        if (weights > max_weight + 1e-12).any():
            raise ConstructionError("max_weight too small for momentum top_k")
        return weights
    window = samples  # lookback sample, ending BEFORE today's close
    if method == "inverse_vol":
        vol = np.std(window, axis=0, ddof=1)
        # avoid infinite leverage / weight from constant-price instruments
        inv = 1.0 / np.maximum(vol, 1e-6)
        return _capped_simplex_projection(inv / inv.sum(), max_weight)
    if method == "min_variance":
        cov = np.atleast_2d(np.cov(window, rowvar=False, ddof=1))
        return _minimum_variance(cov, max_weight=max_weight, shrinkage=shrinkage)
    raise ConstructionError(f"Unknown strategy {method}")


def _strategy(
    prices: pd.DataFrame,
    reference: np.ndarray,
    method: str,
    *,
    first_position: int,
    lookback: int,
    skip: int,
    top_k: int,
    max_weight: float,
    shrinkage: float,
    fee_rate: float,
    capital: float,
    frequency: str,
    annual_risk_free: float,
    evaluation_start: date | None,
) -> AllocationResult:
    full_dates = prices.index
    full_prices = prices.to_numpy(dtype=float)
    full_returns = prices.pct_change(fill_method=None).iloc[1:].to_numpy(dtype=float)
    symbols = list(prices.columns)
    n = len(symbols)

    def target_at(i: int) -> np.ndarray:
        # Full-return row j represents close(j+1)/close(j)-1.
        # Most recent usable row at close i: row i-2 (previous close).
        ending = i - 1  # exclusive, so previous-close returns are last included
        count = lookback + (skip if method == "momentum_top_k" else 0)
        sample = full_returns[ending-count:ending]
        if sample.shape != (count, n):
            raise ConstructionError("Not enough prior observations for signal")
        result = _allocation(method, sample, reference, max_weight=max_weight,
                             shrinkage=shrinkage, top_k=top_k, skip=skip)
        if not np.isfinite(result).all() or (result < -1e-10).any() or abs(result.sum()-1) > 1e-9:
            raise ConstructionError("Invalid strategy weights")
        return result

    dates = full_dates[first_position:]
    matrix = full_prices[first_position:]
    first_target = target_at(first_position)
    funded = capital / (1 + fee_rate)
    positions = funded * first_target / matrix[0]
    initial_fees = capital - funded
    costs = initial_fees
    traded = funded
    trades = [Trade(dates[0].date().isoformat(), symbol, float(positions[j]),
                    float(funded * first_target[j]), float(initial_fees * first_target[j]),
                    "initial") for j, symbol in enumerate(symbols) if first_target[j] > 1e-12]
    values = np.empty(len(dates))
    values[0] = capital  # V4 convention: entry cost appears in next realized return
    history = np.empty((len(dates), n))
    history[0] = first_target
    rebalances = 0
    last_target = first_target.copy()
    for step in range(1, len(dates)):
        absolute_i = first_position + step
        before = positions * matrix[step]
        wealth = float(before.sum())
        if not isfinite(wealth) or wealth <= 0:
            raise ConstructionError("Nonpositive portfolio wealth")
        if method != "input_buy_hold" and _first_close_of_new_period(dates, step, frequency):
            desired = target_at(absolute_i)
            new_pos, fee, changes = _rebalance_at_close(positions, matrix[step], desired, fee_rate)
            for j, symbol in enumerate(symbols):
                if abs(changes[j]) > wealth * 1e-11:
                    trades.append(Trade(dates[step].date().isoformat(), symbol,
                                        float(new_pos[j]-positions[j]), float(changes[j]),
                                        float(abs(changes[j])*fee_rate), "rebalance"))
            costs += fee
            traded += float(np.abs(changes).sum())
            positions = new_pos
            last_target = desired
            rebalances += 1
        values[step] = float(np.dot(positions, matrix[step]))
        history[step] = positions * matrix[step] / values[step]
        if not isfinite(values[step]) or values[step] <= 0:
            raise ConstructionError("Nonfinite wealth")
    nav = pd.Series(values, index=dates, name=method)
    returns = nav.pct_change(fill_method=None).iloc[1:]
    metrics = _metrics(returns, dates, annual_risk_free)
    eval_metrics = None
    if evaluation_start is not None:
        i = int(dates.searchsorted(pd.Timestamp(evaluation_start)))
        if i < 1 or i >= len(dates)-2:
            raise ConstructionError("Evaluation date must leave a prior observation and two future returns")
        # Evaluation start is the first observed return date >= cutoff.
        # Include the prior close only as a comparison anchor.
        eval_returns = returns.iloc[i-1:]
        eval_dates = dates[i-1:]
        eval_metrics = _metrics(eval_returns, eval_dates, annual_risk_free)
    ending = positions * matrix[-1]
    final = float(ending.sum())
    return AllocationResult(method, metrics, eval_metrics, float(nav.iloc[-1]),
                            float(costs), float(traded), rebalances, len(trades),
                            {s: float(ending[j]/final) for j, s in enumerate(symbols)},
                            nav, returns,
                            pd.DataFrame(history, index=dates, columns=symbols),
                            tuple(trades), {s: float(last_target[j]) for j,s in enumerate(symbols)})


def run_construction(
    asset_prices_in_base: Mapping[str, pd.Series],
    reference_weights: Mapping[str, float],
    *,
    capital: float = 20000.0,
    fee_bps: float = 0.0,
    base_currency: str = "CAD",
    annual_risk_free: float = 0.0,
    frequency: str = "quarterly",
    methods: Sequence[str] = METHODS,
    lookback: int = 252,
    momentum_skip: int = 21,
    top_k: int = 2,
    max_weight: float = 0.6,
    shrinkage: float = 0.15,
    evaluation_start: date | None = None,
    benchmark_prices_in_base: pd.Series | None = None,
    benchmark_symbol: str | None = None,
) -> ConstructionReport:
    """Compare long-only strategies on one common evaluation window.

    ALL strategies start after a shared warmup: max lookback+skip for a fair
    comparison, and no strategy sees prices from the current trade close.
    A strategy's eligible universe is supplied by the user; this can introduce
    selection/survivorship bias even when signal calculations are leak-free.
    """
    if base_currency not in {"CAD", "USD"}:
        raise ConstructionError("Currency must be CAD or USD")
    if not isfinite(capital) or capital <= 0:
        raise ConstructionError("Capital must be positive and finite")
    if not isfinite(fee_bps) or not 0 <= fee_bps <= 10000:
        raise ConstructionError("Fee must be within 0..10000 bps")
    if not isfinite(annual_risk_free) or annual_risk_free <= -1:
        raise ConstructionError("Risk-free rate must be greater than -100%")
    if frequency not in FREQUENCIES:
        raise ConstructionError(f"Frequency must be one of: {', '.join(FREQUENCIES)}")
    if not isinstance(lookback, int) or lookback < 30:
        raise ConstructionError("Lookback must be >= 30 common-close returns")
    if not isinstance(momentum_skip, int) or momentum_skip < 0 or momentum_skip > lookback:
        raise ConstructionError("Momentum skip must be within 0..lookback")
    if not methods or len(methods) != len(set(methods)) or set(methods)-set(METHODS):
        raise ConstructionError("Unknown or duplicate methods")
    if not isinstance(top_k, int) or top_k < 1:
        raise ConstructionError("top_k must be a positive integer")
    if not isfinite(max_weight) or not 0 < max_weight <= 1:
        raise ConstructionError("max_weight must be between 0 and 1")
    if not isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ConstructionError("shrinkage must be between 0 and 1")
    if not asset_prices_in_base or set(reference_weights) != set(asset_prices_in_base):
        raise ConstructionError("Reference weights and prices must match")
    symbols = list(asset_prices_in_base)
    n = len(symbols)
    ref = np.array([float(reference_weights[s]) for s in symbols])
    if not np.isfinite(ref).all() or (ref <= 0).any() or abs(float(ref.sum())-1)>1e-9:
        raise ConstructionError("Reference weights must be positive and sum to 1")
    if top_k > n and "momentum_top_k" in methods:
        raise ConstructionError("top_k cannot exceed number of assets")
    if n*max_weight < 1-1e-12 and any(m in methods for m in ("equal_weight", "min_variance", "inverse_vol")):
        raise ConstructionError("max_weight too restrictive for this asset universe")
    if "momentum_top_k" in methods and top_k*max_weight < 1-1e-12:
        raise ConstructionError("max_weight must be at least 1/top_k for momentum")
    if (benchmark_prices_in_base is None)!=(benchmark_symbol is None):
        raise ConstructionError("Benchmark price and name must be provided together")
    if benchmark_symbol in asset_prices_in_base:
        raise ConstructionError("Benchmark must be a different ticker")
    named = {}
    try:
        for symbol, series in asset_prices_in_base.items():
            named[symbol] = _validated_series(series, symbol)
        if benchmark_symbol:
            named[benchmark_symbol] = _validated_series(benchmark_prices_in_base, benchmark_symbol)
        frame = _check_price_frame(pd.concat(named, axis=1, join="inner").sort_index())
    except (ValueError, PerformanceError) as exc:
        raise ConstructionError(str(exc)) from exc
    # Need 1 previous-close buffer beyond all estimation returns.
    warmup = lookback + (momentum_skip if "momentum_top_k" in methods else 0) + 1
    if len(frame) - warmup < 30:
        raise ConstructionError(
            f"Not enough shared observations for V6: {len(frame)} available, "
            f"need at least {warmup+30} (warmup {warmup})"
        )
    common_prices = frame[symbols]
    fee_rate = fee_bps / 10000
    outcomes = {m: _strategy(common_prices, ref, m, first_position=warmup,
                             lookback=lookback, skip=momentum_skip, top_k=top_k,
                             max_weight=max_weight, shrinkage=shrinkage,
                             fee_rate=fee_rate, capital=capital, frequency=frequency,
                             annual_risk_free=annual_risk_free,
                             evaluation_start=evaluation_start)
                for m in methods}
    started = frame.index[warmup:]
    bmetrics = None
    beval = None
    if benchmark_symbol:
        price = frame[benchmark_symbol].iloc[warmup:]
        bret = price.pct_change(fill_method=None).iloc[1:]
        bmetrics = _metrics(bret, started, annual_risk_free)
        if evaluation_start:
            first_eval = int(started.searchsorted(pd.Timestamp(evaluation_start)))
            beval = _metrics(bret.iloc[first_eval-1:], started[first_eval-1:], annual_risk_free)
    notes = [
        "Target weights provided today are applied retroactively as a reference only, not actual personal performance.",
        "Strategy signals use only observations through the PREVIOUS common close, never the current trade close.",
        "Rebalances occur at the first observed common close of a new calendar period, determined without future dates.",
        "Adjusted-close total-return proxies and fractional close fills are idealized; taxes, spread, slippage and market impact are omitted.",
        "The asset universe is selected with hindsight and may exhibit survivorship/selection bias; walk-forward signal timing does NOT establish an untouched out-of-sample result.",
        "Minimum-variance covariance is diagonally shrunk; it does not forecast expected returns.",
        "Momentum rotates among the top assets even if all have negative momentum; there is no cash fallback.",
    ]
    if evaluation_start:
        notes.append("Evaluation metrics summarize a later calendar segment, but model selection after inspecting it invalidates an untouched holdout.")
    return ConstructionReport(started[0].date().isoformat(), started[-1].date().isoformat(),
                              len(started), len(started)-1, lookback, momentum_skip,
                              top_k, max_weight, shrinkage, frequency, capital,
                              base_currency, fee_bps, annual_risk_free,
                              dict(zip(symbols, ref.tolist())), outcomes,
                              evaluation_start.isoformat() if evaluation_start else None,
                              benchmark_symbol, bmetrics, beval, tuple(notes))


def construction_as_dict(report: ConstructionReport, *, include_series: bool = True,
                         include_trades: bool = True) -> dict:
    out = {
        "methodology": "V6 ex-ante rolling allocation research on observed common closes",
        "first_date": report.first_date, "last_date": report.last_date,
        "observation_dates": report.observation_dates, "return_periods": report.return_periods,
        "lookback": report.historical_lookback, "momentum_skip": report.momentum_skip,
        "top_k": report.top_k, "max_weight": report.max_weight,
        "shrinkage": report.shrinkage, "rebalance_frequency": report.rebalance_frequency,
        "initial_capital": report.initial_capital, "base_currency": report.base_currency,
        "fee_bps": report.fee_bps, "risk_free_annual": report.risk_free_annual,
        "reference_weights": report.reference_weights,
        "benchmark_symbol": report.benchmark_symbol,
        "benchmark_metrics": asdict(report.benchmark_metrics) if report.benchmark_metrics else None,
        "benchmark_evaluation_metrics": asdict(report.benchmark_evaluation_metrics) if report.benchmark_evaluation_metrics else None,
        "evaluation_start": report.evaluation_start,
        "warnings": list(report.warnings), "strategies": {},
    }
    for name, row in report.results.items():
        info = {
            "ending_value": row.ending_value, "metrics": asdict(row.metrics),
            "evaluation_metrics": asdict(row.evaluation_metrics) if row.evaluation_metrics else None,
            "total_fees": row.total_fees, "traded_notional": row.traded_notional,
            "rebalance_count": row.rebalance_count, "trade_count": row.trade_count,
            "ending_weights": row.ending_weights, "last_target": row.last_target,
        }
        if include_series:
            info["portfolio_values"] = [{"date": ts.date().isoformat(), "value": float(v)}
                                        for ts, v in row.values.items()]
            info["weights_over_time"] = [
                {"date": ts.date().isoformat(), "weights": {s: float(w) for s,w in r.items()}}
                for ts,r in row.weights_over_time.iterrows()
            ]
        if include_trades:
            info["trades"] = [asdict(t) for t in row.trades]
        out["strategies"][name] = info
    return out
