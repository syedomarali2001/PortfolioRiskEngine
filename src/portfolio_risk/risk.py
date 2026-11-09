"""V5: deterministic portfolio risk analytics; no APIs or file operations.

Convention: VaR/ES are positive loss magnitudes (floor zero). Inputs are
base-currency observed common-close simple returns from a V4 strategy.
Historical tail ES includes fractional probability mass at the cutoff.
Forecasts use strictly preceding observations and are evaluated one step ahead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, isfinite, log, sqrt
from statistics import NormalDist
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk.performance import TRADING_DAYS_PER_YEAR


class RiskError(ValueError):
    """Invalid input or insufficient observations for risk analytics."""


@dataclass(frozen=True)
class TailRisk:
    model: str
    confidence: float
    value_at_risk: float  # positive fraction of capital at risk
    expected_shortfall: float  # positive fraction of capital at risk
    var_amount: float  # base currency
    es_amount: float  # base currency


@dataclass(frozen=True)
class RiskAttribution:
    symbol: str
    weight: float
    standalone_volatility: float  # annualized
    contribution_to_volatility: float  # annualized, may be negative for hedge
    share_of_total_risk: float  # sum of shares = 1, may be negative


@dataclass(frozen=True)
class CoverageTest:
    model: str
    confidence: float
    lookback: int
    tests: int
    breaches: int
    breach_fraction: float
    expected_breaches: float
    kupiec_lr: float | None
    kupiec_p_value: float | None
    # Last forecast uses observations only preceding the final test return.
    last_forecast_date: str | None
    last_var: float | None


@dataclass(frozen=True)
class StressResult:
    label: str
    source: str
    portfolio_return: float
    change_on_capital: float
    description: str


@dataclass(frozen=True)
class RiskReport:
    first_date: str
    last_date: str
    base_currency: str
    strategy: str
    starting_capital: float
    reference_capital: float
    lookback: int
    estimation_observations: int
    selected_strategy_returns: pd.Series
    risk_estimates: tuple[TailRisk, ...]
    attribution: tuple[RiskAttribution, ...]
    portfolio_volatility: float
    diversification_ratio: float | None
    coverage: tuple[CoverageTest, ...]
    stresses: tuple[StressResult, ...]
    warnings: tuple[str, ...]


def _valid_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        raise RiskError("Returns must be a pandas Series with a DatetimeIndex")
    if returns.empty or returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise RiskError("Returns must have sorted unique observation dates")
    if returns.index.tz is not None or not returns.index.equals(returns.index.normalize()):
        raise RiskError("Return dates must be timezone-naive daily dates")
    arr = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(arr).all() or (arr <= -1).any():
        raise RiskError("Returns must be finite and above -100%")
    return pd.Series(arr, index=returns.index, name=returns.name, dtype=float)


def _tail_estimates(
    sample: np.ndarray, confidence: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return historical and normal-model (VaR, ES) in positive loss units."""
    if sample.ndim != 1 or len(sample) < 2 or not np.isfinite(sample).all():
        raise RiskError("Need at least two finite historical returns")
    if not (0 < confidence < 1):
        raise RiskError("Confidence level must be strictly between 0 and 1")
    losses = -sample
    hist_var = max(0.0, float(np.quantile(losses, confidence)))
    # Exact empirical tail mean with fractional mass at the cutoff; avoids
    # instability from np.mean(losses >= VaR) when there are few observations.
    sorted_losses = np.sort(losses)[::-1]
    tail_count = len(sorted_losses) * (1.0 - confidence)
    whole = int(np.floor(tail_count))
    fractional = tail_count - whole
    total = float(sorted_losses[:whole].sum())
    if fractional > 1e-12:
        total += fractional * float(sorted_losses[whole])
    hist_es = max(hist_var, 0.0, total / tail_count)
    mean = float(sample.mean())
    std = float(sample.std(ddof=1))
    z = NormalDist().inv_cdf(confidence)
    phi = np.exp(-z * z / 2.0) / sqrt(2.0 * np.pi)
    normal_var = max(0.0, -mean + z * std)
    normal_es = max(normal_var, 0.0, -mean + std * phi / (1.0 - confidence))
    return (hist_var, hist_es), (normal_var, normal_es)


def _coverage(sample: pd.Series, lookback: int, confidence: float, model: str) -> CoverageTest:
    observations = sample.to_numpy(dtype=float)
    exceptions = 0
    last_forecast = None
    tests = max(0, len(observations) - lookback)
    for i in range(lookback, len(observations)):
        train = observations[i - lookback : i]  # NO look-ahead
        hist, normal = _tail_estimates(train, confidence)
        forecast = hist[0] if model == "historical" else normal[0]
        if -observations[i] > forecast + 1e-12:
            exceptions += 1
        last_forecast = forecast
    expected = tests * (1 - confidence)
    rate = exceptions / tests if tests else 0.0
    lr, pvalue = None, None
    if tests:
        null_breach = 1.0 - confidence
        ll_null = exceptions * log(null_breach) + (tests - exceptions) * log(1 - null_breach)
        ll_alt = (exceptions * log(rate) if exceptions else 0.0) + (
            (tests - exceptions) * log(1 - rate) if exceptions < tests else 0.0
        )
        lr = max(0.0, 2.0 * (ll_alt - ll_null))
        pvalue = erfc(sqrt(lr / 2.0))  # chi-square(1) asymptotic survival function
    return CoverageTest(
        model=model,
        confidence=confidence,
        lookback=lookback,
        tests=tests,
        breaches=exceptions,
        breach_fraction=rate,
        expected_breaches=expected,
        kupiec_lr=lr,
        kupiec_p_value=pvalue,
        last_forecast_date=sample.index[-1].date().isoformat() if tests else None,
        last_var=last_forecast,
    )


def _asset_attribution(
    returns: pd.DataFrame, weights: Mapping[str, float]
) -> tuple[tuple[RiskAttribution, ...], float, float | None]:
    names = list(weights)
    subset = returns[names]
    vals = subset.to_numpy(dtype=float)
    if len(vals) < 2 or not np.isfinite(vals).all():
        raise RiskError("Need at least two complete asset-return periods")
    w = np.asarray([weights[n] for n in names], dtype=float)
    covariance = np.atleast_2d(np.cov(vals, rowvar=False, ddof=1))
    variance = max(0.0, float(w @ covariance @ w))
    sigma = sqrt(variance)
    standalone = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if sigma < 1e-14:
        contributions = np.zeros_like(w)
        shares = np.zeros_like(w)
        ratio = None
    else:
        contributions = w * (covariance @ w) / sigma
        shares = contributions / sigma
        ratio = float(np.dot(w, standalone) / sigma)
    result = tuple(
        RiskAttribution(
            symbol=n,
            weight=float(w[i]),
            standalone_volatility=float(standalone[i] * sqrt(TRADING_DAYS_PER_YEAR)),
            contribution_to_volatility=float(contributions[i] * sqrt(TRADING_DAYS_PER_YEAR)),
            share_of_total_risk=float(shares[i]),
        )
        for i, n in enumerate(names)
    )
    return result, sigma * sqrt(TRADING_DAYS_PER_YEAR), ratio


def analyze_risk(
    asset_prices_in_base: Mapping[str, pd.Series],
    actual_weights: Mapping[str, float],
    strategy_returns: pd.Series,
    *,
    strategy: str = "buy_hold",
    base_currency: str = "CAD",
    starting_capital: float = 20000.0,
    reference_capital: float | None = None,
    lookback: int = 252,
    confidences: tuple[float, ...] = (0.95, 0.99),
    shocks: Mapping[str, float] | None = None,
) -> RiskReport:
    """Analyze one V4 simulation, using only information within the sample.

    Current VaR uses the last lookback strategy returns, e.g. quarterly
    portfolio returns including its simulated trading costs. Rolling coverage
    uses *strictly preceding* observations for each next-period forecast.

    Risk contributions are a local covariance-based estimate at the final
    strategy weights and apply to the trailing asset-return window. They are
    not a full decomposition of historical fee-inclusive strategy volatility.
    """
    if base_currency not in {"CAD", "USD"}:
        raise RiskError("Base currency must be CAD or USD")
    if not isfinite(starting_capital) or starting_capital <= 0:
        raise RiskError("Starting capital must be positive and finite")
    if reference_capital is None:
        reference_capital = starting_capital
    if not isfinite(reference_capital) or reference_capital <= 0:
        raise RiskError("Reference capital must be positive and finite")
    if not isinstance(lookback, int) or lookback < 30:
        raise RiskError("Lookback must be an integer >= 30")
    if not asset_prices_in_base or set(asset_prices_in_base) != set(actual_weights):
        raise RiskError("Price symbols and weights must match")
    if not confidences or len(set(confidences)) != len(confidences):
        raise RiskError("Confidence levels must be unique and nonempty")
    if any(not 0 < a < 1 for a in confidences):
        raise RiskError("Confidence levels must be between 0 and 1")
    weights = {key: float(value) for key, value in actual_weights.items()}
    if any(not isfinite(w) or w < 0 for w in weights.values()):
        raise RiskError("Weights must be finite and nonnegative")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise RiskError("Weights must sum to one")
    if shocks is not None and (set(shocks) - set(weights)):
        raise RiskError("Unrecognized shock symbols")
    if shocks is not None and any(not isfinite(v) or v < -1 for v in shocks.values()):
        raise RiskError("Each shock must be finite and at least -100%")

    series = _valid_returns(strategy_returns)
    # Shared price dates are inherited from V4. Restrict to the exact V4 dates
    # so independently cleaned/converted sources cannot introduce misalignment.
    from portfolio_risk.historical_fx import _validated_series

    named = {k: _validated_series(v, k) for k, v in asset_prices_in_base.items()}
    prices = pd.concat(named, axis=1, join="inner").sort_index()
    asset_returns = prices.pct_change(fill_method=None).iloc[1:]
    if not series.index.isin(asset_returns.index).all():
        raise RiskError("Strategy returns have dates absent from asset histories")
    asset_returns = asset_returns.reindex(series.index)
    if asset_returns.isna().any().any():
        raise RiskError("Missing asset prices on strategy return dates")
    if len(series) < 30:
        raise RiskError("At least 30 strategy returns required for V5 risk analysis")
    tail_n = min(lookback, len(series))
    recent = series.tail(tail_n)
    asset_recent = asset_returns.tail(tail_n)
    estimates = []
    coverage = []
    for confidence in confidences:
        hist, normal = _tail_estimates(recent.to_numpy(), confidence)
        for model, (var, es) in (("historical", hist), ("parametric_normal", normal)):
            estimates.append(
                TailRisk(
                    model,
                    confidence,
                    var,
                    es,
                    var * reference_capital,
                    es * reference_capital,
                )
            )
            coverage.append(_coverage(series, lookback, confidence, model))
    attribution, volatility, diversification = _asset_attribution(asset_recent, weights)
    stresses = []
    worst_one = float(series.min())
    stresses.append(
        StressResult(
            label="Worst observed common-date return",
            source="observed_strategy",
            portfolio_return=worst_one,
            change_on_capital=worst_one * reference_capital,
            description=f"Worst interval among {len(series)} observed V4 strategy return periods",
        )
    )
    if len(series) >= 20:
        rolling = (1.0 + series).rolling(20).apply(np.prod, raw=True) - 1.0
        worst_twenty = float(rolling.min())
        stresses.append(
            StressResult(
                label="Worst observed 20-period compounded return",
                source="observed_strategy",
                portfolio_return=worst_twenty,
                change_on_capital=worst_twenty * reference_capital,
                description="20 consecutive common-close return periods, not 20 calendar days",
            )
        )
    if shocks:
        shock_return = float(sum(weights[s] * shocks.get(s, 0.0) for s in weights))
        stresses.append(
            StressResult(
                label="User-defined instantaneous asset shocks",
                source="hypothetical",
                portfolio_return=shock_return,
                change_on_capital=shock_return * reference_capital,
                description="Single-period shocks on ending weights; unspecified assets unchanged",
            )
        )
    warnings = [
        "Risk estimates describe distributions of historical V4 strategy returns; they are not future-loss guarantees.",
        "Asset risk attribution uses trailing covariance and FINAL simulated portfolio weights; "
        "it does not decompose full historical turnover or trading-fee effects.",
        "Normal VaR/ES assumes normally distributed simple returns; extreme market returns may violate that assumption.",
        "Historical VaR/ES uses only the selected sample; missing pre-inception and crisis periods cannot be inferred.",
        "Rolling VaR coverage uses preceding returns only. Kupiec p-values are asymptotic, "
        "test frequency only (not independence), and are unreliable with few expected breaches.",
    ]
    if len(series) < lookback + 20:
        warnings.append(
            "SHORT_COVERAGE: fewer than 20 validation periods after lookback; do not judge calibration."
        )
    if tail_n < lookback:
        warnings.append(
            f"SHORT_ESTIMATION: using {tail_n} rather than {lookback} requested lookback periods"
        )
    if any(len(series) - lookback < 5 / (1 - confidence) for confidence in confidences):
        warnings.append(
            "SPARSE_EXCEEDANCES: one or more confidence levels have fewer than five expected breaches"
        )
    return RiskReport(
        first_date=series.index[0].date().isoformat(),
        last_date=series.index[-1].date().isoformat(),
        base_currency=base_currency,
        strategy=strategy,
        starting_capital=starting_capital,
        reference_capital=reference_capital,
        lookback=lookback,
        estimation_observations=tail_n,
        selected_strategy_returns=series,
        risk_estimates=tuple(estimates),
        attribution=attribution,
        portfolio_volatility=volatility,
        diversification_ratio=diversification,
        coverage=tuple(coverage),
        stresses=tuple(stresses),
        warnings=tuple(warnings),
    )


def risk_as_dict(report: RiskReport, *, include_series: bool = False) -> dict:
    data = {
        "first_date": report.first_date,
        "last_date": report.last_date,
        "base_currency": report.base_currency,
        "strategy": report.strategy,
        "starting_capital": report.starting_capital,
        "reference_capital": report.reference_capital,
        "lookback": report.lookback,
        "estimation_observations": report.estimation_observations,
        "risk_estimates": [asdict(x) for x in report.risk_estimates],
        "risk_attribution": [asdict(x) for x in report.attribution],
        "annualized_portfolio_volatility_at_final_weights": report.portfolio_volatility,
        "diversification_ratio": report.diversification_ratio,
        "coverage": [asdict(x) for x in report.coverage],
        "stress_scenarios": [asdict(x) for x in report.stresses],
        "warnings": list(report.warnings),
    }
    if include_series:
        data["strategy_returns"] = [
            {"date": d.date().isoformat(), "return": float(v)}
            for d, v in report.selected_strategy_returns.items()
        ]
    return data
