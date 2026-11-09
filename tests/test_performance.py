"""Deterministic, closed-form checks. Nothing fetches market data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.performance import PerformanceError, analyze_performance, performance_as_dict


def prices(values, dates=None):
    if dates is None:
        dates = pd.date_range("2025-01-02", periods=len(values), freq="B")
    return pd.Series(values, index=pd.DatetimeIndex(dates), dtype=float)


def calc(series, weights=None, **kwargs):
    if weights is None:
        weights = {symbol: 1 / len(series) for symbol in series}
    return analyze_performance(series, weights, min_observations=3, **kwargs)


def test_weighted_period_returns_exact_simple_example():
    a = prices([100, 110, 121, 108.9])  # +10%, +10%, -10%
    b = prices([100, 100, 100, 100])
    report = calc({"A": a, "B": b}, {"A": 0.4, "B": 0.6})
    assert list(report.period_returns) == pytest.approx([.04, .04, -.04])
    assert report.portfolio_index.iloc[-1] == pytest.approx(100 * 1.04 * 1.04 * .96)
    assert report.portfolio.cumulative_return == pytest.approx(1.04 * 1.04 * .96 - 1)
    assert report.portfolio.maximum_drawdown == pytest.approx(-.04)
    assert report.portfolio.annualized_volatility == pytest.approx(
        np.std([.04, .04, -.04], ddof=1) * np.sqrt(252)
    )


def test_cagr_uses_elapsed_calendar_time_not_row_count():
    idx = pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2025-12-31"])
    r = calc({"A": prices([100, 105, 110, 121], idx)}, {"A": 1})
    days = (idx[-1] - idx[0]).days
    assert r.portfolio.cagr == pytest.approx(1.21 ** (365.25 / days) - 1)


def test_annual_vol_and_sharpe_with_known_returns():
    history = prices([100, 110, 99, 108.9])  # +.1,-.1,+.1
    r = calc({"X": history}, {"X": 1}, annual_risk_free_rate=0.0)
    ex = np.array([.1, -.1, .1])
    assert r.portfolio.sharpe == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
    assert r.portfolio.sortino == pytest.approx(ex.mean() / np.sqrt(np.mean(np.minimum(ex, 0)**2)) * np.sqrt(252))


def test_zero_vol_renders_null_ratios_not_nan():
    p = prices([100, 100, 100, 100])
    r = calc({"FLAT": p}, {"FLAT": 1})
    assert r.portfolio.sharpe is None and r.portfolio.sortino is None
    assert r.portfolio.cagr == 0 and r.portfolio.maximum_drawdown == 0
    assert r.correlations["FLAT"]["FLAT"] is None
    result = json.dumps(performance_as_dict(r), allow_nan=False)
    assert '"sharpe": null' in result


def test_no_lookahead_and_strict_common_date_intersection():
    weekdays = pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11"])
    crypto = pd.to_datetime(["2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10", "2026-11-11"])
    r = calc({"ETF": prices([100, 101, 102, 103], weekdays),
              "BTC": prices([10, 30, 20, 15, 12, 14], crypto)})
    assert r.observation_dates == 4
    assert list(r.period_returns.index) == list(weekdays[1:])
    # BTC Friday->Monday return incorporates entire weekend, not fabricated zero-days.
    assert r.period_returns.iloc[0] == pytest.approx(.5 * .01 + .5 * .5)


def test_benchmark_tracking_error_beta_and_information_ratio():
    p = prices([100, 110, 110, 121])
    b = prices([200, 220, 220, 242])
    r = calc({"A": p}, {"A": 1}, benchmark_prices_in_base=b, benchmark_symbol="SPY")
    assert r.benchmark is not None
    assert r.benchmark.beta == pytest.approx(1.0)
    assert r.benchmark.tracking_error == pytest.approx(0.0)
    assert r.benchmark.information_ratio is None


def test_benchmark_comparison_uses_same_common_dates():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    benchmark_dates = dates.delete(2)
    p = prices([100, 110, 115, 116, 117], dates)
    b = prices([100, 105, 110, 112], benchmark_dates)
    r = calc({"A": p}, {"A": 1}, benchmark_prices_in_base=b, benchmark_symbol="SPY")
    assert r.observation_dates == 4
    assert r.first_date == "2025-01-01" and r.last_date == "2025-01-07"


def test_correlation_matrix_symmetric_for_nonzero_volatility():
    a = prices([100, 110, 99, 108.9])
    b = prices([50, 45, 49.5, 47.025])
    r = calc({"A": a, "B": b})
    assert r.correlations["A"]["B"] == pytest.approx(r.correlations["B"]["A"])
    assert r.correlations["A"]["A"] == pytest.approx(1)


@pytest.mark.parametrize("weights", [{"A": 1.1}, {"A": -1}, {"A": np.nan}, {"A": 0}])
def test_invalid_weights_rejected(weights):
    with pytest.raises(PerformanceError, match="Weights"):
        calc({"A": prices([1,2,3,4])}, weights)


def test_mismatched_symbol_lists_rejected():
    with pytest.raises(PerformanceError, match="match"):
        calc({"A": prices([1,2,3,4])}, {"B": 1})


def test_nan_and_invalid_asset_prices_rejected():
    with pytest.raises(PerformanceError, match="positive"):
        calc({"A": prices([100, np.nan, 102, 103])})
    with pytest.raises(PerformanceError, match="positive"):
        calc({"A": prices([100, 0, 102, 103])})


def test_short_intersection_rejected():
    with pytest.raises(PerformanceError, match="common close dates"):
        analyze_performance({"A": prices([1,2,3,4])}, {"A": 1}, min_observations=50)


def test_missing_benchmark_pair_rejected():
    with pytest.raises(PerformanceError, match="together"):
        calc({"A": prices([1,2,3,4])}, benchmark_symbol="SPY")


def test_negative_risk_free_invalid():
    with pytest.raises(PerformanceError, match="risk-free"):
        calc({"A": prices([1,2,3,4])}, annual_risk_free_rate=-1.0)


def test_results_have_consistent_index_return_period_count():
    r = calc({"A": prices([2,3,4,5])})
    assert len(r.portfolio_index) == len(r.period_returns) + 1
    assert r.portfolio_index.iloc[0] == 100.0
    assert r.observation_dates == r.return_periods + 1


def test_simple_independent_reference_for_max_drawdown():
    r = calc({"A": prices([100, 120, 90, 99, 80])})
    assert r.portfolio.maximum_drawdown == pytest.approx((80 / 120) - 1)


def test_compounding_does_not_sum_simple_returns():
    r = calc({"A": prices([100, 150, 100, 150])})
    assert r.portfolio.cumulative_return == pytest.approx(.5)
    assert r.portfolio.cumulative_return != pytest.approx(.5 - 1/3 + .5)
