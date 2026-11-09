"""V9.18 optimizer acceptance checks: math, constraints, leakage and benchmarks."""

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v8_insights import InsightError
from portfolio_risk.v83_robustness import _simulated_returns, _validated_rules
from portfolio_risk.v918_three_strategies import (
    _blocks,
    _fast_daily_returns,
    _fast_growth,
    _select,
    analyze_three_strategies,
)


def inputs(n=630, seed=515):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    factors = rng.normal(0, 0.010, n)
    # Deliberate stress case: 90% in unstable low-drift CCL analogue;
    # AAPL/MSFT have stronger historical growth, SPY moderate growth.
    frame = pd.DataFrame(
        {
            "CCL": -0.0006 + 2.8 * factors + rng.normal(0, 0.009, n),
            "AAPL": 0.0014 + 1.15 * factors + rng.normal(0, 0.006, n),
            "MSFT": 0.0011 + 1.03 * factors + rng.normal(0, 0.005, n),
            "SPY_HOLD": 0.0007 + 0.9 * factors + rng.normal(0, 0.004, n),
        },
        index=idx,
    )
    spy = pd.Series(0.0007 + 0.90 * factors + rng.normal(0, 0.004, n), index=idx)
    rsp = pd.Series(0.0006 + 0.80 * factors + rng.normal(0, 0.004, n), index=idx)
    ref = np.array([0.9, 0.0333, 0.0333, 0.0334])
    dates = pd.DatetimeIndex([idx[0] - pd.Timedelta(days=1), *idx])
    return frame, dates, ref, {"SPY": spy, "RSP": rsp}


def rules(max_weight=0.4):
    return {
        "max_weight": max_weight,
        "rebalance": "quarterly",
        "cost_bps": 10.0,
        "train_periods": 252,
        "test_periods": 126,
        "step_periods": 126,
    }


@pytest.mark.parametrize("method", ["daily", "monthly", "quarterly", "annually", "buy_hold"])
def test_fast_cagr_exactly_matches_legacy_self_financing_simulator(method):
    x, _, ref, _ = inputs(290)
    w = np.array([0.1, 0.35, 0.35, 0.2])
    actual, _ = _simulated_returns(x, w, rebalance=method, cost_bps=10.0, starting_weights=ref)
    fast = _fast_daily_returns(x, w, ref, 10, method)
    np.testing.assert_allclose(fast, actual.to_numpy(), atol=1e-12)
    annualized = np.expm1(np.log1p(actual).sum() * 252 / len(x))
    assert _fast_growth(_blocks(x, method), w, ref, 10, len(x)) == pytest.approx(
        annualized, abs=1e-11
    )


def test_growth_optimizer_beats_feasible_manual_alternative_on_training_data():
    x, _, ref, _ = inputs()
    settings = _validated_rules(list(x), ref, rules())
    result, _ = _select(x, ref, settings, 0.0, "growth")
    baseline = np.array([0.0, 0.4, 0.4, 0.2])
    got = _fast_growth(_blocks(x, "quarterly"), result, ref, 10, len(x))
    manual = _fast_growth(_blocks(x, "quarterly"), baseline, ref, 10, len(x))
    assert got >= manual - 0.001
    assert result[0] < 0.05, (
        "A weak concentrated holding should not dominate this constructed growth example"
    )


def test_balanced_and_defensive_are_distinct_goals():
    x, _, ref, _ = inputs()
    settings = _validated_rules(list(x), ref, rules())
    balanced, _ = _select(x, ref, settings, 0.0, "balanced")
    defensive, _ = _select(x, ref, settings, 0.0, "defensive")
    assert np.all(balanced <= 0.400001) and np.all(defensive <= 0.400001)
    b = _fast_daily_returns(x, balanced, ref, 10, "quarterly")
    d = _fast_daily_returns(x, defensive, ref, 10, "quarterly")
    assert d.std() <= b.std() + 1e-5
    assert np.isclose(balanced.sum(), 1) and np.isclose(defensive.sum(), 1)


def test_three_goals_share_aligned_dates_and_include_both_benchmarks():
    x, dates, ref, bench = inputs()
    out = analyze_three_strategies(x, dates, ref, bench, rules=rules())
    assert set(out["strategies"]) == {"growth", "balanced", "defensive"}
    for _key, row in out["strategies"].items():
        assert row["status"] == "evaluated"
        assert set(row["benchmarks_training_metrics"]) == {"SPY", "RSP"}
        assert sum(row["weights"].values()) == pytest.approx(1.0)
        assert row["holdout"]["tests"] >= 1
        assert set(row["holdout"]["benchmark_counts"]) == {"SPY", "RSP"}
        assert (
            row["holdout"]["better"] + row["holdout"]["worse"] + row["holdout"]["similar"]
            == row["holdout"]["tests"]
        )


def test_benchmark_misalignment_rejected():
    x, dates, ref, bench = inputs(400)
    bench["RSP"] = bench["RSP"].iloc[:-1]
    with pytest.raises(InsightError, match="SPY and RSP"):
        analyze_three_strategies(x, dates, ref, bench, rules=rules())


def test_later_period_never_changes_earlier_selected_weights():
    x, dates, ref, bench = inputs()
    a = analyze_three_strategies(x, dates, ref, bench, rules=rules())
    x2 = x.copy()
    x2.iloc[-126:] *= 3
    b = analyze_three_strategies(x2, dates, ref, bench, rules=rules())
    for goal in ("growth", "balanced", "defensive"):
        first = a["strategies"][goal]["holdout"]["folds"][0]["training_weights"]
        second = b["strategies"][goal]["holdout"]["folds"][0]["training_weights"]
        assert first == pytest.approx(second, abs=1e-8)


def test_old_fewer_than_one_fold_report_does_not_fake_success():
    x, dates, ref, bench = inputs(320)
    out = analyze_three_strategies(x, dates, ref, bench, rules=rules())
    assert all(item["holdout"]["status"] == "insufficient" for item in out["strategies"].values())


def test_no_network_insights_adapter_emits_all_three_optimized_paths():
    from portfolio_risk.v8_insights import analyze_insights

    frame, dates, ref, benchmarks = inputs(525)
    # Test the same public analysis path used by the CLI; synthetic closes.
    prices = {
        symbol: pd.Series(np.r_[100.0, 100 * np.cumprod(1 + frame[symbol].to_numpy())], index=dates)
        for symbol in frame
    }
    bench_prices = {
        symbol: pd.Series(np.r_[100.0, 100 * np.cumprod(1 + series.to_numpy())], index=dates)
        for symbol, series in benchmarks.items()
    }
    result = analyze_insights(
        prices,
        dict(zip(frame.columns, ref, strict=False)),
        bench_prices,
        base_currency="USD",
        three_strategies=True,
        robustness=rules(),
        mc_paths=20,
        mc_horizon=30,
    )
    assert result["three_strategy_analysis"]["status"] == "evaluated"
    assert set(result["three_strategy_analysis"]["strategies"]) == {
        "growth",
        "balanced",
        "defensive",
    }
    assert "rolling_robustness" not in result
    assert set(result["benchmarks"]) == {"SPY", "RSP"}
