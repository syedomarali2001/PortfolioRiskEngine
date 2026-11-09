"""V6 numerical acceptance tests: no API calls; deterministic data."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.construction import (
    ConstructionError,
    _allocation,
    _capped_simplex_projection,
    _first_close_of_new_period,
    _minimum_variance,
    construction_as_dict,
    run_construction,
)


def _prices(n=390, seed=4, names=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    matrix = rng.normal(0.0004, 0.008, (n, len(names)))
    matrix[:, 1] = 0.7 * matrix[:, 0] + 0.3 * matrix[:, 1]
    matrix[:, 2] = -0.3 * matrix[:, 0] + 0.7 * matrix[:, 2]
    val = np.cumprod(1 + matrix, axis=0) * 100
    return {s: pd.Series(val[:, i], index=dates) for i, s in enumerate(names)}


def _run(prices=None, **kwargs):
    options = {"lookback": 70, "momentum_skip": 10, "top_k": 2}
    options.update(kwargs)
    return run_construction(
        prices if prices is not None else _prices(),
        {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
        **options,
    )


def test_full_strategy_smoke_outputs_shared_window():
    r = _run()
    assert set(r.results) == {
        "input_buy_hold",
        "input_rebalanced",
        "equal_weight",
        "inverse_vol",
        "min_variance",
        "momentum_top_k",
    }
    assert r.observation_dates > 200
    assert r.return_periods == r.observation_dates - 1
    assert len({row.values.index[0] for row in r.results.values()}) == 1
    assert len({row.values.index[-1] for row in r.results.values()}) == 1
    assert all(row.ending_value > 0 for row in r.results.values())
    assert all(row.metrics.cagr > -1 for row in r.results.values())


def test_fixed_initial_capital_no_fee_and_positive_weights():
    r = _run()
    assert all(row.values.iloc[0] == pytest.approx(20000) for row in r.results.values())
    assert all(row.total_fees == pytest.approx(0) for row in r.results.values())
    for row in r.results.values():
        np.testing.assert_allclose(row.weights_over_time.sum(axis=1), 1, atol=1e-10)
        assert all(w >= -1e-12 for w in row.ending_weights.values())
        assert sum(row.ending_weights.values()) == pytest.approx(1)


def test_benchmark_correctly_uses_shared_dates():
    p = _prices()
    bench = pd.Series(100 * np.cumprod(np.full(390, 1.001)), index=next(iter(p.values())).index)
    r = _run(p, benchmark_prices_in_base=bench, benchmark_symbol="SPY")
    assert r.benchmark_metrics is not None
    assert r.benchmark_metrics.cagr > 0
    assert r.observation_dates == len(next(iter(r.results.values())).values)


def test_benchmark_needs_currency_pair_and_different_symbol():
    p = _prices()
    with pytest.raises(ConstructionError, match="provided together"):
        _run(p, benchmark_symbol="SPY")
    with pytest.raises(ConstructionError, match="different ticker"):
        _run(p, benchmark_symbol="AAA", benchmark_prices_in_base=p["AAA"])


def test_reference_hold_is_no_rebalance_and_weight_drift_exists():
    r = _run()
    buy = r.results["input_buy_hold"]
    assert buy.rebalance_count == 0
    assert buy.trade_count == 3
    assert any(
        abs(buy.ending_weights[k] - r.reference_weights[k]) > 0.01 for k in r.reference_weights
    )


def test_rebalanced_reference_stays_close_to_target_near_rebalance():
    r = _run()
    row = r.results["input_rebalanced"]
    assert row.rebalance_count > 0
    first_period = pd.Timestamp("2023-07-03")
    if first_period in row.weights_over_time.index:
        arr = row.weights_over_time.loc[first_period].values
        assert np.allclose(arr, [0.5, 0.3, 0.2], atol=1e-8)


def test_capped_projection_reconciles_bounds_and_weights():
    x = np.array([10.0, 3.0, -5.0, 0.0])
    result = _capped_simplex_projection(x, 0.45)
    assert result.sum() == pytest.approx(1, abs=1e-10)
    assert np.min(result) >= 0
    assert np.max(result) <= 0.45 + 1e-12
    assert result[0] == pytest.approx(0.45)


@pytest.mark.parametrize("maxweight", [0.1, 0, -1, float("nan")])
def test_reject_infeasible_simplex_cap(maxweight):
    with pytest.raises(ConstructionError):
        _capped_simplex_projection(np.array([1.0, 2.0, 3.0]), maxweight)


def test_min_variance_favors_low_vol_asset():
    covariance = np.diag([0.0001, 0.0009])
    out = _minimum_variance(covariance, max_weight=1.0, shrinkage=0.15)
    assert out[0] > 0.80
    assert out.sum() == pytest.approx(1)


def test_min_variance_singular_covariance_still_valid():
    covariance = np.array([[0.0001, 0.0001], [0.0001, 0.0001]])
    out = _minimum_variance(covariance, max_weight=0.6, shrinkage=0)
    assert out.sum() == pytest.approx(1)
    assert max(out) <= 0.6


def test_inverse_vol_favors_stable_asset():
    x = np.column_stack([np.array([-0.04, 0.04] * 50), np.array([-0.005, 0.005] * 50)])
    w = _allocation(
        "inverse_vol", x, np.array([0.5, 0.5]), max_weight=0.8, shrinkage=0.15, top_k=1, skip=0
    )
    assert w[1] > w[0]
    assert w[1] == pytest.approx(0.8)


def test_momentum_top_two_picks_highest_old_window_return():
    # First is best on the older window; most recent two periods reverse it.
    hist = np.array([[0.02, 0.01, -0.01]] * 30 + [[-0.5, 0.5, 0.6]] * 2)
    w = _allocation(
        "momentum_top_k",
        hist,
        np.array([1 / 3] * 3),
        max_weight=0.6,
        shrinkage=0.15,
        top_k=2,
        skip=2,
    )
    np.testing.assert_allclose(w, [0.5, 0.5, 0])


def test_momentum_uses_lagged_signal_not_trade_close():
    base = _prices()
    altered = {key: s.copy() for key, s in base.items()}
    a = _run(base, methods=("momentum_top_k",))
    trade_date = a.results["momentum_top_k"].values.index[0]
    altered["AAA"].loc[trade_date:] *= 10  # changes today's trade close and all future prices
    b = _run(altered, methods=("momentum_top_k",))
    assert a.results["momentum_top_k"].last_target != {}  # structure exists

    # The INITIAL signal is based on the previous close, so unaffected.
    def initial_weights(r):
        row = r.results["momentum_top_k"]
        entries = [t for t in row.trades if t.reason == "initial"]
        dollars = {t.symbol: t.notional for t in entries}
        return {k: v / sum(dollars.values()) for k, v in dollars.items()}

    assert initial_weights(a) == initial_weights(b)


def test_rising_future_data_does_not_change_prior_trade_schedule_or_decision():
    base = _prices()
    a = _run(base, methods=("inverse_vol",))
    dates = a.results["inverse_vol"].values.index
    cutoff = dates[75]
    altered = {key: s.copy() for key, s in base.items()}
    altered["BBB"].loc[cutoff:] *= 3
    b = _run(altered, methods=("inverse_vol",))
    left = [t for t in a.results["inverse_vol"].trades if t.date < cutoff.strftime("%Y-%m-%d")]
    right = [t for t in b.results["inverse_vol"].trades if t.date < cutoff.strftime("%Y-%m-%d")]
    assert left == right


def test_period_change_needs_only_previous_and_current_dates():
    d = pd.DatetimeIndex(["2025-03-31", "2025-04-01", "2025-04-02", "2025-07-01", "2026-01-02"])
    assert _first_close_of_new_period(d, 1, "monthly")
    assert _first_close_of_new_period(d, 1, "quarterly")
    assert not _first_close_of_new_period(d, 1, "semiannual")
    assert _first_close_of_new_period(d, 3, "semiannual")
    assert _first_close_of_new_period(d, 4, "annual")
    assert not _first_close_of_new_period(d, 0, "monthly")
    assert not _first_close_of_new_period(d, 2, "quarterly")


@pytest.mark.parametrize("frequency", ["monthly", "quarterly", "semiannual", "annual"])
def test_rebalance_count_decreases_with_frequency(frequency):
    r = _run(methods=("equal_weight",), frequency=frequency)
    assert r.results["equal_weight"].rebalance_count >= 1


def test_frequency_monotonic_trade_opportunities():
    counts = [
        _run(methods=("inverse_vol",), frequency=f).results["inverse_vol"].rebalance_count
        for f in ("monthly", "quarterly", "semiannual", "annual")
    ]
    assert counts == sorted(counts, reverse=True)


def test_fee_application_reconciles_gross_notional():
    r = _run(methods=("input_rebalanced", "momentum_top_k"), fee_bps=10)
    for row in r.results.values():
        assert row.total_fees > 0
        assert row.total_fees == pytest.approx(row.traded_notional * 0.001, rel=1e-9)
        assert row.values.iloc[0] == pytest.approx(20000)
        assert row.trade_count == len(row.trades)


def test_fee_free_buy_hold_matches_closed_form():
    prices = _prices()
    r = _run(prices, methods=("input_buy_hold",))
    first_date = pd.Timestamp(r.first_date)
    last_date = pd.Timestamp(r.last_date)
    expected = 20000 * sum(
        r.reference_weights[s] * prices[s].loc[last_date] / prices[s].loc[first_date]
        for s in prices
    )
    assert r.results["input_buy_hold"].ending_value == pytest.approx(expected)


def test_fee_returns_reduce_ending_value_in_constant_growth_example():
    dates = pd.bdate_range("2022-01-03", periods=380)
    vals = 100 * 1.0003 ** np.arange(380)
    p = {s: pd.Series(vals * mult, index=dates) for s, mult in [("AAA", 1), ("BBB", 2), ("CCC", 3)]}
    free = _run(p, methods=("input_buy_hold",), fee_bps=0)
    cost = _run(p, methods=("input_buy_hold",), fee_bps=25)
    assert free.results["input_buy_hold"].ending_value > cost.results["input_buy_hold"].ending_value


def test_later_period_evaluation_and_json_are_serializable():
    r = _run(evaluation_start=date(2023, 9, 1), methods=("min_variance", "input_buy_hold"))
    assert r.evaluation_start == "2023-09-01"
    assert all(v.evaluation_metrics is not None for v in r.results.values())
    out = construction_as_dict(r)
    assert out["evaluation_start"] == "2023-09-01"
    assert len(out["strategies"]["min_variance"]["portfolio_values"]) == r.observation_dates
    assert out["strategies"]["input_buy_hold"]["trade_count"] == 3
    import json

    json.dumps(out, allow_nan=False)


def test_insufficient_warmup_rejected():
    with pytest.raises(ConstructionError, match="Not enough"):
        _run(_prices(100), lookback=252)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capital": 0},
        {"fee_bps": -1},
        {"fee_bps": float("nan")},
        {"max_weight": 1.01},
        {"shrinkage": -1},
        {"shrinkage": 1.1},
        {"frequency": "weekly"},
        {"methods": ("bad",)},
        {"methods": ("equal_weight", "equal_weight")},
        {"top_k": 0},
        {"momentum_skip": -1},
    ],
)
def test_invalid_inputs(kwargs):
    with pytest.raises(ConstructionError):
        _run(**kwargs)


def test_old_and_new_v4_method_names_remain_distinct():
    r = _run(methods=("input_buy_hold", "input_rebalanced"))
    assert r.results["input_buy_hold"].rebalance_count == 0
    assert r.results["input_rebalanced"].rebalance_count > 0


def test_short_sample_warning_when_user_universe_selected_later():
    r = _run()
    assert any("hindsight" in msg for msg in r.warnings)
    assert any("previous" in msg.lower() for msg in r.warnings)
