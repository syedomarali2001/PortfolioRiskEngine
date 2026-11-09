"""V4 numerical and calendar acceptance tests; no market-data network calls."""
from __future__ import annotations

from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.backtest import (
    BacktestError, SCHEDULES, _is_period_end, _rebalance_at_close,
    backtest_as_dict, run_backtest,
)


def prices(a, b, dates=None):
    if dates is None:
        dates = pd.date_range("2025-01-01", periods=len(a), freq="B")
    return {"A": pd.Series(a, index=dates, dtype=float),
            "B": pd.Series(b, index=dates, dtype=float)}


def simulate(a, b, **kwargs):
    return run_backtest(prices(a, b), {"A": .5, "B": .5}, min_observations=3,
                        schedules=("buy_hold",), **kwargs)


def test_all_five_schedules_present():
    p = prices([100, 110, 115, 100], [100, 100, 100, 100])
    result = run_backtest(p, {"A": .5, "B": .5}, min_observations=3)
    assert list(result.strategies) == list(SCHEDULES)


def test_buy_hold_exact_wealth_and_units():
    report = simulate([100, 200, 100], [100, 100, 100], initial_capital=1000)
    hold = report.strategies["buy_hold"]
    assert hold.values.to_list() == pytest.approx([1000, 1500, 1000])
    assert hold.metrics.cumulative_return == pytest.approx(0)
    assert hold.rebalance_count == 0
    assert hold.weights_over_time.loc[pd.Timestamp("2025-01-03"), "A"] == pytest.approx(.5)
    assert hold.trade_count == 2
    assert hold.ending_weights == pytest.approx({"A": .5, "B": .5})
    assert hold.total_fees == 0


def test_buy_hold_weights_drift():
    report = simulate([100, 200, 300], [100, 100, 100], initial_capital=1000)
    hold = report.strategies["buy_hold"]
    assert hold.ending_value == pytest.approx(2000)
    assert hold.weights_over_time.loc[pd.Timestamp("2025-01-03"), "A"] == pytest.approx(.75)
    assert hold.ending_weights["A"] == pytest.approx(.75)
    assert hold.ending_weights["B"] == pytest.approx(.25)
    assert hold.maximum_weight_drift == pytest.approx(.25)


def test_month_end_trade_uses_last_common_close_and_affects_future_only():
    ds = pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-03"])
    p = prices([100, 200, 100], [100, 100, 100], dates=ds)
    result = run_backtest(p, {"A": .5, "B": .5}, initial_capital=1000,
                          min_observations=3, schedules=("buy_hold", "monthly"))
    hold, month = result.strategies["buy_hold"], result.strategies["monthly"]
    assert hold.ending_value == pytest.approx(1000)
    assert month.ending_value == pytest.approx(1125)
    assert month.values.iloc[1] == pytest.approx(1500)  # pre-fee value at close
    assert month.rebalance_count == 1
    trades = [t for t in month.trades if t.reason == "rebalance"]
    assert {t.date for t in trades} == {"2025-01-31"}
    assert sum(t.notional for t in trades) == pytest.approx(0)
    assert {t.symbol for t in trades} == {"A", "B"}


@pytest.mark.parametrize("schedule,trigger", [
    ("monthly", "2025-01-31"),
    ("quarterly", "2025-03-31"),
    ("semiannual", "2025-06-30"),
    ("annual", "2025-12-31"),
])
def test_calendar_period_ends(schedule, trigger):
    idx = pd.date_range("2025-01-02", "2026-02-10", freq="B")
    ix = idx.get_loc(pd.Timestamp(trigger))
    assert _is_period_end(idx, ix, schedule)
    assert not _is_period_end(idx, 0, schedule)
    assert not _is_period_end(idx, len(idx)-1, schedule)


def test_quarterly_not_monthly():
    idx = pd.to_datetime(["2025-01-29", "2025-01-31", "2025-02-03"])
    assert _is_period_end(idx, 1, "monthly")
    assert not _is_period_end(idx, 1, "quarterly")


def test_end_of_data_never_costs_rebalance():
    idx = pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-03"])
    assert not _is_period_end(idx, 2, "monthly")


def test_initial_purchase_fees_reduce_end_value_with_flat_prices():
    rep = simulate([100, 100, 100], [100, 100, 100],
                   fee_bps=100, initial_capital=1000)
    res = rep.strategies["buy_hold"]
    invested = 1000 / 1.01
    assert res.values.to_list() == pytest.approx([1000, invested, invested])
    assert res.ending_value == pytest.approx(invested)
    assert res.initial_fees == pytest.approx(1000 - invested)
    assert res.metrics.cumulative_return == pytest.approx(invested / 1000 - 1)
    assert sum(t.fee for t in res.trades) == pytest.approx(res.total_fees)
    assert sum(abs(t.notional) for t in res.trades) == pytest.approx(res.traded_notional)


def test_zero_cost_rebalancing_matches_target_wealth():
    pos = np.array([5., 5.])
    price = np.array([200., 100.])
    w = np.array([.5, .5])
    q, fee, trades = _rebalance_at_close(pos, price, w, 0.)
    assert q * price == pytest.approx([750., 750.])
    assert sum(trades) == pytest.approx(0)
    assert fee == 0


@pytest.mark.parametrize("bps", [1., 10., 100., 1000.])
def test_fee_aware_rebalance_preserves_cash_and_target_weights(bps):
    pos = np.array([5., 5.])
    px = np.array([200., 100.])
    w = np.array([.5, .5])
    f = bps / 10000
    q, fee, tr = _rebalance_at_close(pos, px, w, f)
    new_wealth = float((q * px).sum())
    assert new_wealth + fee == pytest.approx(1500, abs=1e-8)
    assert fee == pytest.approx(f * sum(abs(tr)), abs=1e-8)
    assert q * px / new_wealth == pytest.approx(w)
    assert (q >= 0).all()


def test_positive_costs_do_not_improve_flat_price_strategy():
    idx = pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-03"])
    p = prices([100, 100, 100], [100, 100, 100], idx)
    report = run_backtest(p, {"A": .5, "B": .5}, initial_capital=1000,
                          fee_bps=50, min_observations=3,
                          schedules=("buy_hold", "monthly"))
    for result in report.strategies.values():
        assert result.ending_value < 1000
        assert result.total_fees > 0


def test_fee_increase_reduces_final_wealth_on_same_path():
    idx = pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-03"])
    data = prices([100, 200, 100], [100, 100, 100], idx)
    common = dict(initial_capital=1000, min_observations=3, schedules=("monthly",))
    x = run_backtest(data, {"A": .5, "B": .5}, fee_bps=0, **common)
    y = run_backtest(data, {"A": .5, "B": .5}, fee_bps=50, **common)
    assert y.strategies["monthly"].ending_value < x.strategies["monthly"].ending_value


def test_benchmark_is_aligned_fee_free_and_not_rebalanced():
    idx = pd.date_range("2025-01-06", periods=5, freq="B")
    p = prices([100]*5, [100]*5, idx)
    bench = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)
    result = run_backtest(p, {"A": .5, "B": .5}, initial_capital=1000,
                          fee_bps=20, min_observations=3,
                          benchmark_prices_in_base=bench, benchmark_symbol="SPY")
    assert result.benchmark_values.iloc[-1] == pytest.approx(1040)
    assert result.benchmark.cumulative_return == pytest.approx(.04)
    assert result.strategies["buy_hold"].ending_value < 1000
    assert "fee-free" in " ".join(result.warnings)


def test_strict_common_date_intersection_discard_weekends():
    idx = pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"])
    p = {"BTC": pd.Series([100, 110, 120, 130], index=idx, dtype=float),
         "ETF": pd.Series([100, 110], index=idx[[0, 3]], dtype=float)}
    # Two dates aren't enough for Sharpe calculations, so confirm the strict
    # sample is rejected instead of inventing weekend returns.
    with pytest.raises(BacktestError, match="shared adjusted-close dates"):
        run_backtest(p, {"BTC": .5, "ETF": .5}, min_observations=3)


@pytest.mark.parametrize("bad_weights", [
    {"A": .5, "B": .6}, {"A": -0.5, "B": 1.5}, {"A": 0., "B": 1.},
    {"A": float('nan'), "B": 1.}, {"A": .5},
])
def test_invalid_weights_rejected(bad_weights):
    with pytest.raises(BacktestError):
        run_backtest(prices([100]*4, [100]*4), bad_weights, min_observations=3)


@pytest.mark.parametrize("config", [
    {"fee_bps": -1}, {"fee_bps": 10001}, {"fee_bps": float("nan")},
    {"initial_capital": -1}, {"initial_capital": float("inf")},
    {"annual_risk_free_rate": -1.5}, {"base_currency": "EUR"},
    {"schedules": ("weekly",)}, {"schedules": ()},
    {"schedules": ("monthly", "monthly")},
])
def test_invalid_settings_rejected(config):
    with pytest.raises(BacktestError):
        run_backtest(prices([100]*4, [100]*4), {"A": .5, "B": .5},
                     min_observations=3, **config)


def test_json_output_is_serializable_with_trade_log():
    report = simulate([100, 110, 90], [100, 100, 100], initial_capital=1000, fee_bps=50)
    payload = backtest_as_dict(report)
    encoded = json.loads(json.dumps(payload, allow_nan=False))
    assert encoded["fee_bps"] == 50
    assert len(encoded["strategies"]["buy_hold"]["trades"]) == 2
    assert len(encoded["strategies"]["buy_hold"]["portfolio_values"]) == 3
    assert len(encoded["strategies"]["buy_hold"]["period_returns"]) == 2
    assert len(encoded["strategies"]["buy_hold"]["weights_over_time"]) == 3
    assert "initial_fees" in encoded["strategies"]["buy_hold"]


def test_strict_price_validation():
    with pytest.raises(BacktestError, match="positive"):
        simulate([100, 0, 100], [100, 100, 100])


def test_unaligned_currency_is_never_inferred_by_pure_engine():
    # Service layer normalizes FX. Pure engine makes no claims about quoted
    # currencies and receives only common base-currency series.
    assert run_backtest(prices([100, 110, 120], [100, 101, 102]),
                        {"A": .5, "B": .5}, min_observations=3).base_currency == "CAD"
