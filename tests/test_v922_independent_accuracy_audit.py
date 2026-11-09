"""Independent accounting/math audits for the three-goal strategy engine.

No web, provider responses, or calls to the production simulator in the
independent expected-value calculations. This is NOT a live-market price audit.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v8_insights import _stats
from portfolio_risk.v83_robustness import _simulated_returns, _validated_rules
from portfolio_risk.v918_three_strategies import (
    _blocks,
    _fast_daily_returns,
    _fast_growth,
    _select,
    analyze_three_strategies,
    ending_drift_weights,
)


def _fixture(days=420):
    dates = pd.bdate_range("2023-11-15", periods=days + 1)
    t = np.arange(days)
    a = 0.0008 + 0.008 * np.sin(t / 17) + 0.004 * np.cos(t / 9)
    b = 0.0002 + 0.007 * np.cos(t / 11) - 0.003 * np.sin(t / 15)
    c = 0.0004 + 0.003 * np.sin(t / 29)
    returns = pd.DataFrame({"A": a, "B": b, "C": c}, index=dates[1:])
    return dates, returns


def _accounting_by_shares(frame, target, starting, schedule, cost_bps):
    """Independent simulator: prices, shares, wealth and transaction costs."""
    price = np.full(len(target), 100.0)
    nav = 1.0
    shares = starting / price
    daily = []
    previous_period = None
    for row_number, (day, row) in enumerate(frame.iterrows()):
        if schedule == "quarterly":
            bucket = (day.year, (day.month - 1) // 3)
        elif schedule == "monthly":
            bucket = (day.year, day.month)
        elif schedule == "annually":
            bucket = day.year
        else:
            bucket = None
        rebalance = (
            row_number == 0
            or schedule == "daily"
            or (schedule != "buy_hold" and bucket != previous_period)
        )
        before = nav
        if rebalance:
            current_shares = shares * price / nav
            traded = np.abs(current_shares - target).sum() / 2
            nav *= 1 - traded * cost_bps / 10000
            shares = nav * target / price
        price = price * (1 + row.to_numpy())
        nav = float((shares * price).sum())
        daily.append(nav / before - 1)
        previous_period = bucket
    return np.array(daily), shares * price / nav


@pytest.mark.parametrize("mode", ["daily", "monthly", "quarterly", "annually", "buy_hold"])
@pytest.mark.parametrize("cost_bps", [0.0, 37.0])
def test_independent_share_accounting_agrees_with_simulator(mode, cost_bps):
    dates, frame = _fixture(420)
    original = np.array([0.70, 0.20, 0.10])
    target = np.array([0.10, 0.25, 0.65])
    expected, last_weights = _accounting_by_shares(frame, target, original, mode, cost_bps)
    actual, _ = _simulated_returns(
        frame, target, rebalance=mode, cost_bps=cost_bps, starting_weights=original
    )
    speedy = _fast_daily_returns(frame, target, original, cost_bps, mode)
    assert actual.to_numpy() == pytest.approx(expected, abs=5e-14)
    assert speedy == pytest.approx(expected, abs=5e-14)
    assert ending_drift_weights(frame, target, mode) == pytest.approx(
        _accounting_by_shares(frame, target, target, mode, 0.0)[1], abs=5e-14
    )
    if mode in ("buy_hold", "monthly", "quarterly", "annually", "daily"):
        growth = _fast_growth(_blocks(frame, mode), target, original, cost_bps, len(frame))
        expected_growth = np.prod(1 + expected) ** (252 / len(expected)) - 1
        assert growth == pytest.approx(expected_growth, abs=1e-12)


def test_hand_computed_two_asset_rebalancing_and_fee():
    dates = pd.to_datetime(["2024-03-28", "2024-03-29", "2024-04-01"])
    frame = pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.10]}, index=dates[1:])
    # Day 1: 50/50 -> 5% gain, day 2: same weights in new quarter -> 5% gain.
    result, _ = _accounting_by_shares(
        frame, np.array([0.5, 0.5]), np.array([0.5, 0.5]), "quarterly", 0.0
    )
    assert result == pytest.approx([0.05, 0.05], abs=1e-14)
    # In buy-and-hold, winning asset drifts: .5/1.05 = 47.619% in B.
    hold, _ = _accounting_by_shares(
        frame, np.array([0.5, 0.5]), np.array([0.5, 0.5]), "buy_hold", 0.0
    )
    assert hold == pytest.approx([0.05, 0.05 / 1.05], abs=1e-14)


def test_independent_metrics_cagr_volatility_sharpe_drawdown_and_es():
    dates, frame = _fixture(420)
    stream, _ = _accounting_by_shares(
        frame, np.array([0.3, 0.2, 0.5]), np.array([0.7, 0.2, 0.1]), "quarterly", 21.0
    )
    rf = 0.04
    result = _stats(pd.Series(stream, index=frame.index), dates, rf)
    nav = np.r_[1.0, np.cumprod(1 + stream)]
    elapsed = (dates[-1] - dates[0]).days
    expected_cagr = nav[-1] ** (365.25 / elapsed) - 1
    daily_rf = (1 + rf) ** (1 / 252) - 1
    expected_sharpe = (stream.mean() - daily_rf) / stream.std(ddof=1) * sqrt(252)
    drawdown = 1 - float(np.min(nav / np.maximum.accumulate(nav)))
    loss = -stream
    es = max(0.0, float(loss[loss >= np.quantile(loss, 0.95)].mean()))
    assert result["cagr"] == pytest.approx(expected_cagr, abs=1e-12)
    assert result["annualized_volatility"] == pytest.approx(
        stream.std(ddof=1) * sqrt(252), abs=1e-12
    )
    assert result["sharpe"] == pytest.approx(expected_sharpe, abs=1e-12)
    assert result["drawdown"] == pytest.approx(drawdown, abs=1e-12)
    assert result["es95"] == pytest.approx(es, abs=1e-12)


def test_optimizer_is_not_worse_than_feasible_manual_candidates_for_each_goal():
    dates, frame = _fixture(300)
    reference = np.array([0.65, 0.25, 0.10])
    rules = _validated_rules(
        list(frame.columns),
        reference,
        dict(
            max_weight=1.0,
            rebalance="quarterly",
            cost_bps=15.0,
            train_periods=150,
            test_periods=60,
            step_periods=60,
        ),
    )
    manual = [
        reference,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.ones(3) / 3,
        np.array([0.5, 0.3, 0.2]),
    ]
    for goal in ("growth", "balanced", "defensive"):
        chosen, _ = _select(frame, reference, rules, 0.03, goal)
        assert chosen is not None
        assert np.all(chosen >= -1e-7) and chosen.sum() == pytest.approx(1.0)

        def score(w, goal=goal):
            stream, _ = _accounting_by_shares(frame, w, reference, "quarterly", 15.0)
            if goal == "growth":
                return np.prod(1 + stream) ** (252 / len(stream)) - 1
            if goal == "balanced":
                return (
                    (stream.mean() - ((1 + 0.03) ** (1 / 252) - 1)) / stream.std(ddof=1) * sqrt(252)
                )
            return -stream.std(ddof=1) * sqrt(252)

        # SLSQP is local; this fixture explicitly requires improvement over
        # simple valid allocations, not a general guarantee of global optimum.
        assert score(chosen) >= max(map(score, manual)) - 1e-5, goal


def test_benchmarks_matched_and_no_holdout_leakage_in_fold_weight_selection():
    dates, frame = _fixture(440)
    spy = pd.Series(0.00035 + 0.004 * np.sin(np.arange(len(frame)) / 5), index=frame.index)
    rsp = pd.Series(0.00025 + 0.003 * np.cos(np.arange(len(frame)) / 7), index=frame.index)
    benchmark = {"SPY": spy, "RSP": rsp}
    baseline = np.array([0.75, 0.15, 0.10])
    rules = dict(
        max_weight=1.0,
        rebalance="quarterly",
        cost_bps=10.0,
        train_periods=150,
        test_periods=60,
        step_periods=60,
    )
    report = analyze_three_strategies(frame, dates, baseline, benchmark, rules=rules, rf=0.02)
    assert set(report["strategies"]) == {"growth", "balanced", "defensive"}
    for goal, item in report["strategies"].items():
        assert item["status"] == "evaluated"
        assert set(item["benchmarks_training_metrics"]) == {"SPY", "RSP"}
        assert item["benchmarks_training_metrics"]["SPY"]["cagr"] == pytest.approx(
            _stats(spy, dates, 0.02)["cagr"]
        )
        assert item["benchmarks_training_metrics"]["RSP"]["sharpe"] == pytest.approx(
            _stats(rsp, dates, 0.02)["sharpe"]
        )
        for fold in item["holdout"]["folds"]:
            assert fold["test_start"] == fold["train_end"]
            assert fold["test_end"] > fold["train_end"]
            assert sum(fold["training_weights"].values()) == pytest.approx(1.0)
            assert set(fold["benchmarks"]) == {"SPY", "RSP"}
            assert fold["primary_better"] == (fold["primary_improvement"] > 1e-9)
            end_index = int(np.searchsorted(dates.values, np.datetime64(fold["train_end"])))
            # Re-run training choice with last 60 days entirely removed.
            constraints = _validated_rules(list(frame), baseline, rules)
            weights, _ = _select(frame.iloc[:end_index], baseline, constraints, 0.02, goal)
            assert np.array(list(fold["training_weights"].values())) == pytest.approx(
                weights, abs=1e-9
            )
    moved = frame.copy()
    moved.iloc[-60:] *= -1.2
    altered = analyze_three_strategies(moved, dates, baseline, benchmark, rules=rules, rf=0.02)
    # Earliest fold cannot depend on future rows.
    for key in ("growth", "balanced", "defensive"):
        assert report["strategies"][key]["holdout"]["folds"][0][
            "training_weights"
        ] == pytest.approx(
            altered["strategies"][key]["holdout"]["folds"][0]["training_weights"], abs=1e-10
        )


def test_nonzero_risk_free_balanced_training_objective_matches_displayed_sharpe():
    """A key audit fix: objective and reported Sharpe use identical rf math."""
    dates, frame = _fixture(235)
    original = np.array([0.7, 0.2, 0.1])
    rules = _validated_rules(
        list(frame),
        original,
        dict(
            max_weight=1.0,
            rebalance="quarterly",
            cost_bps=21.0,
            train_periods=150,
            test_periods=60,
            step_periods=60,
        ),
    )
    weights, diagnostics = _select(frame, original, rules, 0.055, "balanced")
    stream, _ = _accounting_by_shares(frame, weights, original, "quarterly", 21.0)
    expected = (stream.mean() - ((1 + 0.055) ** (1 / 252) - 1)) / stream.std(ddof=1) * sqrt(252)
    reported = _stats(pd.Series(stream, index=frame.index), dates, 0.055)["sharpe"]
    assert reported == pytest.approx(expected, abs=1e-10)
    assert diagnostics["training_objective"] == pytest.approx(-reported, abs=1e-10)
