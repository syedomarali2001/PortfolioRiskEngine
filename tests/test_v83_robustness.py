"""V8.3 offline tests: constraints, chronological isolation, costs, actionable evidence."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v8_insights import InsightError, analyze_insights
from portfolio_risk.v83_actions import portfolio_actions
from portfolio_risk.v83_exposure_actions import exposure_actions
from portfolio_risk.v83_robustness import (
    _box_project,
    _project_candidate,
    _simulated_returns,
    _validated_rules,
    analyze_robustness,
)

NAMES = ["AAPL", "MSFT", "SPY", "GLD"]
REFERENCE = np.array([0.35, 0.2, 0.3, 0.15])


def data(periods=820, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=periods + 1, freq="B")
    x = rng.normal(0, 0.012, (periods, 4)) + 0.00045
    frame = pd.DataFrame(x, index=idx[1:], columns=NAMES)
    return frame, idx


def rules(**kw):
    result = {
        "max_weight": 0.7,
        "train_periods": 252,
        "test_periods": 126,
        "step_periods": 126,
        "rebalance": "monthly",
        "cost_bps": 10.0,
    }
    result.update(kw)
    return result


def test_projection_respects_exact_sum_lower_upper():
    w = _box_project(
        np.array([0.8, 0.2, 0.0]), np.array([0.0, 0.15, 0.1]), np.array([0.45, 0.6, 0.7]), 1.0
    )
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w >= np.array([0.0, 0.15, 0.1]) - 1e-10).all()
    assert (w <= np.array([0.45, 0.6, 0.7]) + 1e-10).all()


def test_projection_refuses_infeasible_bounds():
    with pytest.raises(InsightError, match="infeasible"):
        _box_project(np.array([0.4, 0.6]), np.array([0.6, 0.6]), np.array([1.0, 1.0]), 1.0)


def test_compound_asset_floor_caps_and_group_cap():
    params = rules(
        asset_caps={"AAPL": 0.2, "GLD": 0.15},
        asset_floors={"SPY": 0.2},
        group_caps={"stocks": {"symbols": ["AAPL", "MSFT"], "cap": 0.3}},
    )
    constraints = _validated_rules(NAMES, REFERENCE, params)
    result = _project_candidate(REFERENCE, REFERENCE, constraints)
    assert result is not None
    assert result.sum() == pytest.approx(1.0)
    assert result[0] <= 0.2 + 1e-8
    assert result[3] <= 0.15 + 1e-8
    assert result[2] >= 0.2 - 1e-8
    assert result[0] + result[1] <= 0.3 + 1e-8


@pytest.mark.parametrize(
    "limits",
    [
        {"asset_caps": {"UNKNOWN": 0.2}},
        {"asset_caps": {"AAPL": 0.1}, "asset_floors": {"AAPL": 0.2}},
        {"asset_caps": {s: 0.1 for s in NAMES}},
        {"asset_floors": {s: 0.4 for s in NAMES}},
        {
            "group_caps": {
                "g1": {"symbols": ["SPY", "AAPL"], "cap": 0.4},
                "g2": {"symbols": ["AAPL", "GLD"], "cap": 0.4},
            }
        },
        {"group_caps": {"g1": {"symbols": ["UNKNOWN"], "cap": 0.4}}},
        {"cost_bps": -1},
        {"max_turnover": 1.5},
        {"step_periods": 50, "test_periods": 126},
    ],
)
def test_invalid_limits_fail_closed(limits):
    with pytest.raises(InsightError):
        _validated_rules(NAMES, REFERENCE, rules(**limits))


def test_turnover_limit_for_feasible_baseline():
    params = rules(max_turnover=0.05)
    constraints = _validated_rules(NAMES, REFERENCE, params)
    out = _project_candidate(np.array([0.1, 0.1, 0.7, 0.1]), REFERENCE, constraints)
    assert out is not None
    assert np.abs(out - REFERENCE).sum() / 2 <= 0.05000001


def test_infeasible_baseline_with_tight_turnover_is_not_laundered():
    params = rules(asset_caps={"AAPL": 0.1}, max_turnover=0.01)
    r = _validated_rules(NAMES, REFERENCE, params)
    assert _project_candidate(REFERENCE, REFERENCE, r) is None


def test_monthly_rebalance_and_nonzero_cost_are_auditable():
    r, idx = data(265)
    target = np.array([0.1, 0.3, 0.4, 0.2])
    zero, z_details = _simulated_returns(
        r, target, rebalance="monthly", cost_bps=0, starting_weights=REFERENCE
    )
    costly, c_details = _simulated_returns(
        r, target, rebalance="monthly", cost_bps=25, starting_weights=REFERENCE
    )
    assert c_details["one_way_turnover_sum"] >= 0.25
    assert c_details["cost_fraction_sum_before_compounding"] > 0
    assert c_details["charged_rebalance_events"] > 0
    assert (costly <= zero + 1e-12).all()
    assert (costly < zero).any()
    buy_hold, b_details = _simulated_returns(
        r, target, rebalance="buy_hold", cost_bps=25, starting_weights=REFERENCE
    )
    assert b_details["charged_rebalance_events"] == 1
    assert b_details["one_way_turnover_sum"] == pytest.approx(0.25)
    assert not np.allclose(buy_hold, costly)


def test_walkforward_nonoverlap_and_feasible_portfolio():
    r, idx = data()
    result = analyze_robustness(
        r,
        idx,
        REFERENCE,
        rules=rules(
            asset_caps={"AAPL": 0.20, "GLD": 0.15},
            asset_floors={"SPY": 0.2},
            group_caps={"stocks": {"symbols": ["AAPL", "MSFT"], "cap": 0.3}},
        ),
        priorities={"sharpe": 2.0, "volatility": 1.0, "drawdown": 1.0},
    )
    assert result["status"] == "evaluated"
    assert result["fold_count"] >= 3
    assert result["constraints"]["baseline_compliant"] is False
    for i, f in enumerate(result["folds"]):
        weights = f["weights"]
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["AAPL"] <= 0.200001
        assert weights["GLD"] <= 0.150001
        assert weights["SPY"] >= 0.199999
        assert weights["AAPL"] + weights["MSFT"] <= 0.300001
        assert set(f["verdicts"]) == {"sharpe", "volatility", "drawdown"}
        assert f["train_end"] == f["holdout_start"]
        if i:
            assert result["folds"][i - 1]["holdout_end"] <= f["holdout_start"]
    json.dumps(result, allow_nan=False)


def test_no_future_leakage_first_fold_is_identical_when_late_returns_change():
    original, idx = data(820)
    altered = original.copy()
    altered.iloc[-250:, :] = np.array([0.12, -0.11, 0.09, -0.08])  # severe future regime change
    setup = rules()
    a = analyze_robustness(original, idx, REFERENCE, rules=setup, priorities={"sharpe": 1})
    b = analyze_robustness(altered, idx, REFERENCE, rules=setup, priorities={"sharpe": 1})
    # The changed observations begin strictly after the first two holdout windows.
    # Strategy identity and weights must be identical. Derived floating-point
    # statistics can differ at machine-roundoff scale across NumPy/BLAS builds.
    for i in range(2):
        assert a["folds"][i]["selected"] == b["folds"][i]["selected"]
        assert a["folds"][i]["weights"] == b["folds"][i]["weights"]
        for key in (
            "train_metrics_selected",
            "baseline_holdout_metrics",
            "candidate_holdout_metrics",
        ):
            assert a["folds"][i][key] == pytest.approx(b["folds"][i][key], rel=1e-10, abs=1e-12)

    # Sanity check: the modification really changes later evaluated results.
    # Otherwise the test could pass even if it never exercised different data.
    assert any(
        abs(
            a["folds"][i]["baseline_holdout_metrics"]["cumulative_return"]
            - b["folds"][i]["baseline_holdout_metrics"]["cumulative_return"]
        )
        > 1e-3
        for i in range(2, len(a["folds"]))
    )


def test_insufficient_history_is_explicitly_unavailable():
    r, idx = data(220)
    d = analyze_robustness(r, idx, REFERENCE, rules=rules())
    assert d["status"] == "unavailable"
    assert "Need 378" in d["reason"]


def test_no_candidate_when_infeasible_original_and_tight_turnover():
    r, idx = data(400)
    d = analyze_robustness(
        r, idx, REFERENCE, rules=rules(asset_caps={"AAPL": 0.10}, max_turnover=0.01)
    )
    assert d["status"] == "unavailable"
    assert "No candidates satisfy" in d["reason"]


def test_actionable_benchmark_targets_not_vague_or_guarantees():
    r, dates = data(500)
    prices = {s: pd.Series(np.r_[100, 100 * np.cumprod(1 + r[s])], index=dates) for s in NAMES}
    report = analyze_insights(
        prices,
        dict(zip(NAMES, REFERENCE, strict=False)),
        {"SPY": prices["SPY"], "RSP": prices["GLD"]},
        base_currency="USD",
        mc_paths=20,
        robustness=rules(),
    )
    acts = report["actionable_diagnostics"]
    assert len(acts["benchmark_targets"]) == 12
    assert acts["allocation_change_validation_status"] in {
        "no_consistent_improvement",
        "historical_consistency_observed",
    }
    for row in acts["benchmark_targets"]:
        if row["sentiment"] != "unavailable":
            assert row["target"]["benchmark_value"] is not None
            assert row["research_next_step"]
    assert "No candidate" in acts["conclusion"] or "Candidate" in acts["conclusion"]
    json.dumps(report, allow_nan=False)


def test_overlap_and_missing_exposure_actions_not_buy_sell_advice():
    x = {
        "direct_weights": {"QQQ": 0.3, "SPMO": 0.2, "XEQT.TO": 0.5},
        "unknown_underlying": 0.5,
        "coverage_by_fund": {
            "XEQT.TO": {"portfolio_weight": 0.5, "unknown_fraction_of_fund": 1.0},
            "QQQ": {"portfolio_weight": 0.3, "unknown_fraction_of_fund": 0.0},
            "SPMO": {"portfolio_weight": 0.2, "unknown_fraction_of_fund": 0.0},
        },
        "pairwise_overlap": [
            {"fund_a": "QQQ", "fund_b": "SPMO", "identified_overlap_lower_bound": 0.43}
        ],
    }
    out = exposure_actions(x)
    assert out["data_completeness"] == "incomplete"
    assert {a["type"] for a in out["findings"]} == {
        "identified_fund_overlap",
        "unresolved_holdings",
    }
    assert all(a["research_next_step"] for a in out["findings"])


def test_rollout_preserves_v8_default_and_v83_optin():
    r, dates = data(400)
    prices = {s: pd.Series(np.r_[100, 100 * np.cumprod(1 + r[s])], index=dates) for s in NAMES}
    kwargs = {"base_currency": "USD", "mc_paths": 20}
    previous = analyze_insights(
        prices, dict(zip(NAMES, REFERENCE, strict=False)), {"SPY": prices["SPY"], "RSP": prices["GLD"]}, **kwargs
    )
    assert "rolling_robustness" not in previous
    assert previous["version"] == "8.0"
    current = analyze_insights(
        prices,
        dict(zip(NAMES, REFERENCE, strict=False)),
        {"SPY": prices["SPY"], "RSP": prices["GLD"]},
        robustness=rules(),
        **kwargs,
    )
    assert current["version"] == "8.3"
    assert current["rolling_robustness"]["fold_count"] == 1
    assert current["portfolio"] == previous["portfolio"]


def test_constraint_compliance_is_not_confused_with_outperformance():
    r, idx = data(820)
    report = analyze_robustness(
        r,
        idx,
        REFERENCE,
        rules=rules(
            asset_caps={"AAPL": 0.25, "GLD": 0.2},
            asset_floors={"SPY": 0.25},
            group_caps={"bigtech": {"symbols": ["AAPL", "MSFT"], "cap": 0.4}},
            max_turnover=0.3,
        ),
        priorities={"sharpe": 2, "volatility": 1, "drawdown": 1},
    )
    assert report["constraints"]["baseline_compliant"] is False
    example = report["constraint_compliance_example"]
    assert example["status"] == "feasible_not_risk_validated"
    w = example["weights"]
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["AAPL"] <= 0.250001 and w["GLD"] <= 0.200001 and w["SPY"] >= 0.249999
    assert w["AAPL"] + w["MSFT"] <= 0.400001
    assert example["initial_one_way_turnover"] <= 0.300001
    assert "no claim" in example["interpretation"].lower()
    assert report["constraints"]["max_turnover_scope"].startswith("Initial one-way")
    assert report["unvalidated_trailing_return_periods"] == (
        len(r) - (252 + (report["fold_count"] - 1) * 126 + 126)
    )
    assert report["available_return_history_end"] == str(idx[-1].date())


def test_noncompliant_original_never_given_retain_original_advice():
    robustness = {
        "status": "evaluated",
        "consistent_improvement_on_all_priorities": False,
        "constraints": {"baseline_compliant": False},
        "next_step": "Review example",
    }
    empty_report = {
        "diagnostics": {x: [] for x in ("negative", "positive", "neutral", "unavailable")},
        "risk_contribution": [],
    }
    actions = portfolio_actions(empty_report, robustness)
    assert "violates" in actions["conclusion"].lower()
    assert "compliant" in actions["conclusion"].lower()
    assert actions["next_action"] == "Review example"
