"""V9.13 regression suite: per-goal reports must be real; SPY checks fail closed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v83_robustness import analyze_robustness
from portfolio_risk_ui.v913_strategy_paths import (
    PATHS, PICKED, path_summary, spy_benchmark_evidence, strategy_report_args,
)
from portfolio_risk_ui.v98_presentation import STRATEGY_GOALS

APP = (Path(__file__).resolve().parents[1] / "portfolio_risk_ui" / "app.py").read_text()


def test_three_goals_are_optimized_in_one_market_data_request():
    from portfolio_risk.v918_three_strategies import GOALS
    assert len(GOALS)==3
    assert {goal[1] for goal in GOALS.values()} == {'cagr','sharpe','volatility'}
    assert '"--three-strategies"' in APP
    assert "show_three_results(record['report'])" in APP
    assert 'Analyze all six portfolio strategies' not in APP.split('def page_strategy():',1)[1].split('def main():',1)[0]



def test_no_spy_data_means_no_claim_of_spy_outperformance():
    report = {"rolling_robustness": {"status": "evaluated", "folds": [
        {"candidate_vs_spy": {}, "weights": {"SPY": 1.}}]}}
    comparison = spy_benchmark_evidence(report, "cagr")
    assert comparison["status"] == "unavailable"


def test_spy_market_guard_does_not_pass_weak_growth_or_sharpe():
    folds = []
    for i in range(8):
        folds.append({"candidate_vs_spy": {
            "sharpe": {"candidate_value": 1.1, "spy_value": 1.0, "favorable_difference": .1},
            "cagr": {"candidate_value": .08, "spy_value": .10,
                     "favorable_difference": -.02 if i < 3 else .02},
        }})
    # Five of eight growth periods met SPY, below 75%=6: not sufficient.
    report = {"rolling_robustness": {"status": "evaluated", "folds": folds}}
    evidence = spy_benchmark_evidence(report, "sharpe")
    assert evidence["status"] == "evaluated"
    assert evidence["metrics"]["sharpe"]["matched"] == 8
    assert evidence["metrics"]["cagr"]["matched"] == 5
    assert not evidence["supported"]


def test_same_currency_matched_spy_benchmark_is_recorded_in_actual_robustness():
    rng = np.random.default_rng(913)
    n = 505
    days = pd.date_range("2020-01-01", periods=n+1, freq="B")
    assets = pd.DataFrame(rng.normal(.0004, .012, (n, 3)),
                          index=days[1:], columns=["AAPL", "GLD", "SPY"])
    spy = pd.Series(rng.normal(.0005, .011, n), index=days[1:])
    res = analyze_robustness(assets, days, np.array([.5, .25, .25]),
                             priorities={"sharpe": 1.},
                             rules={"train_periods": 252, "test_periods": 126,
                                    "step_periods": 126, "rebalance": "quarterly"},
                             benchmark_returns={"SPY": spy})
    assert res["status"] == "evaluated"
    assert "SPY" in res["spy_benchmark_method"]
    assert len(res["folds"]) == 2
    for fold in res["folds"]:
        assert fold["spy_holdout_metrics"] is not None
        assert set(fold["candidate_vs_spy"]) >= {"cagr", "sharpe", "volatility"}
    comp = spy_benchmark_evidence({"rolling_robustness":res}, "sharpe")
    assert comp["status"] == "evaluated"
    assert comp["metrics"]["cagr"]["total"] == 2


def test_spy_different_calendar_is_rejected_before_any_claim():
    rng = np.random.default_rng(914)
    days = pd.date_range("2020-01-01", periods=380, freq="B")
    assets = pd.DataFrame(rng.normal(0, .01, (379, 2)), index=days[1:], columns=["A", "B"])
    spy = pd.Series(np.zeros(379), index=pd.date_range("2022-01-01", periods=379, freq="B"))
    with pytest.raises(Exception, match="SPY benchmark returns must match"):
        analyze_robustness(assets, days, np.array([.5,.5]),
                           benchmark_returns={"SPY": spy})


def test_three_strategy_ui_does_not_equate_training_result_with_guarantee():
    from pathlib import Path
    view=(Path(__file__).resolve().parents[1]/'portfolio_risk_ui'/'v918_three_views.py').read_text(encoding='utf-8')
    assert 'not forecasts or trading instructions' in view
    assert 'later historical periods' in view
    assert "record.get('args') != args" in APP
    assert 'scenario_rule' in APP and 'scenario_invalid' in APP
    assert 'maximum historical growth' in APP

