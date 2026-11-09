"""V9.10 tests: saved-value edit, and evidence for a single candidate on six metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v8_insights import OBJECTIVES
from portfolio_risk.v83_robustness import analyze_robustness
from portfolio_risk_ui.core import (PortfolioError, engine_rows,
                                    fingerprint, update_holding_amount)
from portfolio_risk_ui.v910_presentation import all_goal_review, tested_candidate_warning as candidate_cap_note

APP = (Path(__file__).resolve().parents[1] / "portfolio_risk_ui" / "app.py").read_text()


def test_holding_edit_updates_source_record_and_changes_report_signature():
    holdings = [{"symbol": "AAPL", "name": "Apple", "market_value": 8750.0,
                 "weight": 20.0, "currency": "USD", "quote_currency": "USD"},
                {"symbol": "NVDA", "name": "Nvidia", "market_value": 1_000_000.0,
                 "weight": 80.0, "currency": "USD", "quote_currency": "USD"}]
    original = fingerprint(engine_rows(holdings, "values", "USD"), "USD", "2000-01-01", "2026-11-08")
    assert update_holding_amount(holdings, "NVDA", "market_value", 10000.0)
    assert holdings[1]["market_value"] == 10000.0
    changed = fingerprint(engine_rows(holdings, "values", "USD"), "USD", "2000-01-01", "2026-11-08")
    assert original != changed
    assert update_holding_amount(holdings, "NVDA", "market_value", 10000.0) is False
    assert 'on_change=_commit_holding_widget' in APP
    assert 'st.session_state.reports.clear()' in APP.split('def _commit_holding_widget(', 1)[1].split('def page_builder()', 1)[0]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_edits_fail_without_mutating_portfolio(bad):
    rows = [{"symbol": "NVDA", "market_value": 1000.0}]
    with pytest.raises(PortfolioError):
        update_holding_amount(rows, "NVDA", "market_value", bad)
    assert rows[0]["market_value"] == 1000.0


def test_editing_target_weight_updates_same_source_of_truth():
    rows = [{"symbol": "AAPL", "weight": 50.0}, {"symbol": "NVDA", "weight": 50.0}]
    assert update_holding_amount(rows, "AAPL", "weight", 70.0)
    assert update_holding_amount(rows, "NVDA", "weight", 30.0)
    assert [r["weight"] for r in rows] == [70.0, 30.0]


def test_one_rolling_run_produces_all_six_metrics_without_changing_original_rule():
    rng = np.random.default_rng(876)
    n = 510
    dates = pd.date_range("2024-01-01", periods=n+1, freq="B")
    frame = pd.DataFrame(rng.normal(.0004, .014, (n, 3)),
                         columns=["NVDA", "SPY", "GLD"], index=dates[1:])
    result = analyze_robustness(frame, dates, np.array([.9, .07, .03]),
        priorities={"volatility": 1.0},
        rules={"max_weight": .7, "train_periods": 252, "test_periods": 126,
               "step_periods": 126, "rebalance": "quarterly", "cost_bps": 10.0})
    assert result["status"] == "evaluated"
    assert set(result["priorities"]) == {"volatility"}
    assert set(result["per_metric_tally"]) == {"volatility"}
    assert set(result["all_metric_tally"]) == set(OBJECTIVES)
    assert all(set(fold["verdicts"]) == {"volatility"} for fold in result["folds"])
    assert all(set(fold["all_metric_verdicts"]) == set(OBJECTIVES) for fold in result["folds"])
    for metric in OBJECTIVES:
        item = result["all_metric_tally"][metric]
        assert sum(item[k] for k in ("positive", "neutral", "negative", "unavailable")) == result["fold_count"]
    report = {"rolling_robustness": result, "input_weights": {"NVDA": .90, "SPY": .07, "GLD": .03}}
    review = all_goal_review(report)
    assert review["status"] == "evaluated"
    assert len(review["rows"]) == 6
    assert review["passed"] == result["consistent_improvement_on_all_priorities"]
    note = candidate_cap_note(report)
    assert note is not None and "NVDA" in note and "90.00%" in note and "70%" in note


def test_evidence_protects_against_claiming_six_optimizations_and_fake_rows():
    fake = {"rolling_robustness": {"status": "evaluated", "fold_count": 20,
            "per_metric_tally": {"sharpe": {"positive": 15, "neutral": 5, "negative": 0, "unavailable": 0}},
            "consistent_improvement_on_all_priorities": True}}
    view = all_goal_review(fake)
    assert len(view["rows"]) == 1  # older report only had one measured objective
    assert view["rows"][0]["Metric"] == "sharpe"
    assert len(view["strong"]) == 1
    assert "These are not six different portfolios" in APP
    invalid = {"rolling_robustness": {"status": "evaluated", "fold_count": 20,
              "all_metric_tally": {"sharpe": {"positive": 15, "neutral": 10, "negative": 0, "unavailable": 0}}}}
    assert all_goal_review(invalid)["status"] == "unavailable"  # 25 outcomes cannot be 20 tests
