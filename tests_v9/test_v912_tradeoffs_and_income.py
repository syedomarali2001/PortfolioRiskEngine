"""Focused regression checks for V9.12's historical summaries and UI design."""
from pathlib import Path

from portfolio_risk_ui.v912_presentation import (
    income_ranked, strategy_goal_name, strategy_option_label,
    strategy_tradeoff, status_for_summary,
)

APP = (Path(__file__).resolve().parents[1] / "portfolio_risk_ui" / "app.py").read_text()


def example_report():
    original = {"NVDA": .85, "GLD": .15}
    tested = {"NVDA": .65, "GLD": .35}
    verdicts = {
        "cagr": {"sentiment": "negative", "original_value": .18, "candidate_value": -.0054},
        "volatility": {"sentiment": "positive", "original_value": .2183, "candidate_value": .1283},
        "drawdown": {"sentiment": "positive", "original_value": .1292, "candidate_value": .0752},
        "sharpe": {"sentiment": "negative", "original_value": 1.2, "candidate_value": .2},
        "sortino": {"sentiment": "negative", "original_value": 1.3, "candidate_value": .3},
        "es95": {"sentiment": "positive", "original_value": .030, "candidate_value": .019},
    }
    tally = {
        "cagr": {"positive": 11, "neutral": 2, "negative": 15, "unavailable": 0},
        "volatility": {"positive": 28, "neutral": 0, "negative": 0, "unavailable": 0},
        "drawdown": {"positive": 21, "neutral": 7, "negative": 0, "unavailable": 0},
    }
    return {
        "input_weights": original,
        "rolling_robustness": {
            "status": "evaluated", "fold_count": 28, "consistent_improvement_on_all_priorities": False,
            "folds": [{"weights": tested, "selected": "minimum_variance",
                       "holdout_end": "2026-11-03", "train_end": "2026-05-05",
                       "all_metric_verdicts": verdicts}],
            "all_metric_tally": tally,
        },
    }


def test_income_ranks_by_estimated_dollars_and_keeps_missing_last():
    holdings = [
        {"symbol": "GLD", "estimated_annual_gross_income": 0.0},
        {"symbol": "SPY", "estimated_annual_gross_income": 77.35},
        {"symbol": "SCHD", "estimated_annual_gross_income": 3197.7},
        {"symbol": "NO_DATA", "estimated_annual_gross_income": None},
        {"symbol": "AAPL", "estimated_annual_gross_income": 27.09},
    ]
    sorted_holdings = income_ranked(holdings)
    assert [r["symbol"] for r in sorted_holdings] == ["SCHD", "SPY", "AAPL", "GLD", "NO_DATA"]
    assert [r["symbol"] for r in holdings] == ["GLD", "SPY", "SCHD", "NO_DATA", "AAPL"]


def test_tradeoff_uses_same_fold_numbers_and_separate_aggregate_evidence():
    tradeoff = strategy_tradeoff(example_report())
    assert tradeoff["status"] == "evaluated"
    assert tradeoff["passed"] is False
    assert tradeoff["latest_end"] == "2026-11-03"
    metrics = {m["key"]: m for m in tradeoff["measures"]}
    assert metrics["cagr"]["result"] == "Worse"
    assert metrics["cagr"]["original"] == .18
    assert metrics["cagr"]["tested"] == -.0054
    assert round(metrics["volatility"]["delta_pp"], 2) == -9.0
    assert metrics["drawdown"]["result"] == "Better"
    assert "sacrificed annualized growth" in tradeoff["interpretation"]
    historical = {m["key"]: m for m in tradeoff["all_fold_counts"]}
    assert historical["volatility"]["better"] == 28
    assert historical["cagr"]["worse"] == 15
    assert 'does not support' in status_for_summary(example_report()).lower()


def test_unknown_tradeoff_never_invents_outcomes():
    assert strategy_tradeoff({})["status"] == "unavailable"
    assert strategy_tradeoff({"rolling_robustness": {"status": "evaluated", "folds": [{}]}})["status"] == "unavailable"


def test_six_paths_are_run_before_the_user_selects_a_tab():
    assert strategy_goal_name("sharpe:2,volatility:1,drawdown:1") == "Balance growth and risk"
    assert "growth" in strategy_option_label("More annual growth").lower()
    assert 'st.pills("Optimization focus"' not in APP
    assert 'def _strategy_run_all(base_args: list[str])' in APP
    assert 'for index, (goal, short, _metric, _explanation) in enumerate(PATHS' in APP
    assert 'tabs = st.tabs(' in APP
    assert 'def _run_strategy_cached(args: list[str]) -> bool:' in APP
    assert '"signature": _portfolio_signature()' in APP



def test_income_sort_and_chart_tooltips_have_readable_contrast():
    assert 'for holding in income_ranked(r.get("holdings") or []):' in APP
    assert 'Sorted by estimated annual dividend income, highest first' in APP
    # Fix both the tooltip background AND light-colored extra label.
    assert 'hovertemplate="%{x}<br>Estimated value: %{y:,.2f}<extra></extra>"' in APP
    assert 'hovertemplate=label + "<br>%{y}: %{x:.2f}%<extra></extra>"' in APP
    assert 'bgcolor="#132238"' in APP
