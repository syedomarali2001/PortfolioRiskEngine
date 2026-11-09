"""V9.11: allocational examples are real backend fold data, not recommendations."""
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_risk.v83_robustness import analyze_robustness
from portfolio_risk_ui.v911_presentation import (
    latest_candidate_example, candidate_label, format_metric_value, status_group,
)

APP = (Path(__file__).resolve().parents[1]/"portfolio_risk_ui"/"app.py").read_text()


def sample_report():
    base = {"NVDA": .90, "SPY": .10}
    candidate = {"NVDA": .70, "SPY": .30}
    last = {"selected": "equal_weight", "train_end": "2025-01-01",
            "holdout_start": "2025-01-01", "holdout_end": "2025-07-01", "weights": candidate,
            "all_metric_verdicts": {
                "volatility": {"sentiment": "positive", "original_value": .28, "candidate_value": .20},
                "sharpe": {"sentiment": "negative", "original_value": 1.5, "candidate_value": 1.0},
            }}
    prior = {**last, "weights": {"NVDA": .60, "SPY": .40}, "holdout_end": "2024-12-01"}
    return {"input_weights": base,
            "rolling_robustness": {"status": "evaluated", "fold_count": 2,
                                    "folds": [prior, last],
                                    "consistent_improvement_on_all_priorities": False}}


def test_latest_shows_exact_weights_and_holdout_tradeoffs():
    result = latest_candidate_example(sample_report())
    assert result["status"] == "evaluated"
    assert result["selected_label"] == "Equal weights"
    assert result["rows"][0]["Ticker"] == "NVDA"
    assert abs(result["rows"][0]["Change (percentage points)"] + 20.) < 1e-8
    assert abs(result["allocation_turnover_pct"] - 20.) < 1e-8
    assert result["training_end"] == "2025-01-01"
    assert [r["Result"] for r in result["measures"]] == ["Worse", "Better"]
    nvda = next(r for r in result["historical_ranges"] if r["Ticker"] == "NVDA")
    assert nvda["Lowest tested (%)"] == 60
    assert nvda["Highest tested (%)"] == 70
    assert nvda["Median tested (%)"] == 65
    assert result["passed"] is False


def test_invalid_missing_or_incomplete_weights_not_made_up():
    report = sample_report()
    report["rolling_robustness"]["folds"][-1]["weights"].pop("SPY")
    assert latest_candidate_example(report)["status"] == "unavailable"
    report = sample_report()
    report["rolling_robustness"]["folds"][-1]["weights"]["NVDA"] = float("nan")
    assert latest_candidate_example(report)["status"] == "unavailable"
    assert latest_candidate_example({})["status"] == "unavailable"


def test_candidate_labels_and_formatting():
    assert "variance" in candidate_label("minimum_variance").lower()
    assert "blend" in candidate_label("minvar_equal_50").lower()
    assert format_metric_value("sharpe", 1.289) == "1.29"
    assert format_metric_value("cagr", .1289) == "12.89%"
    assert status_group({"passed_independent_check": True})[0] == "Consistent improvement"


def test_ui_presents_three_distinct_historical_paths_after_one_run():
    from portfolio_risk.v918_three_strategies import GOALS
    assert set(GOALS)=={'growth','balanced','defensive'}
    assert '--three-strategies' in APP
    assert "st.button('Optimize my portfolio'" in APP
    assert "show_three_results(record['report'])" in APP
    assert 'st.subheader("How should the target weights be maintained?")' in APP




def test_real_robustness_rows_get_exposed_without_recalculation():
    gen = np.random.default_rng(7399)
    n = 505
    date = pd.date_range("2024-01-01", periods=n+1, freq="B")
    frame = pd.DataFrame(gen.normal(.0005, .012, size=(n, 3)),
                         index=date[1:], columns=["NVDA", "SPY", "GLD"])
    robust = analyze_robustness(frame, date, np.array([.9, .08, .02]),
        priorities={"sharpe": 1.},
        rules={"max_weight": .7, "train_periods": 252, "test_periods": 126,
               "step_periods": 126, "rebalance": "quarterly", "cost_bps": 10.})
    report = {"input_weights": {"NVDA": .9, "SPY": .08, "GLD": .02},
              "rolling_robustness": robust}
    example = latest_candidate_example(report)
    assert example["status"] == "evaluated"
    assert len(example["rows"]) == 3
    assert len(example["measures"]) == 6
    assert example["selected"] == robust["folds"][-1]["selected"]
    np.testing.assert_allclose(sum(r["Tested (%)"] for r in example["rows"]), 100, atol=1e-4)
