"""V9.15 reproducible stress tests for continuous, training-only selection.

Synthetic return series are intentionally not actual securities or forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v83_robustness import (
    _project_candidate,
    _validated_rules,
    analyze_robustness,
)
from portfolio_risk.v915_optimizer import continuous_candidates

NAMES = ["CCL", "AAPL", "SPY", "MSFT", "GLD"]


def broken_portfolio(kind="cruise", n=630, seed=915):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.00035, 0.007, n)
    normal = np.column_stack(
        [
            -0.0006 + 1.6 * market + rng.normal(0, 0.024, n),  # volatile trouble stock
            0.00035 + market + rng.normal(0, 0.007, n),
            market,
            0.00035 + market + rng.normal(0, 0.006, n),
            0.00008 - 0.12 * market + rng.normal(0, 0.006, n),
        ]
    )
    dates = pd.bdate_range("2020-01-02", periods=n + 1)
    w = (
        np.array([0.8, 0.07, 0.06, 0.04, 0.03])
        if kind == "cruise"
        else np.array([0.9, 0.025, 0.025, 0.025, 0.025])
        if kind == "concentrated"
        else np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    )
    return pd.DataFrame(normal, index=dates[1:], columns=NAMES), dates, w


def spec(**kw):
    return {
        "max_weight": 0.4,
        "train_periods": 252,
        "test_periods": 126,
        "step_periods": 126,
        "rebalance": "quarterly",
        "cost_bps": 10.0,
        **kw,
    }


@pytest.mark.parametrize("kind", ["cruise", "concentrated", "diversified"])
def test_continuous_search_respects_limits_and_training_only(kind):
    data, dates, ref = broken_portfolio(kind)
    rules = _validated_rules(NAMES, ref, spec())
    result = continuous_candidates(
        data.iloc[:252], ref, rules, 0.0, {"sharpe": 1}, _project_candidate
    )
    assert len(result) >= 2
    assert any(key.startswith("continuous_sharpe") for key in result)
    assert any(key.startswith("frontier_") for key in result)
    for w in result.values():
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1, abs=1e-7)
        assert w.max() <= 0.4000001
        assert (w >= -1e-10).all()


def test_80pct_ccl_test_finds_real_reallocation_and_lower_later_volatility():
    data, dates, reference = broken_portfolio()
    output = analyze_robustness(data, dates, reference, rules=spec(), priorities={"volatility": 1})
    assert output["status"] == "evaluated"
    assert output["fold_count"] >= 2
    assert all(f["candidate_count"] >= 10 for f in output["folds"])
    latest = output["folds"][-1]
    assert latest["weights"]["CCL"] <= 0.4000001
    assert sum(latest["weights"].values()) == pytest.approx(1, abs=1e-7)
    assert (
        latest["candidate_holdout_metrics"]["annualized_volatility"]
        < latest["baseline_holdout_metrics"]["annualized_volatility"]
    )
    assert any(
        "continuous" in fold["selected"] or "frontier" in fold["selected"]
        for fold in output["folds"]
    )


def test_ninety_percent_concentration_also_reduces_tested_risk():
    data, dates, reference = broken_portfolio("concentrated", seed=916)
    output = analyze_robustness(data, dates, reference, rules=spec(), priorities={"sharpe": 1})
    assert output["status"] == "evaluated"
    for fold in output["folds"]:
        assert fold["weights"]["CCL"] <= 0.4000001
        assert sum(fold["weights"].values()) == pytest.approx(1.0, abs=1e-7)
    assert (
        output["folds"][-1]["candidate_holdout_metrics"]["annualized_volatility"]
        < output["folds"][-1]["baseline_holdout_metrics"]["annualized_volatility"]
    )


def test_future_data_does_not_change_earliest_selected_candidate():
    data, dates, ref = broken_portfolio()
    changed = data.copy()
    changed.iloc[400:] = changed.iloc[400:] * -1.5
    original = analyze_robustness(data, dates, ref, rules=spec(), priorities={"sharpe": 1})
    altered = analyze_robustness(changed, dates, ref, rules=spec(), priorities={"sharpe": 1})
    first = original["folds"][0]
    new = altered["folds"][0]
    assert first["selected"] == new["selected"]
    assert first["weights"] == pytest.approx(new["weights"], abs=1e-10)
    assert first["candidate_holdout_metrics"]["cagr"] == pytest.approx(
        new["candidate_holdout_metrics"]["cagr"]
    )


def test_group_cap_asset_floors_and_max_turnover_with_continuous_search():
    data, _, reference = broken_portfolio("diversified")
    constraints = _validated_rules(
        NAMES,
        reference,
        spec(
            asset_caps={"CCL": 0.15},
            asset_floors={"SPY": 0.2},
            group_caps={"us_tech": {"symbols": ["AAPL", "MSFT"], "cap": 0.3}},
            max_turnover=0.22,
        ),
    )
    result = continuous_candidates(
        data.iloc[:252], reference, constraints, 0.0, {"es95": 1}, _project_candidate
    )
    assert result
    for w in result.values():
        assert w[0] <= 0.1500001
        assert w[2] >= 0.1999999
        assert w[1] + w[3] <= 0.3000001
        assert np.abs(w - reference).sum() / 2 <= 0.2200001
