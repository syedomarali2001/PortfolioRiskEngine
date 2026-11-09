"""V8 pure analytic tests: all data synthetic and offline."""

import json

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v8_insights import InsightError, _details, analyze_insights


def fixture(n=480, seed=44):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    vals = [
        rng.normal(m, s, size=n)
        for m, s in ((0.0005, 0.011), (0.0001, 0.013), (0.0006, 0.01), (0.0003, 0.012))
    ]
    series = [pd.Series(100 * np.cumprod(1 + x), index=dates) for x in vals]
    return (
        {"AAA": series[0], "BBB": series[1]},
        {"AAA": 0.6, "BBB": 0.4},
        {"SPY": series[2], "RSP": series[3]},
    )


def analyze(**kwargs):
    p, w, b = fixture()
    return analyze_insights(p, w, b, **kwargs)


def test_both_benchmarks_are_really_matched_and_diagnostics_explain():
    doc = analyze(priorities={"sharpe": 2, "drawdown": 1})
    assert set(doc["benchmarks"]) == {"SPY", "RSP"}
    assert doc["return_periods"] == 479
    assert sum(len(x) for x in doc["diagnostics"].values()) == 12
    assert all(
        "Materiality threshold" in e["evidence"]
        for k in ("positive", "negative", "neutral")
        for e in doc["diagnostics"][k]
    )
    assert doc["historical_strategy_selection"]["status"] == "evaluated"
    assert (
        doc["historical_strategy_selection"]["train_end"]
        == doc["historical_strategy_selection"]["holdout_start"]
    )
    assert set(doc["historical_strategy_selection"]["objective_weights"]) == {"sharpe", "drawdown"}
    assert doc["risk_contribution"]
    assert sum(x["share_of_total_risk"] for x in doc["risk_contribution"]) == pytest.approx(1)
    assert doc["block_bootstrap_monte_carlo"]["status"] == "evaluated"
    json.dumps(doc, allow_nan=False)


def test_different_conventional_metric_directions_and_threshold():
    a = {
        "sharpe": 1.2,
        "sortino": 1.0,
        "volatility": 0.10,
        "drawdown": 0.12,
        "cagr": 0.09,
        "es95": 0.02,
    }
    b = {
        "sharpe": 0.8,
        "sortino": 1.0,
        "volatility": 0.18,
        "drawdown": 0.25,
        "cagr": 0.08,
        "es95": 0.03,
    }
    assert _details(a, b, "sharpe", "SPY", 300)["sentiment"] == "positive"
    assert _details(a, b, "volatility", "SPY", 300)["sentiment"] == "positive"
    assert _details(a, b, "drawdown", "RSP", 300)["sentiment"] == "positive"
    assert _details(b, a, "volatility", "SPY", 300)["sentiment"] == "negative"
    assert _details(a, b, "cagr", "SPY", 300)["sentiment"] == "neutral"
    assert _details(a, b, "sharpe", "SPY", 90)["sentiment"] == "unavailable"
    assert _details({"sharpe": None}, b, "sharpe", "SPY", 300)["sentiment"] == "unavailable"


def test_scenario_is_not_renormalized_and_metrics_preserved():
    doc = analyze(scenarios={"half": {"AAA": 0.5, "BBB": 0.5}})
    assert doc["user_scenarios"]["half"]["weights"] == {"AAA": 0.5, "BBB": 0.5}
    assert "vs_input" in doc["user_scenarios"]["half"]
    with pytest.raises(InsightError, match="sum to 100"):
        analyze(scenarios={"half": {"AAA": 0.4, "BBB": 0.4}})
    with pytest.raises(InsightError, match="exactly one weight"):
        analyze(scenarios={"half": {"AAA": 1.0}})
    with pytest.raises(InsightError, match="nonnegative"):
        analyze(scenarios={"half": {"AAA": -0.1, "BBB": 1.1}})


def test_optimizer_past_only_selection_does_not_see_late_shocks():
    assets, w, bench = fixture()
    d1 = analyze_insights(assets, w, bench)
    # Alter only last 50 returns in one asset; training sample stays identical.
    mod = assets.copy()
    mod["AAA"] = assets["AAA"].copy()
    last = mod["AAA"].index[-50]
    mod["AAA"].loc[last:] *= 0.7
    d2 = analyze_insights(mod, w, bench)
    c1 = d1["historical_strategy_selection"]
    c2 = d2["historical_strategy_selection"]
    assert c1["selected"] == c2["selected"]
    assert (
        c1["candidates_selected"][c1["selected"]]["training_metrics"]
        == c2["candidates_selected"][c2["selected"]]["training_metrics"]
    )
    assert (
        c1["candidates_selected"]["input"]["holdout_metrics"]
        != c2["candidates_selected"]["input"]["holdout_metrics"]
    )


def test_monte_carlo_fixed_seed_reproduces_and_uses_block_resampling():
    d1 = analyze(seed=10, mc_paths=50, mc_horizon=60)
    d2 = analyze(seed=10, mc_paths=50, mc_horizon=60)
    d3 = analyze(seed=11, mc_paths=50, mc_horizon=60)
    def x(d):
        return d["block_bootstrap_monte_carlo"]["terminal_return_p05_p50_p95"]
    assert x(d1) == x(d2)
    assert x(d1) != x(d3)
    assert d1["block_bootstrap_monte_carlo"]["block_length"] == 5


def test_no_fake_crisis_for_insufficient_historical_window():
    doc = analyze()
    assert all(e["status"] == "unavailable" for e in doc["historical_events"]["portfolio"])
    assert doc["starting_date_sensitivity"]["5y"]["status"] == "unavailable"


def test_short_sample_withholds_sentiment_and_strategy():
    prices, w, bench = fixture(n=90)
    doc = analyze_insights(prices, w, bench)
    assert len(doc["diagnostics"]["unavailable"]) == 12
    assert doc["historical_strategy_selection"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "problem", ["missing_benchmark", "nan", "few_dates", "negative_price", "bad_rf", "max_weight"]
)
def test_bad_data_fails_closed(problem):
    prices, w, bench = fixture()
    if problem == "missing_benchmark":
        bench.pop("RSP")
    if problem == "nan":
        prices["AAA"].iloc[30] = np.nan
    if problem == "negative_price":
        prices["AAA"].iloc[30] = -1
    if problem == "few_dates":
        bench["RSP"] = bench["RSP"].iloc[-3:]
    kw = {}
    if problem == "bad_rf":
        kw["annual_risk_free_rate"] = -1
    if problem == "max_weight":
        kw["max_weight"] = 0.2
    with pytest.raises((InsightError, ValueError)):
        analyze_insights(prices, w, bench, **kw)


def test_same_stock_in_portfolio_and_benchmark_not_dropped():
    prices, w, bench = fixture()
    prices["SPY"] = bench["SPY"]
    w = {"AAA": 0.2, "BBB": 0.3, "SPY": 0.5}
    doc = analyze_insights(prices, w, bench)
    assert doc["input_weights"]["SPY"] == 0.5
    assert "SPY" in doc["benchmarks"]


def test_missing_chronological_dates_do_not_fill_forward():
    prices, w, bench = fixture()
    bench["RSP"] = bench["RSP"].drop(bench["RSP"].index[20:25])
    doc = analyze_insights(prices, w, bench)
    assert doc["shared_close_dates"] == 475


def test_volatility_is_reported_and_scored_using_canonical_annualized_field():
    """Regression: formerly portfolio volatility existed but alerts said unavailable."""
    doc = analyze(priorities={"volatility": 1.0})
    vol_alerts = [
        row
        for group in doc["diagnostics"].values()
        for row in group
        if row["metric"] == "volatility"
    ]
    assert len(vol_alerts) == 2
    for alert in vol_alerts:
        assert alert["sentiment"] in {"positive", "negative", "neutral"}
        assert alert["portfolio_value"] == pytest.approx(doc["portfolio"]["annualized_volatility"])
        assert alert["benchmark_value"] == pytest.approx(
            doc["benchmarks"][alert["benchmark"]]["annualized_volatility"]
        )
    strategy = doc["historical_strategy_selection"]
    candidates = strategy["training_candidate_scorecard"]
    assert (
        candidates["minimum_variance"]["objective_score"] > candidates["input"]["objective_score"]
    )
    winner = strategy["selected"]
    a = strategy["candidates_selected"][winner]["holdout_metrics"]["annualized_volatility"]
    b = strategy["candidates_selected"]["input"]["holdout_metrics"]["annualized_volatility"]
    assessment = strategy["holdout_objective_verdicts"]["volatility"]
    assert assessment["favorable_difference"] == pytest.approx(b - a)
    assert assessment["sentiment"] in {"positive", "negative", "neutral"}


def test_whatif_volatility_reports_nonzero_comparison_when_allocation_changes():
    doc = analyze(scenarios={"equal": {"AAA": 0.5, "BBB": 0.5}})
    scenario = doc["user_scenarios"]["equal"]
    assert "volatility" in scenario["vs_input"]
    assert scenario["vs_input"]["volatility"]["difference"] == pytest.approx(
        scenario["metrics"]["annualized_volatility"] - doc["portfolio"]["annualized_volatility"]
    )


def test_holdout_missing_metric_is_unavailable_not_neutral(monkeypatch):
    import portfolio_risk.v8_insights as mod
    from portfolio_risk.v8_insights import _recommendations

    dates = pd.bdate_range("2020-01-01", periods=150)
    frame = pd.DataFrame(
        {"AAA": 0.001 + np.arange(150) * 0.000001, "BBB": -0.001 + np.arange(150) * 0.000002},
        index=dates,
    )
    base = mod._stats

    def missing_volume(series, dates, rf):
        result = base(series, dates, rf)
        # Simulate a data condition where the chosen metric cannot be derived.
        result["annualized_volatility"] = None
        return result

    monkeypatch.setattr(mod, "_stats", missing_volume)
    outcome = _recommendations(
        frame,
        pd.bdate_range("2019-12-31", periods=151),
        np.array([0.5, 0.5]),
        0.0,
        {"volatility": 1.0},
        0.7,
    )
    assert outcome["holdout_objective_verdicts"]["volatility"]["sentiment"] == "unavailable"
    assert outcome["holdout_objective_verdicts"]["volatility"]["favorable_difference"] is None
