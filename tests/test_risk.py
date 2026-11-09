"""V5 pure numerical tests, fully offline and independent of data providers."""

import math

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.risk import (
    RiskError,
    _asset_attribution,
    _coverage,
    _tail_estimates,
    analyze_risk,
    risk_as_dict,
)


def _synthetic(n=320):
    dates = pd.bdate_range("2024-01-02", periods=n + 1)
    ix = np.arange(n)
    x = 0.003 * np.sin(ix * 0.4) + 0.001 * np.cos(ix * 0.33)
    y = 0.004 * np.cos(ix * 0.5) - 0.001 * np.sin(ix * 0.2)
    a = pd.Series(100.0 * np.r_[1, np.cumprod(1 + x)], index=dates)
    b = pd.Series(100.0 * np.r_[1, np.cumprod(1 + y)], index=dates)
    ret = pd.Series(0.6 * x + 0.4 * y, index=dates[1:])
    return {"AAA": a, "BBB": b}, {"AAA": 0.6, "BBB": 0.4}, ret


def test_historical_tail_uses_exact_fractional_mass():
    # losses [0.2, 0.1, 0.0, -0.1] in descending order.
    hist, normal = _tail_estimates(np.array([-0.2, -0.1, 0.0, 0.1]), 0.75)
    assert hist[0] == pytest.approx(0.125)
    assert hist[1] == pytest.approx(0.20)
    assert normal[1] >= normal[0] >= 0


@pytest.mark.parametrize("alpha", [0.9, 0.95, 0.99])
def test_expected_shortfall_at_least_var(alpha):
    x = np.array([-0.1, -0.02, -0.01, 0.0, 0.03, 0.05, 0.001] * 20)
    for v, e in _tail_estimates(x, alpha):
        assert e >= v >= 0


def test_zero_variance_returns_have_no_spurious_risk():
    for v, e in _tail_estimates(np.full(50, 0.01), 0.95):
        assert v == pytest.approx(0)
        assert e == pytest.approx(0)


def test_var_and_es_increase_with_volatility():
    arr = np.array([-0.02, -0.01, -0.005, 0.005, 0.01, 0.02] * 60)
    h1, n1 = _tail_estimates(arr, 0.95)
    h2, n2 = _tail_estimates(arr * 2, 0.95)
    assert h2[0] == pytest.approx(h1[0] * 2)
    assert h2[1] == pytest.approx(h1[1] * 2)
    assert n2[0] == pytest.approx(n1[0] * 2)
    assert n2[1] == pytest.approx(n1[1] * 2)


def test_current_attribution_reconciles_covar():
    prices, weights, returns = _synthetic()
    rets = pd.concat(prices, axis=1).pct_change(fill_method=None).iloc[1:]
    contributions, vol, ratio = _asset_attribution(rets, weights)
    assert [r.symbol for r in contributions] == ["AAA", "BBB"]
    assert sum(r.contribution_to_volatility for r in contributions) == pytest.approx(vol)
    assert sum(r.share_of_total_risk for r in contributions) == pytest.approx(1.0)
    assert ratio is not None and ratio > 0


def test_perfect_anticorrelation_can_have_negative_risk_contribution():
    n = 50
    a = np.tile(np.array([-0.02, 0.02]), n)
    frame = pd.DataFrame({"AAA": a, "BBB": -a})
    contributions, vol, ratio = _asset_attribution(frame, {"AAA": 0.75, "BBB": 0.25})
    assert vol > 0
    assert contributions[0].share_of_total_risk == pytest.approx(1.5)
    assert contributions[1].share_of_total_risk == pytest.approx(-0.5)
    assert ratio == pytest.approx(2.0)


def test_single_asset_risk_share_100_percent():
    prices, _, ret = _synthetic()
    a = prices["AAA"]
    report = analyze_risk({"AAA": a}, {"AAA": 1.0}, a.pct_change(fill_method=None).iloc[1:])
    assert len(report.attribution) == 1
    assert report.attribution[0].share_of_total_risk == pytest.approx(1.0)


def test_risk_report_shapes_and_stress():
    prices, weights, returns = _synthetic()
    result = analyze_risk(
        prices,
        weights,
        returns,
        strategy="quarterly",
        lookback=60,
        shocks={"AAA": -0.30},
        reference_capital=20000,
    )
    assert result.strategy == "quarterly"
    assert len(result.risk_estimates) == 4
    assert len(result.coverage) == 4
    assert len(result.attribution) == 2
    assert result.estimation_observations == 60
    assert all(x.tests == len(returns) - 60 for x in result.coverage)
    assert result.stresses[-1].source == "hypothetical"
    assert result.stresses[-1].portfolio_return == pytest.approx(-0.18)
    assert result.stresses[-1].change_on_capital == pytest.approx(-3600)
    assert result.stresses[0].portfolio_return == pytest.approx(float(returns.min()))
    output = risk_as_dict(result, include_series=True)
    assert len(output["strategy_returns"]) == len(returns)
    assert output["risk_estimates"][0]["var_amount"] >= 0


def test_one_step_validation_has_no_peeking_at_next_return():
    prices, weights, returns = _synthetic()
    base = _coverage(returns, 60, 0.95, "historical")
    changed = returns.copy()
    changed.iloc[-1] = -0.8
    later = _coverage(changed, 60, 0.95, "historical")
    assert base.last_var == pytest.approx(later.last_var)
    assert later.breaches == base.breaches + 1
    assert later.tests == base.tests


@pytest.mark.parametrize("model", ["historical", "parametric_normal"])
@pytest.mark.parametrize("conf", [0.95, 0.99])
def test_kupiec_p_value_valid(model, conf):
    _, _, returns = _synthetic(410)
    c = _coverage(returns, 252, conf, model)
    assert c.tests == 158
    assert c.expected_breaches == pytest.approx(c.tests * (1 - conf))
    assert 0 <= c.kupiec_p_value <= 1
    assert c.kupiec_lr >= 0


def test_no_validation_periods_reported_honestly():
    prices, weights, returns = _synthetic(50)
    r = analyze_risk(prices, weights, returns, lookback=252)
    assert r.estimation_observations == 50
    assert all(t.tests == 0 and t.kupiec_p_value is None for t in r.coverage)
    assert any("SHORT_COVERAGE" in w for w in r.warnings)


def test_coverage_zero_breaches_is_finite():
    idx = pd.bdate_range("2024-01-02", periods=50)
    # If all returns are positive, risk is zero and there are zero exceedances.
    c = _coverage(pd.Series([0.01] * 50, index=idx), 30, 0.95, "historical")
    assert c.breaches == 0
    assert math.isfinite(c.kupiec_lr)
    assert 0 <= c.kupiec_p_value <= 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.01])
def test_invalid_returns_rejected(bad):
    prices, weights, returns = _synthetic()
    r = returns.copy()
    r.iloc[3] = bad
    with pytest.raises(RiskError):
        analyze_risk(prices, weights, r)


def test_unknown_shock_rejected():
    prices, weights, returns = _synthetic()
    with pytest.raises(RiskError, match="Unrecognized"):
        analyze_risk(prices, weights, returns, shocks={"XYZ": -0.3})


def test_non_normalized_weights_rejected():
    prices, _, returns = _synthetic()
    with pytest.raises(RiskError, match="sum to one"):
        analyze_risk(prices, {"AAA": 0.9, "BBB": 0.9}, returns)


def test_misaligned_return_dates_rejected():
    prices, weights, returns = _synthetic()
    r = returns.copy()
    r.index += pd.Timedelta(days=1)
    with pytest.raises(RiskError, match="dates absent"):
        analyze_risk(prices, weights, r)


def test_not_allowed_negative_weight():
    prices, _, returns = _synthetic()
    with pytest.raises(RiskError, match="nonnegative"):
        analyze_risk(prices, {"AAA": -1, "BBB": 2}, returns)


def test_negative_scenario_min_minus_100_percent():
    prices, weights, returns = _synthetic()
    with pytest.raises(RiskError, match="at least -100"):
        analyze_risk(prices, weights, returns, shocks={"AAA": -1.2})


def test_invalid_lookback_rejected():
    prices, weights, returns = _synthetic()
    with pytest.raises(RiskError, match="Lookback"):
        analyze_risk(prices, weights, returns, lookback=20)


def test_missing_prices_on_strategy_dates():
    prices, weights, returns = _synthetic()
    truncated = {"AAA": prices["AAA"].iloc[:-10], "BBB": prices["BBB"]}
    with pytest.raises(RiskError, match="dates absent"):
        analyze_risk(truncated, weights, returns)
