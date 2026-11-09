"""Deterministic OLS regression checks for descriptive V7 ETF proxy analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.factor_proxy import (
    FactorProxyError,
    analyze_factor_proxies,
    factor_proxy_as_dict,
)


def prices(x):
    dates = pd.bdate_range("2024-01-02", periods=len(x) + 1)
    return pd.Series(np.r_[100.0, 100 * np.cumprod(1 + np.asarray(x))], index=dates)


def test_linear_beta_and_alpha():
    x = np.array([-0.02, 0.013, 0.017, -0.005, 0.025, -0.01] * 20)
    proxy = prices(x)
    portfolio = prices(2 * x + 0.001)
    r = analyze_factor_proxies(portfolio, {"SPY": proxy})["SPY"]
    assert r.beta == pytest.approx(2.0, abs=1e-10)
    assert r.correlation == pytest.approx(1)
    assert r.r_squared == pytest.approx(1)
    assert r.alpha_annualized_arithmetic == pytest.approx(0.252)
    assert r.observations == len(x)


def test_negative_beta():
    x = np.array([-0.03, 0.04, 0.02, -0.01] * 20)
    r = analyze_factor_proxies(prices(-0.5 * x + 0.003), {"SPY": prices(x)})["SPY"]
    assert r.beta == pytest.approx(-0.5)
    assert r.correlation == pytest.approx(-1)
    assert r.r_squared == pytest.approx(1)


def test_no_price_variation_yields_null_not_infinity():
    x = np.linspace(-0.01, 0.02, 80)
    r = analyze_factor_proxies(prices(x), {"FLAT": prices(np.zeros(80))})["FLAT"]
    assert r.beta is None
    assert r.correlation is None
    assert r.r_squared is None
    data = factor_proxy_as_dict({"FLAT": r})
    import json

    json.dumps(data, allow_nan=False)


def test_short_sample_is_rejected():
    with pytest.raises(FactorProxyError, match="shared price"):
        analyze_factor_proxies(
            prices(np.array([0.01] * 10)), {"SPY": prices(np.array([0.02] * 10))}
        )


def test_no_proxy_rejected():
    with pytest.raises(FactorProxyError, match="At least"):
        analyze_factor_proxies(prices(np.array([0.01] * 80)), {})


@pytest.mark.parametrize("min_obs", [0, 2, 3.5])
def test_invalid_min_observations(min_obs):
    with pytest.raises(FactorProxyError):
        analyze_factor_proxies(
            prices(np.array([0.01] * 80)),
            {"SPY": prices(np.array([0.01] * 80))},
            min_observations=min_obs,
        )


def test_nonpositive_prices_rejected():
    p = prices(np.array([0.01] * 80))
    p.iloc[20] = -2
    with pytest.raises(FactorProxyError, match="positive"):
        analyze_factor_proxies(p, {"SPY": prices(np.array([0.02] * 80))})


def test_disjoint_dates_rejected():
    p = prices(np.array([0.01] * 80))
    q = prices(np.array([0.02] * 80))
    q.index = q.index + pd.Timedelta(days=900)
    with pytest.raises(FactorProxyError, match="shared price"):
        analyze_factor_proxies(p, {"SPY": q})


def test_two_univariate_proxies_differ():
    x = np.array([-0.03, 0.04, 0.02, -0.01] * 20)
    r = analyze_factor_proxies(prices(x), {"A": prices(x), "B": prices(-x + 0.003)})
    assert r["A"].beta == pytest.approx(1)
    assert r["B"].beta == pytest.approx(-1)
    assert r["B"].r_squared == pytest.approx(1)
