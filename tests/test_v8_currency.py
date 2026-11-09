from datetime import date
import pandas as pd
import numpy as np
import pytest
from portfolio_risk.v8_currency import compare_reporting_currencies
from portfolio_risk.v8_insights import InsightError
def fixture(n=350):
    rng=np.random.default_rng(44)
    dates=pd.bdate_range('2023-01-02',periods=n)
    def prices():return pd.Series(100*np.cumprod(1+rng.normal(.0003,.01,n)),index=dates)
    return {'AAA':prices(),'BBB':prices()},{'AAA':.6,'BBB':.4},{'SPY':prices(),'RSP':prices()}


def test_currency_comparison_has_exactly_same_dates_and_changes_only_fx_path():
    assets,weights,bench=fixture(n=350)
    fx=pd.Series(np.linspace(1.3,1.5,350),index=assets['AAA'].index)
    doc=compare_reporting_currencies(assets,weights,bench,fx,base_currency='USD')
    assert set(doc['reporting_currency_metrics'])=={'CAD','USD'}
    assert doc['return_periods']==349
    usd=doc['reporting_currency_metrics']['USD']['portfolio']['cagr']
    cad=doc['reporting_currency_metrics']['CAD']['portfolio']['cagr']
    assert usd!=pytest.approx(cad)
    assert doc['start']==str(assets['AAA'].index[0].date())


def test_flat_fx_produces_identical_risk_and_return_metrics():
    a,w,b=fixture(n=340)
    flat=pd.Series(1.3,index=a['AAA'].index)
    out=compare_reporting_currencies(a,w,b,flat,base_currency='CAD')
    for name in ('portfolio','SPY','RSP'):
        for metric in ('cagr','annualized_volatility','maximum_drawdown'):
            # _stats uses positive drawdown depth rather than negative maximum drawdown
            key='drawdown' if metric=='maximum_drawdown' else metric
            assert out['reporting_currency_metrics']['CAD'][name][key]==pytest.approx(
                out['reporting_currency_metrics']['USD'][name][key])


def test_missing_fx_dates_are_not_filled_beyond_limit():
    a,w,b=fixture(n=340)
    bad=pd.Series([1.3,1.3],index=pd.to_datetime(['2023-01-02','2023-01-03']))
    with pytest.raises((ValueError,InsightError)):
        compare_reporting_currencies(a,w,b,bad,base_currency='USD',max_fx_age_days=2)
