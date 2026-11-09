"""V9.20 regression: clear target-weight settings, ending drift, sample cases, editor seeds."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v918_three_strategies import ending_drift_weights
from portfolio_risk.v83_robustness import _simulated_returns
from portfolio_risk_ui.example_portfolios import EXAMPLES, example_holdings, get_example
from portfolio_risk_ui.v920_presentation import (
    REBALANCE_CHOICES, custom_start_weights, editable_percentages, rebalance_explanation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_quarterly_target_rebalance_default_and_exact_constant_weight_option():
    assert next(iter(REBALANCE_CHOICES.values())) == 'quarterly'
    assert list(REBALANCE_CHOICES.values()) == ['quarterly','annually','monthly','daily','buy_hold']
    assert 'drift' in rebalance_explanation('quarterly')
    assert 'NOT stay fixed' in rebalance_explanation('buy_hold')
    assert 'each trading day' in rebalance_explanation('daily')
    text = (ROOT/'portfolio_risk_ui/app.py').read_text()
    assert 'key="v920_cadence"' in text
    assert 'REBALANCE_CHOICES[cadence]' in text
    assert "key='v920_include_custom'" in text
    assert 'Run Overview or Strategy Lab first' not in text


def test_buy_hold_end_weights_drift_and_quarterly_resets_on_first_new_quarter():
    index = pd.to_datetime(['2026-03-30','2026-03-31','2026-04-01'])
    data = pd.DataFrame({'A': [.10,.10,.10], 'B':[0,0,0]},index=index)
    start = np.array([.5,.5])
    ending_buy_hold = ending_drift_weights(data,start,'buy_hold')
    expected_buy_hold = np.array([.5*1.1**3,.5]); expected_buy_hold/=expected_buy_hold.sum()
    np.testing.assert_allclose(ending_buy_hold, expected_buy_hold)
    assert ending_buy_hold[0] > .55
    # April 1 is a new calendar quarter, so rebalancing resets at day start.
    ending_quarterly = ending_drift_weights(data,start,'quarterly')
    np.testing.assert_allclose(ending_quarterly, [1.1/2.1,1/2.1])
    ending_daily = ending_drift_weights(data,start,'daily')
    np.testing.assert_allclose(ending_daily, [1.1/2.1,1/2.1])


@pytest.mark.parametrize('cadence',list(REBALANCE_CHOICES.values()))
def test_ending_allocation_is_finite_and_sums_to_one(cadence):
    idx=pd.bdate_range('2024-01-02',periods=400)
    rng=np.random.default_rng(920)
    returns=pd.DataFrame(rng.normal([.0005,.0002,.0008],[.015,.004,.018],size=(400,3)),index=idx,columns=['A','B','C'])
    target=np.array([.4,.35,.25])
    last=ending_drift_weights(returns,target,cadence)
    assert np.isfinite(last).all()
    assert (last >= 0).all()
    assert last.sum()==pytest.approx(1)
    # Both simulator and ending report must use valid identical cadence names.
    series,metadata=_simulated_returns(returns,target,rebalance=cadence,cost_bps=10.,starting_weights=target)
    assert len(series)==len(returns)
    assert metadata['cadence']==cadence


def test_custom_weights_can_be_constructed_without_running_full_report():
    holdings=[{'symbol':'AAPL','market_value':6000,'currency':'USD'},
              {'symbol':'SPY','market_value':3000,'currency':'USD'},
              {'symbol':'GLD','market_value':1000,'currency':'USD'}]
    assert custom_start_weights(holdings,'USD') == pytest.approx({'AAPL':.6,'SPY':.3,'GLD':.1})
    assert editable_percentages({'AAPL':.333333333,'SPY':.333333333,'GLD':.333333334}) == {
        'AAPL':33.33,'SPY':33.33,'GLD':33.34}
    assert editable_percentages({'AAPL':1.,'SPY':0.})=={'AAPL':100.,'SPY':0.}
    with pytest.raises(ValueError):
        editable_percentages({'AAPL':.7,'SPY':.2})


def test_mixed_currency_custom_weights_require_real_fx_rate():
    holdings=[{'symbol':'AAPL','market_value':100,'currency':'USD'},
              {'symbol':'XEQT.TO','market_value':100,'currency':'CAD'}]
    with pytest.raises(ValueError,match='FX'):
        custom_start_weights(holdings,'USD')
    result=custom_start_weights(holdings,'USD',usd_cad='1.25')
    assert result==pytest.approx({'AAPL':100/180,'XEQT.TO':80/180})


def test_all_sample_scenarios_are_importable_and_have_distinct_test_questions():
    assert len(EXAMPLES)==14
    keys={item.key for item in EXAMPLES}
    assert {'starter','ccl80','nvda90','lagging_consumer','momentum_compare','bond_shock','index_overlap','diversified','tech_only','mixed_fx','dividends','short_history','one_holding','max_holdings'} == keys
    for case in EXAMPLES:
        rows=example_holdings(case.key)
        assert len({r['symbol'] for r in rows}) == len(rows)
        assert sum(r['weight'] for r in rows) == pytest.approx(100.)
        assert case.inspect and case.purpose and case.objective_to_test
        assert (ROOT/'examples/demo_portfolios'/f'{case.key}.csv').exists()
    assert get_example('lagging_consumer').positions[0][0]=='SBUX'
    assert get_example('bond_shock').positions[0][0]=='TLT'
    assert any(s[0]=='SPMO' for s in get_example('momentum_compare').positions)


def test_manual_comparison_uses_chosen_cadence_not_legacy_daily_constant_mix():
    from portfolio_risk.v8_insights import analyze_insights
    from portfolio_risk.v83_robustness import _simulated_returns
    from portfolio_risk.v8_insights import _stats
    dates = pd.bdate_range('2022-01-03', periods=525)
    rng = np.random.default_rng(92002)
    r_a = rng.normal(.0008,.016,len(dates)-1)
    r_b = rng.normal(.0004,.007,len(dates)-1)
    r_spy = rng.normal(.0006,.011,len(dates)-1)
    r_rsp = rng.normal(.0005,.011,len(dates)-1)
    asset = pd.DataFrame({'A':r_a,'B':r_b},index=dates[1:])
    close = lambda returns: pd.Series(np.r_[100.,100*np.cumprod(1+returns)],index=dates)
    prices = {'A':close(r_a),'B':close(r_b)}
    benchmarks = {'SPY':close(r_spy),'RSP':close(r_rsp)}
    custom = {'A':.9,'B':.1}
    original = {'A':.5,'B':.5}
    report = analyze_insights(prices,original,benchmarks,base_currency='USD',
        scenarios={'GUI_WhatIf':custom},three_strategies=True,
        robustness={'max_weight':1.,'rebalance':'quarterly','cost_bps':25.,
                    'train_periods':252,'test_periods':126,'step_periods':126},
        mc_paths=20,mc_horizon=30)
    outcome = report['user_scenarios']['GUI_WhatIf']
    stream,_ = _simulated_returns(asset,np.array([.9,.1]),rebalance='quarterly',
                                  cost_bps=25.,starting_weights=np.array([.5,.5]))
    expected = _stats(stream,dates,0.)
    assert outcome['metrics']['cagr']==pytest.approx(expected['cagr'],abs=1e-12)
    assert outcome['metrics']['sharpe']==pytest.approx(expected['sharpe'],abs=1e-12)
    assert 'quarterly' in outcome['method']
    orig_stats=report['three_strategy_analysis']['strategies']['growth']['original_training_metrics']
    assert outcome['vs_input']['cagr']['difference']==pytest.approx(expected['cagr']-orig_stats['cagr'])
