"""V9.16: training-only objective fidelity and source-data audit.

Synthetic data, no claims about actual CCL, AAPL, MSFT or gold returns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v83_robustness import _score_training, _validated_rules, analyze_robustness

NAMES = ['CCL','AAPL','SPY','MSFT','GLD']
REF = np.array([.80,.07,.06,.04,.03])
RULES = {'max_weight':.4,'train_periods':252,'test_periods':126,
         'step_periods':126,'rebalance':'quarterly','cost_bps':0.0}


def example_data(n=504):
    rng = np.random.default_rng(916)
    dates = pd.bdate_range('2020-01-02', periods=n+1)
    mean = np.array([-.00075,.00100,.00025,.00090,.00005])
    raw = rng.normal(0,.0015,size=(n,len(NAMES))) + mean
    return pd.DataFrame(raw,index=dates[1:],columns=NAMES), dates


def test_growth_goal_favors_the_actual_strong_training_assets():
    data, dates = example_data()
    constraints = _validated_rules(NAMES, REF, RULES)
    candidates, stats, scores = _score_training(data.iloc[:252], dates[:253], REF,
                                                constraints, 0, {'cagr':1.})
    winner = max(scores, key=scores.get)
    weights = candidates[winner]
    assert weights.sum() == pytest.approx(1.0, abs=1e-7)
    assert weights[1] >= .35 and weights[3] >= .35  # AAPL and MSFT
    assert weights[0] < .05 and weights[4] < .05  # CCL and GLD
    assert stats[winner]['cagr'] == pytest.approx(max(item['cagr'] for item in stats.values()))


def test_low_risk_and_growth_are_different_objectives():
    data, dates = example_data()
    rules = _validated_rules(NAMES, REF, RULES)
    growth, _, scores_g = _score_training(data.iloc[:252], dates[:253], REF, rules, 0, {'cagr':1.})
    risk, _, scores_v = _score_training(data.iloc[:252], dates[:253], REF, rules, 0, {'volatility':1.})
    wg = growth[max(scores_g, key=scores_g.get)]
    wv = risk[max(scores_v, key=scores_v.get)]
    assert not np.allclose(wg, wv, atol=.02)


def test_asset_audit_exposes_training_vs_later_reversal_without_leakage():
    data, dates = example_data()
    # CCL had weak history but rallies during the last later test window.
    altered = data.copy()
    altered.iloc[378:, altered.columns.get_loc('CCL')] = .004
    earlier = analyze_robustness(data,dates,REF,priorities={'cagr':1.},rules=RULES)
    later = analyze_robustness(altered,dates,REF,priorities={'cagr':1.},rules=RULES)
    assert later['status'] == 'evaluated'
    latest = later['latest_asset_audit']
    assert set(latest) == set(NAMES)
    assert latest['CCL']['training_cagr'] < latest['AAPL']['training_cagr']
    assert latest['CCL']['holdout_total_return'] > latest['AAPL']['holdout_total_return']
    # The deliberately changed holdout cannot influence the selected weights.
    assert later['folds'][-1]['weights'] == pytest.approx(earlier['folds'][-1]['weights'])


def test_status_cards_and_examples_use_the_requested_clean_layout():
    from pathlib import Path
    app = (Path(__file__).resolve().parents[1] / 'portfolio_risk_ui' / 'app.py').read_text()
    builder = app.split('def page_builder():',1)[1].split('def _render_portfolio_list():',1)[0]
    assert builder.index('_render_portfolio_list()') < builder.index('_render_sample_portfolios()')
    assert "with st.expander(\"Sample portfolios for testing\"" in app
    assert 'load_original_starter' not in app
    assert 'three_strategy_analysis' in (Path(__file__).resolve().parents[1] / 'portfolio_risk_ui' / 'v918_three_views.py').read_text(encoding='utf-8')
    assert 'st.columns(3)' in app
    assert 'latest_asset_audit' in app
