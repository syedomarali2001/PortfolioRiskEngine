from pathlib import Path
from portfolio_risk_ui.v922_result_explanations import (
    goal_improvement, benchmark_insights, portfolio_validation_explanation,
    rating_text, benchmark_validation_explanation,
)

ROOT=Path(__file__).resolve().parents[1]


def fixture(goal='growth'):
    return {'status':'evaluated',
        'original_training_metrics': {'cagr':.10,'sharpe':.5,'annualized_volatility':.30},
        'current_training_metrics': {'cagr':.17,'sharpe':1.0,'annualized_volatility':.15},
        'benchmarks_training_metrics':{
            'SPY':{'cagr':.13,'sharpe':.8,'annualized_volatility':.20},
            'RSP':{'cagr':.11,'sharpe':.65,'annualized_volatility':.19}},
        'holdout':{'tests':8,'better':6,'worse':2,'portfolio_evidence':'consistent',
                   'benchmark_counts':{'SPY':3,'RSP':4},'mean_primary_improvement':.021}}


def test_goal_effects_explain_real_numbers_and_unit_directions():
    case=fixture()
    assert '7.00 percentage points higher' in goal_improvement('growth',case)['headline']
    assert '0.50 higher' in goal_improvement('balanced',case)['headline']
    defensive=goal_improvement('defensive',case)['headline']
    assert '15.00 percentage points lower' in defensive
    assert '(50% reduction)' in defensive
    assert 'from 30.00% to 15.00%' in goal_improvement('growth',case)['tradeoff']
    bench,_=benchmark_insights('growth',case)
    assert 'SPY: 4.00 percentage points more annual growth' in bench
    assert 'RSP: 6.00 percentage points more annual growth' in bench
    bench,_=benchmark_insights('defensive',case)
    assert 'SPY: 5.00 percentage points less annual volatility' in bench
    assert 'SPY: 0.20 higher Sharpe ratio' in benchmark_insights('balanced',case)[0]


def test_true_holdout_status_is_kept_separate_from_training_outcomes():
    case=fixture()
    assert rating_text(case)=='★ Usually improved in historical replays'
    assert 'usually improved' in portfolio_validation_explanation('growth',case)
    assert 'did not consistently beat SPY or RSP' in benchmark_validation_explanation(case)
    case['holdout']['portfolio_evidence']='mixed'
    case['holdout']['mean_primary_improvement']=-.03
    assert 'not consistent' in portfolio_validation_explanation('balanced',case).lower()
    assert 'lost in others' in portfolio_validation_explanation('balanced',case)
    case['holdout']['benchmark_counts']={'SPY':7,'RSP':3}
    assert 'usually beat SPY' in benchmark_validation_explanation(case)


def test_sample_selector_is_short_and_strategy_ui_is_explanatory():
    app=(ROOT/'portfolio_risk_ui'/'app.py').read_text()
    part=app[app.index('def _render_sample_portfolios():'):app.index('def _render_portfolio_list():')]
    assert 'Choose a portfolio' in part and 'st.write(f"**{demo.purpose}**")' in part
    assert 'What a good test should reveal' not in part
    assert 'Illustrative entries' not in part
    assert 'Validation:' not in part
    view=(ROOT/'portfolio_risk_ui'/'v918_three_views.py').read_text()
    assert 'How to read results' not in view
    assert 'Did it improve your portfolio?' in view
    assert 'How does it compare with SPY and RSP?' in view
    assert 'Technical details: individual historical tests' in view
    assert 'goal_improvement(goal, item)' in view
    assert 'benchmark_insights(goal, item)' in view
