"""V9.24: no ticker-specific prose, useful benchmark summaries and honest colors."""
from pathlib import Path

from portfolio_risk_ui.v922_result_explanations import (
    benchmark_three_metric_summary, portfolio_validation_explanation,
    goal_improvement,
)
import pandas as pd
from portfolio_risk_ui.v924_metric_colors import color_proposed_metrics

ROOT = Path(__file__).resolve().parents[1]


def report():
    return {
        'weights': {'VZ': 0, 'COST': 0.55, 'MSFT': 0.45},
        'original_training_metrics': {
            'cagr': 0.07, 'sharpe': 0.6, 'annualized_volatility': 0.2,
            'drawdown': 0.4, 'sortino': 0.8, 'es95': 0.04,
        },
        'current_training_metrics': {
            'cagr': 0.12, 'sharpe': 0.5, 'annualized_volatility': 0.15,
            'drawdown': 0.5, 'sortino': 1.2, 'es95': 0.03,
        },
        'benchmarks_training_metrics': {
            'SPY': {'cagr': 0.10, 'sharpe': 0.70, 'annualized_volatility': 0.14},
            'RSP': {'cagr': 0.08, 'sharpe': 0.60, 'annualized_volatility': 0.25},
        },
        'holdout': {'tests': 8, 'portfolio_evidence': 'not_supported'},
    }


def test_benchmark_key_measure_comparisons_are_directional():
    spy, rsp = benchmark_three_metric_summary(report())
    assert spy['summary'] == 'Better on 1 of 3 key measures'
    assert [x['status'] for x in spy['lines']] == ['better', 'worse', 'worse']
    assert spy['lines'][0]['description'] == '2.00 percentage points higher than SPY (better)'
    assert spy['lines'][1]['description'] == '0.20 ratio points lower than SPY (worse)'
    assert spy['lines'][2]['description'] == '1.00 percentage points higher than SPY (worse)'
    assert rsp['summary'] == 'Better on 2 of 3 key measures'
    assert [x['status'] for x in rsp['lines']] == ['better', 'worse', 'better']
    assert rsp['lines'][2]['description'] == '10.00 percentage points lower than RSP (better)'


def test_ties_do_not_show_as_zero_size_better_or_worse():
    data = report()
    data['current_training_metrics'].update(cagr=0.10, sharpe=0.70,
                                             annualized_volatility=0.14)
    spy = benchmark_three_metric_summary(data)[0]
    assert spy['summary'] == 'Better on 0 of 3 key measures, similar on 3'
    assert all(x['status'] == 'similar' for x in spy['lines'])
    assert all('about the same' in x['description'] for x in spy['lines'])


def test_proposed_metric_styles_are_directionally_correct_and_only_in_proposed_column():
    metrics = ('cagr', 'sharpe', 'annualized_volatility', 'drawdown', 'sortino', 'es95')
    frame = pd.DataFrame({'Measure': list(metrics),
                          'Your portfolio': ['original']*6,
                          'Proposed': ['proposed']*6,
                          'SPY': ['spy']*6, 'RSP': ['rsp']*6})
    styler = color_proposed_metrics(frame, report(), metrics)
    ctx = styler._compute().ctx
    # Columns: measure 0; original 1; proposed 2; SPY 3; RSP 4.
    for i in range(6):
        assert not ctx.get((i, 0)), 'no colored metric labels'
        assert not ctx.get((i, 1)), 'no coloring original values'
        assert not ctx.get((i, 3)), 'no coloring benchmark values'
        assert not ctx.get((i, 4)), 'no coloring benchmark values'
    for good in (0, 2, 4, 5):
        assert any(('color', '#86EFAC') == p for p in ctx[(good, 2)])
    for bad in (1, 3):
        assert any(('color', '#FDA4AF') == p for p in ctx[(bad, 2)])
    assert styler.data.iloc[0]['Proposed'] == 'proposed'


def test_portfolio_specific_ccl_story_replaced_with_general_evidence():
    r = report()
    explanation = portfolio_validation_explanation('growth', r)
    assert 'CCL' not in explanation and 'AAPL' not in explanation
    assert 'An investment that wins over many years' in explanation
    assert 'earlier dates' in explanation
    assert '—' not in explanation
    assert '—' not in goal_improvement('growth', r)['headline']


def test_user_visible_strategy_modules_do_not_include_em_dashes():
    for name in ('v922_result_explanations.py', 'v918_three_views.py', 'v920_presentation.py'):
        assert '—' not in (ROOT/'portfolio_risk_ui'/name).read_text(encoding="utf-8")
    source = (ROOT/'portfolio_risk_ui/app.py').read_text(encoding="utf-8")
    assert 'Optional investment limits —' not in source
    assert 'styled_comparisons(item)' in (ROOT/'portfolio_risk_ui/v918_three_views.py').read_text(encoding="utf-8")
    assert 'benchmark_three_metric_summary(item)' in (ROOT/'portfolio_risk_ui/v918_three_views.py').read_text(encoding="utf-8")
