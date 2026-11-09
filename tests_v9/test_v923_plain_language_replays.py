"""V9.23: explain the same evidence without overstating retrospective gains."""
from pathlib import Path
import copy
import pytest

from portfolio_risk_ui.v922_result_explanations import (
    rating_text, benchmark_insights, portfolio_validation_explanation,
    goal_improvement, benchmark_validation_explanation,
)

ROOT = Path(__file__).resolve().parents[1]


def example():
    return {
        'weights': {'CCL': 0, 'AAPL': 1, 'MSFT': 0, 'SPY': 0, 'GLD': 0},
        'original_training_metrics': {'cagr': .0496, 'sharpe': .32,
                                      'annualized_volatility': .3815},
        'current_training_metrics': {'cagr': .3155, 'sharpe': 1.02,
                                     'annualized_volatility': .3207},
        'benchmarks_training_metrics': {
            'SPY': {'cagr': .1065, 'sharpe': .63, 'annualized_volatility': .1883},
            'RSP': {'cagr': .0997, 'sharpe': .58, 'annualized_volatility': .2005},
        },
        'holdout': {'tests': 8, 'better': 3, 'worse': 5,
                    'portfolio_evidence': 'not_supported',
                    'mean_primary_improvement': -.012,
                    'benchmark_counts': {'SPY': 3, 'RSP': 4}},
    }


def test_better_in_hindsight_can_fail_simulated_past_decisions():
    case = example()
    assert '26.59 percentage points higher' in goal_improvement('growth', case)['headline']
    assert '✕' in rating_text(case) and 'did not consistently hold up' in rating_text(case)
    explanation = portfolio_validation_explanation('growth', case)
    assert 'full historical record' in explanation
    assert 'earlier dates' in explanation
    assert 'An investment that wins over many years' in explanation
    assert 'not' in explanation


def test_growth_benchmark_advantages_are_direct_and_signed():
    case = example()
    text, later = benchmark_insights('growth', case)
    assert 'SPY: 20.90 percentage points more annual growth' in text
    assert 'RSP: 21.58 percentage points more annual growth' in text
    assert '17.00% vs' not in text
    assert 'did not consistently beat SPY or RSP' in later
    case['current_training_metrics']['cagr'] = .06
    text, _ = benchmark_insights('growth', case)
    assert 'SPY: 4.65 percentage points less annual growth' in text
    assert 'RSP: 3.97 percentage points less annual growth' in text


def test_balanced_and_defensive_deltas_have_correct_directions():
    case = example()
    value, _ = benchmark_insights('balanced', case)
    assert 'SPY: 0.39 higher Sharpe ratio' in value
    assert 'RSP: 0.44 higher Sharpe ratio' in value
    case['current_training_metrics']['sharpe'] = .5
    value, _ = benchmark_insights('balanced', case)
    assert 'SPY: 0.13 lower Sharpe ratio' in value
    assert 'RSP: 0.08 lower Sharpe ratio' in value
    case['current_training_metrics']['annualized_volatility'] = .1346
    value, _ = benchmark_insights('defensive', case)
    assert 'SPY: 5.37 percentage points less annual volatility' in value
    assert 'RSP: 6.59 percentage points less annual volatility' in value
    case['current_training_metrics']['annualized_volatility'] = .25
    value, _ = benchmark_insights('defensive', case)
    assert 'SPY: 6.17 percentage points more annual volatility' in value
    assert 'RSP: 4.95 percentage points more annual volatility' in value


def test_earned_star_and_mixed_stay_historically_truthful():
    case = example()
    case['holdout'].update(portfolio_evidence='consistent', better=6, worse=2,
                           benchmark_counts={'SPY': 6, 'RSP': 6})
    assert rating_text(case).startswith('★')
    assert 'usually improved your portfolio' in portfolio_validation_explanation('growth', case)
    assert 'usually beat both benchmarks' in benchmark_validation_explanation(case)
    case['holdout'].update(portfolio_evidence='mixed', better=5, worse=3,
                           benchmark_counts={'SPY': 2, 'RSP': 6})
    assert rating_text(case).startswith('○')
    assert 'not consistent' in portfolio_validation_explanation('growth', case)
    assert 'usually beat RSP' in benchmark_validation_explanation(case)


def test_live_view_explains_past_replays_not_predicted_future():
    source = (ROOT/'portfolio_risk_ui'/'v918_three_views.py').read_text()
    assert 'This is a replay of the' in source and 'not a simulation of what will happen' in source
    assert 'best historical fit performed (with hindsight)' in source
    assert 'Compared with each benchmark over the full history' in source
    assert 'Technical details: individual historical tests' in source
    assert 'benchmark_insights(goal, item)' in source
    optimizer = (ROOT/'src'/'portfolio_risk'/'v918_three_strategies.py').read_text()
    assert "'primary_improvement': score" in optimizer
