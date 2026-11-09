"""V9.21: original-portfolio improvement and benchmark competitiveness differ."""
from pathlib import Path

import pytest
from portfolio_risk_ui.v921_evidence import (
    status_label, benchmark_status_label, average_objective_advantage,
)


def case(*, good=6, bad=2, spy=3, rsp=4, mean=.12, n=8):
    return {'status': 'evaluated', 'holdout': {
        'tests': n, 'better': good, 'worse': bad, 'similar': n-good-bad,
        'mean_primary_improvement': mean,
        'baseline_consistent': good >= .75*n and mean > 0,
        'benchmark_counts': {'SPY': spy, 'RSP': rsp},
        'benchmarks_consistent': spy >= .75*n and rsp >= .75*n,
    }}


def test_balanced_star_is_independent_of_benchmark_performance():
    report = case(good=6,bad=2,spy=3,rsp=4)
    assert status_label(report) == ('★', 'Consistent past improvements')
    assert benchmark_status_label(report) == 'No consistent benchmark advantage'
    # Legacy combined status can still be 'mixed' without overriding the star.
    report['holdout']['status'] = 'mixed'
    assert status_label(report)[0] == '★'


def test_benchmark_superiority_does_not_manufacture_an_improvement_star():
    report = case(good=2,bad=6,spy=8,rsp=8,mean=-.01)
    assert status_label(report)[0] != '★'
    assert benchmark_status_label(report) == 'Consistently ahead of both benchmarks'


def test_partial_benchmark_win_and_lack_of_positive_average():
    assert benchmark_status_label(case(spy=7,rsp=3)) == 'Consistently ahead of one benchmark'
    report = case(good=7,bad=1,spy=8,rsp=8,mean=-.002)
    assert status_label(report)[0] != '★'


def test_rounded_threshold_and_missing_evidence():
    report = case(good=5,bad=2,spy=6,rsp=6,mean=.09,n=7)
    assert status_label(report)[0] != '★'  # self 5/7 < ceil(.75*7) = 6
    assert benchmark_status_label(report) == 'Consistently ahead of both benchmarks'
    insufficient = {'status': 'evaluated', 'holdout': {
        'tests':2, 'better':2, 'worse':0, 'similar':0,
        'benchmark_counts': {'SPY':2,'RSP':2}, 'mean_primary_improvement':.1,
    }}
    assert status_label(insufficient)[0] == '○'
    assert benchmark_status_label(insufficient) == 'Not enough later tests'


def test_average_goal_advantage_uses_correct_units():
    h = case()['holdout']
    assert average_objective_advantage('balanced',h) == '+0.12 Sharpe points'
    assert average_objective_advantage('growth',h) == '+12.00 percentage points of annualized growth'
    assert average_objective_advantage('defensive',h) == '+12.00 percentage points less annual volatility'


def test_legacy_optimized_data_paths_and_user_visible_split_are_preserved():
    root = Path(__file__).resolve().parents[1]
    ui = (root/'portfolio_risk_ui'/'v918_three_views.py').read_text()
    backend = (root/'src'/'portfolio_risk'/'v918_three_strategies.py').read_text()
    assert 'show_three_results' in ui
    assert 'Did it improve your portfolio?' in ui
    assert 'How does it compare with SPY and RSP?' in ui
    assert "'portfolio_evidence': portfolio_evidence" in backend
    assert "'benchmark_evidence': benchmark_evidence" in backend
    assert 'benchmark_evidence_status' not in ui
    assert 'not statistical significance' in ui
