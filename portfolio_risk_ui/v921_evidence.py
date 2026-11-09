"""Independent, historical-only portfolio and benchmark evidence labels.

Neither status is investment advice or a statistical-significance test.
"""
from __future__ import annotations
from typing import Mapping

def status_label(report: Mapping) -> tuple[str, str]:
    """Grade consistent out-of-sample improvement vs the user's OWN portfolio.

    SPY and RSP evidence has an independent grade. A strategy can be helpful
    relative to current weights without beating either diversified benchmark.
    """
    if report.get('status') != 'evaluated':
        return '○', 'Unavailable'
    h = report.get('holdout', {})
    status = h.get('portfolio_evidence')
    if status is None:  # older reports can still be opened safely
        if h.get('tests', 0) < 3:
            status = 'insufficient'
        elif h.get('baseline_consistent'):
            status = 'consistent'
        else:
            status = 'mixed' if h.get('better', 0) > h.get('worse', 0) else 'not_supported'
    return {
        'consistent': ('★', 'Consistent past improvements'),
        'mixed': ('○', 'Mixed historical evidence'),
        'not_supported': ('✕', 'Not supported by past tests'),
        'insufficient': ('○', 'Limited historical evidence'),
    }.get(status, ('○', 'No verified result'))


def benchmark_status_label(report: Mapping) -> str:
    """Separate benchmark judgment; NEVER a requirement for own-portfolio ★."""
    if report.get('status') != 'evaluated':
        return 'Unavailable'
    h = report.get('holdout', {})
    tests = h.get('tests', 0)
    if tests < 3:
        return 'Not enough later tests'
    counts = h.get('benchmark_counts', {})
    threshold = (3*tests + 3)//4  # ceil(75% * number of tests)
    wins = sum(int(counts.get(sym, 0)) >= threshold for sym in ('SPY', 'RSP'))
    if wins == 2:
        return 'Consistently ahead of both benchmarks'
    if wins == 1:
        return 'Consistently ahead of one benchmark'
    return 'No consistent benchmark advantage'


def average_objective_advantage(goal: str, holdout: Mapping) -> str:
    """Average goal improvement over separate later windows, not a p-value."""
    v = float(holdout.get('mean_primary_improvement', 0.))
    if goal == 'balanced':
        return f'{v:+.2f} Sharpe points'
    if goal == 'growth':
        return f'{v*100:+.2f} percentage points of annualized growth'
    return f'{v*100:+.2f} percentage points less annual volatility'

