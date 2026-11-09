"""Human explanations of retrospective portfolios and historical replays.

Presentation only: never change security data, optimization, or evidence grades.
All benchmark deltas describe FULL-HISTORY in-sample metrics; replay conclusions
come exclusively from chronological holdout evidence.
"""
from __future__ import annotations

from math import isfinite
from typing import Mapping

KEYS = {'growth': 'cagr', 'balanced': 'sharpe', 'defensive': 'annualized_volatility'}


def _metric(data: Mapping, key: str) -> float:
    result = float(data[key])
    if not isfinite(result):
        raise ValueError(f'Invalid historical metric: {key}')
    return result


def _relative_change(before: float, after: float) -> float | None:
    return 100 * (after - before) / abs(before) if abs(before) > 1e-12 else None


def _pct_point_change(before: float, after: float) -> str:
    delta = 100 * (after - before)
    return f'{abs(delta):.2f} percentage points'


def goal_improvement(goal: str, report: Mapping) -> dict[str, str]:
    """Full-history hindsight effect, explicitly NOT a later-period validation."""
    key = KEYS[goal]
    original, tested = report['original_training_metrics'], report['current_training_metrics']
    a, b = _metric(original, key), _metric(tested, key)
    if goal == 'growth':
        verb = 'higher' if b > a else 'lower' if b < a else 'unchanged'
        headline = (f'Annual growth: {b:.2%} vs {a:.2%}. '
                    f'{_pct_point_change(a, b)} {verb}')
    elif goal == 'balanced':
        verb = 'higher' if b > a else 'lower' if b < a else 'unchanged'
        headline = f'Sharpe ratio: {b:.2f} vs {a:.2f}. {abs(b-a):.2f} {verb}'
    else:
        verb = 'lower' if b < a else 'higher' if b > a else 'unchanged'
        change = _relative_change(a, b)
        relative = (f' ({abs(change):.0f}% {"reduction" if b < a else "increase"})'
                    if change is not None else '')
        headline = (f'Volatility: {b:.2%} vs {a:.2%}. '
                    f'{_pct_point_change(a, b)} {verb}{relative}')
    growth_a, growth_b = _metric(original, 'cagr'), _metric(tested, 'cagr')
    vol_a = _metric(original, 'annualized_volatility')
    vol_b = _metric(tested, 'annualized_volatility')
    if goal == 'growth':
        tradeoff = ('Volatility increased' if vol_b > vol_a + 1e-8 else
                    'Volatility decreased' if vol_b < vol_a - 1e-8 else 'Volatility unchanged')
        tradeoff += f' from {vol_a:.2%} to {vol_b:.2%}.'
    else:
        tradeoff = (f'Annual growth: {growth_a:.2%} → {growth_b:.2%}. '
                    f'Volatility: {vol_a:.2%} → {vol_b:.2%}.')
    return {'headline': headline, 'tradeoff': tradeoff}


def _benchmark_comparison(goal: str, proposed: float, reference: float, ticker: str) -> str:
    """Simple signed ADVANTAGE; for defensive, lower volatility is better."""
    if goal == 'balanced':
        advantage = proposed - reference
        if abs(advantage) < 1e-10:
            return f'{ticker}: essentially the same historical Sharpe ratio'
        direction = 'higher' if advantage > 0 else 'lower'
        return f'{ticker}: {abs(advantage):.2f} {direction} Sharpe ratio'
    advantage = (reference - proposed) if goal == 'defensive' else (proposed - reference)
    unit = 'annual volatility' if goal == 'defensive' else 'annual growth'
    if abs(advantage) < 1e-10:
        return f'{ticker}: essentially the same historical {unit}'
    direction = ('less' if advantage > 0 else 'more') if goal == 'defensive' else (
        'more' if advantage > 0 else 'less')
    return f'{ticker}: {abs(advantage)*100:.2f} percentage points {direction} {unit}'


def benchmark_insights(goal: str, report: Mapping) -> tuple[str, str]:
    """Signed metric differences for full history; independent replay judgment."""
    key = KEYS[goal]
    proposed = _metric(report['current_training_metrics'], key)
    benchmarks = report['benchmarks_training_metrics']
    explanation = '  |  '.join(
        _benchmark_comparison(goal, proposed, _metric(benchmarks[ticker], key), ticker)
        for ticker in ('SPY', 'RSP')
    )
    return explanation, benchmark_validation_explanation(report)


def portfolio_validation_explanation(goal: str, report: Mapping) -> str:
    """Explain why the hindsight result and real-history walk-forward grade differ."""
    hold = report.get('holdout') or {}
    if int(hold.get('tests', 0)) < 3:
        return 'There were too few historical replays to judge whether the improvement held up.'
    state = hold.get('portfolio_evidence', '')
    if state == 'consistent':
        return ('When we repeatedly went back to earlier dates, picked weights using only the prices '
                'known then, and checked what happened afterward, this approach usually improved '
                'your portfolio on its selected goal. That is encouraging historical evidence, '
                'not a forecast.')
    if state == 'mixed':
        return ('This allocation looks better with the benefit of the full historical record. '
                'But when we simulated decisions at earlier dates without knowing '
                'what would happen next, the approach improved in some periods and lost in others. '
                'The advantage was not consistent.')
    if state == 'not_supported':
        return ('The large improvement above uses the best fit found after seeing the full '
                'historical record. When we simulated decisions using only prices available at '
                'earlier dates, the approach did not show a dependable advantage. '
                'An investment that wins over many years can still lose to another investment '
                'during shorter periods. This explanation applies to any portfolio.')
    return 'The historical replays were inconclusive about whether this approach improves your portfolio.'


def benchmark_validation_explanation(report: Mapping) -> str:
    """Differentiate full-history benchmark lead from replayed competitiveness."""
    hold = report.get('holdout') or {}
    n = int(hold.get('tests', 0))
    if n < 3:
        return 'There were too few historical replays to assess benchmark competitiveness.'
    counts = hold.get('benchmark_counts') or {}
    threshold = (3*n + 3)//4
    spy = int(counts.get('SPY', 0)) >= threshold
    rsp = int(counts.get('RSP', 0)) >= threshold
    if spy and rsp:
        return 'When we replayed past decisions using only information available at the time, this approach usually beat both benchmarks on this goal.'
    if spy or rsp:
        return f'The historical replays usually beat {"SPY" if spy else "RSP"} on this goal, but not both benchmarks.'
    return ('The optimized allocation may look better over the full history, but the approach '
            'did not consistently beat SPY or RSP when we replayed earlier decisions.')


def rating_text(report: Mapping) -> str:
    """Labels refer ONLY to repeatability in later historical replays."""
    state = (report.get('holdout') or {}).get('portfolio_evidence', 'insufficient')
    return {
        'consistent': '★ Usually improved in historical replays',
        'mixed': '○ Improvement was inconsistent',
        'not_supported': '✕ Improvement did not consistently hold up',
        'insufficient': 'Not enough historical tests',
    }.get(state, 'Not enough historical tests')


KEY_METRICS = (
    ('cagr', 'Annual growth', 'percentage points', True),
    ('sharpe', 'Sharpe ratio', 'ratio points', True),
    ('annualized_volatility', 'Volatility', 'percentage points', False),
)


def benchmark_three_metric_summary(report: Mapping) -> list[dict]:
    """Compare the optimized portfolio with each benchmark on 3 useful measures.

    Positive means beneficial: more growth/Sharpe or less volatility. All figures
    refer to the same full historical period. Later evidence remains separate.
    """
    proposed = report['current_training_metrics']
    benchmarks = report['benchmarks_training_metrics']
    answer = []
    for ticker in ('SPY', 'RSP'):
        lines = []
        for key, label, units, higher_is_better in KEY_METRICS:
            candidate = _metric(proposed, key)
            benchmark = _metric(benchmarks[ticker], key)
            delta = candidate - benchmark
            meaningful = delta if higher_is_better else -delta
            magnitude = abs(delta) if key == 'sharpe' else 100 * abs(delta)
            # Avoid calling a rounded 0.00 advantage or disadvantage.
            if round(magnitude, 2) == 0:
                status, phrase = 'similar', f'about the same as {ticker}'
            else:
                status = 'better' if meaningful > 0 else 'worse'
                direction = ('higher' if delta > 0 else 'lower')
                phrase = f'{magnitude:.2f} {units} {direction} than {ticker} ({status})'
            lines.append({'label': label, 'status': status, 'description': phrase})
        wins = sum(line['status'] == 'better' for line in lines)
        ties = sum(line['status'] == 'similar' for line in lines)
        summary = f'Better on {wins} of 3 key measures'
        if ties:
            summary += f', similar on {ties}'
        answer.append({'ticker': ticker, 'summary': summary, 'lines': lines})
    return answer
