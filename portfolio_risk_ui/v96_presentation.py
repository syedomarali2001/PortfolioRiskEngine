"""V9.6 presentation helpers. Financial measures and classifications come from V8.4."""
from __future__ import annotations

import datetime as dt
import math


def historical_window_text(report: dict) -> str:
    """Show *actual* first/last common price dates, never infer an inception date."""
    start = report.get('first_date')
    end = report.get('last_date')
    try:
        first = dt.date.fromisoformat(str(start))
        last = dt.date.fromisoformat(str(end))
        if last < first:
            return 'Historical dates unavailable'
    except (TypeError, ValueError):
        return 'Historical dates unavailable'
    return f'{first:%b} {first.day}, {first.year} to {last:%b} {last.day}, {last.year}'


HISTORY_NOTE = (
    'This starts on the earliest date with usable prices for **every selected investment** '
    'and the comparison ETFs. A newer ETF can shorten the period for the whole portfolio. '
    'It does not mean you bought the investments on that date.'
)

_METRIC_DIRECTION = {
    'cagr': (
        'Average yearly growth was higher than the comparison.',
        'Average yearly growth was lower than the comparison.',
        'Average yearly growth was close to the comparison.'),
    'volatility': (
        'Your returns fluctuated less, giving a smoother historical ride.',
        'Your returns fluctuated more, making the portfolio less steady.',
        'The amount of historical return fluctuation was broadly similar.'),
    'drawdown': (
        'The largest fall from a previous high was smaller.',
        'The largest fall from a previous high was deeper.',
        'The largest historical peak-to-trough fall was broadly similar.'),
    'sharpe': (
        'You received more historical return per unit of total volatility.',
        'You received less historical return per unit of total volatility.',
        'Return relative to total historical volatility was similar.'),
    'sortino': (
        'You received more historical return relative to downside fluctuations.',
        'You received less historical return relative to downside fluctuations.',
        'Return relative to downside fluctuations was similar.'),
    'es95': (
        'Average losses on the worst 5% of trading days were smaller.',
        'Average losses on the worst 5% of trading days were larger.',
        'Average losses on the worst 5% of trading days were similar.'),
}


METRIC_NAMES = {
    'cagr': 'Annual growth (CAGR)',
    'volatility': 'Volatility',
    'annualized_volatility': 'Volatility',
    'drawdown': 'Maximum drawdown',
    'sharpe': 'Sharpe ratio',
    'sortino': 'Sortino ratio',
    'es95': 'Expected shortfall (95%)',
}


def finding_message(metric: str, sentiment: str, difference: str | None = None) -> str:
    """Metric-specific meaning with no new statistical claims or sentiment overrides."""
    metric = metric.lower()
    if metric == 'annualized_volatility':
        metric = 'volatility'
    if sentiment not in ('positive','negative','neutral'):
        return 'This comparison could not be assessed with the available information.'
    explanations = _METRIC_DIRECTION.get(metric, (
        'The observed result was more favorable for this measure.',
        'The observed result was less favorable for this measure.',
        'The observed result was similar for this measure.',
    ))
    statement = explanations[{'positive':0,'negative':1,'neutral':2}[sentiment]]
    # The visible numeric comparison already supplies the gap; repeating it in
    # every insight makes the explanation harder to scan. Classification is
    # determined upstream by V8.4, not recomputed here.
    return statement


def composition_data_with_unknown(weights: dict, kind: str):
    """Add an explicitly separate, non-redistributed unknown chart row."""
    from .v94_presentation import composition_rows
    known, unknown = composition_rows(weights,kind)
    rows = [dict(row, Coverage='Identified') for row in known]
    if math.isfinite(unknown) and unknown > .005:
        rows.append({'Category':'Unclassified / not applicable',
                     'Share of portfolio (%)': round(unknown,2),
                     'Coverage':'Unclassified'})
    return rows, unknown
