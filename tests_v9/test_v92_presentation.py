"""V9.2 presentation-only acceptance checks; no new financial calculations."""
from pathlib import Path

import pytest

from portfolio_risk_ui.presentation import (
    allocation_pie_rows, finding_display, format_metric_for_people,
    overview_notes,
)


@pytest.mark.parametrize('metric,raw,expected', [
    ('cagr', 0.2338, '23.38%'),
    ('volatility', 0.1653, '16.53%'),
    ('drawdown', 0.1934, '19.34%'),
    ('es95', 0.0221, '2.21%'),
    ('sharpe', 1.3601, '1.36'),
    ('sortino', 1.42, '1.42'),
    ('beta', .93, '0.93'),
])
def test_metric_presentation(metric,raw,expected):
    assert format_metric_for_people(metric,raw) == expected


def test_no_fabricated_data_for_missing_or_nonfinite():
    assert format_metric_for_people('cagr',None) == 'Unavailable'
    assert format_metric_for_people('cagr',float('nan')) == 'Unavailable'
    assert format_metric_for_people('cagr','not a number') == 'Unavailable'


def test_percentage_point_differences_are_readable():
    out = finding_display({
        'metric':'volatility','benchmark':'RSP','sentiment':'negative',
        'portfolio_value':.1653, 'benchmark_value':.1388,
        'favorable_difference':-.0265, 'materiality_threshold':.0200,
        'explanation':'Historical variation in returns',
    })
    assert out['title'] == 'NEGATIVE: Volatility vs RSP'
    assert '16.53%' in out['comparison']
    assert '13.88%' in out['comparison']
    assert '2.65 percentage points' in out['comparison']
    assert '2.00 percentage points' in out['comparison']
    assert '0.0265' not in out['comparison']


def test_ratio_difference_not_wrongly_percentified():
    out = finding_display({
        'metric':'sharpe','benchmark':'SPY','sentiment':'neutral',
        'portfolio_value':1.36,'benchmark_value':1.29,
        'favorable_difference':.07,'materiality_threshold':.2,
    })
    assert '1.36' in out['comparison']
    assert '1.29' in out['comparison']
    assert '0.07' in out['comparison']
    assert 'percentage points' not in out['comparison']


def test_donut_preserves_allocation_with_grouped_small_holdings():
    weights={f'T{i}': .01 for i in range(30)}
    weights.update({'AAPL':.35,'SPY':.35})
    x=allocation_pie_rows(weights,max_slices=6)
    assert len(x)==6
    assert sum(r['Weight (%)'] for r in x)==pytest.approx(100.0)
    assert any('Other holdings' in r['Holding'] for r in x)


def test_donut_simple_portfolio_has_each_ticker():
    data=allocation_pie_rows({'AAPL':.35,'SPY':.3,'MSFT':.2,'GLD':.15})
    assert [r['Holding'] for r in data]==['AAPL','SPY','MSFT','GLD']
    assert [r['Weight (%)'] for r in data]==pytest.approx([35,30,20,15])


def test_warning_classification_preserves_unexpected_quality_warning():
    method,quality=overview_notes([
        "Today's portfolio allocations applied retrospectively, not actual investor P&L.",
        '2026-11-01 market price missing for ticker ABC',
    ])
    assert len(method)==1
    assert len(quality)==1
    assert 'missing' in quality[0]


def test_app_has_no_deprecated_width_usage_and_uses_donut():
    app=(Path(__file__).resolve().parents[1]/'portfolio_risk_ui'/'app.py').read_text()
    assert 'use_container_width' not in app
    assert 'width="stretch"' in app
    assert '_allocation_donut(insights["input_weights"])' in app
    assert 'st.subheader("Dividend Income")' in app
    assert '"TRADING CURRENCY"' in app
    assert 'st.subheader("2. Set amount and add to portfolio")' in app


def test_no_backend_changes_in_v92_files():
    root=Path(__file__).resolve().parents[1]
    assert (root/'src'/'portfolio_risk').is_dir()
