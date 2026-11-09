"""V9.9 acceptance tests for honest validation and navigation clarity."""
from pathlib import Path

import pytest

from portfolio_risk_ui.v99_presentation import (
    evidence_requirements, goal_from_priorities, goal_metric_labels,
    metric_decision, validation_summary,
)
from portfolio_risk_ui.v98_presentation import STRATEGY_GOALS

APP = (Path(__file__).resolve().parents[1] / 'portfolio_risk_ui' / 'app.py').read_text(encoding='utf-8')


def sample_report(main, support=None, *, backend_success=False):
    items = {'es95': {'positive': main[0], 'neutral': main[1], 'negative': main[2], 'unavailable': 0}}
    priors = {'es95': 3.0}
    if support is not None:
        priors['drawdown'] = 1.0
        items['drawdown'] = {'positive': support[0], 'neutral': support[1], 'negative': support[2], 'unavailable': 0}
    return {'rolling_robustness': {
        'status': 'evaluated', 'fold_count': 20, 'priorities': priors,
        'per_metric_tally': items, 'consistent_improvement_on_all_priorities': backend_success,
    }}


def test_fifteen_of_twenty_is_enough_but_one_worsening_disqualifies():
    result = validation_summary(sample_report((15, 4, 1)))
    assert evidence_requirements(20) == 15
    assert result['status'] == 'not_passed'
    main = result['primary_verdict']
    assert main['positive_gate_passed'] is True
    assert main['zero_worsening_passed'] is False
    assert main['passed'] is False
    assert '15 of 20' in result['description']
    assert 'minimum of 15' in result['description']
    assert 'requires zero' in result['description']


def test_fifteen_better_zero_worse_passes_single_goal_when_backend_confirms():
    result = validation_summary(sample_report((15, 5, 0), backend_success=True))
    assert result['primary_verdict']['passed']
    assert result['status'] == 'passed'


def test_supporting_goal_can_block_even_when_main_goal_passes():
    report = sample_report((15, 5, 0), support=(6, 14, 0))
    data = validation_summary(report)
    assert data['primary_verdict']['passed']
    assert data['status'] == 'not_passed'
    assert len(data['failed_supporting']) == 1
    assert 'supporting check' in data['description']
    assert 'peak-to-trough' in data['description']


def test_each_goal_has_one_primary_metric_and_supporting_metrics_are_described():
    assert goal_from_priorities(STRATEGY_GOALS['Reduce losses on bad days'][0]) == 'Reduce losses on bad days'
    assert goal_metric_labels(STRATEGY_GOALS['Reduce losses on bad days'][0]) == [
        'losses on the worst trading days', 'smaller peak-to-trough declines']
    for goal, (spec, desc) in STRATEGY_GOALS.items():
        assert goal_from_priorities(spec) == goal
        assert goal_metric_labels(spec) and desc


def test_menu_removes_noisy_sidebar_notice_and_marks_locked_items():
    assert 'st.info("Start in **Portfolio Builder**' not in APP
    assert 'Market data from third parties may be delayed or incomplete. No trading or account access.' not in APP
    assert 'icon=icon if portfolio_ready else ":material/lock:"' in APP
    assert 'builder_icon = ":material/account_balance_wallet:" if portfolio_ready else ":material/arrow_circle_right:"' in APP
    assert 'st.Page(page_builder,title="Portfolio Builder",icon=builder_icon,default=True)' in APP
    assert 'def _current_data_warning():' in APP


def test_dividend_lines_have_very_distinct_colors_and_line_styles():
    assert '#F5A45B' in APP and '#4BD4E5' in APP
    assert 'line_dash="Choice"' in APP
    assert 'symbol="Choice"' in APP
    assert 'hoverlabel=dict(bgcolor="#19283F"' in APP


def test_valuation_removes_warning_but_preserves_uncertainty():
    assert 'st.warning("This is a large valuation gap.' not in APP
    assert 'This simplified cash-flow model depends heavily on its inputs' in APP
    assert 'not a forecast or evidence the share price must move toward the estimate' in APP


def test_three_strategy_results_use_one_shared_analysis_and_hide_advanced_controls():
    from pathlib import Path
    page=APP.split('def page_strategy():',1)[1].split('def main():',1)[0]
    assert '"--three-strategies"' in page
    assert 'Optional investment limits' in page
    assert 'Test my own allocation (optional)' in page
    assert 'show_three_results(record' in page
    assert '_warnings(report)' not in page

