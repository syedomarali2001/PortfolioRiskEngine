"""Regression tests for the V10.1 visual/calculator add-on.

No portfolio engine functions are used by the standalone calculators.
"""
from pathlib import Path
import math
import pytest
import plotly.graph_objects as go

from portfolio_risk_ui.v10_design import THEMES, CHART_PALETTES, categorical_colors, tune_plotly, theme_css
from portfolio_risk_ui.v10_calculators import (
    future_value, monthly_contribution_for_goal, college_future_cost,
    us_rmd_standard, IRS_UNIFORM_LIFETIME)

ROOT = Path(__file__).resolve().parents[1]


def test_compound_interest_known_values():
    assert future_value(1000, 0, 0, 10) == 1000
    assert future_value(1000, 50, 0, 2) == 2200
    assert future_value(1000, 0, 12, 1) == pytest.approx(1000*(1.01)**12)
    assert future_value(1000, 100, 6, 10) > 1000 + 100*120


def test_savings_goal_inverts_future_value():
    for rate in (0.0, 3.0, 7.0):
        for freq in (1, 4, 12, 365):
            goal = 100_000
            pmt = monthly_contribution_for_goal(goal, 7500, rate, 12, freq)
            assert future_value(7500, pmt, rate, 12, freq) == pytest.approx(goal, abs=1e-6)
    assert monthly_contribution_for_goal(500, 2000, 1, 5) == 0


def test_college_total_and_inflation():
    assert college_future_cost(10_000, 0, 4, 0) == 40_000
    assert college_future_cost(10_000, 4, 4, 3) > 40_000
    assert college_future_cost(10_000, 4, 4, 3) < 60_000


def test_irs_standard_rmd_table():
    assert len(IRS_UNIFORM_LIFETIME) == 48
    assert us_rmd_standard(265_000, 73) == (10_000, 26.5)
    assert us_rmd_standard(246_000, 75) == (10_000, 24.6)
    assert us_rmd_standard(2000, 120) == (1000, 2.0)
    with pytest.raises(ValueError): us_rmd_standard(200_000, 72)


def test_invalid_inputs_cannot_silently_fake_results():
    with pytest.raises(ValueError): future_value(-1, 20, 5, 10)
    with pytest.raises(ValueError): future_value(100, 0, 3, 999)
    with pytest.raises(ValueError): monthly_contribution_for_goal(100, 20, 5, 0)
    with pytest.raises(ValueError): college_future_cost(10000, 2, 12, 2)
    with pytest.raises(ValueError): us_rmd_standard(-100, 74)


def test_all_pie_palettes_have_distinct_slices():
    for name in THEMES:
        colors = categorical_colors(name, 6)
        assert len(set(colors)) == 6
        fig = go.Figure(go.Pie(labels=list("ABCDEF"), values=[1]*6))
        tune_plotly(fig, name)
        assert list(fig.data[0].marker.colors) == colors
        assert fig.data[0].marker.line.color == THEMES[name].background


def test_four_dark_themes_have_consistent_controls():
    assert 'Light' not in THEMES
    for name in THEMES:
        css = theme_css(name)
        for piece in ('stAlert', 'stTable', 'stSegmentedControl', 'stExpander'):
            assert piece in css
        assert 'v10-concentration-warning' in css
        assert '#FFF4DE' not in css


def test_calculators_have_independent_sidebar_pages_and_currency_symbols():
    app = (ROOT/'portfolio_risk_ui/app.py').read_text(encoding='utf-8')
    assert '"CALCULATORS"' in app
    for page in ('page_compound','page_savings','page_basic'):
        assert f'st.Page({page}' in app
    assert 'def display_money' in app and 'f"${float(value):,.2f} {currency}"' in app
    assert 'PORTFOLIO_GITHUB_URL' in app
    assert 'v10-brand-mark' not in app
    assert 'Portfolio Optimizer' in app
    assert 'st.Page(page_college' not in app and 'st.Page(page_rmd' not in app
    assert 'run_report' not in (ROOT/'portfolio_risk_ui/v10_calculator_pages.py').read_text(encoding='utf-8')


def test_all_dark_themes_keep_interactive_tables(monkeypatch):
    import sys
    import types
    import pandas as pd
    from portfolio_risk_ui.v10_design import display_table

    calls = []
    fake = types.SimpleNamespace(session_state={'v10_theme': 'Dark Navy'},
                                 dataframe=lambda data, **kwargs: calls.append((data, kwargs)))
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    for name in THEMES:
        fake.session_state['v10_theme'] = name
        data = pd.DataFrame({'Metric': [1]})
        display_table(data, hide_index=True)
        assert calls[-1][0] is data
        assert calls[-1][1]['hide_index'] is True
