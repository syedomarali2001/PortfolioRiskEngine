"""V10.3 baseline, Light removed; no financial calculations changed."""
import plotly.graph_objects as go
from portfolio_risk_ui.v10_design import THEMES, theme_for, theme_css, categorical_colors, tune_plotly


def test_theme_selector_can_only_use_four_dark_appearances():
    assert tuple(THEMES) == ('Dark Navy', 'Ocean Blue', 'Forest Green', 'Burgundy Red')
    assert theme_for('Light') == THEMES['Dark Navy']
    assert all(not theme.is_light for theme in THEMES.values())


def test_chart_data_and_palettes_are_unchanged():
    for name in THEMES:
        colors = categorical_colors(name, 4)
        assert len(set(colors)) == 4
        chart = go.Figure(go.Bar(x=['A', 'B'], y=[4, 7]))
        result = tune_plotly(chart, name)
        assert tuple(result.data[0].y) == (4, 7)
        assert result.layout.template.layout.paper_bgcolor == 'rgb(17,17,17)'
        assert f'--v10-bg:{THEMES[name].background}' in theme_css(name)
