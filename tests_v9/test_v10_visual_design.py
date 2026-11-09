"""V10 visual contract: four dark themes, visible charts and no finance changes."""
from pathlib import Path

import plotly.graph_objects as go

from portfolio_risk_ui.v10_design import THEMES, theme_for, theme_css, tune_plotly

ROOT = Path(__file__).resolve().parents[1]


def _luminance(hex_color):
    channels = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    linear = [v / 12.92 if v <= .04045 else ((v + .055)/1.055)**2.4 for v in channels]
    return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]


def _contrast(a, b):
    x, y = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (x+.05)/(y+.05)


def test_four_distinct_available_appearances():
    assert tuple(THEMES) == ('Dark Navy', 'Ocean Blue', 'Forest Green', 'Burgundy Red')
    assert len({t.background for t in THEMES.values()}) == 4
    assert theme_for('unknown') is THEMES['Dark Navy']
    assert all(not theme.is_light for theme in THEMES.values())


def test_readability_all_four_palettes():
    for theme in THEMES.values():
        for surface in (theme.background, theme.sidebar, theme.panel, theme.elevated):
            assert _contrast(theme.text, surface) >= 7.0, (theme.name, 'main text', surface)
            assert _contrast(theme.muted, surface) >= 4.5, (theme.name, 'secondary', surface)
        assert _contrast(theme.accent, theme.background) >= 3.0, theme.name
        assert _contrast(theme.accent_text, theme.accent) >= 4.5, theme.name


def test_styles_have_shared_sizing_and_safe_scope():
    for name in THEMES:
        css = theme_css(name)
        assert '.block-container' in css
        assert 'max-width:1520px' in css
        assert 'stMetric' in css and 'stTabs' in css
        assert 'valuation_multiple_' in css
        assert 'unsafe' not in css
        assert f'--v10-bg:{THEMES[name].background}' in css
        assert '<style>' in css and '</style>' in css


def test_plotly_dark_charts_follow_theme_without_changing_data():
    for name in THEMES:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['A','B'], y=[3, 8], marker_color='#78A9EB'))
        fig.update_layout(height=360, xaxis_title='Holding', yaxis_title='Allocation')
        output = tune_plotly(fig, name)
        assert output is fig
        assert tuple(output.data[0].y) == (3, 8)
        assert output.layout.height == 360
        assert output.data[0].marker.color == THEMES[name].chart
        assert output.layout.font.color == THEMES[name].text
        assert output.layout.yaxis.gridcolor == THEMES[name].grid
        assert output.layout.template.layout.paper_bgcolor == 'rgb(17,17,17)'


def test_every_visible_plotly_chart_is_themed_and_main_site_is_still_seven_pages():
    app = (ROOT / 'portfolio_risk_ui' / 'app.py').read_text(encoding="utf-8")
    view = (ROOT / 'portfolio_risk_ui' / 'v918_three_views.py').read_text(encoding="utf-8")
    assert app.count('tune_plotly(') == app.count('st.plotly_chart(')
    assert app.count('theme=None)') == app.count('st.plotly_chart(')
    assert view.count('tune_plotly(') == view.count('st.plotly_chart(')
    assert view.count('theme=None)') == view.count('st.plotly_chart(')
    assert 'Portfolio Risk Engine · V10' in app
    assert 'st.selectbox("Appearance"' in app
    assert 'v10_theme' in app
    assert app.count('nav_page(page_') == 6
    assert 'st.Page(page_builder' in app
    assert 'load_example(st.session_state, demo_key)' in app
    assert 'with st.container(border=True, key=f"valuation_multiple_{metric}")' in app
