"""V10 theme rollback guard: V10.4's failed Light overrides must not be present."""
from pathlib import Path
from portfolio_risk_ui.v10_design import THEMES, CHART_PALETTES, theme_css

ROOT = Path(__file__).resolve().parents[1]


def test_no_light_option_or_palette():
    assert tuple(THEMES) == ('Dark Navy', 'Ocean Blue', 'Forest Green', 'Burgundy Red')
    assert set(CHART_PALETTES) == set(THEMES)
    app = (ROOT / 'portfolio_risk_ui' / 'app.py').read_text(encoding='utf8')
    assert 'Preview five coordinated' not in app
    assert 'if st.session_state.get("v10_theme") not in THEMES:' in app


def test_no_v104_light_css_or_badges():
    design = (ROOT / 'portfolio_risk_ui' / 'v10_design.py').read_text(encoding='utf8')
    app = (ROOT / 'portfolio_risk_ui' / 'app.py').read_text(encoding='utf8')
    assert 'v104-status-pill' not in design
    assert '_v104_status_badge' not in app
    for theme in THEMES:
        css = theme_css(theme)
        assert 'v10-concentration-warning' in css
        assert '#FFF5DD' not in css
