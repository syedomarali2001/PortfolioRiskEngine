"""V10.3 targeted UI regression guards, intentionally no optimizer math changes."""
from pathlib import Path
from portfolio_risk_ui.v10_design import theme_css

ROOT = Path(__file__).resolve().parents[1] / 'portfolio_risk_ui'


def test_removed_light_falls_back_to_original_navy():
    from portfolio_risk_ui.v10_design import THEMES, theme_for
    assert 'Light' not in THEMES
    assert theme_for('Light') == THEMES['Dark Navy']
    assert theme_css('Light') == theme_css('Dark Navy')
    assert '#FFF5DD' not in theme_css('Dark Navy')


def test_keyed_holding_cards_and_explicit_light_info():
    app = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'key=f"portfolio_holding_card_{index}"' in app
    assert "class='v10-soft-info'" in app
    assert 'View project on GitHub' in app


def test_stacked_optimizer_evidence_preserves_data():
    view = (ROOT / 'v918_three_views.py').read_text(encoding='utf-8')
    assert 'first, second = st.columns(2)' not in view
    assert 'key=f"optimizer_evidence_portfolio_{goal}"' in view
    assert 'key=f"optimizer_evidence_benchmarks_{goal}"' in view
    assert 'benchmark_three_metric_summary(item)' in view
    assert 'portfolio_validation_explanation(goal, item)' in view
    assert 'v10-concentration-warning' in view
