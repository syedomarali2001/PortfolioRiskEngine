"""V10.2 UI polish, navigation and independent calculator behaviors."""
from pathlib import Path
import re
import pytest
from portfolio_risk_ui.v10_calculators import calculator_press, initial_calculator_state
from portfolio_risk_ui.v10_design import theme_css, categorical_colors

ROOT=Path(__file__).resolve().parents[1]

def press_sequence(*keys):
    state=initial_calculator_state()
    for key in keys:
        state=calculator_press(state,key)
    return state

@pytest.mark.parametrize('keys,expected', [
    (('1','2','+','3','='),'15'),
    (('2','.','5','×','4','='),'10'),
    (('9','÷','3','='),'3'),
    (('9','−','2','='),'7'),
    (('1','0','%'),'0.1'),
    (('1','2','⌫'),'1'),
    (('5','±'),'-5'),
    (('0','.','1','+','0','.','2','='),'0.3'),
    (('6','÷','0','='),'Error'),
])
def test_normal_calculator(keys,expected):
    assert press_sequence(*keys)['display']==expected

def test_clear_and_recover_from_divide_zero():
    s=press_sequence('1','÷','0','=')
    assert calculator_press(s,'5')['display']=='5'
    assert calculator_press(s,'C')==initial_calculator_state()
    with pytest.raises(ValueError): calculator_press(s,'DROP TABLE')

def test_sidebar_three_calculators_no_logo_and_new_name():
    app=(ROOT/'portfolio_risk_ui/app.py').read_text(encoding='utf-8')
    assert app.count('st.Page(page_')==4
    assert 'st.Page(page_basic, title="Calculator"' in app
    assert 'st.Page(page_college' not in app and 'st.Page(page_rmd' not in app
    assert 'v10-brand-mark' not in app
    assert 'Portfolio Optimizer' in app
    assert 'View project on GitHub' in app

def test_money_suffix_consistent_across_app_and_calculators():
    app=(ROOT/'portfolio_risk_ui/app.py').read_text(encoding='utf-8')
    pages=(ROOT/'portfolio_risk_ui/v10_calculator_pages.py').read_text(encoding='utf-8')
    assert 'f"${float(value):,.2f} {currency}"' in app
    assert 'f"${float(value):,.2f} {currency}"' in pages
    assert re.search(r'\{currency\} \$',app) is None
    assert re.search(r'\{ccy\} \$',app) is None

def test_removed_light_theme_cannot_be_selected():
    from portfolio_risk_ui.v10_design import THEMES, theme_for
    assert 'Light' not in THEMES
    assert theme_for('Light') is THEMES['Dark Navy']
    assert 'stBaseButton-primary' not in theme_css('Dark Navy') or 'stMetric' in theme_css('Dark Navy')


def test_theme_palettes_are_not_monochromatic():
    for theme in ('Forest Green', 'Burgundy Red', 'Ocean Blue'):
        assert len(set(categorical_colors(theme, 7)))==7
        # The second color must contrast with the first in the first three slices.
        colors=categorical_colors(theme, 3)
        assert colors[0] != colors[1] != colors[2]
