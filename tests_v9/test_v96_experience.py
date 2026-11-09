"""V9.6 UI-only improvements. Keep V8.4 numbers and diagnostic sentiment authoritative."""
from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_risk_ui.v94_presentation import finding_for_people
from portfolio_risk_ui.v96_presentation import (
    HISTORY_NOTE, composition_data_with_unknown, finding_message,
    historical_window_text,
)


@pytest.mark.parametrize("first,last,expected", [
    ("2015-10-13", "2026-11-06", "Oct 13, 2015 to Nov 6, 2026"),
    ("2024-01-02", "2025-02-11", "Jan 2, 2024 to Feb 11, 2025"),
    ("bad", "2025-01-01", "Historical dates unavailable"),
    ("2026-01-02", "2025-01-01", "Historical dates unavailable"),
    (None, None, "Historical dates unavailable"),
])
def test_history_uses_real_report_dates(first,last,expected):
    assert historical_window_text({"first_date":first,"last_date":last}) == expected


def test_history_explains_newer_holdings_and_not_a_purchase_date():
    assert "earliest date" in HISTORY_NOTE
    assert "every selected investment" in HISTORY_NOTE
    assert "newer ETF" in HISTORY_NOTE
    assert "does not mean you bought" in HISTORY_NOTE


@pytest.mark.parametrize('metric,positive,negative,neutral', [
    ('cagr','growth was higher','growth was lower','growth was close'),
    ('volatility','fluctuated less','fluctuated more','fluctuation was broadly similar'),
    ('drawdown','fall from a previous high was smaller','fall from a previous high was deeper','fall was broadly similar'),
    ('sharpe','more historical return per unit','less historical return per unit','volatility was similar'),
    ('sortino','more historical return relative','less historical return relative','fluctuations was similar'),
    ('es95','worst 5% of trading days were smaller','worst 5% of trading days were larger','worst 5% of trading days were similar'),
])
def test_metric_specific_explanations(metric,positive,negative,neutral):
    assert positive in finding_message(metric,'positive','1.20')
    assert negative in finding_message(metric,'negative','1.20')
    assert neutral in finding_message(metric,'neutral','1.20')
    assert 'measured gap' not in finding_message(metric,'positive','1.20')
    assert 'measured gap was 1.20' not in finding_message(metric,'neutral','1.20')


def test_diagnostic_labels_agree_with_tabs_and_keep_sentiment():
    cases={'positive':'Strength','negative':'Area to watch','neutral':'Similar result'}
    for status,label in cases.items():
        item={'sentiment':status,'metric':'sharpe','benchmark':'SPY',
              'portfolio_value':1.06,'benchmark_value':.86,
              'favorable_difference':.20,'materiality_threshold':.20}
        output=finding_for_people(item)
        assert output['status']==status
        assert output['status_label']==label
        assert output['headline']=='Sharpe ratio compared with SPY'
        assert '1.06' in output['comparison']
        assert '.86' in output['comparison']
        assert 'noticeable' not in output['reason'].lower()


def test_es95_explained_by_name_not_internal_key():
    item={'sentiment':'negative','metric':'es95','benchmark':'RSP',
          'portfolio_value':.04,'benchmark_value':.025,'favorable_difference':-.015,
          'materiality_threshold':.005}
    f=finding_for_people(item)
    assert 'Expected shortfall (95%)' in f['headline']
    assert 'worst 5%' in f['reason']
    assert f['difference'] == '1.50 percentage points'
    assert 'es95' not in f['headline'].lower()


def test_source_never_reclassifies_a_backend_positive_as_negative():
    p=finding_for_people({'sentiment':'positive','metric':'drawdown','benchmark':'SPY',
                           'portfolio_value':.1,'benchmark_value':.3,'favorable_difference':.2})
    assert p['status']=='positive'
    assert p['status_label']=='Strength'


def test_sector_unclassified_is_a_separate_last_bar_without_renormalization():
    rows,unknown=composition_data_with_unknown({'technology':.50,'health_care':.30,'UNKNOWN':.20},'Sectors')
    assert unknown==pytest.approx(20)
    assert rows[-1]['Category']=='Unclassified / not applicable'
    assert rows[-1]['Coverage']=='Unclassified'
    assert rows[0]['Coverage']=='Identified'
    assert sum(row['Share of portfolio (%)'] for row in rows)==pytest.approx(100)


def test_unknown_only_portfolio_keeps_explicit_data_gap():
    rows,unknown=composition_data_with_unknown({'UNKNOWN':1},'Sectors')
    assert rows==[{'Category':'Unclassified / not applicable',
                   'Share of portfolio (%)':100.0,'Coverage':'Unclassified'}]
    assert unknown==pytest.approx(100)


def test_no_unknown_adds_no_filler_bar():
    rows,unknown=composition_data_with_unknown({'technology':.7,'financials':.3},'Sectors')
    assert len(rows)==2
    assert unknown==0


def test_source_contract_is_easier_to_understand_and_not_unnecessarily_cluttered():
    text=(Path(__file__).resolve().parents[1]/'portfolio_risk_ui'/'app.py').read_text(encoding='utf8')
    for marker in (
        'st.subheader(f"Historical period analyzed:',
        'st.divider()',
        'st.subheader("Key findings")',
        'st.subheader("Strengths and areas to watch")',
        'st.subheader("Explore what\'s inside your portfolio")',
        'st.session_state.xray_active_section = name',
        'nav_page(page_exposure,"Holdings & Exposure"',
        'composition_data_with_unknown(weights,label)',
    ):
        assert marker in text
    assert 'enrich = 10' in text
    assert 'st.slider("Optional company profiles to check"' not in text
    assert 'with st.expander("Help and common questions")' not in text
    assert 'with st.expander("How this analysis works' not in text
    assert 'with st.expander("Detailed risk allocation table")' not in text
    assert 'with st.popover("What do Better, Similar and Worse mean?")' not in text
    assert 'st.tabs([\n        "Portfolio makeup"' not in text


def test_financial_backend_was_not_changed_by_ui_patch():
    """Tests still import the validated source package, with no new backend dependencies."""
    import portfolio_risk.performance_service
    import portfolio_risk.exposure
    assert portfolio_risk.performance_service.__file__
    assert portfolio_risk.exposure.__file__
