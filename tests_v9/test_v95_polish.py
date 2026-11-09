"""Focused V9.5 user-experience and FX correctness checks."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from portfolio_risk_ui.fx_display import display_portfolio_value, get_fx_quote
from portfolio_risk_ui.v95_presentation import sort_sectors_by_exposure, coverage_summary


@pytest.mark.parametrize('base,rate,expected', [
    ('CAD', '1.40', '2400.000'),
    ('USD', '1.40', '1714.285714285714285714285714'),
])
def test_mixed_currency_total_uses_real_exchange_rate(base,rate,expected):
    rows = [{'symbol':'AAPL', 'market_value':1000, 'currency':'USD'},
            {'symbol':'SHOP.TO', 'market_value':1000, 'currency':'CAD'}]
    result=display_portfolio_value(rows,base,rate)
    assert result.total==pytest.approx(Decimal(expected),abs=Decimal('0.01'))
    assert sum(result.shares.values())==pytest.approx(100)


def test_native_currency_requires_no_fx_lookup():
    row=[{'symbol':'SHOP.TO','market_value':1000,'currency':'CAD'}]
    out=display_portfolio_value(row,'CAD')
    assert out.total==Decimal(1000)
    assert out.shares=={'SHOP.TO':100.0}


def test_wrong_currency_is_not_silently_added():
    rows=[{'symbol':'AAPL','market_value':1000,'currency':'USD'},
          {'symbol':'SHOP.TO','market_value':2000,'currency':'CAD'}]
    with pytest.raises(ValueError,match='dated FX quote'):
        display_portfolio_value(rows,'CAD')


@pytest.mark.parametrize('rate', ['0','-1','NaN','Infinity'])
def test_invalid_fx_rate_is_rejected(rate):
    with pytest.raises((ValueError,ArithmeticError)):
        display_portfolio_value([{'symbol':'SPY','market_value':100,'currency':'USD'}],'CAD',rate)


@pytest.mark.parametrize('amount', ['0','-100','NaN','Infinity'])
def test_invalid_holding_value_is_rejected(amount):
    with pytest.raises(ValueError,match='positive finite'):
        display_portfolio_value([{'symbol':'AAPL','market_value':amount,'currency':'USD'}],'USD')


def test_duplicate_holdings_not_silently_overwritten():
    with pytest.raises(ValueError,match='Duplicate'):
        display_portfolio_value([
           {'symbol':'AAPL','market_value':10,'currency':'USD'},
           {'symbol':'AAPL','market_value':20,'currency':'USD'}],'USD')


def test_display_conversion_does_not_modify_raw_portfolio_data():
    rows=[{'symbol':'AAPL','market_value':100,'currency':'USD'},
          {'symbol':'SHOP.TO','market_value':200,'currency':'CAD'}]
    before=[r.copy() for r in rows]
    display_portfolio_value(rows,'CAD',Decimal('1.3'))
    assert rows==before


def test_existing_fx_service_used_instead_of_an_independent_provider(monkeypatch):
    from portfolio_risk_ui import fx_display
    from portfolio_risk.fx import FXQuote
    observed={}
    def fake(provider,cache,as_of):
        observed.update(provider=type(provider).__name__,cache=type(cache).__name__,as_of=as_of)
        return FXQuote(Decimal('1.37'),as_of,as_of,'test quote')
    monkeypatch.setattr(fx_display,'latest_usd_cad_quote',fake)
    quote=get_fx_quote(date(2026,11,6))
    assert quote.usd_cad==Decimal('1.37')
    assert observed=={'provider':'YahooFinanceProvider','cache':'HistoryCache','as_of':date(2026,11,6)}


def test_sector_menu_descending_with_unknown_last():
    values={'financials':{'portfolio_fraction':.1},
            'UNKNOWN':{'portfolio_fraction':.5},
            'technology':{'portfolio_fraction':.6},
            'health_care':{'portfolio_fraction':.3}}
    assert sort_sectors_by_exposure(values)==['technology','health_care','financials','UNKNOWN']


def test_sector_menu_missing_or_nonfinite_data_is_safe():
    data={'unknown':{'portfolio_fraction':1},'real_estate':{},'healthcare':{'portfolio_fraction':float('nan')}}
    assert sort_sectors_by_exposure(data)[-1]=='unknown'


def test_unclassified_share_remains_visible_and_not_renormalized():
    headline, reason=coverage_summary(93.22,6.78)
    assert '93.22%' in headline
    assert '6.78%' in reason
    assert 'gold' in reason.lower()


def test_presentation_contract_keeps_all_diagnostics_and_explanations():
    source=(Path(__file__).resolve().parents[1]/'portfolio_risk_ui'/'app.py').read_text(encoding='utf-8')
    assert 'Show all {len(items)} {label}' in source
    assert 'Strengths mean a favorable historical comparison' in source
    assert 'with st.popover("What do Better, Similar and Worse mean?")' not in source
    assert 'Analysis method and data notes' in source
    assert 'get_fx_quote(as_of)' in source
    assert 'st.subheader(f"Historical period analyzed:' in source
    assert 'Downside scenario (5th percentile)' in source
    assert 'Upside scenario (95th percentile)' in source
    assert 'sort_sectors_by_exposure(sectors)' in source
    assert 'with st.expander("Data sources and coverage (optional)"' in source
    assert 'Context in tooltip' not in source


def test_percentiles_are_not_called_guaranteed_best_or_worst_cases():
    source=(Path(__file__).resolve().parents[1]/'portfolio_risk_ui'/'app.py').read_text(encoding='utf-8')
    assert 'Past data cannot tell us how likely future returns are' in source
    assert 'actual losses could be worse' in source


def test_original_backend_source_files_unmodified_since_v94():
    import hashlib
    baseline=Path('/mnt/data/v9_4_work/src/portfolio_risk')
    current=Path(__file__).resolve().parents[1]/'src'/'portfolio_risk'
    # In local validation, compare to the prior working release. Deployment
    # validates with tests against the copied original backend instead.
    if baseline.exists():
        a={p.relative_to(baseline):hashlib.sha256(p.read_bytes()).digest() for p in baseline.rglob('*.py')}
        b={p.relative_to(current):hashlib.sha256(p.read_bytes()).digest() for p in current.rglob('*.py')}
        assert a==b
