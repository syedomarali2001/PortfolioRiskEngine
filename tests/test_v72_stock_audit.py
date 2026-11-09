"""Deterministic V7.2 stock metadata audit tests: no live Yahoo calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from portfolio_risk.cli import main
from portfolio_risk.security_metadata import YahooSecurityMetadataProvider
from portfolio_risk.stock_audit import read_stock_universe, run_stock_audit, stock_report

DATA=Path(__file__).resolve().parents[1]/'examples/stock_universe_v72.csv'


def lookup(symbol):
    if symbol=='RY.TO':
        return {'symbol': symbol,'quoteType':'EQUITY','shortName':'RBC','sector':'Financial Services',
                'industry':'Banks - Diversified','country':'Canada','exchange':'TOR','currency':'CAD'}
    if symbol=='TSM':
        return {'symbol':symbol,'quoteType':'EQUITY','shortName':'TSMC',
                'sector':'Technology','country':'Taiwan','exchange':'NYQ','currency':'USD'}
    if symbol=='MISSING':
        return {}
    if symbol=='DOWN':
        raise RuntimeError('Mock throttle')
    return {'symbol':symbol,'quoteType':'EQUITY','longName':'Test Co',
            'sector':'Technology','industry':'Software','country':'United States',
            'exchange':'NMS','currency':'USD'}


def test_stock_fixture_is_stratified():
    universe=read_stock_universe(DATA)
    assert len(universe)>=100
    assert sum(x[1]=='US_STOCK' for x in universe)>=60
    assert sum(x[1]=='CAD_STOCK' for x in universe)>=25
    assert len({x[2] for x in universe})>=10
    assert {'TSM','AAPL','RY.TO','GIB-A.TO'} <= {x[0] for x in universe}


@pytest.mark.parametrize('csv,regex',[
    ('symbol,market,group\nAAPL,US_STOCK,Test\nAAPL,US_STOCK,Test\n','Duplicate'),
    ('symbol,market,group\nAAPL,US_ETF,Test\n','invalid'),
    ('symbol,market,group\nAAPL,US_STOCK,\n','invalid'),
    ('symbol,market\nAAPL,US_STOCK\n','requires'),
])
def test_stock_universe_validates(tmp_path,csv,regex):
    path=tmp_path/'stock.csv';path.write_text(csv)
    with pytest.raises(ValueError,match=regex):read_stock_universe(path)


def test_stock_audit_true_classification_and_domestic_not_from_exchange():
    universe=(('TSM','US_STOCK','Foreign ADR'),('RY.TO','CAD_STOCK','Canadian'))
    rows=run_stock_audit(universe,YahooSecurityMetadataProvider(getter=lookup),delay_ms=0)
    assert rows[0].company_country=='Taiwan'
    assert rows[0].currency=='USD'
    assert rows[0].recognized_equity
    assert rows[1].company_country=='Canada'
    assert rows[1].sector=='Financial Services'
    report=stock_report(rows)
    assert report['summary']['total']==2
    assert report['summary']['equity_identified']==2
    assert report['by_market']['US_STOCK']['company_country_available']==1
    assert 'revenue' in report['country_definition']


def test_stock_audit_handles_provider_unavailable_and_missing():
    rows=run_stock_audit((('MISSING','US_STOCK','x'),('DOWN','US_STOCK','y')),
              YahooSecurityMetadataProvider(getter=lookup),delay_ms=0)
    assert rows[0].status=='missing'
    assert rows[1].status=='unavailable'
    assert stock_report(rows)['summary']['provider_unavailable']==1


def test_stock_audit_detects_nonstock_input():
    p=YahooSecurityMetadataProvider(getter=lambda s:{'quoteType':'ETF','shortName':'fund'})
    row=run_stock_audit((('SPY','US_STOCK','wrong'),),p,delay_ms=0)[0]
    assert row.type_mismatch and not row.recognized_equity


def test_stock_audit_validates_delay():
    with pytest.raises(ValueError,match='delay'):
        run_stock_audit((('AAPL','US_STOCK','x'),),YahooSecurityMetadataProvider(getter=lookup),delay_ms=-1)


def test_stock_cli_file_and_summary(monkeypatch,tmp_path,capsys):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli,'YahooSecurityMetadataProvider',lambda:YahooSecurityMetadataProvider(getter=lookup))
    universe=tmp_path/'stocks.csv'
    universe.write_text('symbol,market,group\nAAPL,US_STOCK,US tech\nRY.TO,CAD_STOCK,CA bank\n')
    out=tmp_path/'summary.json'
    assert main(['stock-audit',str(universe),'--output',str(out),'--delay-ms','0'])==0
    assert 'stock metadata audit' in capsys.readouterr().out
    obj=json.loads(out.read_text())
    assert obj['summary']['total']==2
    assert obj['by_market']['CAD_STOCK']['equity_identified']==1
    assert main(['stock-audit',str(universe),'--limit','0'])==1


def test_stock_cli_json_output_and_limit(monkeypatch,tmp_path,capsys):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli,'YahooSecurityMetadataProvider',lambda:YahooSecurityMetadataProvider(getter=lookup))
    source=tmp_path/'stock.csv'
    source.write_text('symbol,market,group\nAAPL,US_STOCK,Tech\nRY.TO,CAD_STOCK,Bank\n')
    assert main(['stock-audit',str(source),'--delay-ms','0','--limit','1','--json',
                 '--output',str(tmp_path/'r.json')])==0
    report=json.loads(capsys.readouterr().out)
    assert report['summary']['total']==1
    assert report['assets'][0]['symbol']=='AAPL'
