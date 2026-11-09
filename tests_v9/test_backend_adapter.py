from pathlib import Path
import sys
import json
import pytest

from portfolio_risk_ui.backend import BackendError, run_report, run_fundamentals


@pytest.fixture
def stub_engine(tmp_path):
    project=tmp_path / 'portfolio_risk'
    project.mkdir()
    (project/'__init__.py').write_text('',encoding='utf-8')
    (project/'__main__.py').write_text('''\
import sys, json, pathlib, csv
args=sys.argv[1:]
command=args[0]
out=pathlib.Path(args[args.index("--output")+1])
rows=[]
if command!="fundamentals":
    with open(args[1],newline="") as f:rows=list(csv.DictReader(f))
out.write_text(json.dumps({"command":command,"rows":rows,"symbol":args[1],"argv":args}),encoding="utf-8")
''',encoding='utf-8')
    return tmp_path


def test_invokes_backend_using_cli_arguments(stub_engine):
    report=run_report('dividends',[{'symbol':'AAPL','market_value':500,'currency':'USD'}],
           'USD',['--years','10','--tax-drag','0'],root=stub_engine)
    assert report['command']=='dividends'
    assert report['rows'][0]['symbol']=='AAPL'
    assert report['rows'][0]['currency']=='USD'
    assert '--years' in report['argv']


def test_insights_report(stub_engine):
    report=run_report('insights',[{'symbol':'SPY','market_value':100,'currency':'CAD'}],
                      'CAD',['--start','2020-01-01','--end','2025-01-01'],root=stub_engine)
    assert report['command']=='insights'
    assert report['rows'][0]['currency']=='CAD'


def test_fundamentals_uses_symbol_not_portfolio(stub_engine):
    report=run_fundamentals('aapl',root=stub_engine)
    assert report['command']=='fundamentals'
    assert report['symbol']=='AAPL'


def test_arbitrary_command_is_rejected(stub_engine):
    with pytest.raises(BackendError):run_report('delete',[],"USD",root=stub_engine)


def test_invalid_fundamental_symbol_rejected(stub_engine):
    with pytest.raises(ValueError):run_fundamentals('foo;echo hacked',root=stub_engine)


def test_missing_engine_path():
    with pytest.raises(BackendError,match='directory does not exist'):
        run_report('insights',[],"USD",root=Path('/a/nonexistent/testing/engine'))


def test_engine_failure_is_visible(tmp_path):
    project=tmp_path/'portfolio_risk'
    project.mkdir()
    (project/'__init__.py').write_text('')
    (project/'__main__.py').write_text('raise RuntimeError("example intentional failure")')
    with pytest.raises(BackendError,match='intentional failure'):
        run_report('insights',[],"USD",root=tmp_path)


def test_missing_output_is_visible(tmp_path):
    project=tmp_path/'portfolio_risk'
    project.mkdir()
    (project/'__init__.py').write_text('')
    (project/'__main__.py').write_text('print("no json created")')
    with pytest.raises(BackendError,match='no JSON'):
        run_report('insights',[],"USD",root=tmp_path)
