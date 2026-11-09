"""Offline, deterministic V7.1 contract tests; never contact live provider."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from portfolio_risk.coverage_audit import (
    AuditAsset, load_audit_universe, run_coverage_audit, audit_as_dict,
)
from portfolio_risk.models import PriceHistory
from portfolio_risk.providers.tickerinside import (
    FundRecord, FundUnavailable, TickerInsideClient, TickerInsideError, parse_fund_record,
)
from portfolio_risk.cli import main
from portfolio_risk.exposure import validate_snapshots


FUNDS = {"SPY", "QQQ", "SCHD", "SPMO", "BND", "GLD", "SLV", "IAU", "USO", "PPLT"}


def payload(symbol="QQQ", *, kind=None, holdings=None):
    obj = {
        "ticker": symbol, "name": f"Example {symbol} ETF", "holdings_as_of": "2026-08-31",
        "source_layer": "sec_nport", "filed": "2026-10-27", "positions": 3,
        "stock_pct": 98.0, "holdings": holdings if holdings is not None else {"AAPL": 45, "MSFT": 35, "NVDA": 20},
        "expense_ratio": 0.2, "page_url": f"https://tickerinside.com/etf/{symbol.lower()}/",
        "cite_as": f"TickerInside {symbol} CC BY 4.0",
        "breakdown": {"sectors_pct": {"technology": 100}, "countries_pct": {"US": 100}},
    }
    if kind is not None:
        obj["holdings_kind"] = kind
    return obj


def fake_http(url):
    if url.endswith("/index.json"):
        return {"as_of": "2026-11-06", "tickers": ["QQQ", "SCHD", "BND"],
                "tickers_bond": ["BND"], "tickers_trust": ["GLD"]}
    if "/fund/" in url:
        symbol = url.split("/fund/")[1].split(".json")[0].upper()
        if symbol not in {"QQQ", "SCHD", "BND", "GLD"}:
            raise FundUnavailable("Not covered")
        kind = "bond" if symbol == "BND" else "commodity" if symbol == "GLD" else "stock"
        return payload(symbol, kind=kind)
    raise AssertionError(url)


def client(tmp_path):
    return TickerInsideClient(tmp_path / "fund_cache", getter=fake_http)


def test_index_families_and_scope(tmp_path):
    names, metadata = client(tmp_path).index()
    assert names == {"QQQ", "SCHD", "BND", "GLD"}
    assert metadata["count"] == 4
    with pytest.raises(FundUnavailable):
        client(tmp_path).fund("XEQT.TO")


def test_full_holdings_and_snapshot(tmp_path):
    f = client(tmp_path).fund("QQQ")
    assert f.holdings_as_of == date(2026, 8, 31)
    assert f.source_layer == "sec_nport"
    assert f.known_pct == 100
    assert len(f.holdings) == 3
    assert f.sector_pct["technology"] == 100
    snap = f.to_snapshot()
    validate_snapshots({"QQQ": snap})
    assert sum(h.fraction for h in snap.holdings) == pytest.approx(1)


def test_cache_means_one_network_fetch(tmp_path):
    calls = []
    def counted(url):
        calls.append(url)
        return fake_http(url)
    api = TickerInsideClient(tmp_path, getter=counted)
    first, _ = api.index()
    second, _ = api.index()
    assert first == second
    assert len(calls) == 1
    assert not api.fund("QQQ").fetched_from_cache
    assert api.fund("QQQ").fetched_from_cache
    assert len(calls) == 2
    api.fund("QQQ", refresh=True)
    assert len(calls) == 3


def test_no_cache_means_no_files(tmp_path):
    api = TickerInsideClient(tmp_path / "cache", ttl_hours=0, getter=fake_http)
    api.index(); api.fund("QQQ")
    assert not (tmp_path / "cache").exists()


def test_invalid_weights_fail_closed():
    with pytest.raises(TickerInsideError):
        parse_fund_record(payload(holdings={"AAPL": 100.5, "MSFT": 10}), "QQQ")
    with pytest.raises(TickerInsideError):
        parse_fund_record(payload(holdings={"AAPL": -2}), "QQQ")
    with pytest.raises(TickerInsideError):
        parse_fund_record(payload(holdings={"AAPL": float('nan')}), "QQQ")


def test_wrong_ticker_is_rejected():
    with pytest.raises(TickerInsideError):
        parse_fund_record(payload("QQQ"), "SCHD")


def test_missing_dates_or_non_equity_cannot_be_treated_as_equity():
    f = parse_fund_record(payload(kind="derivative"), "QQQ")
    with pytest.raises(TickerInsideError):
        f.to_snapshot()
    raw = payload()
    raw.pop("holdings_as_of")
    with pytest.raises(TickerInsideError):
        parse_fund_record(raw, "QQQ").to_snapshot()


def test_bond_and_commodity_not_stocks(tmp_path):
    assert client(tmp_path).fund("BND").holdings_kind == "bond"
    assert client(tmp_path).fund("GLD").holdings_kind == "commodity"


def test_audit_universe_30_assets():
    path = Path(__file__).resolve().parents[1] / "examples/coverage_universe_30.csv"
    assets = load_audit_universe(path)
    assert len(assets) == 30
    assert {x.category for x in assets} == {
        'US_ETF','CAD_ETF','US_STOCK','CAD_STOCK','COMMODITY_ETP','CRYPTO'}
    assert all(sum(x.category == name for x in assets) == 5 for name in
               {x.category for x in assets})


def test_audit_status_is_category_aware(tmp_path):
    universe = (
        AuditAsset("QQQ", "US_ETF", "USD"),
        AuditAsset("BND", "US_ETF", "USD"),
        AuditAsset("XEQT.TO", "CAD_ETF", "CAD"),
        AuditAsset("AAPL", "US_STOCK", "USD"),
        AuditAsset("BTC-USD", "CRYPTO", "USD"),
        AuditAsset("GLD", "COMMODITY_ETP", "USD"),
        AuditAsset("VOO", "US_ETF", "USD"),
    )
    rows = run_coverage_audit(universe, client(tmp_path), as_of=date(2026, 11, 7))
    assert [x.holdings_status for x in rows] == [
        'holdings_available', 'holdings_available', 'outside_provider_scope',
        'not_applicable', 'not_applicable', 'holdings_available', 'not_indexed']
    assert rows[0].age_days == 68
    assert rows[1].holdings_kind == 'bond'
    assert rows[0].sector_count == 1
    assert rows[0].holdings_count == 3
    out = audit_as_dict(rows)
    assert out['summary']['by_category']['US_ETF']['holdings_available'] == 2
    assert out['assets'][0]['provider_citation'].startswith('TickerInside')


class MockYahoo:
    def get_history(self, symbol, start, end):
        prices = pd.Series([100.0, 100.1, 100.2],
                           index=pd.to_datetime([start, start.replace(day=start.day+1),
                                                 start.replace(day=start.day+2)]))
        return PriceHistory(symbol, prices, start, end, 'mock', datetime.now(timezone.utc), 'USD')


def test_optional_price_check_independent(tmp_path):
    rows = run_coverage_audit((AuditAsset('AAPL','US_STOCK','USD'),), client(tmp_path),
                              check_prices=True, provider=MockYahoo(),
                              price_cache=__import__('portfolio_risk.cache',fromlist=['HistoryCache']).HistoryCache(tmp_path / 'prices'),
                              as_of=date(2026,11,7))
    assert rows[0].holdings_status == 'not_applicable'
    assert rows[0].price_status == 'quality_warning'  # mock prices stop long before cutoff
    assert rows[0].price_obs == 3


def test_audit_cli_with_mocked_provider(monkeypatch, tmp_path, capsys):
    import portfolio_risk.cli as cli
    original = cli.TickerInsideClient
    monkeypatch.setattr(cli, 'TickerInsideClient', lambda **kwargs: client(tmp_path))
    universe = tmp_path / 'tiny.csv'
    universe.write_text('symbol,category,quote_currency\nQQQ,US_ETF,USD\nXEQT.TO,CAD_ETF,CAD\n')
    out = tmp_path / 'results.json'
    assert main(['coverage-audit',str(universe),'--output',str(out)]) == 0
    output = capsys.readouterr().out
    assert 'holdings_available' in output
    assert 'outside_provider_scope' in output
    import json
    document = json.loads(out.read_text())
    assert document['summary']['total'] == 2


def test_bad_file_rejected(tmp_path):
    csv = tmp_path/'bad.csv'
    csv.write_text('symbol,category,quote_currency\nQQQ,US_ETF,USD\nQQQ,US_ETF,USD\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_audit_universe(csv)


def test_real_portfolio_auto_tickerinside_mocked(monkeypatch, tmp_path, capsys):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli, 'TickerInsideClient', lambda **kwargs: client(tmp_path))
    portfolio = tmp_path / 'holdings.csv'
    portfolio.write_text('symbol,market_value,currency,quote_currency\nQQQ,750,USD,USD\nAAPL,250,USD,USD\n')
    status = main(['exposure', str(portfolio), '--base', 'USD',
                   '--tickerinside', '--as-of', '2026-11-07'])
    text = capsys.readouterr().out
    assert status == 0
    assert 'TickerInside fund-level aggregates' in text
    assert 'AAPL' in text
    assert 'MSFT' in text
    assert 'https://tickerinside.com' in text
