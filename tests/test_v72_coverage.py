"""V7.2 mock-only regression tests: no HTTP requests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from portfolio_risk.cli import main
from portfolio_risk.coverage_audit import AuditAsset, audit_summary, run_coverage_audit
from portfolio_risk.etf_audit import (
    AuditFund,
    etf_report,
    read_etf_universe,
    run_etf_audit,
)
from portfolio_risk.providers.tickerinside import (
    FundUnavailable,
    TickerInsideClient,
    TickerInsideError,
)
from portfolio_risk.security_metadata import (
    SecurityMetadataError,
    YahooSecurityMetadataProvider,
    category_match,
    parse_yahoo_profile,
)

DATA = Path(__file__).resolve().parents[1] / "examples/us_etf_universe_v72.csv"


def mock_payload(symbol, kind="stock", percentage=99.5, as_of="2026-11-03"):
    return {
        "ticker": symbol,
        "name": f"Mock {symbol}",
        "holdings_kind": kind,
        "holdings_as_of": as_of,
        "holdings": {"AAPL": percentage / 2, "MSFT": percentage / 2} if percentage else {},
        "source_layer": "issuer" if kind == "stock" else "sec_nport_bonds",
        "breakdown": {
            "sectors_pct": {"technology": 95} if kind == "stock" else {},
            "countries_pct": {"US": 99} if kind == "stock" else {},
        },
        "expense_ratio": 0.03,
    }


def fake_get(url):
    if url.endswith("index.json"):
        return {
            "as_of": "2026-11-06",
            "tickers": ["SPY", "BND", "QQQ", "MISSING", "OLD"],
            "coverage": {"universe": {"etfs": 5723, "covered": 5386}},
        }
    symbol = url.split("/fund/")[1].split(".json")[0].upper()
    if symbol == "SPY":
        return mock_payload("SPY")
    if symbol == "BND":
        return mock_payload("BND", kind="bond", percentage=45)
    if symbol == "QQQ":
        return mock_payload("QQQ", kind="stock", percentage=97.1, as_of="2026-04-01")
    if symbol == "MISSING":
        raise FundUnavailable("Record missing despite index")
    if symbol == "OLD":
        raise TickerInsideError("API throttled")
    raise AssertionError("Should never fetch " + symbol)


def api(tmp_path):
    return TickerInsideClient(tmp_path / "tcache", getter=fake_get)


def test_large_curated_universe_has_many_groups_and_unique_tickers():
    sample = read_etf_universe(DATA)
    assert len(sample) >= 200
    assert len({row.symbol for row in sample}) == len(sample)
    assert len({row.theme for row in sample}) >= 12
    for required in ("SPY", "QQQ", "BND", "GLD", "TQQQ", "ARKK", "IBIT", "EEM", "EFA", "XLF"):
        assert required in {row.symbol for row in sample}


@pytest.mark.parametrize(
    "body, msg",
    [
        ("symbol,theme,issuer\nSPY,Equity,Firm\nSPY,Other,Firm\n", "Duplicate"),
        ("symbol,theme,issuer\nXEQT.TO,Equity,Firm\n", "U.S."),
        ("symbol,theme,issuer\nBTC-USD,Equity,Firm\n", "U.S."),
        ("symbol,theme,issuer\nSPY,,Firm\n", "required"),
        ("symbol,theme\nSPY,Equity\n", "must contain"),
    ],
)
def test_universe_rejects_misleading_inputs(tmp_path, body, msg):
    f = tmp_path / "universe.csv"
    f.write_text(body)
    with pytest.raises(ValueError, match=msg):
        read_etf_universe(f)


def test_broad_audit_separates_completeness_staleness_and_kind(tmp_path):
    sample = tuple(
        AuditFund(s, "Mixed", "Issuer") for s in ("SPY", "QQQ", "BND", "ZZZZ", "MISSING", "OLD")
    )
    rows, idx = run_etf_audit(sample, api(tmp_path), as_of=date(2026, 11, 7), delay_ms=0)
    report = etf_report(rows, idx)
    stats = report["summary"]
    assert stats["total"] == 6
    assert stats["indexed"] == 5
    assert stats["with_dated_holdings"] == 3
    assert stats["near_full_weight_95pct"] == 2
    assert stats["current_120d"] == 2
    assert stats["equity_lookthrough_safe"] == 2
    assert stats["provider_errors"] == 2
    assert stats["not_indexed"] == 1
    assert report["provider_reported_index"]["coverage"]["universe"]["covered"] == 5386
    assert rows[0].source_url.startswith("https://tickerinside.com")
    assert rows[1].age_days > 120 and "Stale" in rows[1].note
    assert rows[2].holdings_kind == "bond" and rows[2].weight_pct == 45
    assert not rows[2].equity_lookthrough_safe
    assert rows[3].status == "not_indexed"
    assert rows[4].status == "index_record_missing"
    assert rows[5].status == "provider_error"


def test_audit_rejects_negative_delay(tmp_path):
    with pytest.raises(ValueError, match="delay-ms"):
        run_etf_audit((AuditFund("SPY", "Equity", "X"),), api(tmp_path), delay_ms=-5)


def test_audit_progress_and_categories(tmp_path):
    sample = (AuditFund("SPY", "Broad", "A"), AuditFund("BND", "Bonds", "B"))
    progress = []
    rows, info = run_etf_audit(
        sample,
        api(tmp_path),
        delay_ms=0,
        on_progress=lambda i, n, s: progress.append((i, n, s)),
        as_of=date(2026, 11, 7),
    )
    result = etf_report(rows, info)
    assert progress == [(1, 2, "SPY"), (2, 2, "BND")]
    assert result["by_theme"]["Bonds"]["partial_weight"] == 1
    assert result["by_issuer"]["A"]["with_sector_breakdown"] == 1


def test_parse_stock_profile():
    profile = parse_yahoo_profile(
        "AAPL",
        {
            "symbol": "AAPL",
            "quoteType": "EQUITY",
            "longName": "Apple Inc.",
            "country": "United States",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currency": "USD",
            "exchange": "NMS",
            "marketCap": 1000,
        },
    )
    assert profile.status == "ok"
    assert profile.asset_kind == "equity"
    assert profile.sector == "Technology"
    assert profile.industry == "Consumer Electronics"
    assert profile.company_country == "United States"
    assert category_match("US_STOCK", profile)
    assert not category_match("US_ETF", profile)


def test_parse_canadian_stock_and_no_sector_cryptocurrency():
    ca = parse_yahoo_profile(
        "RY.TO", {"quoteType": "EQUITY", "sector": "Financial Services", "country": "Canada"}
    )
    assert category_match("CAD_STOCK", ca) is True
    btc = parse_yahoo_profile(
        "BTC-USD", {"quoteType": "CRYPTOCURRENCY", "shortName": "Bitcoin USD"}
    )
    assert category_match("CRYPTO", btc) is True
    assert btc.sector is None and btc.company_country is None


@pytest.mark.parametrize(
    "qtype, kind",
    [
        ("ETF", "fund"),
        ("MUTUALFUND", "fund"),
        ("ETN", "fund"),
        ("FUTURE", "commodity_future"),
        ("CURRENCY", "currency"),
        ("INDEX", "index"),
    ],
)
def test_asset_kind_mapping(qtype, kind):
    p = parse_yahoo_profile("TEST", {"quoteType": qtype})
    assert p.asset_kind == kind


def test_missing_metadata_is_not_automatically_wrong_ticker():
    missing = parse_yahoo_profile("AAPL", {})
    assert missing.status == "missing"
    assert category_match("US_STOCK", missing) is None


def test_wrong_symbol_metadata_fails_closed():
    with pytest.raises(SecurityMetadataError, match="mismatched"):
        parse_yahoo_profile("AAPL", {"symbol": "MSFT", "quoteType": "EQUITY"})


def test_invalid_market_cap_ignored():
    for value in (-1, float("inf"), True, "100"):
        assert (
            parse_yahoo_profile("AAPL", {"quoteType": "EQUITY", "marketCap": value}).market_cap
            is None
        )


def test_yahoo_failures_report_unavailable_not_wrong_category():
    provider = YahooSecurityMetadataProvider(
        getter=lambda ticker: (_ for _ in ()).throw(RuntimeError("Rate limited"))
    )
    record = provider.lookup("AAPL")
    assert record.status == "unavailable" and "Rate limited" in record.note
    assert category_match("US_STOCK", record) is None


def test_metadata_audit_records_fields_without_changing_etf_status(tmp_path):
    yahoo = YahooSecurityMetadataProvider(
        getter=lambda s: {
            "symbol": s,
            "quoteType": "EQUITY",
            "sector": "Technology",
            "industry": "Software",
            "country": "United States",
            "currency": "USD",
        }
    )
    rows = run_coverage_audit(
        (AuditAsset("AAPL", "US_STOCK", "USD"),),
        api(tmp_path),
        as_of=date(2026, 11, 7),
        check_metadata=True,
        metadata_provider=yahoo,
    )
    assert rows[0].holdings_status == "not_applicable"
    assert rows[0].metadata_status == "ok"
    assert rows[0].metadata_sector == "Technology"
    assert rows[0].metadata_category_match is True
    assert audit_summary(rows)["metadata_sector_available"] == 1


def test_metadata_audit_flags_type_mismatch(tmp_path):
    yahoo = YahooSecurityMetadataProvider(getter=lambda s: {"quoteType": "ETF"})
    rows = run_coverage_audit(
        (AuditAsset("AAPL", "US_STOCK", "USD"),),
        api(tmp_path),
        check_metadata=True,
        metadata_provider=yahoo,
    )
    assert rows[0].metadata_category_match is False
    assert audit_summary(rows)["category_mismatches"] == 1


def test_etf_cli_reports_summary_and_json(monkeypatch, tmp_path, capsys):
    import portfolio_risk.cli as cli

    monkeypatch.setattr(cli, "TickerInsideClient", lambda **kwargs: api(tmp_path))
    small = tmp_path / "selected.csv"
    small.write_text("symbol,theme,issuer\nSPY,Broad,A\nBND,Bonds,B\n")
    report = tmp_path / "out.json"
    assert main(["etf-audit", str(small), "--output", str(report), "--delay-ms", "0"]) == 0
    stdout = capsys.readouterr().out
    assert "near-full" in stdout
    assert "Bonds" in stdout
    import json

    obj = json.loads(report.read_text())
    assert obj["summary"]["total"] == 2
    assert obj["summary"]["near_full_weight_95pct"] == 1


def test_etf_cli_limit_and_errors(monkeypatch, tmp_path, capsys):
    import portfolio_risk.cli as cli

    monkeypatch.setattr(cli, "TickerInsideClient", lambda **kwargs: api(tmp_path))
    src = tmp_path / "u.csv"
    src.write_text("symbol,theme,issuer\nSPY,Broad,A\nBND,Bonds,B\n")
    assert (
        main(
            [
                "etf-audit",
                str(src),
                "--output",
                str(tmp_path / "r.json"),
                "--delay-ms",
                "0",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    import json

    assert json.loads((tmp_path / "r.json").read_text())["summary"]["total"] == 1
    assert main(["etf-audit", str(src), "--limit", "0"]) == 1


def test_yahoo_metadata_cli_integration(monkeypatch, tmp_path, capsys):
    import portfolio_risk.cli as cli

    monkeypatch.setattr(cli, "TickerInsideClient", lambda **kwargs: api(tmp_path))
    monkeypatch.setattr(
        cli,
        "YahooSecurityMetadataProvider",
        lambda: YahooSecurityMetadataProvider(
            getter=lambda s: {
                "symbol": s,
                "quoteType": "EQUITY",
                "sector": "Technology",
                "country": "United States",
            }
        ),
    )
    file = tmp_path / "audit.csv"
    file.write_text("symbol,category,quote_currency\nAAPL,US_STOCK,USD\n")
    target = tmp_path / "coverage.json"
    assert main(["coverage-audit", str(file), "--check-metadata", "--output", str(target)]) == 0
    import json

    result = json.loads(target.read_text())
    assert result["summary"]["metadata_checked"] == 1
    assert result["assets"][0]["metadata_sector"] == "Technology"
    assert "Yahoo metadata" in capsys.readouterr().out


def test_exposure_direct_stock_classification(monkeypatch, tmp_path, capsys):
    import portfolio_risk.cli as cli

    monkeypatch.setattr(
        cli,
        "YahooSecurityMetadataProvider",
        lambda: YahooSecurityMetadataProvider(
            getter=lambda s: {
                "quoteType": "EQUITY",
                "symbol": s,
                "sector": "Technology",
                "country": "United States",
            }
        ),
    )
    file = tmp_path / "portfolio.csv"
    file.write_text("symbol,market_value,currency,quote_currency\nAAPL,1000,USD,USD\n")
    assert main(["exposure", str(file), "--base", "USD", "--auto-stock-metadata", "--json"]) == 0
    import json

    report = json.loads(capsys.readouterr().out)
    assert report["by_sector"]["technology"] == pytest.approx(1)
    assert report["by_country"]["US"] == pytest.approx(1)


def test_default_coverage_audit_does_not_trigger_metadata(tmp_path):
    rows = run_coverage_audit((AuditAsset("AAPL", "US_STOCK", "USD"),), api(tmp_path))
    assert rows[0].metadata_status == "not_checked"
