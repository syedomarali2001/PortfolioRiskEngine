"""V7.3: conservative limits, provider verified issuer stocks, unusual futures.

All tests use fake providers, never external HTTP.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from portfolio_risk.cli import main
from portfolio_risk.etf_audit import AuditFund, etf_report, run_etf_audit
from portfolio_risk.exposure import SecurityMetadata, analyze_exposure
from portfolio_risk.exposure_support import assess_exposure_support
from portfolio_risk.providers.tickerinside import (
    TickerInsideClient,
    TickerInsideError,
    parse_fund_record,
)
from portfolio_risk.security_metadata import YahooSecurityMetadataProvider

TODAY = date(2026, 11, 7)


def record(symbol="SPY", kind="unknown", holdings=None, category=True, **extras):
    data = {
        "ticker": symbol,
        "name": f"Test {symbol}",
        "holdings_kind": kind,
        "source_layer": "issuer",
        "holdings_as_of": "2026-11-05",
        "holdings": holdings or {"AAPL": 70.0, "MSFT": 29.8},
        "breakdown": {
            "sectors_pct": {"technology": 70, "financials": 29.8},
            "countries_pct": {"US": 99.8},
            "sector_coverage_pct": 99.8,
            "other_funds_pct": 0.0,
        },
    }
    if category:
        data["category"] = (
            category
            if isinstance(category, dict)
            else {
                "asset_class": "equity",
                "basis": "holdings",
                "as_of": "2026-11-05",
                "measured": {"stocks": 2},
            }
        )
    data.update(extras)
    return data


def fake_fund(symbol):
    if symbol == "BND":
        return record(symbol, "bond", {"US TREASURY 2028": 40.0}, False)
    if symbol == "GLD":
        return record(symbol, "commodity", {"GOLD": 100.0}, False)
    if symbol == "TQQQ":
        return record(symbol, "derivative", {"ESZ6": -60.0, "AAPL": 70}, False)
    return record(symbol)


def fake_http(url):
    if url.endswith("/index.json"):
        return {"tickers": ["SPY", "BND", "GLD", "TQQQ"]}
    sym = url.split("/fund/")[1].split(".json")[0].upper()
    return fake_fund(sym)


def api(tmp_path):
    return TickerInsideClient(tmp_path / "cache", getter=fake_http)


def test_verified_issuer_equity_can_be_expanded_even_if_kind_unknown():
    fund = parse_fund_record(record(), "SPY")
    assert fund.holdings_kind == "unknown"
    assert fund.equity_validation == "issuer_evidence"
    assert fund.exposure_support == "equity"
    snap = fund.to_snapshot()
    assert len(snap.holdings) == 2
    assert all(h.asset_class == "equity" for h in snap.holdings)


@pytest.mark.parametrize(
    "change",
    [
        {"holdings_kind": "derivative"},
        {
            "category": {
                "asset_class": "bond",
                "basis": "holdings",
                "as_of": "2026-11-05",
                "measured": {"stocks": 2},
            }
        },
        {
            "category": {
                "asset_class": "equity",
                "basis": "market_data",
                "as_of": "2026-11-05",
                "measured": {"stocks": 2},
            }
        },
        {
            "category": {
                "asset_class": "equity",
                "basis": "holdings",
                "as_of": "2026-08-31",
                "measured": {"stocks": 2},
            }
        },
        {"breakdown": {"sector_coverage_pct": 45, "other_funds_pct": 0}},
        {"breakdown": {"sector_coverage_pct": 100, "other_funds_pct": 12}},
    ],
)
def test_bad_or_unverified_fund_evidence_never_turns_into_equity(change):
    f = parse_fund_record(record(**change), "SPY")
    assert f.exposure_support == "price_only"
    with pytest.raises(TickerInsideError):
        f.to_snapshot()


def test_signed_futures_positions_preserved_as_exclusion_not_invented_shares():
    doc = record("SPY", holdings={"AAPL": 99, "ESZ6": -9, "MFSZ6": 1})
    rec = parse_fund_record(doc, "SPY")
    assert rec.known_pct == 99
    assert len(rec.non_equity_positions) == 2
    assert rec.equity_validation == "mixed_positions"
    assert rec.exposure_support == "price_only"
    with pytest.raises(TickerInsideError):
        rec.to_snapshot()


def test_negative_actual_stock_still_rejected():
    with pytest.raises(TickerInsideError, match="AAPL"):
        parse_fund_record(record(holdings={"AAPL": -20}), "SPY")


def test_only_physical_and_spot_assets_expand_as_asset_classes():
    metal = parse_fund_record(fake_fund("GLD"), "GLD")
    assert metal.exposure_support == "spot_asset"
    assert metal.to_snapshot().holdings[0].asset_class == "commodity"
    coin = parse_fund_record(record("IBIT", "commodity", {"BITCOIN": 100}, False), "IBIT")
    assert coin.to_snapshot().holdings[0].asset_class == "crypto"
    futures = parse_fund_record(fake_fund("TQQQ"), "TQQQ")
    assert futures.exposure_support == "price_only"
    assert parse_fund_record(fake_fund("BND"), "BND").exposure_support == "price_only"


def test_support_contract_stale_partial_and_fund_exclusion(tmp_path):
    docs = {s: api(tmp_path).fund(s) for s in ("SPY", "BND", "GLD")}
    snaps = {
        s: rec.to_snapshot() for s, rec in docs.items() if rec.exposure_support != "price_only"
    }
    decisions = assess_exposure_support(
        ["SPY", "BND", "GLD", "XEQT.TO", "AAPL", "BTC-USD"],
        classifications={
            "XEQT.TO": SecurityMetadata("XEQT.TO", "etf"),
            "AAPL": SecurityMetadata("AAPL", "equity", "Technology", "US"),
            "BTC-USD": SecurityMetadata("BTC-USD", "crypto"),
        },
        snapshots=snaps,
        provider_records=docs,
        cutoff=TODAY,
        max_age_days=120,
    )
    assert decisions["SPY"]["exposure_status"] == "supported"
    assert decisions["BND"]["exposure_status"] == "price_only"
    assert decisions["XEQT.TO"]["exposure_status"] == "price_only"
    assert decisions["BTC-USD"]["exposure_status"] == "supported"
    assert decisions["AAPL"]["exposure_status"] == "supported"
    older = assess_exposure_support(
        ["SPY"],
        classifications={},
        snapshots=snaps,
        provider_records=docs,
        cutoff=TODAY,
        max_age_days=1,
    )
    assert older["SPY"]["exposure_status"] == "partial"


def test_etf_audit_accepts_verified_issuer_and_keeps_futures_diagnostic(tmp_path):
    entries = tuple(AuditFund(s, "test", "test") for s in ("SPY", "TQQQ", "BND", "GLD"))
    rows, meta = run_etf_audit(entries, api(tmp_path), as_of=TODAY, delay_ms=0)
    report = etf_report(rows, meta)
    assert rows[0].equity_lookthrough_safe
    assert rows[0].equity_validation == "issuer_evidence"
    assert rows[1].excluded_non_equity_positions == 1
    assert rows[1].status == "holdings_available"  # formerly a provider error
    assert rows[1].exposure_support == "price_only"
    assert report["summary"]["provider_verified_equity"] == 1
    assert report["summary"]["mixed_non_equity_positions"] == 1


def fake_profile(symbol):
    if symbol == "AAPL":
        return {
            "symbol": symbol,
            "quoteType": "EQUITY",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
        }
    if symbol == "XEQT.TO":
        return {"symbol": symbol, "quoteType": "ETF", "longName": "Example CAD ETF"}
    if symbol == "BTC-USD":
        return {"symbol": symbol, "quoteType": "CRYPTOCURRENCY"}
    return {"symbol": symbol, "quoteType": "ETF"}


def run_portfolio(monkeypatch, tmp_path, capsys, lines, strict=False):
    import portfolio_risk.cli as cli

    monkeypatch.setattr(cli, "TickerInsideClient", lambda **kwargs: api(tmp_path))
    monkeypatch.setattr(
        cli,
        "YahooSecurityMetadataProvider",
        lambda: YahooSecurityMetadataProvider(getter=fake_profile),
    )
    f = tmp_path / "p.csv"
    f.write_text(
        "symbol,market_value,currency,quote_currency\n"
        + "\n".join(f"{sym},{value},USD,USD" for sym, value in lines)
        + "\n"
    )
    args = [
        "exposure",
        str(f),
        "--base",
        "USD",
        "--tickerinside",
        "--auto-stock-metadata",
        "--as-of",
        "2026-11-07",
        "--json",
    ]
    if strict:
        args.append("--strict-exposure")
    code = main(args)
    stdout, stderr = capsys.readouterr()
    return code, json.loads(stdout) if code == 0 else None, stderr


def test_cli_mixes_stock_with_issuer_etf_aggregate_and_normalizes_sector(
    monkeypatch, tmp_path, capsys
):
    status, result, err = run_portfolio(
        monkeypatch, tmp_path, capsys, [("SPY", 600), ("AAPL", 400)]
    )
    assert status == 0, err
    assert result["asset_support"]["SPY"]["exposure_status"] == "supported"
    assert result["asset_support"]["AAPL"]["exposure_status"] == "supported"
    assert result["by_sector"]["technology"] == pytest.approx(0.6 * 0.7 + 0.4)
    assert result["by_sector"]["financials"] == pytest.approx(0.6 * 0.298)
    assert result["by_country"]["US"] == pytest.approx(0.6 * 0.998 + 0.4)
    assert result["coverage_by_fund"]["SPY"]["sector_country_source"].startswith("TickerInside")
    assert result["restricted_exposure_assets"] == []
    assert "AAPL" in result["identified_underlying"]
    assert "ETF sector/country aggregates" in " ".join(result["warnings"])


def test_cli_warns_for_canadian_fund_and_bonds_without_claiming_prices_fail(
    monkeypatch, tmp_path, capsys
):
    status, result, err = run_portfolio(
        monkeypatch, tmp_path, capsys, [("XEQT.TO", 500), ("BND", 500)]
    )
    assert status == 0, err
    assert result["asset_support"]["XEQT.TO"]["exposure_status"] == "price_only"
    assert result["asset_support"]["BND"]["exposure_status"] == "price_only"
    assert result["asset_support"]["BND"]["price_risk_status"] == "not_checked_here"
    assert result["unknown_underlying"] == pytest.approx(1)
    status, doc, err = run_portfolio(
        monkeypatch, tmp_path, capsys, [("XEQT.TO", 500), ("BND", 500)], True
    )
    assert status == 1
    assert "Strict exposure rejected" in err


def test_cli_supported_stock_crypto_and_gold_can_pass_strict(monkeypatch, tmp_path, capsys):
    status, result, err = run_portfolio(
        monkeypatch, tmp_path, capsys, [("AAPL", 300), ("BTC-USD", 200), ("GLD", 500)], True
    )
    assert status == 0, err
    assert result["by_asset_class"]["commodity"] == pytest.approx(0.5)
    assert result["by_asset_class"]["crypto"] == pytest.approx(0.2)
    assert result["by_asset_class"]["equity"] == pytest.approx(0.3)
    assert result["pairwise_overlap"] == []


def test_cli_rejects_unknown_when_strict(monkeypatch, tmp_path, capsys):
    status, doc, err = run_portfolio(monkeypatch, tmp_path, capsys, [("UNIDENTIFIED", 100)], True)
    assert status == 1
    assert "Strict exposure rejected" in err


def test_small_cash_position_is_reported_as_cash_without_false_equity_overlap():
    p = record("SPY", holdings={"AAPL": 99.5, "CASH": 0.5})
    fund = parse_fund_record(p, "SPY")
    assert fund.exposure_support == "equity"
    assert fund.known_pct == 100
    snap = fund.to_snapshot()
    assert [(h.symbol, h.asset_class) for h in snap.holdings] == [
        ("AAPL", "equity"),
        ("CASH", "cash"),
    ]
    assert sum(h.fraction for h in snap.holdings) == pytest.approx(1)


def test_negative_cash_suggests_leverage_and_blocks_lookthrough():
    fund = parse_fund_record(record("SPY", holdings={"AAPL": 100, "CASH": -3}), "SPY")
    assert fund.exposure_support == "price_only"
    assert "CASH" in fund.non_equity_positions[0]


def test_unknown_and_bond_symbols_never_count_as_stock_overlap():
    from decimal import Decimal

    from portfolio_risk.accounting import value_portfolio
    from portfolio_risk.exposure import Constituent, FundSnapshot
    from portfolio_risk.models import Holding, PortfolioInput

    p = PortfolioInput(
        (Holding("FUND_A", Decimal("50"), "USD"), Holding("FUND_B", Decimal("50"), "USD")),
        "fixture.csv",
    )
    val = value_portfolio(p, base_currency="USD")
    funds = {
        "FUND_A": FundSnapshot(
            "FUND_A",
            (
                Constituent("BOND1", 0.4, "bond"),
                Constituent("UNKNOWN1", 0.3, "UNKNOWN"),
                Constituent("AAPL", 0.3, "equity"),
            ),
            "fixture",
            TODAY,
        ),
        "FUND_B": FundSnapshot(
            "FUND_B",
            (
                Constituent("BOND1", 0.4, "bond"),
                Constituent("UNKNOWN1", 0.3, "UNKNOWN"),
                Constituent("AAPL", 0.2, "equity"),
            ),
            "fixture",
            TODAY,
        ),
    }
    output = analyze_exposure(val, snapshots=funds, analysis_date=TODAY)
    assert output.pairwise_overlap[0]["identified_overlap_lower_bound"] == pytest.approx(0.2)
    assert output.pairwise_overlap[0]["common_identified_symbols"] == ["AAPL"]


def test_manual_bond_snapshot_is_price_only_even_with_all_weights():
    from portfolio_risk.exposure import Constituent, FundSnapshot

    snap = FundSnapshot("BND", (Constituent("UST", 1, "bond"),), "manual file", TODAY)
    result = assess_exposure_support(
        ["BND"],
        classifications={},
        snapshots={"BND": snap},
        provider_records={},
        cutoff=TODAY,
        max_age_days=120,
    )
    assert result["BND"]["exposure_status"] == "price_only"


def test_nested_unknown_exposure_blocks_strict_full_claim():
    from portfolio_risk.exposure import Constituent, FundSnapshot

    snap = FundSnapshot("ETF", (Constituent("NESTED", 1, "etf"),), "manual file", TODAY)
    result = assess_exposure_support(
        ["ETF"],
        classifications={},
        snapshots={"ETF": snap},
        provider_records={},
        cutoff=TODAY,
        max_age_days=120,
        effective_coverage={"ETF": 0.0},
    )
    assert result["ETF"]["exposure_status"] == "partial"
    assert "nested" in result["ETF"]["reason"]
