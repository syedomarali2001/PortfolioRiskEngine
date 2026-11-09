"""Offline subprocess smoke tests using the user's *actual* uploaded V8.4 source.

No provider requests. Validates V8.4 CLI flags/output contracts, not live market data.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_risk_ui.backend import run_fundamentals, run_report
from portfolio_risk_ui.presentation import (holding_support_rows, inspect_risk_contribution,
                                            overlap_rows, read_exposure_breakdowns, what_if_flag)
from portfolio_risk_ui.scenarios import parse_group_limits, prepare_what_if
from portfolio_risk_ui.core import PortfolioError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="module")
def synthetic_dividends():
    rows = [dict(symbol="SCHD", market_value=10000, currency="USD"),
            dict(symbol="AAPL", market_value=10000, currency="USD"),
            dict(symbol="GLD", market_value=5000, currency="USD")]
    return run_report("dividends", rows, "USD", [
        "--as-of", "2025-11-07",
        "--snapshot-file", str(EXAMPLES / "v84_dividend_demo.json"),
        "--years", "10", "--price-growth", "4", "--dividend-growth", "3",
        "--tax-drag", "15"], root=ROOT)


def test_real_v84_dividend_cli_contract(synthetic_dividends):
    assert synthetic_dividends["version"] == "8.4"
    assert synthetic_dividends["income_summary"]["estimated_annual_gross_income_covered_holdings"] == 600
    assert synthetic_dividends["income_summary"]["income_coverage_fraction"] == 1


def test_real_v84_tax_assumption(synthetic_dividends):
    assert synthetic_dividends["simulation"]["assumed_effective_distribution_tax_drag"] == pytest.approx(.15)
    cash = synthetic_dividends["simulation"]["annual_results"][0]["cash"]
    assert cash["yearly_gross_distributions"] == pytest.approx(600)
    assert cash["yearly_net_cash_withdrawn"] == pytest.approx(510)


def test_real_v84_fundamental_command():
    r = run_fundamentals("DEMO", ["--file",str(EXAMPLES/"v8_fundamentals_demo.json")],root=ROOT)
    assert r["symbol"] == "DEMO"
    assert r["dcf"]["status"] == "scenario_analysis"
    assert "price_to_fcf" in r["multiples"]


def test_real_v84_exposure_cli_json_on_stdout():
    rows = [dict(symbol="AAPL", market_value=6000, currency="CAD"),
            dict(symbol="MSFT", market_value=4000, currency="CAD")]
    r = run_report("exposure", rows, "CAD", [
        "--as-of", "2026-11-07", "--holdings-file",str(EXAMPLES/"exposure_demo_holdings.csv"),
        "--assets-file",str(EXAMPLES/"exposure_demo_assets.csv"),"--drilldown"],root=ROOT)
    assert r["analysis_date"] == "2026-11-07"
    assert r["direct_weights"] == {"AAPL": pytest.approx(.6),"MSFT":pytest.approx(.4)}
    assert "by_sector" in r
    assert "asset_support" in r
    assert read_exposure_breakdowns(r)["Sectors"] == r["by_sector"]
    assert isinstance(overlap_rows(r), list)
    assert len(holding_support_rows(r)) == 2


def test_risk_display_reads_actual_v84_name():
    report={"risk_contribution":[{"symbol":"AAPL","weight":.30,"share_of_total_risk":.65},
                                  {"symbol":"GLD","weight":.70,"share_of_total_risk":.35}]}
    x=inspect_risk_contribution(report)
    assert x[0]["Ticker"] == "AAPL"
    assert x[0]["Share of volatility risk (%)"] == pytest.approx(65)


def test_what_if_rejects_missing_ticker():
    with pytest.raises(PortfolioError,match="include every"):
        prepare_what_if([{"Ticker":"SPY","Target weight (%)":100}], ["AAPL","SPY"])


def test_what_if_validates_one_hundred_percent():
    with pytest.raises(PortfolioError,match="100%"):
        prepare_what_if([{"Ticker":"SPY","Target weight (%)":90},
                         {"Ticker":"AAPL","Target weight (%)":8}], ["SPY","AAPL"])


def test_what_if_emits_v84_accepted_syntax():
    scenario = prepare_what_if([{"Ticker":"SPY","Target weight (%)":65},
                                {"Ticker":"AAPL","Target weight (%)":35}], ["SPY","AAPL"])
    assert scenario == "GUI_WhatIf:SPY=65.00000000,AAPL=35.00000000"


def test_group_caps_preserve_engine_syntax():
    assert parse_group_limits("bigtech:AAPL,MSFT=40",["AAPL","MSFT","SPY"]) == ["bigtech:AAPL,MSFT=40"]


def test_group_cap_no_duplicates():
    with pytest.raises(PortfolioError,match="only one"):
        parse_group_limits("tech:AAPL,MSFT=50\nother:MSFT,SPY=50",["AAPL","MSFT","SPY"])


def test_group_cap_only_user_holdings():
    with pytest.raises(PortfolioError,match="portfolio tickers"):
        parse_group_limits("tech:AAPL,META=30",["AAPL","MSFT"])


def test_group_cap_out_of_range():
    with pytest.raises(PortfolioError,match="between 0% and 100%"):
        parse_group_limits("tech:AAPL,MSFT=150",["AAPL","MSFT"])
