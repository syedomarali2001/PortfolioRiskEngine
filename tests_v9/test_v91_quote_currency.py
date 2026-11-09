"""V9.1 regression guards for market-currency metadata and CSV round-trips."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from portfolio_risk.input import parse_portfolio
from portfolio_risk_ui.backend import run_report
from portfolio_risk_ui.core import (PortfolioError, engine_rows,
                                     import_values_csv, to_csv, validate_holdings)


def row(**values):
    return {"symbol": "AAPL", "market_value": 1000, "currency": "CAD", **values}


def test_explicit_quote_currency_is_separate_from_investment_currency():
    result = validate_holdings([row(quote_currency="USD")], "values")
    assert result[0]["currency"] == "CAD"
    assert result[0]["quote_currency"] == "USD"


def test_auto_quote_currency_must_not_be_guessed():
    result = validate_holdings([row()], "values")
    assert "quote_currency" not in result[0]


def test_blank_quote_currency_is_auto():
    assert "quote_currency" not in validate_holdings([row(quote_currency="")], "values")[0]


def test_case_insensitive_explicit_quote_currency():
    assert validate_holdings([row(quote_currency="usd")], "values")[0]["quote_currency"] == "USD"


@pytest.mark.parametrize("invalid", ["EUR", "UNKNOWN", "1", "BTC", "auto-selected"])
def test_invalid_quote_currency_rejected(invalid):
    with pytest.raises(PortfolioError, match="quote currency"):
        validate_holdings([row(quote_currency=invalid)], "values")


def test_quote_currency_survives_weight_mode_conversion():
    rows = [{"symbol": "AAPL", "weight": 40, "currency": "CAD", "quote_currency": "USD"},
            {"symbol": "XEQT.TO", "weight": 60, "currency": "CAD", "quote_currency": "CAD"}]
    out = engine_rows(rows, "weights", "CAD", total_value=10000)
    assert out == [dict(symbol="AAPL", market_value=4000, currency="CAD", quote_currency="USD"),
                   dict(symbol="XEQT.TO", market_value=6000, currency="CAD", quote_currency="CAD")]


def test_quote_currency_survives_csv_export_and_import():
    rows = [row(quote_currency="USD"),
            {"symbol": "MSFT", "market_value": 500, "currency": "CAD"}]
    data = to_csv(rows).encode()
    assert csv.DictReader(io.StringIO(data.decode())).fieldnames == [
        "symbol", "market_value", "currency", "quote_currency"]
    out = import_values_csv(data)
    assert out[0]["quote_currency"] == "USD"
    assert "quote_currency" not in out[1]


def test_optional_quote_currency_in_real_v84_parser(tmp_path):
    csv_path = tmp_path / "holdings.csv"
    csv_path.write_text(to_csv([row(quote_currency="USD")]), encoding="utf-8")
    holdings = parse_portfolio(csv_path).holdings
    assert len(holdings) == 1
    assert holdings[0].currency == "CAD"
    assert holdings[0].quote_currency == "USD"


def test_gui_bridge_sends_explicit_quote_to_actual_v84_dividends():
    root = Path(__file__).resolve().parents[1]
    report = run_report("dividends", [
        dict(symbol="SCHD",market_value=10000,currency="USD",quote_currency="USD"),
        dict(symbol="AAPL",market_value=10000,currency="USD",quote_currency="USD"),
        dict(symbol="GLD",market_value=5000,currency="USD",quote_currency="USD")],
        "USD", ["--as-of","2025-11-07", "--snapshot-file",str(root/"examples"/"v84_dividend_demo.json")],
        root=root)
    assert report["income_summary"]["income_coverage_fraction"] == 1.0
    assert report["income_summary"]["estimated_annual_gross_income_covered_holdings"] == pytest.approx(600)


def test_live_typing_and_portfolio_cards_are_included():
    source = (Path(__file__).resolve().parents[1] / "portfolio_risk_ui" / "app.py").read_text()
    assert 'live="350ms"' in source
    assert "with st.container(border=True):" in source
    assert "Market quote" in source
    assert "from streamlit_searchbox import" not in source


def test_actual_v84_currency_resolution_reproduces_previous_failure():
    from portfolio_risk.performance_service import _quote_currency
    with pytest.raises(ValueError, match="quote currency unavailable"):
        _quote_currency("AAPL", None, None)
    assert _quote_currency("AAPL", "USD", None) == "USD"
