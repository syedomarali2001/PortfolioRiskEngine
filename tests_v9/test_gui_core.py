import csv
import io
import json
import math

import pytest

from portfolio_risk_ui.core import (
    PortfolioError, dividend_summary, engine_rows, fingerprint, import_values_csv,
    normalized_symbol, pct, safe_portfolio_nominal_total, show_complete_dividend_yield,
    to_csv, validate_holdings,
)


@pytest.fixture
def usd_rows():
    return [{"symbol": "AAPL", "market_value": 1000, "currency": "USD"},
            {"symbol": "SPY", "market_value": 3000, "currency": "USD"}]


def test_single_hold():
    assert validate_holdings([{"symbol":"aapl", "market_value":20,"currency":"usd"}],"values")[0]["symbol"] == "AAPL"


def test_usd_total(usd_rows):
    assert safe_portfolio_nominal_total(usd_rows,"USD") == 4000


def test_do_not_sum_mixed_currencies():
    assert safe_portfolio_nominal_total([{"symbol":"AAPL","market_value":10,"currency":"USD"},
        {"symbol":"XEQT.TO","market_value":11,"currency":"CAD"}],"USD") is None


@pytest.mark.parametrize("invalid", ["", "a b", "AAPL; rm -rf", "AAPL/ZZ", "hello$", "AAA'B"])
def test_reject_unsafe_tickers(invalid):
    with pytest.raises(PortfolioError): normalized_symbol(invalid)


@pytest.mark.parametrize("good", ["AAPL", "XEQT.TO", "BRK-B", "BTC-USD", "USDCAD=X", "^GSPC"])
def test_supported_symbol_forms(good):
    assert normalized_symbol(good)==good


def test_duplicate_ticker():
    with pytest.raises(PortfolioError, match="Duplicate"):
        validate_holdings([{"symbol":"aapl","market_value":10,"currency":"USD"},
                           {"symbol":"AAPL","market_value":15,"currency":"CAD"}],"values")


def test_missing_holdings():
    with pytest.raises(PortfolioError):validate_holdings([],"values")


def test_configured_holding_limit():
    with pytest.raises(PortfolioError):validate_holdings([{"symbol":"SPY","market_value":10,"currency":"USD"},
       {"symbol":"AAPL","market_value":15,"currency":"USD"}],"values",max_holdings=1)


@pytest.mark.parametrize("value",[0,-100,float('nan'),float('inf'),"x",None])
def test_bad_values(value):
    with pytest.raises(PortfolioError):validate_holdings([{"symbol":"SPY","market_value":value,"currency":"USD"}],"values")


def test_bad_currency():
    with pytest.raises(PortfolioError):validate_holdings([{"symbol":"SPY","market_value":10,"currency":"EUR"}],"values")


def test_weight_exact_100():
    assert len(validate_holdings([{"symbol":"AAPL","weight":70,"currency":"USD"},
        {"symbol":"SPY","weight":30,"currency":"USD"}],"weights"))==2


def test_weight_does_not_normalize():
    with pytest.raises(PortfolioError,match="100%"):
        validate_holdings([{"symbol":"SPY","weight":90,"currency":"USD"}],"weights")


def test_weight_conversion():
    rows=engine_rows([{"symbol":"AAPL","weight":55,"currency":"USD"},
        {"symbol":"SPY","weight":45,"currency":"CAD"}],"weights","CAD",total_value=10000)
    assert rows == [{"symbol":"AAPL","market_value":5500,"currency":"CAD"},
                    {"symbol":"SPY","market_value":4500,"currency":"CAD"}]


def test_weight_total_reconciles():
    rows=engine_rows([{"symbol":"AAPL","weight":33.3333,"currency":"USD"},
        {"symbol":"SPY","weight":33.3333,"currency":"USD"},
        {"symbol":"QQQ","weight":33.3334,"currency":"USD"}],"weights","USD",total_value=100.01)
    assert sum(row["market_value"] for row in rows)==pytest.approx(100.01)


def test_zero_hypothetical_total_rejected():
    with pytest.raises(PortfolioError):engine_rows([{"symbol":"SPY","weight":100,"currency":"USD"}],"weights","USD",total_value=0)


def test_csv_roundtrip(usd_rows):
    text=to_csv(usd_rows)
    assert import_values_csv(text.encode()) == validate_holdings(usd_rows,"values")


def test_csv_bom():
    text="\ufeffsymbol,market_value,currency\nAAPL,100,USD\n"
    assert import_values_csv(text.encode())[0]["symbol"]=="AAPL"


def test_csv_wrong_headers():
    with pytest.raises(PortfolioError):import_values_csv(b"ticker,shares\nAAPL,10\n")


def test_csv_extra_columns_rejected():
    with pytest.raises(PortfolioError):import_values_csv(b"symbol,market_value,currency\nAAPL,10,USD,extra\n")


def test_csv_size_limit():
    with pytest.raises(PortfolioError):import_values_csv(b"x"*1_000_001)


def test_fingerprint_changes_with_allocation(usd_rows):
    x=fingerprint(usd_rows,"USD","2020-01-01","2021-01-01")
    usd_rows[0]["market_value"]=1001
    assert x != fingerprint(usd_rows,"USD","2020-01-01","2021-01-01")


def test_fingerprint_changes_with_history_dates(usd_rows):
    assert fingerprint(usd_rows,"USD","2020-01-01","2021-01-01") != fingerprint(usd_rows,"USD","2020-01-01","2022-01-01")


def test_income_requires_complete_coverage():
    summary={"income_coverage_fraction":0.50, "fully_covered_portfolio_yield":None}
    assert show_complete_dividend_yield(summary) is False


def test_income_complete_coverage():
    assert show_complete_dividend_yield({"income_coverage_fraction":1.0,"fully_covered_portfolio_yield":0.03}) is True


def test_pct_none():
    assert pct(None)=="Unavailable"


def test_dividend_report_passthrough():
    x={"income_summary":{"estimated_annual_gross_income_covered_holdings":100.25}}
    assert dividend_summary(x)["estimated_annual_gross_income_covered_holdings"]==100.25
