from __future__ import annotations

from decimal import Decimal

import pytest

from portfolio_risk.input import PortfolioFormatError, normalize_symbol, parse_portfolio


def _write(tmp_path, content, suffix=".csv"):
    path = tmp_path / f"portfolio{suffix}"
    path.write_text(content, encoding="utf-8")
    return path


def test_sample_file_imported():
    portfolio = parse_portfolio("examples/sample_portfolio.csv")
    assert len(portfolio.holdings) == 6
    assert portfolio.holdings[0].symbol == "XEQT.TO"
    assert portfolio.holdings[0].market_value == Decimal("8000")
    assert portfolio.holdings[0].currency == "CAD"


def test_txt_and_tsv_are_supported(tmp_path):
    for suffix, separator in [(".txt", ","), (".tsv", "\t")]:
        path = _write(
            tmp_path,
            f"symbol{separator}market_value{separator}currency\nqqq{separator}25.40{separator}usd\n",
            suffix,
        )
        result = parse_portfolio(path)
        assert result.holdings[0].symbol == "QQQ"
        assert result.holdings[0].market_value == Decimal("25.40")


def test_whitespace_and_header_case(tmp_path):
    path = _write(tmp_path, " Symbol , Market_Value , Currency \n brk.b , 19.90 , cad \n")
    assert parse_portfolio(path).holdings[0].symbol == "BRK.B"


def test_normalize_yahoo_style_symbols():
    assert normalize_symbol(" btc-usd ") == "BTC-USD"
    assert normalize_symbol("xEQt.to") == "XEQT.TO"
    assert normalize_symbol("usdcad=x") == "USDCAD=X"


@pytest.mark.parametrize("value", ["0", "-4", "NaN", "Infinity", "abc", ""])
def test_invalid_amounts_rejected(tmp_path, value):
    path = _write(tmp_path, f"symbol,market_value,currency\nQQQ,{value},USD\n")
    with pytest.raises(PortfolioFormatError):
        parse_portfolio(path)


@pytest.mark.parametrize("symbol", ["NOT A TICKER", "BAD/SYM", "", "QQQ;DROP"])
def test_invalid_symbols_rejected(tmp_path, symbol):
    path = _write(tmp_path, f"symbol,market_value,currency\n{symbol},10,USD\n")
    with pytest.raises(PortfolioFormatError):
        parse_portfolio(path)


def test_duplicate_symbol_rejected(tmp_path):
    path = _write(tmp_path, "symbol,market_value,currency\nQQQ,10,USD\nqqq,20,CAD\n")
    with pytest.raises(PortfolioFormatError, match="duplicate"):
        parse_portfolio(path)


def test_bad_currency_rejected(tmp_path):
    path = _write(tmp_path, "symbol,market_value,currency\nQQQ,10,EUR\n")
    with pytest.raises(PortfolioFormatError, match="CAD or USD"):
        parse_portfolio(path)


def test_missing_header_field_rejected(tmp_path):
    path = _write(tmp_path, "symbol,market_value\nQQQ,10\n")
    with pytest.raises(PortfolioFormatError, match="Missing required"):
        parse_portfolio(path)


def test_extra_unquoted_comma_rejected(tmp_path):
    path = _write(tmp_path, "symbol,market_value,currency\nQQQ,1,000,USD\n")
    with pytest.raises(PortfolioFormatError, match="incorrect number"):
        parse_portfolio(path)


def test_empty_portfolio_rejected(tmp_path):
    path = _write(tmp_path, "symbol,market_value,currency\n")
    with pytest.raises(PortfolioFormatError, match="no holdings"):
        parse_portfolio(path)
