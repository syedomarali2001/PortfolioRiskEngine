from pathlib import Path

import pytest

from portfolio_risk.input import PortfolioFormatError, parse_portfolio


def test_optional_quote_currency_v3_file():
    p = parse_portfolio("examples/sample_portfolio_v3.csv")
    assert len(p.holdings) == 6
    assert p.holdings[0].quote_currency == "CAD"
    assert p.holdings[1].quote_currency == "USD"


def test_v1_and_v2_legacy_file_keeps_optional_quote_none():
    p = parse_portfolio("examples/sample_portfolio.csv")
    assert all(h.quote_currency is None for h in p.holdings)


def test_bad_quote_currency_rejected(tmp_path: Path):
    f = tmp_path / "bad.csv"
    f.write_text("symbol,market_value,currency,quote_currency\nQQQ,10,CAD,EUR\n")
    with pytest.raises(PortfolioFormatError, match="quote_currency"):
        parse_portfolio(f)


def test_quote_currency_is_not_market_value_currency(tmp_path: Path):
    f = tmp_path / "different.csv"
    f.write_text("symbol,market_value,currency,quote_currency\nQQQ,100,CAD,USD\n")
    p = parse_portfolio(f)
    assert p.holdings[0].currency == "CAD"
    assert p.holdings[0].quote_currency == "USD"
