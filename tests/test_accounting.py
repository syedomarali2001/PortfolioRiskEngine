from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from portfolio_risk.accounting import valuation_as_dict, value_portfolio
from portfolio_risk.fx import FXRateError, manual_fx_quote
from portfolio_risk.models import Holding, PortfolioInput


def make_input(*positions):
    return PortfolioInput(
        tuple(Holding(symbol, Decimal(value), currency) for symbol, value, currency in positions),
        "fixture.csv",
    )


def test_sample_exact_totals_weights_and_concentration():
    p = make_input(
        ("XEQT.TO", "8000", "CAD"),
        ("QQQ", "4000", "USD"),
        ("SCHD", "3000", "USD"),
        ("SPMO", "2000", "USD"),
        ("BTC-USD", "1500", "USD"),
        ("GLD", "1500", "USD"),
    )
    q = manual_fx_quote("1.35", date(2026, 11, 7))
    result = value_portfolio(p, base_currency="CAD", fx_quote=q)
    assert result.total_value == Decimal("24200.00")
    assert result.holdings[0].value_in_base == Decimal("8000")
    assert result.holdings[1].value_in_base == Decimal("5400.00")
    assert result.holdings[2].value_in_base == Decimal("4050.00")
    assert abs(sum((h.weight for h in result.holdings), Decimal(0)) - 1) < Decimal("1e-25")
    assert result.holdings[0].weight == Decimal("8000") / Decimal("24200.00")
    assert result.concentration.largest_weight == result.holdings[0].weight
    assert result.concentration.top_three_weight == Decimal("17450.00") / result.total_value
    assert result.concentration.effective_holdings == 1 / result.concentration.herfindahl_index
    assert dict(result.by_entered_currency) == {"CAD": Decimal("8000"), "USD": Decimal("16200.00")}


def test_base_usd_inverts_quote_and_uses_decimal():
    p = make_input(("XEQT.TO", "135", "CAD"), ("QQQ", "100", "USD"))
    out = value_portfolio(
        p, base_currency="USD", fx_quote=manual_fx_quote("1.35", date(2026, 11, 7))
    )
    assert out.total_value == Decimal("200")
    assert [h.weight for h in out.holdings] == [Decimal("0.5"), Decimal("0.5")]
    assert out.concentration.herfindahl_index == Decimal("0.50")
    assert out.concentration.effective_holdings == Decimal("2")
    assert out.concentration.top_three_weight == Decimal("1.0")


def test_all_same_currency_needs_no_quote():
    p = make_input(("A", "20", "CAD"), ("B", "80", "CAD"))
    val = value_portfolio(p, base_currency="CAD")
    assert val.total_value == 100
    assert val.fx_quote is None
    assert val.concentration.largest_weight == Decimal("0.8")
    assert val.concentration.effective_holdings == 1 / Decimal("0.68")


def test_mixed_currency_requires_rate():
    p = make_input(("A", "1", "CAD"), ("B", "1", "USD"))
    with pytest.raises(FXRateError, match="required"):
        value_portfolio(p, base_currency="CAD")


def test_one_holding_is_100_percent_and_hhi_one():
    p = make_input(("A", "50", "USD"))
    val = value_portfolio(p, base_currency="USD")
    assert val.holdings[0].weight == 1
    assert val.concentration.herfindahl_index == 1
    assert val.concentration.effective_holdings == 1


@pytest.mark.parametrize("bad", ["JPY", "", "EURO"])
def test_reject_unsupported_base_currency(bad):
    with pytest.raises(ValueError, match="base_currency"):
        value_portfolio(make_input(("A", "1", "CAD")), base_currency=bad)


def test_defensive_invalid_input_even_when_parser_is_bypassed():
    with pytest.raises(ValueError, match="positive"):
        value_portfolio(make_input(("A", "-1", "CAD")))
    with pytest.raises(ValueError, match="positive"):
        value_portfolio(make_input(("A", "NaN", "CAD")))


def test_json_formatting_preserves_decimal_text_and_meaning():
    q = manual_fx_quote("1.30", date(2026, 11, 7))
    p = make_input(("A", "100", "USD"), ("B", "70", "CAD"))
    val = valuation_as_dict(value_portfolio(p, base_currency="CAD", fx_quote=q))
    assert val["total_value"] == "200.00"
    assert val["holdings"][0]["value_in_base"] == "130.00"
    assert val["fx_quote"]["rate_cad_per_usd"] == "1.30"
    assert "not equivalent" in val["limitations"][1]


def test_decimals_are_not_rounded_before_weight_calculation():
    p = make_input(("A", "0.01", "CAD"), ("B", "0.02", "CAD"))
    val = value_portfolio(p)
    assert val.total_value == Decimal("0.03")
    assert val.holdings[0].weight == Decimal("0.01") / Decimal("0.03")
