import pytest

from portfolio_risk.fx import fx_ticker


def test_fx_pairs():
    assert fx_ticker("USD", "CAD") == "USDCAD=X"
    assert fx_ticker("cad", "usd") == "CADUSD=X"
    assert fx_ticker("USD", "USD") is None


def test_unsupported_fx_currency():
    with pytest.raises(ValueError):
        fx_ticker("EUR", "USD")
