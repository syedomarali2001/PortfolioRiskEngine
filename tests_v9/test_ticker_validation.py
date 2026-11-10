"""Offline regression tests for source-verified manual ticker entry."""
import pandas as pd

from portfolio_risk_ui.ticker_validation import verify_ticker


def test_realistic_price_response_is_accepted():
    frame = pd.DataFrame({"Close": [101.2, 102.7]})
    result = verify_ticker(" xeqt.to ", fetch_prices=lambda sym: frame if sym == "XEQT.TO" else None)
    assert result.ok


def test_random_ticker_is_not_accepted():
    result = verify_ticker("MEOWDSFA", fetch_prices=lambda sym: pd.DataFrame())
    assert not result.ok
    assert "No recent market prices" in result.message


def test_provider_outage_fails_closed_without_exposing_traceback():
    def offline(symbol):
        raise RuntimeError("provider tokens and internal traceback must not be displayed")
    result = verify_ticker("AAPL", fetch_prices=offline)
    assert not result.ok
    assert "try again" in result.message.lower()
    assert "tokens" not in result.message


def test_nonpositive_or_nan_prices_rejected():
    frame = pd.DataFrame({"Close": [None, 0, -5.0]})
    assert not verify_ticker("AAPL", fetch_prices=lambda symbol: frame).ok


def test_missing_close_column_rejected():
    frame = pd.DataFrame({"Volume": [100, 200]})
    assert not verify_ticker("AAPL", fetch_prices=lambda symbol: frame).ok
