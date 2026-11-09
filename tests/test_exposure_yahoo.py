import pandas as pd
import pytest

from portfolio_risk.exposure import ExposureError
from portfolio_risk.exposure_yahoo import _top_holding_rows


def test_yahoo_dataframe_top_holdings():
    frame = pd.DataFrame(
        {"Name": ["Apple", "Microsoft"], "Holding Percent": [0.08, 0.05]}, index=["AAPL", "MSFT"]
    )
    result = _top_holding_rows(frame)
    assert result[0].symbol == "AAPL"
    assert result[0].fraction == 0.08
    assert result[1].asset_class == "UNKNOWN"


def test_yahoo_dataframe_symbol_column():
    frame = pd.DataFrame({"Symbol": ["AAPL"], "Holding Percent": [0.08]})
    assert _top_holding_rows(frame)[0].symbol == "AAPL"


def test_yahoo_missing_weights_fails_closed():
    frame = pd.DataFrame({"Name": ["Apple"]}, index=["AAPL"])
    with pytest.raises(ExposureError):
        _top_holding_rows(frame)


def test_yahoo_percent_8_vs_fraction_008_rejected():
    frame = pd.DataFrame({"Holding Percent": [8.0]}, index=["AAPL"])
    with pytest.raises(ExposureError, match="fraction"):
        _top_holding_rows(frame)


def test_yahoo_no_rows_fails():
    with pytest.raises(ExposureError):
        _top_holding_rows(pd.DataFrame())
