from __future__ import annotations

import pandas as pd
import pytest

from portfolio_risk.fx import FXRateError
from portfolio_risk.historical_fx import convert_price_history_to_base


def series(items, name="price"):
    return pd.Series(
        [v for d, v in items], index=pd.to_datetime([d for d, v in items]), dtype=float, name=name
    )


def test_weekend_crypto_uses_previous_friday_fx():
    crypto = series([("2026-11-06", 100), ("2026-11-07", 110), ("2026-11-08", 120)])
    rates = series([("2026-11-06", 1.35), ("2026-11-09", 1.50)])
    result = convert_price_history_to_base(
        crypto, quote_currency="USD", base_currency="CAD", usd_cad_prices=rates
    )
    assert list(result) == pytest.approx([135, 148.5, 162])
    assert list(result.index) == list(crypto.index)


def test_inverted_rate_converts_canadian_security_to_usd():
    canadian = series([("2026-11-06", 135)])
    fx = series([("2026-11-06", 1.35)])
    out = convert_price_history_to_base(
        canadian, quote_currency="CAD", base_currency="USD", usd_cad_prices=fx
    )
    assert out.iloc[0] == pytest.approx(100)


def test_same_currency_no_fx_required_and_no_side_effects():
    asset = series([("2026-11-06", 10)])
    out = convert_price_history_to_base(asset, quote_currency="USD", base_currency="USD")
    assert out.equals(asset)
    assert out is not asset


def test_missing_earlier_fx_rejected_no_lookahead():
    asset = series([("2026-11-06", 10)])
    fx = series([("2026-11-07", 1.40)])
    with pytest.raises(FXRateError, match="on or before"):
        convert_price_history_to_base(
            asset, quote_currency="USD", base_currency="CAD", usd_cad_prices=fx
        )


def test_stale_fx_rejected():
    asset = series([("2026-11-10", 10)])
    fx = series([("2026-11-01", 1.40)])
    with pytest.raises(FXRateError, match="too old"):
        convert_price_history_to_base(
            asset, quote_currency="USD", base_currency="CAD", usd_cad_prices=fx
        )


def test_malformed_price_or_fx_rejected():
    asset = series([("2026-11-06", -1)])
    fx = series([("2026-11-06", 1.4)])
    with pytest.raises(ValueError, match="positive"):
        convert_price_history_to_base(
            asset, quote_currency="USD", base_currency="CAD", usd_cad_prices=fx
        )
    asset = series([("2026-11-06", 10)])
    with pytest.raises(ValueError, match="positive"):
        convert_price_history_to_base(
            asset,
            quote_currency="USD",
            base_currency="CAD",
            usd_cad_prices=series([("2026-11-06", 0)]),
        )


def test_quote_currency_is_never_inferred_from_asset_ticker():
    asset = series([("2026-11-06", 10)])
    with pytest.raises(ValueError, match="Currency must"):
        convert_price_history_to_base(asset, quote_currency="UNKNOWN", base_currency="CAD")
