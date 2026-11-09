"""As-of FX alignment for future performance and backtesting modules.

Converts daily *observed* asset price dates. FX may be carried forward from
its last known close, within a user-defined maximum calendar-day age. Asset
prices are never forward-filled or back-filled by this function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_risk.fx import FXRateError, _currency


def _validated_series(series: pd.Series, label: str) -> pd.Series:
    if not isinstance(series, pd.Series) or series.empty:
        raise ValueError(f"{label}: nonempty pandas Series required")
    idx = series.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None:
        raise ValueError(f"{label}: expected timezone-naive daily DatetimeIndex")
    if not idx.is_monotonic_increasing or idx.has_duplicates or not idx.equals(idx.normalize()):
        raise ValueError(f"{label}: dates must be unique, sorted, and normalized")
    values = pd.to_numeric(series, errors="coerce")
    data = values.to_numpy(dtype=float)
    if not np.isfinite(data).all() or not (data > 0).all():
        raise ValueError(f"{label}: values must be positive and finite")
    return values.astype(float)


def convert_price_history_to_base(
    asset_prices: pd.Series,
    *,
    quote_currency: str,
    base_currency: str,
    usd_cad_prices: pd.Series | None = None,
    max_fx_age_days: int = 5,
) -> pd.Series:
    """Convert a quoted price history without FX look-ahead.

    ``usd_cad_prices`` means CAD per one USD, indexed by actual observation
    date. ``quote_currency`` must be known explicitly; the denomination of
    the user's *entered market value* is not sufficient to establish it.

    Does not determine a common portfolio trading calendar, impute missing
    asset prices, or calculate historical returns. Those belong in V3/V4.
    """
    src, dst = _currency(quote_currency), _currency(base_currency)
    asset = _validated_series(asset_prices, "asset_prices")
    if max_fx_age_days < 0:
        raise ValueError("max_fx_age_days must not be negative")
    if src == dst:
        return asset.copy()
    if usd_cad_prices is None:
        raise FXRateError("A USD/CAD daily history is required for historical conversion")
    rates = _validated_series(usd_cad_prices, "usd_cad_prices")

    # Use only FX observations on or before each asset price date (never bfill).
    observed_dates = pd.Series(rates.index, index=rates.index).reindex(asset.index, method="ffill")
    if observed_dates.isna().any():
        missing = asset.index[observed_dates.isna()][0].date()
        raise FXRateError(f"No historical FX quote on or before asset date {missing}")
    age_days = (asset.index - pd.DatetimeIndex(observed_dates.to_numpy())).days
    if (age_days > max_fx_age_days).any():
        stale_idx = int(np.argmax(age_days > max_fx_age_days))
        day = asset.index[stale_idx].date()
        raise FXRateError(
            f"Historical FX quote too old for asset date {day}: {age_days[stale_idx]} days"
        )
    aligned = rates.reindex(asset.index, method="ffill")
    converted = asset * aligned if src == "USD" else asset / aligned
    converted.name = asset_prices.name
    return converted
