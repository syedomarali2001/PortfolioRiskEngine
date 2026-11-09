"""Fail-fast price checks, followed by explicit coverage/staleness warnings."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from portfolio_risk.models import PriceHistory, QualityReport


class PriceQualityError(ValueError):
    """Prices are unusable for quantitative calculations."""


def validate_history(history: PriceHistory) -> QualityReport:
    series = history.prices
    if not isinstance(series, pd.Series) or series.empty:
        raise PriceQualityError(f"{history.symbol}: empty price history")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise PriceQualityError(f"{history.symbol}: expected a DatetimeIndex")
    if series.index.tz is not None:
        raise PriceQualityError(f"{history.symbol}: dates must be timezone-naive local days")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise PriceQualityError(f"{history.symbol}: dates must be unique and sorted")
    if not series.index.equals(series.index.normalize()):
        raise PriceQualityError(f"{history.symbol}: daily indices must have midnight timestamps")
    clean = pd.to_numeric(series, errors="coerce")
    if clean.isna().any() or (~(clean > 0)).any():
        raise PriceQualityError(f"{history.symbol}: prices must be positive and finite")
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise PriceQualityError(f"{history.symbol}: prices must be finite")
    first = series.index[0].date()
    last = series.index[-1].date()
    if first < history.start or last >= history.end:
        raise PriceQualityError(
            f"{history.symbol}: provider returned data outside requested window"
        )
    warnings: list[str] = []
    if (first - history.start).days > 14:
        warnings.append(
            f"SHORT_HISTORY: starts {(first - history.start).days} days after requested start"
        )
    if (history.end - timedelta(days=1) - last).days > 7:
        warnings.append(f"STALE_END: last close is {last} (end-exclusive request {history.end})")
    if len(series) > 1:
        max_gap = max(
            (right - left).days for left, right in zip(series.index[:-1], series.index[1:], strict=False)
        )
        if max_gap > 10:
            warnings.append(f"LARGE_GAP: maximum gap between observations is {max_gap} days")
    return QualityReport(history.symbol, len(series), first, last, tuple(warnings))
