"""Orchestrates provider, validation and cache; independent of the user interface."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from portfolio_risk.cache import HistoryCache
from portfolio_risk.models import PriceHistory, QualityReport
from portfolio_risk.providers.base import MarketDataProvider
from portfolio_risk.quality import validate_history


@dataclass(frozen=True)
class HistoryResult:
    history: PriceHistory
    quality: QualityReport
    cached: bool


def get_validated_history(
    provider: MarketDataProvider,
    cache: HistoryCache,
    symbol: str,
    start: date,
    end: date,
    *,
    refresh: bool = False,
) -> HistoryResult:
    if start >= end:
        raise ValueError("start must be before end (end date is exclusive)")
    if not refresh:
        saved = cache.get(symbol, start, end)
        if saved is not None:
            try:
                return HistoryResult(saved, validate_history(saved), cached=True)
            except ValueError:
                pass  # corrupt cache content is never trusted
    history = provider.get_history(symbol, start, end)
    if history.symbol != symbol or history.start != start or history.end != end:
        raise ValueError(f"Provider returned mismatched metadata for {symbol}")
    if not history.adjusted:
        raise ValueError(f"{symbol}: expected adjusted daily closes")
    quality = validate_history(history)
    cache.put(history)
    return HistoryResult(history, quality, cached=False)
