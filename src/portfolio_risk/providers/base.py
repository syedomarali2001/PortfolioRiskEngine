"""Vendor-independent interface. Other providers may be added later."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from portfolio_risk.models import PriceHistory


class MarketDataError(RuntimeError):
    """Historical prices could not be obtained from the selected provider."""


class MarketDataProvider(Protocol):
    def get_history(self, symbol: str, start: date, end: date) -> PriceHistory:
        """Fetch daily adjusted closes for [start, end)."""
        ...
