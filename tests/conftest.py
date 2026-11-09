from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from portfolio_risk.models import PriceHistory


@pytest.fixture
def price_history_factory():
    def make(symbol="QQQ", values=(100.0, 101.0, 99.0), dates=None, start=None, end=None):
        if dates is None:
            dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
        return PriceHistory(
            symbol=symbol,
            prices=pd.Series(values, index=pd.to_datetime(dates), name="adjusted_close"),
            start=start or date(2026, 1, 5),
            end=end or date(2026, 1, 9),
            source="OfflineTestProvider",
            fetched_at=datetime.now(timezone.utc),
            quote_currency="USD",
            adjusted=True,
        )
    return make
