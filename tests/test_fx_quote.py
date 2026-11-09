from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from portfolio_risk.cache import HistoryCache
from portfolio_risk.fx import (
    FXQuote, FXRateError, convert_amount, latest_usd_cad_quote, manual_fx_quote,
    validate_usd_cad,
)
from portfolio_risk.models import PriceHistory


class FakeFXProvider:
    def __init__(self, observations):
        self.observations = observations
        self.calls = 0

    def get_history(self, symbol, start, end):
        self.calls += 1
        assert symbol == "USDCAD=X"
        index = pd.DatetimeIndex([pd.Timestamp(d) for d, v in self.observations
                                  if start <= date.fromisoformat(d) < end])
        prices = pd.Series([v for d, v in self.observations
                            if start <= date.fromisoformat(d) < end], index=index, dtype=float)
        return PriceHistory(symbol, prices, start, end, "Mock FX provider",
                            datetime.now(timezone.utc), "CAD", True)


def test_asof_weekend_uses_friday_not_monday(tmp_path):
    p = FakeFXProvider([("2026-11-06", 1.35), ("2026-11-09", 1.50)])
    q = latest_usd_cad_quote(p, HistoryCache(tmp_path), date(2026, 11, 7))
    assert q.observed_on == date(2026, 11, 6)
    assert q.as_of == date(2026, 11, 7)
    assert q.age_days == 1
    assert q.usd_cad == Decimal("1.35")
    assert p.calls == 1


def test_fx_cache_is_reused(tmp_path):
    provider = FakeFXProvider([("2026-11-06", 1.35)])
    cache = HistoryCache(tmp_path)
    first = latest_usd_cad_quote(provider, cache, date(2026, 11, 7))
    second = latest_usd_cad_quote(provider, cache, date(2026, 11, 7))
    assert not first.cached and second.cached
    assert provider.calls == 1


def test_reject_rate_that_is_too_old(tmp_path):
    provider = FakeFXProvider([("2026-10-28", 1.31)])
    with pytest.raises(FXRateError, match="old"):
        latest_usd_cad_quote(provider, HistoryCache(tmp_path), date(2026, 11, 7))


def test_no_observation_before_cutoff(tmp_path):
    provider = FakeFXProvider([])
    with pytest.raises(ValueError, match="empty price history"):
        latest_usd_cad_quote(provider, HistoryCache(tmp_path), date(2026, 11, 7))


def test_no_leakage_from_later_fx_observations():
    with pytest.raises(FXRateError, match="later"):
        FXQuote(Decimal("1.3"), date(2026, 1, 1), date(2026, 1, 2), "test")


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity", "abc"])
def test_bad_fx_quotes_rejected(value):
    with pytest.raises(FXRateError):
        validate_usd_cad(value)


def test_fx_arithmetic_and_identity():
    assert convert_amount(Decimal("100"), "USD", "CAD", Decimal("1.35")) == Decimal("135.00")
    assert convert_amount(Decimal("135"), "CAD", "USD", Decimal("1.35")) == Decimal("100")
    assert convert_amount(Decimal("50"), "CAD", "CAD", None) == Decimal("50")
    with pytest.raises(FXRateError):
        convert_amount(Decimal("1"), "USD", "CAD", None)
    with pytest.raises(ValueError):
        convert_amount(Decimal("1"), "JPY", "CAD", Decimal("1.35"))


def test_manual_rate_marks_its_origin():
    q = manual_fx_quote("1.33", date(2026, 11, 7))
    assert q.observed_on == q.as_of
    assert "not independently verified" in q.source
