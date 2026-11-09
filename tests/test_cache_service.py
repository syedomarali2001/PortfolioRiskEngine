from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from portfolio_risk.cache import HistoryCache
from portfolio_risk.service import get_validated_history


class FakeProvider:
    def __init__(self, history):
        self.history = history
        self.calls = 0

    def get_history(self, symbol, start, end):
        self.calls += 1
        return replace(self.history, symbol=symbol, start=start, end=end)


def test_cache_roundtrip(tmp_path, price_history_factory):
    cache = HistoryCache(tmp_path)
    original = price_history_factory()
    cache.put(original)
    loaded = cache.get(original.symbol, original.start, original.end)
    assert loaded is not None
    assert list(loaded.prices) == [100.0, 101.0, 99.0]
    assert loaded.quote_currency == "USD"
    assert cache.get("GLD", original.start, original.end) is None


def test_expired_cache_ignored(tmp_path, price_history_factory):
    cache = HistoryCache(tmp_path)
    history = replace(price_history_factory(), fetched_at=datetime.now(timezone.utc) - timedelta(days=3))
    cache.put(history)
    assert cache.get(history.symbol, history.start, history.end) is None


def test_broken_cache_ignored(tmp_path, price_history_factory):
    cache = HistoryCache(tmp_path)
    original = price_history_factory()
    cache.put(original)
    cache._path(original.symbol, original.start, original.end).write_text("{bad JSON")
    assert cache.get(original.symbol, original.start, original.end) is None


def test_service_reuses_cache_and_refreshes(tmp_path, price_history_factory):
    h = price_history_factory()
    provider = FakeProvider(h)
    cache = HistoryCache(tmp_path)
    a = get_validated_history(provider, cache, "QQQ", h.start, h.end)
    b = get_validated_history(provider, cache, "QQQ", h.start, h.end)
    c = get_validated_history(provider, cache, "QQQ", h.start, h.end, refresh=True)
    assert not a.cached
    assert b.cached
    assert not c.cached
    assert provider.calls == 2


def test_service_rejects_mismatched_provider_metadata(tmp_path, price_history_factory):
    h = price_history_factory(symbol="QQQ")
    class WrongProvider:
        def get_history(self, symbol, start, end):
            return h
    import pytest
    with pytest.raises(ValueError, match="mismatched"):
        get_validated_history(WrongProvider(), HistoryCache(tmp_path), "GLD", date(2026, 1, 5), date(2026, 1, 9))
