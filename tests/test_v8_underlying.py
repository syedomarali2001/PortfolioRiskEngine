from datetime import date, datetime, timedelta, timezone

import pytest

from portfolio_risk.exposure import Constituent, FundSnapshot, SecurityMetadata
from portfolio_risk.security_metadata import YahooSecurityMetadataProvider
from portfolio_risk.v8_underlying import enrich_top_equities

S = {
    "SPY": FundSnapshot(
        "SPY",
        (Constituent("AAPL", 0.60, "equity"), Constituent("MSFT", 0.4, "equity")),
        "mock",
        date(2026, 11, 5),
    ),
    "QQQ": FundSnapshot(
        "QQQ",
        (Constituent("AAPL", 0.5, "equity"), Constituent("NVDA", 0.5, "equity")),
        "mock",
        date(2026, 11, 5),
    ),
}


def test_enrichment_capped_ranked_no_guesses_and_cached(tmp_path):
    calls = []

    def getter(s):
        calls.append(s)
        return {
            "symbol": s,
            "quoteType": "EQUITY",
            "sector": "Technology",
            "country": "United States",
        }

    provider = YahooSecurityMetadataProvider(getter=getter)
    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    args = ({"SPY": 0.5, "QQQ": 0.5}, S, {})
    classes, stats = enrich_top_equities(
        *args, limit=2, metadata_provider=provider, cache_dir=tmp_path, now=now
    )
    assert stats["attempted"] == 2
    assert calls == ["AAPL", "NVDA"]  # weighted portfolio exposure AAPL=.55, NVDA=.25, MSFT=.2
    assert classes["AAPL"].sector == "technology"
    assert "MSFT" not in classes
    calls.clear()
    second, report = enrich_top_equities(
        *args, limit=2, metadata_provider=provider, cache_dir=tmp_path, now=now + timedelta(days=1)
    )
    assert calls == []
    assert report["cached"] == 2
    assert second["NVDA"].country == "US"
    calls.clear()
    enrich_top_equities(
        *args, limit=2, metadata_provider=provider, cache_dir=tmp_path, now=now + timedelta(days=9)
    )
    assert calls == ["AAPL", "NVDA"]


def test_non_equity_profile_not_accepted(tmp_path):
    provider = YahooSecurityMetadataProvider(
        getter=lambda s: {"symbol": s, "quoteType": "ETF", "sector": "Technology"}
    )
    classes, stats = enrich_top_equities(
        {"SPY": 1}, S, {}, limit=1, metadata_provider=provider, cache_dir=tmp_path
    )
    assert classes == {}
    assert stats["enriched_count"] == 0
    assert stats["unavailable"]


def test_do_not_recheck_existing_verified_metadata(tmp_path):
    provider = YahooSecurityMetadataProvider(
        getter=lambda s: (_ for _ in ()).throw(AssertionError("Do not fetch"))
    )
    classes, stats = enrich_top_equities(
        {"SPY": 1},
        S,
        {"AAPL": SecurityMetadata("AAPL", "equity", "technology", "US")},
        limit=0,
        metadata_provider=provider,
        cache_dir=tmp_path,
    )
    assert stats["attempted"] == 0
    assert classes["AAPL"].sector == "technology"


@pytest.mark.parametrize("limit", [-1, 201])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        enrich_top_equities({"SPY": 1}, S, {}, limit=limit)
