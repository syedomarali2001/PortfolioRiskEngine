from datetime import date, datetime, timezone

import pytest

from portfolio_risk.exposure import Constituent, FundSnapshot
from portfolio_risk.security_metadata import YahooSecurityMetadataProvider
from portfolio_risk.v8_underlying import enrich_top_equities


def test_sector_focus_individual_classification_not_issuer_label(tmp_path):
    snapshot = FundSnapshot(
        "FOO",
        (Constituent("TECH", 0.6, "equity"), Constituent("BANK", 0.4, "equity")),
        "synthetic",
        date(2026, 8, 1),
    )
    info = {
        "TECH": {
            "symbol": "TECH",
            "quoteType": "EQUITY",
            "sector": "Technology",
            "country": "United States",
        },
        "BANK": {
            "symbol": "BANK",
            "quoteType": "EQUITY",
            "sector": "Financial Services",
            "country": "United States",
        },
    }
    caller = []

    def getter(s):
        caller.append(s)
        return info[s]

    provider = YahooSecurityMetadataProvider(getter=getter)
    kwargs = {
        "portfolio_weights": {"FOO": 1.0},
        "snapshots": {"FOO": snapshot},
        "classifications": {},
        "limit": 2,
        "metadata_provider": provider,
        "cache_dir": tmp_path,
        "focus_sector": "financials",
        "fund_sector_pct": {"FOO": {"financials": 40.0, "technology": 60.0}},
        "now": datetime(2026, 11, 7, tzinfo=timezone.utc),
    }
    classes, report = enrich_top_equities(**kwargs)
    assert report["attempted"] == 2
    assert report["matching_sector_symbols"] == ["BANK"]
    assert classes["BANK"].sector == "financials"
    assert classes["TECH"].sector == "technology"
    assert report["note"].startswith("Sector focus ranks unknown")
    assert set(caller) == {"BANK", "TECH"}
    classes2, report2 = enrich_top_equities(
        **{
            **kwargs,
            "classifications": {},
            "metadata_provider": YahooSecurityMetadataProvider(
                getter=lambda s: (_ for _ in ()).throw(AssertionError("network"))
            ),
        }
    )
    assert report2["cached"] == 2
    assert classes2["BANK"].sector == "financials"


def test_focus_with_zero_reported_sector_is_no_op(tmp_path):
    snap = FundSnapshot("FUND", (Constituent("TECH", 1.0, "equity"),), "mock", date(2026, 8, 1))
    classes, report = enrich_top_equities(
        {"FUND": 1.0},
        {"FUND": snap},
        {},
        limit=20,
        focus_sector="financials",
        fund_sector_pct={"FUND": {"technology": 100.0}},
        cache_dir=tmp_path,
        metadata_provider=YahooSecurityMetadataProvider(
            getter=lambda s: (_ for _ in ()).throw(AssertionError("should not query"))
        ),
    )
    assert report["attempted"] == 0
    assert classes == {}


def test_no_silent_sector_inference_when_profile_missing(tmp_path):
    snap = FundSnapshot("FUND", (Constituent("BANK", 1.0, "equity"),), "mock", date(2026, 8, 1))
    classes, report = enrich_top_equities(
        {"FUND": 1.0},
        {"FUND": snap},
        {},
        limit=1,
        focus_sector="financials",
        fund_sector_pct={"FUND": {"financials": 100.0}},
        cache_dir=tmp_path,
        metadata_provider=YahooSecurityMetadataProvider(
            getter=lambda s: {"symbol": s, "quoteType": "ETF"}
        ),
    )
    assert report["matching_sector_symbols"] == []
    assert report["enriched_count"] == 0
    assert classes == {}


def test_sector_focus_reports_actual_portfolio_weights_separate_from_ranking(tmp_path):
    snap = FundSnapshot(
        "FUND",
        (Constituent("BANK", 0.4, "equity"), Constituent("TECH", 0.6, "equity")),
        "mock",
        date(2026, 8, 1),
    )
    provider = YahooSecurityMetadataProvider(
        getter=lambda symbol: {
            "symbol": symbol,
            "quoteType": "EQUITY",
            "sector": "Financial Services" if symbol == "BANK" else "Technology",
            "country": "United States",
        }
    )
    classes, report = enrich_top_equities(
        {"FUND": 0.5},
        {"FUND": snap},
        {},
        limit=2,
        focus_sector="financials",
        fund_sector_pct={"FUND": {"financials": 20.0, "technology": 80.0}},
        cache_dir=tmp_path,
        metadata_provider=provider,
    )
    assert report["selected_top_equity_weights"]["BANK"] == pytest.approx(0.2)
    assert report["selection_ranking_scores"]["BANK"] == pytest.approx(0.04)
    assert report["selected_top_equity_weights"]["TECH"] == pytest.approx(0.3)
    assert report["selection_ranking_scores"]["TECH"] == pytest.approx(0.06)
    assert report["matching_sector_symbols"] == ["BANK"]
    assert "ONLY for lookup prioritization" in report["ranking_method"]
