"""V7 deterministic coverage, nested funds, overlap, and validation tests."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from portfolio_risk.accounting import value_portfolio
from portfolio_risk.exposure import (
    UNKNOWN, Constituent, ExposureError, FundSnapshot, SecurityMetadata,
    analyze_exposure, exposure_as_dict,
)
from portfolio_risk.models import Holding, PortfolioInput


CUTOFF = date(2026, 11, 7)


def portfolio(*parts):
    return value_portfolio(PortfolioInput(
        tuple(Holding(s, Decimal(str(n)), "CAD") for s, n in parts), "test.csv"))


def snap(symbol, *holdings, as_of=date(2026, 11, 1), source="official issuer"):
    return FundSnapshot(symbol, tuple(holdings), source, as_of)


def h(symbol, percent, kind="equity", sector="Technology", country="US"):
    return Constituent(symbol, percent / 100, kind, sector, country)


def run(val, snaps=None, classes=None, **kwargs):
    return analyze_exposure(val, snapshots=snaps, classifications=classes,
                            analysis_date=CUTOFF, **kwargs)


def test_direct_stock_known_exposure():
    r = run(portfolio(("AAPL", 100)), classes={"AAPL": SecurityMetadata("AAPL", "equity", "Technology", "US")})
    assert r.identified_underlying["AAPL"] == 1
    assert r.by_sector["Technology"] == 1
    assert r.by_country["US"] == 1
    assert r.unknown_underlying == 0


def test_direct_unknown_is_not_falsely_classified():
    r = run(portfolio(("XYZ", 100)))
    assert r.by_asset_class[UNKNOWN] == 1
    assert r.by_sector[UNKNOWN] == 1
    assert r.identified_underlying["XYZ"] == 1
    assert r.unknown_underlying == 0  # direct ticker known, classification isn't


def test_known_crypto_not_assigned_a_stock_sector_or_country():
    r = run(portfolio(("BTC-USD", 100)), classes={"BTC-USD": SecurityMetadata("BTC-USD", "crypto")})
    assert r.by_asset_class["crypto"] == 1
    assert r.by_country[UNKNOWN] == 1


def test_partial_etf_unknown_not_conjured_as_known():
    r = run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 30))})
    assert r.identified_underlying["AAPL"] == pytest.approx(.30)
    assert r.unknown_underlying == pytest.approx(.70)
    assert r.by_sector[UNKNOWN] == pytest.approx(.70)
    assert r.coverage_by_fund["FUND"]["identified_fraction_of_fund"] == pytest.approx(.3)


def test_missing_fund_snapshot_is_unknown_not_fund_symbol_leaf():
    r = run(portfolio(("ETF", 100)), classes={"ETF": SecurityMetadata("ETF", "etf")})
    assert r.unknown_underlying == pytest.approx(1)
    assert r.identified_underlying == {}
    assert r.by_asset_class[UNKNOWN] == 1


def test_nested_etf_recursion_and_unknown_propagation():
    funds = {
        "A": snap("A", h("B", 40, "etf"), h("AAPL", 20)),
        "B": snap("B", h("MSFT", 50)),
    }
    r = run(portfolio(("A", 100)), funds)
    assert r.identified_underlying["AAPL"] == pytest.approx(.2)
    assert r.identified_underlying["MSFT"] == pytest.approx(.2)
    assert r.unknown_underlying == pytest.approx(.6)
    assert r.by_asset_class["equity"] == pytest.approx(.4)


def test_same_equity_inside_multiple_funds_aggregates_correctly():
    funds = {
        "A": snap("A", h("AAPL", 40)),
        "B": snap("B", h("AAPL", 20)),
    }
    r = run(portfolio(("A", 60), ("B", 40)), funds)
    assert r.identified_underlying["AAPL"] == pytest.approx(.32)
    assert r.unknown_underlying == pytest.approx(.68)
    assert r.pairwise_overlap[0]["identified_overlap_lower_bound"] == pytest.approx(.2)


def test_overlap_does_not_count_generic_cash_as_shared_security():
    funds = {
        "A": snap("A", h("CASH", 50, "cash", "", ""), h("AAPL", 10)),
        "B": snap("B", h("CASH", 70, "cash", "", ""), h("MSFT", 10)),
    }
    r = run(portfolio(("A", 50), ("B", 50)), funds)
    assert r.pairwise_overlap[0]["identified_overlap_lower_bound"] == 0
    assert r.pairwise_overlap[0]["common_identified_symbols"] == []


def test_overlap_is_fund_weight_percentage_not_portfolio_weight():
    funds = {
        "A": snap("A", h("AAPL", 20)),
        "B": snap("B", h("AAPL", 40)),
    }
    r = run(portfolio(("A", 10), ("B", 90)), funds)
    assert r.pairwise_overlap[0]["identified_overlap_lower_bound"] == pytest.approx(.2)


def test_full_known_funds_produce_zero_unknown():
    r = run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 60), h("MSFT", 40))})
    assert r.unknown_underlying == pytest.approx(0)
    assert r.by_sector["Technology"] == pytest.approx(1)


def test_partitions_sum_to_one():
    val = portfolio(("A", 1), ("BTC", 2), ("MSFT", 4))
    r = run(val, {"A": snap("A", h("AAPL", 30))},
            {"BTC": SecurityMetadata("BTC", "crypto"),
             "MSFT": SecurityMetadata("MSFT", "equity", "Technology", "US")})
    assert sum(r.by_asset_class.values()) == pytest.approx(1)
    assert sum(r.by_sector.values()) == pytest.approx(1)
    assert sum(r.by_country.values()) == pytest.approx(1)
    assert sum(r.identified_underlying.values()) + r.unknown_underlying == pytest.approx(1)


def test_stock_metadata_fills_missing_disclosure_sector():
    r = run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 100, "equity", "", ""))},
            {"AAPL": SecurityMetadata("AAPL", "equity", "Technology", "US")})
    assert r.by_sector["Technology"] == 1
    assert r.by_country["US"] == 1


def test_stale_snapshot_is_warning_not_automatic_exclusion():
    r = run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 100), as_of=date(2025, 1, 1))})
    assert any("days old" in x for x in r.warnings)
    assert r.identified_underlying["AAPL"] == 1


def test_unverified_asof_is_visible():
    r = run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 100), as_of=None)})
    assert r.coverage_by_fund["FUND"]["as_of"] is None
    assert any("unverified" in s.lower() for s in r.warnings)


def test_future_snapshot_rejected():
    with pytest.raises(ExposureError, match="AFTER"):
        run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 100), as_of=date(2026, 11, 8))})


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -0.1, 0, 1.01])
def test_invalid_weights_rejected(bad):
    with pytest.raises(ExposureError):
        run(portfolio(("FUND", 100)), {"FUND": snap("FUND", Constituent("AAPL", bad, "equity"))})


def test_constituents_over_100_percent_rejected():
    with pytest.raises(ExposureError, match="exceed"):
        run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 70), h("MSFT", 40))})


def test_duplicate_constituent_rejected():
    with pytest.raises(ExposureError, match="duplicate"):
        run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 20), h("AAPL", 10))})


def test_self_referential_fund_rejected():
    with pytest.raises(ExposureError, match="itself"):
        run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("FUND", 20, "etf"))})


def test_indirect_fund_cycle_rejected():
    funds = {"A": snap("A", h("B", 40, "etf")), "B": snap("B", h("A", 50, "etf"))}
    with pytest.raises(ExposureError, match="Cyclic"):
        run(portfolio(("A", 100)), funds)


def test_cycle_detected_even_when_child_mislabeled_equity():
    funds = {"A": snap("A", h("B", 50)), "B": snap("B", h("A", 50))}
    with pytest.raises(ExposureError, match="Cyclic"):
        run(portfolio(("A", 100)), funds)


def test_bad_snapshot_key_rejected():
    with pytest.raises(ExposureError, match="does not match"):
        run(portfolio(("ETF", 100)), {"ETF": snap("OTHER", h("AAPL", 30))})


def test_bad_max_age_rejected():
    with pytest.raises(ExposureError, match="nonnegative"):
        run(portfolio(("FUND", 100)), max_snapshot_age_days=-1)


def test_json_output_is_serializable():
    import json
    r = run(portfolio(("FUND", 100)), {"FUND": snap("FUND", h("AAPL", 25))})
    data = exposure_as_dict(r)
    assert json.loads(json.dumps(data, allow_nan=False))["unknown_underlying"] == pytest.approx(.75)
    assert data["analysis_date"] == CUTOFF.isoformat()


def test_demo_warning():
    r = run(portfolio(("FUND", 100)), demo_data=True)
    assert any("SYNTHETIC" in s for s in r.warnings)
