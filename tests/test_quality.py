from __future__ import annotations

from datetime import date

import pytest

from portfolio_risk.quality import PriceQualityError, validate_history


def test_good_history_passes(price_history_factory):
    report = validate_history(price_history_factory())
    assert report.passed
    assert report.observation_count == 3
    assert report.last_date == date(2026, 1, 7)


@pytest.mark.parametrize(
    "prices",
    [(1.0, 0.0, 3.0), (1.0, -2.0, 3.0), (1.0, float("nan"), 3.0), (1.0, float("inf"), 3.0)],
)
def test_nonpositive_or_nonfinite_prices_rejected(price_history_factory, prices):
    with pytest.raises(PriceQualityError):
        validate_history(price_history_factory(values=prices))


def test_out_of_window_rejected(price_history_factory):
    with pytest.raises(PriceQualityError, match="outside"):
        validate_history(price_history_factory(dates=["2026-01-04", "2026-01-06", "2026-01-07"]))


def test_duplicate_dates_rejected(price_history_factory):
    with pytest.raises(PriceQualityError, match="unique"):
        validate_history(price_history_factory(dates=["2026-01-05", "2026-01-05", "2026-01-07"]))


def test_warn_short_and_stale_history(price_history_factory):
    history = price_history_factory(start=date(2025, 1, 1), end=date(2026, 2, 1))
    result = validate_history(history)
    assert not result.passed
    assert any(w.startswith("SHORT_HISTORY:") for w in result.warnings)
    assert any(w.startswith("STALE_END:") for w in result.warnings)


def test_large_date_gap_warned(price_history_factory):
    history = price_history_factory(
        dates=["2026-01-05", "2026-01-20", "2026-01-21"], end=date(2026, 1, 23)
    )
    report = validate_history(history)
    assert any(w.startswith("LARGE_GAP:") for w in report.warnings)
