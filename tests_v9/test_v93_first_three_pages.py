"""V9.3 first-three-page usability & no-fabricated-data regression checks."""
from datetime import date
from pathlib import Path

import pytest

from portfolio_risk.performance_service import _quote_currency
from portfolio_risk_ui.periods import AUTO_FETCH_START, actual_window_label, requested_window
from portfolio_risk_ui.quotes import (
    VERIFIED_LISTINGS, quote_currency_from_search, verified_currency,
)
from portfolio_risk_ui.presentation import (finding_display, inspect_risk_contribution,
                                             overview_notes)


class FakeTicker:
    def __init__(self, metadata=None, fast=None, info=None, fails=()):
        self.metadata=metadata
        self.fast=fast
        self.info=info
        self.fails=set(fails)

    def get_history_metadata(self):
        if "history" in self.fails:raise ValueError("Yahoo metadata blocked")
        return self.metadata

    def get_fast_info(self):
        if "fast" in self.fails:raise ValueError("Yahoo fast info blocked")
        return self.fast

    def get_info(self):
        if "info" in self.fails:raise ValueError("Yahoo quote blocked")
        return self.info


def test_spmo_currency_verified_from_issuer_without_yahoo():
    match = verified_currency("SPMO",ticker_factory=lambda symbol: (_ for _ in ()).throw(RuntimeError("shouldn't contact provider")))
    assert match.currency == "USD"
    assert "Invesco" in match.source
    # This is a separately verified listing, NOT an assumed USD input currency.
    assert _quote_currency("SPMO",match.currency,None)=="USD"


def test_other_tickers_are_not_assumed_usd_when_provider_unavailable():
    assert verified_currency("MYTICKER",lambda x: FakeTicker()) is None
    with pytest.raises(ValueError,match="currency unavailable"):
        _quote_currency("MYTICKER",None,None)


def test_search_result_currency_is_explicit_only():
    assert quote_currency_from_search({"currency":"usd"}).currency=="USD"
    assert quote_currency_from_search({"currency":"cad"}).currency=="CAD"
    assert quote_currency_from_search({"market":"us_market"}) is None
    assert quote_currency_from_search({"currency":"EUR"}) is None


def test_quote_lookup_prefers_confirmed_history_metadata():
    data=verified_currency("XEQT.TO",lambda x: FakeTicker(metadata={"currency":"CAD"},fast={"currency":"USD"}))
    assert data.currency == "CAD"
    assert data.source == "Yahoo history metadata"


def test_quote_lookup_uses_fast_quote_if_history_metadata_is_missing():
    data=verified_currency("TEST123",lambda x: FakeTicker(metadata={},fast={"currency":"CAD"}))
    assert data.currency=="CAD"


def test_quote_lookup_uses_fallback_profile_when_both_missing():
    data=verified_currency("SOME123",lambda x: FakeTicker(metadata={},fast={},info={"currency":"USD"}))
    assert data.currency=="USD"


def test_quote_lookup_ignores_bad_metadata_and_continues():
    data=verified_currency("ABCD123",lambda x: FakeTicker(fails={"history","fast"},info={"currency":"CAD"}))
    assert data.currency=="CAD"


def test_quote_lookup_handles_all_provider_failures_without_guess():
    assert verified_currency("ABCD123",lambda x: FakeTicker(fails={"history","fast","info"})) is None


def test_exact_auto_history_starts_well_before_spmo_inception():
    a,b=requested_window(True,date(2023,11,8),date(2026,11,8),date(2026,11,7))
    assert a==date(2000,1,1)==AUTO_FETCH_START
    assert b==date(2026,11,8)


def test_manual_history_window_is_not_overridden():
    a,b=requested_window(False,date(2021,1,2),date(2025,12,31),date(2026,11,7))
    assert (a,b)==(date(2021,1,2),date(2025,12,31))


def test_display_date_is_from_report_not_requested_start():
    report={"first_date":"2015-10-09","last_date":"2026-11-06"}
    assert actual_window_label(report)=="2015-10-09 to 2026-11-06"
    assert actual_window_label({})=="Shared history dates unavailable"


def test_no_data_or_old_2023_date_is_fabricated():
    assert actual_window_label({"first_date":None,"last_date":None})=="Shared history dates unavailable"


def test_percentage_and_risk_difference_keep_two_decimal_places():
    cards=inspect_risk_contribution({"risk_contribution":[
        {"symbol":"SPMO","weight":.285714,"share_of_total_risk":.323584}]})
    row=cards[0]
    assert f"{row['Share of volatility risk (%)']:.2f}%"=="32.36%"
    assert f"{row['Allocation (%)']:.2f}%"=="28.57%"


def test_unavailable_risk_does_not_get_zero_imputed():
    assert inspect_risk_contribution({"risk_contribution":[
        {"symbol":"SPY","weight":.3,"share_of_total_risk":None}]}) == []


def test_existing_metric_explanations_remain_in_report():
    f=finding_display({"metric":"cagr", "benchmark":"SPY", "sentiment":"positive",
                       "portfolio_value":.25,"benchmark_value":.20,
                       "favorable_difference":.05, "materiality_threshold":.02})
    assert "25.00%" in f["comparison"]
    assert "20.00%" in f["comparison"]
    assert "5.00 percentage points" in f["comparison"]


def test_methodology_warnings_are_preserved_not_deleted():
    methods,quality=overview_notes(["Today's portfolio allocations applied retrospectively, not actual investor P&L.",
                                    "SPMO: STALE_END: last close was too old"])
    assert len(methods)==1 and len(quality)==1


def test_gui_source_contains_expected_three_page_design_contracts():
    app=(Path(__file__).resolve().parents[1]/"portfolio_risk_ui"/"app.py").read_text(encoding="utf-8")
    for marker in ["Use longest shared history", "historical_window_text(report)",
                   "_allocation_donut(insights[\"input_weights\"])",
                   "SPY · S&P 500 ETF", "RSP · Equal-weight S&P 500 ETF",
                   '"Compare against", ["SPY", "RSP", "Both"]',
                   "Which investments drive the most ups and downs?",
                   "Expected shortfall (95%)", "Explore historical risks and possible outcomes",
                   'with st.expander("Analysis method and data notes"',
                   "_cached_quote_currency(holding[\"symbol\"])"]:
        assert marker in app
    assert "st.subheader(\"2. Set amount and add to portfolio\")" in app
    assert "use_container_width" not in app
    # Five yellow warnings at bottom of risk page were replaced by a closed expander.
    assert "_warnings(r)" not in app.split("def page_risk():",1)[1].split("def _bar_from_weights(",1)[0]
