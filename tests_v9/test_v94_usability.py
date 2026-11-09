"""V9.4 readability contract tests: display-only, never alter V8.4 math."""
from pathlib import Path

import pytest

from portfolio_risk_ui.v94_presentation import (
    RISK_METRIC_HELP, benchmark_status, composition_rows, exposure_notes,
    finding_for_people, nice_category, select_overview_findings,
    simplify_insights_warnings, snapshot_rows, support_for_people,
)


@pytest.mark.parametrize("metric,ours,benchmark,diff,status,expected", [
    ("cagr", .2052, .1424, .0628, "positive", "6.28 percentage points"),
    ("volatility", .184, .1766, -.0074, "neutral", "0.74 percentage points"),
    ("drawdown", .2809, .3372, .0563, "positive", "5.63 percentage points"),
    ("sharpe", 1.11, .84, .27, "positive", "0.27"),
    ("es95", .0271, .0269, -.0002, "neutral", "0.02 percentage points"),
])
def test_finding_keeps_units_and_backend_sentiment(metric,ours,benchmark,diff,status,expected):
    out=finding_for_people({"metric":metric,"benchmark":"SPY","sentiment":status,
           "portfolio_value":ours,"benchmark_value":benchmark,
           "favorable_difference":diff,"materiality_threshold":.02 if metric!="sharpe" else .2})
    assert out["status"]==status
    assert out["difference"]==expected
    assert "materiality threshold" not in out["comparison"].lower()
    assert "SPY" in out["headline"]


def test_neutral_does_not_mean_positive():
    result=finding_for_people({"metric":"sharpe","benchmark":"SPY","sentiment":"neutral",
              "portfolio_value":1.11,"benchmark_value":1.07,
              "favorable_difference":.04,"materiality_threshold":.2})
    assert result["status_label"]=="Similar result"
    assert "Sharpe ratio" in result["headline"]
    assert "was similar" in result["reason"]


def test_negative_finding_explicitly_unfavorable():
    result=finding_for_people({"metric":"volatility","benchmark":"RSP","sentiment":"negative",
              "portfolio_value":.19,"benchmark_value":.14,
              "favorable_difference":-.05,"materiality_threshold":.02})
    assert result["status_label"]=="Area to watch"
    assert "fluctuated more" in result["reason"]
    assert "2.00 percentage points" in result["rule"]


def test_missing_finding_never_invented():
    result=finding_for_people({"metric":"volatility","sentiment":"unavailable"})
    assert "Unavailable" in result["comparison"]
    assert result["difference"] is None


def test_summary_prioritizes_distinct_findings_and_negative():
    d={"negative":[{"metric":"volatility","benchmark":"RSP"}],
       "positive":[{"metric":"sharpe","benchmark":"SPY"},{"metric":"sharpe","benchmark":"RSP"},
                   {"metric":"sortino","benchmark":"SPY"}],
       "neutral":[{"metric":"es95","benchmark":"RSP"}]}
    x=select_overview_findings(d,4)
    assert len(x)==4
    assert x[0]["metric"]=="volatility"
    assert {v["metric"] for v in x}=={"volatility","sharpe","sortino","es95"}


def test_benchmark_status_uses_engine_classification():
    report={"diagnostics":{"positive":[{"metric":"volatility","benchmark":"SPY"}],
             "neutral":[{"metric":"sharpe","benchmark":"SPY"}],"negative":[]}}
    assert benchmark_status(report,"annualized_volatility")=="positive"
    assert benchmark_status(report,"sharpe")=="neutral"
    assert benchmark_status(report,"sortino") is None


@pytest.mark.parametrize("key", ["cagr","annualized_volatility","sharpe","drawdown", "sortino", "calmar", "es95", "beta"])
def test_explainer_includes_useful_context(key):
    assert len(RISK_METRIC_HELP[key][1]) >= 70


def test_sector_unknown_is_not_renormalized_or_conflated_with_gold():
    known,unknown=composition_rows({"technology":.6195,"UNKNOWN":.12,"industrials":.2605},"Sectors")
    assert unknown==pytest.approx(12.0)
    assert sum(r["Share of portfolio (%)"] for r in known)+unknown==pytest.approx(100.0)
    assert [r["Category"] for r in known]==["Technology","Industrials"]


def test_country_unknown_preserved_separately():
    known,unknown=composition_rows({"US":.85,"UNKNOWN":.15},"Countries (company domicile)")
    assert known[0]["Category"]=="United States"
    assert unknown==pytest.approx(15.0)


def test_asset_class_labels_are_human_readable():
    assert nice_category("equity","Asset classes")=="Stocks / equities"
    assert nice_category("commodity","Asset classes")=="Commodities"
    assert nice_category("communication_services","Sectors")=="Communication Services"


def test_bad_or_missing_weights_do_not_become_fake_exposures():
    known,unknown=composition_rows({"technology":float('nan'),"UNKNOWN":0,"energy":-.1},"Sectors")
    assert known==[] and unknown==0


def test_support_rows_leave_price_only_separate_from_unknown():
    rows=support_for_people([{"Ticker":"SPY","Constituent coverage":"supported","Reason":"Verified"},
        {"Ticker":"XEQT.TO","Constituent coverage":"price_only","Reason":"Canadian ETF"}])
    assert rows[0]["Available information"]=="Detailed coverage"
    assert rows[1]["Available information"]=="Price history only"
    assert rows[1]["Reason"]=="Canadian ETF"


def test_fund_snapshot_preserves_provenance_and_rounds():
    result=snapshot_rows({"SPY":{"identified_fraction":.99868,"snapshot_as_of":"2026-11-05",
                                   "all_positions":503,"source":"Issuer"}})
    assert result[0]["Identified holdings (%)"]==pytest.approx(99.87)
    assert result[0]["Holdings reported as of"]=="2026-11-05"
    assert result[0]["Holdings listed"]==503
    assert result[0]["Data source"]=="Issuer"


def test_exposure_notes_are_never_dropped():
    messages=["Issuer sector/country aggregates come from fund-level provider reports",
              "GLD: fund snapshot is 131 days old", "SPMO: parser error"]
    routine, caution=exposure_notes(messages)
    assert sorted(routine+caution)==sorted(messages)
    assert "SPMO: parser error" in caution


def test_expected_history_notes_grouped_without_erasure():
    entries=["GLD: SHORT_HISTORY: starts 1783 days after requested start",
             "CALENDAR_ALIGNMENT: only 2785 shared closes compared with 6753 days for AAPL",
             "Today's portfolio allocations applied retrospectively, not actual investor P&L.",
             "AAPL: STALE_END: last close too old"]
    summary, important=simplify_insights_warnings(entries)
    assert len(summary)==3
    assert important==[entries[-1]]


def test_source_has_grouped_findings_and_exposure_navigation():
    source=(Path(__file__).resolve().parents[1]/"portfolio_risk_ui"/"app.py").read_text(encoding="utf-8")
    for token in ['st.tabs([f"Strengths ({len(positives)})"',
                  "Strengths mean a favorable historical comparison",
                  'nav_descriptions = [',
                  'st.session_state.xray_active_section = name',
                  '"Data sources and coverage (optional)"',
                  "coverage_summary(max(0.0, 100.0-unknown), unknown)",
                  "st.badge(badge, color=color)"]:
        assert token in source
    assert "materiality threshold:" not in source.split('def _sentiment_card(',1)[1].split('def _diagnostics(',1)[0]
    assert '_warnings(r)' not in source.split('def page_exposure():',1)[1].split('def _income_table(',1)[0]
