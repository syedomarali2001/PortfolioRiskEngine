"""V9.8 regressions: UX changes must not alter the validated V8.4 math."""
from pathlib import Path

import pytest

from portfolio_risk.v8_insights import OBJECTIVES
from portfolio_risk_ui.core import PortfolioError
from portfolio_risk_ui.scenarios import prepare_what_if, parse_group_limits
from portfolio_risk_ui.v98_presentation import (
    STRATEGY_GOALS, format_group_cap, holdout_chart_rows, holdout_takeaway,
    market_valuation_gap, scenario_allocation_check,
)

APP = (Path(__file__).resolve().parents[1] / "portfolio_risk_ui" / "app.py").read_text()


@pytest.mark.parametrize("total,valid", [(100., True), (100.02, False), (99.98, False),
                                       (99.999, False), (100.00001, True)])
def test_what_if_total_is_exact_and_user_notified(total, valid):
    rows = [{"Ticker": "AAPL", "Target weight (%)": 50},
            {"Ticker": "MSFT", "Target weight (%)": total - 50}]
    result, observed = scenario_allocation_check(rows)
    assert result is valid
    assert observed == pytest.approx(total)
    if not valid:
        with pytest.raises(PortfolioError):
            prepare_what_if(rows, ["AAPL", "MSFT"])


def test_alternative_is_automatically_run_or_blocked_not_silently_skipped():
    block=APP.split('def page_strategy():',1)[1].split('def main():',1)[0]
    assert 'scenario_invalid=True' in block
    assert 'disabled=scenario_invalid' in block
    assert "args.extend(['--scenario', scenario_rule])" in block
    assert 'This does not place trades.' in block



def test_six_strategy_goals_only_use_validated_v84_objectives():
    assert len(STRATEGY_GOALS) == 6
    assert "More annual growth" in STRATEGY_GOALS
    assert "Reduce losses on bad days" in STRATEGY_GOALS
    for spec, description in STRATEGY_GOALS.values():
        assert description
        for part in spec.split(","):
            key, weight = part.split(":")
            assert key in OBJECTIVES
            assert float(weight) > 0


def test_group_names_are_user_labels_not_predefined_and_ui_selects_tickers():
    group = format_group_cap('My tech', ['MSFT','AAPL'],42,['MSFT','AAPL','SPY'])
    assert group == 'My_tech:MSFT,AAPL=42'
    assert parse_group_limits(group,['MSFT','AAPL','SPY']) == [group]
    assert "st.multiselect('Choose investments for the group'" in APP
    assert 'This is a label you choose, not a predefined financial category.' in APP



@pytest.mark.parametrize("name,tickers,percent", [
    ("", ["AAPL"], 25), ("Bad!", ["AAPL"], 25),
    ("Tech", ["NOPE"], 25), ("Tech", [], 25), ("Tech", ["AAPL"], 120),
])
def test_invalid_group_rejected(name, tickers, percent):
    with pytest.raises(PortfolioError):
        format_group_cap(name, tickers, percent, ["AAPL"])


def test_dcf_gap_is_display_only_not_an_expected_return():
    gap = market_valuation_gap(530.61, 125.62)
    assert gap is not None
    assert gap["percent"] == pytest.approx(-76.325, abs=.005)
    assert gap["large"] is True
    assert gap["direction"] == "below"
    assert "not a price forecast" in APP.lower()
    assert "peer multiples are not currently verified" in APP.lower()
    assert market_valuation_gap(None, 125.62) is None
    assert market_valuation_gap(0., 100.) is None


def test_market_price_has_ticker_in_label_and_ebitda_is_expanded():
    assert 'f"{symbol} market price used for comparison"' in APP
    assert "earnings before interest, taxes, depreciation and amortization" in APP


def test_validation_chart_uses_backend_counts_and_rules():
    report = {"rolling_robustness": {
        "status": "evaluated", "fold_count": 20,
        "priorities": {"sharpe": .5, "volatility": .5},
        "consistent_improvement_on_all_priorities": False,
        "per_metric_tally": {
            "sharpe": {"positive": 7, "neutral": 4, "negative": 9, "unavailable": 0},
            "volatility": {"positive": 16, "neutral": 3, "negative": 1, "unavailable": 0},
        }
    }}
    rows = holdout_chart_rows(report)
    assert len(rows) == 2
    assert rows[0]["Better"] == 7 and rows[0]["Worse"] == 9
    assert "7 of 20" in holdout_takeaway(report)
    assert "9" in holdout_takeaway(report)
    assert "75%" in holdout_takeaway(report)
    assert 'go.Bar(name=label' in APP


def test_valuation_and_risk_explained_and_clear_hover_colors():
    assert "SPY sensitivity" in APP
    assert "Context only" in APP
    assert 'bgcolor="#19283F"' in APP
    assert 'color="#FFFFFF"' in APP
    assert "only a few hundred paths" in APP


def test_navigation_uses_cards_as_direct_controls_not_duplicate_tabs():
    assert 'st.session_state.xray_active_section = name' in APP
    assert 'st.button(name, key=f"exposure_nav_' in APP
    assert 'st.segmented_control("Choose an analysis"' not in APP
    assert 'portfolio_ready = _portfolio_signature() is not None' in APP
    assert 'icon=icon if portfolio_ready else ":material/lock:"' in APP
    assert '.st-key-cancel_selected_ticker button' in APP
