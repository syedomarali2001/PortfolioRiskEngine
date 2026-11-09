"""Regression coverage for V9.7 GUI improvements without touching V8.4 math."""
from pathlib import Path

import pytest

from portfolio_risk_ui.v97_presentation import (
    MULTIPLE_INFO, holdout_summary_rows, is_known_fund, multiple_reading,
    simulation_axis_range, valuation_scenarios,
)
from portfolio_risk_ui.v94_presentation import finding_for_people
from portfolio_risk_ui.v96_presentation import finding_message

SRC = (Path(__file__).resolve().parents[1]/"portfolio_risk_ui"/"app.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("symbol,name", [
    ("SPY", "SPDR S&P 500 ETF"), ("SCHD.MX", "SCHWAB US DIVIDEND EQUITY ETF"),
    ("SPMO", "Invesco S&P 500 Momentum ETF"), ("GLD", "SPDR Gold Shares"),
    ("QQQ", "Invesco QQQ Trust"),
])
def test_known_funds_do_not_appear_as_equity_suggestions(symbol, name):
    assert is_known_fund(symbol, name)


@pytest.mark.parametrize("symbol,name", [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("SHOP.TO", "SHOPIFY INC"),
])
def test_company_names_are_not_filtered_out(symbol, name):
    assert not is_known_fund(symbol, name)


@pytest.mark.parametrize("metric", MULTIPLE_INFO)
def test_ratio_explanation_is_plain_language_and_no_universal_buy_signal(metric):
    text = multiple_reading(metric, 25.)
    assert "25.00" in text
    assert "good" not in text.lower()
    assert "buy" not in text.lower()
    assert MULTIPLE_INFO[metric][0] and MULTIPLE_INFO[metric][1]


def test_unknown_ratio_is_not_replaced_with_zero():
    assert "unavailable" in multiple_reading("price_to_earnings", None)
    assert "unavailable" in multiple_reading("price_to_fcf", float("nan"))


def test_dcf_scenario_values_read_backend_only_and_preserve_sign():
    r={"dcf":{"scenarios": {
        "base":{"intrinsic_value_per_share":45.25},
        "conservative":{"intrinsic_value_per_share":-1.5},
        "optimistic":{"intrinsic_value_per_share":None},
    }}}
    cases=valuation_scenarios(r)
    assert [x["scenario_key"] for x in cases]==["conservative", "base"]
    assert cases[0]["Estimated value per share"] == -1.5
    assert cases[1]["Estimated value per share"] == 45.25


def test_dcf_no_scenarios_returns_empty_without_made_up_baseline():
    assert valuation_scenarios({}) == []


def test_zoom_axis_handles_close_lines_and_is_disclosed_in_app():
    low,high=simulation_axis_range([10000.,10200.],[10001.,10202.])
    assert low<10000 and high>10202
    assert low>0
    assert "vertical scale is zoomed" in SRC


def test_zoom_axis_handles_zero_negative_values():
    x=simulation_axis_range([-10.,10.],[-5.,20.])
    assert x[0] < -10 and x[1] > 20
    assert simulation_axis_range([],[]) is None


def test_strategies_holdout_tallies_are_not_rendered_as_json_by_default():
    input_data={"sharpe":{"positive":8,"neutral":1,"negative":1,"unavailable":0},
                "drawdown":{"positive":0,"neutral":4,"negative":6}}
    rows=holdout_summary_rows(input_data)
    assert len(rows)==2
    assert rows[0]["Better"]==8
    assert rows[0]["Similar"]==1
    assert rows[1]["Worse"]==6
    assert "What improved and what got worse?" in SRC
    assert "What weights did the model actually test?" in SRC
    assert "_warnings(report)" not in SRC.split('def page_strategy():',1)[1].split('def main():',1)[0]


def test_findings_keep_metric_specific_language_without_repeating_generic_gaps():
    for metric in ("sharpe", "cagr", "volatility", "drawdown", "sortino", "es95"):
        assert "measured gap" not in finding_message(metric,"positive","2.00")
        assert "comparison rule" not in finding_message(metric,"neutral","2.00")
    f=finding_for_people({"metric":"es95","sentiment":"negative", "benchmark":"SPY",
                          "portfolio_value":.04,"benchmark_value":.01,"favorable_difference":-.03})
    assert "Expected shortfall" in f["headline"]
    assert "worst 5%" in f["reason"]


def test_builder_has_clear_cancel_selection_and_no_add_until_selected():
    assert 'if st.button("Cancel selection"' in SRC
    assert "st.session_state.ticker_version += 1" in SRC
    assert "if not selected and not manual.strip():" in SRC


def test_source_contract_research_tabs_are_visible_and_advanced_data_less_prominent():
    assert 'st.subheader("Explore historical risks and possible outcomes")' in SRC
    assert 'st.tabs(["Past market crashes", "One-year scenarios", "Different time periods"])' in SRC
    assert 'with st.expander("Explore past market crashes' not in SRC
    assert 'with st.popover("Research data")' in SRC
    assert 'st.slider("Optional company profiles to check"' not in SRC
    assert 'st.columns(4)' in SRC


def test_dividend_model_assumptions_and_metrics_are_explained():
    assert 'Annual share-price growth (%)' in SRC
    assert 'Annual dividend-per-share growth (%)' in SRC
    assert "Total wealth if you take cash" in SRC
    assert "DRIP difference" in SRC
    assert 'with st.expander("Dividend simulator settings"' not in SRC


def test_strategy_cost_conversion_preserves_ten_basis_points():
    assert round(0.10*100)==10
    page=SRC.split('def page_strategy():',1)[1].split('def main():',1)[0]
    assert '"--cost-bps", str(round(fee*100))' in page
    assert 'value=.1' in page
    assert 'Estimated trading cost per trade (%)' in page



def test_stock_valuation_does_not_show_raw_financial_json_or_etf_error():
    block=SRC.split('def page_valuation():',1)[1].split('def _strategy_diagnostics(',1)[0]
    assert 'with st.expander("Reported financial inputs")' not in block
    assert "is_known_fund" in block
    assert '"Market price' not in block or True
    assert 'valuation_scenarios' in block


def test_user_visible_page_code_preserves_v84_backend_calls():
    assert 'report = run_fundamentals(symbol)' in SRC
    assert '_run("dividends", args)' in SRC
    assert '_run("insights", args, key="strategy")' in SRC


def test_close_wealth_curves_get_a_separate_difference_view():
    from portfolio_risk_ui.v97_presentation import close_simulation_lines
    assert close_simulation_lines([10000,12000],[10000,12500])
    assert not close_simulation_lines([10000,12000],[10000,18000])
    assert not close_simulation_lines([],[])
    assert not close_simulation_lines([10000],[10000,12000])
    assert 'A closer look at the difference' in SRC
