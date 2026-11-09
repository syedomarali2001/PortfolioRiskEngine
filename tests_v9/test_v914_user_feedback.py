"""V9.14 regression: no duplicate Plotly element ids, clear status, safe finance DCF."""
from pathlib import Path
from datetime import date

from portfolio_risk.v8_fundamentals import analyze_fundamentals, simple_dcf_limitation
from portfolio_risk_ui.v913_strategy_paths import path_signal

APP = (Path(__file__).resolve().parents[1] / "portfolio_risk_ui" / "app.py").read_text()


def test_six_tab_charts_have_unique_per_goal_streamlit_keys():
    assert 'chart_key=f"v914_{summary[\'primary\']}_validation"' in APP
    assert 'chart_key=f"v914_{summary[\'primary\']}_allocation"' in APP
    assert 'st.plotly_chart(fig, width="stretch", key=chart_key, theme=None)' in APP
    assert 'def _strategy_test_chart(rows: list[dict], count: int, *, chart_key:' in APP
    assert 'def _strategy_allocation_chart(rows: list[dict], *, chart_key:' in APP


def test_visible_progress_and_historical_meaning():
    assert "This can take several minutes" in APP
    assert 'progress.progress(index / len(PATHS)' in APP
    assert "a portfolio using only earlier data" in APP
    assert "It does not test the exact same weights 30 times" in APP
    assert "All six measures in this historical example" in APP


def test_tri_state_labels_not_investment_recommendations():
    base = {"example": {"status":"evaluated"}, "review":{"status":"evaluated"}, "own_passed":False, "supported":False}
    assert path_signal({**base, "supported":True})[0] == "★"
    assert path_signal({**base, "own_passed":True})[0] == "○"
    assert path_signal(base)[0] == "✕"
    assert path_signal({**base, "example":{"status":"unavailable"}})[0] == "○"
    assert path_signal({**base, "review":{"status":"unavailable"}})[0] == "○"


def test_generic_dcf_suppressed_for_insurance_heavy_conglomerates():
    record = dict(symbol="BRK-B", period_end="2025-12-31", statement_currency="USD",
                  quote_currency="USD", free_cash_flow=50e9, shares_outstanding=2e9,
                  cash=400e9, total_debt=100e9, price=570.89, market_cap=1e12,
                  net_income=30e9, ebitda=40e9)
    result = analyze_fundamentals(record, as_of=date(2026,11,8))
    assert result["dcf"]["status"] == "unavailable"
    assert not result["dcf"]["scenarios"]
    assert result["multiples"]["price_to_earnings"] is not None
    assert "Berkshire" in result["dcf"]["reason"]
    record["symbol"] = "AAPL"
    assert analyze_fundamentals(record,as_of=date(2026,11,8))["dcf"]["status"] == "scenario_analysis"
    assert simple_dcf_limitation("BRK.B")
    assert simple_dcf_limitation("ABC", sector="Financial Services")
    assert not simple_dcf_limitation("ABC", sector="Technology")
