"""V9.15.1 demo selectors, import-ready CSVs, and important edge cases."""
from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_risk_ui.core import PortfolioError, import_values_csv
from portfolio_risk_ui.example_portfolios import (
    EXAMPLES, example_csv, example_holdings, get_example, load_example,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_sample_is_bounded_unique_and_importable():
    assert len(EXAMPLES) >= 8
    assert len({example.key for example in EXAMPLES}) == len(EXAMPLES)
    for example in EXAMPLES:
        holdings = example_holdings(example.key)
        assert 1 <= len(holdings) <= 40
        assert sum(holding["weight"] for holding in holdings) == pytest.approx(100)
        assert len({holding["symbol"] for holding in holdings}) == len(holdings)
        assert all(holding["market_value"] > 0 for holding in holdings)
        assert all(holding["quote_currency"] in ("USD", "CAD") for holding in holdings)
        exported = example_csv(example.key)
        imported = import_values_csv(exported.encode())
        assert [(r["symbol"], r["market_value"], r["currency"], r["quote_currency"])
                for r in imported] == [
                    (r["symbol"], r["market_value"], r["currency"], r["quote_currency"])
                    for r in holdings]
        stored_file = ROOT / "examples" / "demo_portfolios" / f"{example.key}.csv"
        assert stored_file.read_bytes() == exported.encode("utf-8")


def test_original_starter_allocation_is_preserved():
    holdings = example_holdings("starter")
    assert [(x["symbol"], x["market_value"], x["weight"]) for x in holdings] == [
        ("AAPL", 8750.0, 35), ("MSFT", 5000.0, 20),
        ("SPY", 7500.0, 30), ("GLD", 3750.0, 15),
    ]


def test_concentration_and_limiting_cases_are_not_mislabelled_as_safe():
    for key, sym, concentration in [
        ("ccl80", "CCL", 80), ("nvda90", "NVDA", 90),
        ("one_holding", "SPY", 100),
    ]:
        assert next(x["weight"] for x in example_holdings(key) if x["symbol"] == sym) == concentration
        assert "recommend" not in get_example(key).purpose.lower()
    assert len(example_holdings("max_holdings")) == 40
    with pytest.raises(PortfolioError, match="configured limit"):
        example_holdings("max_holdings", max_holdings=20)


def test_mixed_currency_demo_distinguishes_amount_and_trading_currency():
    holdings = example_holdings("mixed_fx")
    assert {x["currency"] for x in holdings} == {"CAD", "USD"}
    assert any(x["currency"] != x["quote_currency"] for x in holdings)
    assert next(x for x in holdings if x["symbol"] == "SPY")["quote_currency"] == "USD"
    assert "NOT FX-adjusted" in get_example("mixed_fx").note


def test_loading_sample_replaces_holdings_and_invalidates_stale_results():
    state = {"holdings": [{"symbol":"OLD"}], "entry_mode":"weights", "base":"CAD",
             "reports":{"risk":1}, "strategy_cache":{"old":1}, "editor_version":2,
             "ticker_version":7, "selected_ticker":{"symbol":"OLD"}, "last_search_query":"old",
             "quote_sources":{"OLD":"unknown"}}
    load_example(state, "ccl80")
    assert len(state["holdings"]) == 5 and state["holdings"][0]["symbol"] == "CCL"
    assert state["entry_mode"] == "values"
    assert state["base"] == "CAD"  # chosen reporting currency preserved
    assert state["editor_version"] == 3 and state["ticker_version"] == 8
    assert state["reports"] == state["strategy_cache"] == state["quote_sources"] == {}
    assert state["selected_ticker"] is None and state["last_search_query"] == ""


def test_rejected_sample_does_not_mutate_existing_portfolio():
    state = {"holdings": [{"symbol": "SPY"}], "reports": {"risk": 1}}
    with pytest.raises(PortfolioError):
        load_example(state, "max_holdings", max_holdings=10)
    assert state == {"holdings": [{"symbol": "SPY"}], "reports": {"risk": 1}}
    with pytest.raises(PortfolioError, match="Unknown"):
        load_example(state, "not_a_real_example")


def test_demo_selector_exists_on_builder_even_when_portfolio_is_populated():
    app = (ROOT / 'portfolio_risk_ui' / 'app.py').read_text(encoding='utf-8')
    builder = app.split('def page_builder():',1)[1].split('def _render_sample_portfolios():',1)[0]
    chooser = app.split('def _render_sample_portfolios():',1)[1].split('def _render_portfolio_list():',1)[0]
    assert builder.count('_render_sample_portfolios()') == 2
    assert 'with st.expander("Sample portfolios for testing"' in chooser
    assert 'load_example(st.session_state, demo_key)' in chooser
    assert 'replaces the current portfolio' in chooser

