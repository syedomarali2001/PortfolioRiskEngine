import json
from datetime import date

import pandas as pd
import pytest

from portfolio_risk.v8_fundamentals import (
    FundamentalsError,
    analyze_fundamentals,
    yahoo_fundamentals_snapshot,
)

BASE = dict(
    symbol="ABC",
    period_end="2025-12-31",
    statement_currency="USD",
    free_cash_flow=100.0,
    cash=20.0,
    total_debt=60.0,
    shares_outstanding=10.0,
    price=15.0,
    market_cap=150.0,
    revenue=500.0,
    net_income=40.0,
    ebitda=60.0,
)


def test_dcf_scenarios_ordered_multiples_and_json():
    d = analyze_fundamentals(BASE, as_of=date(2026, 11, 7))
    cases = d["dcf"]["scenarios"]
    assert (
        cases["conservative"]["intrinsic_value_per_share"]
        < cases["base"]["intrinsic_value_per_share"]
        < cases["optimistic"]["intrinsic_value_per_share"]
    )
    assert d["multiples"]["price_to_fcf"] == 1.5
    assert d["multiples"]["price_to_earnings"] == 3.75
    assert d["multiples"]["ev_to_ebitda"] == pytest.approx(190 / 60)
    assert d["used_free_cash_flow"] == 100
    json.dumps(d, allow_nan=False)


def test_fcf_derived_only_from_negative_capex_convention():
    rec = {**BASE, "free_cash_flow": None, "operating_cash_flow": 200, "capital_expenditures": -65}
    assert analyze_fundamentals(rec)["used_free_cash_flow"] == 135
    rec["capital_expenditures"] = 65
    with pytest.raises(FundamentalsError, match="NEGATIVE"):
        analyze_fundamentals(rec)


def test_negative_fcf_and_missing_cash_cannot_fake_dcf():
    for changes in (
        {"free_cash_flow": -100},
        {"free_cash_flow": None},
        {"cash": None},
        {"shares_outstanding": None},
    ):
        d = analyze_fundamentals({**BASE, **changes})
        assert d["dcf"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "changes",
    [
        {"statement_currency": "EUR"},
        {"quote_currency": "CAD"},
        {"period_end": "2030-01-01"},
        {"cash": -100},
        {"shares_outstanding": 0},
        {"market_cap": float("nan")},
    ],
)
def test_fundamental_validation(changes):
    with pytest.raises(FundamentalsError):
        analyze_fundamentals({**BASE, **changes}, as_of=date(2026, 11, 7))


def test_invalid_dcf_parameters_rejected():
    for params in (
        {"growth": 0.1, "discount": 0.03, "terminal": 0.03},
        {"growth": 0.2, "discount": 0.08, "terminal": 0.1},
        {"growth": 5, "discount": 0.08, "terminal": 0},
    ):
        with pytest.raises(FundamentalsError):
            analyze_fundamentals(BASE, scenarios={"bad": params})


def test_yahoo_adapter_dated_same_fiscal_period_and_no_live_calls():
    dates = pd.to_datetime(["2025-12-31", "2024-12-31"])

    class Ticker:
        def get_info(self):
            return {
                "quoteType": "EQUITY",
                "financialCurrency": "USD",
                "currency": "USD",
                "sharesOutstanding": 10.0,
                "currentPrice": 15.0,
            }

        def get_cashflow(self, freq="yearly"):
            return pd.DataFrame(
                [[100, 90], [210, 180], [-110, -90]],
                index=["FreeCashFlow", "OperatingCashFlow", "CapitalExpenditure"],
                columns=dates,
            )

        def get_income_stmt(self, freq="yearly"):
            return pd.DataFrame(
                [[500, 450], [60, 50]], index=["TotalRevenue", "EBITDA"], columns=dates
            )

        def get_balance_sheet(self, freq="yearly"):
            return pd.DataFrame(
                [[20, 10], [60, 70]], index=["CashAndCashEquivalents", "TotalDebt"], columns=dates
            )

    record = yahoo_fundamentals_snapshot(
        "ABC", as_of=date(2026, 11, 7), ticker_factory=lambda s: Ticker()
    )
    assert record["period_end"] == "2025-12-31"
    assert record["free_cash_flow"] == 100
    assert analyze_fundamentals(record)["dcf"]["status"] == "scenario_analysis"


def test_yahoo_no_future_fiscal_rows():
    dates = pd.to_datetime(["2026-12-31", "2025-12-31"])

    class Ticker:
        def get_info(self):
            return {"quoteType": "EQUITY", "financialCurrency": "USD", "currency": "USD"}

        def get_cashflow(self, freq="yearly"):
            return pd.DataFrame([[900, 100]], index=["FreeCashFlow"], columns=dates)

        def get_income_stmt(self, freq="yearly"):
            return pd.DataFrame()

        def get_balance_sheet(self, freq="yearly"):
            return pd.DataFrame()

    record = yahoo_fundamentals_snapshot(
        "ABC", as_of=date(2026, 11, 7), ticker_factory=lambda s: Ticker()
    )
    assert record["period_end"] == "2025-12-31"
    assert record["free_cash_flow"] == 100


def test_peer_multiples_requires_two_valid_peers():
    from portfolio_risk.v8_fundamentals import compare_peer_multiples

    a = {**BASE, "symbol": "DEF", "market_cap": 240.0, "price": 24.0}
    b = {**BASE, "symbol": "GHI", "market_cap": 180.0, "price": 18.0}
    result = compare_peer_multiples(BASE, [a, b], as_of=date(2026, 11, 7))
    x = result["multiples"]["price_to_earnings"]
    assert x["status"] == "evaluated"
    assert x["peer_median"] == pytest.approx((6 + 4.5) / 2)
    assert x["subject_vs_peer_median_fraction"] == pytest.approx(3.75 / 5.25 - 1)
    one = compare_peer_multiples(BASE, [a], as_of=date(2026, 11, 7))
    assert one["multiples"]["price_to_earnings"]["status"] == "unavailable"
    with pytest.raises(FundamentalsError, match="unique"):
        compare_peer_multiples(BASE, [a, a])
    with pytest.raises(FundamentalsError, match="subject"):
        compare_peer_multiples(BASE, [BASE])
