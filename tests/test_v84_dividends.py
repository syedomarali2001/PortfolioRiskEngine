"""V8.4 offline reproducibility and protection against overconfident dividend claims."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from portfolio_risk.accounting import value_portfolio
from portfolio_risk.cli import main
from portfolio_risk.models import Holding, PortfolioInput
from portfolio_risk.v84_dividend_provider import (
    fetch_yahoo_distribution_snapshot,
    get_cached_dividends,
)
from portfolio_risk.v84_dividends import DividendError, evaluate_dividends

CUT = date(2025, 11, 7)


def val(*holdings, base="USD", fx=None):
    p = PortfolioInput(
        tuple(Holding(s, Decimal(str(v)), ccy) for s, v, ccy in holdings), source_file="test.csv"
    )
    return value_portfolio(p, base_currency=base, fx_quote=fx)


def snapshot(sym="AAA", price=100, dps=2, prev=1.6, *, currency="USD"):
    return {
        "symbol": sym,
        "quote_currency": currency,
        "close_price": price,
        "price_date": "2025-11-06",
        "history_first_date": "2022-10-01",
        "history_last_date": "2025-11-06",
        "dividends": [
            {"date": f"{year}-{month:02d}-15", "amount": v / 4}
            for year, v in [(2024, prev), (2025, dps)]
            for month in [2, 5, 8, 10]
        ],
        "source": "Offline synthetic",
    }


def evaluate(recs=None, holdings=None, **kwargs):
    recs = recs if recs is not None else {"AAA": snapshot()}
    holdings = holdings or [("AAA", 10000, "USD")]
    return evaluate_dividends(val(*holdings), recs, as_of=CUT, **kwargs)


def test_yield_income_and_hist_growth_and_no_price_return_double_count():
    report = evaluate()
    a = report["holdings"][0]
    assert a["status"] == "evaluated"
    assert a["trailing_distribution_yield"] == pytest.approx(0.02)
    assert a["estimated_annual_gross_income"] == pytest.approx(200)
    assert a["historical_trailing_year_distribution_growth"] == pytest.approx(0.25)
    assert a["observed_trailing_distribution_events"] == 4
    assert report["income_summary"]["fully_covered_portfolio_yield"] == pytest.approx(0.02)
    assert "unadjusted close" in report["method"].lower()
    assert json.loads(json.dumps(report, allow_nan=False))["version"] == "8.4"


def test_monthly_is_average_not_schedule():
    assert evaluate()["holdings"][0]["estimated_monthly_average_gross_income"] == pytest.approx(
        200 / 12
    )


def test_no_payout_history_with_full_coverage_is_zero_not_unknown():
    s = snapshot()
    s["dividends"] = []
    r = evaluate({"AAA": s})
    assert r["holdings"][0]["status"] == "evaluated"
    assert r["holdings"][0]["trailing_distribution_yield"] == 0
    assert r["holdings"][0]["historical_trailing_year_distribution_growth"] is None


def test_insufficient_price_history_is_unavailable_not_zero():
    s = snapshot()
    s["history_first_date"] = "2025-08-01"
    s["dividends"] = []
    r = evaluate({"AAA": s})
    assert r["holdings"][0]["status"] == "unavailable"
    assert r["income_summary"]["fully_covered_portfolio_yield"] is None
    assert r["simulation"]["status"] == "unavailable"


def test_missing_provider_data_reports_unavailable_and_zero_coverage():
    r = evaluate({})
    assert r["holdings"][0]["status"] == "unavailable"
    assert r["income_summary"]["income_coverage_fraction"] == 0
    assert r["income_summary"]["known_income_divided_by_total_portfolio_value"] is None


def test_partial_coverage_never_claims_full_portfolio_yield():
    r = evaluate({"AAA": snapshot()}, [("AAA", 10000, "USD"), ("BBB", 10000, "USD")])
    assert r["income_summary"]["income_coverage_fraction"] == 0.5
    assert r["income_summary"]["trailing_distribution_yield_on_covered_value"] == pytest.approx(
        0.02
    )
    assert r["income_summary"]["known_income_divided_by_total_portfolio_value"] == pytest.approx(
        0.01
    )
    assert r["income_summary"]["fully_covered_portfolio_yield"] is None
    assert r["simulation"]["modeled_start_value"] == 10000
    assert r["simulation"]["unmodeled_start_value"] == 10000


def test_cad_usd_dividend_conversion_even_if_entered_values_are_cad():
    canadian = val(("AAA", 14000, "CAD"), base="CAD")
    report = evaluate_dividends(canadian, {"AAA": snapshot()}, as_of=CUT, fx_rate=Decimal("1.4"))
    assert report["holdings"][0]["estimated_annual_gross_income"] == pytest.approx(280)
    assert report["holdings"][0]["trailing_distribution_yield"] == pytest.approx(0.02)
    nofx = evaluate_dividends(canadian, {"AAA": snapshot()}, as_of=CUT)
    assert nofx["holdings"][0]["status"] == "unavailable"


def test_tax_affects_distributions_not_price_market_value():
    untaxed = evaluate(years=2, price_growth=0.0, dividend_growth=0.0)
    taxed = evaluate(years=2, price_growth=0.0, dividend_growth=0.0, tax_drag=0.2)
    u = untaxed["simulation"]["end_comparison"]
    t = taxed["simulation"]["end_comparison"]
    assert t["cash"]["invested_market_value"] == pytest.approx(10000)
    assert u["cash"]["invested_market_value"] == pytest.approx(10000)
    assert u["cash"]["cumulative_net_cash_withdrawn"] == pytest.approx(400)
    assert t["cash"]["cumulative_net_cash_withdrawn"] == pytest.approx(320)
    assert t["drip"]["total_wealth"] == pytest.approx(10322.56)
    assert t["cash"]["total_wealth_including_withdrawn_cash"] == pytest.approx(10320)
    assert t["drip"]["total_wealth"] > t["cash"]["total_wealth_including_withdrawn_cash"]
    assert t["drip"]["yearly_gross_distributions"] > t["cash"]["yearly_gross_distributions"]


def test_cash_does_not_gain_interest_and_drip_never_twice_same_year():
    r = evaluate(years=1, price_growth=0.0, dividend_growth=0.0)
    e = r["simulation"]["end_comparison"]
    assert e["drip"]["total_wealth"] == pytest.approx(10200)
    assert e["cash"]["total_wealth_including_withdrawn_cash"] == pytest.approx(10200)


def test_negative_price_growth_and_zero_dividend_growth_in_sim():
    r = evaluate(years=10, price_growth=-0.05, dividend_growth=-1.0)
    assert r["simulation"]["end_comparison"]["drip"]["total_wealth"] > 0
    assert r["simulation"]["annual_results"][1]["cash"]["yearly_net_cash_withdrawn"] == 0


def test_priors_missing_growth_remains_unknown():
    s = snapshot()
    s["history_first_date"] = "2024-11-01"
    s["dividends"] = [p for p in s["dividends"] if p["date"] >= "2024-11-01"]
    r = evaluate({"AAA": s})
    assert r["holdings"][0]["status"] == "evaluated"
    assert r["holdings"][0]["historical_trailing_year_distribution_growth"] is None


@pytest.mark.parametrize(
    "years,pg,dg,tax",
    [
        (0, 0.02, 0.02, 0),
        (41, 0.02, 0.02, 0),
        (True, 0.02, 0.02, 0),
        (1, -1, 0.02, 0),
        (1, 0.02, -1.01, 0),
        (1, 0.02, 0.02, 1.01),
        (1, float("nan"), 0.02, 0),
        (1, 0.02, float("inf"), 0),
    ],
)
def test_invalid_projection_controls_rejected(years, pg, dg, tax):
    with pytest.raises(DividendError):
        evaluate(years=years, price_growth=pg, dividend_growth=dg, tax_drag=tax)


@pytest.mark.parametrize(
    "changes",
    [
        {"symbol": "BBB"},
        {"quote_currency": "EUR"},
        {"close_price": 0},
        {"close_price": float("nan")},
        {"price_date": "2025-10-01"},
        {"history_last_date": "2025-09-01"},
        {"history_first_date": "2026-01-01"},
        {"dividends": None},
    ],
)
def test_invalid_provider_snapshot_fails_closed(changes):
    s = snapshot()
    s.update(changes)
    r = evaluate({"AAA": s})
    assert r["holdings"][0]["status"] == "unavailable"


def test_duplicate_dividend_events_fails_closed():
    s = snapshot()
    s["dividends"].append(dict(s["dividends"][0]))
    assert evaluate({"AAA": s})["holdings"][0]["status"] == "unavailable"


def test_yahoo_adapter_uses_raw_close_actions_not_adjusted_and_ignores_future():
    frame = pd.DataFrame(
        {"Close": [99.0, 100.0], "Dividends": [0.0, 2.0]},
        index=pd.to_datetime(["2025-11-05", "2025-11-06"]),
    )

    class Yahoo:
        def history(self, **kwargs):
            assert kwargs["auto_adjust"] is False
            assert kwargs["actions"] is True
            assert kwargs["end"] == "2025-11-08"
            return frame

        def get_history_metadata(self):
            return {"currency": "USD"}

    got = fetch_yahoo_distribution_snapshot("AAA", as_of=CUT, ticker_factory=lambda s: Yahoo())
    assert got["close_price"] == 100
    assert got["dividends"] == [{"date": "2025-11-06", "amount": 2.0}]


def test_yahoo_missing_dividend_column_not_false_zero():
    class Yahoo:
        def history(self, **kwargs):
            return pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2025-11-06"]))

    with pytest.raises(DividendError, match="Dividends"):
        fetch_yahoo_distribution_snapshot("AAA", as_of=CUT, ticker_factory=lambda s: Yahoo())


def test_yahoo_currency_disagreement_not_silent():
    class Yahoo:
        def history(self, **kwargs):
            return pd.DataFrame(
                {"Close": [100.0], "Dividends": [1.0]}, index=pd.to_datetime(["2025-11-06"])
            )

        def get_history_metadata(self):
            return {"currency": "CAD"}

    with pytest.raises(DividendError, match="disagrees"):
        fetch_yahoo_distribution_snapshot(
            "AAA", as_of=CUT, expected_currency="USD", ticker_factory=lambda s: Yahoo()
        )


def test_yahoo_missing_currency_requires_explicit_or_unavailable():
    class Yahoo:
        def history(self, **kwargs):
            return pd.DataFrame(
                {"Close": [100.0], "Dividends": [1.0]}, index=pd.to_datetime(["2025-11-06"])
            )

        def get_history_metadata(self):
            return {}

        fast_info = {}

    with pytest.raises(DividendError, match="Quote currency unavailable"):
        fetch_yahoo_distribution_snapshot("AAA", as_of=CUT, ticker_factory=lambda s: Yahoo())
    got = fetch_yahoo_distribution_snapshot(
        "AAA", as_of=CUT, expected_currency="USD", ticker_factory=lambda s: Yahoo()
    )
    assert got["quote_currency"] == "USD"


def test_same_date_cache_avoids_second_live_call(tmp_path):
    calls = []
    frame = pd.DataFrame(
        {"Close": [100.0], "Dividends": [1.0]}, index=pd.to_datetime(["2025-11-06"])
    )

    class Yahoo:
        def history(self, **kwargs):
            calls.append(1)
            return frame

        def get_history_metadata(self):
            return {"currency": "USD"}

    arg = dict(as_of=CUT, cache_dir=tmp_path, ticker_factory=lambda s: Yahoo())
    a = get_cached_dividends("AAA", **arg)
    b = get_cached_dividends("AAA", **arg)
    assert len(calls) == 1 and a == b
    get_cached_dividends("AAA", refresh=True, **arg)
    assert len(calls) == 2


def test_dividend_cli_offline_json_and_exact_snapshot_date(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    snap = root / "examples/v84_dividend_demo.json"
    holdings = root / "examples/v84_dividend_demo_portfolio.csv"
    out = tmp_path / "out.json"
    args = [
        "dividends",
        str(holdings),
        "--base",
        "USD",
        "--as-of",
        "2025-11-07",
        "--snapshot-file",
        str(snap),
        "--tax-drag",
        "15",
        "--years",
        "5",
        "--output",
        str(out),
        "--json",
    ]
    assert main(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["income_summary"]["income_coverage_fraction"] == 1
    assert data["simulation"]["assumed_effective_distribution_tax_drag"] == pytest.approx(0.15)
    assert json.loads(out.read_text()) == data
    assert (
        main(
            [
                "dividends",
                str(holdings),
                "--base",
                "USD",
                "--as-of",
                "2025-11-06",
                "--snapshot-file",
                str(snap),
            ]
        )
        != 0
    )


def test_cli_includes_manual_fx_when_input_values_already_in_base(tmp_path, capsys):
    # When CAD holding amount represents USD-listed stock, dividends need FX.
    p = tmp_path / "p.csv"
    p.write_text("symbol,market_value,currency,quote_currency\nAAA,14000,CAD,USD\n")
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"as_of": "2025-11-07", "snapshots": {"AAA": snapshot()}}))
    args = [
        "dividends",
        str(p),
        "--as-of",
        "2025-11-07",
        "--base",
        "CAD",
        "--snapshot-file",
        str(s),
        "--usd-cad",
        "1.4",
        "--json",
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["holdings"][0]["estimated_annual_gross_income"] == pytest.approx(280)
    assert result["fx_snapshot"]["rate_cad_per_usd"] == "1.4"


def test_offline_cli_rejects_explicit_quote_currency_conflict(tmp_path, capsys):
    p = tmp_path / "p.csv"
    p.write_text("symbol,market_value,currency,quote_currency\nAAA,10000,USD,CAD\n")
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"as_of": "2025-11-07", "snapshots": {"AAA": snapshot()}}))
    assert (
        main(
            [
                "dividends",
                str(p),
                "--base",
                "USD",
                "--as-of",
                "2025-11-07",
                "--snapshot-file",
                str(s),
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["holdings"][0]["status"] == "unavailable"
    assert result["income_summary"]["fully_covered_portfolio_yield"] is None
    assert any("conflicts" in x for x in result["warnings"])
