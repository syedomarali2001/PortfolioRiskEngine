import json
from datetime import date, datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.accounting import value_portfolio
from portfolio_risk.cli import main
from portfolio_risk.fx import manual_fx_quote
from portfolio_risk.models import Holding, PortfolioInput, PriceHistory
from portfolio_risk.v8_service import run_insights_from_market_data


class Cache:
    def get(self, *args):
        return None

    def put(self, *args):
        pass


class Provider:
    def __init__(self, seed=5):
        self.called = []
        self.dates = pd.bdate_range("2024-01-03", periods=470)
        rng = np.random.default_rng(seed)
        self.data = {}
        for symbol, mean, vol, ccy in [
            ("AAA", 0.0004, 0.01, "USD"),
            ("BBB", 0.0002, 0.016, "USD"),
            ("SPY", 0.0003, 0.012, "USD"),
            ("RSP", 0.0002, 0.011, "USD"),
            ("RY.TO", 0.0003, 0.012, "CAD"),
        ]:
            self.data[symbol] = (
                pd.Series(
                    100 * np.cumprod(1 + rng.normal(mean, vol, len(self.dates))), index=self.dates
                ),
                ccy,
            )
        self.data["USDCAD=X"] = (pd.Series([1.35] * len(self.dates), index=self.dates), "CAD")

    def get_history(self, symbol, start, end):
        self.called.append(symbol)
        s, ccy = self.data[symbol]
        part = s.loc[(s.index.date >= start) & (s.index.date < end)]
        return PriceHistory(
            symbol, part, start, end, "Offline mock", datetime.now(timezone.utc), ccy
        )


def test_insights_service_fetches_both_benchmarks_and_fx_and_keeps_cad_weights():
    p = PortfolioInput(
        (Holding("AAA", Decimal(100), "USD", "USD"), Holding("RY.TO", Decimal(100), "CAD", "CAD")),
        "test.csv",
    )
    v = value_portfolio(p, base_currency="CAD", fx_quote=manual_fx_quote("1.35", date(2025, 10, 1)))
    provider = Provider()
    d = run_insights_from_market_data(
        p, v, provider, Cache(), start=date(2024, 1, 3), end=date(2025, 11, 1)
    )
    assert d["base_currency"] == "CAD"
    assert set(d["benchmarks"]) == {"SPY", "RSP"}
    assert d["input_weights"]["AAA"] == pytest.approx(135 / 235)
    assert "USDCAD=X" in provider.called
    assert d["market_data"]["SPY"]["quote_currency"] == "USD"
    json.dumps(d, allow_nan=False)


def test_cli_insights_uses_benchmarks_and_scenario_with_json_output(tmp_path, monkeypatch, capsys):
    import portfolio_risk.cli as cli

    provider = Provider()
    monkeypatch.setattr(cli, "YahooFinanceProvider", lambda: provider)
    monkeypatch.setattr(cli, "HistoryCache", lambda x: Cache())
    path = tmp_path / "p.csv"
    path.write_text(
        "symbol,market_value,currency,quote_currency\nAAA,600,USD,USD\nBBB,400,USD,USD\n"
    )
    output = tmp_path / "report.json"
    args = [
        "insights",
        str(path),
        "--base",
        "USD",
        "--start",
        "2024-01-03",
        "--end",
        "2025-11-01",
        "--as-of",
        "2025-11-01",
        "--scenario",
        "half:AAA=50,BBB=50",
        "--priorities",
        "sharpe:2,drawdown:1",
        "--json",
        "--output",
        str(output),
        "--mc-paths",
        "40",
    ]
    code = main(args)
    out, err = capsys.readouterr()
    assert code == 0, err
    doc = json.loads(out)
    assert doc["user_scenarios"]["half"]["weights"]["AAA"] == 0.5
    assert doc["historical_strategy_selection"]["status"] == "evaluated"
    assert doc["diagnosis_summary"]["interpretation"]
    assert doc == json.loads(output.read_text())
    assert set(provider.called) == {"AAA", "BBB", "SPY", "RSP"}


def test_cli_rejects_missing_scenario_allocations(tmp_path, monkeypatch, capsys):
    import portfolio_risk.cli as cli

    monkeypatch.setattr(cli, "YahooFinanceProvider", lambda: Provider())
    monkeypatch.setattr(cli, "HistoryCache", lambda x: Cache())
    path = tmp_path / "p.csv"
    path.write_text("symbol,market_value,currency\nAAA,600,USD\nBBB,400,USD\n")
    args = [
        "insights",
        str(path),
        "--base",
        "USD",
        "--start",
        "2024-01-03",
        "--end",
        "2025-11-01",
        "--as-of",
        "2025-11-01",
        "--scenario",
        "wrong:AAA=100",
        "--json",
    ]
    code = main(args)
    assert code == 1
    assert "exactly one weight" in capsys.readouterr().err


def test_cli_fundamentals_offline_file_and_invalid_currency(tmp_path, capsys):
    path = tmp_path / "abc.json"
    data = {
        "symbol": "ABC",
        "period_end": "2025-12-31",
        "statement_currency": "USD",
        "free_cash_flow": 100,
        "shares_outstanding": 10,
        "cash": 20,
        "total_debt": 60,
        "price": 15,
    }
    path.write_text(json.dumps(data))
    code = main(["fundamentals", "ABC", "--file", str(path), "--as-of", "2026-11-07", "--json"])
    out, err = capsys.readouterr()
    assert code == 0, err
    assert json.loads(out)["dcf"]["status"] == "scenario_analysis"
    data["quote_currency"] = "CAD"
    path.write_text(json.dumps(data))
    code = main(["fundamentals", "ABC", "--file", str(path), "--as-of", "2026-11-07"])
    assert code == 1
    assert "currencies differ" in capsys.readouterr().err


def test_cli_exposure_drilldown_and_enrichment_remain_auditable(tmp_path, monkeypatch, capsys):
    import portfolio_risk.cli as cli
    from portfolio_risk.providers.tickerinside import parse_fund_record
    from portfolio_risk.security_metadata import YahooSecurityMetadataProvider

    def data(sym):
        return {
            "ticker": sym,
            "name": "Fake SPY",
            "holdings_kind": "unknown",
            "source_layer": "issuer",
            "holdings_as_of": "2026-11-05",
            "holdings": {"AAPL": 70.0, "MSFT": 29.8},
            "breakdown": {
                "sectors_pct": {"technology": 70.0, "financials": 29.8},
                "countries_pct": {"US": 99.8},
                "sector_coverage_pct": 99.8,
                "other_funds_pct": 0,
            },
            "category": {
                "asset_class": "equity",
                "basis": "holdings",
                "as_of": "2026-11-05",
                "measured": {"stocks": 2},
            },
        }

    class API:
        def index(self, refresh=False):
            return {"SPY"}, {}

        def fund(self, symbol, refresh=False):
            return parse_fund_record(data(symbol), symbol)

    profiles = {
        "AAPL": {
            "symbol": "AAPL",
            "quoteType": "EQUITY",
            "sector": "Technology",
            "country": "United States",
        },
        "MSFT": {
            "symbol": "MSFT",
            "quoteType": "EQUITY",
            "sector": "Financial Services",
            "country": "United States",
        },
        "XEQT.TO": {"symbol": "XEQT.TO", "quoteType": "ETF"},
    }
    monkeypatch.setattr(cli, "TickerInsideClient", lambda **kwargs: API())
    monkeypatch.setattr(
        cli,
        "YahooSecurityMetadataProvider",
        lambda: YahooSecurityMetadataProvider(getter=lambda s: profiles[s]),
    )
    path = tmp_path / "port.csv"
    path.write_text(
        "symbol,market_value,currency,quote_currency\nSPY,500,USD,USD\nAAPL,300,USD,USD\nXEQT.TO,200,USD,CAD\n"
    )
    args = [
        "exposure",
        str(path),
        "--base",
        "USD",
        "--as-of",
        "2026-11-07",
        "--tickerinside",
        "--auto-stock-metadata",
        "--drilldown",
        "--enrich-underlyings",
        "2",
        "--underlying-metadata-cache",
        str(tmp_path / "cache"),
        "--json",
    ]
    code = cli.main(args)
    out, err = capsys.readouterr()
    assert code == 0, err
    d = json.loads(out)
    assert d["unknown_underlying"] == pytest.approx(0.201)
    assert d["drilldown"]["unknown_holdings_by_fund"]["XEQT.TO"][
        "unknown_portfolio_fraction"
    ] == pytest.approx(0.2)
    assert (
        d["drilldown"]["metadata_enrichment"]["enriched_count"] == 1
    )  # AAPL already classified; MSFT looked up
    sector = d["drilldown"]["sector_drilldown"]
    assert sector["technology"]["portfolio_fraction"] == pytest.approx(0.65)
    assert sector["financials"]["known_company_contributions"]["MSFT"] == pytest.approx(0.5 * 0.298)
    assert sector["financials"]["portfolio_fraction"] == pytest.approx(0.5 * 0.298)
    assert d["drilldown"]["fund_top_holdings"]["SPY"]["top_holdings"][0]["symbol"] == "AAPL"


def test_insights_missing_rsp_currency_metadata_uses_narrow_usd_benchmark_fallback():
    portfolio = PortfolioInput((Holding("AAA", Decimal(100), "USD", "USD"),), "test.csv")
    valuation = value_portfolio(portfolio, base_currency="USD")
    provider = Provider()
    series, _ = provider.data["RSP"]
    provider.data["RSP"] = (series, None)  # Exactly the Yahoo history metadata failure seen live.
    out = run_insights_from_market_data(
        portfolio,
        valuation,
        provider,
        Cache(),
        start=date(2024, 1, 3),
        end=date(2025, 11, 1),
        mc_paths=20,
    )
    assert out["market_data"]["RSP"]["quote_currency"] == "USD"
    assert out["market_data"]["RSP"]["quote_currency_source"] == "benchmark_symbol_default"
    assert any(
        "RSP: provider quote currency unavailable" in w for w in out["data_quality_warnings"]
    )
    assert out["market_data"]["SPY"]["quote_currency_source"] == "provider_metadata"
    assert out["benchmarks"]["RSP"]


def test_insights_wrong_reported_benchmark_currency_is_error_not_silent_override():
    from portfolio_risk.v8_insights import InsightError

    portfolio = PortfolioInput((Holding("AAA", Decimal(100), "USD", "USD"),), "test.csv")
    valuation = value_portfolio(portfolio, base_currency="USD")
    provider = Provider()
    series, _ = provider.data["RSP"]
    provider.data["RSP"] = (series, "CAD")
    with pytest.raises(InsightError, match="provider reports CAD"):
        run_insights_from_market_data(
            portfolio,
            valuation,
            provider,
            Cache(),
            start=date(2024, 1, 3),
            end=date(2025, 11, 1),
            mc_paths=20,
        )


def test_cli_insights_rsp_currency_missing_still_writes_complete_report(
    tmp_path, monkeypatch, capsys
):
    import portfolio_risk.cli as cli

    provider = Provider()
    series, _ = provider.data["RSP"]
    provider.data["RSP"] = (series, None)
    monkeypatch.setattr(cli, "YahooFinanceProvider", lambda: provider)
    monkeypatch.setattr(cli, "HistoryCache", lambda x: Cache())
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text(
        "symbol,market_value,currency,quote_currency\nAAA,600,USD,USD\nBBB,400,USD,USD\n"
    )
    output = tmp_path / "v8_insights_report.json"
    code = cli.main(
        [
            "insights",
            str(portfolio),
            "--base",
            "USD",
            "--start",
            "2024-01-03",
            "--end",
            "2025-11-01",
            "--as-of",
            "2025-11-01",
            "--priorities",
            "sharpe:2,drawdown:1,volatility:1",
            "--scenario",
            "equal:AAA=50,BBB=50",
            "--compare-currencies",
            "--mc-paths",
            "20",
            "--output",
            str(output),
        ]
    )
    stdout, stderr = capsys.readouterr()
    assert code == 0, stderr
    assert output.is_file()
    report = json.loads(output.read_text())
    assert report["market_data"]["RSP"]["quote_currency_source"] == "benchmark_symbol_default"
    assert "currency_comparison" in report
    assert any(
        "RSP: provider quote currency unavailable" in w for w in report["data_quality_warnings"]
    )


def test_v8_currency_comparison_handles_one_adjacent_fx_bar_only():
    """Regression for Yahoo returning an FX bar outside requested end/start."""
    from dataclasses import replace

    class AdjacentFX(Provider):
        def get_history(self, symbol, start, end):
            hist = super().get_history(symbol, start, end)
            if symbol == "USDCAD=X":
                extra = pd.Series(
                    [1.351], index=pd.DatetimeIndex([pd.Timestamp(start) - pd.Timedelta(days=1)])
                )
                return replace(hist, prices=pd.concat((extra, hist.prices)).sort_index())
            return hist

    portfolio = PortfolioInput((Holding("AAA", Decimal(100), "USD", "USD"),), "test.csv")
    valuation = value_portfolio(portfolio, base_currency="USD")
    out = run_insights_from_market_data(
        portfolio,
        valuation,
        AdjacentFX(),
        Cache(),
        start=date(2024, 1, 3),
        end=date(2025, 11, 1),
        compare_currencies=True,
        mc_paths=20,
    )
    assert out["currency_comparison"]["status"] == "evaluated"
    assert any("out-of-window FX bar" in w for w in out["data_quality_warnings"])


def test_v8_currency_comparison_rejects_large_fx_boundary_mismatch():
    from dataclasses import replace

    class FarFX(Provider):
        def get_history(self, symbol, start, end):
            hist = super().get_history(symbol, start, end)
            if symbol == "USDCAD=X":
                far = pd.Series(
                    [1.35], index=pd.DatetimeIndex([pd.Timestamp(start) - pd.Timedelta(days=30)])
                )
                return replace(hist, prices=pd.concat((far, hist.prices)).sort_index())
            return hist

    portfolio = PortfolioInput((Holding("AAA", Decimal(100), "USD", "USD"),), "test.csv")
    valuation = value_portfolio(portfolio, base_currency="USD")
    out = run_insights_from_market_data(
        portfolio,
        valuation,
        FarFX(),
        Cache(),
        start=date(2024, 1, 3),
        end=date(2025, 11, 1),
        compare_currencies=True,
        mc_paths=20,
    )
    assert out["currency_comparison"]["status"] == "unavailable"
    assert "outside requested window" in out["currency_comparison"]["reason"]
