import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.cli import main
from portfolio_risk.models import PriceHistory


class Cache:
    def get(self, *args):
        return None

    def put(self, *args):
        pass


class MarketMock:
    def __init__(self):
        self.called = []
        self.idx = pd.bdate_range("2024-01-03", periods=450)
        rng = np.random.default_rng(43)
        self.prices = {
            s: pd.Series(
                100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(self.idx))), index=self.idx
            )
            for s in ("AAA", "BBB", "SPY", "RSP")
        }

    def get_history(self, symbol, start, end):
        self.called.append(symbol)
        series = self.prices[symbol]
        subset = series.loc[(series.index.date >= start) & (series.index.date < end)]
        return PriceHistory(
            symbol, subset, start, end, "Synthetic mock", datetime.now(timezone.utc), "USD"
        )


def test_cli_robust_constraints_and_json(tmp_path, monkeypatch, capsys):
    import portfolio_risk.cli as cli

    provider = MarketMock()
    monkeypatch.setattr(cli, "YahooFinanceProvider", lambda: provider)
    monkeypatch.setattr(cli, "HistoryCache", lambda x: Cache())
    port = tmp_path / "p.csv"
    port.write_text(
        "symbol,market_value,currency,quote_currency\nAAA,600,USD,USD\nBBB,400,USD,USD\n"
    )
    output = tmp_path / "v83.json"
    argv = [
        "insights",
        str(port),
        "--base",
        "USD",
        "--start",
        "2024-01-03",
        "--end",
        "2025-11-01",
        "--as-of",
        "2025-11-01",
        "--robustness",
        "--asset-cap",
        "AAA=50",
        "--asset-floor",
        "BBB=20",
        "--group-cap",
        "basket:AAA,BBB=100",
        "--turnover-cap",
        "40",
        "--rebalance",
        "quarterly",
        "--cost-bps",
        "20",
        "--train-periods",
        "252",
        "--test-periods",
        "126",
        "--step-periods",
        "126",
        "--mc-paths",
        "20",
        "--json",
        "--output",
        str(output),
    ]
    code = main(argv)
    out, err = capsys.readouterr()
    assert code == 0, err
    report = json.loads(out)
    assert report == json.loads(output.read_text())
    assert report["rolling_robustness"]["status"] == "evaluated"
    assert report["rolling_robustness"]["constraints"]["cost_bps"] == 20
    assert report["rolling_robustness"]["constraints"]["rebalance"] == "quarterly"
    assert report["rolling_robustness"]["fold_count"] >= 1
    for fold in report["rolling_robustness"]["folds"]:
        assert fold["weights"]["AAA"] <= 0.500001
        assert sum(fold["weights"].values()) == pytest.approx(1.0)
    assert len(report["actionable_diagnostics"]["benchmark_targets"]) == 12
    assert set(provider.called) == {"AAA", "BBB", "SPY", "RSP"}


def test_cli_infeasible_limits_rejected_with_evidence(tmp_path, monkeypatch, capsys):
    import portfolio_risk.cli as cli

    monkeypatch.setattr(cli, "YahooFinanceProvider", lambda: MarketMock())
    monkeypatch.setattr(cli, "HistoryCache", lambda x: Cache())
    port = tmp_path / "p.csv"
    port.write_text(
        "symbol,market_value,currency,quote_currency\nAAA,600,USD,USD\nBBB,400,USD,USD\n"
    )
    base = [
        "insights",
        str(port),
        "--base",
        "USD",
        "--start",
        "2024-01-03",
        "--end",
        "2025-11-01",
        "--as-of",
        "2025-11-01",
        "--mc-paths",
        "20",
    ]
    assert main(base + ["--asset-cap", "AAA=20", "--asset-cap", "BBB=20"]) == 1
    assert "infeasible" in capsys.readouterr().err.lower()
    assert main(base + ["--asset-cap", "AAA=30", "--asset-cap", "AAA=40"]) == 1
    assert "unique" in capsys.readouterr().err.lower()
    assert main(base + ["--asset-cap", "BOGUS=20"]) == 1
    assert "unknown portfolio ticker" in capsys.readouterr().err.lower()


def test_cli_sector_focus_yahoo_lookup_mock(tmp_path, monkeypatch, capsys):
    import portfolio_risk.cli as cli
    from portfolio_risk.providers.tickerinside import parse_fund_record
    from portfolio_risk.security_metadata import YahooSecurityMetadataProvider

    class API:
        def index(self, refresh=False):
            return {"QQQ"}, {}

        def fund(self, symbol, refresh=False):
            return parse_fund_record(
                {
                    "ticker": "QQQ",
                    "name": "Mock QQQ",
                    "holdings_kind": "unknown",
                    "source_layer": "issuer",
                    "holdings_as_of": "2026-11-06",
                    "holdings": {"TECH": 55.0, "BANK": 44.9},
                    "breakdown": {
                        "sectors_pct": {"technology": 55.0, "financials": 44.9},
                        "countries_pct": {"US": 99.9},
                        "sector_coverage_pct": 99.9,
                        "other_funds_pct": 0,
                    },
                    "category": {
                        "asset_class": "equity",
                        "basis": "holdings",
                        "as_of": "2026-11-06",
                        "measured": {"stocks": 2},
                    },
                },
                symbol,
            )

    info = {
        "TECH": {
            "symbol": "TECH",
            "quoteType": "EQUITY",
            "sector": "Technology",
            "country": "United States",
        },
        "BANK": {
            "symbol": "BANK",
            "quoteType": "EQUITY",
            "sector": "Financial Services",
            "country": "United States",
        },
    }
    requests = []

    def get(s):
        requests.append(s)
        return info[s]

    monkeypatch.setattr(cli, "TickerInsideClient", lambda **kwargs: API())
    monkeypatch.setattr(
        cli, "YahooSecurityMetadataProvider", lambda: YahooSecurityMetadataProvider(getter=get)
    )
    p = tmp_path / "p.csv"
    p.write_text("symbol,market_value,currency,quote_currency\nQQQ,1000,USD,USD\n")
    result = main(
        [
            "exposure",
            str(p),
            "--base",
            "USD",
            "--as-of",
            "2026-11-07",
            "--tickerinside",
            "--drilldown",
            "--focus-sector",
            "financials",
            "--focus-limit",
            "2",
            "--underlying-metadata-cache",
            str(tmp_path / "meta"),
            "--json",
        ]
    )
    out, err = capsys.readouterr()
    assert result == 0, err
    doc = json.loads(out)
    report = doc["drilldown"]["sector_enrichment"]["financials"]
    assert report["matching_sector_symbols"] == ["BANK"]
    assert doc["drilldown"]["sector_drilldown"]["financials"]["known_company_contributions"][
        "BANK"
    ] == pytest.approx(0.449)
    assert doc["drilldown"]["fund_top_holdings"]["QQQ"][
        "classified_constituent_fraction"
    ] == pytest.approx(0.999)
    assert "exposure_actions" in doc
    assert set(requests) == {"BANK", "TECH"}
