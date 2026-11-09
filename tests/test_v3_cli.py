"""End-to-end CLI smoke tests using offline quote histories."""
from datetime import date, datetime, timezone
import json

import pandas as pd

from portfolio_risk.cli import main
from portfolio_risk.models import PriceHistory


class MemoryCache:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args):
        return None

    def put(self, *args):
        pass


class FakeProvider:
    def get_history(self, symbol, start, end):
        dates = pd.date_range("2026-01-05", "2026-03-31", freq="B")
        values = {"XEQT.TO": 100, "QQQ": 200, "SCHD": 60,
                  "SPMO": 90, "BTC-USD": 300, "GLD": 70,
                  "USDCAD=X": 1.35, "SPY": 100}
        prices = pd.Series(
            [values[symbol] * (1.0 + i * .0005) for i in range(len(dates))]
            if symbol != "USDCAD=X" else [1.35] * len(dates),
            index=dates, dtype=float,
        )
        prices = prices.loc[(prices.index.date >= start) & (prices.index.date < end)]
        return PriceHistory(symbol, prices, start, end, "Offline stub",
                            datetime.now(timezone.utc), quote_currency=None)


def test_v3_cli_json_with_faked_live_data(monkeypatch, capsys):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli, "YahooFinanceProvider", FakeProvider)
    monkeypatch.setattr(cli, "HistoryCache", MemoryCache)
    code = main([
        "performance", "examples/sample_portfolio_v3.csv", "--base", "CAD",
        "--usd-cad", "1.35", "--start", "2026-01-05", "--end", "2026-04-01",
        "--benchmark", "SPY", "--benchmark-currency", "USD", "--risk-free", "3",
        "--json",
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    d = json.loads(captured.out)
    assert d["base_currency"] == "CAD"
    assert d["benchmark"]["symbol"] == "SPY"
    assert d["return_periods"] > 30
    assert len(d["weights"]) == 6
    assert "USDCAD=X" in d["market_data"]
    assert len(d["portfolio_index"]) == d["return_periods"] + 1
    assert d["annual_risk_free_rate"] == .03


def test_v3_cli_human_friendly(monkeypatch, capsys):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli, "YahooFinanceProvider", FakeProvider)
    monkeypatch.setattr(cli, "HistoryCache", MemoryCache)
    code = main([
        "performance", "examples/sample_portfolio_v3.csv", "--base", "CAD",
        "--usd-cad", "1.35", "--start", "2026-01-05", "--end", "2026-04-01",
    ])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "CAGR" in out.out
    assert "Sharpe" in out.out
    assert "Sortino" in out.out
    assert "Return correlations" in out.out
    assert "HYPOTHETICAL" in out.out


def test_v3_cli_date_and_input_error_exit_code(capsys):
    assert main(["performance", "examples/sample_portfolio_v3.csv",
                 "--start", "2090-01-01", "--end", "2090-05-01"]) == 2
    assert "DATE ERROR" in capsys.readouterr().err
