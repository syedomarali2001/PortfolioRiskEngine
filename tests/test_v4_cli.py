"""V4 CLI smoke tests without network access."""
from datetime import datetime, timezone
import json

import pandas as pd

from portfolio_risk.cli import main
from portfolio_risk.models import PriceHistory


class Cache:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args):
        return None

    def put(self, *args):
        pass


class Provider:
    def get_history(self, symbol, start, end):
        ds = pd.date_range("2026-01-05", "2026-05-15", freq="B")
        start_px = {"XEQT.TO": 100, "QQQ": 200, "SCHD": 60,
                    "SPMO": 90, "BTC-USD": 300, "GLD": 70,
                    "USDCAD=X": 1.35, "SPY": 100}[symbol]
        vals = ([1.35]*len(ds) if symbol == "USDCAD=X" else
                [start_px * (1.0 + i*.001) for i in range(len(ds))])
        ser = pd.Series(vals, index=ds, dtype=float)
        ser = ser.loc[(ser.index.date >= start) & (ser.index.date < end)]
        return PriceHistory(symbol, ser, start, end, "Stub", datetime.now(timezone.utc), None)


def test_cli_json_and_cost_and_benchmark(monkeypatch, capsys):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli, "YahooFinanceProvider", Provider)
    monkeypatch.setattr(cli, "HistoryCache", Cache)
    result = main([
        "backtest", "examples/sample_portfolio_v3.csv",
        "--base", "CAD", "--usd-cad", "1.35", "--capital", "20000",
        "--start", "2026-01-05", "--end", "2026-05-16",
        "--fee-bps", "10", "--risk-free", "3",
        "--benchmark", "SPY", "--benchmark-currency", "USD", "--json",
    ])
    output = capsys.readouterr()
    assert result == 0, output.err
    obj = json.loads(output.out)
    assert obj["fee_bps"] == 10
    assert obj["benchmark_symbol"] == "SPY"
    assert obj["base_currency"] == "CAD"
    assert obj["annual_risk_free_rate"] == .03
    assert set(obj["strategies"]) == {"buy_hold", "monthly", "quarterly", "semiannual", "annual"}
    assert obj["strategies"]["buy_hold"]["total_fees"] > 0
    assert len(obj["strategies"]["annual"]["portfolio_values"]) > 30


def test_cli_human_output_and_defaults(monkeypatch, capsys):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli, "YahooFinanceProvider", Provider)
    monkeypatch.setattr(cli, "HistoryCache", Cache)
    result = main([
        "backtest", "examples/sample_portfolio_v3.csv", "--base", "CAD",
        "--usd-cad", "1.35", "--start", "2026-01-05", "--end", "2026-05-16",
    ])
    output = capsys.readouterr()
    assert result == 0, output.err
    assert "V4 historical backtest" in output.out
    assert "semiannual" in output.out
    assert "buy_hold" in output.out
    assert "0.00 bps" in output.out
    assert "CAGR" in output.out


def test_cli_schedule_subset(monkeypatch, capsys):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli, "YahooFinanceProvider", Provider)
    monkeypatch.setattr(cli, "HistoryCache", Cache)
    assert main([
        "backtest", "examples/sample_portfolio_v3.csv", "--usd-cad", "1.35",
        "--start", "2026-01-05", "--end", "2026-05-16",
        "--strategies", "buy_hold", "quarterly", "--json",
    ]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert list(obj["strategies"]) == ["buy_hold", "quarterly"]


def test_cli_invalid_fee_gracefully_fails(capsys):
    rc = main([
        "backtest", "examples/sample_portfolio_v3.csv",
        "--start", "2026-01-01", "--end", "2026-05-16", "--fee-bps", "-10",
    ])
    assert rc == 2
    assert "INPUT ERROR" in capsys.readouterr().err
