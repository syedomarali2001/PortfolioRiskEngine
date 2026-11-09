"""V6 CLI smoke tests with mocked market data, no internet required."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from portfolio_risk.cli import main
from portfolio_risk.models import PriceHistory


class Cache:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        return None

    def put(self, *args, **kwargs):
        pass


class Provider:
    def get_history(self, symbol, start, end):
        ix = pd.bdate_range("2024-01-01", "2026-10-15")
        k = {
            "XEQT.TO": 1,
            "QQQ": 2,
            "SCHD": 3,
            "SPMO": 4,
            "BTC-USD": 5,
            "GLD": 6,
            "USDCAD=X": 7,
            "SPY": 8,
        }[symbol]
        x = np.arange(len(ix))
        if symbol == "USDCAD=X":
            prices = np.full(len(ix), 1.35)
        else:
            periodic = 0.002 * np.sin(x / (7 + k)) + 0.001 * np.cos(x / (11 + k))
            prices = 100 * np.cumprod(1.0002 + k * 0.00002 + periodic)
        vals = pd.Series(prices, index=ix).loc[
            lambda s: (s.index.date >= start) & (s.index.date < end)
        ]
        return PriceHistory(
            symbol, vals, start, end, "MockProvider", datetime.now(timezone.utc), None
        )


def setup(monkeypatch):
    import portfolio_risk.cli as cli

    monkeypatch.setattr(cli, "YahooFinanceProvider", Provider)
    monkeypatch.setattr(cli, "HistoryCache", Cache)


def test_cli_construct_plaintext(monkeypatch, capsys):
    setup(monkeypatch)
    result = main(
        [
            "construct",
            "examples/sample_portfolio_v3.csv",
            "--base",
            "CAD",
            "--start",
            "2024-01-01",
            "--end",
            "2026-10-16",
            "--usd-cad",
            "1.35",
            "--lookback",
            "126",
            "--momentum-skip",
            "10",
            "--capital",
            "20000",
            "--fee-bps",
            "10",
            "--rebalance",
            "quarterly",
            "--risk-free",
            "3",
            "--benchmark",
            "SPY",
            "--benchmark-currency",
            "USD",
        ]
    )
    io = capsys.readouterr()
    assert result == 0, io.err
    assert "V6 portfolio construction" in io.out
    assert "input_buy_hold" in io.out
    assert "min_variance" in io.out
    assert "momentum_top_k" in io.out
    assert "Benchmark SPY" in io.out
    assert "Signals exclude" in io.out
    assert '{"Method"' not in io.out


def test_cli_construct_json_result(monkeypatch, capsys):
    setup(monkeypatch)
    result = main(
        [
            "construct",
            "examples/sample_portfolio_v3.csv",
            "--base",
            "USD",
            "--start",
            "2024-01-01",
            "--end",
            "2026-10-16",
            "--usd-cad",
            "1.35",
            "--lookback",
            "80",
            "--momentum-skip",
            "10",
            "--methods",
            "input_buy_hold",
            "equal_weight",
            "min_variance",
            "--evaluation-start",
            "2025-06-01",
            "--json",
        ]
    )
    io = capsys.readouterr()
    assert result == 0, io.err
    payload = json.loads(io.out)
    assert set(payload["strategies"]) == {"input_buy_hold", "equal_weight", "min_variance"}
    assert payload["evaluation_start"] == "2025-06-01"
    assert all(item["evaluation_metrics"] is not None for item in payload["strategies"].values())
    assert payload["market_data"]["USDCAD=X"]["source"] == "MockProvider"
    assert all(
        not item["last_target"] or sum(item["last_target"].values()) > 0.999999
        for item in payload["strategies"].values()
    )


def test_cli_reject_invalid_inputs_before_api(capsys):
    code = main(["construct", "examples/sample_portfolio_v3.csv", "--fee-bps", "-10"])
    assert code == 2
    assert "INPUT ERROR" in capsys.readouterr().err


def test_cli_no_historical_period_returns_clear_error(monkeypatch, capsys):
    setup(monkeypatch)
    code = main(
        [
            "construct",
            "examples/sample_portfolio_v3.csv",
            "--start",
            "2026-06-01",
            "--end",
            "2026-10-16",
            "--usd-cad",
            "1.35",
            "--lookback",
            "252",
        ]
    )
    assert code == 1
    assert "CONSTRUCTION ERROR" in capsys.readouterr().err


def test_cli_reject_evaluation_after_sample(monkeypatch, capsys):
    setup(monkeypatch)
    code = main(
        [
            "construct",
            "examples/sample_portfolio_v3.csv",
            "--start",
            "2024-01-01",
            "--end",
            "2026-10-16",
            "--usd-cad",
            "1.35",
            "--evaluation-start",
            "2027-01-01",
        ]
    )
    assert code == 1
    assert "CONSTRUCTION ERROR" in capsys.readouterr().err
