"""CLI V5 integration using an offline stub, no live API dependency."""
from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from portfolio_risk.cli import main, _parse_shocks
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
        ds = pd.date_range("2025-01-01", "2026-09-30", freq="B")
        offset = {"XEQT.TO": 1, "QQQ": 2, "SCHD": 3,
                  "SPMO": 4, "BTC-USD": 5, "GLD": 6,
                  "USDCAD=X": 7}[symbol]
        import numpy as np
        i = np.arange(len(ds))
        vals = ([1.35]*len(ds) if symbol == "USDCAD=X" else
                (100 * np.cumprod(1 + .001 * np.sin(i / (3 + offset)) + .0003 * offset)))
        ser = pd.Series(vals, index=ds, dtype=float)
        ser = ser.loc[(ser.index.date >= start) & (ser.index.date < end)]
        return PriceHistory(symbol, ser, start, end, "Stub", datetime.now(timezone.utc), None)


def setup(monkeypatch):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli, "YahooFinanceProvider", Provider)
    monkeypatch.setattr(cli, "HistoryCache", Cache)


def test_risk_json_integration(monkeypatch, capsys):
    setup(monkeypatch)
    code = main(["risk", "examples/sample_portfolio_v3.csv", "--base", "CAD",
                 "--usd-cad", "1.35", "--start", "2025-01-01",
                 "--end", "2026-10-01", "--strategy", "quarterly",
                 "--lookback", "252", "--fee-bps", "10", "--capital", "20000",
                 "--shock", "BTC-USD=-30", "--json"])
    out = capsys.readouterr()
    assert code == 0, out.err
    obj = json.loads(out.out)
    assert obj["strategy"] == "quarterly"
    assert obj["lookback"] == 252
    assert len(obj["coverage"]) == 4
    assert len(obj["risk_attribution"]) == 6
    assert len(obj["risk_estimates"]) == 4
    assert obj["stress_scenarios"][-1]["source"] == "hypothetical"
    assert obj["market_data"]["USDCAD=X"]["source"] == "Stub"


def test_risk_plain_text_integration(monkeypatch, capsys):
    setup(monkeypatch)
    code = main(["risk", "examples/sample_portfolio_v3.csv", "--base", "CAD",
                 "--usd-cad", "1.35", "--start", "2025-01-01",
                 "--end", "2026-10-01", "--lookback", "80"])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "V5 portfolio risk" in out.out
    assert "Value at Risk" in out.out
    assert "Rolling one-step-ahead" in out.out
    assert "buy_hold" in out.out


def test_v4_header_fix(monkeypatch, capsys):
    setup(monkeypatch)
    code = main(["backtest", "examples/sample_portfolio_v3.csv", "--usd-cad", "1.35",
                 "--start", "2025-01-01", "--end", "2026-10-01"])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "Strategy" in out.out and "End value" in out.out
    assert "{'Strategy':" not in out.out


@pytest.mark.parametrize("item", ["BTC-USD", "BTC-USD=abc", "BTC-USD=-110", "BTC-USD=nan"])
def test_bad_shock_syntax(item):
    with pytest.raises(ValueError):
        _parse_shocks([item])


def test_shock_duplicate_rejected():
    with pytest.raises(ValueError):
        _parse_shocks(["QQQ=-10", "QQQ=-20"])


def test_unknown_shock_symbol_fails_gracefully(monkeypatch, capsys):
    setup(monkeypatch)
    code = main(["risk", "examples/sample_portfolio_v3.csv", "--usd-cad", "1.35",
                 "--start", "2025-01-01", "--end", "2026-10-01", "--shock", "UNKNOWN=-20"])
    assert code == 1
    assert "RISK ERROR" in capsys.readouterr().err


def test_invalid_lookback_rejected_before_api(capsys):
    code = main(["risk", "examples/sample_portfolio_v3.csv", "--lookback", "5"])
    assert code == 2
    assert "INPUT ERROR" in capsys.readouterr().err
