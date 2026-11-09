"""No-network V7 factor ETF-proxy CLI integration."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from portfolio_risk.cli import main
from portfolio_risk.models import PriceHistory


class Cache:
    def __init__(self, *a, **kw): pass
    def get(self, *a, **kw): return None
    def put(self, *a, **kw): pass


class Provider:
    def get_history(self, symbol, start, end):
        ix = pd.bdate_range("2024-01-01", "2026-10-15")
        mapping = {"XEQT.TO": 1, "QQQ": 2, "SCHD": 3, "SPMO": 4,
                   "BTC-USD": 5, "GLD": 6, "SPY": 7, "MTUM": 8,
                   "USDCAD=X": 9}
        n = mapping[symbol]
        j = np.arange(len(ix))
        values = (np.full(len(ix), 1.35) if symbol == "USDCAD=X" else
                  100*np.cumprod(1.0003+n*.00002+.003*np.sin(j/(7+n))))
        s = pd.Series(values, index=ix)
        s = s.loc[(s.index.date >= start) & (s.index.date < end)]
        return PriceHistory(symbol, s, start, end, "OfflineProvider",
                            datetime.now(timezone.utc), None)


def setup(monkeypatch):
    import portfolio_risk.cli as cli
    monkeypatch.setattr(cli, "YahooFinanceProvider", Provider)
    monkeypatch.setattr(cli, "HistoryCache", Cache)


def args():
    return ["factors", "examples/sample_portfolio_v3.csv", "--base", "CAD",
            "--start", "2024-01-01", "--end", "2026-10-16",
            "--usd-cad", "1.35", "--proxies", "SPY:USD", "MTUM:USD"]


def test_cli_factor_proxy_json(monkeypatch, capsys):
    setup(monkeypatch)
    assert main(args() + ["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["proxies"]["SPY"]["observations"] > 400
    assert data["proxies"]["MTUM"]["beta"] is not None
    assert "OLS" in data["method"]


def test_cli_factor_proxy_plaintext(monkeypatch, capsys):
    setup(monkeypatch)
    assert main(args()) == 0
    out = capsys.readouterr().out
    assert "V7 historical ETF proxy sensitivity" in out
    assert "MTUM" in out and "SPY" in out
    assert "not demonstrated trading alpha" in out


def test_cli_factor_input_invalid_proxy(capsys):
    assert main(args()[:-2] + ["BADFORMAT"]) == 1
    assert "FACTOR PROXY ERROR" in capsys.readouterr().err


def test_cli_factor_no_future_dates(capsys):
    assert main(["factors", "examples/sample_portfolio_v3.csv",
                 "--start", "2099-01-01"]) == 2
    assert "DATE ERROR" in capsys.readouterr().err
