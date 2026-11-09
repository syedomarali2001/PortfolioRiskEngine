"""Non-financial glue for scenario presentation. All results originate in V8.4."""
from __future__ import annotations

from .core import PortfolioError, normalized_symbol
from .presentation import what_if_flag


def prepare_what_if(rows: list[dict], expected_tickers: list[str]) -> str:
    expected = [normalized_symbol(x) for x in expected_tickers]
    parsed = {}
    for row in rows:
        symbol = normalized_symbol(row.get("Ticker"))
        if symbol in parsed:
            raise PortfolioError(f"Duplicate ticker in scenario: {symbol}")
        try:
            parsed[symbol] = float(row.get("Target weight (%)"))
        except (ValueError, TypeError) as exc:
            raise PortfolioError(f"{symbol}: enter a numeric scenario allocation") from exc
    if set(parsed) != set(expected):
        raise PortfolioError("Scenario must include every original portfolio holding exactly once")
    return what_if_flag(parsed)


def parse_group_limits(text: str, allowed_tickers: list[str]) -> list[str]:
    """Parse one group per line: e.g. bigtech:AAPL,MSFT=40.

    This only validates UI input; V8.4 has independent feasibility checks.
    """
    groups = []
    allowed = set(map(normalized_symbol, allowed_tickers))
    assigned = set()
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if raw.count(":") != 1 or raw.count("=") != 1:
            raise PortfolioError("Each group needs NAME:TICKER,TICKER=MAX%, one group per line")
        name, rhs = raw.split(":")
        members, cap = rhs.split("=")
        if not name.replace("_", "").isalnum():
            raise PortfolioError("Group names may contain only letters, digits, and underscores")
        tickers = [normalized_symbol(s) for s in members.split(",")]
        if not tickers or any(sym not in allowed for sym in tickers):
            raise PortfolioError("Group limits must reference portfolio tickers")
        if len(set(tickers)) != len(tickers) or any(sym in assigned for sym in tickers):
            raise PortfolioError("A ticker may appear in only one named group")
        assigned.update(tickers)
        try:
            val = float(cap)
        except (ValueError, TypeError) as exc:
            raise PortfolioError("Group caps must be numeric percentages") from exc
        if not 0 <= val <= 100:
            raise PortfolioError("Group caps must be between 0% and 100%")
        groups.append(f"{name}:{','.join(tickers)}={val:g}")
    return groups
