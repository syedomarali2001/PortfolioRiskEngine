"""Ticker lookup for the add-holdings popup. Best-effort, never a price source."""
from __future__ import annotations
from .core import normalized_symbol, PortfolioError
from .quotes import quote_currency_from_search


def lookup(query: str) -> list[tuple[str, dict]]:
    q = str(query).strip()
    if len(q) < 1:
        return []
    try:
        import yfinance as yf
        response = yf.Search(q, max_results=10, news_count=0, lists_count=0,
                             include_research=False, include_cb=False, timeout=8,
                             raise_errors=False)
        rows = response.quotes or []
    except Exception:
        return []
    found = []
    seen = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            symbol = normalized_symbol(raw.get("symbol", ""))
        except PortfolioError:
            continue
        typ = str(raw.get("quoteType", "")).upper()
        if typ not in {"EQUITY", "ETF", "MUTUALFUND", "CRYPTOCURRENCY"}:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        title = raw.get("shortname") or raw.get("longname") or symbol
        display = f"{symbol} · {title}"
        info = {"symbol": symbol, "name": str(title), "type": typ}
        evidence = quote_currency_from_search(raw)
        if evidence is not None:
            info["quote_currency"] = evidence.currency
            info["currency_source"] = evidence.source
        found.append((display[:130], info))
    return found[:10]
