"""Conservative GUI trading-currency resolution.

Do not infer trading currency from the denomination of a user's holding value,
from ticker suffixes, or from US/CAD reporting currency. A known issuer-listed
security or explicit provider currency may be used; otherwise ask the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .core import normalized_symbol


@dataclass(frozen=True)
class CurrencyEvidence:
    currency: str
    source: str


# Narrow, independently source-verified listings. Never treat this as a rule
# for other US-listed tickers. SPMO is a US NYSE Arca listing in USD.
# https://www.invesco.com/us/en/financial-products/etfs/invesco-sp-500-momentum-etf.html
# https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy
VERIFIED_LISTINGS: dict[str, CurrencyEvidence] = {
    "SPMO": CurrencyEvidence("USD", "Invesco US / NYSE Arca listing"),
    "SPY": CurrencyEvidence("USD", "State Street US listing"),
}


def quote_currency_from_search(raw: dict) -> CurrencyEvidence | None:
    """Trust only an explicit USD/CAD quote currency, never 'us_market' etc."""
    currency = str(raw.get("currency") or raw.get("quoteCurrency") or "").strip().upper()
    if currency in {"USD", "CAD"}:
        return CurrencyEvidence(currency, "Yahoo search result")
    return None


def verified_currency(symbol: str, ticker_factory: Callable | None = None) -> CurrencyEvidence | None:
    """Best effort issuer or provider lookup; unavailable is NOT USD by default."""
    symbol = normalized_symbol(symbol)
    if symbol in VERIFIED_LISTINGS:
        return VERIFIED_LISTINGS[symbol]
    if ticker_factory is None:
        try:
            from yfinance import Ticker
        except ImportError:
            return None
        ticker_factory = Ticker
    try:
        ticker = ticker_factory(symbol)
    except Exception:
        return None
    sources = [
        ("Yahoo history metadata", lambda: ticker.get_history_metadata()),
        ("Yahoo fast quote metadata", lambda: ticker.get_fast_info()),
        ("Yahoo quote profile", lambda: ticker.get_info()),
    ]
    for source, fetch in sources:
        try:
            metadata = fetch()
            if not isinstance(metadata, dict):
                if hasattr(metadata, "get"):
                    metadata = {"currency": metadata.get("currency")}
                else:
                    continue
            evidence = quote_currency_from_search(metadata)
            if evidence is not None:
                return CurrencyEvidence(evidence.currency, source)
        except Exception:
            continue
    return None
