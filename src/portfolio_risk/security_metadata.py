"""Best-effort Yahoo security identity and metadata (not an ownership feed).

The provider does not guarantee a match, sector, industry, or domicile. Never
silently infer a country from a listing suffix or use market domicile as
revenue/FX exposure. Yahoo market-data terms apply to live use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from portfolio_risk.input import normalize_symbol


class SecurityMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class SecurityProfile:
    symbol: str
    status: str
    quote_type: str | None = None
    asset_kind: str = "unknown"
    long_name: str | None = None
    exchange: str | None = None
    currency: str | None = None
    sector: str | None = None
    industry: str | None = None
    company_country: str | None = None
    website: str | None = None
    market_cap: float | None = None
    note: str = ""
    source: str = "Yahoo Finance / yfinance (best effort; personal-use terms)"

    def as_dict(self) -> dict:
        return asdict(self)


TYPE_TO_KIND = {
    "EQUITY": "equity",
    "ETF": "fund",
    "MUTUALFUND": "fund",
    "ETN": "fund",
    "CLOSED_END_FUND": "fund",
    "CRYPTOCURRENCY": "crypto",
    "FUTURE": "commodity_future",
    "INDEX": "index",
    "CURRENCY": "currency",
}
EXPECTED_KINDS = {
    "US_STOCK": {"equity"},
    "CAD_STOCK": {"equity"},
    "US_ETF": {"fund"},
    "CAD_ETF": {"fund"},
    "COMMODITY_ETP": {"fund"},
    "CRYPTO": {"crypto"},
}


def _str(info: dict, *keys: str) -> str | None:
    for key in keys:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_yahoo_profile(symbol: str, payload: object) -> SecurityProfile:
    sym = normalize_symbol(symbol)
    if not isinstance(payload, dict):
        raise SecurityMetadataError(f"{sym}: company info is not a dictionary")
    reported = _str(payload, "symbol")
    if reported is not None and normalize_symbol(reported) != sym:
        raise SecurityMetadataError(f"{sym}: Yahoo returned mismatched ticker {reported}")
    quote_type = _str(payload, "quoteType")
    kind = TYPE_TO_KIND.get(quote_type.upper(), "unknown") if quote_type else "unknown"
    market_cap = payload.get("marketCap")
    if isinstance(market_cap, bool) or not isinstance(market_cap, (int, float)) or market_cap < 0:
        market_cap = None
    elif not __import__("math").isfinite(market_cap):
        market_cap = None
    key_count = sum(
        bool(_str(payload, key))
        for key in ("shortName", "longName", "quoteType", "sector", "country", "currency")
    )
    return SecurityProfile(
        symbol=sym,
        status="ok" if key_count and quote_type else "partial" if key_count else "missing",
        quote_type=quote_type,
        asset_kind=kind,
        long_name=_str(payload, "longName", "shortName", "displayName"),
        exchange=_str(payload, "exchange", "fullExchangeName"),
        currency=_str(payload, "currency"),
        sector=_str(payload, "sector"),
        industry=_str(payload, "industry"),
        company_country=_str(payload, "country"),
        website=_str(payload, "website"),
        market_cap=float(market_cap) if market_cap is not None else None,
        note="Company country = reported company location, not revenue/FX/geographic holdings exposure",
    )


def category_match(category: str, profile: SecurityProfile) -> bool | None:
    """None = provider cannot verify identity, not a failed ticker."""
    if profile.asset_kind == "unknown":
        return None
    expected = EXPECTED_KINDS.get(category.upper())
    return profile.asset_kind in expected if expected else None


class YahooSecurityMetadataProvider:
    def __init__(self, getter: Callable[[str], dict] | None = None):
        self.getter = getter or self._fetch

    @staticmethod
    def _fetch(symbol: str) -> dict:
        import yfinance as yf

        return yf.Ticker(symbol).get_info()

    def lookup(self, symbol: str) -> SecurityProfile:
        sym = normalize_symbol(symbol)
        try:
            return parse_yahoo_profile(sym, self.getter(sym))
        except Exception as exc:
            # Ticker may exist but metadata may be blocked / rate-limited.
            return SecurityProfile(
                sym,
                status="unavailable",
                note=f"Metadata lookup unavailable: {type(exc).__name__}: {str(exc)[:160]}",
            )
