"""Display-only CAD/USD valuation from the existing V8.4 FX service.

Never use a mixed-currency sum without a dated, positive provider quote.
The backend independently performs its own dated conversions for analytics.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from portfolio_risk.fx import FXQuote, convert_amount, latest_usd_cad_quote
from portfolio_risk.cache import HistoryCache
from portfolio_risk.providers.yahoo import YahooFinanceProvider


@dataclass(frozen=True)
class DisplayValue:
    total: Decimal
    shares: dict[str, float]
    base: str


def get_fx_quote(as_of: dt.date, cache_dir: str | Path = 'data/cache/v1') -> FXQuote:
    """Reuse V8.4's Yahoo-powered, recent, dated USD/CAD quote validation."""
    return latest_usd_cad_quote(YahooFinanceProvider(), HistoryCache(cache_dir), as_of)


def display_portfolio_value(holdings: list[dict], base: str,
                            usd_cad: Decimal | str | None = None) -> DisplayValue:
    if base not in {"CAD", "USD"}:
        raise ValueError('Reporting currency must be CAD or USD')
    if not holdings:
        raise ValueError('At least one holding required')
    totals: dict[str, Decimal] = {}
    for row in holdings:
        symbol = str(row['symbol'])
        if symbol in totals:
            raise ValueError('Duplicate holdings cannot be combined')
        value = Decimal(str(row['market_value']))
        if not value.is_finite() or value <= 0:
            raise ValueError('Each holding must be a positive finite value')
        currency = str(row['currency']).upper()
        if currency != base and usd_cad is None:
            raise ValueError('A dated FX quote is needed for mixed currencies')
        totals[symbol] = convert_amount(value, currency, base,
                                        None if usd_cad is None else Decimal(str(usd_cad)))
    total = sum(totals.values(), Decimal('0'))
    return DisplayValue(total=total,
                        shares={ticker: float(v/total*100) for ticker,v in totals.items()}, base=base)
