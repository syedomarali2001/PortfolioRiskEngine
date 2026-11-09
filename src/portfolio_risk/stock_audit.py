"""V7.2 stratified stock metadata coverage audit (Yahoo, best effort).

A US-listed ADR may be domiciled outside the US. Canadian listings need not be
Canadian-domiciled; country here is *reported company location*, not revenue.
"""
from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from portfolio_risk.input import normalize_symbol
from portfolio_risk.security_metadata import YahooSecurityMetadataProvider, category_match


@dataclass(frozen=True)
class AuditedStock:
    symbol: str
    market: str  # user-curated LISTING universe: US_STOCK or CAD_STOCK
    group: str
    status: str
    quote_type: str | None
    recognized_equity: bool
    type_mismatch: bool
    company_name: str | None
    sector: str | None
    industry: str | None
    company_country: str | None
    exchange: str | None
    currency: str | None
    note: str


def read_stock_universe(path: str | Path) -> tuple[tuple[str, str, str], ...]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fp:
        reader=csv.DictReader(fp)
        if not reader.fieldnames or not {"symbol", "market", "group"}.issubset({x.strip().lower() for x in reader.fieldnames}):
            raise ValueError("Stock audit CSV requires symbol,market,group")
        reader.fieldnames=[x.strip().lower() for x in reader.fieldnames]
        rows=[];seen=set()
        for line,row in enumerate(reader,2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed stock audit row {line}")
            symbol=normalize_symbol(row['symbol'])
            market=row['market'].strip().upper();group=row['group'].strip()
            if market not in {'US_STOCK','CAD_STOCK'} or not group:
                raise ValueError(f"Stock audit row {line}: invalid market/group")
            if symbol in seen:
                raise ValueError(f"Duplicate stock ticker: {symbol}")
            seen.add(symbol);rows.append((symbol,market,group))
    if not rows:
        raise ValueError("Stock audit universe empty")
    return tuple(rows)


def run_stock_audit(universe: tuple[tuple[str,str,str],...],
                    provider: YahooSecurityMetadataProvider, *, delay_ms: int = 150,
                    on_progress: Callable[[int,int,str],None] | None = None) -> list[AuditedStock]:
    if not 0 <= delay_ms <= 10_000:
        raise ValueError('Invalid delay-ms')
    rows=[]
    for i,(symbol,market,group) in enumerate(universe,1):
        if on_progress:
            on_progress(i,len(universe),symbol)
        profile=provider.lookup(symbol)
        match=category_match(market,profile)
        rows.append(AuditedStock(symbol,market,group,profile.status,profile.quote_type,
                        profile.asset_kind=='equity',match is False,
                        profile.long_name,profile.sector,profile.industry,
                        profile.company_country,profile.exchange,profile.currency,profile.note))
        if delay_ms:
            time.sleep(delay_ms/1000)
    return rows


def stock_summary(rows: list[AuditedStock]) -> dict:
    return {
        'total':len(rows),
        'metadata_ok':sum(x.status=='ok' for x in rows),
        'equity_identified':sum(x.recognized_equity for x in rows),
        'sector_available':sum(bool(x.sector) for x in rows),
        'industry_available':sum(bool(x.industry) for x in rows),
        'company_country_available':sum(bool(x.company_country) for x in rows),
        'exchange_available':sum(bool(x.exchange) for x in rows),
        'currency_available':sum(bool(x.currency) for x in rows),
        'provider_unavailable':sum(x.status=='unavailable' for x in rows),
        'category_mismatches':sum(x.type_mismatch for x in rows),
        'by_type':dict(sorted(Counter(x.quote_type or 'UNKNOWN' for x in rows).items())),
    }


def stock_report(rows: list[AuditedStock]) -> dict:
    return {
      'description':'Curated mixed US/Canadian listed stock metadata diagnostic sample, not a full exchange census',
      'provider':'Yahoo Finance via yfinance, best effort; personal-use/data licensing restrictions',
      'country_definition':'Reported company location, NOT listing country or geography of revenues',
      'summary':stock_summary(rows),
      'by_market':{m:stock_summary([x for x in rows if x.market==m]) for m in sorted({x.market for x in rows})},
      'by_group':{m:stock_summary([x for x in rows if x.group==m]) for m in sorted({x.group for x in rows})},
      'assets':[asdict(x) for x in rows],
    }
