"""V7.1 market-wide coverage audit. Categories are declared test cases, not guessed.

TickerInside supplies US ETF/ETP holdings; Yahoo optionally checks adjusted
price availability for ALL categories. No historical ETF constituents assumed.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from portfolio_risk.cache import HistoryCache
from portfolio_risk.input import normalize_symbol
from portfolio_risk.providers.tickerinside import (
    FundUnavailable, TickerInsideClient, TickerInsideError,
)
from portfolio_risk.providers.yahoo import YahooFinanceProvider
from portfolio_risk.security_metadata import YahooSecurityMetadataProvider, category_match
from portfolio_risk.service import get_validated_history

CATEGORIES = frozenset({"US_ETF", "CAD_ETF", "US_STOCK", "CAD_STOCK", "COMMODITY_ETP", "CRYPTO"})


@dataclass(frozen=True)
class AuditAsset:
    symbol: str
    category: str
    quote_currency: str


@dataclass(frozen=True)
class AuditResult:
    symbol: str
    category: str
    holdings_status: str
    holdings_count: int | None = None
    listed_pct: float | None = None
    holdings_as_of: str | None = None
    age_days: int | None = None
    source_layer: str | None = None
    holdings_kind: str | None = None
    sector_count: int = 0
    country_count: int = 0
    has_expense_ratio: bool = False
    price_status: str = "not_checked"
    price_obs: int | None = None
    price_last: str | None = None
    provider_note: str = ""
    source_url: str | None = None
    provider_citation: str | None = None
    top_holdings: tuple[tuple[str, float], ...] = ()
    sectors: tuple[tuple[str, float], ...] = ()
    countries: tuple[tuple[str, float], ...] = ()
    metadata_status: str = "not_checked"
    metadata_quote_type: str | None = None
    metadata_asset_kind: str | None = None
    metadata_category_match: bool | None = None
    metadata_name: str | None = None
    metadata_sector: str | None = None
    metadata_industry: str | None = None
    metadata_company_country: str | None = None
    metadata_exchange: str | None = None
    metadata_currency: str | None = None
    metadata_note: str = ""


def load_audit_universe(path: str | Path) -> tuple[AuditAsset, ...]:
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"Audit universe not found: {file}")
    with file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"symbol", "category", "quote_currency"}.issubset(
                {str(n).strip().lower() for n in reader.fieldnames}):
            raise ValueError("Audit CSV requires symbol,category,quote_currency")
        reader.fieldnames = [n.strip().lower() for n in reader.fieldnames]
        output = []
        seen = set()
        for row_no, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed audit CSV row {row_no}")
            if not any(str(v).strip() for v in row.values()):
                continue
            sym = normalize_symbol(row["symbol"])
            category = row["category"].strip().upper()
            currency = row["quote_currency"].strip().upper()
            if sym in seen:
                raise ValueError(f"Duplicate audit symbol {sym}")
            if category not in CATEGORIES or currency not in {"CAD", "USD"}:
                raise ValueError(f"Row {row_no}: unsupported category or quote currency")
            seen.add(sym)
            output.append(AuditAsset(sym, category, currency))
    if not output:
        raise ValueError("Audit universe is empty")
    return tuple(output)


def run_coverage_audit(
    universe: tuple[AuditAsset, ...], client: TickerInsideClient,
    *, check_prices: bool = False, provider=None, price_cache=None,
    as_of: date | None = None, market_days: int = 45,
    refresh: bool = False, on_progress: Callable[[int, int, str], None] | None = None,
    check_metadata: bool = False, metadata_provider: YahooSecurityMetadataProvider | None = None,
) -> list[AuditResult]:
    cutoff = as_of or date.today()
    if market_days < 10:
        raise ValueError("market_days must be >= 10")
    if check_prices:
        provider = provider or YahooFinanceProvider()
        price_cache = price_cache or HistoryCache()
    fund_symbols, index_meta = client.index(refresh=refresh)
    if check_metadata:
        metadata_provider = metadata_provider or YahooSecurityMetadataProvider()
    results: list[AuditResult] = []
    total = len(universe)
    for i, asset in enumerate(universe, 1):
        if on_progress:
            on_progress(i, total, asset.symbol)
        fields = {"symbol": asset.symbol, "category": asset.category,
                  "holdings_status": "not_applicable", "provider_note": "Not an ETF constituent question"}
        sym = asset.symbol
        if asset.category in {"US_ETF", "COMMODITY_ETP"}:
            if sym not in fund_symbols:
                fields.update(holdings_status="not_indexed", provider_note="Not in TickerInside US fund index")
            else:
                try:
                    record = client.fund(sym, refresh=refresh)
                    age = (cutoff - record.holdings_as_of).days if record.holdings_as_of else None
                    count = len(record.holdings)
                    fields.update(
                        holdings_status="holdings_available" if count and record.holdings_as_of else "fund_without_holdings",
                        holdings_count=count,
                        listed_pct=round(record.known_pct, 3),
                        holdings_as_of=record.holdings_as_of.isoformat() if record.holdings_as_of else None,
                        age_days=age,
                        source_layer=record.source_layer,
                        holdings_kind=record.holdings_kind,
                        sector_count=len(record.sector_pct),
                        country_count=len(record.country_pct),
                        has_expense_ratio=record.expense_ratio_pct is not None,
                        source_url=record.source_url,
                        provider_citation=record.cite_as,
                        top_holdings=tuple(sorted(record.holdings.items(), key=lambda x: -x[1])[:5]),
                        sectors=tuple(sorted(record.sector_pct.items(), key=lambda x: -x[1])[:5]),
                        countries=tuple(sorted(record.country_pct.items(), key=lambda x: -x[1])[:5]),
                        provider_note=("Holdings percentages are gross/nonstandard; not safe for long-only expansion" if record.known_pct > 100.00001 else
                                       "Holdings file is dated after analysis cutoff" if age is not None and age < 0 else
                                       "Fund exists but disclosure date is unavailable" if not record.holdings_as_of else ""),
                    )
                except (TickerInsideError, ValueError) as exc:
                    fields.update(holdings_status="provider_error", provider_note=str(exc)[:250])
        elif asset.category == "CAD_ETF":
            fields.update(holdings_status="outside_provider_scope", provider_note="Canadian ETF: TickerInside US-only; not a failure of price-data coverage")
        if check_prices:
            try:
                result = get_validated_history(provider, price_cache, sym,
                                              cutoff - timedelta(days=market_days), cutoff + timedelta(days=1),
                                              refresh=refresh)
                fields.update(price_status="ok" if result.quality.passed else "quality_warning",
                              price_obs=result.quality.observation_count,
                              price_last=result.quality.last_date.isoformat())
                if result.quality.warnings:
                    fields["provider_note"] += " | Price: " + "; ".join(result.quality.warnings)
            except Exception as exc:
                fields.update(price_status="unavailable", provider_note=(fields["provider_note"] +
                              f" | Price unavailable: {exc}")[:350])
        if check_metadata:
            profile = metadata_provider.lookup(sym)
            fields.update(
                metadata_status=profile.status,
                metadata_quote_type=profile.quote_type,
                metadata_asset_kind=profile.asset_kind,
                metadata_category_match=category_match(asset.category, profile),
                metadata_name=profile.long_name,
                metadata_sector=profile.sector,
                metadata_industry=profile.industry,
                metadata_company_country=profile.company_country,
                metadata_exchange=profile.exchange,
                metadata_currency=profile.currency,
                metadata_note=profile.note,
            )
        results.append(AuditResult(**fields))
    return results


def audit_summary(results: list[AuditResult]) -> dict:
    by_category = {}
    for category in sorted({x.category for x in results}):
        items = [x for x in results if x.category == category]
        by_category[category] = {
            "tickers": len(items),
            "holdings_available": sum(x.holdings_status == "holdings_available" for x in items),
            "not_indexed": sum(x.holdings_status == "not_indexed" for x in items),
            "outside_scope": sum(x.holdings_status == "outside_provider_scope" for x in items),
            "price_available": sum(x.price_status in {"ok", "quality_warning"} for x in items),
        }
    return {"total": len(results), "by_category": by_category,
            "tickerinside_funds_with_holdings": sum(x.holdings_status == "holdings_available" for x in results),
            "price_checked": sum(x.price_status != "not_checked" for x in results),
            "metadata_checked": sum(x.metadata_status != "not_checked" for x in results),
            "metadata_identity_available": sum(x.metadata_status == "ok" for x in results),
            "metadata_sector_available": sum(bool(x.metadata_sector) for x in results),
            "metadata_industry_available": sum(bool(x.metadata_industry) for x in results),
            "metadata_company_country_available": sum(bool(x.metadata_company_country) for x in results),
            "category_mismatches": sum(x.metadata_category_match is False for x in results)}


def audit_as_dict(results: list[AuditResult]) -> dict:
    return {"provider": "TickerInside (US ETF/ETP holdings, CC BY 4.0)",
            "attribution": "https://tickerinside.com",
            "description": "Category declared by example CSV; price availability is separate from constituent availability",
            "summary": audit_summary(results), "assets": [asdict(x) for x in results]}
