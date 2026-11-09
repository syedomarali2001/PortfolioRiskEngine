"""V7.2: stratified, documented U.S. ETF holdings data audit.

Universe is a *curated diagnostic sample* with unambiguous themes and labels.
It is not an independent representative random sample of all listed ETFs.
Use separate denominators for indexed, retrieved, equity-look-through eligible,
classified, fresh and weight-complete funds. Network failures != not indexed.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from portfolio_risk.input import normalize_symbol
from portfolio_risk.providers.tickerinside import (
    FundUnavailable,
    TickerInsideClient,
    TickerInsideError,
)


@dataclass(frozen=True)
class AuditFund:
    symbol: str
    theme: str
    issuer: str


@dataclass(frozen=True)
class AuditedFund:
    symbol: str
    theme: str
    issuer: str
    indexed: bool
    status: str
    positions: int | None = None
    weight_pct: float | None = None
    disclosed_on: str | None = None
    age_days: int | None = None
    source_layer: str | None = None
    holdings_kind: str | None = None
    sector_count: int = 0
    country_count: int = 0
    expense_ratio_available: bool = False
    equity_lookthrough_safe: bool = False
    near_full_weight: bool = False
    current_120d: bool = False
    note: str = ""
    source_url: str | None = None
    exposure_support: str = "unknown"
    equity_validation: str = "unverified"
    excluded_non_equity_positions: int = 0


def read_etf_universe(path: str | Path) -> tuple[AuditFund, ...]:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"symbol", "theme", "issuer"}.issubset(
            {x.strip().lower() for x in reader.fieldnames}
        ):
            raise ValueError("ETF audit CSV must contain symbol,theme,issuer")
        reader.fieldnames = [x.strip().lower() for x in reader.fieldnames]
        out = []
        seen = set()
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed ETF universe row {line}")
            symbol = normalize_symbol(row["symbol"])
            if symbol in seen:
                raise ValueError(f"Duplicate ETF symbol: {symbol}")
            if symbol.endswith(".TO") or symbol.endswith("-USD"):
                raise ValueError(f"ETF audit expects U.S. fund tickers, got {symbol}")
            theme, issuer = row["theme"].strip(), row["issuer"].strip()
            if not theme or not issuer:
                raise ValueError(f"Row {line}: theme and issuer are required")
            out.append(AuditFund(symbol, theme, issuer))
            seen.add(symbol)
    if not out:
        raise ValueError("ETF audit universe is empty")
    return tuple(out)


def run_etf_audit(
    universe: tuple[AuditFund, ...],
    client: TickerInsideClient,
    *,
    as_of: date | None = None,
    refresh: bool = False,
    delay_ms: int = 100,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[AuditedFund], dict]:
    if not 0 <= delay_ms <= 10_000:
        raise ValueError("--delay-ms must be between 0 and 10000")
    cutoff = as_of or date.today()
    fund_index, meta = client.index(refresh=refresh)
    results = []
    for num, entry in enumerate(universe, 1):
        if on_progress:
            on_progress(num, len(universe), entry.symbol)
        if entry.symbol not in fund_index:
            results.append(
                AuditedFund(entry.symbol, entry.theme, entry.issuer, False, "not_indexed")
            )
            continue
        try:
            rec = client.fund(entry.symbol, refresh=refresh)
            age = (cutoff - rec.holdings_as_of).days if rec.holdings_as_of else None
            full = 95 <= rec.known_pct <= 100.00001
            equity = (
                rec.is_equity_holdings
                and bool(rec.holdings)
                and bool(rec.holdings_as_of)
                and rec.known_pct <= 100.00001
            )
            status = (
                "holdings_available" if rec.holdings and rec.holdings_as_of else "no_dated_holdings"
            )
            note = []
            if age is not None and age < 0:
                note.append("Disclosed after audit as-of date")
            if age is not None and age > 120:
                note.append("Stale (>120 days)")
            if rec.known_pct < 95:
                note.append("Less than 95% of fund weight identified")
            if not equity:
                note.append("Not eligible for equity-only look-through without more validation")
            if rec.non_equity_positions:
                note.append(
                    f"{len(rec.non_equity_positions)} futures/cash positions excluded from stock holdings"
                )
            results.append(
                AuditedFund(
                    entry.symbol,
                    entry.theme,
                    entry.issuer,
                    True,
                    status,
                    positions=len(rec.holdings),
                    weight_pct=round(rec.known_pct, 4),
                    disclosed_on=rec.holdings_as_of.isoformat() if rec.holdings_as_of else None,
                    age_days=age,
                    source_layer=rec.source_layer,
                    holdings_kind=rec.holdings_kind,
                    sector_count=len(rec.sector_pct),
                    country_count=len(rec.country_pct),
                    expense_ratio_available=rec.expense_ratio_pct is not None,
                    equity_lookthrough_safe=equity and age is not None and age >= 0,
                    near_full_weight=full,
                    current_120d=age is not None and 0 <= age <= 120,
                    note="; ".join(note),
                    source_url=rec.source_url,
                    exposure_support=rec.exposure_support,
                    equity_validation=rec.equity_validation,
                    excluded_non_equity_positions=len(rec.non_equity_positions),
                )
            )
        except FundUnavailable as exc:
            results.append(
                AuditedFund(
                    entry.symbol,
                    entry.theme,
                    entry.issuer,
                    True,
                    "index_record_missing",
                    note=str(exc)[:180],
                )
            )
        except (TickerInsideError, ValueError, OSError) as exc:
            message = str(exc)
            nonstandard = any(
                term in message.lower()
                for term in ("gross", "nonstandard", "reported weights exceed", "exceed 105")
            )
            results.append(
                AuditedFund(
                    entry.symbol,
                    entry.theme,
                    entry.issuer,
                    True,
                    "nonstandard_weights" if nonstandard else "provider_error",
                    note=message[:180],
                )
            )
        if delay_ms:
            time.sleep(delay_ms / 1000)
    return results, meta


def _stats(rows: list[AuditedFund]) -> dict:
    n = len(rows)
    def count(pred):
        return sum(bool(pred(x)) for x in rows)
    return {
        "total": n,
        "indexed": count(lambda x: x.indexed),
        "with_dated_holdings": count(lambda x: x.status == "holdings_available"),
        "equity_lookthrough_safe": count(lambda x: x.equity_lookthrough_safe),
        "near_full_weight_95pct": count(lambda x: x.near_full_weight),
        "current_120d": count(lambda x: x.current_120d),
        "with_sector_breakdown": count(lambda x: x.sector_count > 0),
        "with_country_breakdown": count(lambda x: x.country_count > 0),
        "provider_errors": count(lambda x: x.status in {"provider_error", "index_record_missing"}),
        "nonstandard_weights": count(lambda x: x.status == "nonstandard_weights"),
        "partial_weight": count(lambda x: x.weight_pct is not None and x.weight_pct < 95),
        "not_indexed": count(lambda x: x.status == "not_indexed"),
        "fund_types": dict(
            sorted(Counter(x.holdings_kind or "unknown" for x in rows if x.indexed).items())
        ),
        "price_only_exposure": count(lambda x: x.exposure_support == "price_only"),
        "provider_verified_equity": count(lambda x: x.equity_validation == "issuer_evidence"),
        "mixed_non_equity_positions": count(lambda x: x.excluded_non_equity_positions > 0),
    }


def etf_report(rows: list[AuditedFund], index_metadata: dict) -> dict:
    by_theme = {}
    by_issuer = {}
    for field, output in (("theme", by_theme), ("issuer", by_issuer)):
        keys = sorted({getattr(x, field) for x in rows})
        for key in keys:
            output[key] = _stats([x for x in rows if getattr(x, field) == key])
    return {
        "description": "Curated U.S. ETF diagnostic sample; results not statistically representative of all U.S. listed ETFs",
        "provider": "TickerInside (CC BY 4.0)",
        "attribution": "https://tickerinside.com",
        "provider_reported_index": index_metadata,
        "summary": _stats(rows),
        "by_theme": by_theme,
        "by_issuer": by_issuer,
        "assets": [asdict(x) for x in rows],
    }
