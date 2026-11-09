"""Optional bounded Yahoo metadata enrichment for largest *verified equity* holdings.

Avoid hundreds/thousands of requests; cache lookups, show unresolved companies.
Provider quote type must say EQUITY before metadata is used. This NEVER infers
security type from ticker spelling and doesn't silently classify whole ETFs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from portfolio_risk.exposure import UNKNOWN, FundSnapshot, SecurityMetadata
from portfolio_risk.exposure_classification import SECTORS, canonical_country, canonical_sector
from portfolio_risk.security_metadata import SecurityProfile, YahooSecurityMetadataProvider


def enrich_top_equities(
    portfolio_weights: Mapping[str, float],
    snapshots: Mapping[str, FundSnapshot],
    classifications: Mapping[str, SecurityMetadata],
    *,
    limit: int = 25,
    metadata_provider: YahooSecurityMetadataProvider | None = None,
    cache_dir: Path | None = None,
    ttl_days: int = 7,
    now: datetime | None = None,
    focus_sector: str | None = None,
    fund_sector_pct: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[dict[str, SecurityMetadata], dict]:
    if not 0 <= limit <= 200 or ttl_days < 0:
        raise ValueError("Enrichment limit must be 0..200; TTL nonnegative")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("Enrichment time must be timezone-aware")
    cache_dir = cache_dir or Path("data/cache/v8_stock_metadata")
    result = dict(classifications)
    normalized_focus = canonical_sector(focus_sector) if focus_sector else None
    if normalized_focus is not None and normalized_focus not in set(SECTORS.values()):
        raise ValueError("Focus sector must have a recognized sector name, e.g. financials")
    sector_mixes = fund_sector_pct or {}
    contributions = {}
    actual_contributions = {}
    for fund, record in snapshots.items():
        w = portfolio_weights.get(fund, 0.0)
        for child in record.holdings:
            if child.asset_class == "equity" and child.fraction > 0:
                actual_contributions[child.symbol] = (
                    actual_contributions.get(child.symbol, 0.0) + w * child.fraction
                )
                if normalized_focus is None:
                    ranking_weight = w * child.fraction
                else:
                    # The issuer's sector mix identifies relevant FUNDS, not which
                    # holdings in those funds belong to the requested sector.
                    mix = sector_mixes.get(fund, {})
                    relevance = (
                        sum(
                            float(v)
                            for k, v in mix.items()
                            if canonical_sector(k) == normalized_focus
                        )
                        / 100.0
                    )
                    if not 0 < relevance <= 1.0001:
                        continue
                    ranking_weight = w * child.fraction * relevance
                contributions[child.symbol] = contributions.get(child.symbol, 0.0) + ranking_weight
    eligible = [
        (s, v)
        for s, v in sorted(contributions.items(), key=lambda kv: (-kv[1], kv[0]))
        if not (s in result and result[s].asset_class == "equity" and result[s].sector != UNKNOWN)
    ]
    selected = eligible[:limit]
    provider = metadata_provider or YahooSecurityMetadataProvider()
    enriched = []
    unavailable = {}
    cache_hits = []
    matched = []
    for symbol, _ in selected:
        profile = None
        path = cache_dir / (hashlib.sha256(symbol.encode()).hexdigest() + ".json")
        try:
            if path.exists() and ttl_days:
                doc = json.loads(path.read_text(encoding="utf-8"))
                stored = datetime.fromisoformat(doc["retrieved_at"])
                if (
                    stored.tzinfo is not None
                    and doc.get("symbol") == symbol
                    and timedelta() <= current - stored <= timedelta(days=ttl_days)
                ):
                    profile = SecurityProfile(**doc["profile"])
                    cache_hits.append(symbol)
        except (ValueError, OSError, KeyError, TypeError):
            profile = None
        if profile is None:
            profile = provider.lookup(symbol)
            if profile.asset_kind == "equity" and profile.status in ("ok", "partial"):
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(
                            {
                                "symbol": symbol,
                                "retrieved_at": current.isoformat(),
                                "profile": profile.as_dict(),
                            },
                            allow_nan=False,
                        ),
                        encoding="utf-8",
                    )
                except OSError:
                    pass  # cache failure is not a reason to reject verified metadata
        if profile.asset_kind != "equity" or profile.sector is None:
            unavailable[symbol] = profile.note or "No verified equity sector data"
            continue
        result[symbol] = SecurityMetadata(
            symbol,
            "equity",
            canonical_sector(profile.sector),
            canonical_country(profile.company_country),
        )
        enriched.append(symbol)
        if normalized_focus is not None and canonical_sector(profile.sector) == normalized_focus:
            matched.append(symbol)
    return result, {
        "attempted": len(selected),
        "limit": limit,
        "cached": len(cache_hits),
        "enriched_count": len(enriched),
        "enriched_symbols": enriched,
        "unavailable": unavailable,
        "selected_top_equity_weights": {s: float(actual_contributions[s]) for s, _ in selected},
        "selection_ranking_scores": {s: float(score) for s, score in selected},
        "ranking_method": (
            "Portfolio equity weight times the issuer-reported fraction of the target sector, "
            "used ONLY for lookup prioritization; not a company-sector probability"
            if normalized_focus
            else "Total portfolio equity weight"
        ),
        "source": "Yahoo Finance company sector/domicile via yfinance; personal use, best effort",
        "focus_sector": normalized_focus,
        "matching_sector_symbols": matched,
        "note": (
            "Sector focus ranks unknown names from funds with issuer-reported exposure; "
            "a company is added to the requested sector ONLY after independent stock metadata verification."
            if normalized_focus
            else "Only the top verified equity positions were attempted; other constituents stay unclassified."
        ),
    }
