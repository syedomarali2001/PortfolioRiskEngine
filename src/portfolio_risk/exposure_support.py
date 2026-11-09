"""Explicit capability contracts for current-snapshot look-through analysis.

This does NOT make claims about historical-price availability. Price-risk analysis
uses the separate price provider, and may be available for price-only assets.
"""

from __future__ import annotations

from datetime import date
from typing import Mapping

from portfolio_risk.exposure import UNKNOWN, FundSnapshot, SecurityMetadata
from portfolio_risk.providers.tickerinside import FundRecord


def assess_exposure_support(
    symbols: list[str],
    *,
    classifications: Mapping[str, SecurityMetadata],
    snapshots: Mapping[str, FundSnapshot],
    provider_records: Mapping[str, FundRecord],
    cutoff: date,
    max_age_days: int,
    unsupported_metadata: Mapping[str, str] | None = None,
    effective_coverage: Mapping[str, float] | None = None,
) -> dict[str, dict]:
    """Return one decision per portfolio ticker; never silently assume unknown=equity."""
    output = {}
    unsupported_metadata = unsupported_metadata or {}
    effective_coverage = effective_coverage or {}
    for symbol in symbols:
        rec = provider_records.get(symbol)
        snap = snapshots.get(symbol)
        meta = classifications.get(symbol)
        status = "unverified"
        rationale = (
            "Asset type is unverified; enable --auto-stock-metadata or supply classifications"
        )
        kinds = "unknown"
        if rec is not None:
            kinds = rec.holdings_kind
            if rec.exposure_support == "price_only":
                status = "price_only"
                rationale = (
                    f"Fund {kinds} holdings cannot be safely expanded into stock overlap; "
                    f"verification={rec.equity_validation}. Bond/futures/derivative look-through unsupported"
                )
            else:
                status, rationale = (
                    "supported",
                    "Validated fund look-through with dated provider holdings",
                )
                if not rec.holdings_as_of or rec.holdings_as_of > cutoff:
                    status, rationale = (
                        "partial",
                        "Fund holdings date is unavailable or after analysis cutoff",
                    )
                elif (cutoff - rec.holdings_as_of).days > max_age_days:
                    status, rationale = (
                        "partial",
                        f"Fund holdings are {(cutoff - rec.holdings_as_of).days} days old",
                    )
                if rec.known_pct < 95:
                    status = "partial"
                    rationale += f"; only {rec.known_pct:.2f}% of fund weight is identified"
                if rec.known_pct > 100.00001:
                    status, rationale = (
                        "price_only",
                        "Fund gross holdings exceed 100%; not valid long-only exposure",
                    )
                if snap is None:
                    status, rationale = (
                        "price_only",
                        "Validated fund data could not be converted to an exposure snapshot",
                    )
        elif snap is not None:
            kinds = "fund_snapshot"
            fraction = sum(h.fraction for h in snap.holdings)
            status = "supported" if fraction >= 0.95 and snap.as_of is not None else "partial"
            rationale = "User-supplied or alternative fund snapshot; verify its disclosure and holding types"
            if any(h.asset_class in {"bond", "other", "UNKNOWN"} for h in snap.holdings):
                status = "price_only"
                rationale = "Manual snapshot includes bond, other or unverified constituent type; not validated for equity overlap"
            if (
                snap.as_of is None
                or snap.as_of > cutoff
                or (cutoff - snap.as_of).days > max_age_days
            ):
                if status != "price_only":
                    status = "partial"
            if fraction < 0.95:
                rationale += f"; identified weight {fraction:.1%}"
        elif meta is not None:
            kinds = meta.asset_class
            if meta.asset_class == "equity":
                status = (
                    "supported"
                    if meta.sector not in ("", UNKNOWN) and meta.country not in ("", UNKNOWN)
                    else "partial"
                )
                rationale = (
                    "Direct equity with sector and company location"
                    if status == "supported"
                    else "Direct equity recognized, but sector or company location is missing"
                )
            elif meta.asset_class == "crypto":
                status, rationale = (
                    "supported",
                    "Direct cryptocurrency identified (no company-sector/country applicable)",
                )
            elif meta.asset_class == "commodity":
                status, rationale = (
                    "supported",
                    "Direct physical commodity classification (stock sector not applicable)",
                )
            elif meta.asset_class in {"etf", "bond"}:
                status, rationale = (
                    "price_only",
                    "Fund/bond constituents unavailable or unsupported; price-risk may still work",
                )
            else:
                rationale = "Asset classification unknown; exposure cannot be validated"
        if symbol in unsupported_metadata:
            status, rationale = "price_only", unsupported_metadata[symbol]
        if (
            symbol in effective_coverage
            and status == "supported"
            and effective_coverage[symbol] < 0.95 - 1e-9
        ):
            status = "partial"
            rationale += f"; only {effective_coverage[symbol]:.1%} underlying exposure resolved after nested look-through"
        output[symbol] = {
            "exposure_status": status,
            "instrument_kind": kinds,
            "reason": rationale,
            "price_risk_status": "not_checked_here",
            "note": "Price-based analysis may work separately if history can be retrieved; sector/country do not apply to all types",
        }
    return output
