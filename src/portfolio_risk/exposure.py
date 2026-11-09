"""V7: transparent, snapshot-based look-through exposure mathematics.

No prices, no third-party requests, and no inferred ETF constituents. Unknown
weights remain explicit. This is CURRENT snapshot analysis, not historical
constituent or factor history.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Mapping

from portfolio_risk.accounting import PortfolioValuation


class ExposureError(ValueError):
    """Malformed, inconsistent or cyclic look-through exposure data."""


ASSET_CLASSES = frozenset(
    {"equity", "etf", "bond", "cash", "commodity", "crypto", "other", "UNKNOWN"}
)
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SecurityMetadata:
    symbol: str
    asset_class: str = "UNKNOWN"
    sector: str = UNKNOWN
    country: str = UNKNOWN


@dataclass(frozen=True)
class Constituent:
    symbol: str
    fraction: float  # of ONE fund, 0..1
    asset_class: str
    sector: str = UNKNOWN
    country: str = UNKNOWN


@dataclass(frozen=True)
class FundSnapshot:
    symbol: str
    holdings: tuple[Constituent, ...]
    source: str
    as_of: date | None  # None means a retrieved/disclosure date could not be verified.
    retrieved_on: date | None = None


@dataclass(frozen=True)
class ExposureReport:
    base_currency: str
    total_value: str
    analysis_date: date
    direct_weights: dict[str, float]
    snapshot_fx: dict | None
    identified_underlying: dict[str, float]
    by_asset_class: dict[str, float]
    by_sector: dict[str, float]
    by_country: dict[str, float]
    unknown_underlying: float
    coverage_by_fund: dict[str, dict]
    pairwise_overlap: tuple[dict, ...]
    warnings: tuple[str, ...]


def _positive_weight(value: float, context: str) -> None:
    if not isfinite(value) or not (0 < value <= 1.0 + 1e-9):
        raise ExposureError(f"{context}: weight must be greater than zero and at most 100%")


def validate_snapshots(snapshots: Mapping[str, FundSnapshot]) -> None:
    """Fail closed for impossible weights, conflicts and cycles."""
    for fund_key, fund in snapshots.items():
        if fund_key != fund.symbol:
            raise ExposureError(f"Snapshot key {fund_key} does not match {fund.symbol}")
        if not fund.source.strip():
            raise ExposureError(f"{fund_key}: source is required")
        seen: set[str] = set()
        total = 0.0
        for h in fund.holdings:
            if h.symbol == fund_key:
                raise ExposureError(f"{fund_key}: a fund cannot contain itself")
            if h.symbol in seen:
                raise ExposureError(f"{fund_key}: duplicate underlying symbol {h.symbol}")
            seen.add(h.symbol)
            _positive_weight(h.fraction, f"{fund_key}/{h.symbol}")
            if h.asset_class not in ASSET_CLASSES:
                raise ExposureError(f"{fund_key}/{h.symbol}: unsupported asset class")
            total += h.fraction
        if total > 1.0000001:
            raise ExposureError(f"{fund_key}: constituent weights exceed 100%")

    def visit(sym: str, path: set[str]) -> None:
        if sym in path:
            raise ExposureError("Cyclic fund holdings: " + " -> ".join([*sorted(path), sym]))
        fund = snapshots.get(sym)
        if fund is None:
            return
        for item in fund.holdings:
            if item.symbol in snapshots:
                visit(item.symbol, path | {sym})

    for sym in snapshots:
        visit(sym, set())


def _look_through(
    symbol: str,
    meta: SecurityMetadata,
    snapshots: Mapping[str, FundSnapshot],
    classifications: Mapping[str, SecurityMetadata],
) -> tuple[
    dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float], float
]:
    """Return normalized leaf symbols, classes, sectors, countries, unresolved share.

    A constituent of an unknown nested ETF remains UNKNOWN; it is not treated
    as the shares of that ETF's ticker nor given its own sector/country.
    """
    leaf: dict[str, float] = defaultdict(float)
    comparable_leaf: dict[str, float] = defaultdict(float)
    types: dict[str, float] = defaultdict(float)
    sectors: dict[str, float] = defaultdict(float)
    countries: dict[str, float] = defaultdict(float)
    unknown = 0.0

    def descend(sym: str, item: SecurityMetadata, amount: float) -> None:
        nonlocal unknown
        snap = snapshots.get(sym)
        if snap is not None:
            used = 0.0
            for child in snap.holdings:
                used += child.fraction
                known = classifications.get(child.symbol)
                descend(
                    child.symbol,
                    SecurityMetadata(
                        child.symbol,
                        child.asset_class
                        if child.asset_class != UNKNOWN
                        else (known.asset_class if known else UNKNOWN),
                        child.sector
                        if child.sector not in ("", UNKNOWN)
                        else (known.sector if known else UNKNOWN),
                        child.country
                        if child.country not in ("", UNKNOWN)
                        else (known.country if known else UNKNOWN),
                    ),
                    amount * child.fraction,
                )
            remainder = max(0.0, 1 - used)
            unknown += amount * remainder
            types[UNKNOWN] += amount * remainder
            sectors[UNKNOWN] += amount * remainder
            countries[UNKNOWN] += amount * remainder
            return
        if item.asset_class == "etf":
            # Missing underlying holdings for a nested ETF are NOT known shares.
            unknown += amount
            types[UNKNOWN] += amount
            sectors[UNKNOWN] += amount
            countries[UNKNOWN] += amount
            return
        leaf[sym] += amount
        # Security overlap is only defined here for verified equities. Unknown
        # tickers, bonds, metal units and derivatives must not enter the
        # common-STOCK overlap denominator.
        if item.asset_class == "equity" and sym not in {"CASH", "USD", "CAD"}:
            comparable_leaf[sym] += amount
        types[item.asset_class] += amount
        sectors[item.sector if item.asset_class == "equity" else UNKNOWN] += amount
        countries[item.country if item.asset_class == "equity" else UNKNOWN] += amount

    descend(symbol, meta, 1.0)
    return dict(leaf), dict(comparable_leaf), dict(types), dict(sectors), dict(countries), unknown


def analyze_exposure(
    valuation: PortfolioValuation,
    *,
    snapshots: Mapping[str, FundSnapshot] | None = None,
    classifications: Mapping[str, SecurityMetadata] | None = None,
    analysis_date: date | None = None,
    max_snapshot_age_days: int = 120,
    demo_data: bool = False,
    fund_aggregates: Mapping[str, tuple[Mapping[str, float], Mapping[str, float]]] | None = None,
) -> ExposureReport:
    """Compute portfolio-level lower-bound overlap and exposure coverage.

    Inputs are present-day user market values. Weights are normalized through
    V2; snapshot dates are NEVER applied backwards into historical positions.
    """
    if max_snapshot_age_days < 0:
        raise ExposureError("Maximum snapshot age must be nonnegative")
    snapshots = dict(snapshots or {})
    classifications = dict(classifications or {})
    fund_aggregates = dict(fund_aggregates or {})
    validate_snapshots(snapshots)
    if set(fund_aggregates) - set(snapshots):
        raise ExposureError("Fund-level aggregate classification requires a verified fund snapshot")
    cutoff = analysis_date or date.today()
    for sym, snap in snapshots.items():
        if snap.as_of is not None and snap.as_of > cutoff:
            raise ExposureError(f"{sym}: snapshot {snap.as_of} is AFTER analysis cutoff {cutoff}")

    weights = {h.symbol: float(h.weight) for h in valuation.holdings}
    if not weights:
        raise ExposureError("Portfolio must not be empty")
    for sym, meta in classifications.items():
        if meta.symbol != sym or meta.asset_class not in ASSET_CLASSES:
            raise ExposureError(f"Invalid asset classification for {sym}")

    leaf_total: dict[str, float] = defaultdict(float)
    types: dict[str, float] = defaultdict(float)
    sectors: dict[str, float] = defaultdict(float)
    countries: dict[str, float] = defaultdict(float)
    unresolved = 0.0
    expanded_by_fund: dict[str, dict[str, float]] = {}
    coverage: dict[str, dict] = {}
    warnings: list[str] = []

    for sym, weight in weights.items():
        if sym in snapshots:
            meta = SecurityMetadata(sym, "etf")
        else:
            meta = classifications.get(sym, SecurityMetadata(sym))
            if sym not in classifications:
                warnings.append(
                    f"{sym}: no classification or fund snapshot; asset type and geography UNKNOWN"
                )
        leaf, comparable_leaf, by_type, by_sector, by_country, missing = _look_through(
            sym, meta, snapshots, classifications
        )
        if sym in snapshots or meta.asset_class == "etf":
            expanded_by_fund[sym] = comparable_leaf
            snapshot = snapshots.get(sym)
            known = max(0.0, min(1.0, 1.0 - missing))
            coverage[sym] = {
                "portfolio_weight": weight,
                "identified_fraction_of_fund": known,
                "unknown_fraction_of_fund": 1.0 - known,
                "source": None if snapshot is None else snapshot.source,
                "as_of": None
                if snapshot is None or snapshot.as_of is None
                else snapshot.as_of.isoformat(),
                "retrieved_on": None
                if snapshot is None or snapshot.retrieved_on is None
                else snapshot.retrieved_on.isoformat(),
            }
            if snapshot is None:
                warnings.append(f"{sym}: no holdings snapshot; underlying exposure unknown")
            elif snapshot.as_of is None:
                warnings.append(
                    f"{sym}: holdings disclosure date unverified; retrieval is not an as-of date"
                )
            elif (cutoff - snapshot.as_of).days > max_snapshot_age_days:
                warnings.append(
                    f"{sym}: fund snapshot is {(cutoff - snapshot.as_of).days} days old"
                )
            if missing > 1e-9:
                warnings.append(f"{sym}: {missing:.1%} of underlying exposure unresolved")

        for name, amt in leaf.items():
            leaf_total[name] += weight * amt
        for name, amt in by_type.items():
            types[name] += weight * amt
        # The issuer's fund-level sector/country breakdown is a separate source of
        # metadata. It is not an assignment to individual constituent securities.
        # Use it only for explicitly validated fund snapshots, not unknown funds.
        if sym in fund_aggregates:
            from portfolio_risk.exposure_classification import canonical_country, canonical_sector

            groups = fund_aggregates[sym]

            def apply_group(raw, existing, destination, canon, label, weight, sym):
                if not raw:
                    for name, amt in existing.items():
                        destination[name] += weight * amt
                    return
                values = {}
                for name, value in raw.items():
                    if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                        raise ExposureError(f"{sym}: invalid {label} percentage")
                    key = canon(name)
                    values[key] = values.get(key, 0.0) + float(value) / 100.0
                total = sum(values.values())
                if total > 1.001:
                    warnings.append(
                        f"{sym}: {label} aggregate exceeds 100%; using constituent classification instead"
                    )
                    for name, amt in existing.items():
                        destination[name] += weight * amt
                    return
                for name, amount in values.items():
                    destination[name] += weight * amount
                destination[UNKNOWN] += weight * max(0.0, 1.0 - total)

            apply_group(groups[0], by_sector, sectors, canonical_sector, "sector", weight, sym)
            apply_group(groups[1], by_country, countries, canonical_country, "country", weight, sym)
            if sym in coverage:
                coverage[sym]["sector_country_source"] = (
                    "TickerInside fund-level aggregate (not per-security labels)"
                )
        else:
            for name, amt in by_sector.items():
                sectors[name] += weight * amt
            for name, amt in by_country.items():
                countries[name] += weight * amt
        unresolved += weight * missing

    # If an asset has no metadata, it is a KNOWN direct ticker but its sector,
    # country and asset class are UNKNOWN. It isn't missing underlying shares.
    pairwise = []
    names = sorted(expanded_by_fund)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = set(expanded_by_fund[a]) & set(expanded_by_fund[b])
            min_shared = sum(min(expanded_by_fund[a][s], expanded_by_fund[b][s]) for s in common)
            pairwise.append(
                {
                    "fund_a": a,
                    "fund_b": b,
                    "identified_overlap_lower_bound": min_shared,
                    "common_identified_symbols": sorted(common),
                    "unresolved_a": coverage[a]["unknown_fraction_of_fund"],
                    "unresolved_b": coverage[b]["unknown_fraction_of_fund"],
                }
            )
    pairwise.sort(key=lambda x: (-x["identified_overlap_lower_bound"], x["fund_a"], x["fund_b"]))
    dates = {sym: snap.as_of for sym, snap in snapshots.items() if sym in expanded_by_fund}
    if len({d for d in dates.values() if d is not None}) > 1:
        warnings.append(
            "Fund disclosures have different as-of dates; pairwise overlap mixes snapshots"
        )
    if demo_data:
        warnings.insert(0, "SYNTHETIC DEMO holdings: not actual ETF disclosures or investment data")
    if fund_aggregates:
        warnings.append(
            "ETF sector/country aggregates come from fund-level provider reports, not constituent-level metadata"
        )
        warnings.append(
            "Mixed ETF and direct-stock sector/country classifications may use differing provider taxonomies"
        )
    warnings.append("Country means company domicile/classification, not revenue or FX exposure")
    warnings.append("Exposure reflects supplied snapshots today, not historic fund composition")
    warnings.append(
        "Identified common-EQUITY overlap is a LOWER BOUND; unknown, bond, cash, crypto and commodity positions excluded"
    )
    # Protect sum-to-1 invariants from numerical noise.
    for label, part in (("asset class", types), ("sector", sectors), ("country", countries)):
        if abs(sum(part.values()) - 1.0) > 1e-7:
            raise ExposureError(f"{label} exposure fractions do not sum to 100%")
    if abs(sum(leaf_total.values()) + unresolved - 1.0) > 1e-7:
        raise ExposureError("Underlying security coverage doesn't reconcile to total portfolio")

    def clean(d: Mapping[str, float]) -> dict[str, float]:
        return dict(
            sorted(
                ((s, max(0.0, v)) for s, v in d.items() if v > 1e-13), key=lambda x: (-x[1], x[0])
            )
        )

    return ExposureReport(
        base_currency=valuation.base_currency,
        total_value=str(valuation.total_value),
        analysis_date=cutoff,
        direct_weights=clean(weights),
        snapshot_fx=(
            None
            if valuation.fx_quote is None
            else {
                "rate_cad_per_usd": str(valuation.fx_quote.usd_cad),
                "observed_on": valuation.fx_quote.observed_on.isoformat(),
                "requested_cutoff": valuation.fx_quote.as_of.isoformat(),
                "source": valuation.fx_quote.source,
            }
        ),
        identified_underlying=clean(leaf_total),
        by_asset_class=clean(types),
        by_sector=clean(sectors),
        by_country=clean(countries),
        unknown_underlying=unresolved,
        coverage_by_fund=coverage,
        pairwise_overlap=tuple(pairwise),
        warnings=tuple(warnings),
    )


def exposure_as_dict(report: ExposureReport) -> dict:
    return {
        "base_currency": report.base_currency,
        "total_value": report.total_value,
        "analysis_date": report.analysis_date.isoformat(),
        "direct_weights": report.direct_weights,
        "snapshot_fx": report.snapshot_fx,
        "identified_underlying": report.identified_underlying,
        "by_asset_class": report.by_asset_class,
        "by_sector": report.by_sector,
        "by_country": report.by_country,
        "unknown_underlying": report.unknown_underlying,
        "coverage_by_fund": report.coverage_by_fund,
        "pairwise_overlap": list(report.pairwise_overlap),
        "warnings": list(report.warnings),
        "limitations": [
            "Only common VERIFIED equity positions (excluding unknown, bond, cash, crypto and commodities) are compared; overlap is a LOWER BOUND.",
            "Current fund holdings are not historical portfolio constituents or a trading backtest.",
            "Country is security domicile/classification, not revenue mix or currency sensitivity.",
            "Sector/country UNKNOWN explicitly includes missing metadata, non-equities and unsupplied holdings.",
            "ETF aggregate sectors/countries are issuer-reported fund-level allocations, not individual stock classifications.",
        ],
    }
