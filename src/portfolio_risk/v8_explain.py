"""V8 holdings/sector/overlap provenance drill-down (current dated snapshots only)."""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Mapping

from portfolio_risk.exposure import ExposureReport, FundSnapshot, SecurityMetadata, UNKNOWN
from portfolio_risk.exposure_classification import canonical_sector


def explain_exposure(
    report: ExposureReport, *, snapshots: Mapping[str, FundSnapshot],
    classifications: Mapping[str, SecurityMetadata] | None = None,
    fund_aggregates: Mapping[str, tuple[Mapping[str, float], Mapping[str, float]]] | None = None,
    top_n: int = 25,
) -> dict:
    """Do not assign an issuer's fund-level sector allocation to particular names.

    Companies are classified only when direct per-stock metadata exists. Sector
    allocation BY INVESTMENT can use issuer fund-level totals, but is labeled as
    such and never represented as verified security-level allocation.
    """
    if not 1 <= top_n <= 100:
        raise ValueError("top_n must be 1..100")
    classes = dict(classifications or {})
    aggregates = dict(fund_aggregates or {})
    weights = report.direct_weights
    fund_details = {}
    by_sector_fund: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    known_companies: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_fund_stock = {}
    for sym, w in weights.items():
        if sym in snapshots:
            snap = snapshots[sym]
            holdings = sorted(snap.holdings, key=lambda x: (-x.fraction, x.symbol))
            names = []
            eligible = {}
            assigned_sector_pct = 0.
            for item in holdings:
                effective_meta = classes.get(item.symbol)
                sector = (canonical_sector(item.sector) if item.sector not in ("", UNKNOWN)
                          else canonical_sector(effective_meta.sector) if effective_meta and
                          effective_meta.sector not in ("", UNKNOWN) else UNKNOWN)
                if item.asset_class == "equity":
                    eligible[item.symbol] = item.fraction
                    if sector != UNKNOWN:
                        known_companies[sector][item.symbol] += w*item.fraction
                        assigned_sector_pct += item.fraction
                names.append({"symbol":item.symbol, "fund_fraction":item.fraction,
                              "portfolio_fraction":w*item.fraction,
                              "asset_class":item.asset_class, "verified_company_sector":sector})
            per_fund_stock[sym] = eligible
            fund_details[sym] = {"portfolio_fraction": w,
                                 "snapshot_as_of": str(snap.as_of) if snap.as_of else None,
                                 "source": snap.source,
                                 "identified_fraction": report.coverage_by_fund.get(sym,{}).get("identified_fraction_of_fund",0.),
                                 "unidentified_portfolio_fraction":w*report.coverage_by_fund.get(sym,{}).get("unknown_fraction_of_fund",1.),
                                 "top_holdings": names[:top_n],
                                 "all_positions":len(names),
                                 "classified_constituent_fraction":assigned_sector_pct,
                                 "note":"Company-sector field is UNKNOWN unless a constituent classification was actually supplied."}
            if sym in aggregates and aggregates[sym][0]:
                aggregate = aggregates[sym][0]
                total = sum(v for v in aggregate.values() if isinstance(v,(int,float)) and isfinite(v) and v>=0)
                if total <= 100.1:
                    for sector, pct in aggregate.items():
                        if not isfinite(pct) or pct < 0:
                            raise ValueError(f"{sym}: invalid sector aggregate")
                        by_sector_fund[canonical_sector(sector)][sym] += w*pct/100
                    by_sector_fund[UNKNOWN][sym] += w*max(0.,1-total/100)
                else:
                    by_sector_fund[UNKNOWN][sym] += w
            else:
                by_sector_fund[UNKNOWN][sym] += w
        else:
            per_fund_stock[sym] = {}
            meta = classes.get(sym)
            if meta and meta.asset_class == "equity":
                sector = canonical_sector(meta.sector)
                by_sector_fund[sector][sym] += w
                if sector != UNKNOWN:
                    known_companies[sector][sym] += w
            else:
                by_sector_fund[UNKNOWN][sym] += w
    # This is the source breakdown of sector *portfolio* exposure. It is
    # intentionally separate from per-company known-sector attribution, which
    # may be much smaller than fund-level issuer sector allocations.
    sector_view = {}
    for sector, fund_map in sorted(by_sector_fund.items()):
        sector_view[sector] = {
            "portfolio_fraction": sum(fund_map.values()),
            "by_investment": dict(sorted(fund_map.items(),key=lambda kv:(-kv[1],kv[0]))),
            "known_company_contributions": dict(sorted(known_companies.get(sector,{}).items(),
                                                      key=lambda kv:(-kv[1],kv[0]))[:top_n]),
            "known_company_total_fraction": sum(known_companies.get(sector,{}).values()),
            "unattributed_company_fraction": max(0., sum(fund_map.values()) - sum(known_companies.get(sector,{}).values())),
        }
    overlaps = []
    funds = sorted([s for s in weights if s in snapshots])
    for i, a in enumerate(funds):
        for b in funds[i+1:]:
            shared = set(per_fund_stock[a]) & set(per_fund_stock[b])
            contributors = [{"symbol":s, "fund_a_fraction":per_fund_stock[a][s],
                             "fund_b_fraction":per_fund_stock[b][s],
                             "identified_overlap_contribution": min(per_fund_stock[a][s],per_fund_stock[b][s]),
                             "portfolio_exposure_via_a":weights[a]*per_fund_stock[a][s],
                             "portfolio_exposure_via_b":weights[b]*per_fund_stock[b][s]}
                            for s in shared]
            contributors.sort(key=lambda x:(-x["identified_overlap_contribution"],x["symbol"]))
            overlaps.append({"fund_a":a,"fund_b":b,
                             "overlap_lower_bound":sum(x["identified_overlap_contribution"] for x in contributors),
                             "shared_stock_count":len(contributors),
                             "largest_contributors":contributors[:top_n],
                             "warnings":"Lower bound from different fund snapshots; not look-through of missing funds."})
    overlaps.sort(key=lambda x:(-x["overlap_lower_bound"],x["fund_a"],x["fund_b"]))
    return {"fund_top_holdings":fund_details,
            "sector_drilldown":sector_view,
            "overlap_contributors":overlaps,
            "underlying_top_holdings":list(report.identified_underlying.items())[:top_n],
            "unknown_holdings_by_fund":{s:{"unknown_portfolio_fraction":c["portfolio_weight"]*c["unknown_fraction_of_fund"],
                                           "source":c["source"]}
                                        for s,c in report.coverage_by_fund.items() if c["unknown_fraction_of_fund"]>1e-9},
            "limitations":["Sector-by-investment can use issuer-level sector allocations independent of stock identities.",
                           "Sector-by-stock lists ONLY individually classified companies; unattributed amounts are explicit.",
                           "Two different issuer/ticker classification taxonomies may differ; do not force totals to reconcile.",
                           "Missing ETF constituents are not assumed to have zero economic overlap.",
                           "Stock overlap excludes cash, bonds, commodities, futures and unidentified security types.",
                           "Dated holdings snapshots do not represent the historical composition of a backtest."]}
