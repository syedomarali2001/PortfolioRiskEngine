"""Fuse three *observed* V7.2 audits into a descriptive capability matrix.

Never extrapolate from a selected sample to universal market coverage; keep
price, metadata and constituents as different dimensions. No API calls here.
"""

from __future__ import annotations

from datetime import date


def build_capability_report(etfs: dict, stocks: dict, mixed: dict) -> dict:
    for name, value in (("ETF", etfs), ("stock", stocks), ("mixed", mixed)):
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("summary"), dict)
            or not isinstance(value.get("assets"), list)
        ):
            raise ValueError(f"Invalid {name} audit report")
    groups = mixed["summary"].get("by_category", {})
    for key in ("US_ETF", "CAD_ETF", "US_STOCK", "CAD_STOCK", "COMMODITY_ETP", "CRYPTO"):
        if key not in groups:
            raise ValueError(f"Mixed audit missing category {key}")
    equity = etfs["summary"]
    stock_by = stocks.get("by_market", {})
    for key in ("US_STOCK", "CAD_STOCK"):
        if key not in stock_by:
            raise ValueError(f"Stock audit missing market {key}")

    def price(m):
        d = groups[m]
        return {"tested": d.get("tickers", 0), "available": d.get("price_available", 0)}

    return {
        "title": "Portfolio Risk Engine V7.2 observed data capability matrix",
        "generated_on": date.today().isoformat(),
        "scope_note": "Curated samples only; rates are not an estimate of entire U.S./Canadian market coverage.",
        "capabilities": {
            "US_ETF": {
                "sampled_prices": price("US_ETF"),
                "funds_audited": equity.get("total", 0),
                "funds_indexed": equity.get("indexed", 0),
                "funds_with_dated_holdings": equity.get("with_dated_holdings", 0),
                "funds_with_95pct_weight": equity.get("near_full_weight_95pct", 0),
                "funds_with_sectors": equity.get("with_sector_breakdown", 0),
                "funds_with_countries": equity.get("with_country_breakdown", 0),
                "funds_with_recent_120d_data": equity.get("current_120d", 0),
                "funds_safe_for_equity_lookthrough": equity.get("equity_lookthrough_safe", 0),
                "note": "Bond, derivative, commodity and crypto-linked ETP constituents are not equities; count separately.",
            },
            "CAD_ETF": {
                "sampled_prices": price("CAD_ETF"),
                "sampled_holdings_available": groups["CAD_ETF"].get("holdings_available", 0),
                "note": "TickerInside is U.S.-focused; Canadian ETF holdings require a separate provider.",
            },
            "US_STOCK": {
                "sampled_prices": price("US_STOCK"),
                "stock_metadata": stock_by["US_STOCK"],
                "note": "Reported company country != geographic revenue or exposure.",
            },
            "CAD_STOCK": {
                "sampled_prices": price("CAD_STOCK"),
                "stock_metadata": stock_by["CAD_STOCK"],
                "note": "Canadian listing does not prove Canadian company domicile.",
            },
            "COMMODITY_ETP": {
                "sampled_prices": price("COMMODITY_ETP"),
                "sampled_with_constituent_data": groups["COMMODITY_ETP"].get(
                    "holdings_available", 0
                ),
                "note": "Physical bullion trusts may be modeled; oil futures and derivatives need distinct exposure logic.",
            },
            "CRYPTO": {
                "sampled_prices": price("CRYPTO"),
                "note": "Crypto coin price history is not ETF holdings, and sector/country labels are generally not applicable.",
            },
        },
        "etf_by_theme": etfs.get("by_theme", {}),
        "stocks_by_group": stocks.get("by_group", {}),
        "limitations": [
            "Price availability checks were short recent windows; historical depth is not demonstrated.",
            "ETF holdings are current-or-last-known snapshots, not historical as-of constituents for backtests.",
            "Provider metadata unavailable may be transient rate limiting, not a permanently unsupported ticker.",
            "A reported holdings date can be several months old even if the request succeeds.",
            "No claim of all-market completeness or full Canadian ETF look-through is supported by these audits.",
        ],
    }
