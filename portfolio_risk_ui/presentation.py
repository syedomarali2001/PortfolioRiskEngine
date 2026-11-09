"""Read-only presentation/validation helpers for V8.4 JSON contracts.

No portfolio return, dividend, DCF or optimization mathematics is calculated here.
The small transformations only prepare *existing* engine outputs for display.
"""
from __future__ import annotations

import math

from .core import PortfolioError, normalized_symbol


def inspect_risk_contribution(report: dict) -> list[dict]:
    """Map the V8.4 'risk_contribution' output to chartable rows."""
    items = report.get("risk_contribution") or []
    out = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("share_of_total_risk"), (int, float)):
            continue
        out.append({
            "Ticker": item.get("symbol", "Unknown"),
            "Allocation (%)": 100 * item.get("weight", 0),
            "Share of volatility risk (%)": 100 * item["share_of_total_risk"],
        })
    return sorted(out, key=lambda r: -r["Share of volatility risk (%)"])


def read_exposure_breakdowns(report: dict) -> dict[str, dict]:
    """Actual V8.4 JSON keys: by_sector, by_asset_class and by_country."""
    return {label: report.get(key) or {} for label, key in (
        ("Sectors", "by_sector"),
        ("Asset classes", "by_asset_class"),
        ("Countries (company domicile)", "by_country"),
    )}


def holding_support_rows(report: dict) -> list[dict]:
    out = []
    for ticker, x in (report.get("asset_support") or {}).items():
        if not isinstance(x, dict):
            continue
        out.append({"Ticker": ticker,
                    "Constituent coverage": x.get("exposure_status", "unknown"),
                    "Reason": x.get("reason", "")})
    return out


def overlap_rows(report: dict) -> list[dict]:
    values = (report.get("drilldown") or {}).get("overlap_contributors") or []
    return [{"Fund A": x.get("fund_a"), "Fund B": x.get("fund_b"),
             "Identified common equity overlap (min %)": 100*x.get("overlap_lower_bound", 0),
             "Identified overlapping companies": x.get("shared_stock_count", 0)}
            for x in values if isinstance(x, dict)]


def what_if_flag(weights: dict[str, float]) -> str:
    """Safely build an explicit V8.4 --scenario parameter, without renormalizing."""
    if not weights:
        raise PortfolioError("A what-if scenario must include all your portfolio tickers")
    clean = {}
    for ticker, v in weights.items():
        symbol = normalized_symbol(ticker)
        try:
            val = float(v)
        except (TypeError, ValueError) as exc:
            raise PortfolioError(f"Invalid scenario weight for {symbol}") from exc
        if not math.isfinite(val) or val < 0 or val > 100:
            raise PortfolioError(f"{symbol}: what-if weight must be between 0% and 100%")
        if symbol in clean:
            raise PortfolioError(f"Duplicate what-if ticker {symbol}")
        clean[symbol] = val
    if abs(sum(clean.values()) - 100.0) > 0.0001:
        raise PortfolioError(f"What-if weights must total 100% (currently {sum(clean.values()):.4f}%)")
    return "GUI_WhatIf:" + ",".join(f"{s}={n:.8f}" for s, n in clean.items())

# Metric-specific presentation only. Financial values and classifications
# originate in the validated V8.4 reports; no result is recalculated here.
_PERCENT_METRICS = {
    "cagr", "annualized_volatility", "volatility", "drawdown", "es95",
    "var95", "tracking_error", "tracking_error_annualized",
    "cumulative_return", "sortino_downside_deviation", "active_return",
}
_FRIENDLY_METRICS = {
    "cagr": "Annual growth (CAGR)",
    "volatility": "Volatility", "annualized_volatility": "Volatility",
    "drawdown": "Maximum drawdown", "es95": "Expected shortfall (95%)",
    "sharpe": "Sharpe ratio", "sortino": "Sortino ratio",
    "tracking_error_annualized": "Tracking error",
}


def format_metric_for_people(metric: str, value: object) -> str:
    """Convert return fractions to %; leave dimensionless ratios as numbers."""
    if value is None:
        return "Unavailable"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "Unavailable"
    except (TypeError, ValueError, OverflowError):
        return "Unavailable"
    if metric.lower() in _PERCENT_METRICS:
        return f"{100 * number:.2f}%"
    return f"{number:.2f}"


def finding_display(item: dict) -> dict:
    """Return a readable, faithful version of a V8.4 diagnostic item."""
    metric = str(item.get("metric") or "Metric").lower()
    label = _FRIENDLY_METRICS.get(metric, metric.replace("_", " ").title())
    benchmark = str(item.get("benchmark") or "benchmark")
    ours = format_metric_for_people(metric, item.get("portfolio_value"))
    other = format_metric_for_people(metric, item.get("benchmark_value"))
    threshold = format_metric_for_people(metric, item.get("materiality_threshold"))
    diff = item.get("favorable_difference")
    is_percent = metric in _PERCENT_METRICS
    try:
        difference = abs(float(diff))
    except (ValueError, TypeError):
        difference = None
    if difference is not None and not math.isfinite(difference):
        difference = None
    if difference is None:
        comparison = f"Portfolio: {ours} · {benchmark}: {other}."
    elif is_percent:
        comparison = (f"Portfolio: {ours} · {benchmark}: {other}. "
                      f"Difference: {difference * 100:.2f} percentage points "
                      f"(materiality threshold: {threshold.replace('%', '')} percentage points).")
    else:
        comparison = (f"Portfolio: {ours} · {benchmark}: {other}. "
                      f"Difference: {difference:.2f} "
                      f"(materiality threshold: {threshold}).")
    sentiment = str(item.get("sentiment") or "unavailable").upper()
    return {"title": f"{sentiment}: {label} vs {benchmark}",
            "comparison": comparison,
            "explanation": str(item.get("explanation") or ""),
            "metric": metric,
            "sentiment": sentiment}


def allocation_pie_rows(weights: dict, *, max_slices: int = 9) -> list[dict]:
    """Compress only the visualization; preserve every original weight in sum."""
    if max_slices < 2:
        raise ValueError("max_slices must be at least 2")
    good = []
    for symbol, weight in weights.items():
        try:
            number = float(weight)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            good.append((str(symbol), number))
    good.sort(key=lambda x: (-x[1], x[0]))
    if len(good) > max_slices:
        head, tail = good[:max_slices-1], good[max_slices-1:]
        good = head + [(f"Other holdings ({len(tail)})", sum(v for _,v in tail))]
    return [{"Holding": s, "Weight (%)": w*100} for s,w in good]


def overview_notes(warnings: list) -> tuple[list[str], list[str]]:
    """Split expected research caveats from nonstandard provider/quality notices."""
    known = (
        "today's portfolio allocations applied retrospectively",
        "spy/rsp are etf proxies",
        "positive and negative alerts indicate",
        "joint calendar excludes non-shared closes",
        "selecting among backtests can overfit",
    )
    methodology, quality = [], []
    for item in warnings:
        message = str(item.get("message") or item) if isinstance(item, dict) else str(item)
        if message.lower().startswith(known):
            methodology.append(message)
        else:
            quality.append(message)
    return methodology, quality
