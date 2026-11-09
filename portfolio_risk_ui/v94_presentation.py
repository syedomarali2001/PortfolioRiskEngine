"""Accessible V9.4 presentation helpers; never change the V8.4 calculations."""
from __future__ import annotations

import math
import re

from .presentation import format_metric_for_people
from .v96_presentation import finding_message, METRIC_NAMES

RISK_METRIC_HELP = {
    "cagr": ("Historical annual growth", "Annualized growth over the analyzed period, if the same portfolio mix had been held throughout. This is not the account's actual return or a forecast."),
    "annualized_volatility": ("Portfolio ups and downs", "Annualized variability of the portfolio's historical returns. Smaller means a smoother historical ride; it does not indicate the maximum possible loss."),
    "sharpe": ("Return for total risk taken", "Historical return above the analysis's assumed risk-free rate, divided by return variability. Higher may indicate a better return-to-risk trade-off; depends on the period and risk-free assumption."),
    "drawdown": ("Largest historical decline", "The biggest fall from a previous portfolio peak to a later low during the measured period. Smaller declines are generally preferable."),
    "sortino": ("Return for downside risk", "Historical return relative to downside fluctuations only. Higher generally means more return per unit of downside variation."),
    "calmar": ("Growth versus worst decline", "Historical annual growth divided by the largest peak-to-trough decline. Higher means stronger growth relative to the worst observed decline."),
    "es95": ("Average of worst daily losses", "Average historical loss on the worst 5% of daily returns. This is a daily tail-risk estimate, not an annual loss prediction."),
    "beta": ("Market sensitivity", "How closely your portfolio's returns have moved with SPY. Roughly 1 means similar sensitivity to SPY; below or above 1 means smaller or larger market-linked moves, not overall risk."),
}

FRIENDLY_METRIC = {
    "cagr": "Annual growth", "volatility": "Portfolio fluctuations",
    "annualized_volatility": "Portfolio fluctuations", "drawdown": "Largest historical decline",
    "sharpe": "Return relative to total risk", "sortino": "Return relative to downside risk",
    "es95": "Loss on the worst days", "calmar": "Growth relative to worst decline",
}

_PERCENT = {"cagr", "volatility", "annualized_volatility", "drawdown", "es95", "var95"}


def finding_for_people(item: dict) -> dict:
    """Translate a preclassified V8.4 finding. Never override its sentiment."""
    metric = str(item.get("metric") or "").lower()
    status = str(item.get("sentiment") or "unavailable").lower()
    benchmark = str(item.get("benchmark") or "the comparison")
    label = METRIC_NAMES.get(metric, metric.replace("_", " ").title())
    ours = format_metric_for_people(metric, item.get("portfolio_value"))
    other = format_metric_for_people(metric, item.get("benchmark_value"))
    numerical_diff = item.get("favorable_difference")
    threshold = item.get("materiality_threshold")
    diff = None
    threshold_num = None
    try:
        if numerical_diff is not None and math.isfinite(float(numerical_diff)):
            diff = float(numerical_diff)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        if threshold is not None and math.isfinite(float(threshold)):
            threshold_num = abs(float(threshold))
    except (TypeError, ValueError, OverflowError):
        pass
    if diff is None:
        change = None
    elif metric in _PERCENT:
        change = f"{abs(diff)*100:.2f} percentage points"
    else:
        change = f"{abs(diff):.2f}"
    # V8.4 supplies sentiment; the title always names the metric and benchmark.
    head = f"{label} compared with {benchmark}"
    reason = finding_message(metric, status, change)
    if threshold_num is None:
        rule = "A comparison cutoff was not provided."
    elif metric in _PERCENT:
        rule = f"A difference of at least {threshold_num*100:.2f} percentage points to earn a positive or negative label."
    else:
        rule = f"A difference of at least {threshold_num:.2f} to earn a positive or negative label."
    return {"status": status, "headline": head, "comparison": f"Your portfolio: {ours} · {benchmark}: {other}",
            "difference": change, "reason": reason, "rule": rule,
            "explanation": str(item.get("explanation") or ""), "metric": metric,
            "benchmark": benchmark,
            "status_label": {"positive":"Strength", "negative":"Area to watch",
                             "neutral":"Similar result", "unavailable":"Not assessed"}.get(status,"Not assessed")}


def select_overview_findings(diagnostics: dict, maximum: int = 4) -> list[dict]:
    """Select diverse highlights rather than four versions of the same metric."""
    groups = {key: [x for x in diagnostics.get(key, []) if isinstance(x, dict)]
              for key in ("negative", "positive", "neutral")}
    chosen = []
    seen = set()
    # A concern should never be silently crowded out by repeated good news.
    for key in ("negative", "positive", "neutral", "positive"):
        for item in groups[key]:
            metric = str(item.get("metric") or "")
            if metric not in seen and len(chosen) < maximum:
                chosen.append(item)
                seen.add(metric)
                break
    for key in ("negative", "positive", "neutral"):
        for item in groups[key]:
            if len(chosen) == maximum:
                return chosen
            if item not in chosen:
                chosen.append(item)
    return chosen


def benchmark_status(report: dict, metric: str, benchmark: str = "SPY") -> str | None:
    if metric == "annualized_volatility":
        metric = "volatility"
    for group in ("negative", "positive", "neutral"):
        for item in (report.get("diagnostics") or {}).get(group, []):
            if item.get("metric") == metric and item.get("benchmark") == benchmark:
                return group
    return None


def nice_category(label: str, kind: str = "Sectors") -> str:
    text = str(label).strip()
    key = text.upper()
    if key == "UNKNOWN":
        return "Not classified / not applicable"
    if kind == "Countries (company domicile)":
        return {"US":"United States", "USA":"United States", "CA":"Canada", "GB":"United Kingdom"}.get(key, text.replace("_", " ").title())
    if kind == "Asset classes":
        return {"equity":"Stocks / equities", "commodity":"Commodities", "crypto":"Cryptocurrency"}.get(text.lower(), text.replace("_", " ").title())
    return text.replace("_", " ").title()


def composition_rows(weights: dict, kind: str) -> tuple[list[dict], float]:
    """Keep unknown category separate, not hidden or redistributed."""
    known, unknown = [], 0.0
    for name, fraction in (weights or {}).items():
        if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or fraction <= 0:
            continue
        if str(name).upper() == "UNKNOWN":
            unknown += 100 * fraction
        else:
            known.append({"Category": nice_category(name, kind), "Share of portfolio (%)": 100 * fraction})
    known.sort(key=lambda item: (-item["Share of portfolio (%)"], item["Category"]))
    return known, unknown


def support_for_people(rows: list[dict]) -> list[dict]:
    labels = {"supported":"Detailed coverage", "partial":"Partial coverage", "price_only":"Price history only",
              "unverified":"Not verified", "unknown":"Unknown"}
    return [{"Investment": r["Ticker"], "Available information": labels.get(str(r.get("Constituent coverage") or "").lower(), "Not verified"),
             "Reason": r.get("Reason") or "No further details supplied"} for r in rows]


def snapshot_rows(funds: dict) -> list[dict]:
    rows = []
    for name, data in (funds or {}).items():
        if not isinstance(data, dict):
            continue
        frac = data.get("identified_fraction")
        count = data.get("all_positions")
        rows.append({"Fund": name, "Holdings reported as of": data.get("snapshot_as_of") or "Not supplied",
                     "Identified holdings (%)": round(100 * frac, 2) if isinstance(frac, (int, float)) else None,
                     "Holdings listed": count if isinstance(count, int) else None,
                     "Data source": str(data.get("source") or "Not supplied")})
    return rows


def exposure_notes(warnings: list) -> tuple[list[str], list[str]]:
    """Separate routine disclosure notes from data-quality warnings (without dropping any)."""
    routine = []
    caution = []
    indicators = ("snapshot", "fund disclosures", "issuer sector", "taxonomies", "country means",
                  "lower bound", "overlap", "unknown, bond", "current", "historical", "today", "dated")
    for item in warnings or []:
        text = str(item.get("message") or item) if isinstance(item, dict) else str(item)
        if any(token in text.lower() for token in indicators):
            routine.append(text)
        else:
            caution.append(text)
    return routine, caution


def simplify_insights_warnings(warnings: list) -> tuple[list[str], list[str]]:
    """Summarize expected longest-history notices; leave all raw notices available."""
    messages = [str(w.get("message") or w) if isinstance(w, dict) else str(w) for w in warnings or []]
    short = [w for w in messages if "SHORT_HISTORY" in w]
    calendar = [w for w in messages if "CALENDAR_ALIGNMENT" in w]
    familiar = [w for w in messages if w.lower().startswith((
        "today's portfolio allocations", "spy/rsp are etf proxies", "positive and negative alerts",
        "joint calendar excludes", "selecting among backtests"))]
    other = [w for w in messages if w not in short + calendar + familiar]
    summary = []
    if short:
        summary.append("Some investments started trading later than others. The analysis starts when all selected investments have comparable prices.")
    if calendar:
        summary.append("Only trading dates shared by every holding and benchmark were included. Missing prices were not invented.")
    if familiar:
        summary.append("Results show how today's allocation would have performed historically: not your actual past trades or future returns.")
    return summary, other
