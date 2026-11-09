"""V9.8 presentation and validation explanations, without changing V8.4 calculations."""
from __future__ import annotations

import math

from .core import PortfolioError, normalized_symbol


STRATEGY_GOALS = {
    "Balance growth and risk": ("sharpe:2,volatility:1,drawdown:1",
        "Favor a higher historical return for the amount of fluctuation taken."),
    "More annual growth": ("cagr:3,sharpe:1",
        "Prioritize annual growth, even if the resulting mix could fluctuate more. This does not guarantee higher returns."),
    "Smoother returns": ("volatility:3,sharpe:1,drawdown:1",
        "Put more weight on lower day-to-day price fluctuations."),
    "Smaller losses": ("drawdown:3,sharpe:1,volatility:1",
        "Put more weight on limiting the largest historical peak-to-trough fall."),
    "Better downside-adjusted returns": ("sortino:3,sharpe:1",
        "Prioritize returns relative to downside fluctuations, rather than all volatility."),
    "Reduce losses on bad days": ("es95:3,drawdown:1",
        "Prioritize smaller average losses across the worst 5% of historical trading days."),
}


def scenario_allocation_check(rows: list[dict]) -> tuple[bool, float]:
    """Exact displayed scenario totals; backend keeps the authoritative strict validation."""
    amounts = [float(row.get("Target weight (%)") or 0) for row in rows]
    if any(not math.isfinite(n) or n < 0 or n > 100 for n in amounts):
        return False, sum(amounts)
    total = sum(amounts)
    return abs(total - 100.0) <= 0.0001, total


def format_group_cap(name: str, members: list[str], percent: float, allowed: list[str]) -> str:
    """Generate the validated V8.4 syntax from a user-selected group, not a predefined category."""
    label = name.strip().replace(" ", "_")
    if not label or not label.replace("_", "").isalnum():
        raise PortfolioError("Give your group a short name using letters, numbers or underscores")
    allowed_set = {normalized_symbol(x) for x in allowed}
    selected = [normalized_symbol(x) for x in members]
    if not selected or len(set(selected)) != len(selected) or any(x not in allowed_set for x in selected):
        raise PortfolioError("Choose one or more distinct holdings from your portfolio")
    if not isinstance(percent, (int, float)) or not math.isfinite(float(percent)) or not 0 <= percent <= 100:
        raise PortfolioError("The maximum group allocation must be between 0% and 100%")
    return f"{label}:{','.join(selected)}={float(percent):g}"


def market_valuation_gap(price: object, estimated: object) -> dict | None:
    """Compare two backend-provided prices; never interpret the difference as a forecast."""
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (price, estimated)):
        return None
    if price <= 0:
        return None
    percent = (float(estimated) / float(price) - 1.0) * 100.0
    return {"percent": percent, "direction": "above" if percent > 0 else "below",
            "large": abs(percent) >= 30.0}


def holdout_chart_rows(report: dict) -> list[dict]:
    """Display counts supplied by V8.4, with no modeled or fabricated trials."""
    rob = report.get("rolling_robustness") or {}
    tally = rob.get("per_metric_tally") or {}
    names = {
        "sharpe": "Return per unit of risk", "volatility": "Smoother returns",
        "drawdown": "Smaller peak-to-trough fall", "cagr": "Annual growth",
        "sortino": "Return relative to downside risk", "es95": "Losses on worst days",
    }
    rows = []
    for key in (rob.get("priorities") or tally):
        item = tally.get(key)
        if not isinstance(item, dict):
            continue
        rows.append({"Goal": names.get(key, key.replace("_", " ").capitalize()),
                     "Metric": key,
                     "Better": int(item.get("positive") or 0),
                     "Similar": int(item.get("neutral") or 0),
                     "Worse": int(item.get("negative") or 0),
                     "Not assessed": int(item.get("unavailable") or 0)})
    return rows


def holdout_takeaway(report: dict) -> str:
    """Explain the observed evidence and exact V8.4 rule, without changing its verdict."""
    rob = report.get("rolling_robustness") or {}
    rows = holdout_chart_rows(report)
    if not rows or rob.get("status") != "evaluated":
        return "Not enough comparable historical tests were available for a conclusion."
    count = int(rob.get("fold_count") or 0)
    if rob.get("consistent_improvement_on_all_priorities") is True:
        return "Every selected goal passed the historical consistency rule. Review the allocations, costs and risks before drawing conclusions."
    # Name the relevant blocker rather than repeating generic notices.
    blockers = [r for r in rows if r["Worse"] > 0 or r["Better"] < math.ceil(0.75 * count) or r["Not assessed"]]
    if not blockers:
        return "The strategy did not satisfy all validation requirements; inspect the individual tests below."
    worst = max(blockers, key=lambda x: (x["Worse"], -x["Better"]))
    return (f"The biggest obstacle was {worst['Goal'].lower()}: the new mix improved in "
            f"{worst['Better']} of {count} tests but worsened in {worst['Worse']}. "
            "To recommend an improvement, each selected goal must improve in at least 75% "
            "of tests, with zero tests showing a meaningful worsening. This standard was not met.")
