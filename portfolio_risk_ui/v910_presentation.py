"""V9.10: six-metric assessment of the SAME forward-selected candidates.

This report is descriptive. Selection and strict overall approval remain owned
by the existing V8.4 engine; the additional outcomes are out-of-sample evidence.
"""
from __future__ import annotations

import math

METRICS = (
    ("cagr", "Annual growth"),
    ("sharpe", "Return per unit of risk"),
    ("sortino", "Return relative to downside risk"),
    ("volatility", "Smoother returns"),
    ("drawdown", "Smaller worst decline"),
    ("es95", "Smaller losses on bad days"),
)


def all_goal_review(report: dict) -> dict:
    """Read backend-supplied tallies, never claim independent optimization runs."""
    rob = report.get("rolling_robustness") or {}
    count = int(rob.get("fold_count") or 0)
    if rob.get("status") != "evaluated" or count <= 0:
        return {"status": "unavailable", "count": count, "rows": [], "findings": [],
                "strong": [], "passed": False}
    tally = rob.get("all_metric_tally") or rob.get("per_metric_tally") or {}
    rows = []
    findings = []
    strong = []
    for metric, goal in METRICS:
        item = tally.get(metric)
        if not isinstance(item, dict):
            continue  # older reports have only priorities, do not invent results
        better = int(item.get("positive") or 0)
        similar = int(item.get("neutral") or 0)
        worse = int(item.get("negative") or 0)
        missing = int(item.get("unavailable") or 0)
        if any(n < 0 for n in (better, similar, worse, missing)) or better + similar + worse + missing != count:
            continue
        passed = better >= math.ceil(0.75 * count) and worse == 0 and missing == 0
        if passed:
            status = "Consistent historical improvement"
            strong.append(goal)
        elif better >= math.ceil(0.75 * count) and worse:
            status = "Often better, but some worse periods"
        elif better > worse and better > 0:
            status = "Improved in some periods"
        elif worse > better:
            status = "Worsened more often than improved"
        else:
            status = "No clear advantage"
        detail = (f"Better in {better} of {count} tests, similar in {similar}, "
                  f"worse in {worse}." +
                  (f" {missing} could not be evaluated." if missing else ""))
        rows.append({"Goal": goal, "Metric": metric, "Better": better,
                     "Similar": similar, "Worse": worse, "Not assessed": missing})
        findings.append({"goal": goal, "summary": status + ".", "detail": detail,
                         "passed_independent_check": passed})
    return {"status": "evaluated" if rows else "unavailable", "count": count,
            "rows": rows, "findings": findings, "strong": strong,
            "passed": rob.get("consistent_improvement_on_all_priorities") is True}


def tested_candidate_warning(report: dict) -> str | None:
    """Explain what the default allocation cap means for concentrated portfolios."""
    rob = report.get("rolling_robustness") or {}
    rules = rob.get("constraints") or {}
    cap = rules.get("max_weight")
    weights = report.get("input_weights") or {}
    if not isinstance(weights, dict) or not isinstance(cap, (float, int)):
        return None
    try:
        symbol, largest = max(((name, float(weight)) for name, weight in weights.items()),
                              key=lambda pair: pair[1])
    except (ValueError, TypeError):
        return None
    if not math.isfinite(largest) or not 0 < cap <= 1 or largest <= cap + 1e-8:
        return None
    return (f"Concentration note: {symbol} is {largest:.2%} of your original portfolio. "
            f"The tested candidate allocations limit a single holding to {cap:.0%} by default. "
            "This helps generate diversified alternatives, but does not ensure that "
            "they will outperform the original in future markets.")
