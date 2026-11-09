"""V9.12: honest, human-readable portfolio trade-offs with no recalculation.

All historical numbers come from the V8.4 robustness report. In particular,
latest-fold changes are not projected future returns or a live recommendation.
"""
from __future__ import annotations

import math

from .v911_presentation import latest_candidate_example
from .v910_presentation import all_goal_review
from .v98_presentation import STRATEGY_GOALS


STRATEGY_OPTION_LABELS = {
    "Balance growth and risk": "Balanced growth and risk",
    "More annual growth": "Seek higher growth, accept more risk",
    "Smoother returns": "Aim for smaller ups and downs",
    "Smaller losses": "Aim for smaller market downturns",
    "Better downside-adjusted returns": "Favor return relative to downside risk",
    "Reduce losses on bad days": "Reduce losses on bad days",
}


def strategy_option_label(goal: str) -> str:
    return STRATEGY_OPTION_LABELS.get(goal, goal)


def strategy_goal_name(priorities: str | None) -> str:
    for name, (parameters, _) in STRATEGY_GOALS.items():
        if parameters == priorities:
            return name
    return "Balance growth and risk" if priorities is None else "Custom focus"


def income_ranked(holdings: list[dict]) -> list[dict]:
    """Rank by ACTUAL estimated annual distribution dollars, not yield or name."""
    def amount(row: dict) -> float:
        value = row.get("estimated_annual_gross_income")
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
        return -math.inf
    return sorted(holdings, key=lambda row: (-amount(row), str(row.get("symbol", ""))))


def _metric_change(example: dict, name: str) -> dict | None:
    row = next((r for r in example.get("measures", []) if r.get("Key") == name), None)
    if row is None:
        return None
    return {
        "label": row["Metric"], "key": name,
        "original": row["Your portfolio"], "tested": row["Tested allocation"],
        "result": row["Result"],
        "delta_pp": (row["Tested allocation"] - row["Your portfolio"]) * 100,
    }


def strategy_tradeoff(report: dict) -> dict:
    """Explain a *single* out-of-sample candidate and the separate all-fold evidence.

    Keep all-fold consistency and single-fold values separate, and never call an
    unapproved candidate a recommendation or claim expected future returns.
    """
    example = latest_candidate_example(report)
    review = all_goal_review(report)
    if example.get("status") != "evaluated":
        return {"status": "unavailable", "reason": example.get("reason"), "evidence": []}
    selected = [_metric_change(example, k) for k in ("cagr", "volatility", "drawdown", "sharpe", "sortino", "es95")]
    metrics = [r for r in selected if r is not None]
    by_metric = {r["key"]: r for r in metrics}
    positives = [r["label"] for r in metrics if r["result"] == "Better"]
    negatives = [r["label"] for r in metrics if r["result"] == "Worse"]
    similarities = [r["label"] for r in metrics if r["result"] == "Similar"]
    # Make this interpretation quantitative: never infer gain/loss from the
    # optimization focus or an unverified prediction.
    consistency = []
    for metric_key in ("cagr", "volatility", "drawdown", "sharpe", "sortino", "es95"):
        measure = by_metric.get(metric_key)
        if measure:
            row = next((x for x in review.get("rows", []) if x["Metric"] == metric_key), None)
            if row:
                consistency.append({"label": measure["label"], "key": metric_key,
                                    "better": row["Better"], "similar": row["Similar"], "worse": row["Worse"],
                                    "total": review["count"]})
    growth, volatility = by_metric.get("cagr"), by_metric.get("volatility")
    interpretation = None
    if growth and volatility:
        growth_state, vol_state = growth["result"], volatility["result"]
        if growth_state == "Worse" and vol_state == "Better":
            interpretation = ("This tested mix reduced fluctuations but sacrificed annualized growth "
                              "in this historical window. Lower risk did not mean higher returns.")
        elif growth_state == "Better" and vol_state == "Worse":
            interpretation = ("This tested mix gained annualized growth but had larger fluctuations "
                              "in this historical window. More return came with more risk.")
        elif growth_state == "Better" and vol_state == "Better":
            interpretation = ("This tested mix had higher annualized growth and smaller fluctuations "
                              "in this historical window, though this did not happen in every test.")
        elif growth_state == "Worse" and vol_state == "Worse":
            interpretation = ("This tested mix had lower annualized growth and larger fluctuations "
                              "in this historical window.")
    return {"status": "evaluated", "latest_end": example["holdout_end"],
            "interpretation": interpretation,
            "selected_label": example["selected_label"], "measures": metrics,
            "positives": positives, "negatives": negatives, "similarities": similarities,
            "all_fold_counts": consistency, "fold_count": review.get("count", 0),
            "passed": review.get("passed") is True}


def status_for_summary(report: dict) -> str:
    review = all_goal_review(report)
    if review["status"] != "evaluated":
        return "Not enough historical evidence to assess this approach."
    return ("The selection method passed its historical consistency rules. This does not "
            "guarantee future results or identify one permanently optimal allocation."
            if review["passed"] else
            "The selection method did not pass all required historical checks. "
            "Some risk measures may improve, but the evidence does not support an overall allocation recommendation.")
