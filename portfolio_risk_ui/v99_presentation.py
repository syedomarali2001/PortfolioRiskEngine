"""V9.9: explanatory UI logic for V8.4 validation, no changes to model rules."""
from __future__ import annotations

import math

from .v98_presentation import STRATEGY_GOALS, holdout_chart_rows


def goal_from_priorities(raw: str | None) -> str:
    """Recover the goal used for a saved report; don't use the current unsubmitted picker."""
    for label, (spec, _) in STRATEGY_GOALS.items():
        if spec == raw:
            return label
    return "Selected strategy"


def goal_metric_labels(spec: str) -> list[str]:
    names = {"sharpe": "return relative to overall risk", "volatility": "smoother returns",
             "drawdown": "smaller peak-to-trough declines", "cagr": "annual growth",
             "sortino": "return relative to downside risk", "es95": "losses on the worst trading days"}
    return [names.get(part.split(":", 1)[0], part.split(":", 1)[0])
            for part in spec.split(",") if part.strip()]


def evidence_requirements(count: int) -> int:
    """Keep the same 75%-or-more rule as src/portfolio_risk/v83_robustness.py."""
    return max(1, math.ceil(0.75 * count)) if count > 0 else 0


def metric_decision(row: dict, count: int) -> dict:
    """Explain *both* gates, so 15/20 positive + one negative is not mistaken for a pass."""
    needed = evidence_requirements(count)
    better = int(row["Better"])
    worse = int(row["Worse"])
    unknown = int(row.get("Not assessed", 0))
    enough = better >= needed and count > 0
    clean = worse == 0 and unknown == 0
    passes = enough and clean
    intro = (f"Improved in {better} of {count} tests ({better / count:.0%})" if count else "No completed tests")
    criterion = (f"The minimum of {needed} improvements was met" if enough
                 else f"At least {needed} improvements are required")
    remainder = [criterion]
    if worse:
        remainder.append(f"However, {worse} test{'s' if worse != 1 else ''} showed meaningful worsening, and the rule requires zero")
    if unknown:
        remainder.append(f"{unknown} test{'s' if unknown != 1 else ''} could not be assessed")
    if passes:
        remainder.append("This measure passed the historical consistency check")
    return {"passed": passes, "message": intro + ". " + ". ".join(remainder) + ".",
            "required": needed, "positive_gate_passed": enough,
            "zero_worsening_passed": worse == 0}



def validation_summary(report: dict) -> dict:
    """Put the selected main metric first, explain supporting checks separately."""
    rob = report.get("rolling_robustness") or {}
    count = int(rob.get("fold_count") or 0)
    rows = holdout_chart_rows(report)
    if rob.get("status") != "evaluated" or count <= 0 or not rows:
        return {"status": "unavailable", "primary": None, "supporting": [], "description": "Not enough completed historical tests are available."}
    main = rows[0]
    supporting = rows[1:]
    main_verdict = metric_decision(main, count)
    failed_support = [r for r in supporting if not metric_decision(r, count)["passed"]]
    overall = rob.get("consistent_improvement_on_all_priorities") is True
    if overall:
        conclusion = "Every required measure passed the historical consistency check. Past success is not a forecast."
    elif not main_verdict["passed"]:
        conclusion = "The main goal did not pass all requirements. " + main_verdict["message"]
        if failed_support:
            conclusion += f" {len(failed_support)} supporting check{'s' if len(failed_support) != 1 else ''} also did not pass."
    elif failed_support:
        labels = ", ".join(r["Goal"].lower() for r in failed_support)
        conclusion = ("Your main goal met the rule, but the overall strategy was not approved because "
                      f"the supporting check{'s' if len(failed_support) != 1 else ''} for {labels} did not pass.")
    else:
        conclusion = "The validation report did not approve the strategy; review individual tests for the remaining requirements."
    return {"status": "passed" if overall else "not_passed", "primary": main,
            "supporting": supporting, "count": count, "description": conclusion,
            "primary_verdict": main_verdict, "failed_supporting": failed_support}
