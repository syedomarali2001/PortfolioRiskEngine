"""Evidence-based V8.3 explanations. Suggestions are tests, NEVER trade instructions."""

from __future__ import annotations

from math import isfinite

from portfolio_risk.v8_insights import THRESHOLDS


def portfolio_actions(report: dict, robustness: dict | None = None) -> dict:
    """Give benchmark targets and a falsifiable research next step for each metric.

    No vague verdicts, no language-model generated prescriptions, no claims that
    exceeding an arbitrary materiality threshold proves a statistical effect.
    """
    actions = []
    for level in ("negative", "positive", "neutral", "unavailable"):
        for item in report["diagnostics"][level]:
            metric = item["metric"]
            benchmark = item["benchmark"]
            value = item["benchmark_value"]
            higher = metric not in {"volatility", "drawdown", "es95"}
            threshold = THRESHOLDS[metric]
            if level == "unavailable":
                target = None
                next_step = "Obtain at least 126 aligned observations with defined metrics before comparing."
            else:
                target = {
                    "comparison": "at_least" if higher else "at_most",
                    "benchmark_value": value,
                    "benchmark": benchmark,
                    "metric": metric,
                    "materiality": threshold,
                    "strong_advantage_target": value + threshold
                    if higher
                    else max(0.0, value - threshold),
                }
                if level == "negative":
                    next_step = (
                        f"Test constrained allocation changes targeting {metric}; "
                        "compare with the original and both benchmarks in separate future holdout windows."
                    )
                elif level == "positive":
                    next_step = (
                        f"Preserve the demonstrated {metric} advantage when testing trade-offs; "
                        "verify it survives new holdouts and trading costs."
                    )
                else:
                    next_step = (
                        f"No material advantage established for {metric}; "
                        f"use the stated threshold to decide whether a change is meaningful."
                    )
            actions.append(
                {
                    "metric": metric,
                    "benchmark": benchmark,
                    "sentiment": level,
                    "observed_difference_favorable": item["favorable_difference"],
                    "target": target,
                    "evidence": item["evidence"],
                    "research_next_step": next_step,
                }
            )
    strongest_risk = []
    for entry in sorted(report["risk_contribution"], key=lambda x: -x["share_of_total_risk"]):
        share = float(entry["share_of_total_risk"])
        weight = float(entry["weight"])
        if isfinite(share) and share > 0.35 and share > 1.3 * weight:
            strongest_risk.append(
                {
                    "asset": entry["symbol"],
                    "sentiment": "negative",
                    "portfolio_weight": weight,
                    "volatility_risk_share": share,
                    "rule": "Flag when one asset contributes >35% of volatility and >1.3x its portfolio weight",
                    "evidence": f"{entry['symbol']}: {weight:.2%} allocation produces {share:.2%} of modeled volatility risk.",
                    "target": "Investigate risk share <=35% (heuristic, not universal optimum)",
                    "research_next_step": f"Test a small reduction in {entry['symbol']} weight within stated constraints; reject if holdout results do not improve chosen metrics.",
                }
            )
    status = "not_evaluated"
    if robustness is not None:
        if robustness.get("status") != "evaluated":
            status = "unavailable"
        elif robustness.get("consistent_improvement_on_all_priorities"):
            status = "historical_consistency_observed"
        else:
            status = "no_consistent_improvement"
    if status == "historical_consistency_observed":
        conclusion = (
            "Candidate-selection process improved every chosen priority in at least 75% of "
            "nonoverlapping holdouts without material deterioration; inspect weights/costs, not a future guarantee."
        )
    elif status == "no_consistent_improvement":
        if (
            robustness is not None
            and robustness.get("constraints", {}).get("baseline_compliant") is False
        ):
            conclusion = (
                "No tested candidate-selection process consistently improved all priorities. "
                "The original allocation violates the user-specified limits: do not portray it as compliant. "
                "A constraint-compliance example is available, but is NOT validated as a risk improvement."
            )
        else:
            conclusion = (
                "No candidate-selection process consistently improved all priorities in the tested windows. "
                "Do not recommend changing allocation on this evidence."
            )
    elif status == "unavailable":
        conclusion = "Rolling validation unavailable; do not claim an allocation improvement."
    else:
        conclusion = "Rolling validation was not requested; benchmark differences alone do not justify changing holdings."
    negative = sorted(
        (x for x in actions if x["sentiment"] == "negative"),
        key=lambda x: -abs(x["observed_difference_favorable"]) / THRESHOLDS[x["metric"]],
    )
    return {
        "benchmark_targets": actions,
        "negative_findings_priority": negative,
        "concentration_findings": strongest_risk,
        "allocation_change_validation_status": status,
        "conclusion": conclusion,
        "next_action": (
            robustness.get("next_step")
            if robustness and robustness.get("next_step")
            else "Run rolling constrained evaluation with --robustness before treating hypothetical alternatives as improvements."
        ),
        "policy": "Descriptive benchmark gaps and predefined thresholds; user-specific risk tolerance, not investment advice or statistical significance.",
    }
