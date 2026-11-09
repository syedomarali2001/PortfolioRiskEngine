"""V9.13: six independent strategy paths with careful out-of-sample evidence.

All numerical comparisons derive from V8.4 return/report outputs. A fold's
latest candidate is ONE historically chosen allocation; it is not the same
portfolio tested in all earlier folds, nor a validated future allocation.
"""
from __future__ import annotations

import math

from .v910_presentation import all_goal_review
from .v911_presentation import latest_candidate_example
from .v912_presentation import strategy_tradeoff

PATHS = (
    ("Balance growth and risk", "Balanced", "sharpe", "More return for the volatility taken"),
    ("More annual growth", "Higher growth", "cagr", "Seek higher annualized return, potentially with bigger swings"),
    ("Smoother returns", "Smoother returns", "volatility", "Reduce fluctuations in the portfolio's value"),
    ("Smaller losses", "Smaller declines", "drawdown", "Reduce the largest historical peak-to-trough decline"),
    ("Better downside-adjusted returns", "Downside-adjusted", "sortino", "Seek higher return relative to harmful fluctuations"),
    ("Reduce losses on bad days", "Bad-day protection", "es95", "Reduce losses during the worst historical trading days"),
)

PICKED = {row[0]: row for row in PATHS}


def strategy_report_args(base_args: list[str], priorities: str) -> list[str]:
    """Return a fresh argument list for a distinct optimization objective."""
    args = list(base_args)
    if "--priorities" in args:
        pos = args.index("--priorities")
        args[pos + 1] = priorities
    else:
        args.extend(["--priorities", priorities])
    return args


def spy_benchmark_evidence(report: dict, primary: str) -> dict:
    """Compare candidate to SPY on identical later dates, in reporting currency.

    An aggregate pass needs >=75% of valid fold comparisons. There is no
    invented result if the historical report predates SPY fold evidence.
    """
    robust = report.get("rolling_robustness") or {}
    folds = robust.get("folds") or []
    if robust.get("status") != "evaluated" or not folds:
        return {"status": "unavailable", "reason": "No completed historical tests"}
    metrics = (primary, "cagr", "sharpe")
    evidence = {}
    for metric in dict.fromkeys(metrics):
        valid = []
        for fold in folds:
            value = (fold.get("candidate_vs_spy") or {}).get(metric) or {}
            gap = value.get("favorable_difference")
            if isinstance(gap, (float, int)) and math.isfinite(gap):
                valid.append({"gap": float(gap), "candidate": value.get("candidate_value"),
                              "spy": value.get("spy_value")})
        if len(valid) != len(folds):
            return {"status": "unavailable", "reason": "SPY comparison is not available for every test"}
        met = sum(v["gap"] >= -1e-12 for v in valid)
        evidence[metric] = {
            "matched": met, "total": len(valid), "required": math.ceil(0.75 * len(valid)),
            "last": valid[-1], "sufficient": met >= math.ceil(0.75 * len(valid))
        }
    return {"status": "evaluated", "metrics": evidence,
            "supported": all(evidence[m]["sufficient"] for m in (primary, "cagr", "sharpe"))}


def path_summary(report: dict, goal: str) -> dict:
    """Qualify a path without conflating its latest allocation with a guarantee."""
    _, short_name, primary, detail = PICKED[goal]
    example = latest_candidate_example(report)
    review = all_goal_review(report)
    trade = strategy_tradeoff(report)
    spy = spy_benchmark_evidence(report, primary)
    own_passed = review.get("passed") is True
    supported = bool(own_passed and spy.get("supported") is True)
    primary_row = next((r for r in review.get("rows", []) if r.get("Metric") == primary), None)
    if example.get("status") != "evaluated" or review.get("status") != "evaluated":
        status = "Not enough historical evidence"
        explanation = "This path did not produce a complete set of comparable results."
    elif supported:
        status = "Passed strict historical checks"
        explanation = ("This selection method met its original-portfolio checks and matched or "
                       "exceeded SPY on its target, growth and risk-adjusted return in at least "
                       "75% of the same later test periods. This is not a future guarantee.")
    elif not own_passed:
        status = "Mixed evidence, not validated"
        explanation = ("This approach did not pass the strict consistency checks against your existing "
                       "portfolio. Some measures may still improve, but the historical evidence "
                       "does not establish dependable outperformance or justify an automatic allocation change.")
    elif spy.get("status") != "evaluated":
        status = "Market comparison not verified"
        explanation = ("The method met its original-portfolio checks, but matching-date "
                       "comparisons with SPY are unavailable. Do not interpret this as "
                       "better than the market.")
    else:
        status = "Improvement over original, not confirmed against SPY"
        explanation = ("The method passed its original-portfolio checks, but its results "
                       "did not meet our stricter same-date SPY reference check. It is not "
                       "presented as market-beating or as a recommendation.")
    return {"goal": goal, "short_name": short_name, "primary": primary, "detail": detail,
            "status": status, "explanation": explanation, "supported": supported,
            "own_passed": own_passed, "example": example, "review": review,
            "trade": trade, "spy": spy, "primary_row": primary_row}


def path_signal(summary: dict) -> tuple[str, str]:
    """User-facing signal reflects evidence, not a buy/sell verdict.

    ★: full strict historical and SPY checks passed;
    ○: evidence is incomplete or mixed;
    ✕: failed original-portfolio historical approval checks.
    """
    if summary.get("supported"):
        return "★", "Passed strict historical checks"
    if ((summary.get("example") or {}).get("status") != "evaluated" or
            (summary.get("review") or {}).get("status") != "evaluated"):
        return "○", "Insufficient historical evidence"
    if summary.get("own_passed"):
        return "○", "Improved against baseline, but SPY confirmation was insufficient"
    # Do not show a frightening red failure when a strategy substantially
    # improves its stated goal but misses an extra strict pass condition.
    row = summary.get("primary_row") or {}
    better, worse = int(row.get("Better", 0)), int(row.get("Worse", 0))
    if better > worse and better > 0:
        return "○", "Promising in some windows, not consistently validated"
    return "✕", "No convincing improvement on the selected goal"
