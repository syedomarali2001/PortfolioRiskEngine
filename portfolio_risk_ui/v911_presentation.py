"""V9.11: explain a tested reallocation without inventing investment advice.

Each robustness fold makes a DIFFERENT training-only choice and evaluates that
choice on later observations. A latest-fold example is not an overall optimum,
and an aggregation across folds is not a validated single static allocation.
"""
from __future__ import annotations

import math
from statistics import median

from .v910_presentation import METRICS

LABELS = {
    "input": "Original holdings",
    "adjusted_input": "Starting weights adjusted to the allocation limits",
    "equal_weight": "Equal weights",
    "inverse_volatility": "Weights that favor historically less volatile investments",
    "minimum_variance": "Lowest estimated portfolio variance",
}


def candidate_label(raw: object) -> str:
    """Explain the existing finite candidate family, without renaming its math."""
    key = str(raw or "")
    if key in LABELS:
        return LABELS[key]
    if key.startswith("continuous_"):
        readable = {"sharpe": "return per unit of risk", "sortino": "downside-adjusted return",
                    "volatility": "lower portfolio volatility", "drawdown": "lower sustained-loss exposure",
                    "es95": "smaller losses on bad days", "cagr": "higher historical growth"}
        goal = readable.get(key.split("continuous_", 1)[1], "risk and return")
        return "Constrained continuous search for " + goal
    if key.startswith("frontier_"):
        return "Constrained growth-risk frontier allocation"
    if key.startswith("minvar_equal_"):
        return "Blend of minimum variance and equal weights"
    if key.startswith("inverse_input_"):
        return "Blend of inverse volatility and original holdings"
    return key.replace("_", " ").capitalize() if key else "Not specified"


def _valid_weights(source: object) -> dict[str, float] | None:
    if not isinstance(source, dict) or not source:
        return None
    try:
        weights = {str(ticker): float(weight) for ticker, weight in source.items()}
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) and x >= 0 for x in weights.values()):
        return None
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        return None
    return weights


def latest_candidate_example(report: dict) -> dict:
    """Return exact, existing latest-fold weights; never infer missing candidates.

    No chosen weights are generated here. Weights and outcomes must be present
    in the original robustness report, in the SAME fold, with consistent keys.
    """
    rob = report.get("rolling_robustness") or {}
    folds = rob.get("folds") or []
    baseline = _valid_weights(report.get("input_weights"))
    if rob.get("status") != "evaluated" or not folds or baseline is None:
        return {"status": "unavailable", "reason": "No complete historical allocation example was provided."}
    latest = folds[-1]
    if not isinstance(latest, dict):
        return {"status": "unavailable", "reason": "The latest test is incomplete."}
    tested = _valid_weights(latest.get("weights"))
    if tested is None or baseline.keys() != tested.keys():
        return {"status": "unavailable", "reason": "A complete allocation for the latest test is unavailable."}

    allocation_rows = [{
        "Ticker": ticker,
        "Original (%)": baseline[ticker] * 100,
        "Tested (%)": tested[ticker] * 100,
        "Change (percentage points)": (tested[ticker] - baseline[ticker]) * 100,
    } for ticker in baseline]
    allocation_rows.sort(key=lambda row: (-abs(row["Change (percentage points)"]), row["Ticker"]))
    turnover_pct = sum(abs(tested[k] - baseline[k]) for k in baseline) * 50

    verdicts = latest.get("all_metric_verdicts") or latest.get("verdicts") or {}
    measures = []
    for metric, title in METRICS:
        evidence = verdicts.get(metric)
        if not isinstance(evidence, dict):
            continue
        old, new = evidence.get("original_value"), evidence.get("candidate_value")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (old, new)):
            continue
        direction = "Lower is favorable" if metric in ("volatility", "drawdown", "es95") else "Higher is favorable"
        measures.append({"Metric": title, "Key": metric,
                         "Your portfolio": float(old), "Tested allocation": float(new),
                         "Result": {"positive": "Better", "negative": "Worse",
                                    "neutral": "Similar"}.get(evidence.get("sentiment"), "Not assessed"),
                         "Context": direction})

    historical_ranges = []
    valid_folds = [w for fold in folds if isinstance(fold, dict)
                   if (w := _valid_weights(fold.get("weights"))) is not None and w.keys() == baseline.keys()]
    for ticker in baseline:
        observed = [w[ticker] * 100 for w in valid_folds]
        if observed:
            historical_ranges.append({"Ticker": ticker,
                                      "Lowest tested (%)": min(observed),
                                      "Median tested (%)": median(observed),
                                      "Highest tested (%)": max(observed)})
    historical_ranges.sort(key=lambda row: (-abs(row["Median tested (%)"] - baseline[row["Ticker"]]*100), row["Ticker"]))

    return {"status": "evaluated", "selected": latest.get("selected"),
            "selected_label": candidate_label(latest.get("selected")),
            "training_end": latest.get("train_end"),
            "holdout_start": latest.get("holdout_start"),
            "holdout_end": latest.get("holdout_end"),
            "rows": allocation_rows, "measures": measures,
            "historical_ranges": historical_ranges,
            "valid_fold_count": len(valid_folds),
            "allocation_turnover_pct": turnover_pct,
            "fold_count": int(rob.get("fold_count") or len(folds)),
            "passed": rob.get("consistent_improvement_on_all_priorities") is True}


def status_group(item: dict) -> tuple[str, str]:
    """Compact visual status, based only on recorded per-metric results."""
    if item.get("passed_independent_check"):
        return "Consistent improvement", "green"
    # The named result should not say 'improvement' if regressions dominated.
    label = str(item.get("summary") or "No clear advantage").rstrip(".")
    if label.startswith("Worsened"):
        return "More often worse", "red"
    if label.startswith("Often better"):
        return "Improved often, with setbacks", "amber"
    if label.startswith("Improved"):
        return "Mixed, some improvement", "amber"
    return "No clear advantage", "gray"


def format_metric_value(key: str, value: float) -> str:
    return f"{value:.2f}" if key in {"sharpe", "sortino"} else f"{value * 100:.2f}%"
