"""V9.19 three-goal portfolio optimization and matched-date benchmarking.

Search each goal independently using *training-only* asset returns. One shared
market-data request powers all three analyses. Neither benchmark prices nor
holdout returns influence selected weights. Historical allocations are not
personalized investment recommendations or forward-looking forecasts.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_risk.v8_insights import InsightError, _metric, _stats
from portfolio_risk.v83_robustness import (
    _feasible,
    _project_candidate,
    _simulated_returns,
    _validated_rules,
)

GOALS = {
    "growth": ("Maximum growth", "cagr"),
    "balanced": ("Balanced growth and risk", "sharpe"),
    "defensive": ("Capital protection", "volatility"),
}


def _blocks(frame: pd.DataFrame, cadence: str):
    """Gross asset returns between scheduled rebalancing dates."""
    data = frame.to_numpy(dtype=float)
    if cadence == "daily":
        return 1 + data
    if cadence == "buy_hold":
        return np.prod(1 + data, axis=0, keepdims=True)
    idx = frame.index
    if cadence == "monthly":
        periods = [(x.year, x.month) for x in idx]
    elif cadence == "quarterly":
        periods = [(x.year, (x.month - 1) // 3) for x in idx]
    else:
        periods = [x.year for x in idx]
    splits = [i for i in range(1, len(idx)) if periods[i] != periods[i - 1]]
    return np.array([np.prod(1 + group, axis=0) for group in np.split(data, splits)])


def _fast_growth(blocks, weights, reference, cost_bps, days):
    """Terminal CAGR with periodic self-financing rebalancing and simple trade costs.

    Each rebalance is priced at the *start* of the period, consistent with the
    underlying daily simulator. All intermediate wealth cancels in a period's
    gross factor; this makes optimization fast without a Python loop per day.
    """
    gross = blocks @ weights
    if np.any(gross <= 0):
        return -1e6
    cost_rate = cost_bps / 10000
    turnover = np.abs(weights - reference).sum() / 2
    all_turnover = [turnover]
    if len(blocks) > 1:
        drift = blocks[:-1] * weights / gross[:-1, None]
        all_turnover.extend((np.abs(drift - weights).sum(axis=1) / 2).tolist())
    costs = 1 - cost_rate * np.asarray(all_turnover)
    if np.any(costs <= 0):
        return -1e6
    return float(np.expm1((np.log(gross).sum() + np.log(costs).sum()) * 252 / days))


def _prepared_paths(frame, cadence):
    """Precompute cumulative gross returns ONCE for a training window."""
    data = frame.to_numpy(dtype=float)
    if cadence == "daily":
        return ("daily", data)
    if cadence == "buy_hold":
        blocks = [data]
    else:
        idx = frame.index
        if cadence == "monthly":
            groups = [(x.year, x.month) for x in idx]
        elif cadence == "quarterly":
            groups = [(x.year, (x.month - 1) // 3) for x in idx]
        else:
            groups = [x.year for x in idx]
        splits = [i for i in range(1, len(idx)) if groups[i] != groups[i - 1]]
        blocks = np.split(data, splits)
    return ("blocks", [np.cumprod(1 + block, axis=0) for block in blocks])


def _returns_from_paths(prepared, weights, reference, cost_bps):
    """Exact daily drift and transaction-cost returns, with precomputed paths."""
    cost = cost_bps / 10000.0
    kind, data = prepared
    if kind == "daily":
        gross = 1 + data @ weights
        last_weights = weights[None, :] * (1 + data[:-1]) / gross[:-1, None]
        previous = np.vstack((reference, last_weights))
        turnovers = np.abs(previous - weights).sum(axis=1) / 2
        return gross * (1 - cost * turnovers) - 1
    results = []
    previous = reference
    for asset_wealth in data:
        trade = np.abs(previous - weights).sum() / 2.0
        portfolio_wealth = asset_wealth @ weights
        prior_wealth = np.r_[1.0, portfolio_wealth[:-1]]
        daily = portfolio_wealth / prior_wealth - 1
        daily[0] = (1 - cost * trade) * (1 + daily[0]) - 1
        results.append(daily)
        previous = weights * asset_wealth[-1] / portfolio_wealth[-1]
    return np.concatenate(results)


def _fast_daily_returns(frame, weights, reference, cost_bps, cadence):
    """Exact equivalent of V8.3 self-financing drift/rebalancing simulation."""
    return _returns_from_paths(_prepared_paths(frame, cadence), weights, reference, cost_bps)


def _feasible_starts(frame, reference, rules):
    n = len(reference)
    means = frame.mean().to_numpy(dtype=float)
    seeds = [reference, np.ones(n) / n]
    # Asset-focused starts are important for high-growth searches: they keep
    # the optimizer from getting stuck near the minimum-risk solution.
    for index in np.argsort(-means)[: min(n, 4)]:
        seeds.append(np.eye(n)[index])
    # Also inspect the lowest-volatility asset, but do not prefer it for growth.
    seeds.append(np.eye(n)[int(np.argmin(frame.std().to_numpy()))])
    out = []
    for candidate in seeds:
        try:
            projected = _project_candidate(candidate, reference, rules)
        except InsightError:
            projected = None
        if projected is not None and not any(np.allclose(projected, x, atol=1e-9) for x in out):
            out.append(projected)
    return out


def _select(frame, reference, rules, rf, goal, starts=None):
    """Constrained global-seed SLSQP; use truthful goal-specific objectives.

    Growth is maximized using actual requested rebalancing cadence/costs.
    Balanced uses annualized mean excess return / covariance risk (Sharpe).
    Defensive minimizes annualized variance, not gold weight or a proxy score.
    """
    seeds = starts if starts is not None else _feasible_starts(frame, reference, rules)
    if not seeds:
        return None, {"reason": "No valid allocations satisfy the limits"}
    # Balanced and defensive objectives use the EXACT training-period daily
    # returns produced by the requested rebalancing cadence and costs.
    # No SPY or RSP data enters any objective.
    blocks = _blocks(frame, rules["rebalance"]) if goal == "growth" else None
    prepared = _prepared_paths(frame, rules["rebalance"]) if goal != "growth" else None

    def objective(w):
        if goal == "growth":
            return -_fast_growth(blocks, w, reference, rules["cost_bps"], len(frame))
        daily = _returns_from_paths(prepared, w, reference, rules["cost_bps"])
        volatility = max(1e-12, float(daily.std(ddof=1) * np.sqrt(252)))
        if goal == "balanced":
            return (
                -(float(daily.mean()) - ((1.0 + rf) ** (1.0 / 252) - 1.0))
                / max(float(daily.std(ddof=1)), 1e-8)
                * np.sqrt(252)
            )
        return volatility

    lin = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1)}]
    for _, ix, cap in rules["groups"]:
        lin.append(
            {
                "type": "ineq",
                "fun": lambda w, indices=ix, maxcap=cap: float(maxcap - w[indices].sum()),
            }
        )
    turnover = rules["max_turnover"]
    if turnover is not None:
        lin.append(
            {"type": "ineq", "fun": lambda w: float(turnover - np.abs(w - reference).sum() / 2)}
        )
    candidates = [a for a in seeds if _feasible(a, rules)]
    # Multi-start protects against nonconvex Sharpe objectives and local minima.
    for init in seeds[: min(4, len(seeds))]:
        fit = minimize(
            objective,
            init,
            bounds=list(zip(rules["floors"], rules["caps"], strict=False)),
            constraints=lin,
            method="SLSQP",
            options={"maxiter": 85, "ftol": 1e-8},
        )
        if not np.isfinite(fit.x).all():
            continue
        try:
            w = _project_candidate(fit.x, reference, rules)
        except InsightError:
            continue
        if w is not None and _feasible(w, rules):
            candidates.append(w)
    if not candidates:
        return None, {"reason": "No feasible optimizer result"}
    winning = min(candidates, key=objective)
    # Honor exact constraints, and avoid misleading round-off drift on output.
    score = float(objective(winning))
    return winning, {
        "training_objective": score,
        "candidates_considered": len(candidates),
        "method": "SLSQP continuous search plus independently checked feasible starts",
        "objective": goal,
    }


def _metrics(frame, dates, weights, reference, rules, rf):
    stream, costs = _simulated_returns(
        frame,
        weights,
        rebalance=rules["rebalance"],
        cost_bps=rules["cost_bps"],
        starting_weights=reference,
    )
    return _stats(stream, dates, rf), costs


def _fold_metrics(frame, dates, reference, benchmarks, rules, rf, goal, end):
    train = frame.iloc[:end]
    test_n = rules["test_periods"]
    hold = frame.iloc[end : end + test_n]
    weights, audit = _select(train, reference, rules, rf, goal)
    if weights is None:
        return None
    test_dates = dates[end : end + test_n + 1]
    orig, orig_cost = _metrics(hold, test_dates, reference, reference, rules, rf)
    alt, alt_cost = _metrics(hold, test_dates, weights, reference, rules, rf)
    bm = {
        symbol: _stats(prices.iloc[end : end + test_n], test_dates, rf)
        for symbol, prices in benchmarks.items()
    }
    key = GOALS[goal][1]
    candidate_metric = _metric(alt, key)
    original_metric = _metric(orig, key)
    score = (
        float(original_metric - candidate_metric)
        if goal == "defensive"
        else float(candidate_metric - original_metric)
    )
    return {
        "train_end": str(dates[end].date()),
        "test_start": str(dates[end].date()),
        "test_end": str(dates[end + test_n].date()),
        "training_weights": dict(zip(frame.columns, map(float, weights), strict=False)),
        "original_metrics": orig,
        "tested_metrics": alt,
        "benchmarks": bm,
        "primary_improvement": score,
        "primary_better": score > 1e-9,
        "beats_benchmarks_on_goal": {
            sym: (
                float(_metric(alt, key) - _metric(metrics, key)) > 0
                if goal != "defensive"
                else float(_metric(metrics, key) - _metric(alt, key)) > 0
            )
            for sym, metrics in bm.items()
        },
        "training_audit": audit,
        "original_trading": orig_cost,
        "tested_trading": alt_cost,
    }


def ending_drift_weights(frame: pd.DataFrame, target: np.ndarray, cadence: str) -> np.ndarray:
    """End-of-simulation asset weight fractions under the same rebalance schedule.

    Fees decrease portfolio wealth; proportional transaction fees do not change
    the relative asset weights at the post-rebalance start of a period.
    """
    current = np.asarray(target, dtype=float).copy()
    last_period = None
    for i, (when, row) in enumerate(frame.iterrows()):
        if cadence == "monthly":
            period = (when.year, when.month)
        elif cadence == "quarterly":
            period = (when.year, (when.month - 1) // 3)
        elif cadence == "annually":
            period = when.year
        else:
            period = None
        if i == 0 or cadence == "daily" or (cadence != "buy_hold" and period != last_period):
            current = target.copy()
        current = current * (1 + row.to_numpy(dtype=float))
        current = current / current.sum()
        last_period = period
    return current


def analyze_three_strategies(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    reference: np.ndarray,
    benchmarks: Mapping[str, pd.Series],
    *,
    rules: Mapping | None = None,
    rf: float = 0.0,
    max_folds: int = 8,
) -> dict:
    """One shared price calendar, three goals, chronological no-leakage tests.

    Reference holdings can violate candidate caps; they remain an honest
    comparator, never a falsely eligible optimization candidate. The final
    proposed weights use ALL known prices; the holdouts evaluate the process.
    """
    if (
        len(frame) + 1 != len(dates)
        or not frame.index.equals(dates[1:])
        or not np.isfinite(frame.to_numpy()).all()
        or np.any(frame.to_numpy() <= -1)
        or len(frame) < 30
        or not np.isclose(reference.sum(), 1)
    ):
        raise InsightError("Three-strategy optimization requires valid aligned returns and weights")
    if set(benchmarks) != {"SPY", "RSP"} or any(
        not b.index.equals(frame.index) for b in benchmarks.values()
    ):
        raise InsightError("SPY and RSP must use the identical dates and currency as holdings")
    constraints = _validated_rules(list(frame), reference, rules or {})
    train_n, test_n, step_n = (
        constraints[k] for k in ("train_periods", "test_periods", "step_periods")
    )
    ends = list(range(train_n, len(frame) - test_n + 1, step_n))
    if max_folds and len(ends) > max_folds:
        ends = ends[-max_folds:]
    results = {}
    for goal, (label, primary) in GOALS.items():
        proposed, audit = _select(frame, reference, constraints, rf, goal)
        if proposed is None:
            results[goal] = {"status": "unavailable", **audit}
            continue
        current, costs = _metrics(frame, dates, proposed, reference, constraints, rf)
        original, _ = _metrics(frame, dates, reference, reference, constraints, rf)
        market = {symbol: _stats(series, dates, rf) for symbol, series in benchmarks.items()}
        folds = [
            result
            for end in ends
            if (
                result := _fold_metrics(
                    frame, dates, reference, benchmarks, constraints, rf, goal, end
                )
            )
            is not None
        ]
        better = sum(f["primary_better"] for f in folds)
        worse = sum(f["primary_improvement"] < -1e-9 for f in folds)
        valid = len(folds)
        benchmark_counts = {
            sym: sum(f["beats_benchmarks_on_goal"][sym] for f in folds) for sym in ("SPY", "RSP")
        }
        # A strong majority is meaningful even if one later window lost.  Also
        # require a positive average objective improvement: four negligible
        # victories cannot hide one catastrophic defeat.
        mean_advantage = float(np.mean([f["primary_improvement"] for f in folds])) if folds else 0.0
        baseline_consistent = (
            valid >= 3 and better >= int(np.ceil(0.75 * valid)) and mean_advantage > 1e-9
        )
        benchmarks_consistent = valid >= 3 and all(
            value >= int(np.ceil(0.75 * valid)) for value in benchmark_counts.values()
        )
        # Track two distinct questions. A credible improvement versus the
        # investor's existing allocation is not the same as sustained
        # outperformance against SPY and RSP. Preserve the legacy combined
        # status for existing consumers; the V9.21 UI displays both separately.
        portfolio_evidence = (
            "insufficient"
            if valid < 3
            else "consistent"
            if baseline_consistent
            else "mixed"
            if better > worse
            else "not_supported"
        )
        benchmark_evidence = (
            "insufficient"
            if valid < 3
            else "consistent_both"
            if benchmarks_consistent
            else "consistent_one"
            if any(v >= int(np.ceil(0.75 * valid)) for v in benchmark_counts.values())
            else "not_consistent"
        )
        status = (
            "insufficient"
            if valid < 3
            else "historically_consistent"
            if baseline_consistent and benchmarks_consistent
            else "mixed"
            if better > worse
            else "not_supported"
        )
        results[goal] = {
            "status": "evaluated",
            "label": label,
            "primary_metric": primary,
            "weights": dict(zip(frame.columns, map(float, proposed), strict=False)),
            "ending_weights": dict(
                zip(
                    frame.columns,
                    map(float, ending_drift_weights(frame, proposed, constraints["rebalance"])),
                 strict=False)
            ),
            "current_training_metrics": current,
            "original_training_metrics": original,
            "benchmarks_training_metrics": market,
            "training_audit": audit,
            "trading": costs,
            "holdout": {
                "status": status,
                "tests": valid,
                "better": better,
                "worse": worse,
                "similar": valid - better - worse,
                "portfolio_evidence": portfolio_evidence,
                "benchmark_evidence": benchmark_evidence,
                "baseline_consistent": baseline_consistent,
                "mean_primary_improvement": mean_advantage,
                "benchmark_counts": benchmark_counts,
                "benchmarks_consistent": benchmarks_consistent,
                "folds": folds,
            },
        }
    return {
        "status": "evaluated",
        "version": "9.21",
        "history_start": str(dates[0].date()),
        "history_end": str(dates[-1].date()),
        "observations": len(frame),
        "candidate_cap": constraints["max_weight"],
        "rebalancing": constraints["rebalance"],
        "trading_cost_bps": constraints["cost_bps"],
        "validation": {
            "max_folds": max_folds,
            "tested_folds": len(ends),
            "train_periods_min": train_n,
            "holdout_periods": test_n,
            "step_periods": step_n,
            "note": "Each later test used only prices available before that test. Final proposed weights use all available observations and are not a forward return forecast.",
        },
        "strategies": results,
    }
