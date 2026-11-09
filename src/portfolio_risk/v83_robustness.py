"""V8.3 reproducible allocation constraints and forward-only rolling evaluation.

No market-data calls. Nonoverlapping, explicitly named portfolio groups only; no
classification guesses. Each fold reselects from its own preceding training data.
All price returns must have been validated, aligned, and base-currency converted.
"""

from __future__ import annotations

from math import isfinite
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk.v8_insights import (
    THRESHOLDS,
    InsightError,
    _candidate_weights,
    _finite,
    _metric,
    _preferences,
    _score,
    _stats,
)


def _box_project(target: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto box with fixed sum, via monotone bisection."""
    if (
        np.any(lo < -1e-10)
        or np.any(lo > hi + 1e-10)
        or total < lo.sum() - 1e-8
        or total > hi.sum() + 1e-8
    ):
        raise InsightError("Allocation limits are infeasible: minimums/maximums cannot sum to 100%")
    lower = float(np.min(lo - target)) - 1
    upper = float(np.max(hi - target)) + 1
    for _ in range(90):
        mid = (lower + upper) / 2
        if np.clip(target + mid, lo, hi).sum() < total:
            lower = mid
        else:
            upper = mid
    weights = np.clip(target + (lower + upper) / 2, lo, hi)
    if abs(weights.sum() - total) > 1e-7:
        raise InsightError("Constrained projection failed to satisfy its requested total")
    return weights


def _validated_rules(names: list[str], reference: np.ndarray, options: Mapping) -> dict:
    max_weight = float(options.get("max_weight", 0.7))
    if not isfinite(max_weight) or not 0 < max_weight <= 1:
        raise InsightError("max_weight must be in (0,1]")
    floor_map = dict(options.get("asset_floors") or {})
    cap_map = dict(options.get("asset_caps") or {})
    groups = dict(options.get("group_caps") or {})
    unknown = (set(floor_map) | set(cap_map)) - set(names)
    if unknown:
        raise InsightError(
            f"Allocation limit refers to unknown portfolio ticker(s): {', '.join(sorted(unknown))}"
        )
    lo = np.zeros(len(names), dtype=float)
    hi = np.full(len(names), max_weight, dtype=float)
    for index, sym in enumerate(names):
        if sym in floor_map:
            lo[index] = float(floor_map[sym])
        if sym in cap_map:
            hi[index] = min(hi[index], float(cap_map[sym]))
    if not (
        np.isfinite(lo).all()
        and np.isfinite(hi).all()
        and np.all(lo >= 0)
        and np.all(hi <= 1)
        and np.all(lo <= hi)
    ):
        raise InsightError("Every asset floor/cap must be finite, 0..100%, and floor <= cap")
    seen = set()
    parts = []
    printed = {}
    for label, spec in groups.items():
        if not label or not isinstance(spec, Mapping):
            raise InsightError("Group caps must have a name, tickers and maximum fraction")
        tickers = list(spec.get("symbols") or [])
        maximum = float(spec.get("cap", float("nan")))
        if (
            not tickers
            or len(set(tickers)) != len(tickers)
            or any(t not in names for t in tickers)
            or any(t in seen for t in tickers)
            or not isfinite(maximum)
            or not 0 <= maximum <= 1
        ):
            raise InsightError(
                f"Group {label} has missing/overlapping/unknown tickers or invalid cap"
            )
        seen.update(tickers)
        indices = np.array([names.index(t) for t in tickers], dtype=int)
        parts.append((label, indices, maximum))
        printed[label] = {"symbols": tickers, "cap": maximum}
    for sym in names:
        if sym not in seen:
            parts.append((f"asset:{sym}", np.array([names.index(sym)], dtype=int), 1.0))
    groups_lo = np.array([lo[ix].sum() for _, ix, _ in parts])
    groups_hi = np.array([min(hi[ix].sum(), maximum) for _, ix, maximum in parts])
    if (
        np.any(groups_lo > groups_hi + 1e-9)
        or groups_lo.sum() > 1 + 1e-9
        or groups_hi.sum() < 1 - 1e-9
    ):
        raise InsightError("Group/asset limits are infeasible: cannot allocate 100%")
    turnover = options.get("max_turnover")
    if turnover is not None and (not isfinite(float(turnover)) or not 0 <= float(turnover) <= 1):
        raise InsightError("max_turnover must be a one-way fraction in [0,1]")
    cost_bps = float(options.get("cost_bps", 0.0))
    if not isfinite(cost_bps) or not 0 <= cost_bps <= 500:
        raise InsightError("Trading cost must be 0..500 basis points per one-way traded dollar")
    rebalance = options.get("rebalance", "monthly")
    if rebalance not in {"daily", "monthly", "quarterly", "annually", "buy_hold"}:
        raise InsightError("Rebalance must be daily/monthly/quarterly/annually/buy_hold")
    train = int(options.get("train_periods", 252))
    test = int(options.get("test_periods", 126))
    step = int(options.get("step_periods", 126))
    if train < 126 or test < 30 or step < test or step > 504 or train > 3000 or test > 504:
        raise InsightError("Require training >=126, test >=30, test <= step <=504, training <=3000")
    return {
        "max_weight": max_weight,
        "floors": lo,
        "caps": hi,
        "groups": parts,
        "group_caps": printed,
        "max_turnover": float(turnover) if turnover is not None else None,
        "cost_bps": cost_bps,
        "rebalance": rebalance,
        "train_periods": train,
        "test_periods": test,
        "step_periods": step,
        "user_bounds": {"floors": floor_map, "caps": cap_map},
    }


def _project_candidate(target: np.ndarray, reference: np.ndarray, rules: dict) -> np.ndarray | None:
    parts = rules["groups"]
    lo, hi = rules["floors"], rules["caps"]
    targets = np.array([target[ix].sum() for _, ix, _ in parts])
    mins = np.array([lo[ix].sum() for _, ix, _ in parts])
    maxs = np.array([min(hi[ix].sum(), cap) for _, ix, cap in parts])
    grouped = _box_project(targets, mins, maxs, 1.0)
    out = np.zeros(len(target))
    for part, amount in zip(parts, grouped, strict=False):
        _, indices, _ = part
        out[indices] = _box_project(target[indices], lo[indices], hi[indices], float(amount))
    # Turnover is one-way: half the sum of absolute weight changes.
    limit = rules["max_turnover"]
    turnover = float(np.abs(out - reference).sum() / 2)
    if limit is not None and turnover > limit + 1e-8:
        if _feasible(reference, rules):
            out = reference + (out - reference) * (limit / turnover)
        else:
            return None  # An infeasible baseline cannot safely be interpolated.
    return out if _feasible(out, rules) else None


def _feasible(w: np.ndarray, rules: dict) -> bool:
    return bool(
        (
            bool(np.isfinite(w).all())
            and abs(w.sum() - 1) < 1e-7
            and np.all(w >= rules["floors"] - 1e-8)
            and np.all(w <= rules["caps"] + 1e-8)
            and all(w[ix].sum() <= cap + 1e-8 for _, ix, cap in rules["groups"])
        )
    )


def _trading_period(d: pd.Timestamp, cadence: str):
    if cadence == "monthly":
        return (d.year, d.month)
    if cadence == "quarterly":
        return (d.year, (d.month - 1) // 3)
    if cadence == "annually":
        return d.year
    return None


def _simulated_returns(
    asset: pd.DataFrame,
    target: np.ndarray,
    *,
    rebalance: str,
    cost_bps: float,
    starting_weights: np.ndarray | None = None,
) -> tuple[pd.Series, dict]:
    """Self-financing drifting weights, with proportional one-way turnover costs.

    Each fold is an isolated hypothetical entry from its baseline holdings.
    Rebalance at the first matched daily return of each new calendar period.
    """
    current = target.copy() if starting_weights is None else starting_weights.copy()
    cost_rate = cost_bps / 10000.0
    records = []
    turnover_total = 0.0
    total_cost = 0.0
    events = 0
    last_period = None
    for index, (when, row) in enumerate(asset.iterrows()):
        current_period = _trading_period(when, rebalance)
        do_rebalance = (
            index == 0
            or rebalance == "daily"
            or (rebalance != "buy_hold" and current_period != last_period)
        )
        traded = 0.0
        if do_rebalance:
            traded = float(np.abs(target - current).sum() / 2)
            current = target.copy()
            if traded > 1e-12:
                events += 1
        cost = traded * cost_rate
        daily = float(current @ row.to_numpy(dtype=float))
        if daily <= -1 or not isfinite(daily):
            raise InsightError("Invalid daily portfolio return under drift simulation")
        records.append((1 - cost) * (1 + daily) - 1)
        total_cost += cost
        turnover_total += traded
        # End-of-day weights drift until the next scheduled rebalance.
        current = current * (1 + row.to_numpy(dtype=float)) / (1 + daily)
        last_period = current_period
    return pd.Series(records, index=asset.index, dtype=float), {
        "one_way_turnover_sum": turnover_total,
        "cost_fraction_sum_before_compounding": total_cost,
        "charged_rebalance_events": events,
        "cost_bps": cost_bps,
        "cadence": rebalance,
    }


def _score_training(
    data: pd.DataFrame,
    dates: pd.DatetimeIndex,
    reference: np.ndarray,
    rules: dict,
    rf: float,
    priorities: Mapping[str, float],
) -> tuple[dict, dict, dict]:
    raw = _candidate_weights(data, reference, 1.0)
    raw["adjusted_input"] = reference.copy()
    # V9.15 searches the continuous weight space with explicit box, group,
    # sum and turnover constraints, using ONLY this training window.
    from portfolio_risk.v915_optimizer import continuous_candidates

    raw.update(continuous_candidates(data, reference, rules, rf, priorities, _project_candidate))
    projected = {}
    for label, target in raw.items():
        if label == "input":
            if _feasible(reference, rules):
                projected[label] = reference.copy()
            continue
        p = _project_candidate(target, reference, rules)
        if p is not None and not any(
            np.allclose(p, v, atol=1e-8, rtol=0) for v in projected.values()
        ):
            projected[label] = p
    if not projected:
        return {}, {}, {"reason": "No candidates satisfy all caps and turnover limits"}
    scores = {}
    for key, w in projected.items():
        series, _ = _simulated_returns(
            data,
            w,
            rebalance=rules["rebalance"],
            cost_bps=rules["cost_bps"],
            starting_weights=reference,
        )
        scores[key] = _stats(series, dates, rf)
    totals = {k: 0.0 for k in projected}
    for objective, importance in priorities.items():
        values = np.array([_score(scores[k], objective) for k in projected], dtype=float)
        low, high = values.min(), values.max()
        scale = (values - low) / (high - low) if high - low > 1e-12 else np.full(len(values), 0.5)
        for k, part in zip(projected, scale, strict=False):
            totals[k] += float(importance * part)
    return projected, scores, totals


def analyze_robustness(
    asset_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    reference: np.ndarray,
    *,
    priorities: Mapping[str, float] | None = None,
    rules: Mapping | None = None,
    rf: float = 0.0,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
) -> dict:
    """Expanding train, non-overlapping chronological holdout folds; NO holdout-based selection."""
    if len(dates) != len(asset_returns) + 1 or not asset_returns.index.equals(dates[1:]):
        raise InsightError(
            "Rolling evaluation requires aligned return dates and their preceding close"
        )
    if not np.isfinite(asset_returns.to_numpy()).all() or np.any(asset_returns.to_numpy() <= -1):
        raise InsightError("Rolling evaluation requires finite returns greater than -100%")
    if (
        not np.isfinite(reference).all()
        or np.any(reference < 0)
        or not np.isclose(reference.sum(), 1)
    ):
        raise InsightError("Invalid baseline weights for rolling evaluation")
    # Optional benchmark returns must be on the same verified trading calendar
    # and converted to the same reporting currency upstream.
    spy_returns = None
    if benchmark_returns is not None and "SPY" in benchmark_returns:
        spy_returns = benchmark_returns["SPY"]
        if (
            not isinstance(spy_returns, pd.Series)
            or not spy_returns.index.equals(asset_returns.index)
            or not np.isfinite(spy_returns.to_numpy(dtype=float)).all()
            or (spy_returns.to_numpy(dtype=float) <= -1).any()
        ):
            raise InsightError("SPY benchmark returns must match the historical trading dates")
    selected = _preferences(priorities)
    constraints = _validated_rules(list(asset_returns), reference, rules or {})
    baseline_compliant = _feasible(reference, constraints)
    projected_reference = _project_candidate(reference, reference, constraints)
    # This is a transparent constraint-compliance example, NEVER a risk-validated winner.
    compliance_example = (
        {
            "status": "already_compliant",
            "weights": dict(zip(asset_returns.columns, map(float, reference), strict=False)),
            "initial_one_way_turnover": 0.0,
            "interpretation": "Original portfolio satisfies the requested target-weight limits.",
        }
        if baseline_compliant
        else {
            "status": "feasible_not_risk_validated",
            "weights": dict(zip(asset_returns.columns, map(float, projected_reference), strict=False)),
            "initial_one_way_turnover": float(np.abs(projected_reference - reference).sum() / 2),
            "interpretation": "Example satisfying target-weight limits; no claim of better investment outcomes.",
        }
        if projected_reference is not None
        else {
            "status": "unavailable",
            "weights": None,
            "initial_one_way_turnover": None,
            "interpretation": "A feasible reallocation could not be identified under these limits and entry-turnover cap.",
        }
    )
    train_n = constraints["train_periods"]
    test_n = constraints["test_periods"]
    step = constraints["step_periods"]
    if len(asset_returns) < train_n + test_n:
        return {
            "status": "unavailable",
            "reason": f"Need {train_n + test_n} aligned returns; only {len(asset_returns)} available",
            "constraints": {**_rules_out(constraints), "baseline_compliant": baseline_compliant},
            "fold_count": 0,
        }
    folds = []
    for train_end in range(train_n, len(asset_returns) - test_n + 1, step):
        train = asset_returns.iloc[:train_end]
        hold = asset_returns.iloc[train_end : train_end + test_n]
        candidates, train_stats, training_scores = _score_training(
            train, dates[: train_end + 1], reference, constraints, rf, selected
        )
        if not candidates:
            return {
                "status": "unavailable",
                "reason": training_scores["reason"],
                "constraints": {
                    **_rules_out(constraints),
                    "baseline_compliant": baseline_compliant,
                },
                "fold_count": 0,
            }
        winning = sorted(candidates, key=lambda k: (-training_scores[k], k != "input", k))[0]
        baseline_returns, baseline_costs = _simulated_returns(
            hold,
            reference,
            rebalance=constraints["rebalance"],
            cost_bps=constraints["cost_bps"],
            starting_weights=reference,
        )
        candidate_returns, candidate_costs = _simulated_returns(
            hold,
            candidates[winning],
            rebalance=constraints["rebalance"],
            cost_bps=constraints["cost_bps"],
            starting_weights=reference,
        )
        observed_dates = dates[train_end : train_end + test_n + 1]
        original = _stats(baseline_returns, observed_dates, rf)
        tested = _stats(candidate_returns, observed_dates, rf)
        spy_stats = (
            _stats(spy_returns.iloc[train_end : train_end + test_n], observed_dates, rf)
            if spy_returns is not None
            else None
        )
        # This is a separate out-of-sample ETF comparison, not another
        # candidate for the optimizer to select using future information.
        spy_relative = {}
        if spy_stats is not None:
            for metric in THRESHOLDS:
                lhs, rhs = _metric(tested, metric), _metric(spy_stats, metric)
                difference = (
                    _score(tested, metric) - _score(spy_stats, metric)
                    if _finite(lhs) and _finite(rhs)
                    else None
                )
                spy_relative[metric] = {
                    "candidate_value": lhs,
                    "spy_value": rhs,
                    "favorable_difference": difference,
                    "at_least_as_good": difference is not None and difference >= -1e-12,
                }
        all_evidence = {}
        for metric in THRESHOLDS:
            change = (
                _score(tested, metric) - _score(original, metric)
                if _finite(_metric(original, metric)) and _finite(_metric(tested, metric))
                else None
            )
            threshold = THRESHOLDS[metric]
            sentiment = (
                "unavailable"
                if change is None
                else "positive"
                if change >= threshold
                else "negative"
                if change <= -threshold
                else "neutral"
            )
            all_evidence[metric] = {
                "sentiment": sentiment,
                "favorable_difference": change,
                "threshold": threshold,
                "original_value": _metric(original, metric),
                "candidate_value": _metric(tested, metric),
            }
        # Existing V8.4 verdict contract remains scoped to selection priorities.
        # Extra metrics are only observational, evaluated on that SAME candidate.
        evidence = {metric: all_evidence[metric] for metric in selected}
        folds.append(
            {
                "train_start": str(dates[0].date()),
                "train_end": str(dates[train_end].date()),
                "holdout_start": str(dates[train_end].date()),
                "holdout_end": str(dates[train_end + test_n].date()),
                "selected": winning,
                "weights": dict(zip(asset_returns.columns, map(float, candidates[winning]), strict=False)),
                "training_score": training_scores[winning],
                "candidate_count": len(candidates),
                "train_metrics_selected": train_stats[winning],
                "baseline_holdout_metrics": original,
                "candidate_holdout_metrics": tested,
                "spy_holdout_metrics": spy_stats,
                "candidate_vs_spy": spy_relative,
                "baseline_trading": baseline_costs,
                "candidate_trading": candidate_costs,
                "verdicts": evidence,
                "all_metric_verdicts": all_evidence,
            }
        )
    tally = {}
    for objective in selected:
        counts = {
            key: sum(f["verdicts"][objective]["sentiment"] == key for f in folds)
            for key in ("positive", "negative", "neutral", "unavailable")
        }
        tally[objective] = {
            **counts,
            "materially_positive_fraction": counts["positive"] / len(folds),
        }
    # Descriptive report for the six objectives, not six optimizations.
    all_tally = {}
    for objective in THRESHOLDS:
        counts = {
            key: sum(f["all_metric_verdicts"][objective]["sentiment"] == key for f in folds)
            for key in ("positive", "negative", "neutral", "unavailable")
        }
        all_tally[objective] = {
            **counts,
            "materially_positive_fraction": counts["positive"] / len(folds),
        }
    stable = all(
        tally[k]["positive"] >= max(1, int(np.ceil(0.75 * len(folds))))
        and tally[k]["negative"] == 0
        and tally[k]["unavailable"] == 0
        for k in selected
    )
    newest = folds[-1]
    # Only the last completed fold needs this descriptive source-data audit.
    # It is calculated AFTER selection and is never fed back into the optimizer.
    latest_training_end = train_n + (len(folds) - 1) * step
    latest_train = asset_returns.iloc[:latest_training_end]
    latest_holdout = asset_returns.iloc[latest_training_end : latest_training_end + test_n]
    latest_asset_audit = {}
    for symbol in asset_returns.columns:
        train_asset = _stats(latest_train[symbol], dates[: latest_training_end + 1], rf)
        hold_asset = _stats(
            latest_holdout[symbol],
            dates[latest_training_end : latest_training_end + test_n + 1],
            rf,
        )
        latest_asset_audit[symbol] = {
            "training_cagr": float(train_asset["cagr"]),
            "training_volatility": float(train_asset["annualized_volatility"]),
            "holdout_total_return": float(hold_asset["cumulative_return"]),
        }
    if stable:
        conclusion = "Consistent historical holdout improvement under selected thresholds; not a future guarantee"
        next_step = "Inspect the latest fold's candidate weights, costs and concentration before treating it as a hypothetical allocation"
    elif baseline_compliant:
        conclusion = "No consistently validated improvement across all chosen priorities; do not recommend an allocation change on performance evidence"
        next_step = "Retain the original as a compliant comparison; test other priorities or constraints in new holdouts"
    else:
        conclusion = (
            "No consistent improvement across all priorities. The original portfolio VIOLATES requested limits; "
            "neither retaining it as compliant nor claiming risk improvement is justified."
        )
        next_step = (
            "Review the separately labeled constraint-compliance example, or revise the stated limits; "
            "validate investment results independently before choosing an allocation."
        )
    unvalidated_tail = len(asset_returns) - (train_n + (len(folds) - 1) * step + test_n)
    return {
        "status": "evaluated",
        "fold_count": len(folds),
        "priorities": selected,
        "constraints": {**_rules_out(constraints), "baseline_compliant": baseline_compliant},
        "folds": folds,
        "per_metric_tally": tally,
        "all_metric_tally": all_tally,
        "spy_benchmark_method": (
            "Matched-date, same-currency SPY ETF returns; "
            "candidate includes modeled trading costs, benchmark ETF price history "
            "includes fund expenses but assumes zero extra brokerage trading costs."
            if spy_returns is not None
            else "Not provided"
        ),
        "consistent_improvement_on_all_priorities": stable,
        "conclusion": conclusion,
        "next_step": next_step,
        "latest_fold_candidate": newest["weights"],
        "constraint_compliance_example": compliance_example,
        "latest_asset_audit": latest_asset_audit,
        "unvalidated_trailing_return_periods": unvalidated_tail,
        "most_recent_complete_holdout_end": newest["holdout_end"],
        "available_return_history_end": str(dates[-1].date()),
        "limitations": [
            "Each fold re-optimizes using only prior observed training returns; evaluation windows do not overlap.",
            "Fold portfolios start from the original weights independently, not as continuous realized trading history.",
            "Trading costs use a flat bps assumption; no bid-ask, taxes, market impact, cashflows or financing.",
            "Rebalancing follows matched observed close sessions; no weekend or nontrading-day forward-fill.",
            "Training-only continuous constrained optimizer with deterministic starts; numerical solutions are local and not guaranteed globally optimal.",
            "Threshold-based verdicts are descriptive, not statistical significance or future forecasts.",
        ],
    }


def _rules_out(rules: dict) -> dict:
    return {
        "max_weight": rules["max_weight"],
        "asset_limits": rules["user_bounds"],
        "group_caps": rules["group_caps"],
        "max_turnover": rules["max_turnover"],
        "max_turnover_scope": "Initial one-way change from original allocation to candidate target ONLY; not cumulative holdout rebalancing turnover",
        "rebalance": rules["rebalance"],
        "cost_bps": rules["cost_bps"],
        "train_periods": rules["train_periods"],
        "test_periods": rules["test_periods"],
        "step_periods": rules["step_periods"],
    }
