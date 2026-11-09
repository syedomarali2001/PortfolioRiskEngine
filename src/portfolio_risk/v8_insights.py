"""V8 explainable portfolio diagnostics and pre-registered, holdout-tested allocations.

Pure analytical module: no network, UI, or hindsight-based historical holdings.
All portfolios use a synthetic DAILY constant-mix strategy on *shared observed*
close dates. All inputs must be pre-converted into the SAME base currency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from typing import Mapping

import numpy as np
import pandas as pd

from portfolio_risk.construction import _capped_simplex_projection, _minimum_variance
from portfolio_risk.performance import _check_price_frame, _metrics
from portfolio_risk.risk import _asset_attribution

BENCHMARKS = ("SPY", "RSP")
OBJECTIVES = ("sharpe", "sortino", "volatility", "drawdown", "cagr", "es95")
DEFINITIONS = {
    "sharpe": "Annualized excess return per unit of total return volatility (higher is better).",
    "sortino": "Annualized excess return per unit of downside deviation (higher is better).",
    "volatility": "Annualized standard deviation of periodic returns; lower means smaller historical swings.",
    "drawdown": "Largest historical peak-to-trough decline, reported as a positive loss fraction (lower is better).",
    "cagr": "Geometrically annualized historical growth rate (higher is better).",
    "es95": "Average loss in the worst 5% of daily outcomes; historical Expected Shortfall (lower is better).",
    "calmar": "CAGR divided by maximum drawdown depth. Undefined for zero drawdown.",
}
# Deliberately conservative, *materiality rules*, NOT industry certifications.
THRESHOLDS = {"sharpe": .20, "sortino": .30, "volatility": .02,
              "drawdown": .05, "cagr": .02, "es95": .005}


class InsightError(ValueError):
    pass


def _finite(value: float | None) -> bool:
    return value is not None and isfinite(value)


def _weight_map(raw: Mapping[str, float], names: list[str], *, positive: bool = False) -> np.ndarray:
    if set(raw) != set(names):
        raise InsightError("Allocation must give exactly one weight for every portfolio ticker")
    w = np.asarray([raw[n] for n in names], dtype=float)
    if not np.isfinite(w).all() or np.any(w < 0) or (positive and np.any(w <= 0)):
        raise InsightError("Allocation weights must be finite and nonnegative")
    if not np.isclose(w.sum(), 1., atol=1e-8, rtol=0):
        raise InsightError("Allocation weights must sum to 100%; never silently renormalize")
    return w


def _stats(returns: pd.Series, dates: pd.DatetimeIndex, rf: float) -> dict:
    values = _metrics(returns.astype(float), dates, rf)
    output = asdict(values)
    output["drawdown"] = -output.pop("maximum_drawdown")
    daily_losses = -returns.to_numpy(dtype=float)
    cut = float(np.quantile(daily_losses, .95))
    output["es95"] = max(0., float(np.mean(daily_losses[daily_losses >= cut])))
    output["calmar"] = (output["cagr"] / output["drawdown"] if output["drawdown"] > 1e-12 else None)
    return output


def _metric(stats: dict, objective: str) -> float | None:
    """Resolve user-facing objective names to the canonical metric storage keys."""
    if objective == "volatility":
        return stats.get("annualized_volatility", stats.get("volatility"))
    return stats.get(objective)


def _score(stats: dict, objective: str) -> float:
    value = _metric(stats, objective)
    if not _finite(value):
        return -1e15
    return -float(value) if objective in {"volatility", "drawdown", "es95"} else float(value)


def _details(port: dict, benchmark: dict, metric: str, symbol: str, n: int) -> dict:
    lhs, rhs = _metric(port, metric), _metric(benchmark, metric)
    threshold = THRESHOLDS[metric]
    if not _finite(lhs) or not _finite(rhs):
        sentiment = "unavailable"
        evidence = f"{metric}: unavailable or undefined for this sample."
        diff = None
    else:
        direction = -1 if metric in {"volatility", "drawdown", "es95"} else 1
        diff = direction * (float(lhs) - float(rhs))
        sentiment = ("unavailable" if n < 126 else
                     "positive" if diff >= threshold else
                     "negative" if diff <= -threshold else "neutral")
        comparison = "better" if diff > 0 else "worse" if diff < 0 else "equal"
        evidence = (f"Portfolio {metric} {lhs:.4f} vs {symbol} {rhs:.4f}; "
                    f"{abs(diff):.4f} {comparison} (favorable direction). "
                    f"Materiality threshold {threshold:.4f}.")
        if n < 126:
            evidence += " Fewer than 126 common return periods; sentiment withheld."
    return {"metric": metric, "benchmark": symbol, "sentiment": sentiment,
            "portfolio_value": lhs, "benchmark_value": rhs,
            "favorable_difference": diff, "materiality_threshold": threshold,
            "evidence": evidence, "explanation": DEFINITIONS[metric],
            "periods": n, "rule": "Positive/negative only for material difference with >=126 returns"}


def _candidate_weights(train: pd.DataFrame, reference: np.ndarray, cap: float) -> dict[str, np.ndarray]:
    n = len(reference)
    candidate = {"input": reference, "equal_weight": np.ones(n) / n}
    if n == 1:
        return {"input": reference}
    vol = train.std(ddof=1).to_numpy(dtype=float)
    inv = 1. / np.maximum(vol, 1e-6)
    candidate["inverse_volatility"] = _capped_simplex_projection(inv / inv.sum(), cap)
    cov = np.atleast_2d(train.cov().to_numpy(dtype=float))
    candidate["minimum_variance"] = _minimum_variance(cov, max_weight=cap, shrinkage=.15)
    # Diversified, transparent candidate set. These are training-time decisions,
    # not an exhaustive search advertised as a global optimum.
    for alpha in (.25, .5, .75):
        mixed = alpha * candidate["minimum_variance"] + (1-alpha) * candidate["equal_weight"]
        candidate[f"minvar_equal_{int(alpha*100)}"] = mixed
        mixed = alpha * candidate["inverse_volatility"] + (1-alpha) * reference
        if mixed.max() <= cap + 1e-9:
            candidate[f"inverse_input_{int(alpha*100)}"] = mixed
    return {name: w for name, w in candidate.items() if np.isfinite(w).all() and abs(w.sum()-1)<1e-7}


def _preferences(objectives: Mapping[str, float] | None) -> dict[str, float]:
    selected = dict(objectives or {"sharpe": 1.})
    if not selected or any(k not in OBJECTIVES or not _finite(v) or v <= 0 for k, v in selected.items()):
        raise InsightError(f"Priorities must be positive weights for {', '.join(OBJECTIVES)}")
    total = sum(selected.values())
    return {k: float(v) / total for k, v in selected.items()}


def _recommendations(asset_rets: pd.DataFrame, dates: pd.DatetimeIndex, reference: np.ndarray,
                     rf: float, priorities: dict[str, float], cap: float) -> dict:
    if not .0 < cap <= 1:
        raise InsightError("max_weight must be between 0 and 1")
    if len(reference) == 1:
        return {"status":"unavailable", "reason":"Single-asset portfolio: no allocation trade-off to optimize"}
    if cap * len(reference) < 1-1e-9:
        raise InsightError("max_weight cap cannot accommodate all assets")
    count = len(asset_rets)
    split = int(count * .7)
    if split < 80 or count - split < 30:
        return {"status": "unavailable", "reason": "Need at least 80 training and 30 later evaluation returns"}
    train = asset_rets.iloc[:split]
    test = asset_rets.iloc[split:]
    names = list(asset_rets)
    candidates = _candidate_weights(train, reference, cap)
    scored = {}
    for name, weights in candidates.items():
        tr = pd.Series(train.to_numpy() @ weights, index=train.index)
        scored[name] = _stats(tr, dates[:split+1], rf)
    # rank normalized objectives on TRAINING PERIOD ONLY; no future-based selection
    scores = {name: 0. for name in candidates}
    for objective, importance in priorities.items():
        vals = np.asarray([_score(scored[name], objective) for name in candidates])
        lo, hi = vals.min(), vals.max()
        points = (vals-lo)/(hi-lo) if hi-lo > 1e-12 else np.full(len(vals), .5)
        for name, value in zip(candidates, points):
            scores[name] += importance * float(value)
    # If the score ties, favor existing allocation and avoid gratuitous changes.
    ordered = sorted(candidates, key=lambda s: (-scores[s], s != "input", s))
    winner = ordered[0]
    # Expose all TRAINING candidates and their Pareto frontier, while showing
    # later outcomes only for the preselected winner and original input.
    frontier=[]
    for candidate in candidates:
        dominated = any(
            other != candidate and
            all(_score(scored[other],obj) >= _score(scored[candidate],obj) - 1e-12 for obj in priorities) and
            any(_score(scored[other],obj) > _score(scored[candidate],obj) + 1e-12 for obj in priorities)
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    output = {}
    for name in sorted(set(("input", winner))):
        weights = candidates[name]
        tr = pd.Series(test.to_numpy() @ weights, index=test.index)
        output[name] = {"weights": {s: float(v) for s, v in zip(names, weights)},
                        "training_metrics": scored[name],
                        "holdout_metrics": _stats(tr, dates[split:], rf),
                        "training_objective_score": scores[name]}
    delta = {k: _metric(output[winner]["holdout_metrics"], k) - _metric(output["input"]["holdout_metrics"], k)
             for k in priorities if _finite(_metric(output[winner]["holdout_metrics"], k)) and
             _finite(_metric(output["input"]["holdout_metrics"], k))}
    holdout_verdicts = {}
    for objective in priorities:
        candidate_value = _metric(output[winner]["holdout_metrics"], objective)
        original_value = _metric(output["input"]["holdout_metrics"], objective)
        threshold = THRESHOLDS[objective]
        if not _finite(candidate_value) or not _finite(original_value):
            verdict, difference = "unavailable", None
            evidence = f"Holdout {objective}: undefined for candidate and/or original portfolio; comparison withheld."
        else:
            difference = _score(output[winner]["holdout_metrics"], objective) - _score(output["input"]["holdout_metrics"], objective)
            verdict = "positive" if difference >= threshold else "negative" if difference <= -threshold else "neutral"
            evidence = (f"Holdout {objective}: candidate {candidate_value}, original {original_value}; "
                        f"favorable difference {difference:.4f} (threshold {threshold:.4f})")
        holdout_verdicts[objective] = {"sentiment": verdict, "favorable_difference": difference,
                                       "threshold": threshold, "evidence": evidence}
    return {"status": "evaluated", "selected": winner,
            "holdout_objective_verdicts":holdout_verdicts,
            "outperforming_holdout_on_all_priorities":bool(holdout_verdicts) and all(
                x['sentiment']=='positive' for x in holdout_verdicts.values()),
            "objective_weights": priorities, "candidate_count": len(candidates),
            "training_candidate_scorecard": {name:{"weights":dict(zip(names, map(float,w))),
                                                   "objective_score":scores[name],
                                                   "training_metrics":scored[name]}
                                            for name,w in candidates.items()},
            "training_pareto_frontier": frontier,
            "train_start": dates[0].date().isoformat(),
            "train_end": dates[split].date().isoformat(),
            "holdout_start": dates[split].date().isoformat(),
            "holdout_end": dates[-1].date().isoformat(),
            "candidates_selected": output, "holdout_raw_metric_differences_vs_input": delta,
            "selection_rule": "Min-max normalized preference scores on training data only; later holdout never used to select",
            "limitations": ["Historical, hypothetical DAILY constant-weight rebalancing; no fees, spreads, taxes or execution.",
                            "Only a small deterministic set of candidate allocations; not a global optimum.",
                            "A single chronological holdout does not prove future returns; training still overfits possible trends.",
                            "Return weighting assumes daily common close dates; weekend trading is not forward filled."]}


def _event_replay(returns: pd.Series) -> list[dict]:
    windows = (("2008 financial crisis", "2007-10-01", "2009-03-31"),
               ("2020 pandemic selloff", "2020-02-19", "2020-03-31"),
               ("2022 tightening", "2022-01-01", "2022-12-31"))
    result = []
    for label, start, end in windows:
        slice_ = returns.loc[start:end]
        # Do not present a partial crisis interval as a full event replay.
        if returns.index.min() > pd.Timestamp(start) or returns.index.max() < pd.Timestamp(end):
            result.append({"event": label, "status": "unavailable", "reason": "Incomplete historical event window"})
        elif len(slice_) < 10:
            result.append({"event": label, "status": "unavailable", "reason": "Insufficient common close dates"})
        else:
            wealth = np.r_[1., np.cumprod(1+slice_.to_numpy())]
            max_dd = float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
            result.append({"event": label, "status": "evaluated", "periods": len(slice_),
                           "total_return": float(wealth[-1]-1), "maximum_drawdown": max_dd,
                           "first_return_date": str(slice_.index[0].date()),
                           "last_return_date": str(slice_.index[-1].date())})
    return result


def _monte_carlo(returns: pd.Series, paths: int, horizon: int, seed: int) -> dict:
    if paths < 20 or paths > 5000 or not 10 <= horizon <= 1260:
        raise InsightError("Monte Carlo requires 20..5000 paths and 10..1260 periods")
    if len(returns) < 60:
        return {"status": "unavailable", "reason": "Less than 60 common returns"}
    rng = np.random.default_rng(seed)
    sample = returns.to_numpy(dtype=float)
    block = min(5, len(sample))
    starts = rng.integers(0, len(sample)-block+1, size=(paths, (horizon+block-1)//block))
    picks = (starts[..., None] + np.arange(block)).reshape(paths, -1)[:, :horizon]
    simulated = sample[picks]
    paths_wealth = np.cumprod(1+simulated, axis=1)
    terminal = paths_wealth[:, -1] - 1
    worst = np.min(paths_wealth / np.maximum.accumulate(np.c_[np.ones(paths), paths_wealth], axis=1)[:,1:] - 1, axis=1)
    return {"status": "evaluated", "paths": paths, "horizon_periods": horizon,
            "seed": seed, "block_length": block, "terminal_return_p05_p50_p95": [float(v) for v in np.quantile(terminal,[.05,.5,.95])],
            "drawdown_p05_p50_p95": [float(v) for v in np.quantile(worst,[.05,.5,.95])],
            "limitations": "Block bootstrap repeats historically observed return blocks; not a forecast or stress-test guarantee"}


def analyze_insights(
    asset_prices: Mapping[str, pd.Series], weights: Mapping[str, float],
    benchmarks: Mapping[str, pd.Series], *, base_currency: str = "CAD",
    annual_risk_free_rate: float = 0., scenarios: Mapping[str, Mapping[str, float]] | None = None,
    priorities: Mapping[str, float] | None = None, max_weight: float = .7,
    min_observations: int = 60, mc_paths: int = 300, mc_horizon: int = 252,
    seed: int = 42,
    robustness: Mapping | None = None,
    three_strategies: bool = False,
) -> dict:
    """One shared calendar for portfolio, scenarios and BOTH ETF benchmark proxies.

    Data is required to be pre-converted to base currency. No hidden substitution
    or silently dropped assets/benchmark prices. Use historical snapshot-weight
    backtests only as research scenarios, never as actual account history.
    """
    if base_currency not in {"CAD", "USD"}:
        raise InsightError("base_currency must be CAD or USD")
    if not isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1:
        raise InsightError("Invalid risk-free rate")
    if not isinstance(min_observations, int) or min_observations < 30:
        raise InsightError("min_observations must be >=30")
    if set(benchmarks) != set(BENCHMARKS):
        raise InsightError("Both SPY and RSP benchmark prices are required")
    if not asset_prices:
        raise InsightError("No assets to analyze")
    names = list(asset_prices)
    reference = _weight_map(weights, names)
    scenario_weights = {name: _weight_map(w, names) for name, w in (scenarios or {}).items()}
    if any(not name.strip() or name == "input" for name in scenario_weights):
        raise InsightError("Scenario names must be nonempty and different from input")
    try:
        named = {f"A:{sym}": series for sym, series in asset_prices.items()}
        named.update({f"B:{sym}": benchmarks[sym] for sym in BENCHMARKS})
        frame = _check_price_frame(pd.concat(named, axis=1, join="inner").sort_index())
    except (ValueError, TypeError) as exc:
        raise InsightError(f"Invalid or missing common price data: {exc}") from exc
    if len(frame) < min_observations:
        raise InsightError(f"Only {len(frame)} shared observed closes (need {min_observations})")
    dates = frame.index
    ret = frame.pct_change(fill_method=None).iloc[1:]
    asset = ret[[f"A:{s}" for s in names]].copy()
    asset.columns = names
    portfolio = pd.Series(asset.to_numpy() @ reference, index=ret.index, name="portfolio")
    original_stats = _stats(portfolio, dates, annual_risk_free_rate)
    benchmark_stats = {s: _stats(ret[f"B:{s}"], dates, annual_risk_free_rate) for s in BENCHMARKS}
    alerts = [_details(original_stats, benchmark_stats[s], k, s, len(ret))
              for k in OBJECTIVES for s in BENCHMARKS]
    # Relative portfolio sensitivities on exactly the same observations.
    relative = {}
    for symbol in BENCHMARKS:
        b = ret[f"B:{symbol}"]
        var = float(b.var(ddof=1))
        cov = float(portfolio.cov(b))
        active = portfolio-b
        te = float(active.std(ddof=1))
        relative[symbol]={"beta":cov/var if var>1e-14 else None,
                          "correlation":float(portfolio.corr(b)) if var>1e-14 and portfolio.std(ddof=1)>1e-14 else None,
                          "tracking_error_annualized":te*np.sqrt(252),
                          "information_ratio":float(active.mean())/te*np.sqrt(252) if te>1e-14 else None,
                          "average_active_return_annualized_arithmetic":float(active.mean())*252,
                          "method":"Matched simple returns; annual active return is arithmetic approximation"}
    # Distinguish economic implications (positive/negative) from statistical
    # confidence. Relative differences are descriptive; no p-value is invented.
    attribution, total_vol, diversification = _asset_attribution(asset, dict(zip(names, reference)))
    comparisons = {}
    for label, w in scenario_weights.items():
        returns = pd.Series(asset.to_numpy() @ w, index=asset.index)
        metrics = _stats(returns, dates, annual_risk_free_rate)
        comparisons[label] = {"weights": dict(zip(names, map(float, w))),
                              "metrics": metrics,
                              "vs_input": {k: {"difference": _metric(metrics,k)-_metric(original_stats,k),
                                                "favorable": _score(metrics,k)-_score(original_stats,k)}
                                           for k in OBJECTIVES if _finite(_metric(metrics,k)) and _finite(_metric(original_stats,k))},
                              "method": "Hypothetical constant-weight historical scenario; ex-post, not a prediction"}
    priorities_ = _preferences(priorities)
    suggestion = _recommendations(asset, dates, reference, annual_risk_free_rate, priorities_, max_weight)
    date_windows = {}
    for years in (1, 3, 5):
        beginning = dates[-1] - pd.DateOffset(years=years)
        part = portfolio.loc[beginning:]
        if len(part) < 30 or dates[0] > beginning:
            date_windows[f"{years}y"] = {"status": "unavailable", "reason": "Insufficient full history"}
        else:
            pos = dates.get_loc(part.index[0])
            date_windows[f"{years}y"] = {"status": "evaluated", "metrics": _stats(part, dates[pos-1:], annual_risk_free_rate),
                                         "benchmarks":{s:_stats(ret[f"B:{s}"].loc[beginning:], dates[pos-1:], annual_risk_free_rate)
                                                       for s in BENCHMARKS},
                                         "start": str(dates[pos-1].date()), "end": str(dates[-1].date())}
    diag = {"positive": [a for a in alerts if a["sentiment"] == "positive"],
            "negative": [a for a in alerts if a["sentiment"] == "negative"],
            "neutral": [a for a in alerts if a["sentiment"] == "neutral"],
            "unavailable": [a for a in alerts if a["sentiment"] == "unavailable"]}
    from portfolio_risk.v83_robustness import analyze_robustness
    from portfolio_risk.v83_actions import portfolio_actions
    rolling = (analyze_robustness(asset,dates,reference,priorities=priorities_,
                                  rules=robustness,rf=annual_risk_free_rate,
                                  benchmark_returns={"SPY":ret["B:SPY"]})
               if robustness is not None and not three_strategies else None)
    result = {"version": "8.3" if rolling is not None else "8.0", "method": "hypothetical DAILY constant-mix; matched observed closes; no taxes/fees",
            "base_currency": base_currency, "first_date": str(dates[0].date()),
            "last_date": str(dates[-1].date()), "shared_close_dates": len(dates),
            "return_periods": len(ret), "risk_free_annual": annual_risk_free_rate,
            "input_weights": dict(zip(names, map(float, reference))),
            "portfolio": original_stats, "benchmarks": benchmark_stats,
            "relative_to_benchmarks":relative,
            "benchmark_proxies": {"SPY":"S&P 500 market-cap-weight ETF total-return proxy",
                                  "RSP":"S&P 500 equal-weight ETF total-return proxy"},
            "diagnostics": diag,
            "diagnosis_summary": {
                "strengths": sorted(diag["positive"], key=lambda x:-x["favorable_difference"])[:3],
                "weaknesses": sorted(diag["negative"], key=lambda x:x["favorable_difference"])[:3],
                "neutral_count":len(diag["neutral"]),
                "unavailable_count":len(diag["unavailable"]),
                "interpretation":"Separate strengths and weaknesses; no unsupported single overall score or vague mixed-signal verdict"
            },
            "diagnostic_policy": {"thresholds": THRESHOLDS,
                                  "min_periods_for_sentiment":126,
                                  "interpretation":"Descriptive materiality rules, NOT statistical significance or buy/sell advice"},
            "risk_contribution": [asdict(x) for x in attribution],
            "portfolio_volatility_reconstructed": total_vol,
            "diversification_ratio": diversification,
            "user_scenarios": comparisons,
            "historical_strategy_selection": suggestion,
            "historical_events": {"portfolio":_event_replay(portfolio),
                                  **{s:_event_replay(ret[f"B:{s}"]) for s in BENCHMARKS}},
            "starting_date_sensitivity": date_windows,
            "block_bootstrap_monte_carlo": _monte_carlo(portfolio, mc_paths, mc_horizon, seed),
            "warnings": (["Fewer than 252 returns; annualized metrics are unstable"] if len(ret)<252 else []) +
                        ["Today's portfolio allocations applied retrospectively, not actual investor P&L.",
                         "SPY/RSP are ETF proxies, not fee-free index series; currency conversion must be matched.",
                         "Positive and negative alerts indicate observed differences, never certainty about future returns.",
                         "Joint calendar excludes non-shared closes; weekends and missing sessions are NOT forward filled.",
                         "Selecting among backtests can overfit; favor holdout results and inspect drawdown trade-offs."],
            "metric_definitions": DEFINITIONS}

    if rolling is not None:
        result["rolling_robustness"] = rolling
        result["legacy_strategy_disclaimer"] = (
            "historical_strategy_selection is the original V8 single 70/30 split: "
            "unconstrained, daily constant-mix, zero trading costs. "
            "Use rolling_robustness for V8.3 limit-aware and cost-aware evaluation.")
    result["actionable_diagnostics"] = portfolio_actions(result, rolling)
    if three_strategies:
        from portfolio_risk.v918_three_strategies import analyze_three_strategies
        from portfolio_risk.v83_robustness import _validated_rules, _simulated_returns
        effective_rules = robustness or {"max_weight": max_weight}
        result["three_strategy_analysis"] = analyze_three_strategies(
            asset, dates, reference,
            {symbol: ret[f"B:{symbol}"] for symbol in BENCHMARKS},
            rules=effective_rules,
            rf=annual_risk_free_rate,
        )
        # V8's standalone what-if table is a daily constant-weight comparison.
        # Strategy Lab must instead evaluate manual allocations using precisely
        # the SAME selected cadence and trade-cost assumptions as its optima.
        if scenario_weights:
            consistent_rules = _validated_rules(names, reference, effective_rules)
            original_m = result['three_strategy_analysis']['strategies']['growth']['original_training_metrics']
            for label, w in scenario_weights.items():
                stream, trades = _simulated_returns(
                    asset, w, rebalance=consistent_rules['rebalance'],
                    cost_bps=consistent_rules['cost_bps'], starting_weights=reference)
                new_metrics = _stats(stream, dates, annual_risk_free_rate)
                outcome = comparisons[label]
                outcome['metrics'] = new_metrics
                outcome['vs_input'] = {
                    k: {'difference': _metric(new_metrics,k)-_metric(original_m,k),
                        'favorable': _score(new_metrics,k)-_score(original_m,k)}
                    for k in OBJECTIVES
                    if _finite(_metric(new_metrics,k)) and _finite(_metric(original_m,k))
                }
                outcome['method'] = (f"Historical manual allocation with {consistent_rules['rebalance']} "
                                     "rebalance and explicit one-way trading costs. Not an out-of-sample claim.")
                outcome['trading'] = trades
    return result
