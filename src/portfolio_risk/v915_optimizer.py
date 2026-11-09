"""V9.15 continuous constrained training-only portfolio search.

Weights are optimized using only the caller's TRAINING returns. The objectives
are deliberately transparent *training proxies*, never out-of-sample verdicts.
Out-of-sample performance and transaction costs are still evaluated by the
existing V8.3 walk-forward engine. Deterministic seeds, no random search or
future-dependent weight selection.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ImportError as exc:
    raise ImportError("Continuous portfolio optimization requires scipy>=1.11. Install the project's updated requirements.") from exc


def _training_proxies(frame: pd.DataFrame, risk_free: float):
    r = frame.to_numpy(dtype=float)
    mu = r.mean(axis=0)
    cov = np.atleast_2d(np.cov(r, rowvar=False, ddof=1))
    # Covariance shrinkage reduces singularity and noise in correlated holdings.
    cov = .85 * cov + .15 * np.diag(np.diag(cov))
    cov += np.eye(len(mu)) * 1e-10
    daily_rf = risk_free / 252.
    # Rolling 21- and 63-day sums approximate the risk of a sustained decline.
    roll = {}
    for horizon in (21, 63):
        if len(r) >= horizon:
            cum = np.vstack((np.zeros((1, r.shape[1])), np.cumsum(r, axis=0)))
            roll[horizon] = (cum[horizon:] - cum[:-horizon])

    def annual_log_growth(w):
        daily = r @ w
        if (daily <= -1).any():
            return -1e4
        return float(252 * np.log1p(daily).mean())

    def total_vol(w):
        return float(np.sqrt(252 * max(w @ cov @ w, 1e-12)))

    def proxy(w, goal):
        daily = r @ w
        excess = daily - daily_rf
        return_annual = 252 * float(excess.mean())
        volatility = total_vol(w)
        if goal == 'cagr':
            return annual_log_growth(w)
        if goal == 'volatility':
            return -volatility
        if goal == 'sharpe':
            return return_annual / max(volatility, .005)
        if goal == 'sortino':
            downside = np.sqrt(252 * max(np.mean(np.minimum(excess, 0.)**2), 1e-9))
            return return_annual / max(downside, .005)
        if goal == 'es95':
            worst = max(1, int(np.ceil(len(daily) * .05)))
            return -float(np.partition(-daily, -worst)[-worst:].mean())
        if goal == 'drawdown':
            # Convex approximation to large persistent losses, not exact MDD.
            # Exact historical maximum drawdown is assessed independently later.
            losses = []
            for horizon, samples in roll.items():
                path_return = samples @ w
                worst = max(1, int(np.ceil(len(path_return) * .05)))
                losses.append(float(np.partition(-path_return, -worst)[-worst:].mean()) / np.sqrt(horizon))
            return -(.7 * max(losses) + .3 * volatility / np.sqrt(252)) if losses else -volatility
        raise ValueError(f'Unknown optimizer objective: {goal}')

    return proxy, annual_log_growth, total_vol


def continuous_candidates(data: pd.DataFrame, reference: np.ndarray, rules: Mapping,
                          risk_free: float, priorities: Mapping[str, float],
                          project) -> dict[str, np.ndarray]:
    """Search a continuous feasible simplex using deterministic SLSQP starts.

    `project` is the engine's own validated group/cap/turnover projection,
    applied after numerical optimization as a final fail-closed check.
    """
    n = len(reference)
    if n < 2:
        return {}
    proxy, annual_growth, volatility = _training_proxies(data, risk_free)
    lo = np.asarray(rules['floors'], dtype=float)
    hi = np.asarray(rules['caps'], dtype=float)
    if np.any(lo > hi) or hi.sum() < 1-1e-8 or lo.sum() > 1+1e-8:
        return {}
    # Deterministic starts; optimize in the continuous feasible set rather
    # than choosing only from templates and their predetermined blends.
    starts = [reference, np.full(n, 1/n),
              np.eye(n)[int(np.argmax(data.mean().to_numpy()))]]
    usable = []
    for s in starts:
        candidate = project(np.asarray(s, dtype=float), reference, rules)
        if candidate is not None and not any(np.allclose(candidate, p) for p in usable):
            usable.append(candidate)
    if not usable:
        return {}
    linear = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.}]
    for _name, indices, cap in rules['groups']:
        linear.append({'type': 'ineq', 'fun': lambda w, ix=indices, mx=cap: mx - w[ix].sum()})
    turnover = rules.get('max_turnover')
    if turnover is not None:
        linear.append({'type': 'ineq',
                       'fun': lambda w: turnover - .5 * np.abs(w-reference).sum()})

    # With large portfolios, evaluate a small representative set of feasible
    # focus corners rather than multiplying each fold by dozens of templates.
    if n <= 8:
        focus_indices = range(n)
    else:
        means = data.mean().to_numpy(dtype=float)
        sigmas = data.std(ddof=1).to_numpy(dtype=float)
        focus_indices = sorted(set([int(np.argmax(reference))] +
                                   list(np.argsort(-means)[:3]) +
                                   list(np.argsort(sigmas)[:3])))
    candidates = {}
    for i in focus_indices:
        ticker = data.columns[i]
        corner = project(np.eye(n)[i], reference, rules)
        if corner is not None and not any(np.allclose(corner, p, atol=1e-6, rtol=0)
                                          for p in candidates.values()):
            candidates[f'feasible_focus_{ticker}'] = corner
    goals = list(priorities)
    # Each goal gets its own optimized solution. Also sample the efficient
    # risk/return frontier, where neither all-growth nor all-min-variance is
    # necessarily the best compromise.
    # Optimize the stated objective, not an undisclosed diversification target.
    # The weight/group caps already enforce diversification. A large generic
    # L2 penalty can dominate annual growth or expected shortfall and choose
    # the wrong candidate for the user's requested goal.
    methods = [(f'continuous_{goal}',
                lambda w, goal=goal: -proxy(w, goal)) for goal in goals]
    for penalty in (.35, 1.25, 3.0, 7.0):
        methods.append((f'frontier_{penalty:g}', lambda w, k=penalty:
                        -(annual_growth(w) - k * volatility(w)**2)))
    # 2 restarts for the direct goal, one for each frontier point.
    for name, objective in methods:
        tested = usable[:2] if name.startswith('continuous_') else usable[:1]
        best = None
        best_score = float('inf')
        for x0 in tested:
            result = minimize(objective, x0, method='SLSQP',
                              bounds=list(zip(lo, hi)), constraints=linear,
                              options={'maxiter': 90, 'ftol': 5e-8, 'disp': False})
            # Never trust an unverified optimizer result to satisfy the rules.
            if not np.isfinite(result.x).all():
                continue
            # Snap numerically indistinguishable SLSQP solutions to stable
            # five-decimal weights, then reproject to preserve exact limits.
            # This also prevents changes in unrelated future array buffers
            # from causing meaningless 1e-7 allocation differences.
            final = project(np.round(result.x, 5), reference, rules)
            if final is None:
                continue
            value = float(objective(final))
            if np.isfinite(value) and value < best_score:
                best, best_score = final, value
        if best is not None and not any(np.allclose(best, p, rtol=0, atol=1e-6) for p in candidates.values()):
            candidates[name] = best
    return candidates
