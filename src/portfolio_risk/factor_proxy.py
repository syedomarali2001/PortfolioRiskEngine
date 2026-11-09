"""V7 descriptive market/style PROXY sensitivity, not causal factor exposure.

Regresses buy-and-hold portfolio simple returns against a single investable ETF
proxy at a time; regressors are NOT orthogonal and are not Fama-French factors.
The result measures in-sample historical co-movement, not an investable alpha.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np
import pandas as pd


class FactorProxyError(ValueError):
    """Price history invalid or insufficient for descriptive proxy regression."""


@dataclass(frozen=True)
class ProxySensitivity:
    symbol: str
    first_date: str
    last_date: str
    observations: int
    beta: float | None
    correlation: float | None
    r_squared: float | None
    alpha_annualized_arithmetic: float | None


def analyze_factor_proxies(
    portfolio_values: pd.Series, proxy_prices: Mapping[str, pd.Series],
    *, min_observations: int = 60,
) -> dict[str, ProxySensitivity]:
    if not isinstance(min_observations, int) or min_observations < 3:
        raise FactorProxyError("min_observations must be an integer >=3")
    if not proxy_prices:
        raise FactorProxyError("At least one ETF proxy is required")
    results = {}
    for sym, price in proxy_prices.items():
        if not isinstance(portfolio_values, pd.Series) or not isinstance(price, pd.Series):
            raise FactorProxyError("Portfolio and proxy data must be pandas Series")
        frame = pd.concat({"portfolio": portfolio_values, "proxy": price}, axis=1, join="inner")
        frame = frame.dropna().sort_index()
        if len(frame) < min_observations:
            raise FactorProxyError(f"{sym}: only {len(frame)} shared price dates (need {min_observations})")
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
            raise FactorProxyError(f"{sym}: invalid time index")
        vals = frame.to_numpy(dtype=float)
        if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
            raise FactorProxyError(f"{sym}: all aligned prices must be positive and finite")
        returns = frame.pct_change(fill_method=None).iloc[1:]
        y = returns["portfolio"].to_numpy(dtype=float)
        x = returns["proxy"].to_numpy(dtype=float)
        var = float(np.var(x, ddof=1))
        yvar = float(np.var(y, ddof=1))
        if var <= 1e-20 or yvar <= 1e-20:
            beta = corr = r2 = alpha = None
        else:
            beta = float(np.cov(x, y, ddof=1)[0, 1] / var)
            corr = float(np.corrcoef(x, y)[0, 1])
            corr = min(1.0, max(-1.0, corr))
            r2 = corr**2
            alpha = float((y.mean() - beta*x.mean()) * 252)  # arithmetic, not compounding
        results[sym] = ProxySensitivity(sym, frame.index[0].date().isoformat(),
                                        frame.index[-1].date().isoformat(), len(returns),
                                        beta, corr, r2, alpha)
    return results


def factor_proxy_as_dict(items: Mapping[str, ProxySensitivity]) -> dict:
    from dataclasses import asdict
    return {
        "method": "One-at-a-time OLS on aligned historical common-close simple returns",
        "proxies": {sym: asdict(m) for sym, m in items.items()},
        "limitations": [
            "Investable ETFs are imperfect, correlated STYLE/MARKET PROXIES, not isolated academic factors.",
            "Each beta is a separate univariate regression; DO NOT add or interpret as causal exposures.",
            "Alpha is an arithmetic regression intercept annualized x252, NOT proven trading alpha.",
            "Past co-movement does not reliably forecast future factor loadings.",
            "Historical portfolio is a hypothetical strategy using today's asset universe.",
        ],
    }
