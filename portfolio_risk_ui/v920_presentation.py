"""V9.20: plain-language allocation settings and safe custom editor seed values.

No portfolio return, risk statistic, or FX rate is invented here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .fx_display import display_portfolio_value

# Displayed order sets the recommended default: quarterly TARGET rebalancing.
# All five cadences already exist in the underlying V8.3 simulator.
REBALANCE_CHOICES = {
    'Quarterly: restore target weights (default)': 'quarterly',
    'Yearly: restore target weights': 'annually',
    'Monthly: restore target weights': 'monthly',
    'Daily: constant-weight research model': 'daily',
    'Buy and hold: allow weights to drift': 'buy_hold',
}

_REBALANCE_HELP = {
    'quarterly': 'Every three months the simulation trades back to the target percentages. Weights still drift with prices between rebalances. This is the default; simulated trades may incur costs.',
    'annually': 'Once each year the simulation restores the target percentages. Weights can drift during the year; costs are charged on the modeled trades.',
    'monthly': 'Once a month the simulation restores target percentages. This keeps weights closer to target, but usually involves more trading and potential costs.',
    'daily': 'Research-only constant-weight approximation: rebalance each trading day. Weights are reset before each daily return, but trading may be frequent and costly; taxes and market impact are not included.',
    'buy_hold': 'Buy and hold: establish the initial percentages once and make no later rebalancing trades. Winners grow into larger weights and laggards become smaller; the target percentages do NOT stay fixed.',
}


def rebalance_explanation(cadence: str) -> str:
    if cadence not in _REBALANCE_HELP:
        raise ValueError('Unsupported rebalancing schedule')
    return _REBALANCE_HELP[cadence]


def custom_start_weights(holdings: list[dict], base: str, usd_cad: str | None = None) -> dict[str, float]:
    """Derive accurate starting allocation fractions from entered holdings.

    Same-currency holdings need no quote. Mixed currency requires an explicit,
    valid FX conversion from the existing dated quote service (UI fetches it).
    """
    result = display_portfolio_value(holdings, base, usd_cad)
    return {sym: value/100 for sym,value in result.shares.items()}


def editable_percentages(weights: Mapping[str, float]) -> dict[str, float]:
    """Round fractional shares to editable % at two decimals, totaling 100.00.

    The largest holding absorbs any rounding remainder. The source values must
    be nonnegative and sum to one: invalid inputs are never silently repaired.
    """
    if not weights:
        raise ValueError('At least one investment is required')
    vals = {str(symbol): float(weight) for symbol, weight in weights.items()}
    if (any(not math.isfinite(v) or v < -1e-10 for v in vals.values())
            or not math.isclose(sum(vals.values()), 1., rel_tol=0, abs_tol=1e-5)):
        raise ValueError('Weights must be nonnegative fractions totaling 100%')
    rounded = {sym: round(max(0., v)*100, 2) for sym,v in vals.items()}
    biggest = max(vals, key=vals.get)
    rounded[biggest] = round(rounded[biggest] + (100 - sum(rounded.values())), 2)
    if not math.isclose(sum(rounded.values()), 100., rel_tol=0, abs_tol=1e-8):
        raise ValueError('Could not display an exact 100% allocation')
    return rounded
