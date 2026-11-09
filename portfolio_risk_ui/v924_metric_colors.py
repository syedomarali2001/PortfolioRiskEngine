"""Pure presentation styling for existing portfolio statistics.

The statistical values are unchanged. Better/worse is always relative to the
user's original portfolio, not an investment recommendation.
"""
from __future__ import annotations

from typing import Mapping
from math import isfinite
import pandas as pd

MORE_IS_BETTER = {'cagr', 'sharpe', 'sortino'}
LESS_IS_BETTER = {'annualized_volatility', 'drawdown', 'es95'}


def color_proposed_metrics(frame: pd.DataFrame, report: Mapping,
                           metric_keys: tuple[str, ...], light: bool = False) -> pd.io.formats.style.Styler:
    """Tint only Proposed values, leaving original, benchmark and allocation neutral."""
    proposed = report['current_training_metrics']
    original = report['original_training_metrics']

    def paint_row(row: pd.Series) -> list[str]:
        key = metric_keys[int(row.name)]
        try:
            left = float(original[key])
            right = float(proposed[key])
        except (KeyError, TypeError, ValueError):
            return ['' for _ in row.index]
        if not isfinite(left) or not isfinite(right):
            return ['' for _ in row.index]
        if key in MORE_IS_BETTER:
            improvement = right - left
        elif key in LESS_IS_BETTER:
            improvement = left - right
        else:
            raise ValueError(f'No verified direction for metric: {key}')
        if abs(improvement) < 1e-8:
            style = ('color: #52657B; font-weight: 600;' if light else 'color: #AAB4C5; font-weight: 600;')
        elif improvement > 0:
            style = ('color: #075737; background-color: #D6F6E7; font-weight: 700;' if light else 'color: #86EFAC; background-color: #17432F; font-weight: 700;')
        else:
            style = ('color: #9B1B36; background-color: #FCE2E6; font-weight: 700;' if light else 'color: #FDA4AF; background-color: #4B222A; font-weight: 700;')
        return [style if col == 'Proposed' else '' for col in row.index]

    return frame.style.apply(paint_row, axis=1)
