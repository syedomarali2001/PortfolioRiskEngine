"""V8 matched-date CAD vs USD risk/return comparison, no currency forecast."""
from __future__ import annotations

from typing import Mapping
import pandas as pd
import numpy as np

from portfolio_risk.historical_fx import convert_price_history_to_base
from portfolio_risk.performance import _check_price_frame
from portfolio_risk.v8_insights import BENCHMARKS, InsightError, _stats, _weight_map


def compare_reporting_currencies(
    prices_in_base: Mapping[str,pd.Series], weights: Mapping[str,float],
    benchmarks_in_base: Mapping[str,pd.Series], usd_cad: pd.Series,
    *, base_currency: str, rf: float=0., max_fx_age_days: int=5,
    min_observations: int=60,
) -> dict:
    """Both currencies evaluated on EXACTLY the same observed dates.

    Trading currency and economic FX exposure differ. This only describes how
    converting the same synthetic constant-mix NAV alters reported returns.
    """
    if base_currency not in {'CAD','USD'}:
        raise InsightError('Base currency must be CAD or USD')
    if set(benchmarks_in_base)!=set(BENCHMARKS):
        raise InsightError('SPY/RSP benchmark prices both required')
    names=list(prices_in_base)
    w=_weight_map(weights,names)
    other='USD' if base_currency=='CAD' else 'CAD'
    native={**{f"A:{k}":v for k,v in prices_in_base.items()},
            **{f"B:{k}":v for k,v in benchmarks_in_base.items()}}
    adjusted={}
    for name,series in native.items():
        adjusted[name]=convert_price_history_to_base(
            series,quote_currency=base_currency,base_currency=other,
            usd_cad_prices=usd_cad,max_fx_age_days=max_fx_age_days)
    # Intersect both valuation bases, not just their last-price calendar dates.
    frame=pd.concat({**{f"{base_currency}:{s}":v for s,v in native.items()},
                     **{f"{other}:{s}":v for s,v in adjusted.items()}},axis=1,join='inner').sort_index()
    frame=_check_price_frame(frame)
    if len(frame)<min_observations:
        raise InsightError('Insufficient jointly observed dates for matched FX view')
    pct=frame.pct_change(fill_method=None).iloc[1:]
    metrics={}
    for currency in (base_currency,other):
        synthetic=pd.Series(pct[[f"{currency}:A:{n}" for n in names]].to_numpy()@w,index=pct.index)
        metrics[currency]={"portfolio":_stats(synthetic,frame.index,rf),
                           "SPY":_stats(pct[f"{currency}:B:SPY"],frame.index,rf),
                           "RSP":_stats(pct[f"{currency}:B:RSP"],frame.index,rf)}
    return {"status":"evaluated","start":str(frame.index[0].date()),
            "end":str(frame.index[-1].date()),"return_periods":len(pct),
            "reporting_currency_metrics":metrics,
            "limitations":["Constant weights are hypothetically rebalanced DAILY in each reporting currency.",
                           "This converts the same economic securities, not a hedge overlay or FX hedge recommendation.",
                           "Trading denomination does not measure currency of underlying company revenues.",
                           "Currency comparison uses historical USD/CAD and a strictly matched subset of dates."]}
