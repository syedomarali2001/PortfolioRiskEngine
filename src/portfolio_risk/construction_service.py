"""V6 orchestration adapter: reuse V4's validated adjusted prices and FX."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from portfolio_risk.backtest_service import BacktestRun
from portfolio_risk.construction import ConstructionError, ConstructionReport, run_construction


@dataclass(frozen=True)
class ConstructionRun:
    report: ConstructionReport
    market_data: dict[str, dict]
    quality_warnings: tuple[str, ...]


def run_construction_from_backtest(
    backtest_run: BacktestRun,
    *,
    frequency: str = "quarterly",
    lookback: int = 252,
    momentum_skip: int = 21,
    top_k: int = 2,
    max_weight: float = 0.6,
    shrinkage: float = 0.15,
    evaluation_start: date | None = None,
    methods: tuple[str, ...] | None = None,
) -> ConstructionRun:
    backtest = backtest_run.report
    prices = backtest_run.prices_in_base
    if prices is None:
        raise ConstructionError("V4 backtest run did not include validated historical prices")
    names = list(backtest.target_weights)
    from portfolio_risk.construction import METHODS

    report = run_construction(
        {s: prices[s] for s in names},
        backtest.target_weights,
        capital=backtest.initial_capital,
        fee_bps=backtest.fee_bps,
        base_currency=backtest.base_currency,
        annual_risk_free=backtest.risk_free_annual,
        frequency=frequency,
        methods=methods or METHODS,
        lookback=lookback,
        momentum_skip=momentum_skip,
        top_k=top_k,
        max_weight=max_weight,
        shrinkage=shrinkage,
        evaluation_start=evaluation_start,
        benchmark_prices_in_base=(
            prices.get(backtest.benchmark_symbol) if backtest.benchmark_symbol else None
        ),
        benchmark_symbol=backtest.benchmark_symbol,
    )
    return ConstructionRun(report, backtest_run.source_info, backtest_run.quality_warnings)
