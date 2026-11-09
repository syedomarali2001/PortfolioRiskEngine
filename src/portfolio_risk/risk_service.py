"""V5 composition: reuse verified V4 price ingestion and strategy simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from portfolio_risk.backtest_service import BacktestRun
from portfolio_risk.risk import RiskError, RiskReport, analyze_risk


@dataclass(frozen=True)
class RiskRun:
    report: RiskReport
    market_data: dict[str, dict]
    quality_warnings: tuple[str, ...]


def run_risk_from_backtest(
    backtest_run: BacktestRun,
    *,
    strategy: str = "buy_hold",
    lookback: int = 252,
    shocks: Mapping[str, float] | None = None,
    confidences: tuple[float, ...] = (0.95, 0.99),
) -> RiskRun:
    """Analyze already-fetched prices and a V4 strategy without another API call."""
    report = backtest_run.report
    if backtest_run.prices_in_base is None:
        raise RiskError("V5 requires price histories retained by the V4 data service")
    if strategy not in report.strategies:
        raise RiskError(f"Strategy {strategy!r} was not simulated")
    strat = report.strategies[strategy]
    prices = {s: backtest_run.prices_in_base[s] for s in report.target_weights}
    risk = analyze_risk(
        prices,
        strat.ending_weights,
        strat.returns,
        strategy=strategy,
        base_currency=report.base_currency,
        starting_capital=report.initial_capital,
        reference_capital=report.initial_capital,
        lookback=lookback,
        confidences=confidences,
        shocks=shocks,
    )
    return RiskRun(risk, backtest_run.source_info, backtest_run.quality_warnings)
