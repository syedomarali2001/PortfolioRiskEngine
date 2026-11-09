"""CLI uses reusable backend services; dashboard is deferred until V9."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from portfolio_risk.accounting import valuation_as_dict, value_portfolio
from portfolio_risk.backtest import BacktestError, SCHEDULES, backtest_as_dict
from portfolio_risk.backtest_service import run_backtest_from_market_data
from portfolio_risk.cache import HistoryCache
from portfolio_risk.construction import (
    ConstructionError, FREQUENCIES, METHODS, construction_as_dict,
)
from portfolio_risk.construction_service import run_construction_from_backtest
from portfolio_risk.fx import FXRateError, fx_ticker, latest_usd_cad_quote, manual_fx_quote, validate_usd_cad
from portfolio_risk.exposure import ExposureError, SecurityMetadata, analyze_exposure, exposure_as_dict
from portfolio_risk.exposure_classification import canonical_sector, canonical_country, SECTORS
from portfolio_risk.exposure_support import assess_exposure_support
from portfolio_risk.exposure_data import load_asset_metadata, load_fund_snapshots
from portfolio_risk.exposure_yahoo import fetch_yahoo_top_holdings
from portfolio_risk.providers.tickerinside import TickerInsideClient, TickerInsideError, FundUnavailable
from portfolio_risk.etf_audit import read_etf_universe, run_etf_audit, etf_report
from portfolio_risk.stock_audit import read_stock_universe, run_stock_audit, stock_report
from portfolio_risk.capability_report import build_capability_report
from portfolio_risk.security_metadata import YahooSecurityMetadataProvider
from portfolio_risk.coverage_audit import load_audit_universe, run_coverage_audit, audit_as_dict, audit_summary
from portfolio_risk.factor_proxy import FactorProxyError, factor_proxy_as_dict
from portfolio_risk.factor_proxy_service import run_factor_proxy_from_backtest
from portfolio_risk.input import PortfolioFormatError, parse_portfolio
from portfolio_risk.providers.base import MarketDataError
from portfolio_risk.providers.yahoo import YahooFinanceProvider
from portfolio_risk.performance import PerformanceError, performance_as_dict
from portfolio_risk.performance_service import run_performance
from portfolio_risk.service import get_validated_history
from portfolio_risk.risk import RiskError, risk_as_dict
from portfolio_risk.risk_service import run_risk_from_backtest
from portfolio_risk.v8_insights import InsightError, OBJECTIVES
from portfolio_risk.v8_service import run_insights_from_market_data
from portfolio_risk.v8_explain import explain_exposure
from portfolio_risk.v8_underlying import enrich_top_equities
from portfolio_risk.v8_fundamentals import FundamentalsError, analyze_fundamentals, yahoo_fundamentals_snapshot, compare_peer_multiples
from portfolio_risk.v84_dividends import DividendError, evaluate_dividends
from portfolio_risk.v84_dividend_provider import get_cached_dividends


def _date_arg(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected YYYY-MM-DD") from exc


def _rate_arg(raw: str) -> Decimal:
    try:
        return validate_usd_cad(raw)
    except FXRateError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_v8_scenarios(raw: list[str]) -> dict[str, dict[str,float]]:
    """NAME:AAA=50,BBB=50. Exactly 100%; no silent reweighting."""
    result={}
    for item in raw:
        if ':' not in item:
            raise InsightError("--scenario requires NAME:SYMBOL=PERCENT,SYMBOL=PERCENT")
        label, values=item.split(':',1)
        label=label.strip()
        if not label or label in result:
            raise InsightError("Scenario names must be unique and nonempty")
        weights={}
        for assignment in values.split(','):
            if '=' not in assignment:
                raise InsightError("Every scenario ticker needs =PERCENT")
            symbol,value=assignment.split('=',1)
            symbol=symbol.strip().upper()
            if not symbol or symbol in weights:
                raise InsightError("Scenario tickers must be unique and nonempty")
            try:
                weights[symbol]=float(value)/100.
            except ValueError as exc:
                raise InsightError("Scenario weight must be numeric percent") from exc
        result[label]=weights
    return result


def _parse_v8_priorities(raw: str) -> dict[str,float]:
    selected={}
    for entry in raw.split(','):
        name,sep,value=entry.partition(':')
        name=name.strip().lower()
        if name not in OBJECTIVES or name in selected:
            raise InsightError(f"Priority must be one of {', '.join(OBJECTIVES)}, without duplicates")
        try:
            selected[name]=float(value) if sep else 1.
        except ValueError as exc:
            raise InsightError("Priority importance must be a positive number") from exc
    return selected



def _v83_percent_assignments(raw: list[str], flag: str) -> dict[str,float]:
    """Exact ticker -> fraction; no silent percent renormalization."""
    result={}
    for spec in raw:
        ticker,sep,pct=spec.partition('=')
        ticker=ticker.strip().upper()
        if not sep or not ticker or ticker in result:
            raise InsightError(f"{flag} needs unique TICKER=PERCENT entries")
        try:
            fraction=float(pct)/100
        except ValueError as exc:
            raise InsightError(f"{flag} requires numeric percentage") from exc
        if not math.isfinite(fraction) or not 0<=fraction<=1:
            raise InsightError(f"{flag}: percentage must be 0..100")
        result[ticker]=fraction
    return result


def _v83_group_limits(raw: list[str]) -> dict[str,dict]:
    result={}
    for spec in raw:
        name,sep,part=spec.partition(':')
        group_symbols,eq,pct=part.partition('=')
        label=name.strip()
        symbols=[sym.strip().upper() for sym in group_symbols.split(',')]
        if not sep or not eq or not label or label in result or any(not s for s in symbols):
            raise InsightError("--group-cap requires UNIQUE_NAME:TICKER,TICKER=PERCENT")
        try:
            maximum=float(pct)/100
        except ValueError as exc:
            raise InsightError("--group-cap percentage must be numeric") from exc
        if not math.isfinite(maximum) or not 0<=maximum<=1:
            raise InsightError("--group-cap must be between 0 and 100 percent")
        result[label]={"symbols":symbols,"cap":maximum}
    return result


def _print_valuation(summary) -> None:
    print(f"\nPortfolio valuation  |  base={summary.base_currency}")
    print("Entered market values are user-supplied snapshots; no securities were repriced.")
    if summary.fx_quote:
        q = summary.fx_quote
        method = "CACHE" if q.cached else ("MANUAL" if q.source.startswith("User-supplied") else "LIVE")
        print(f"USD/CAD: {q.usd_cad} CAD per USD | FX cutoff {q.as_of} | "
              f"observed {q.observed_on} ({q.age_days}d old) | {method}: {q.source}")
    else:
        print("FX conversion not required: all market values are already in the base currency.")
    print(f"\n{'Symbol':<14}{'Input':>20}{'Base value':>22}{'Weight':>12}")
    print("-" * 68)
    for h in summary.holdings:
        original = f"{h.entered_value:,.2f} {h.entered_currency}"
        converted = f"{h.value_in_base:,.2f} {summary.base_currency}"
        print(f"{h.symbol:<14}{original:>20}{converted:>22}{h.weight * 100:>11.2f}%")
    print("-" * 68)
    print(f"TOTAL: {summary.total_value:,.2f} {summary.base_currency}")
    c = summary.concentration
    print(f"Largest holding: {c.largest_weight * 100:.2f}% | "
          f"Top three: {c.top_three_weight * 100:.2f}%")
    print(f"HHI: {c.herfindahl_index:.6f} | "
          f"Effective holdings (1/HHI): {c.effective_holdings:.2f}")
    print("Valuation-currency breakdown (NOT underlying market/FX exposure):")
    for currency, value in summary.by_entered_currency:
        print(f"  {currency}: {value:,.2f} {summary.base_currency} ({value / summary.total_value * 100:.2f}%)")
    print("V2 is accounting and capital concentration only; risk/performance analysis comes later.")


def _print_performance(run) -> None:
    report = run.report
    stats = report.portfolio
    def pct(number: float) -> str:
        return f"{number * 100:,.2f}%"
    def ratio(number: float | None) -> str:
        return "N/A" if number is None else f"{number:.3f}"

    print(f"\nV3 historical performance (HYPOTHETICAL daily constant-mix, {report.base_currency})")
    print(f"Common close dates: {report.first_date} .. {report.last_date} | "
          f"{report.observation_dates} observations / {report.return_periods} return periods")
    print(f"Assumed annual risk-free rate: {report.risk_free_annual * 100:.2f}%")
    print("Returns use adjusted prices and historical FX. They are NOT your actual account P&L.")
    print(f"\nPortfolio cumulative return: {pct(stats.cumulative_return)}")
    print(f"CAGR: {pct(stats.cagr)} | annualized volatility: {pct(stats.annualized_volatility)}")
    print(f"Sharpe: {ratio(stats.sharpe)} | Sortino: {ratio(stats.sortino)} | "
          f"maximum drawdown: {pct(stats.maximum_drawdown)}")
    print(f"Worst common-date period: {pct(stats.worst_period_return)} | "
          f"best: {pct(stats.best_period_return)}")
    print(f"\n{'Symbol':<14}{'Weight':>11}{'CAGR':>13}{'Volatility':>14}{'Sharpe':>11}{'Max DD':>12}")
    print("-" * 75)
    for symbol, m in report.assets.items():
        print(f"{symbol:<14}{pct(report.weights[symbol]):>11}{pct(m.cagr):>13}"
              f"{pct(m.annualized_volatility):>14}{ratio(m.sharpe):>11}{pct(m.maximum_drawdown):>12}")
    if report.benchmark:
        b = report.benchmark
        print(f"\nBenchmark {b.symbol}: CAGR {pct(b.metrics.cagr)} | "
              f"volatility {pct(b.metrics.annualized_volatility)} | "
              f"max DD {pct(b.metrics.maximum_drawdown)}")
        print(f"Beta: {ratio(b.beta)} | tracking error: {pct(b.tracking_error)} | "
              f"information ratio: {ratio(b.information_ratio)}")
    print("\nReturn correlations (on common dates):")
    names = list(report.assets)
    print(f"{'':12}" + "".join(f"{n:>12}" for n in names))
    for a in names:
        row = "".join(f"{ratio(report.correlations[a][b]):>12}" for b in names)
        print(f"{a:<12}{row}")
    if run.quality_warnings or report.warnings:
        print("\nNotes / warnings:")
        for msg in (*run.quality_warnings, *report.warnings):
            print(f"  - {msg}")
    print("For trade-based simulations use `backtest`; for risk use `risk`; for allocation rules use `construct`.")


def _print_backtest(run) -> None:
    report = run.report
    print(f"\nV4 historical backtest (SIMULATED, {report.base_currency})")
    print(f"Shared closes: {report.first_date} .. {report.last_date} | "
          f"{report.observation_dates} observations / {report.return_periods} return periods")
    print(f"Starting capital: {report.initial_capital:,.2f} {report.base_currency} | "
          f"fee: {report.fee_bps:.2f} bps per traded notional | "
          f"assumed risk-free: {report.risk_free_annual * 100:.2f}%")
    print("Semiannual = twice per calendar year (June and December); "
          "rebalances at final available common close of each completed period.")
    print(f"\n{'Strategy':<14}{'End value':>15}{'CAGR':>12}{'Volatility':>13}"
          f"{'Sharpe':>10}{'Max DD':>12}{'Fees':>13}{'Rebals':>9}")
    print("-" * 98)
    for name, result in report.strategies.items():
        m = result.metrics
        sh = "N/A" if m.sharpe is None else f"{m.sharpe:.3f}"
        print(f"{name:<14}{result.ending_value:>15,.2f}"
              f"{m.cagr * 100:>11.2f}%{m.annualized_volatility * 100:>12.2f}%"
              f"{sh:>10}{m.maximum_drawdown * 100:>11.2f}%"
              f"{result.total_fees:>13,.2f}{result.rebalance_count:>9}")
    if report.benchmark:
        bm = report.benchmark
        print(f"\nBenchmark {report.benchmark_symbol}: fee-free buy-and-hold index | "
              f"CAGR {bm.cagr * 100:.2f}% | "
              f"volatility {bm.annualized_volatility * 100:.2f}% | "
              f"max DD {bm.maximum_drawdown * 100:.2f}%")
    if "buy_hold" in report.strategies:
        hold = report.strategies["buy_hold"]
        print("\nBuy-and-hold ending weight drift (target -> final):")
        for symbol, target in report.target_weights.items():
            print(f"  {symbol:<12} {target * 100:>6.2f}% -> "
                  f"{hold.ending_weights[symbol] * 100:>6.2f}%")
    print("\nStrategy fees include initial purchases; costs default to zero. "
          "Current portfolio weights are applied retrospectively.")
    for warning in (*run.quality_warnings, *report.warnings):
        print(f"  NOTE: {warning}")


def _parse_shocks(items: list[str]) -> dict[str, float]:
    """Parse --shock SYMBOL=-30 into a -0.30 simple return shock."""
    result = {}
    for entry in items:
        if "=" not in entry:
            raise ValueError("Each --shock must be SYMBOL=PERCENT, e.g. BTC-USD=-30")
        symbol, raw = entry.split("=", 1)
        symbol = symbol.strip().upper()
        if not symbol or symbol in result:
            raise ValueError("Shock symbols must be unique and nonempty")
        try:
            fraction = float(raw) / 100.0
        except ValueError as exc:
            raise ValueError(f"Invalid shock percentage for {symbol}") from exc
        if not math.isfinite(fraction) or fraction < -1:
            raise ValueError("Shock percentages must be finite and at least -100")
        result[symbol] = fraction
    return result


def _print_risk(run) -> None:
    report = run.report
    print(f"\nV5 portfolio risk analysis (HISTORICAL SIMULATION, {report.base_currency})")
    print(f"V4 strategy: {report.strategy} | dates: {report.first_date} .. {report.last_date}")
    print(f"Estimation: last {report.estimation_observations} common-close returns | "
          f"rolling lookback: {report.lookback}")
    print(f"Loss amounts illustrated on {report.reference_capital:,.2f} "
          f"{report.base_currency} of reference capital, not an actual account balance.")
    print("\nOne-period Value at Risk and Expected Shortfall (positive potential losses):")
    print(f"{'Model':<21}{'Confidence':>12}{'VaR':>12}{'ES':>12}"
          f"{'VaR amount':>17}{'ES amount':>17}")
    print('-' * 91)
    for e in report.risk_estimates:
        print(f"{e.model:<21}{e.confidence * 100:>11.0f}%"
              f"{e.value_at_risk * 100:>11.2f}%{e.expected_shortfall * 100:>11.2f}%"
              f"{e.var_amount:>17,.2f}{e.es_amount:>17,.2f}")
    print(f"\nAnnualized volatility at final strategy weights: {report.portfolio_volatility * 100:.2f}%")
    if report.diversification_ratio is not None:
        print(f"Diversification ratio: {report.diversification_ratio:.3f}")
    print("\nAttribution of ending-weight portfolio volatility "
          "(covariance-based; risk shares may be negative):")
    print(f"{'Symbol':<14}{'Final weight':>15}{'Standalone vol':>17}"
          f"{'Risk share':>14}{'Vol contribution':>20}")
    print('-' * 80)
    for a in report.attribution:
        print(f"{a.symbol:<14}{a.weight * 100:>14.2f}%"
              f"{a.standalone_volatility * 100:>16.2f}%"
              f"{a.share_of_total_risk * 100:>13.2f}%"
              f"{a.contribution_to_volatility * 100:>19.2f}%")
    print("\nRolling one-step-ahead VaR backtest (strictly prior returns used):")
    print(f"{'Model':<21}{'Conf':>8}{'Tests':>9}{'Breaches':>11}"
          f"{'Expected':>11}{'Kupiec p':>13}")
    print('-' * 75)
    for c in report.coverage:
        pvalue = 'N/A' if c.kupiec_p_value is None else f"{c.kupiec_p_value:.3f}"
        print(f"{c.model:<21}{c.confidence * 100:>7.0f}%"
              f"{c.tests:>9}{c.breaches:>11}{c.expected_breaches:>11.1f}{pvalue:>13}")
    print("\nStress scenarios (historical events are restricted to this sample):")
    for stress in report.stresses:
        print(f"  {stress.label}: {stress.portfolio_return * 100:+.2f}% "
              f"({stress.change_on_capital:+,.2f} {report.base_currency}) "
              f"[{stress.source}]")
    print("\nAssumptions and warnings:")
    for note in (*run.quality_warnings, *report.warnings):
        print(f"  - {note}")


def _print_construction(run) -> None:
    report = run.report
    print(f"\nV6 portfolio construction (HISTORICAL SIMULATION, {report.base_currency})")
    print(f"Common evaluation dates: {report.first_date} .. {report.last_date} | "
          f"{report.observation_dates} observations / {report.return_periods} returns")
    print(f"Initial capital: {report.initial_capital:,.2f} | fee {report.fee_bps:.2f} bps "
          f"| risk-free {100 * report.risk_free_annual:.2f}%")
    print(f"Rebalance: first shared close of new {report.rebalance_frequency} period | signal lookback: "
          f"{report.historical_lookback} common-close returns | "
          f"momentum skip: {report.momentum_skip} | top K: {report.top_k}")
    print(f"Long-only max weight: {report.max_weight:.0%} | "
          f"min-var diagonal shrinkage: {report.shrinkage:.0%}")
    print("Signals exclude the current trade-close return. Starting universe and "
          "reference holdings were chosen with hindsight.")
    print(f"\n{'Method':<21}{'End value':>15}{'CAGR':>11}{'Vol':>10}"
          f"{'Sharpe':>10}{'Max DD':>11}{'Fees':>13}{'Rebals':>9}")
    print("-" * 100)
    for name, row in report.results.items():
        m = row.metrics
        sharpe = "N/A" if m.sharpe is None else f"{m.sharpe:.3f}"
        print(f"{name:<21}{row.ending_value:>15,.2f}"
              f"{100*m.cagr:>10.2f}%{100*m.annualized_volatility:>9.2f}%"
              f"{sharpe:>10}{100*m.maximum_drawdown:>10.2f}%"
              f"{row.total_fees:>13,.2f}{row.rebalance_count:>9}")
    if report.benchmark_metrics is not None:
        bm = report.benchmark_metrics
        print(f"\nBenchmark {report.benchmark_symbol}: fee-free buy-and-hold | "
              f"CAGR {bm.cagr*100:.2f}% | Vol {bm.annualized_volatility*100:.2f}% "
              f"| Max DD {bm.maximum_drawdown*100:.2f}%")
    if report.evaluation_start:
        print(f"\nLATER-PERIOD COMPARISON (from {report.evaluation_start}; "
              "not an untouched holdout):")
        for name, row in report.results.items():
            if row.evaluation_metrics:
                m = row.evaluation_metrics
                print(f"  {name:<20} CAGR {m.cagr*100:>7.2f}% | "
                      f"Sharpe {'N/A' if m.sharpe is None else f'{m.sharpe:.3f}'} | "
                      f"Max DD {m.maximum_drawdown*100:.2f}%")
        if report.benchmark_evaluation_metrics:
            print(f"  {report.benchmark_symbol:<20} benchmark CAGR "
                  f"{report.benchmark_evaluation_metrics.cagr*100:.2f}%")
    print("\nLast target weights by rule:")
    for name, row in report.results.items():
        weights = ", ".join(f"{s} {w*100:.1f}%" for s,w in row.last_target.items() if w>1e-8)
        print(f"  {name:<20} {weights}")
    print("\nMethodological limitations:")
    for msg in (*run.quality_warnings, *report.warnings):
        print(f"  - {msg}")


def _print_exposure(report) -> None:
    print(f"\nV7 exposure intelligence (CURRENT SNAPSHOT; {report.base_currency})")
    print(f"Analysis cutoff: {report.analysis_date} | portfolio value: {float(report.total_value):,.2f} {report.base_currency}")
    print(f"Identified securities (may include unexpanded funds): {100 * (1-report.unknown_underlying):.2f}% | "
          f"Unresolved constituents: {report.unknown_underlying*100:.2f}% of portfolio")
    print("\nPortfolio holdings by entered capital weight:")
    for symbol, weight in report.direct_weights.items():
        print(f"  {symbol:<13} {weight*100:>7.2f}%")
    print("\nIdentified underlying exposures (NOT all holdings if coverage is partial):")
    for symbol, weight in list(report.identified_underlying.items())[:25]:
        print(f"  {symbol:<13} {weight*100:>7.2f}% of portfolio")
    if len(report.identified_underlying) > 25:
        print(f"  ... {len(report.identified_underlying)-25} more; use --json for all")
    for title, data in (("Asset classes", report.by_asset_class),
                        ("Sectors", report.by_sector), ("Company countries/domiciles", report.by_country)):
        print(f"\n{title} (UNKNOWN is not zero):")
        for item, weight in data.items():
            print(f"  {item:<26} {weight*100:>7.2f}%")
    print("\nETF holdings coverage:")
    if not report.coverage_by_fund:
        print("  No ETF/fund snapshots identified. Provide --holdings-file or --yahoo-top-holdings.")
    for symbol, info in report.coverage_by_fund.items():
        print(f"  {symbol:<13} {info['identified_fraction_of_fund']*100:>6.1f}% identified | "
              f"as_of={info['as_of'] or 'UNVERIFIED'} | source={info['source'] or 'MISSING'}")
    print("\nETF pairwise identified common-security overlap (LOWER BOUND):")
    if not report.pairwise_overlap:
        print("  Need two funds with holdings data for overlap comparison")
    for pair in report.pairwise_overlap:
        print(f"  {pair['fund_a']} / {pair['fund_b']}: "
              f"at least {pair['identified_overlap_lower_bound']*100:.2f}% | "
              f"shared: {', '.join(pair['common_identified_symbols']) or 'none identified'}")
    print("\nWarnings / methodology:")
    for note in report.warnings:
        print(f"  - {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="portfolio-risk",
        description="Portfolio Risk Engine V8.4: explainable risk, portfolio strategy and dividend research.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate_cmd = commands.add_parser("validate", help="Validate portfolio file (offline)")
    validate_cmd.add_argument("portfolio", type=Path)

    fetch_cmd = commands.add_parser("fetch", help="Fetch and audit price histories (network on cache miss)")
    fetch_cmd.add_argument("portfolio", type=Path)
    fetch_cmd.add_argument("--start", type=_date_arg, default=date.today() - timedelta(days=90))
    fetch_cmd.add_argument("--end", type=_date_arg, default=date.today() + timedelta(days=1), help="Exclusive upper date")
    fetch_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    fetch_cmd.add_argument("--refresh", action="store_true")
    fetch_cmd.add_argument("--include-fx", action="store_true", help="Also fetch a USD/CAD FX quotation")

    analyze_cmd = commands.add_parser("analyze", help="Calculate FX-normalized weights and concentration")
    analyze_cmd.add_argument("portfolio", type=Path)
    analyze_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    analyze_cmd.add_argument("--as-of", type=_date_arg, default=date.today(),
                             help="FX observation cutoff date, NOT a historical portfolio valuation")
    analyze_cmd.add_argument("--usd-cad", type=_rate_arg, default=None,
                             help="Supply an explicit CAD-per-USD rate; bypasses FX API")
    analyze_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    analyze_cmd.add_argument("--refresh", action="store_true")
    analyze_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    analyze_cmd.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")

    perf_cmd = commands.add_parser(
        "performance", help="Historical risk-adjusted performance of a synthetic constant-mix portfolio"
    )
    perf_cmd.add_argument("portfolio", type=Path,
                          help="Portfolio CSV with optional quote_currency column")
    perf_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    perf_cmd.add_argument("--start", type=_date_arg, default=date.today() - timedelta(days=730))
    perf_cmd.add_argument("--end", type=_date_arg, default=date.today() + timedelta(days=1),
                          help="Exclusive upper bound")
    perf_cmd.add_argument("--as-of", type=_date_arg, default=date.today(),
                          help="Cutoff for current snapshot weights' FX (not historical holdings)")
    perf_cmd.add_argument("--usd-cad", type=_rate_arg, default=None,
                          help="Optional manual FX rate for CURRENT weights only")
    perf_cmd.add_argument("--risk-free", type=float, default=0.0,
                          help="Assumed annual risk-free rate in percent; default 0")
    perf_cmd.add_argument("--benchmark", default=None,
                          help="Optional Yahoo benchmark ticker, e.g., SPY")
    perf_cmd.add_argument("--benchmark-currency", choices=["CAD", "USD"], default=None)
    perf_cmd.add_argument("--min-observations", type=int, default=30)
    perf_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    perf_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    perf_cmd.add_argument("--refresh", action="store_true")
    perf_cmd.add_argument("--json", action="store_true", help="Machine-readable output including index")

    backtest_cmd = commands.add_parser(
        "backtest", help="Simulate buy-and-hold and calendar rebalancing strategies (V4)"
    )
    backtest_cmd.add_argument("portfolio", type=Path,
                              help="Portfolio file; include quote_currency when provider metadata is unavailable")
    backtest_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    backtest_cmd.add_argument("--start", type=_date_arg, default=date.today() - timedelta(days=730))
    backtest_cmd.add_argument("--end", type=_date_arg, default=date.today() + timedelta(days=1),
                              help="Exclusive historical price end date")
    backtest_cmd.add_argument("--as-of", type=_date_arg, default=date.today(),
                              help="Snapshot FX cutoff for CURRENT weights, not backtest start")
    backtest_cmd.add_argument("--usd-cad", type=_rate_arg, default=None,
                              help="Manual FX for snapshot weights only; historical FX still uses API")
    backtest_cmd.add_argument("--capital", type=float, default=20000.0,
                              help="Simulated initial capital in base currency (default 20000)")
    backtest_cmd.add_argument("--fee-bps", type=float, default=0.0,
                              help="Cost per side, on gross traded value, in basis points (default 0)")
    backtest_cmd.add_argument("--strategies", nargs="+", choices=SCHEDULES,
                              default=list(SCHEDULES),
                              help="Select from buy_hold monthly quarterly semiannual annual; default all")
    backtest_cmd.add_argument("--risk-free", type=float, default=0.0,
                              help="Assumed annual risk-free rate in percent")
    backtest_cmd.add_argument("--benchmark", default=None, help="Optional benchmark, e.g. SPY")
    backtest_cmd.add_argument("--benchmark-currency", choices=["CAD", "USD"], default=None)
    backtest_cmd.add_argument("--min-observations", type=int, default=30)
    backtest_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    backtest_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    backtest_cmd.add_argument("--refresh", action="store_true")
    backtest_cmd.add_argument("--json", action="store_true",
                              help="Structured output, including trades and daily portfolio values")

    risk_cmd = commands.add_parser(
        "risk", help="V5 portfolio risk attribution, VaR/ES, stress and rolling validation"
    )
    risk_cmd.add_argument("portfolio", type=Path)
    risk_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    risk_cmd.add_argument("--start", type=_date_arg, default=date.today() - timedelta(days=730))
    risk_cmd.add_argument("--end", type=_date_arg, default=date.today() + timedelta(days=1),
                          help="Exclusive historical upper bound")
    risk_cmd.add_argument("--as-of", type=_date_arg, default=date.today(),
                          help="Snapshot-weight FX cutoff, not historical portfolio state")
    risk_cmd.add_argument("--usd-cad", type=_rate_arg, default=None,
                          help="Manual USD/CAD for snapshot weights only")
    risk_cmd.add_argument("--capital", type=float, default=20000.0,
                          help="Simulated initial capital and risk amount reference")
    risk_cmd.add_argument("--fee-bps", type=float, default=0.0)
    risk_cmd.add_argument("--risk-free", type=float, default=0.0,
                          help="V4 strategy metric risk-free assumption in percent")
    risk_cmd.add_argument("--strategy", choices=SCHEDULES, default="buy_hold")
    risk_cmd.add_argument("--lookback", type=int, default=252,
                          help="Trailing common-close returns for VaR/ES and rolling forecasts")
    risk_cmd.add_argument("--shock", action="append", default=[], metavar="SYMBOL=PERCENT",
                          help="Optional hypothetical one-period shock; may repeat")
    risk_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    risk_cmd.add_argument("--min-observations", type=int, default=30)
    risk_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    risk_cmd.add_argument("--refresh", action="store_true")
    risk_cmd.add_argument("--json", action="store_true", help="Structured results, for future V9 UI")

    construct_cmd = commands.add_parser(
        "construct", help="V6 walk-forward allocation strategies; no return forecast or dashboard"
    )
    construct_cmd.add_argument("portfolio", type=Path)
    construct_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    construct_cmd.add_argument("--start", type=_date_arg, default=date.today()-timedelta(days=1095))
    construct_cmd.add_argument("--end", type=_date_arg, default=date.today()+timedelta(days=1))
    construct_cmd.add_argument("--as-of", type=_date_arg, default=date.today(),
                               help="Cutoff for supplied holdings' snapshot FX")
    construct_cmd.add_argument("--usd-cad", type=_rate_arg, default=None,
                               help="Manual snapshot FX; historic price FX still fetched")
    construct_cmd.add_argument("--capital", type=float, default=20000)
    construct_cmd.add_argument("--fee-bps", type=float, default=0)
    construct_cmd.add_argument("--risk-free", type=float, default=0)
    construct_cmd.add_argument("--rebalance", choices=FREQUENCIES, default="quarterly")
    construct_cmd.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    construct_cmd.add_argument("--lookback", type=int, default=252,
                               help="Number of older common-close return periods for estimation")
    construct_cmd.add_argument("--momentum-skip", type=int, default=21,
                               help="Exclude recent periods from the momentum signal")
    construct_cmd.add_argument("--top-k", type=int, default=2)
    construct_cmd.add_argument("--max-weight", type=float, default=.6,
                               help="Upper bound on non-reference per-asset portfolio weights")
    construct_cmd.add_argument("--shrinkage", type=float, default=.15,
                               help="Shrink covariance toward diagonal for min-variance")
    construct_cmd.add_argument("--evaluation-start", type=_date_arg, default=None,
                               help="Optional later-period performance report, not an untouched holdout")
    construct_cmd.add_argument("--benchmark", default=None)
    construct_cmd.add_argument("--benchmark-currency", choices=["CAD", "USD"], default=None)
    construct_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    construct_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    construct_cmd.add_argument("--refresh", action="store_true")
    construct_cmd.add_argument("--json", action="store_true")

    insights_cmd = commands.add_parser(
        "insights", help="V8 SPY/RSP diagnostics, portfolio scenarios and held-out allocation testing")
    insights_cmd.add_argument("portfolio", type=Path)
    insights_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    insights_cmd.add_argument("--start", type=_date_arg, default=date.today()-timedelta(days=1460))
    insights_cmd.add_argument("--end", type=_date_arg, default=date.today()+timedelta(days=1))
    insights_cmd.add_argument("--as-of", type=_date_arg, default=date.today())
    insights_cmd.add_argument("--usd-cad", type=_rate_arg, default=None)
    insights_cmd.add_argument("--risk-free", type=float, default=0., help="Annual risk-free rate in percent")
    insights_cmd.add_argument("--scenario", action="append", default=[],
                              help="NAME:AAPL=50,SPY=50; include every portfolio ticker, sum to 100%")
    insights_cmd.add_argument("--priorities", default="sharpe:1",
                              help="Preference weights for training-only strategy selection, e.g. sharpe:2,drawdown:1")
    insights_cmd.add_argument("--max-weight", type=float, default=.7,
                              help="Maximum candidate asset weight (original portfolio is not reweighted)")
    insights_cmd.add_argument("--three-strategies", action="store_true",
        help="V9.18: optimize growth, Sharpe and minimum volatility together using shared market data")
    insights_cmd.add_argument("--robustness", action="store_true",
        help="V8.3 chronological expanding-window strategy validation with practical constraints")
    insights_cmd.add_argument("--asset-cap", action="append", default=[],
        help="V8.3 maximum weight for a ticker: GLD=15 (percent); repeatable")
    insights_cmd.add_argument("--asset-floor", action="append", default=[],
        help="V8.3 minimum weight for a ticker: SPY=20 (percent); repeatable")
    insights_cmd.add_argument("--group-cap", action="append", default=[],
        help="V8.3 maximum combined allocation: crypto:BTC-USD,ETH-USD=10 (percent); groups may not overlap")
    insights_cmd.add_argument("--turnover-cap", type=float, default=None,
        help="V8.3 maximum one-way initial turnover from original weights, percent")
    insights_cmd.add_argument("--rebalance", choices=["daily","monthly","quarterly","annually","buy_hold"],
        default="monthly", help="V8.3 rebalancing policy for rolling tests; default monthly")
    insights_cmd.add_argument("--cost-bps", type=float, default=10.,
        help="V8.3 proportional one-way turnover cost in basis points (default 10)")
    insights_cmd.add_argument("--train-periods", type=int, default=252)
    insights_cmd.add_argument("--test-periods", type=int, default=126)
    insights_cmd.add_argument("--step-periods", type=int, default=126)
    insights_cmd.add_argument("--min-observations", type=int, default=60)
    insights_cmd.add_argument("--compare-currencies",action="store_true",
                              help="Also compare the exact portfolio in CAD vs USD on strictly matched FX dates")
    insights_cmd.add_argument("--mc-paths", type=int, default=300)
    insights_cmd.add_argument("--mc-horizon", type=int, default=252)
    insights_cmd.add_argument("--seed", type=int, default=42)
    insights_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    insights_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    insights_cmd.add_argument("--refresh", action="store_true")
    insights_cmd.add_argument("--json", action="store_true")
    insights_cmd.add_argument("--output", type=Path, default=None)

    fundamentals_cmd = commands.add_parser(
        "fundamentals", help="V8 financial statement multiples and illustrative scenario DCF")
    fundamentals_cmd.add_argument("symbol", help="Single equity ticker, e.g. AAPL or RY.TO")
    fundamentals_cmd.add_argument("--file", type=Path, default=None,
                                  help="Optional offline JSON statement snapshot (no Yahoo calls)")
    fundamentals_cmd.add_argument("--peer-file", type=Path, action="append", default=[],
                                  help="Additional peer financial snapshot JSON files; repeat for 2+ peers")
    fundamentals_cmd.add_argument("--peer-ticker", action="append", default=[],
                                  help="Optional live Yahoo peer ticker; repeat for comparison")
    fundamentals_cmd.add_argument("--as-of", type=_date_arg, default=date.today())
    fundamentals_cmd.add_argument("--output", type=Path, default=None)
    fundamentals_cmd.add_argument("--json", action="store_true")

    dividends_cmd = commands.add_parser(
        "dividends", help="V8.4 optional historical dividend yield, income and illustrative DRIP")
    dividends_cmd.add_argument("portfolio", type=Path,
        help="Portfolio CSV with market_value/currency and optional quote_currency")
    dividends_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    dividends_cmd.add_argument("--as-of", type=_date_arg, default=date.today())
    dividends_cmd.add_argument("--usd-cad", type=_rate_arg, default=None,
        help="Optional explicit CAD per USD rate for snapshot and distribution currency conversions")
    dividends_cmd.add_argument("--years", type=int, default=10, help="Projection horizon (1..40 years)")
    dividends_cmd.add_argument("--price-growth", type=float, default=4.,
        help="User-assumed annual future share price growth, percent; default 4")
    dividends_cmd.add_argument("--dividend-growth", type=float, default=3.,
        help="User-assumed future per-share dividend growth, percent; default 3")
    dividends_cmd.add_argument("--tax-drag", type=float, default=0.,
        help="Optional user-assumed effective distribution withholding/tax drag, percent (default 0)")
    dividends_cmd.add_argument("--snapshot-file", type=Path, default=None,
        help="Offline combined snapshot JSON, key snapshots:{SYMBOL:{...}}; no ticker HTTP")
    dividends_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/dividends"))
    dividends_cmd.add_argument("--refresh", action="store_true")
    dividends_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    dividends_cmd.add_argument("--output", type=Path, default=None)
    dividends_cmd.add_argument("--json", action="store_true")

    coverage_cmd = commands.add_parser(
        "coverage-audit", help="V7.1: inspect 30+ assets, US ETF holdings coverage and optional market prices"
    )
    coverage_cmd.add_argument("universe", type=Path,
                              help="Audit CSV with symbol,category,quote_currency")
    coverage_cmd.add_argument("--check-prices", action="store_true",
                              help="Also query Yahoo adjusted-price coverage for every asset (30 requests)")
    coverage_cmd.add_argument("--market-days", type=int, default=45)
    coverage_cmd.add_argument("--check-metadata", action="store_true",
                              help="Also fetch best-effort Yahoo identity/sector/industry/company domicile")
    coverage_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    coverage_cmd.add_argument("--tickerinside-cache", type=Path, default=Path("data/cache/tickerinside"))
    coverage_cmd.add_argument("--no-cache", action="store_true", help="Do not write TickerInside responses to disk")
    coverage_cmd.add_argument("--refresh", action="store_true")
    coverage_cmd.add_argument("--json", action="store_true", help="JSON to stdout for later comparison")
    coverage_cmd.add_argument("--output", type=Path, default=None,
                              help="Optional JSON report file; not required")

    etf_cmd = commands.add_parser(
        "etf-audit", help="V7.2: broad categorized U.S. ETF holdings audit (no price requests)"
    )
    etf_cmd.add_argument("universe", type=Path, help="CSV: symbol,theme,issuer")
    etf_cmd.add_argument("--output", type=Path, default=Path("us_etf_v72_report.json"))
    etf_cmd.add_argument("--offset", type=int, default=0,
                         help="Skip this many rows for chunked audits (zero-based)")
    etf_cmd.add_argument("--limit", type=int, default=None,
                         help="Optional smaller initial smoke test; full report omits this flag")
    etf_cmd.add_argument("--delay-ms", type=int, default=100,
                         help="Pace between fund requests (default 100 ms)")
    etf_cmd.add_argument("--refresh", action="store_true")
    etf_cmd.add_argument("--tickerinside-cache", type=Path, default=Path("data/cache/tickerinside"))
    etf_cmd.add_argument("--no-cache", action="store_true")
    etf_cmd.add_argument("--json", action="store_true")

    stock_cmd = commands.add_parser(
        "stock-audit", help="V7.2: stratified U.S./Canadian stock metadata audit via Yahoo"
    )
    stock_cmd.add_argument("universe", type=Path, help="CSV: symbol,market,group")
    stock_cmd.add_argument("--output", type=Path, default=Path("stock_v72_report.json"))
    stock_cmd.add_argument("--offset", type=int, default=0,
                           help="Skip this many rows for chunked audits (zero-based)")
    stock_cmd.add_argument("--limit", type=int, default=None)
    stock_cmd.add_argument("--delay-ms", type=int, default=150)
    stock_cmd.add_argument("--json", action="store_true")

    matrix_cmd = commands.add_parser(
        "capability-report", help="Combine three V7.2 JSON audit reports into an observed capability matrix"
    )
    matrix_cmd.add_argument("--etfs", required=True, type=Path)
    matrix_cmd.add_argument("--stocks", required=True, type=Path)
    matrix_cmd.add_argument("--mixed", required=True, type=Path)
    matrix_cmd.add_argument("--output", type=Path, default=Path("v72_capabilities.json"))

    exposure_cmd = commands.add_parser(
        "exposure", help="V7 current ETF look-through, sector, country and overlap analysis"
    )
    exposure_cmd.add_argument("portfolio", type=Path)
    exposure_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    exposure_cmd.add_argument("--as-of", type=_date_arg, default=date.today(),
                              help="Current data cutoff; snapshots after this date are rejected")
    exposure_cmd.add_argument("--usd-cad", type=_rate_arg, default=None,
                              help="Manual CAD per USD rate for entered values; optional")
    exposure_cmd.add_argument("--holdings-file", type=Path, default=None,
                              help="Issuer/research ETF constituents CSV with source and disclosure dates")
    exposure_cmd.add_argument("--assets-file", type=Path, default=None,
                              help="Direct assets and constituents: asset class, sector, country")
    exposure_cmd.add_argument("--yahoo-top-holdings", nargs="*", default=[], metavar="ETF",
                              help="OPTIONAL: fetch top holdings for listed funds (partial/unverified date)")
    exposure_cmd.add_argument("--tickerinside", action="store_true",
                              help="Automatically look up all US ETFs in portfolio via TickerInside")
    exposure_cmd.add_argument("--auto-stock-metadata", action="store_true",
                              help="Best-effort Yahoo classifications for direct stocks / crypto / fund types")
    exposure_cmd.add_argument("--metadata-limit", type=int, default=40,
                              help="Maximum number of direct tickers to classify (default 40)")
    exposure_cmd.add_argument("--drilldown", action="store_true",
                              help="V8 ETF top holdings, overlap contributors and sector-by-investment breakdown")
    exposure_cmd.add_argument("--top-holdings", type=int, default=25)
    exposure_cmd.add_argument("--enrich-underlyings", type=int, default=0,
                              help="Optional top-N underlying equity Yahoo metadata lookups (0..200; cached)")
    exposure_cmd.add_argument("--underlying-metadata-cache", type=Path,
                              default=Path("data/cache/v8_stock_metadata"))
    exposure_cmd.add_argument("--focus-sector", action="append", default=[],
                              help="V8.3 enrich individual companies for this sector (e.g. financials); requires --drilldown")
    exposure_cmd.add_argument("--focus-limit", type=int, default=40,
                              help="Max metadata attempts per focus sector (0..200; shared cache)")
    exposure_cmd.add_argument("--strict-exposure", action="store_true",
                              help="Reject the exposure report if any holding is partial, price-only or unverified")
    exposure_cmd.add_argument("--max-holdings-age-days", type=int, default=120)
    exposure_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    exposure_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    exposure_cmd.add_argument("--refresh", action="store_true")
    exposure_cmd.add_argument("--json", action="store_true", help="Structured output for V9 GUI")

    factors_cmd = commands.add_parser(
        "factors", help="V7 optional descriptive beta against investable ETF proxies"
    )
    factors_cmd.add_argument("portfolio", type=Path)
    factors_cmd.add_argument("--base", choices=["CAD", "USD"], default="CAD")
    factors_cmd.add_argument("--start", type=_date_arg, default=date.today()-timedelta(days=1095))
    factors_cmd.add_argument("--end", type=_date_arg, default=date.today()+timedelta(days=1))
    factors_cmd.add_argument("--as-of", type=_date_arg, default=date.today())
    factors_cmd.add_argument("--usd-cad", type=_rate_arg, default=None)
    factors_cmd.add_argument("--proxies", nargs="+", default=["SPY:USD", "MTUM:USD"],
                             metavar="TICKER:CURRENCY", help="e.g., SPY:USD MTUM:USD")
    factors_cmd.add_argument("--capital", type=float, default=20000)
    factors_cmd.add_argument("--min-observations", type=int, default=60)
    factors_cmd.add_argument("--max-fx-age-days", type=int, default=5)
    factors_cmd.add_argument("--cache-dir", type=Path, default=Path("data/cache/v1"))
    factors_cmd.add_argument("--refresh", action="store_true")
    factors_cmd.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "capability-report":
        try:
            result = build_capability_report(
                json.loads(args.etfs.read_text(encoding="utf-8")),
                json.loads(args.stocks.read_text(encoding="utf-8")),
                json.loads(args.mixed.read_text(encoding="utf-8")),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False), encoding="utf-8")
            print("V7.2 capability matrix (curated samples, not all-market coverage):")
            for category, row in result['capabilities'].items():
                prices=row['sampled_prices']
                print(f"  {category:<15} sampled prices: {prices['available']}/{prices['tested']}", end='')
                if category == 'US_ETF':
                    print(f"; dated constituents: {row['funds_with_dated_holdings']}/{row['funds_audited']}"
                          f"; near-full: {row['funds_with_95pct_weight']}"
                          f"; sectors: {row['funds_with_sectors']}")
                elif 'stock_metadata' in row:
                    metadata=row['stock_metadata']
                    print(f"; sector: {metadata['sector_available']}/{metadata['total']}"
                          f"; country: {metadata['company_country_available']}/{metadata['total']}")
                else:
                    print()
            print(f"Saved {args.output}")
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"CAPABILITY REPORT ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "stock-audit":
        try:
            universe = read_stock_universe(args.universe)
            if args.offset < 0 or args.offset >= len(universe):
                raise ValueError("--offset must select at least one remaining entry")
            universe = universe[args.offset:]
            if args.limit is not None:
                if args.limit < 1:
                    raise ValueError("--limit must be positive")
                universe = universe[:args.limit]
            if not args.json:
                print(f"V7.2 stock metadata audit: {len(universe)} securities; Yahoo Finance", flush=True)
            rows = run_stock_audit(universe, YahooSecurityMetadataProvider(), delay_ms=args.delay_ms,
                on_progress=None if args.json else (lambda i,n,s: print(f"  [{i:03}/{n:03}] {s}", flush=True)))
            report = stock_report(rows)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            if args.json:
                print(json.dumps(report, indent=2, allow_nan=False))
            else:
                print("\nOverall: "+", ".join(f"{k}={v}" for k,v in report['summary'].items() if isinstance(v,int)))
                for market,stats in report['by_market'].items():
                    print(f"  {market}: {stats['total']} symbols, {stats['equity_identified']} equities, "
                          f"{stats['sector_available']} sectors, {stats['industry_available']} industries, "
                          f"{stats['company_country_available']} company countries, "
                          f"{stats['provider_unavailable']} lookup errors")
                print(f"Report saved to {args.output}")
                print("Company domicile != revenue exposure; unavailable metadata != unavailable prices.")
        except (ValueError, OSError) as exc:
            print(f"STOCK AUDIT ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "etf-audit":
        try:
            universe = read_etf_universe(args.universe)
            if args.offset < 0 or args.offset >= len(universe):
                raise ValueError("--offset must select at least one remaining entry")
            universe = universe[args.offset:]
            if args.limit is not None:
                if args.limit < 1:
                    raise ValueError("--limit must be positive")
                universe = universe[:args.limit]
            client = TickerInsideClient(cache_dir=args.tickerinside_cache,
                                        ttl_hours=0 if args.no_cache else 24)
            if not args.json:
                print(f"V7.2 curated U.S. ETF audit: {len(universe)} funds; TickerInside", flush=True)
            rows, index_info = run_etf_audit(
                universe, client, refresh=args.refresh, delay_ms=args.delay_ms,
                on_progress=None if args.json else (lambda i,n,s: print(f"  [{i:03}/{n:03}] {s}", flush=True)),
            )
            result = etf_report(rows, index_info)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            if args.json:
                print(json.dumps(result, indent=2, allow_nan=False))
            else:
                print("\nAggregate: " + ", ".join(f"{k}={v}" for k, v in result["summary"].items()
                     if isinstance(v, int)))
                print("By theme (indexed / holdings / >=95% weight / <=120d / sector fields):")
                for theme, stats in result["by_theme"].items():
                    print(f"  {theme:<28} {stats['indexed']:>3}/{stats['total']:<3} indexed  "
                          f"{stats['with_dated_holdings']:>3} dated  "
                          f"{stats['near_full_weight_95pct']:>3} near-full  "
                          f"{stats['current_120d']:>3} recent  "
                          f"{stats['with_sector_breakdown']:>3} sector")
                print(f"Report saved to {args.output}")
                print("Curated diagnostic sample; not a statistically representative all-market coverage rate.")
                print("Credit TickerInside (CC BY 4.0), https://tickerinside.com")
        except (ValueError, OSError, TickerInsideError) as exc:
            print(f"ETF AUDIT ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "coverage-audit":
        try:
            universe = load_audit_universe(args.universe)
            api = TickerInsideClient(cache_dir=args.tickerinside_cache,
                                     ttl_hours=0 if args.no_cache else 24)
            if not args.json:
                print(f"V7.1 coverage audit: {len(universe)} securities; "+
                      "ETF holdings via TickerInside, prices via Yahoo only if requested", flush=True)
            report = run_coverage_audit(
                universe, api, check_prices=args.check_prices,
                provider=YahooFinanceProvider() if args.check_prices else None,
                price_cache=HistoryCache(args.cache_dir) if args.check_prices else None,
                market_days=args.market_days, refresh=args.refresh,
                check_metadata=args.check_metadata,
                metadata_provider=YahooSecurityMetadataProvider() if args.check_metadata else None,
                on_progress=None if args.json else (lambda i,n,t:
                    print(f"  [{i:02}/{n:02}] {t}", flush=True)),
            )
            out = audit_as_dict(report)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
            if args.json:
                print(json.dumps(out, indent=2, allow_nan=False))
            else:
                print(f"\n{'Symbol':<13}{'Category':<17}{'ETF holdings':<24}"
                      f"{'Positions':>10}{'Weight %':>11}{'Date':>14}{'Sector':>9}{'Country':>9}{'Price':>12}")
                print('-'*119)
                for row in report:
                    positions = str(row.holdings_count) if row.holdings_count is not None else '-'
                    weight = f"{row.listed_pct:.1f}%" if row.listed_pct is not None else '-'
                    print(f"{row.symbol:<13}{row.category:<17}{row.holdings_status:<24}"
                          f"{positions:>10}{weight:>11}{str(row.holdings_as_of or '-'):>14}"
                          f"{row.sector_count:>9}{row.country_count:>9}{row.price_status:>12}")
                print("\nSummary:")
                for category, stats in audit_summary(report)["by_category"].items():
                    print(f"  {category:<16} {stats['tickers']} tickers; "
                          f"{stats['holdings_available']} ETFs with holdings; "
                          f"{stats['price_available']} price series available")
                print("\nSectors/countries are fund-level summaries when present, not per-security labels.")
                print("Use --json or --output audit.json for as-of dates, top five positions, "
                      "sector/country breakdowns and issuer/filing provenance.")
                if args.check_metadata:
                    print("\nYahoo metadata (company domicile is NOT economic geographic exposure):")
                    for row in report:
                        print(f"  {row.symbol:<13} {row.metadata_status:<12} "
                              f"{str(row.metadata_quote_type or '-'): <15} "
                              f"sector={row.metadata_sector or '-'}  "
                              f"country={row.metadata_company_country or '-'}  "
                              f"category_match={row.metadata_category_match}")
                    print("Sector and domicile are best-effort classifications; unavailable != unsupported.")
                print("Provider credit: TickerInside (CC BY 4.0), https://tickerinside.com")
                if not args.check_prices:
                    print("Price availability NOT checked; rerun with --check-prices for all 30 assets.")
        except (ValueError, TickerInsideError, OSError) as exc:
            print(f"COVERAGE AUDIT ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "fundamentals":
        try:
            symbol=args.symbol.strip().upper()
            if not symbol:
                raise FundamentalsError("Ticker cannot be empty")
            if args.as_of > date.today():
                raise FundamentalsError("Analysis cutoff cannot be in the future")
            if args.file:
                record=json.loads(args.file.read_text(encoding="utf-8-sig"))
                if not isinstance(record,dict) or str(record.get("symbol","")).upper()!=symbol:
                    raise FundamentalsError("Statement file's ticker must match requested ticker")
            else:
                if args.as_of != date.today():
                    raise FundamentalsError("Yahoo live quotes are not historical vintages; use --file for past as-of")
                record=yahoo_fundamentals_snapshot(symbol,as_of=args.as_of)
            result=analyze_fundamentals(record,as_of=args.as_of)
            if args.peer_file or args.peer_ticker:
                peer_records=[]
                for peer_file in args.peer_file:
                    peer_records.append(json.loads(peer_file.read_text(encoding="utf-8-sig")))
                for peer in args.peer_ticker:
                    if args.as_of != date.today():
                        raise FundamentalsError("Live Yahoo peer market quotes are not point-in-time; use --peer-file with historical cutoff")
                    peer_records.append(yahoo_fundamentals_snapshot(peer.upper(),as_of=args.as_of))
                result['peer_comparison']=compare_peer_multiples(record,peer_records,as_of=args.as_of)
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
            if args.json:
                print(json.dumps(result,indent=2,allow_nan=False))
            else:
                print(f"V8 fundamental valuation: {symbol} | fiscal period {result['fiscal_period_end']} "
                      f"| {result['statement_currency']}")
                print("Multiples:",result['multiples'])
                if result['dcf']['status']=="unavailable":
                    print("DCF UNAVAILABLE:",result['dcf']['reason'])
                else:
                    for name,case in result['dcf']['scenarios'].items():
                        print(f"  {name}: {case['intrinsic_value_per_share']:.2f} {result['statement_currency']} "
                              f"(growth={case['growth']:.1%}, discount={case['discount_rate']:.1%}, "
                              f"terminal={case['terminal_growth']:.1%})")
                if 'peer_comparison' in result:
                    print("Peer relative multiples:")
                    for metric,data in result['peer_comparison']['multiples'].items():
                        if data['status']=='evaluated':
                            print(f"  {metric}: {data['subject_value']:.2f} vs median {data['peer_median']:.2f} "
                                  f"({data['subject_vs_peer_median_fraction']:+.1%} relative)")
                        else:
                            print(f"  {metric}: unavailable ({data['reason']})")
                for line in result['warnings']+result['limitations']:
                    print("NOTE:",line)
        except (ValueError,OSError,FundamentalsError,TypeError) as exc:
            print(f"FUNDAMENTALS ERROR: {exc}",file=sys.stderr)
            return 1
        return 0

    try:
        portfolio = parse_portfolio(args.portfolio)
    except PortfolioFormatError as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2

    if args.command == "dividends":
        if args.as_of > date.today() or args.max_fx_age_days < 0:
            print("DIVIDENDS INPUT ERROR: as-of cannot be future; FX maximum age must be nonnegative",file=sys.stderr)
            return 2
        try:
            if args.snapshot_file is not None:
                raw=json.loads(args.snapshot_file.read_text(encoding="utf-8-sig"))
                if not isinstance(raw,dict) or not isinstance(raw.get("snapshots"),dict):
                    raise DividendError("Offline JSON needs a top-level snapshots ticker mapping")
                all_snaps=raw["snapshots"]
                if raw.get("as_of") and str(raw["as_of"])!=args.as_of.isoformat():
                    raise DividendError("Offline dividend snapshot as_of must match --as-of")
                snapshots={sym:all_snaps[sym] for sym in (h.symbol for h in portfolio.holdings)
                           if sym in all_snaps}
                fetch_warnings=[]
            else:
                snapshots={}
                fetch_warnings=[]
                for h in portfolio.holdings:
                    try:
                        snapshots[h.symbol]=get_cached_dividends(
                            h.symbol,as_of=args.as_of,cache_dir=args.cache_dir,
                            expected_currency=h.quote_currency,refresh=args.refresh)
                    except DividendError as exc:
                        fetch_warnings.append(f"{h.symbol}: live dividend data unavailable — {exc}")
            # User-explicit quote currency is a strict cross-check, even for
            # offline data. Do not use a plausible but contradicted dividend FX.
            for holding in portfolio.holdings:
                snap=snapshots.get(holding.symbol)
                if (holding.quote_currency and isinstance(snap,dict)
                        and str(snap.get('quote_currency') or '').upper()!=holding.quote_currency):
                    fetch_warnings.append(
                        f"{holding.symbol}: dividend snapshot quote currency conflicts with "
                        f"portfolio quote_currency={holding.quote_currency}; excluded")
                    snapshots.pop(holding.symbol,None)
            # Market value is not the same as traded security currency. Dividend
            # conversion can require FX even if all entered values are in base.
            currencies={str(snap.get('quote_currency') or '').upper()
                        for snap in snapshots.values() if isinstance(snap,dict)}
            needs_fx=(any(h.currency!=args.base for h in portfolio.holdings)
                      or any(ccy in {'USD','CAD'} and ccy!=args.base for ccy in currencies))
            fx_quote=None
            if needs_fx:
                fx_quote=(manual_fx_quote(args.usd_cad,args.as_of) if args.usd_cad is not None
                          else latest_usd_cad_quote(YahooFinanceProvider(),HistoryCache(Path("data/cache/v1")),
                              args.as_of,max_age_days=args.max_fx_age_days,refresh=args.refresh))
            valuation=value_portfolio(portfolio,base_currency=args.base,fx_quote=fx_quote)
            result=evaluate_dividends(
                valuation,snapshots,as_of=args.as_of,years=args.years,
                price_growth=args.price_growth/100.,dividend_growth=args.dividend_growth/100.,
                tax_drag=args.tax_drag/100.,fx_rate=fx_quote.usd_cad if fx_quote else None)
            result['warnings']=[*fetch_warnings,*result['warnings']]
            result['fx_snapshot']=None if fx_quote is None else {
                'rate_cad_per_usd':str(fx_quote.usd_cad),
                'observed_on':fx_quote.observed_on.isoformat(),
                'source':fx_quote.source,
            }
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
            if args.json:
                print(json.dumps(result,indent=2,allow_nan=False))
            else:
                summary=result['income_summary']
                print(f"V8.4 DIVIDENDS | {args.as_of} | {args.base} | covered {summary['income_coverage_fraction']:.1%}")
                if summary['status']=='evaluated':
                    print(f"  Historical TTM yield on covered assets: {summary['trailing_distribution_yield_on_covered_value']:.2%}")
                    print(f"  Estimated gross annual income (covered): {summary['estimated_annual_gross_income_covered_holdings']:,.2f} {args.base}")
                    print(f"  Mean monthly amount (NOT a monthly payout): {summary['estimated_monthly_average_gross_income_covered_holdings']:,.2f} {args.base}")
                for row in result['holdings']:
                    if row['status']=='evaluated':
                        print(f"  {row['symbol']}: TTM yield {row['trailing_distribution_yield']:.2%}; gross annual {row['estimated_annual_gross_income']:,.2f}")
                    else:
                        print(f"  {row['symbol']}: unavailable — {row['reason']}")
                if result['simulation']['status']=='illustrative':
                    end=result['simulation']['end_comparison']
                    print(f"  {args.years}-year cash total wealth: {end['cash']['total_wealth_including_withdrawn_cash']:,.2f} {args.base}")
                    print(f"  {args.years}-year DRIP market value: {end['drip']['total_wealth']:,.2f} {args.base}")
                print("  NOTE: Hypothetical future growth; no personal income/capital-gains tax calculation.")
                print("  Use --json or --output to inspect dated distributions, uncertainty and annual projections.")
        except (DividendError,FXRateError,MarketDataError,ValueError,OSError,TypeError) as exc:
            print(f"DIVIDENDS ERROR: {exc}",file=sys.stderr)
            return 1
        return 0

    if args.command == "insights":
        if (args.start >= args.end or args.end > date.today()+timedelta(days=1)
                or args.as_of > date.today() or args.as_of < args.start
                or not math.isfinite(args.risk_free) or args.risk_free <= -100
                or args.max_fx_age_days < 0 or args.min_observations < 30):
            print("V8 INPUT ERROR: invalid date window, rates or observation minimum",file=sys.stderr)
            return 2
        try:
            choices=_parse_v8_scenarios(args.scenario)
            priority=_parse_v8_priorities(args.priorities)
            robust_requested=(args.robustness or args.three_strategies or bool(args.asset_cap or args.asset_floor or args.group_cap)
                              or args.turnover_cap is not None or args.rebalance!='monthly'
                              or args.cost_bps!=10 or args.train_periods!=252
                              or args.test_periods!=126 or args.step_periods!=126)
            robust_options=None
            if robust_requested:
                if args.turnover_cap is not None and not 0<=args.turnover_cap<=100:
                    raise InsightError("--turnover-cap must be 0..100 percent")
                robust_options={"max_weight":args.max_weight,
                    "asset_caps":_v83_percent_assignments(args.asset_cap,"--asset-cap"),
                    "asset_floors":_v83_percent_assignments(args.asset_floor,"--asset-floor"),
                    "group_caps":_v83_group_limits(args.group_cap),
                    "max_turnover":args.turnover_cap/100 if args.turnover_cap is not None else None,
                    "rebalance":args.rebalance,"cost_bps":args.cost_bps,
                    "train_periods":args.train_periods,"test_periods":args.test_periods,
                    "step_periods":args.step_periods}
            provider=YahooFinanceProvider()
            cache=HistoryCache(args.cache_dir)
            fx_quote=None
            if any(h.currency!=args.base for h in portfolio.holdings):
                fx_quote=(manual_fx_quote(args.usd_cad,args.as_of) if args.usd_cad is not None
                          else latest_usd_cad_quote(provider,cache,args.as_of,
                              max_age_days=args.max_fx_age_days,refresh=args.refresh))
            valuation=value_portfolio(portfolio,base_currency=args.base,fx_quote=fx_quote)
            result=run_insights_from_market_data(
                portfolio,valuation,provider,cache,start=args.start,end=args.end,
                rf=args.risk_free/100.,scenarios=choices,priorities=priority,
                max_weight=args.max_weight,min_observations=args.min_observations,
                max_fx_age_days=args.max_fx_age_days,refresh=args.refresh,
                mc_paths=args.mc_paths,mc_horizon=args.mc_horizon,seed=args.seed,
                compare_currencies=args.compare_currencies,robustness=robust_options,
                three_strategies=args.three_strategies)
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
            if args.json:
                print(json.dumps(result,indent=2,allow_nan=False))
            else:
                print(f"V8 INSIGHTS | {result['base_currency']} | {result['first_date']} to "
                      f"{result['last_date']} | {result['return_periods']} aligned returns")
                for label,metrics in (("Portfolio",result['portfolio']),*result['benchmarks'].items()):
                    sharpe=metrics['sharpe']
                    print(f"  {label:<12} CAGR {metrics['cagr']:.2%} | vol {metrics['annualized_volatility']:.2%} "
                          f"| Sharpe {sharpe:.2f}" if sharpe is not None else
                          f"  {label:<12} Sharpe unavailable")
                for label in ("positive","negative","neutral","unavailable"):
                    print(f"  {label.upper()} ({len(result['diagnostics'][label])})")
                    for alert in result['diagnostics'][label][:4]:
                        print("    ",alert['evidence'])
                if 'rolling_robustness' in result:
                    rolling=result['rolling_robustness']
                    print("V8.3 ROLLING VALIDATION:", rolling.get('conclusion',rolling.get('reason')))
                    print("  Folds:",rolling.get('fold_count',0))
                    if rolling.get('next_step'):print("  Next:",rolling['next_step'])
                print("ACTIONABLE SUMMARY:",result['actionable_diagnostics']['conclusion'])
                chosen=result['historical_strategy_selection']
                print("LEGACY V8 SINGLE SPLIT (DAILY, UNCONSTRAINED):" if 'rolling_robustness' in result
                      else "TRAIN/HOLDOUT STRATEGY:",
                      chosen['selected'] if chosen['status']=='evaluated' else chosen['reason'])
                for warning in result['warnings']:
                    print("  NOTE:",warning)
                print("Use --json or --output to inspect full diagnostics, attribution, scenarios and simulations")
        except (ValueError,InsightError,FXRateError,MarketDataError,OSError) as exc:
            print(f"V8 INSIGHTS ERROR: {exc}",file=sys.stderr)
            return 1
        return 0

    if args.command == "factors":
        if (args.start >= args.end or args.end > date.today()+timedelta(days=1)
                or args.as_of > date.today() or args.as_of < args.start):
            print("DATE ERROR: invalid factor sample or snapshot cutoff", file=sys.stderr)
            return 2
        if (args.max_fx_age_days < 0 or args.min_observations < 3
                or not math.isfinite(args.capital) or args.capital <= 0):
            print("INPUT ERROR: invalid factor sample, capital or FX age", file=sys.stderr)
            return 2
        try:
            from portfolio_risk.input import normalize_symbol
            symbols = {}
            for label in args.proxies:
                parts = label.split(":")
                if len(parts) != 2 or parts[1].upper() not in {"CAD", "USD"}:
                    raise FactorProxyError("Use TICKER:CAD or TICKER:USD for each proxy")
                sym = normalize_symbol(parts[0])
                if sym in symbols:
                    raise FactorProxyError(f"Duplicate factor proxy {sym}")
                symbols[sym] = parts[1].upper()
            provider = YahooFinanceProvider()
            cache = HistoryCache(args.cache_dir)
            fx_quote = None
            if any(h.currency != args.base for h in portfolio.holdings):
                fx_quote = (manual_fx_quote(args.usd_cad, args.as_of)
                            if args.usd_cad is not None else latest_usd_cad_quote(
                                provider, cache, args.as_of,
                                max_age_days=args.max_fx_age_days, refresh=args.refresh))
            valuation = value_portfolio(portfolio, base_currency=args.base, fx_quote=fx_quote)
            bt = run_backtest_from_market_data(
                portfolio, valuation, provider, cache, start=args.start, end=args.end,
                initial_capital=args.capital, schedules=("buy_hold",),
                min_observations=args.min_observations,
                max_fx_age_days=args.max_fx_age_days, refresh=args.refresh,
            )
            metrics = run_factor_proxy_from_backtest(
                bt, provider, cache, symbols=symbols, start=args.start, end=args.end,
                max_fx_age_days=args.max_fx_age_days,
                min_observations=args.min_observations, refresh=args.refresh,
            )
        except (ValueError, FactorProxyError, BacktestError, FXRateError,
                MarketDataError, OSError) as exc:
            print(f"FACTOR PROXY ERROR: {exc}", file=sys.stderr)
            return 1
        output = factor_proxy_as_dict(metrics)
        if args.json:
            print(json.dumps(output, indent=2, allow_nan=False))
        else:
            print("\nV7 historical ETF proxy sensitivity (NOT an orthogonal factor model)")
            print("Each beta is separate and uses the buy-and-hold historical simulation in base currency.")
            print(f"{'Proxy':<14}{'Periods':>10}{'Beta':>12}{'Corr':>12}{'R^2':>12}{'Alpha*':>13}")
            print('-'*73)
            for sym, item in metrics.items():
                fmt = lambda n: 'N/A' if n is None else f'{n:.3f}'
                alpha = 'N/A' if item.alpha_annualized_arithmetic is None else f'{item.alpha_annualized_arithmetic*100:.2f}%'
                print(f"{sym:<14}{item.observations:>10}{fmt(item.beta):>12}"
                      f"{fmt(item.correlation):>12}{fmt(item.r_squared):>12}{alpha:>13}")
            print("*Alpha is annualized regression intercept, not demonstrated trading alpha.")
            for warning in output['limitations']:
                print(f"  - {warning}")
        return 0

    if args.command == "exposure":
        if args.as_of > date.today() or args.max_holdings_age_days < 0:
            print("INPUT ERROR: invalid exposure cutoff or holdings age", file=sys.stderr)
            return 2
        if args.max_fx_age_days < 0:
            print("INPUT ERROR: invalid FX age", file=sys.stderr)
            return 2
        try:
            classes = load_asset_metadata(args.assets_file) if args.assets_file else {}
            snapshots = load_fund_snapshots(args.holdings_file) if args.holdings_file else {}
            for raw in args.yahoo_top_holdings:
                from portfolio_risk.input import normalize_symbol
                symbol = normalize_symbol(raw)
                if symbol not in {h.symbol for h in portfolio.holdings}:
                    raise ExposureError(f"{symbol}: not a direct holding; select an ETF ticker from your portfolio")
                if symbol in snapshots:
                    raise ExposureError(f"{symbol}: already supplied in --holdings-file")
                snapshots[symbol] = fetch_yahoo_top_holdings(symbol)
            tickerinside_data = {}
            metadata_unsupported = {}
            if args.tickerinside:
                api = TickerInsideClient()
                supported, _ = api.index(refresh=args.refresh)
                for holding in portfolio.holdings:
                    symbol = holding.symbol
                    if symbol not in supported or symbol in snapshots:
                        continue
                    try:
                        item = api.fund(symbol, refresh=args.refresh)
                        tickerinside_data[symbol] = item
                        if item.exposure_support in {"equity", "spot_asset"} and item.holdings_as_of is not None and item.known_pct <= 100.00001:
                            snapshots[symbol] = item.to_snapshot()
                        else:
                            classes.setdefault(symbol, SecurityMetadata(symbol, 'etf'))
                    except (TickerInsideError, ExposureError) as exc:
                        metadata_unsupported[symbol] = f"Holdings retrieval/validation failed: {exc}; only price-risk analysis may work"
                        print(f"TICKERINSIDE WARNING: {symbol}: {exc}", file=sys.stderr)
            if args.auto_stock_metadata:
                if not 1 <= args.metadata_limit <= 500:
                    raise ExposureError("--metadata-limit must be between 1 and 500")
                from portfolio_risk.exposure import UNKNOWN
                lookup = YahooSecurityMetadataProvider()
                for holding in portfolio.holdings[:args.metadata_limit]:
                    symbol = holding.symbol
                    if symbol in snapshots or (symbol in classes and classes[symbol].asset_class != "UNKNOWN"):
                        continue
                    profile = lookup.lookup(symbol)
                    if profile.status == "unavailable":
                        print(f"YAHOO METADATA WARNING: {symbol}: {profile.note}", file=sys.stderr)
                        continue
                    if profile.asset_kind in {"commodity_future", "index", "currency"}:
                        metadata_unsupported[symbol] = (
                            f"{profile.asset_kind} is outside V7 equity/spot-asset exposure scope; "
                            "price-risk may still work")
                        continue
                    kind = {"equity": "equity", "fund": "etf", "crypto": "crypto"}.get(profile.asset_kind)
                    if kind:
                        classes[symbol] = SecurityMetadata(
                            symbol, kind, canonical_sector(profile.sector) if profile.sector and kind == "equity" else UNKNOWN,
                            canonical_country(profile.company_country) if profile.company_country and kind == "equity" else UNKNOWN,
                        )
                if len(portfolio.holdings) > args.metadata_limit:
                    print("YAHOO METADATA WARNING: direct asset classification capped; "
                          "remaining classifications UNKNOWN", file=sys.stderr)
            fx_quote = None
            if any(h.currency != args.base for h in portfolio.holdings):
                fx_quote = (manual_fx_quote(args.usd_cad, args.as_of)
                            if args.usd_cad is not None else latest_usd_cad_quote(
                                YahooFinanceProvider(), HistoryCache(args.cache_dir), args.as_of,
                                max_age_days=args.max_fx_age_days, refresh=args.refresh))
            valuation = value_portfolio(portfolio, base_currency=args.base, fx_quote=fx_quote)
            enrichment_report=None
            sector_enrichment={}
            if args.focus_sector and (not args.drilldown or not 0 <= args.focus_limit <= 200):
                raise ExposureError("--focus-sector requires --drilldown and --focus-limit 0..200")
            if args.drilldown and args.enrich_underlyings:
                classes,enrichment_report=enrich_top_equities(
                    {h.symbol:float(h.weight) for h in valuation.holdings},snapshots,classes,
                    limit=args.enrich_underlyings,cache_dir=args.underlying_metadata_cache,
                    metadata_provider=YahooSecurityMetadataProvider())
            elif args.enrich_underlyings:
                raise ExposureError("--enrich-underlyings requires --drilldown")
            if args.focus_sector:
                sector_map={s: f.sector_pct for s,f in tickerinside_data.items()
                            if s in snapshots and f.exposure_support=='equity'}
                for focus in args.focus_sector:
                    key=canonical_sector(focus)
                    if key in sector_enrichment or key not in set(SECTORS.values()):
                        raise ExposureError("Focus sectors must be distinct, canonical names (e.g. financials)")
                    classes,sector_enrichment[key]=enrich_top_equities(
                        {h.symbol:float(h.weight) for h in valuation.holdings},
                        snapshots,classes,focus_sector=key,fund_sector_pct=sector_map,
                        limit=args.focus_limit,cache_dir=args.underlying_metadata_cache,
                        metadata_provider=YahooSecurityMetadataProvider())
            demo = bool(args.holdings_file and any(
                'SYNTHETIC' in snap.source.upper() for snap in snapshots.values()))
            aggregate = {s: (f.sector_pct, f.country_pct)
                         for s, f in tickerinside_data.items()
                         if s in snapshots and f.exposure_support == "equity"}
            report = analyze_exposure(
                valuation, snapshots=snapshots, classifications=classes,
                analysis_date=args.as_of, max_snapshot_age_days=args.max_holdings_age_days,
                demo_data=demo, fund_aggregates=aggregate,
            )
            support = assess_exposure_support(
                [h.symbol for h in portfolio.holdings], classifications=classes,
                snapshots=snapshots, provider_records=tickerinside_data,
                cutoff=args.as_of, max_age_days=args.max_holdings_age_days,
                unsupported_metadata=metadata_unsupported,
                effective_coverage={s: c["identified_fraction_of_fund"]
                                    for s, c in report.coverage_by_fund.items()},
            )
            if args.strict_exposure:
                blocked = {s: x["reason"] for s, x in support.items() if x["exposure_status"] != "supported"}
                if blocked:
                    raise ExposureError("Strict exposure rejected: " + "; ".join(f"{s}: {r}" for s, r in blocked.items()))
        except (ValueError, ExposureError, FXRateError, MarketDataError, OSError) as exc:
            print(f"EXPOSURE ERROR: {exc}", file=sys.stderr)
            return 1
        drilldown = None
        if args.drilldown:
            try:
                drilldown=explain_exposure(
                    report,snapshots=snapshots,classifications=classes,
                    fund_aggregates=aggregate,top_n=args.top_holdings)
            except ValueError as exc:
                print(f"EXPOSURE DRILLDOWN ERROR: {exc}",file=sys.stderr)
                return 1
        if args.json:
            document = exposure_as_dict(report)
            if drilldown is not None:
                document["drilldown"] = drilldown
                if enrichment_report is not None:
                    document["drilldown"]["metadata_enrichment"] = enrichment_report
                if sector_enrichment:
                    document["drilldown"]["sector_enrichment"] = sector_enrichment
            document["asset_support"] = support
            document["restricted_exposure_assets"] = [s for s, x in support.items() if x["exposure_status"] != "supported"]
            if drilldown is not None:
                from portfolio_risk.v83_exposure_actions import exposure_actions
                document["exposure_actions"] = exposure_actions(document)
            print(json.dumps(document, indent=2, allow_nan=False))
        else:
            _print_exposure(report)
            if drilldown is not None:
                if enrichment_report is not None:
                    print("  Underlying stock metadata: "
                          f"{enrichment_report['enriched_count']}/{enrichment_report['attempted']} enriched "
                          f"({enrichment_report['cached']} cache hits)")
                print("\nV8 sector-by-investment breakdown (issuer sector totals may differ from per-stock classifications):")
                for sector,data in sorted(drilldown['sector_drilldown'].items(),
                                          key=lambda x:-x[1]['portfolio_fraction']):
                    print(f"  {sector}: {data['portfolio_fraction']:.2%} of TOTAL portfolio")
                    for investment,weight in list(data['by_investment'].items())[:5]:
                        print(f"    {investment}: {weight:.2%} portfolio contribution")
                print("\nV8 fund top holdings:")
                for fund,data in drilldown['fund_top_holdings'].items():
                    print(f"  {fund} (as-of {data['snapshot_as_of']}; "
                          f"unresolved portfolio fraction {data['unidentified_portfolio_fraction']:.2%}):")
                    for item in data['top_holdings'][:10]:
                        print(f"    {item['symbol']} {item['fund_fraction']:.2%} of fund, "
                              f"{item['portfolio_fraction']:.2%} portfolio")
                for item in drilldown['overlap_contributors'][:5]:
                    print(f"  OVERLAP {item['fund_a']} / {item['fund_b']} "
                          f"{item['overlap_lower_bound']:.2%} common-stock lower bound")
                    for company in item['largest_contributors'][:5]:
                        print(f"    {company['symbol']}: "
                              f"{company['identified_overlap_contribution']:.2%} overlap contribution")
                print("Company-by-sector coverage is limited to individual securities whose metadata is available.")
            print("\nV7.3 supported exposure by portfolio ticker (prices evaluated separately):")
            for symbol, entry in support.items():
                print(f"  {symbol}: {entry['exposure_status'].upper()} — {entry['reason']}")
            if any(x['exposure_status'] != 'supported' for x in support.values()):
                print("  NOTE: Unsupported exposure is not a claim that historical-price/risk analysis fails.")
            if args.tickerinside and tickerinside_data:
                print("\nTickerInside fund-level aggregates (CC BY 4.0; dated snapshots):")
                for sym, data in tickerinside_data.items():
                    print(f"  {sym}: {len(data.holdings)} reported positions; holdings date {data.holdings_as_of}; "
                          f"source {data.source_layer}; validation {data.equity_validation}; {data.source_url}")
                    if data.non_equity_positions:
                        print(f"    Excluded futures/cash positions: {len(data.non_equity_positions)}; "
                              "these are not interpreted as equity shares")
                    if data.sector_pct:
                        print("    Sectors: "+", ".join(f"{k} {v:.1f}%" for k, v in
                              sorted(data.sector_pct.items(),key=lambda kv:-kv[1])[:5]))
                    if data.country_pct:
                        print("    Countries: "+", ".join(f"{k} {v:.1f}%" for k, v in
                              sorted(data.country_pct.items(),key=lambda kv:-kv[1])[:5]))
                print("  Data: TickerInside (CC BY 4.0), https://tickerinside.com")
        return 0

    if args.command == "construct":
        if args.start >= args.end or args.end > date.today()+timedelta(days=1):
            print("DATE ERROR: invalid period (end date is exclusive)", file=sys.stderr)
            return 2
        if args.as_of > date.today() or args.as_of < args.start:
            print("DATE ERROR: invalid --as-of cutoff", file=sys.stderr)
            return 2
        if (args.max_fx_age_days < 0 or args.lookback < 30 or
                args.momentum_skip < 0 or args.momentum_skip > args.lookback or
                not math.isfinite(args.capital) or args.capital <= 0 or
                not math.isfinite(args.fee_bps) or not 0 <= args.fee_bps <= 10000 or
                not math.isfinite(args.risk_free) or args.risk_free <= -100 or
                not math.isfinite(args.max_weight) or not 0 < args.max_weight <= 1 or
                not math.isfinite(args.shrinkage) or not 0 <= args.shrinkage <= 1 or
                len(args.methods) != len(set(args.methods)) or args.top_k <= 0):
            print("INPUT ERROR: invalid construction options", file=sys.stderr)
            return 2
        if (not args.benchmark) and args.benchmark_currency is not None:
            print("INPUT ERROR: benchmark currency requires a benchmark", file=sys.stderr)
            return 2
        try:
            provider = YahooFinanceProvider()
            cache = HistoryCache(args.cache_dir)
            fx_quote = None
            if any(h.currency != args.base for h in portfolio.holdings):
                fx_quote = (manual_fx_quote(args.usd_cad, args.as_of)
                            if args.usd_cad is not None else latest_usd_cad_quote(
                                provider, cache, args.as_of,
                                max_age_days=args.max_fx_age_days, refresh=args.refresh))
            valuation = value_portfolio(portfolio, base_currency=args.base, fx_quote=fx_quote)
            benchmark = args.benchmark.strip().upper() if args.benchmark else None
            bt = run_backtest_from_market_data(
                portfolio, valuation, provider, cache,
                start=args.start, end=args.end,
                initial_capital=args.capital, fee_bps=args.fee_bps,
                schedules=("buy_hold",), annual_risk_free_rate=args.risk_free/100,
                benchmark_symbol=benchmark, benchmark_quote_currency=args.benchmark_currency,
                min_observations=args.lookback+(args.momentum_skip if "momentum_top_k" in args.methods else 0)+31,
                max_fx_age_days=args.max_fx_age_days, refresh=args.refresh,
            )
            output = run_construction_from_backtest(
                bt, frequency=args.rebalance, lookback=args.lookback,
                momentum_skip=args.momentum_skip, top_k=args.top_k,
                max_weight=args.max_weight, shrinkage=args.shrinkage,
                evaluation_start=args.evaluation_start, methods=tuple(args.methods),
            )
        except (ValueError, ConstructionError, BacktestError, FXRateError,
                MarketDataError, OSError) as exc:
            print(f"CONSTRUCTION ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            result = construction_as_dict(output.report)
            result["market_data"] = output.market_data
            result["data_quality_warnings"] = list(output.quality_warnings)
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            _print_construction(output)
        return 0

    if args.command == "risk":
        if args.start >= args.end or args.end > date.today() + timedelta(days=1):
            print("DATE ERROR: invalid date window (end is exclusive)", file=sys.stderr)
            return 2
        if args.as_of > date.today() or args.as_of < args.start:
            print("DATE ERROR: --as-of must be after start and not future", file=sys.stderr)
            return 2
        if (args.max_fx_age_days < 0 or args.min_observations < 3 or
                args.lookback < 30 or not math.isfinite(args.capital) or args.capital <= 0 or
                not math.isfinite(args.fee_bps) or not 0 <= args.fee_bps <= 10000 or
                not math.isfinite(args.risk_free) or args.risk_free <= -100):
            print("INPUT ERROR: invalid risk options, fees, capital or lookback", file=sys.stderr)
            return 2
        try:
            shocks = _parse_shocks(args.shock)
            provider = YahooFinanceProvider()
            cache = HistoryCache(args.cache_dir)
            fx_quote = None
            if any(h.currency != args.base for h in portfolio.holdings):
                fx_quote = (manual_fx_quote(args.usd_cad, args.as_of)
                            if args.usd_cad is not None else latest_usd_cad_quote(
                                provider, cache, args.as_of,
                                max_age_days=args.max_fx_age_days, refresh=args.refresh))
            valuation = value_portfolio(portfolio, base_currency=args.base, fx_quote=fx_quote)
            bt_run = run_backtest_from_market_data(
                portfolio, valuation, provider, cache,
                start=args.start, end=args.end,
                initial_capital=args.capital, fee_bps=args.fee_bps,
                schedules=(args.strategy,), annual_risk_free_rate=args.risk_free / 100,
                min_observations=args.min_observations,
                max_fx_age_days=args.max_fx_age_days, refresh=args.refresh,
            )
            result = run_risk_from_backtest(
                bt_run, strategy=args.strategy, lookback=args.lookback,
                shocks=shocks or None,
            )
        except (ValueError, RiskError, BacktestError, FXRateError,
                MarketDataError, OSError) as exc:
            print(f"RISK ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            output = risk_as_dict(result.report, include_series=True)
            output["market_data"] = result.market_data
            output["data_quality_warnings"] = list(result.quality_warnings)
            print(json.dumps(output, indent=2, allow_nan=False))
        else:
            _print_risk(result)
        return 0

    if args.command == "backtest":
        if args.start >= args.end or args.end > date.today() + timedelta(days=1):
            print("DATE ERROR: invalid date window (end is exclusive)", file=sys.stderr)
            return 2
        if args.as_of > date.today() or args.as_of < args.start:
            print("DATE ERROR: --as-of must be after the start and not in the future",
                  file=sys.stderr)
            return 2
        if args.max_fx_age_days < 0 or args.min_observations < 3:
            print("INPUT ERROR: invalid FX age or minimum observations", file=sys.stderr)
            return 2
        if (not math.isfinite(args.capital) or args.capital <= 0 or
                not math.isfinite(args.fee_bps) or not 0 <= args.fee_bps <= 10000 or
                not math.isfinite(args.risk_free) or args.risk_free <= -100 or
                len(args.strategies) != len(set(args.strategies))):
            print("INPUT ERROR: invalid initial capital, fees, risk-free rate or duplicate strategy",
                  file=sys.stderr)
            return 2
        if not args.benchmark and args.benchmark_currency is not None:
            print("INPUT ERROR: benchmark currency requires --benchmark", file=sys.stderr)
            return 2
        benchmark = args.benchmark.upper().strip() if args.benchmark else None
        try:
            provider = YahooFinanceProvider()
            cache = HistoryCache(args.cache_dir)
            fx_quote = None
            if any(h.currency != args.base for h in portfolio.holdings):
                fx_quote = (manual_fx_quote(args.usd_cad, args.as_of)
                            if args.usd_cad is not None else latest_usd_cad_quote(
                                provider, cache, args.as_of,
                                max_age_days=args.max_fx_age_days, refresh=args.refresh,
                            ))
            valuation = value_portfolio(portfolio, base_currency=args.base, fx_quote=fx_quote)
            run = run_backtest_from_market_data(
                portfolio, valuation, provider, cache,
                start=args.start, end=args.end,
                initial_capital=args.capital, fee_bps=args.fee_bps,
                schedules=tuple(args.strategies),
                annual_risk_free_rate=args.risk_free / 100.0,
                benchmark_symbol=benchmark, benchmark_quote_currency=args.benchmark_currency,
                min_observations=args.min_observations,
                max_fx_age_days=args.max_fx_age_days, refresh=args.refresh,
            )
        except (ValueError, BacktestError, FXRateError, MarketDataError, OSError) as exc:
            print(f"BACKTEST ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            output = backtest_as_dict(run.report)
            output["market_data"] = run.source_info
            output["data_quality_warnings"] = list(run.quality_warnings)
            output["valuation_weights_fx"] = None if valuation.fx_quote is None else {
                "rate_cad_per_usd": str(valuation.fx_quote.usd_cad),
                "observed_on": valuation.fx_quote.observed_on.isoformat(),
                "source": valuation.fx_quote.source,
            }
            print(json.dumps(output, indent=2, allow_nan=False))
        else:
            _print_backtest(run)
        return 0

    if args.command == "performance":
        if args.start >= args.end or args.end > date.today() + timedelta(days=1):
            print("DATE ERROR: invalid analysis window (end is exclusive; no future data)", file=sys.stderr)
            return 2
        if args.as_of > date.today() or args.as_of < args.start:
            print("DATE ERROR: --as-of must be within/after the window start and not future",
                  file=sys.stderr)
            return 2
        if args.max_fx_age_days < 0 or args.min_observations < 3:
            print("INPUT ERROR: invalid FX age or minimum observations", file=sys.stderr)
            return 2
        benchmark = args.benchmark.upper().strip() if args.benchmark else None
        try:
            provider = YahooFinanceProvider()
            cache = HistoryCache(args.cache_dir)
            fx_quote = None
            if any(h.currency != args.base for h in portfolio.holdings):
                fx_quote = (manual_fx_quote(args.usd_cad, args.as_of)
                            if args.usd_cad is not None else latest_usd_cad_quote(
                                provider, cache, args.as_of,
                                max_age_days=args.max_fx_age_days,
                                refresh=args.refresh,
                            ))
            valuation = value_portfolio(portfolio, base_currency=args.base, fx_quote=fx_quote)
            result = run_performance(
                portfolio, valuation, provider, cache,
                start=args.start, end=args.end,
                benchmark_symbol=benchmark,
                benchmark_quote_currency=args.benchmark_currency,
                annual_risk_free_rate=args.risk_free / 100.0,
                min_observations=args.min_observations,
                max_fx_age_days=args.max_fx_age_days,
                refresh=args.refresh,
            )
        except (ValueError, FXRateError, MarketDataError, PerformanceError, OSError) as exc:
            print(f"PERFORMANCE ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            output = performance_as_dict(result.report)
            output["market_data"] = result.source_info
            output["data_quality_warnings"] = list(result.quality_warnings)
            output["valuation_weights_fx"] = None if valuation.fx_quote is None else {
                "rate_cad_per_usd": str(valuation.fx_quote.usd_cad),
                "observed_on": valuation.fx_quote.observed_on.isoformat(),
                "source": valuation.fx_quote.source,
            }
            print(json.dumps(output, indent=2, allow_nan=False))
        else:
            _print_performance(result)
        return 0

    if args.command == "analyze":
        if args.as_of > date.today():
            print("DATE ERROR: --as-of must not be in the future", file=sys.stderr)
            return 2
        if args.max_fx_age_days < 0:
            print("FX ERROR: --max-fx-age-days must be nonnegative", file=sys.stderr)
            return 2
        try:
            fx_quote = None
            if any(h.currency != args.base for h in portfolio.holdings):
                if args.usd_cad is not None:
                    fx_quote = manual_fx_quote(args.usd_cad, args.as_of)
                else:
                    fx_quote = latest_usd_cad_quote(
                        YahooFinanceProvider(), HistoryCache(args.cache_dir), args.as_of,
                        max_age_days=args.max_fx_age_days, refresh=args.refresh,
                    )
            summary = value_portfolio(portfolio, base_currency=args.base, fx_quote=fx_quote)
        except (FXRateError, MarketDataError, ValueError, OSError) as exc:
            print(f"VALUATION ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(valuation_as_dict(summary), indent=2))
        else:
            _print_valuation(summary)
        return 0

    print(f"Validated {len(portfolio.holdings)} positions: {args.portfolio}")
    for holding in portfolio.holdings:
        print(f"  {holding.symbol:<12} {holding.market_value:>12} {holding.currency}")
    if args.command == "validate":
        print("PASS: input format valid. No API requests or FX conversions were performed.")
        return 0

    if args.start >= args.end:
        print("DATE ERROR: --start must be before --end", file=sys.stderr)
        return 2
    provider = YahooFinanceProvider()
    cache = HistoryCache(args.cache_dir)
    symbols = [holding.symbol for holding in portfolio.holdings]
    if args.include_fx and len({holding.currency for holding in portfolio.holdings}) > 1:
        fx = fx_ticker("USD", "CAD")
        if fx:
            symbols.append(fx)
    failures = 0
    print(f"\nAdjusted daily close prices: [{args.start}, {args.end})  |  provider=Yahoo")
    for symbol in symbols:
        try:
            result = get_validated_history(provider, cache, symbol, args.start, args.end,
                                           refresh=args.refresh)
            quality = result.quality
            where = "CACHE" if result.cached else "LIVE"
            price_ccy = result.history.quote_currency or "UNKNOWN"
            print(f"  {symbol:<12} [{where}] {quality.observation_count:>4} rows | "
                  f"{quality.first_date} .. {quality.last_date} | quote={price_ccy}")
            for warning in quality.warnings:
                print(f"      WARNING: {warning}")
        except (MarketDataError, ValueError, OSError) as exc:
            failures += 1
            print(f"  {symbol:<12} [FAILED] {exc}")
    print(f"\nResult: {len(symbols) - failures}/{len(symbols)} symbols fetched and validated.")
    print("Fetch checks market data. Use 'analyze', 'performance', or 'backtest' for portfolio analytics.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
