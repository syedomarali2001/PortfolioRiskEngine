# V8.2 Live Validation Fix — Diagnostics, Strategy Priorities, FX Boundaries

Copy over an existing V8.1 installation. This update leaves V1–V7 and all
other V8 commands unchanged.

## Problem observed in 2026-11-07 live report

- Portfolio/benchmark volatility values were calculated as `annualized_volatility`,
  but V8 diagnostic rules, what-if comparisons, and historical strategy objective
  scoring asked for `volatility`. Alerts said `unavailable`, and a user-selected
  volatility objective contributed no discriminatory training score. Holdout
  comparisons displayed `None` and incorrectly returned neutral.
- Requested optional USD/CAD comparison returned `unavailable` because Yahoo's
  FX result included a bar outside the requested history window.

## Fixes

- The `volatility` *objective* is mapped to the stored annualized volatility
  metric everywhere the research engine reads it. The output report retains
  `annualized_volatility` as the canonical historical-statistics key.
- Holdout objective verdicts are marked `unavailable` if either side is missing.
- The V8-only USD/CAD historical-FX request may remove adjacent out-of-window
  observations if every such observation is at most 7 calendar days from a
  requested boundary. The retained data are validated and cached; this is
  logged in `data_quality_warnings`. A longer discrepancy or invalid in-window
  FX history is rejected. The price validation for stocks/ETFs remains strict.
- A later valid run may pick a different training winner because volatility
  preference actually contributes to the optimization ranking now.

## Verification

Run from the PortfolioRiskEngine root:

```powershell
python -m pytest
python -m portfolio_risk insights examples\v8_insights_portfolio.csv --base USD --start 2022-11-01 --end 2026-11-08 --as-of 2026-11-07 --priorities sharpe:2,drawdown:1,volatility:1 --scenario less_tech:AAPL=20,MSFT=10,SPY=50,GLD=20 --compare-currencies --output v8_insights_report_v82.json
```

Expected offline suite at packaging: 431 passing. Confirm the new JSON
includes `volatility` in each scenario's `vs_input`, actual `volatility`
positive/negative/neutral benchmark diagnostics, numeric volatility holdout
verdict values, and a `currency_comparison.status` of `evaluated` if live FX
has clean/adjacent-only data. If Yahoo still returns invalid FX, the currency
comparison may correctly stay `unavailable`; report its reason rather than
fabricate a currency conversion. All other analytics should be produced.

IMPORTANT: Hypothetical daily constant-mix allocations, 0% modeled costs,
ETF benchmarks are total-return proxies and not a prediction or recommendation.
