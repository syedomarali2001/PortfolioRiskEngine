# V8.3 Portfolio Improvement & Robustness — Test Guide

**Status:** backend update built and tested offline; live results must be checked on your own computer. **No GUI**. Existing V1–V8.2 CLI commands continue to work.

## Purpose

V8.3 adds *research tools*, not trade recommendations or a promise to find a perfect portfolio:

1. Explicit allocation constraints (per ticker floors/caps, non-overlapping named group caps, one-way turnover cap). Constraints are **not inferred from ticker names**. An impossible 100%-sum allocation raises a clear error. The existing portfolio **remains the comparison baseline**, even if it violates the new candidate limits.
2. Multiple forward-only, expanding-train and non-overlapping holdout windows. Each fold regenerates candidate weights using only earlier data. Candidate selection is based on training metrics; the later test period never determines that fold's winner.
3. Periodic *drift-aware* rebalancing with the selected cadence, an explicit one-way turnover metric, and flat estimated transaction costs. Unlike the legacy V8 insights *daily constant-weight* report, V8.3 rolling tests simulate changing weights between rebalances. The two methodologies are labeled separately and can yield different results.
4. Benchmark-targeted and concentration-based **actionable diagnostics**: evidence, sentiment, comparison threshold, a numeric target and a falsifiable next test. If the later windows do not confirm consistent improvement, it explicitly says not to recommend changing the allocation on this evidence.
5. Optional sector-focused constituent metadata lookups. The source ETF's published sector totals identify relevant funds, **not individual holdings**. Only an independently verified company sector is added to named contributors. Requests are capped and share the existing cache.
6. A small set of ETF overlap and unresolved holdings quality prompts. Unknown exposure is never treated as zero or a reason to sell a fund.

## Step 1 — Install

Extract `PortfolioRiskEngine_V8_3_Update.zip` into your existing `PortfolioRiskEngine` directory, replacing matching source, docs, examples and tests. Do **not** delete your existing `data/cache`, prior versions, local reports or ZIP backups. This update adds no dependencies.

```powershell
python -m pytest
```

Run from the project root. The expected offline test count is printed in the accompanying release note. If an import error appears, confirm that the project's editable installation / Python path is still configured as it was for V8.2.

## Step 2 — Real market-data insights with practical limits

This reuses the same **AAPL 35%, MSFT 20%, SPY 30%, GLD 15%** sample portfolio and matched SPY / RSP benchmarks.

```powershell
python -m portfolio_risk insights examples\v8_insights_portfolio.csv --base USD --start 2022-11-01 --end 2026-11-08 --as-of 2026-11-07 --priorities sharpe:2,volatility:1,drawdown:1 --robustness --asset-cap AAPL=25 --asset-cap GLD=20 --asset-floor SPY=25 --group-cap bigtech:AAPL,MSFT=40 --turnover-cap 30 --rebalance quarterly --cost-bps 10 --train-periods 252 --test-periods 126 --step-periods 126 --output v83_robustness_report.json
```

If you want the full console JSON, append `--json`; the `--output` command above already saves the complete report.

**Fields to inspect:**

- `rolling_robustness.status`, `fold_count`, `constraints.baseline_compliant`
- `rolling_robustness.folds`: training dates, holdout dates, chosen candidate weights, original/candidate metrics, transaction costs and per-priority verdicts
- `rolling_robustness.per_metric_tally`: actual win/neutral/loss counts for the priority metrics
- `rolling_robustness.constraint_compliance_example`: separately labeled feasible target allocation *without* a claim of better returns; especially important when `baseline_compliant=false`
- `rolling_robustness.unvalidated_trailing_return_periods`: recent observations omitted from the final full-length holdout (they are not silently used as validation evidence)
- `rolling_robustness.consistent_improvement_on_all_priorities`, `conclusion`, `next_step`
- `actionable_diagnostics`: actionable benchmark targets, concentrated-volatility evidence and validation status
- `historical_strategy_selection`: **legacy** single 70/30 daily constant-weight test, retained for backward compatibility; it is **not constrained by V8.3**. Prefer the V8.3 rolling output for cost-aware multi-window decisions.

Each rolling holdout starts from the original sample allocation separately. These folds do **not** represent continuous real investor account transactions. Annualized historical metrics remain uncertain; user constraints and bps fees are assumptions. A candidate is chosen from a modest, fixed family, **not a global optimum**. The program avoids treating neutral results as positive validation. The reported consistency rule requires a material positive difference in at least 75% of non-overlapping holdout windows and **zero materially negative windows** for **every selected priority**.

### Constrained candidate eligibility

In the example, **AAPL was originally 35%** but the cap is **25%**. The original portfolio is shown for comparison, but **is not eligible as a recommended constrained allocation**. Group caps apply to ticker symbols explicitly listed by the user; no inferred 'technology' or 'crypto' group membership. Names of group members cannot overlap across groups. `--turnover-cap` is *initial* one-way turnover measured as half of total absolute changes from the original allocation to each candidate target. The cumulative turnover printed for an entire holdout additionally includes periodic rebalances and may exceed this initial cap; the two numbers must not be conflated. V8.3.2 reports both explicitly.

When the specified turnover is too small to satisfy the other caps, the report says candidate validation is unavailable rather than breaking limits or hiding the shortfall.

## Step 3 — Targeted ETF Financials drill-down

```powershell
python -m portfolio_risk exposure examples\sample_portfolio_v3.csv --base CAD --as-of 2026-11-07 --tickerinside --auto-stock-metadata --drilldown --enrich-underlyings 25 --focus-sector financials --focus-limit 40 --json > v83_sector_drilldown.json
```

Check `drilldown.sector_enrichment.financials`, `drilldown.sector_drilldown.financials`, and `exposure_actions`. The bounded provider lookups can be rate-limited; therefore the result may contain fewer classified Financials companies than the sector aggregate would suggest. The issuer's sector total and the verified stock sum **must remain separately identified**. A failed query is missing evidence, not a negative stock classification. XEQT.TO may still lack constituent data; avoid inventing it.

If live metadata calls take too long, reduce `--focus-limit` to 10. Existing positive metadata cache entries are reused.

## Step 4 — Optional negative/infeasible-input checks

Run an intentionally impossible cap combination on the four-asset example:

```powershell
python -m portfolio_risk insights examples\v8_insights_portfolio.csv --base USD --start 2022-11-01 --end 2026-11-08 --as-of 2026-11-07 --robustness --asset-cap AAPL=10 --asset-cap MSFT=10 --asset-cap SPY=10 --asset-cap GLD=10
```

**Expected:** clear *infeasible limits* error. Do not include this intentionally failing command as a normal success test. Likewise, the existing SPY/RSP comparison, V8.2 DCF, and V7.3 ETF provenance safeguards must still pass in the inherited suite.

## V9 interface requirements

Keep V8.3's `rolling_robustness` and `actionable_diagnostics` JSON as independent backend contracts. The GUI should surface clear fold-by-fold evidence, a **cost assumption** control, asset/group constraint controls and a prominent message when optimization fails holdout validation. Do not hide warnings behind a default green/red portfolio score.

## Known limitations and boundaries

- Data from free sources can be stale, incomplete, unavailable or subject to personal-use terms; don't assume redistribution rights.
- Current ETF holdings snapshots are **not historical** exposure data; the portfolio's historical risk metrics do not use historical issuer holdings.
- Risk contribution is a volatility decomposition, **not** a guarantee about future losses.
- Trading costs are flat bps assumptions; taxes, corporate actions / funding constraints, market impact and share-lot rounding are not individually modelled.
- Some portfolio assets or user-selected groups cannot satisfy tight caps. No silent renormalization or constraint violations are allowed.
- Macro-economic forecasts, realized dividend tax effects and full retirement cashflow simulation are out of scope.

## V8.3.2 Reporting clarification

When the original allocation violates mandatory user limits, the engine will not suggest simply retaining it as a compliant choice, even if no tested allocation improves all selected risk metrics. It reports a feasible constraint-compliance example separately from historical improvement evidence. This example is not a recommendation or a risk-validated strategy. The JSON also distinguishes sector-enrichment `selected_top_equity_weights` (actual total-portfolio equity fractions) from `selection_ranking_scores` (only an API lookup priority heuristic).
