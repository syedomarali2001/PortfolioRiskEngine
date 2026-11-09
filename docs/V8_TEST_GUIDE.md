# Portfolio Risk Engine V8 — Explainable Diagnostics, Alternatives & Fundamentals

**Scope:** The Python analysis backend. No Streamlit/UI (V9), no paid services,
no investment advice, no claim of globally optimal allocations or data coverage.

## Installation

Unzip `PortfolioRiskEngine_V8_Update.zip` over the existing V7.3 project root,
replacing matching files. Keep your own `data/cache`, `.env` (if any), and
version backups. Run from the project root in PowerShell:

```powershell
python -m pytest
python -m portfolio_risk --help
```

`pyproject.toml` retains the V1-V7 required dependencies; no new packages.

## A. Portfolio diagnostics — same SPY/RSP dates and currency

```powershell
python -m portfolio_risk insights examples\v8_insights_portfolio.csv --base USD --start 2022-11-01 --end 2026-11-08 --as-of 2026-11-07 --priorities sharpe:2,drawdown:1,volatility:1 --scenario less_tech:AAPL=20,MSFT=10,SPY=50,GLD=20 --compare-currencies --output v8_insights_report.json
```

Optional `--json` writes the full report to stdout (use `> file.json` if desired);
`--output` writes it directly while also printing a human-readable summary.

**Inputs:** The portfolio CSV is the same market-value snapshot input as V1-V7,
with optional `quote_currency`. The `--scenario` percentages apply to the **same
assets** and must include every ticker and sum to **100.00%**. No missing assets
or silent reweighting. Zero-weight holdings are allowed in a hypothetical
scenario, but not in the entered snapshot portfolio.

**The report covers:**
- Portfolio/SPY/RSP metrics on the **same observed common-close dates**, after
  historical conversion into the requested CAD or USD base. SPY and RSP are
  ETF proxies, *not* the fee-free index series. Because both benchmark
  instruments are specifically USD-listed, missing Yahoo currency metadata for
  SPY/RSP alone is resolved to USD and logged as
  `benchmark_symbol_default`; conflicting CAD/USD metadata is rejected.
  Other unknown ticker quotation currencies are NEVER guessed.
- Sharpe, Sortino, CAGR, volatility, maximum drawdown depth, historical
  Expected Shortfall (95%), Calmar and relative beta/tracking error/information
  ratio versus both benchmarks, when defined.
- Per-metric, per-benchmark `positive`, `negative`, `neutral`, `unavailable`
  diagnostics. Every message contains values, direction, materiality rule,
  period length and a plain-language definition. `neutral` means **a
  measured but immaterial difference**; insufficient data is `unavailable`,
  **never** neutral. Below 126 returns, ratings are withheld.
- Deterministic top-three strengths and weaknesses, **without** collapsing
  opposite-direction outcomes into a vague overall verdict.
- Volatility risk attribution, user-provided what-if allocations, simple 5-day
  block-bootstrap Monte Carlo with a fixed seed, historical event-window
  results (only for complete windows), 1/3/5-year start-window sensitivity.
- Small allocation search among the input allocation, equal weight,
  inverse-volatility, minimum variance and predetermined blends. Preference
  scores are calculated on the **earlier 70%** of returns; only the selected
  candidate and current portfolio are evaluated on the **later 30%**.
  The training candidate scorecard and training-only Pareto frontier are
  inspectable. The holdout report explicitly says if each user-prioritized
  metric improved, worsened or changed insignificantly; no guarantee is made.
- Optional `--compare-currencies` prints a separate **matched-date** CAD/USD
  performance view using historical USD/CAD. This is reporting-currency
  sensitivity, **not** a full economic FX-risk or hedge simulator.

**Important:** Static input/scenario allocations are hypothetically rebalanced
**daily** for the V8 insights comparison (zero trading costs/taxes), regardless
of the account's real transactions. Use V4 and V6 for cost-aware calendar
rebalancing and walk-forward allocations. A selected strategy can perform
worse on the holdout period; that is **evidence, not an error**. Do not select
another candidate by inspecting holdout outcomes—this would leak the test set.
A single 70/30 split is a diagnostic, not a production investment model.

The materiality thresholds are **transparent research settings** (e.g. 0.20
Sharpe difference, 2 percentage points annual volatility, 5 points drawdown
or 2 points CAGR), not statistically validated universal definitions of
'good'. Never claim a ratio is excellent without context and sample quality.

## B. Exposure explainability drill-down

```powershell
python -m portfolio_risk exposure examples\sample_portfolio_v3.csv --base CAD --tickerinside --auto-stock-metadata --drilldown --enrich-underlyings 25 --top-holdings 25 --json > v8_exposure_drilldown.json
```

`--drilldown` without enrichment works offline when supplying cached
`--holdings-file`/`--assets-file`; it returns ETF-specific top holdings,
portfolio-effective constituent weights, portfolio-sector contributions **by
investment**, overlap contributors per common stock, and the amount still
unidentified for each ETF. Use `--enrich-underlyings 0` (default) for no
additional Yahoo calls; `1..200` restricts optional **top underlying equity**
lookups and caches successful profiles for 7 days in
`data/cache/v8_stock_metadata`.

`by_investment` can use TickerInside *fund-level* reported sector percentages.
`known_company_contributions` requires *individually classified* security
metadata from Yahoo (or your `--assets-file`). These are independent data
layers and can disagree by taxonomy or date. The report NEVER fabricates
individual company sectors from a fund's aggregate, and carries an
`unattributed_company_fraction`. XEQT/CAD ETF constituent gaps remain
identified by name/portfolio weight. Missing ETF overlap is UNKNOWN, not zero.

## C. Individual-stock fundamentals, multiples and DCF

Offline, reproducible example (all figures **synthetic**):

```powershell
python -m portfolio_risk fundamentals DEMO --file examples\v8_fundamentals_demo.json --peer-file examples\v8_fundamentals_peer_one.json --peer-file examples\v8_fundamentals_peer_two.json --as-of 2026-11-07 --json > v8_dcf_demo.json
```

Live best-effort research example (requires Yahoo availability; no API key):

```powershell
python -m portfolio_risk fundamentals AAPL --json --output v8_AAPL_fundamentals.json
```

Optional `--peer-ticker MSFT --peer-ticker GOOG` runs extra Yahoo financial
requests; comparison needs **2 valid peers per multiple**. Peer choice is not
validated as economically comparable. A statement file has explicit
`symbol`, `period_end`, `statement_currency`, optional matching
`quote_currency`, and numeric FCF/CFO/capex, earnings, revenue, EBITDA, cash,
debt, shares, market price fields. No currency mixing or invented missing
fundamentals. When annual FCF is positive and cash, debt and share counts are
available, display conservative/base/optimistic *assumption-driven* five-year
DCF sensitivities with terminal-value share. Do not present DCF as an
objective price target. Negative FCF or missing inputs produces
`dcf.status: unavailable`; financial firms and leveraged products may need a
different valuation model.

**Warning:** `yfinance` accesses Yahoo Finance's personal research data; it is
unofficial and fields/access can change. Yahoo changed financial-statement
vocabulary in 2026. V8 parses several row aliases but does not guarantee
availability or historically point-in-time financials. It does NOT silently
use today's live quote with an earlier user-requested `--as-of`: for a past
cutoff you must use an externally vetted statement file.

## D. Regression and local live verification

1. `python -m pytest` — no internet; verifies V1–V7 and all new V8 invariants,
   including invalid data, inconsistent currencies, synthetic CLI/provider
   integration, missing peers, missing FX, deterministic Monte Carlo,
   wrong-way alerts, train/holdout separation and provider metadata caching.
2. Run **A** with `--output v8_insights_report.json`, inspect the `benchmarks`,
   `diagnostics`, `historical_strategy_selection`, `user_scenarios`,
   `relative_to_benchmarks`, and `currency_comparison` fields.
3. Run **B** and inspect `drilldown.sector_drilldown`,
   `drilldown.unknown_holdings_by_fund`, and
   `drilldown.overlap_contributors`. Expect XEQT's unresolved constituent weight
   to remain visible; common-stock overlap uses dated snapshots and is a lower
   bound. New Yahoo metadata requests may be rate limited.
4. Run **C** offline; then optionally attempt live AAPL. Live fundamentals can
   be unavailable even when Yahoo adjusted-close history works.
5. Change one `--scenario` to intentionally sum to 90%: command must **reject**
   it. Change `--priorities` to `drawdown:3,sharpe:1` and verify the selected
   training allocation can change (no promise that it will on every sample).
6. Re-run V7 `etf-audit`, `exposure --strict-exposure`, `risk`, `construct`
   as needed to verify existing modes after installing over your local project.

Please share `v8_insights_report.json`, `v8_exposure_drilldown.json`,
`v8_dcf_demo.json`, and the pytest summary before we finalize V8's live
validation/overview. The internet-dependent tests cannot be certified from
offline mocks.

## Excluded / future work

- No investment recommendations, forecasts or guarantee of a best portfolio.
- No 24-hour monitoring, alerts via email/SMS, brokerage execution, tax
  optimization, dividend event prediction or public web data redistribution.
- No bond duration/credit analytics, derivatives notional exposure, historical
  ETF constituent backtests, or complete Canadian ETF holdings coverage.
- No equity-issuer fundamental projection calibration or cross-sector
  universally meaningful peer-comparison model.
- UI/interactive charts remain for V9. V8 JSON is the structured backend
  source for tooltips, warnings and future plots.
