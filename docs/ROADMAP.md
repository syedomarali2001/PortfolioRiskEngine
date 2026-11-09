# Project Roadmap — Portfolio Risk Engine

**Goal:** Help investors understand portfolio risk and hidden exposure; explore
alternative allocations and strategies with transparent assumptions and
reproducible analysis. No guaranteed profitable strategy or stock forecast.

| Version | Target | Exit criteria / deliverable |
|---|---|---|
| V1 | Foundation and market-data ingestion | CSV/TXT parser, validation, Yahoo provider, currency quotes, cache, CLI, offline tests. **Built**. |
| V2 | Portfolio accounting and CAD/USD FX | FX-normalize entered market values, calculate capital weights/concentration; add no-look-ahead historical FX conversion. **Implemented.** |
| V3 | Performance analytics | Historical metrics, correlations, benchmark comparisons, strict market-date alignment. **Completed and verified locally.** |
| V4 | Backtesting | Buy-and-hold, monthly/quarterly/semiannual/annual rebalancing, proportional costs (zero default), benchmark and trade audit. **Completed and verified locally.** |
| V5 | Risk modelling | Risk attribution, VaR/Expected Shortfall, stress tests and rolling validation. **Completed and verified locally.** |
| V6 | Portfolio construction and strategies | Equal weight, inverse-volatility, minimum-variance and momentum top-K strategies; past-only signals, cost-aware walk-forward simulation and optional later-period analysis. **Completed and verified locally: 235 tests.** |
| V7 | ETF exposure intelligence | Current issuer-snapshot look-through, overlap lower bounds, sector/country unknown coverage, Yahoo metadata and factor-proxy betas. **V7.3 completed after 241-fund live rerun, 110-stock checks and 380 offline tests.** |
| V8 | Explainable analytics and fundamental valuation | **V8.2 completed; 431 tests and live AAPL + benchmark/exposure validation:** SPY/RSP matched-date diagnostics, bounded explanatory alerts, sector/overlap drilldown, what-if allocations, 70/30 held-out selection by chosen metrics, risk attribution, bootstrapped Monte Carlo, crisis/time-window analysis, CAD/USD comparison, and optional Yahoo/manual fundamentals, peer multiples, scenario DCF. |
| V8.3 | Final backend improvement and evidence | **Live constrained/sector tests completed; V8.3.2 reporting correction passed 462 development tests:** investment/group constraints, turnover/cadence/flat cost assumptions, expanding multi-fold holdouts, sector-targeted verification and actionable benchmark targets. |
| V8.4 | Small optional dividend income extension | **Backend implementation with 498 development tests; awaiting Windows/live verification:** historical gross distributions, known-data coverage, year-by-year cash-vs-DRIP, explicit price/distribution growth and user-supplied distribution-only tax drag. No capital-gains or personal tax engine. |
| V9 | Last major stage: integrated GUI | Streamlit + Plotly application calling V1–V8.4 tested services, ticker-search portfolio builder (CSV optional), compact Overview, Risk & Performance, Exposure X-Ray, Strategy Lab, Stock Valuation, optional Income & Dividends. A public demo requires data-licensing/privacy review. |

## Standing principles

1. Each version is modular, reproducible, testable *without* internet access.
2. Sources, quotation units, currencies, calendar dates, stale data, and missing
   information are explicit. Never invent missing market prices or exposures.
3. Don't infer actual past account performance from today's allocation.
4. No result promises investment returns; uncertainty and limitations are shown.
5. Clearly disclose which V7 asset classes have identified holdings versus only prices;
   do not claim unsupported bonds are entirely unavailable when bond prices work.
   Treat each individual statement source and the model's discount/growth assumptions as limitations; V8 fundamentals were successfully live-tested for AAPL only.
6. Numerical models live in backend modules; the GUI remains a thin adapter.

**V1 status:** Complete: 39 offline tests passed, 7/7 live symbols fetched, and cache reuse confirmed on the developer's computer.

**V7 closure:** V7.2's 241-fund, 110-stock and 30-asset live reports were followed
by V7.3's 241-fund rerun and a mixed real portfolio exposure report. The
final V7 overview is in `docs/V7_OVERVIEW.txt`. The completed V7.3 parser
continues to restrict bond, derivative and missing Canadian ETF look-through.

### V7.3 post-audit safety outcome (completed)
- ETF category/holdings evidence now governs validated stock look-through;
  cash, signed futures and non-equities remain distinct.
- Direct stock sector data and issuer-level fund sector/country data retain
  different provenance and are not assigned to individual securities.
- `supported`, `partial`, `price_only`, `unverified` per ticker; optional
  `--strict-exposure` rejects insufficient exposure data.
- 241-fund V7.3 live rerun evaluated; 380 inherited offline regression tests.
- No paid APIs, no bond duration/derivatives exposure/Canadian ETF constituent claims.

### V8 test policy
- Run complete offline pytest suite after every new module; verify copy-over ZIP against clean V7.3 installation.
- Compare SPY and RSP with the exact same currency, dates and return methodology.
- Diagnostics must be evidence-led and independent per benchmark, with positive/negative/neutral/unavailable states and explicit materiality thresholds.
- Allocate using training data only; show later holdout results without claiming future alpha.
- Partial unknown ETF exposures stay unresolved; optional capped company metadata cannot be treated as full constituent classification.
- Financial valuation uses verifiable same-currency dated figures or reports unavailable; live Yahoo data is not archival point-in-time data.
- V8.2 validation closed after 431 tests and real AAPL/benchmark results. See `docs/V8_TEST_GUIDE.md` for prior commands and `docs/V8_3_TEST_GUIDE.md` for the new V8.3 validation. No further GUI work before backend closure.
