# V8.4 — Optional Dividend & Distribution Income Patch

## Scope

Small backend-only module. Uses dated Yahoo **unadjusted Close** prices and `Dividends` cash event history independently from the adjusted-price return/backtesting engine. No changes to V1–V8.3 pricing, risk, portfolio strategies, ETF look-through or DCF calculations. No new dependencies or paid APIs.

Provides: trailing 365-day distribution-per-share, estimated annual **gross** income for the user's supplied current holding values, portfolio yield where all holdings have verified coverage, individual contributions and previous-year growth if there are two complete observation windows. Illustrative cash withdrawal versus DRIP projections (up to 40 years) use **user-supplied** annual price growth, per-share distribution growth and optional **distribution-only** effective tax/withholding drag.

**Not a personal tax calculator, trading system, dividend forecast, total-return backtest or proof of future yields.** ETF distributions may contain capital gains, interest or return of capital and may not be ordinary dividends; Yahoo cash events do not reliably classify distribution tax character. No Canadian account-specific deductions, credits or treaty calculations. DRIP may still incur taxes in taxable accounts. No modeled tax on capital gains, no investor tax filing advice.

## 1. Installation

Extract `PortfolioRiskEngine_V8_4_Update.zip` **over your existing V8.3.2 project root**. Replace matching files. Keep your existing data caches and original reports.

```powershell
python -m pytest
```

Expected in the supplied update: **498 passed** (462 inherited tests + 36 new tests). This number was verified in the development environment; it still needs your Windows check. If the project is not installed in editable mode, run it from the same environment you used for the previous 462 tests.

## 2. Reproducible offline example (NO network)

```powershell
python -m portfolio_risk dividends examples\v84_dividend_demo_portfolio.csv --base USD --as-of 2025-11-07 --snapshot-file examples\v84_dividend_demo.json --years 10 --price-growth 4 --dividend-growth 3 --tax-drag 15 --output v84_demo_income.json
```

The `SCHD`, `AAPL` and `GLD` prices/distributions in this particular file are **entirely synthetic**. They are test fixtures, not any actual ticker's historical payouts. The constructed starting value is $25,000 USD. Expected offline output: 100% data coverage, **2.40% trailing synthetic yield**, **$600 gross estimated annual income**, **$50 monthly arithmetic average** (not a monthly payout schedule), and an illustrative 10-year comparison. The future 4% price and 3% distribution growth rates are **assumptions**. The 15% withholding/tax drag is a user-defined number, not a tax law calculation.

Use `--json` to print the complete report, or inspect the JSON saved by `--output`.

## 3. Actual live Yahoo dividend test

```powershell
python -m portfolio_risk dividends examples\v84_dividend_live_portfolio.csv --base USD --years 10 --price-growth 4 --dividend-growth 3 --tax-drag 0 --output v84_live_dividends.json
```

The live portfolio uses **hypothetical, user-entered** values for SCHD, AAPL and SPY, not your personal account. The app fetches about three years of unadjusted Yahoo prices and cash distribution events. It also verifies quote currency. If a ticker's metadata, dividend history or price is missing or stale, that ticker is **unavailable**, not assumed to have a zero dividend.

To bypass saved dividend snapshots and recheck the provider:

```powershell
python -m portfolio_risk dividends examples\v84_dividend_live_portfolio.csv --base USD --refresh --output v84_live_dividends_refreshed.json
```

Provider availability, field names and cash events are best effort. **Do not expect a particular live yield or payment amount**; these change. Send `v84_live_dividends.json` for one final real-provider review before closing V8.4.

## 4. Optional CAD/USD conversion

- Portfolio `currency` is the denomination of the **entered value**, not necessarily the exchange quote currency.
- `quote_currency` (optional column) names the actual ticker quote currency and is cross-checked with the provider if supplied.
- If distributions are USD but user-entered values are CAD, the engine independently obtains USD/CAD for dividend calculations—even if all holding values are already CAD.
- `--usd-cad 1.40` can override the live FX request. The report indicates the FX source/date. Do not mistake a user-supplied rate for a historical point-in-time FX observation.
- Only CAD and USD quote currencies are currently supported.

## 5. How to read the report

- `holdings[].trailing_distribution_yield`: observed preceding 365-day distributions per share / latest non-adjusted price. This isn't promised forward yield.
- `holdings[].estimated_annual_gross_income`: inferred fractional shares from user-supplied investment amount and raw share price multiplied by trailing per-share distributions.
- `holdings[].historical_trailing_year_distribution_growth`: comparison against the previous nonoverlapping 365 days, only when both windows have full verified coverage and prior-year payouts were positive; otherwise `null`. Not an input to the projection.
- `income_summary.income_coverage_fraction`: what proportion of **market value** was evaluated. If less than 100%, a complete portfolio dividend yield is withheld; the report separately exposes the covered-value yield and known income contribution.
- `simulation.annual_results`: dollar-level yearly comparison of withdrawing **net** cash versus reinvesting **net** cash through fractional shares. For a fair total-wealth comparison, cash wealth includes **previously withdrawn cash** (assumed to earn zero interest).
- `simulation.modeled_start_value`: only verified holdings enter the projection. `unmodeled_start_value` is omitted entirely from *both* methods, not assumed to have a guaranteed zero yield.
- `simulation.assumed_effective_distribution_tax_drag`: an optional flat user-supplied assumption applied to cash distributions, not capital gains.

The simulator independently specifies **share-price appreciation** and **per-share payout growth**. Portfolio dividend yield is not assumed constant; it changes when those growth rates differ. Each year's payouts are modeled at year-end using the opening share count, and DRIP shares earn distributions only in later years. No fractional brokerage constraints or fees are included.

## 6. Final V9 placement

- Portfolio Overview: compact optional gross income summary (trailing yield, estimated annual and monthly arithmetic average, plus coverage indicator).
- Income & Dividends page: verified holding-level income, historical payout changes, cash-vs-DRIP scenarios, small optional `tax drag` toggle/percentage.
- No tax page and **no impact** on other portfolio-risk, benchmark, DCF or optimization computations.
- Users will primarily enter securities via ticker autocomplete and editable value/weight fields; CSV import is secondary.
