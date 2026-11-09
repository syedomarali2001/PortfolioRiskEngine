# V7.2 Data Capability Audit Guide (not final V7 overview)

## Goals

Decide, from **observed** evidence, which features Portfolio Risk Engine supports for arbitrary holdings. Keep distinct: short-window historical prices; company identity and sector/industry/domicile; dated ETF constituents and weight coverage; specialty instruments; and historical depth. A successful API response alone is not sufficient.

## Installation

Extract `PortfolioRiskEngine_V7_2_Update.zip` directly over the active `PortfolioRiskEngine` folder (no extra nesting). No new dependencies. Preserve caches, local portfolio files and any original overview files. Do NOT create final `V7_OVERVIEW.txt` yet—finish local verification first.

## Offline test

```powershell
python -m pytest
```

Expected: **359 passed** with the packaged new tests and all prior V1–V7.1 tests.

## Live test 1 — 241 US ETFs

```powershell
python -m portfolio_risk etf-audit examples\us_etf_universe_v72.csv --output us_etf_v72_report.json
```

The fixture contains 241 distinct fund symbols in 14 thematic groups: broad US, small/midcap, international developed/emerging, sectors, factors/styles, dividends, bonds/Treasuries, commodities/metals, leveraged/inverse, thematic, crypto-linked, active/alternative and niche single-country ETFs. It is a **curated cross-sectional diagnostic sample**, not a statistically representative exchange-wide sample. For each fund the audit checks provider-index discovery, actual downloadable holdings, weights, percent of NAV identified, source, file date, freshness, holdings kind, sector/country breakdown and fee field. A provider-reported universe count is recorded separately; do not equate that claim with independent full coverage validation.

For a faster initial test:

```powershell
python -m portfolio_risk etf-audit examples\us_etf_universe_v72.csv --limit 20 --output us_etf_v72_smoke.json
```

The full run uses a modest 100 ms pause between fund requests. The 24-hour TickerInside cache avoids repeated requests during a retry. You may split requests using `--offset 0 --limit 80`, then `--offset 80 --limit 80`, then `--offset 160` with **different output filenames**. Such partial reports do not combine automatically into one total; run full audit if you want a single complete denominated sample report. Respect the API's access policies.

**ETF report fields:** `indexed`; `with_dated_holdings`; `near_full_weight_95pct`; `current_120d`; `equity_lookthrough_safe`; `with_sector_breakdown`; `with_country_breakdown`; `partial_weight`; `nonstandard_weights`; `provider_errors`; `fund_types`. Counts are calculated from the tested sample, by theme as well as overall. An indexed fund with no dated positions is different from an index miss or a transient HTTP error.

**Specialty limitations:** Bonds may have price history but only partial bond constituent data, which cannot be treated as equity overlap. Commodity trusts, futures exposure, leveraged/inverse and other synthetic structures require different semantics. Report the missing information honestly instead of normalizing partial holdings to 100%.

## Live test 2 — 110 individual equities (75 US-listed + 35 Canadian-listed)

```powershell
python -m portfolio_risk stock-audit examples\stock_universe_v72.csv --output stock_v72_report.json
```

The Yahoo metadata audit checks whether the security is recognized as `EQUITY`, its company name, sector, industry, reported company location, exchange and quote currency. It reports absent fields and provider errors separately by market and group. This uses Yahoo `get_info()`; the provider may rate limit. Use `--limit` and `--offset` for diagnostic subsets if necessary, and treat unavailable metadata as unverified, not as definitive unsupported ticker.

**Country caveat:** `company_country` is a company headquarters/location classification from Yahoo. It is not necessarily its listing exchange (e.g. U.S.-listed international ADRs), sales geography or currency exposure.

## Live test 3 — mixed 30-asset universe (prices + metadata)

```powershell
python -m portfolio_risk coverage-audit examples\coverage_universe_30.csv --check-prices --check-metadata --output mixed_v72_report.json
```

This repeats V7.1's five-each U.S. ETFs, Canadian ETFs, U.S. stocks, Canadian stocks, commodity ETPs, and crypto currencies. It adds issuer/type/company classification metadata, without mislabeling a coin or direct stock as lacking ETF constituents. The **price check uses a recent 45-calendar-day request**, not long-term history.

Known V7.1 results as of November 7, 2026: all 30 sample tickers returned recent prices, all five U.S. funds had some dated holdings, four of five commodity ETPs had physical holdings records, and no Canadian ETF constituents were available from TickerInside. **Do not copy these results forward as V7.2 findings**; rerun and inspect fresh files.

## Combine the three reports

```powershell
python -m portfolio_risk capability-report --etfs us_etf_v72_report.json --stocks stock_v72_report.json --mixed mixed_v72_report.json --output v72_capabilities.json
```

This **does not call any API**. It fuses separate price, holdings and metadata dimensions into a capability matrix and retains the sample caveats. The matrix will not claim that all U.S. ETFs work because 241 selected examples passed.

### Bonus: current direct-stock classification in a user portfolio

```powershell
python -m portfolio_risk exposure examples\sample_portfolio_v3.csv --base CAD --tickerinside --auto-stock-metadata
```

`--auto-stock-metadata` enriches **direct** holdings only, up to 40 by default. It does not fetch metadata for every underlying constituent of large ETFs because this may involve thousands of Yahoo requests. Fund-level sector/country totals returned by TickerInside remain separate from the portfolio's individually classified constituent sector/country totals; do not misinterpret the latter as full fund classification. `--metadata-limit 40` is adjustable.

## Acceptance criteria for V7 and V9 feature labels

1. Cite observed numerator/denominator for each audited feature and type; never use untested universal claims.
2. Treat ticker identity, quoted prices, historical depth, direct company metadata, equity constituents and derivative/bond exposures as **different questions**.
3. Report holdings weights and actual disclosure dates, with stale samples labelled; never conflate file retrieval with disclosure date.
4. Classify missing data as outside scope, not indexed, malformed/nonstandard, rate limited, or genuinely unavailable whenever possible.
5. Preserve V1–V7.1 commands and automated tests.
6. Review the **three raw JSONs plus the combined matrix** before finalizing the V7 overview and deciding which asset types to advertise on GitHub.

## Providers and licences

- TickerInside: https://tickerinside.com/api/ — free current snapshot API with attribution (CC BY 4.0 for the covered current data; separate history terms). Source dates and kind may be incomplete.
- yfinance/Yahoo: free, best-effort *personal/research* price and company-metadata adapter; access may be throttled; do not assume unrestricted commercial redistribution rights.
- No paid API added in V7.2.
