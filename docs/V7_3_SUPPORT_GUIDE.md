# V7.3 — FINAL V7 EXPOSURE SUPPORT / RE-AUDIT GUIDE

This guide accompanies the code. **Do not write the final V7_OVERVIEW.txt until local live tests have been checked.** V7.2 results are the live baseline; all V7.3 tests are deterministic local/mocked tests until re-run using the external services.

## Workflow and commands

Extract V7.3's copy-over ZIP at the project root, preserving the prior V1–V6 docs and user-generated caches/reports. Then run:

```powershell
cd "$HOME\Desktop\PortfolioRiskEngine"
python -m pytest
python -m portfolio_risk etf-audit examples\us_etf_universe_v72.csv --output us_etf_v73_report.json
python -m portfolio_risk exposure examples\sample_portfolio_v3.csv --base CAD --tickerinside --auto-stock-metadata
python -m portfolio_risk exposure examples\sample_portfolio_v3.csv --base CAD --tickerinside --auto-stock-metadata --json
python -m portfolio_risk exposure examples\v73_supported_portfolio.csv --base USD --tickerinside --auto-stock-metadata --strict-exposure
python -m portfolio_risk exposure examples\v73_price_only_portfolio.csv --base CAD --tickerinside --auto-stock-metadata --strict-exposure
```

The full ETF test is 241 funds and has a built-in 100 ms pause per fund. The index and fund results are cached automatically by default for 24 hours. For a first small subset use `--limit 25 --output v73_smoke.json`; do not describe its summary as the full audit. For genuinely fresh provider data use `--refresh` (may take longer and trigger provider throttling). The `--json` exposure command prints output to stdout; to save it in PowerShell:

```powershell
python -m portfolio_risk exposure examples\sample_portfolio_v3.csv --base CAD --tickerinside --auto-stock-metadata --json > v73_exposure.json
```

Warning text may appear on stderr separately. The output includes `asset_support` and `restricted_exposure_assets` per ticker. The optional strict mode:

```powershell
python -m portfolio_risk exposure examples\sample_portfolio_v3.csv --base CAD --tickerinside --auto-stock-metadata --strict-exposure
```

**Expected behaviour:** refuse to generate a detailed exposure report if any fund's look-through is unavailable, too old, less than 95% identified (including unresolved nested funds), contains a non-equity manual constituent, or if any input lacks verified classification. A normal run does not refuse: it reports unknown portions, warnings and the reason per ticker. `--strict-exposure` is specifically for V7 detailed exposure, not price-based risk/performance calculations.

## What each status means

| `exposure_status` | Interpretation | Typical inputs |
| --- | --- | --- |
| `supported` | Known direct equity/crypto/physical commodity type, or qualified dated equity/spot-asset fund snapshot; provider categories used for stock sector and company location where available | AAPL, RY.TO, BTC-USD, qualified SPY |
| `partial` | Report may be useful but some holdings, stock metadata or as-of disclosure are incomplete or stale | Equity ETF with 90% identified or an ETF snapshot older than `--max-holdings-age-days` (default 120) |
| `price_only` | **No detailed exposure guarantee**; price-based risk analysis is independent and might still work when historical prices exist | Canadian ETFs (no free full holdings provider), bond ETFs, futures/derivatives/leveraged funds, oil-futures ETPs |
| `unverified` | A usable direct ticker was entered but identity cannot be established from supplied classifications or Yahoo metadata | Unknown ticker or blocked/rate-limited metadata request |

These are descriptions of **underlying exposure**; `price_risk_status: not_checked_here` means V7.3 hasn't attempted to retrieve risk-return history in that command. Use the `coverage-audit ... --check-prices` command for a price availability check or V3–V6 risk commands for actual analysis. No universal ticker coverage is claimed.

## Engineering safety and precision

- TickerInside is free and U.S.-ETF-focused. The V7.2 live 241-fund audit showed 239 indexed, 205 with dated holdings, 164 with at least 95% of NAV identified, 151 with sector breakdowns, and 152 with country breakdowns; **these numbers are baseline, not V7.3 results**. The strict equity eligibility of 63/241 was conservative because the provider labels some genuine issuer stock holdings as `unknown`.
- V7.3 accepts `unknown` holding type only when the provider provides independent contextual **equity category, holdings-based method, matching as-of date, positive stock count, at least 95% sector classification, and at most 0.5% reported other-fund weight**. Explicit bond/derivative/multi-asset holding types are never turned into equities. Issuer category evidence isn't an independent audit of every stock ticker.
- Cash is separately represented; recognized futures and swaps are not equity shares. Signed or nonstandard derivatives trigger a `price_only` decision instead of causing the provider record to disappear. Unrecognized negative stock weights, gross totals above 100%, bad dates, missing fund identifiers and invalid percentages are rejected safely.
- Direct stock metadata from Yahoo and ETF sector/country aggregates from TickerInside are combined in unified sector and country keys. This is **fund-level** sector/geography for an ETF; individual stock constituents do not thereby inherit invented industry labels. Country is classified company location/domicile, not currency or revenue mix.
- Stock-vs-ETF overlap uses *identified common equities* only, excluding UNKNOWN types, bonds, cash, physical metals, crypto, derivative positions and ETFs without verified constituents. The reported shared weight remains a lower bound. Different holdings disclosure dates and stale files must be presented to users.
- Direct gold/silver/platinum/palladium exchange-traded products and some spot crypto ETPs may support a commodity/coin exposure classification when the API confirms a simple underlying spot asset. Futures-based commodities are **not** supported for detailed exposure. A coin's company sector/country classification is not applicable. Historical prices can work independently.
- Canadian ETF constituent look-through is **not supported** automatically in this free version. Canadian direct stocks are supported for tested price risk and basic classifications; do not claim 100% Canadian stock market coverage. No duration, yield-curve, credit, derivative delta, gross/net futures leverage or physical commodity spot backtest is implemented.
- `TickerInside (CC BY 4.0)` attribution and source URL should accompany published ETF outputs. Yahoo data use follows its own terms. Historical ETF constituent snapshots are not used for as-of historical backtesting.

## Final decision after live verification

Review `us_etf_v73_report.json` for `provider_verified_equity`, `equity_lookthrough_safe`, `mixed_non_equity_positions` and the per-fund `equity_validation` fields. Compare with V7.2's 63/241 baseline; not all 241 funds are equity ETFs. Some newly eligible funds could still be stale or less than 95% complete. Review `v73_exposure.json` for per-ticker statuses, consolidated sector/country and known/unknown weights, then write the final V7_OVERVIEW.txt with actual observed results. Do **not** claim the parser changes guarantee 100% market coverage or accurate exposure to bonds/derivatives.
