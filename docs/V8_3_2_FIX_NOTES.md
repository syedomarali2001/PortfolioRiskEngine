# V8.3.2 — constraint-compliance and sector-weight reporting clarification

This is a small corrective update to V8.3.1. No changes to optimization objective selection, historical return calculations, risk metrics, benchmark comparisons, market-data adapters, or ETF constituent holdings.

## Live-test findings addressed

1. **Noncompliant starting portfolio:** the original allocation stays as an audit baseline even when it violates user-declared hard limits. When no candidate selection process consistently improves every chosen risk objective, the report must NOT suggest retaining the original *as a compliant allocation*. It now separates performance findings from constraint compliance. `rolling_robustness.constraint_compliance_example` is a projected feasible target, **not** a risk-validated improvement, trading instruction, or globally minimal-cost solution.
2. **Turnover meaning:** `max_turnover` limits the **initial** one-way allocation change from the original weights to the proposed target. A subsequent historical holdout may incur additional turnover from quarterly (or other) rebalancing. The output now explicitly distinguishes these quantities.
3. **Unused history:** a full-length chronological holdout may not fit at the end of the requested history; report the number of unvalidated trailing daily returns and the last evaluated holdout date. Do not count partial windows as successful folds.
4. **Sector metadata lookup ranking:** `selected_top_equity_weights` now gives actual total-portfolio weight through the known ETF constituents; the new `selection_ranking_scores` field is the issuer-sector-weighted search-priority score. These must not be conflated; issuer fund-level sector totals do not prove the classification of any individual company.

## Installation and validation

Copy over the matching files in the root of the installed V8.3/V8.3.1 project. Retain caches and reports. Run `python -m pytest` (expected 462 passed), then rerun the two V8.3 live commands only if you want fresh report JSON with corrected text and metadata labels. There are **no production changes** to price-return arithmetic, annualized-risk metrics, or the discovered ETF constituent weights. Prior live JSONs remain valid for their numerical results, but their action wording and focused lookup weight-label detail are superseded by this fix.

## Scope caution

Neither the projected compliance example nor the best historical allocation is investment advice. A target-weight cap may be temporarily exceeded by realized drifting weights between scheduled rebalances, and a capped initial allocation change does not cap cumulative trading across a holdout.
