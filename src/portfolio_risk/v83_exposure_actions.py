"""V8.3 bounded, evidence-based data-coverage and overlap research prompts."""

from __future__ import annotations


def exposure_actions(document: dict) -> dict:
    direct = document["direct_weights"]
    findings = []
    for ticker, row in document["coverage_by_fund"].items():
        unidentified = row["portfolio_weight"] * row["unknown_fraction_of_fund"]
        if unidentified >= 0.05:
            findings.append(
                {
                    "type": "unresolved_holdings",
                    "sentiment": "negative",
                    "investment": ticker,
                    "portfolio_fraction_unresolved": unidentified,
                    "evidence": f"{ticker} contributes {unidentified:.2%} of the WHOLE portfolio as unknown underlying exposure.",
                    "target": "For a complete constituent-level analysis, resolve this fund's source holdings (data-quality objective, not a trading target).",
                    "research_next_step": "Request/retrieve a dated constituent disclosure; until then exclude this holding from stock-level diversification claims.",
                }
            )
    for pair in document["pairwise_overlap"]:
        a, b = pair["fund_a"], pair["fund_b"]
        overlap = pair["identified_overlap_lower_bound"]
        if overlap >= 0.35 and direct.get(a, 0) >= 0.10 and direct.get(b, 0) >= 0.10:
            findings.append(
                {
                    "type": "identified_fund_overlap",
                    "sentiment": "negative",
                    "funds": [a, b],
                    "identified_fund_level_overlap": overlap,
                    "fund_portfolio_weights": [direct[a], direct[b]],
                    "evidence": f"{a} and {b} share at least {overlap:.2%} identified fund-weighted common equities. Both exceed 10% of the portfolio.",
                    "target": "Test whether an alternative allocation reduces identified fund-level overlap below the 35% screening heuristic without worsening chosen risk metrics.",
                    "research_next_step": "Inspect shared stock contributors and compare alternatives in the What-If + rolling evaluation workflow; overlap alone is not a sell signal.",
                }
            )
    unidentified = document.get("unknown_underlying", 0.0)
    return {
        "findings": findings,
        "unresolved_portfolio_fraction": unidentified,
        "data_completeness": "incomplete"
        if unidentified > 0.05
        else "relatively_complete_for_supported_constituents",
        "limitations": [
            "Screening thresholds of 5% unresolved and 35% overlap are transparent heuristics, not market laws.",
            "Fund-level overlap is a lower bound with dated holdings; a missing fund may overlap materially.",
            "Incomplete data is a research limitation, not itself an investment recommendation.",
        ],
    }
