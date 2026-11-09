"""Educational example portfolios for demos and regression testing.

Examples are deliberately constructed scenarios, not investment suggestions.
No live market price, FX rate, return, or recommendation is embedded here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

from .core import MAX_HOLDINGS, PortfolioError, engine_rows, to_csv, validate_holdings


@dataclass(frozen=True)
class ExamplePortfolio:
    key: str
    label: str
    purpose: str
    inspect: str
    # symbol, friendly name, illustrative percentage, amount currency, trading currency
    positions: tuple[tuple[str, str, float, str, str], ...]
    total: float = 100_000.0
    note: str = ""
    objective_to_test: str = "Compare Growth, Balanced and Defensive against your original portfolio and both benchmarks. Stars are earned only from later-period evidence."


def _p(symbol: str, name: str, pct: float, amount: str = "USD", trading: str = "USD"):
    return (symbol, name, pct, amount, trading)


_FORTY = (
    "AAPL MSFT NVDA AMZN GOOGL META JPM BAC WFC V MA JNJ UNH ABBV LLY PG KO PEP "
    "COST WMT HD LOW XOM CVX CAT DE GE RTX LMT NEE DUK PLD AMT SPY QQQ DIA IWM SCHD GLD TLT"
).split()

EXAMPLES: tuple[ExamplePortfolio, ...] = (
    ExamplePortfolio(
        "starter", "Starter: four-investment baseline",
        "A familiar four-holding benchmark for understanding how mathematical targets differ.",
        "Expect modest trade-offs rather than guaranteed improvements. Compare all three objectives and check whether any historical star is deserved.",
        (_p("AAPL", "Apple", 35), _p("MSFT", "Microsoft", 20),
         _p("SPY", "SPDR S&P 500 ETF", 30), _p("GLD", "SPDR Gold Shares", 15)),
        total=25_000.0,
    ),
    ExamplePortfolio(
        "ccl80", "Stress test: 80% Carnival (CCL)",
        "Concentrated cyclical stock: a strong test of whether Defensive lowers risk, Balanced improves Sharpe, and Growth finds a more productive historical allocation.",
        "Risk should usually fall when the concentration is reduced; a growth star is NOT guaranteed. Compare the later-period wins against SPY and RSP.",
        (_p("CCL", "Carnival Corporation", 80), _p("AAPL", "Apple", 7),
         _p("SPY", "SPDR S&P 500 ETF", 6), _p("MSFT", "Microsoft", 4),
         _p("GLD", "SPDR Gold Shares", 3)),
    ),
    ExamplePortfolio(
        "nvda90", "Stress test: 90% NVIDIA (NVDA)",
        "Concentrated high-growth stock: challenges Growth (which may favor staying concentrated) while testing whether Balanced and Defensive can reduce risk.",
        "Check whether the optimizer responsibly acknowledges growth versus concentration risk and whether Defensive earns later-period support.",
        (_p("NVDA", "NVIDIA", 90), _p("AAPL", "Apple", 4),
         _p("SPY", "SPDR S&P 500 ETF", 3), _p("MSFT", "Microsoft", 2),
         _p("GLD", "SPDR Gold Shares", 1)),
    ),
    ExamplePortfolio(
        "lagging_consumer", "Optimization test: struggling consumer stocks",
        "A realistic consumer-stock-heavy mix that gives Growth and Balanced a choice between individual businesses, broad equities, momentum and gold.",
        "Try all three goals. Look for meaningful growth/risk improvement against the starting mix; verify whether any improvement holds up in unseen periods.",
        (_p("SBUX", "Starbucks", 55), _p("NKE", "Nike", 25),
         _p("SPY", "SPDR S&P 500 ETF", 10),
         _p("SPMO", "Invesco S&P 500 Momentum ETF", 5),
         _p("GLD", "SPDR Gold Shares", 5)),
        note="SPMO started later than the individual stocks. Shared historical coverage begins only when every selected investment has usable prices.",
    ),
    ExamplePortfolio(
        "momentum_compare", "Optimization test: small-cap tilt versus momentum",
        "A small-cap-heavy mix compared with diversified equities, technology, momentum and gold.",
        "See whether the growth optimizer chooses a focused ETF, whether Balanced differs, and whether either approach beats SPY/RSP in later tests.",
        (_p("IWM", "iShares Russell 2000 ETF", 60),
         _p("SPY", "SPDR S&P 500 ETF", 15),
         _p("QQQ", "Invesco QQQ Trust", 10),
         _p("SPMO", "Invesco S&P 500 Momentum ETF", 10),
         _p("GLD", "SPDR Gold Shares", 5)),
        note="A Growth winner may be concentrated in a single ETF; that is historical optimization, not an endorsement of 100% allocation.",
    ),
    ExamplePortfolio(
        "bond_shock", "Optimization test: long-duration bond concentration",
        "A bond-heavy portfolio built to reveal interest-rate sensitivity and test whether stocks or gold can offset large bond drawdowns.",
        "Pay special attention to Defensive: did volatility improve, did drawdown follow, and did the result actually beat SPY/RSP on the same dates?",
        (_p("TLT", "iShares 20+ Year Treasury Bond ETF", 65),
         _p("IEF", "iShares 7-10 Year Treasury Bond ETF", 20),
         _p("SPY", "SPDR S&P 500 ETF", 10),
         _p("GLD", "SPDR Gold Shares", 5)),
    ),
    ExamplePortfolio(
        "index_overlap", "Feature demo: overlapping index ETFs",
        "Three ETFs tracking nearly the same index, with a direct Apple position.",
        "In Holdings & Exposure, check the ETF overlap and underlying Apple exposure; compare the holdings with the number of distinct companies actually owned.",
        (_p("SPY", "SPDR S&P 500 ETF", 40), _p("VOO", "Vanguard S&P 500 ETF", 30),
         _p("IVV", "iShares Core S&P 500 ETF", 20), _p("AAPL", "Apple", 10)),
        note="ETF holdings snapshots depend on provider availability and may have different reporting dates.",
    ),
    ExamplePortfolio(
        "dividends", "Feature demo: dividend income",
        "A dividend-oriented portfolio for showcasing income estimates and DRIP scenarios.",
        "Review which holdings generate income, yearly payments, and cash versus reinvestment projections; check provider coverage.",
        (_p("SCHD", "Schwab U.S. Dividend Equity ETF", 50),
         _p("VYM", "Vanguard High Dividend Yield ETF", 20),
         _p("DGRO", "iShares Core Dividend Growth ETF", 15),
         _p("SPY", "SPDR S&P 500 ETF", 10), _p("MSFT", "Microsoft", 5)),
        note="This is an income feature demo, not a guarantee of yield or future distributions.",
    ),
    ExamplePortfolio(
        "mixed_fx", "Feature demo: mixed CAD and USD",
        "Combines Canadian-listed and U.S.-listed investments, including a U.S. ETF valued in CAD.",
        "Check reporting-currency conversion, separate amount/trading currencies, and how historical price dates are aligned.",
        (_p("XEQT.TO", "iShares Core Equity ETF Portfolio", 30, "CAD", "CAD"),
         _p("SHOP.TO", "Shopify, Toronto listing", 10, "CAD", "CAD"),
         _p("SPY", "SPDR S&P 500 ETF", 30, "CAD", "USD"),
         _p("SCHD", "Schwab U.S. Dividend Equity ETF", 20, "USD", "USD"),
         _p("GLD", "SPDR Gold Shares", 10, "USD", "USD")),
        note="Displayed percentages are illustrative entry proportions, NOT FX-adjusted portfolio weights. CAD values are converted using available dated FX data.",
    ),
    ExamplePortfolio(
        "diversified", "Comparison: diversified multi-asset",
        "Diversified stocks, income, bonds, and gold, intended as a contrast to concentrated examples.",
        "See whether the optimizer can justify changes to an already-diversified portfolio instead of always claiming improvement.",
        (_p("SPY", "SPDR S&P 500 ETF", 30),
         _p("QQQ", "Invesco QQQ Trust", 15),
         _p("SCHD", "Schwab U.S. Dividend Equity ETF", 20),
         _p("IEF", "iShares 7-10 Year Treasury Bond ETF", 20),
         _p("GLD", "SPDR Gold Shares", 15)),
    ),
    ExamplePortfolio(
        "tech_only", "Stress test: highly correlated technology",
        "Four investments that may share underlying company and market exposure.",
        "Check technology sector concentration, QQQ/direct-stock overlap, and whether different tickers actually diversify risk.",
        (_p("AAPL", "Apple", 25), _p("MSFT", "Microsoft", 25),
         _p("NVDA", "NVIDIA", 25), _p("QQQ", "Invesco QQQ Trust", 25)),
    ),
    ExamplePortfolio(
        "short_history", "Edge case: recent ETF shortens history",
        "Includes a spot-Bitcoin ETF that began trading in 2024.",
        "Check the clearly displayed history start date, unavailable older crash replays, and insufficient-data explanations.",
        (_p("IBIT", "iShares Bitcoin Trust ETF", 35),
         _p("SPY", "SPDR S&P 500 ETF", 35),
         _p("SCHD", "Schwab U.S. Dividend Equity ETF", 20),
         _p("GLD", "SPDR Gold Shares", 10)),
        note="Some sources may not support IBIT; a clear unavailable-data result is acceptable for this edge case.",
    ),
    ExamplePortfolio(
        "one_holding", "Edge case: one investment (SPY)",
        "A one-security portfolio for checking single-asset reporting and optimizer limits.",
        "The optimizer has no alternative allocation with one investment; the app should explain this instead of inventing an improvement.",
        (_p("SPY", "SPDR S&P 500 ETF", 100),),
    ),
    ExamplePortfolio(
        "max_holdings", "Scale test: 40 holdings (may be slow)",
        "Fills the default 40-investment limit to test large portfolio editing, data coverage, and runtime.",
        "Check that the 40th entry works, a 41st is blocked, and a missing price is disclosed rather than silently fabricated.",
        tuple(_p(sym, sym, 2.5) for sym in _FORTY),
        note="This is an intentionally expensive data-coverage and runtime test. Start with Overview; optimizer performance may depend on provider limits and ticker history.",
    ),
)

_BY_KEY = {item.key: item for item in EXAMPLES}
assert len(_BY_KEY) == len(EXAMPLES)


def get_example(key: str) -> ExamplePortfolio:
    if key not in _BY_KEY:
        raise PortfolioError(f"Unknown sample portfolio: {key}")
    return _BY_KEY[key]


def example_holdings(key: str, *, max_holdings: int = MAX_HOLDINGS) -> list[dict]:
    example = get_example(key)
    if len(example.positions) > max_holdings:
        raise PortfolioError(f"This demo has {len(example.positions)} investments, but your configured limit is {max_holdings}.")
    result = []
    for symbol, name, pct, amount, trading in example.positions:
        result.append({
            "symbol": symbol, "name": name, "market_value": round(example.total * pct / 100.0, 2),
            "weight": pct, "currency": amount, "quote_currency": trading,
        })
    if abs(sum(row["weight"] for row in result) - 100) > 1e-8:
        raise PortfolioError("Sample portfolio allocations must add to 100%")
    validate_holdings(result, "values", max_holdings=max_holdings)
    return result


def example_csv(key: str) -> str:
    return to_csv(example_holdings(key, max_holdings=200))


def load_example(state: MutableMapping, key: str, *, max_holdings: int = MAX_HOLDINGS) -> None:
    """Replace builder session portfolio and invalidate all previously computed reports."""
    rows = example_holdings(key, max_holdings=max_holdings)  # validate before mutation
    state["holdings"] = rows
    state["entry_mode"] = "values"
    state["hypothetical_total"] = float(get_example(key).total)
    state["editor_version"] = int(state.get("editor_version", 0)) + 1
    state["ticker_version"] = int(state.get("ticker_version", 0)) + 1
    state["selected_ticker"] = None
    state["last_search_query"] = ""
    state["pending_quote_symbol"] = None
    state["quote_sources"] = {}
    state["reports"] = {}
    state["strategy_cache"] = {}
