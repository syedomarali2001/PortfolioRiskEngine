"""Pure, deterministic portfolio accounting from user-entered market values.

No market API calls, historical returns, ticker inference, dividend assumptions,
or price projections occur here. This layer can be reused by a future GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from portfolio_risk.fx import FXQuote, convert_amount
from portfolio_risk.models import PortfolioInput


@dataclass(frozen=True)
class ValuedHolding:
    symbol: str
    entered_value: Decimal
    entered_currency: str
    value_in_base: Decimal
    base_currency: str
    weight: Decimal  # 0..1; not a percent value


@dataclass(frozen=True)
class Concentration:
    largest_weight: Decimal
    top_three_weight: Decimal
    herfindahl_index: Decimal
    effective_holdings: Decimal


@dataclass(frozen=True)
class PortfolioValuation:
    source_file: str
    base_currency: str
    total_value: Decimal
    holdings: tuple[ValuedHolding, ...]
    concentration: Concentration
    # Grouped by the denomination of entered market values; not an exposure analysis.
    by_entered_currency: tuple[tuple[str, Decimal], ...]
    fx_quote: FXQuote | None


def value_portfolio(
    portfolio: PortfolioInput,
    *,
    base_currency: str = "CAD",
    fx_quote: FXQuote | None = None,
) -> PortfolioValuation:
    """Calculate a snapshot of portfolio values and capital concentration.

    All user-entered market values are assumed to describe one portfolio
    snapshot. FX must be supplied whenever an entered currency differs from
    the selected base currency. No historical price series are required.
    """
    base = base_currency.upper().strip()
    if base not in {"CAD", "USD"}:
        raise ValueError("base_currency must be CAD or USD")
    if not portfolio.holdings:
        raise ValueError("Portfolio must contain at least one holding")
    fx = fx_quote.usd_cad if fx_quote else None
    interim: list[tuple[str, Decimal, str, Decimal]] = []
    for item in portfolio.holdings:
        if not item.market_value.is_finite() or item.market_value <= 0:
            raise ValueError(f"{item.symbol}: market value must be positive and finite")
        converted = convert_amount(item.market_value, item.currency, base, fx)
        interim.append((item.symbol, item.market_value, item.currency, converted))

    total = sum((row[3] for row in interim), Decimal("0"))
    if not total.is_finite() or total <= 0:
        raise ValueError("Portfolio total must be positive and finite")

    holdings = tuple(
        ValuedHolding(symbol, entered, currency, converted, base, converted / total)
        for symbol, entered, currency, converted in interim
    )
    weights = sorted((row.weight for row in holdings), reverse=True)
    hhi = sum((weight * weight for weight in weights), Decimal("0"))
    concentration = Concentration(
        largest_weight=weights[0],
        top_three_weight=sum(weights[:3], Decimal("0")),
        herfindahl_index=hhi,
        effective_holdings=Decimal("1") / hhi,
    )
    breakdown: dict[str, Decimal] = {}
    for row in holdings:
        breakdown[row.entered_currency] = (
            breakdown.get(row.entered_currency, Decimal("0")) + row.value_in_base
        )
    return PortfolioValuation(
        source_file=portfolio.source_file,
        base_currency=base,
        total_value=total,
        holdings=holdings,
        concentration=concentration,
        by_entered_currency=tuple(sorted(breakdown.items())),
        fx_quote=fx_quote if any(h.currency != base for h in portfolio.holdings) else None,
    )


def valuation_as_dict(valuation: PortfolioValuation) -> dict:
    """JSON-ready model with financial values encoded as decimal strings."""
    fx = valuation.fx_quote
    return {
        "source_file": valuation.source_file,
        "base_currency": valuation.base_currency,
        "total_value": str(valuation.total_value),
        "holdings": [
            {
                "symbol": h.symbol,
                "entered_value": str(h.entered_value),
                "entered_currency": h.entered_currency,
                "value_in_base": str(h.value_in_base),
                "weight": str(h.weight),
            }
            for h in valuation.holdings
        ],
        "concentration": {
            "largest_weight": str(valuation.concentration.largest_weight),
            "top_three_weight": str(valuation.concentration.top_three_weight),
            "herfindahl_index": str(valuation.concentration.herfindahl_index),
            "effective_holdings": str(valuation.concentration.effective_holdings),
        },
        "by_entered_currency": {ccy: str(value) for ccy, value in valuation.by_entered_currency},
        "fx_quote": None if fx is None else {
            "pair": "USD/CAD",
            "rate_cad_per_usd": str(fx.usd_cad),
            "cutoff_date": fx.as_of.isoformat(),
            "observed_on": fx.observed_on.isoformat(),
            "age_calendar_days": fx.age_days,
            "source": fx.source,
            "cached": fx.cached,
        },
        "limitations": [
            "Entered values are supplied by the user; no security prices were used to value holdings.",
            "Entered-value currencies are not equivalent to underlying geographic or currency risk.",
            "Capital concentration is not volatility contribution or a forecast of future returns.",
        ],
    }
