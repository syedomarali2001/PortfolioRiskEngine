"""V8.4: auditable distribution income and illustrative cash vs DRIP.

Isolated from adjusted-price return/backtesting calculations. No tax advice and no
price forecasts: the growth and distribution-tax assumptions are user inputs.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
import math

from portfolio_risk.accounting import PortfolioValuation
from portfolio_risk.fx import convert_amount


class DividendError(ValueError):
    """Input or model contract error, rather than a provider outage."""


def _dt(value, label: str) -> date:
    try:
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise DividendError(f"Invalid {label}: YYYY-MM-DD required") from exc


def _nonnegative(raw, label: str, *, strictly_positive: bool = False) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DividendError(f"{label} must be numeric") from exc
    if not math.isfinite(v) or v < 0 or (strictly_positive and v == 0):
        raise DividendError(f"{label} must be finite and {'positive' if strictly_positive else 'nonnegative'}")
    return v


def evaluate_dividends(
    valuation: PortfolioValuation,
    snapshots: dict[str, dict],
    *,
    as_of: date,
    years: int = 10,
    price_growth: float = 0.04,
    dividend_growth: float = 0.03,
    tax_drag: float = 0.0,
    fx_rate: Decimal | None = None,
) -> dict:
    """Evaluate dated cash payments relative to *unadjusted* spot price.

    Portfolio values are user-supplied snapshots, not fetched live. Only holdings
    with sufficient dated price + dividend history enter the income simulation.
    Unknown dividend status never quietly becomes a zero yield or a full forecast.
    """
    if isinstance(years, bool) or not isinstance(years, int) or not 1 <= years <= 40:
        raise DividendError("Projection years must be an integer from 1 to 40")
    for name, value, lower, upper in (
        ("price_growth", price_growth, -0.95, 1.0),
        ("dividend_growth", dividend_growth, -1.0, 1.0),
        ("tax_drag", tax_drag, 0.0, 1.0),
    ):
        try:
            valid = math.isfinite(float(value)) and lower <= float(value) <= upper
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise DividendError(f"{name} must be finite and between {lower} and {upper}")
    as_of = _dt(as_of, "analysis date")
    if not isinstance(snapshots, dict):
        raise DividendError("Dividend snapshots must be keyed by ticker")

    trailing_start = as_of - timedelta(days=365)
    prior_start = trailing_start - timedelta(days=365)
    fx = fx_rate if fx_rate is not None else (valuation.fx_quote.usd_cad if valuation.fx_quote is not None else None)
    findings = []
    eligible = []
    warnings = []

    for holding in valuation.holdings:
        symbol = holding.symbol
        row = {
            "symbol": symbol, "portfolio_weight": float(holding.weight),
            "market_value_in_base": float(holding.value_in_base),
            "status": "unavailable", "reason": None,
        }
        snap = snapshots.get(symbol)
        try:
            if not isinstance(snap, dict):
                raise DividendError("No verified dividend-price snapshot supplied")
            if str(snap.get("symbol", "")).upper() != symbol:
                raise DividendError("Snapshot ticker does not match portfolio holding")
            currency = str(snap.get("quote_currency") or "").strip().upper()
            if currency not in {"USD", "CAD"}:
                raise DividendError("Quote currency unavailable; cannot infer dividend conversion")
            raw_price = _nonnegative(snap.get("close_price"), "Close price", strictly_positive=True)
            quote_day = _dt(snap.get("price_date"), "price_date")
            first = _dt(snap.get("history_first_date"), "history_first_date")
            last = _dt(snap.get("history_last_date"), "history_last_date")
            if not first <= last <= as_of or not first <= quote_day <= last:
                raise DividendError("Price/history dates are inconsistent or after analysis cutoff")
            if (as_of - quote_day).days > 10:
                raise DividendError("Unadjusted price is stale (>10 calendar days)")
            if first > trailing_start + timedelta(days=7):
                raise DividendError("Less than one complete trailing year of observed dividend history")
            if last < trailing_start or (as_of - last).days > 10:
                raise DividendError("Distribution history is stale or incomplete")
            payments = snap.get("dividends")
            if not isinstance(payments, list):
                raise DividendError("Missing verified cash-distribution event history")
            paid = {}
            for payment in payments:
                if not isinstance(payment, dict):
                    raise DividendError("Dividend payments must be dated objects")
                pd = _dt(payment.get("date"), "payment date")
                amount = _nonnegative(payment.get("amount"), "Dividend amount", strictly_positive=True)
                if pd < first or pd > last:
                    raise DividendError("Dividend event falls outside reported history coverage")
                if pd in paid:
                    raise DividendError("Duplicate cash-distribution event date")
                paid[pd] = amount
            trailing = sum(v for day, v in paid.items() if trailing_start < day <= as_of)
            prior = sum(v for day, v in paid.items() if prior_start < day <= trailing_start)
            price_base = float(convert_amount(Decimal(str(raw_price)), currency, valuation.base_currency, fx))
            dps_base = float(convert_amount(Decimal(str(trailing)), currency, valuation.base_currency, fx))
            if not math.isfinite(price_base) or price_base <= 0:
                raise DividendError("Converted price invalid")
            shares = float(holding.value_in_base) / price_base
            yield_ttm = trailing / raw_price
            annual_income = shares * dps_base
            growth_history = ((trailing / prior) - 1.0) if (
                first <= prior_start + timedelta(days=7) and prior > 0
            ) else None
            if not math.isfinite(yield_ttm) or not math.isfinite(annual_income):
                raise DividendError("Income calculation overflow")
            if growth_history is not None and not math.isfinite(growth_history):
                growth_history = None
            row.update({
                "status": "evaluated", "reason": None, "quote_currency": currency,
                "raw_close_price": raw_price, "price_date": quote_day.isoformat(),
                "dividends_per_share_trailing_year": trailing,
                "dividends_per_share_previous_year": prior if first <= prior_start + timedelta(days=7) else None,
                "trailing_distribution_yield": yield_ttm,
                "estimated_annual_gross_income": annual_income,
                "estimated_monthly_average_gross_income": annual_income / 12.0,
                "historical_trailing_year_distribution_growth": growth_history,
                "observed_trailing_distribution_events": sum(trailing_start < d <= as_of for d in paid),
                "history_first_date": first.isoformat(), "history_last_date": last.isoformat(),
                "source": str(snap.get("source") or "User-supplied verified snapshot"),
                "limitations": ["Last-365-day distribution yield is historical, not a forward yield.",
                                "Market value is user entered; fractional shares are inferred from price."],
            })
            eligible.append({"symbol": symbol, "value": float(holding.value_in_base),
                             "shares": shares, "base_price": price_base, "base_dps": dps_base})
        except (DividendError, ValueError, TypeError, OverflowError) as exc:
            row["reason"] = str(exc)
            warnings.append(f"{symbol}: dividend estimate unavailable — {exc}")
        findings.append(row)

    total_portfolio = float(valuation.total_value)
    eligible_value = sum(x["value"] for x in eligible)
    gross_income = sum(h["estimated_annual_gross_income"] for h in findings if h["status"] == "evaluated")
    coverage = eligible_value / total_portfolio
    summary = {
        "status": "evaluated" if eligible_value > 0 else "unavailable",
        "portfolio_value": total_portfolio,
        "income_covered_value": eligible_value,
        "income_excluded_value": total_portfolio - eligible_value,
        "income_coverage_fraction": coverage,
        "estimated_annual_gross_income_covered_holdings": gross_income if eligible else None,
        "estimated_monthly_average_gross_income_covered_holdings": gross_income / 12 if eligible else None,
        "trailing_distribution_yield_on_covered_value": gross_income / eligible_value if eligible else None,
        "known_income_divided_by_total_portfolio_value": gross_income / total_portfolio if eligible else None,
        "fully_covered_portfolio_yield": (gross_income / total_portfolio if eligible and abs(coverage-1)<1e-9 else None),
        "note": ("Coverage excludes unavailable holdings; known income / total portfolio is a lower-bound "
                 "contribution, not a complete portfolio yield or a guaranteed income floor."),
    }

    if not eligible:
        sim = {"status": "unavailable", "reason": "No holdings with sufficient dated distribution histories"}
    else:
        # Dividends from prior trailing year act as year-1 payout assumption.
        # Payments are modeled at each year-end after the yearly price change;
        # additional DRIP shares start earning dividends in the following year.
        state = {x["symbol"]: dict(x) for x in eligible}
        cash_withdrawn = 0.0
        gross_cash_cumulative = 0.0
        reinvested_total = 0.0
        years_output = []
        for n in range(1, years+1):
            gross_cash = 0.0
            net_cash = 0.0
            gross_drip = 0.0
            net_drip = 0.0
            market_cash = 0.0
            market_drip = 0.0
            next_gross_cash = 0.0
            next_gross_drip = 0.0
            for ticker, item in state.items():
                py = item["base_price"] * (1.0 + price_growth)**n
                dps = item["base_dps"] * (1.0 + dividend_growth)**(n-1)
                cash_g = item["shares"] * dps
                drip_g = item.get("drip_shares", item["shares"]) * dps
                cash_net = cash_g * (1.0 - tax_drag)
                drip_net = drip_g * (1.0 - tax_drag)
                shares_drip = item.get("drip_shares", item["shares"]) + drip_net / py
                item["drip_shares"] = shares_drip
                gross_cash += cash_g
                net_cash += cash_net
                gross_drip += drip_g
                net_drip += drip_net
                market_cash += item["shares"] * py
                market_drip += shares_drip * py
                next_gross_cash += item["shares"] * dps * (1+dividend_growth)
                next_gross_drip += shares_drip * dps * (1+dividend_growth)
            cash_withdrawn += net_cash
            gross_cash_cumulative += gross_cash
            reinvested_total += net_drip
            vals = (market_cash, market_drip, net_cash, net_drip, cash_withdrawn, gross_cash, gross_drip)
            if not all(math.isfinite(v) and v >= 0 for v in vals):
                raise DividendError("Projection overflows; lower horizon or growth assumptions")
            years_output.append({
                "year": n, "cash": {
                    "yearly_gross_distributions": gross_cash,
                    "yearly_net_cash_withdrawn": net_cash,
                    "cumulative_net_cash_withdrawn": cash_withdrawn,
                    "invested_market_value": market_cash,
                    "total_wealth_including_withdrawn_cash": market_cash + cash_withdrawn,
                    "next_year_estimated_gross_distribution_income": next_gross_cash,
                },
                "drip": {
                    "yearly_gross_distributions": gross_drip,
                    "yearly_net_reinvested_distributions": net_drip,
                    "cumulative_net_distributions_reinvested": reinvested_total,
                    "invested_market_value": market_drip,
                    "total_wealth": market_drip,
                    "next_year_estimated_gross_distribution_income": next_gross_drip,
                },
            })
        sim = {
            "status": "illustrative", "modeled_start_value": eligible_value,
            "unmodeled_start_value": total_portfolio - eligible_value,
            "projection_years": years,
            "assumed_annual_price_growth": price_growth,
            "assumed_annual_distribution_per_share_growth": dividend_growth,
            "assumed_effective_distribution_tax_drag": tax_drag,
            "annual_results": years_output,
            "end_comparison": years_output[-1],
            "limitations": [
                "Hypothetical deterministic scenarios, NOT forecasts, recommendations or tax calculations.",
                "Only covered holdings are modeled; no invented income for unavailable tickers.",
                "Dividend growth and price growth are independent user assumptions, NOT inferred forecasts.",
                "Future payments occur at year-end and fractional DRIP purchases use the modeled year-end price.",
                "Cash withdrawals accumulate without interest; total wealth includes invested holdings plus withdrawn net cash.",
                "Tax drag applies only to modeled distributions; not capital gains, currency changes, tax credits or estate tax.",
                "No additional deposits, share splits, fees, cash interest, transaction costs or withholding recoveries.",
                "Historical return calculations elsewhere already embed distributions through adjusted prices; do not add this simulation to historical CAGR.",
            ],
        }

    return {
        "version": "8.4", "analysis_type": "cash_distribution_income_and_simulation",
        "as_of": as_of.isoformat(), "base_currency": valuation.base_currency,
        "method": "Trailing 365 calendar day cash distributions / latest unadjusted close; hypothetical year-end cash/DRIP",
        "holdings": findings, "income_summary": summary, "simulation": sim,
        "warnings": warnings,
        "data_policy": [
            "Dividend events may include ETF distributions with mixed income/tax character; not pure taxable dividends.",
            "Provider dividend data and raw prices may be corrected or unavailable; quote currency must be verified.",
            "Gross yield is TTM realized payments divided by unadjusted price, not a forecast yield.",
            "Historical growth compares trailing and preceding 365-day distribution sums where full coverage exists.",
            "Market values and FX are a user supplied/current snapshot; this is not a historical account ledger.",
            "Tax setting is user-supplied simplification only; TFSA/RRSP/taxable treatment is not inferred.",
        ],
    }
