"""CAD/USD exchange-rate conventions and auditable as-of FX retrieval.

USD/CAD is quoted as Canadian dollars per one US dollar. Converted values
use a single observed rate on or before the requested FX cutoff date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from portfolio_risk.cache import HistoryCache
from portfolio_risk.providers.base import MarketDataProvider
from portfolio_risk.service import get_validated_history

SUPPORTED_CURRENCIES = frozenset({"CAD", "USD"})


class FXRateError(ValueError):
    """FX data is missing, invalid or too old for the requested cutoff."""


def _currency(raw: str) -> str:
    result = raw.upper().strip()
    if result not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Currency must be CAD or USD, got {raw!r}")
    return result


def fx_ticker(from_currency: str, to_currency: str) -> str | None:
    """Yahoo-style pair symbol, or None for same-currency values (V1 contract)."""
    src, dst = _currency(from_currency), _currency(to_currency)
    return None if src == dst else f"{src}{dst}=X"


def validate_usd_cad(value: Decimal | str) -> Decimal:
    """Require a finite, positive USD/CAD quote without floating-point rounding."""
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise FXRateError("USD/CAD rate must be a valid decimal number") from exc
    if not rate.is_finite() or rate <= 0:
        raise FXRateError("USD/CAD rate must be positive and finite")
    return rate


@dataclass(frozen=True)
class FXQuote:
    """An auditable quote for CAD/USD conversion.

    ``as_of`` is the requested date; ``observed_on`` is the latest available
    actual observation on or before that date, and may be earlier on weekends.
    """

    usd_cad: Decimal
    as_of: date
    observed_on: date
    source: str
    cached: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "usd_cad", validate_usd_cad(self.usd_cad))
        if self.observed_on > self.as_of:
            raise FXRateError("FX observation cannot be later than the requested cutoff")

    @property
    def age_days(self) -> int:
        return (self.as_of - self.observed_on).days


def manual_fx_quote(rate: Decimal | str, as_of: date) -> FXQuote:
    """Explicit user-supplied rate; no network request is needed."""
    return FXQuote(
        usd_cad=validate_usd_cad(rate),
        as_of=as_of,
        observed_on=as_of,
        source="User-supplied USD/CAD rate (not independently verified)",
    )


def latest_usd_cad_quote(
    provider: MarketDataProvider,
    cache: HistoryCache,
    as_of: date,
    *,
    max_age_days: int = 5,
    refresh: bool = False,
) -> FXQuote:
    """Load a historical USD/CAD closing quote and enforce its age.

    For example a Saturday cutoff can use Friday's quote, but not one from
    the following Monday. The start of the request precedes the cutoff so
    weekend and bank holidays don't cause needless failures.
    """
    if not isinstance(as_of, date):
        raise FXRateError("as_of must be a date")
    if max_age_days < 0:
        raise FXRateError("max_age_days must not be negative")
    start = as_of - timedelta(days=max(14, max_age_days + 9))
    end = as_of + timedelta(days=1)  # end-exclusive
    result = get_validated_history(provider, cache, "USDCAD=X", start, end, refresh=refresh)
    series = result.history.prices
    eligible = series[series.index.date <= as_of]
    if eligible.empty:
        raise FXRateError(f"No USD/CAD rate available on or before {as_of}")
    date_of_quote = eligible.index[-1].date()
    age = (as_of - date_of_quote).days
    if age > max_age_days:
        raise FXRateError(
            f"USD/CAD quote is {age} calendar days old ({date_of_quote}); "
            f"maximum allowed is {max_age_days}"
        )
    rate = validate_usd_cad(str(float(eligible.iloc[-1])))
    return FXQuote(
        usd_cad=rate,
        as_of=as_of,
        observed_on=date_of_quote,
        source=result.history.source,
        cached=result.cached,
    )


def convert_amount(
    amount: Decimal,
    from_currency: str,
    to_currency: str,
    usd_cad: Decimal | None,
) -> Decimal:
    """Convert a stated cash value (not a security price) to CAD or USD."""
    src, dst = _currency(from_currency), _currency(to_currency)
    if not isinstance(amount, Decimal) or not amount.is_finite() or amount < 0:
        raise ValueError("Amount must be a finite, nonnegative Decimal")
    if src == dst:
        return amount
    if usd_cad is None:
        raise FXRateError("A USD/CAD quote is required for mixed-currency valuation")
    rate = validate_usd_cad(usd_cad)
    return amount * rate if src == "USD" else amount / rate
