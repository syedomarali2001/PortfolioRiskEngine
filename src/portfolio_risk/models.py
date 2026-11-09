"""Typed input and market-data contracts shared by the CLI and future GUI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

import pandas as pd


@dataclass(frozen=True)
class Holding:
    symbol: str
    market_value: Decimal
    currency: str
    # Trading quotation currency (distinct from the currency of the entered market value).
    # Optional for V1/V2 backwards compatibility; required for V3 if provider metadata is absent.
    quote_currency: str | None = None


@dataclass(frozen=True)
class PortfolioInput:
    holdings: tuple[Holding, ...]
    source_file: str


@dataclass(frozen=True)
class PriceHistory:
    """Daily adjusted closes; end date is exclusive (as in Yahoo Finance).

    ``quote_currency`` is the market-data quotation currency; it is *not*
    necessarily equal to the currency used for the user's market_value input.
    """

    symbol: str
    prices: pd.Series
    start: date
    end: date
    source: str
    fetched_at: datetime
    quote_currency: str | None = None
    adjusted: bool = True


@dataclass(frozen=True)
class QualityReport:
    symbol: str
    observation_count: int
    first_date: date
    last_date: date
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.warnings
