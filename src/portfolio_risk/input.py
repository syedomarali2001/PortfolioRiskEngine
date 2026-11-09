"""Strict CSV/TXT portfolio imports, without market-data/network side effects."""

from __future__ import annotations

import csv
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from portfolio_risk.models import Holding, PortfolioInput


class PortfolioFormatError(ValueError):
    """The supplied portfolio file does not meet the V1 schema."""


# Yahoo-style symbols: equities, .TO listings, BTC-USD, FX =X, preferred shares.
_SYMBOL_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.^=\-]{0,31}$")
_REQUIRED = {"symbol", "market_value", "currency"}
_SUPPORTED_CURRENCIES = {"CAD", "USD"}


def normalize_symbol(raw: str) -> str:
    symbol = str(raw).strip().upper()
    if not _SYMBOL_RE.fullmatch(symbol):
        raise PortfolioFormatError(f"Invalid ticker symbol: {raw!r}")
    return symbol


def parse_portfolio(path: str | Path) -> PortfolioInput:
    file_path = Path(path)
    if file_path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
        raise PortfolioFormatError("Use a .csv, .txt or .tsv portfolio file")
    if not file_path.is_file():
        raise PortfolioFormatError(f"Portfolio file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as file:
            header = file.readline()
            if not header.strip():
                raise PortfolioFormatError("The portfolio file is empty")
            delimiter = "\t" if "\t" in header and "," not in header else ","
            file.seek(0)
            reader = csv.DictReader(file, delimiter=delimiter)
            if not reader.fieldnames:
                raise PortfolioFormatError("A header row is required")
            header_fields = [item.strip().lower() for item in reader.fieldnames]
            if len(header_fields) != len(set(header_fields)):
                raise PortfolioFormatError("Duplicate column name in header")
            if not _REQUIRED.issubset(header_fields):
                missing = ", ".join(sorted(_REQUIRED - set(header_fields)))
                raise PortfolioFormatError(f"Missing required column(s): {missing}")
            reader.fieldnames = header_fields

            holdings: list[Holding] = []
            seen: set[str] = set()
            for row_num, row in enumerate(reader, start=2):
                if not any(
                    (value or "").strip() for value in row.values() if isinstance(value, str)
                ):
                    continue
                if None in row or any(value is None for value in row.values()):
                    raise PortfolioFormatError(f"Row {row_num}: incorrect number of columns")
                symbol = normalize_symbol(row["symbol"])
                if symbol in seen:
                    raise PortfolioFormatError(
                        f"Row {row_num}: duplicate symbol {symbol}; aggregate it explicitly first"
                    )
                seen.add(symbol)
                currency = row["currency"].strip().upper()
                if currency not in _SUPPORTED_CURRENCIES:
                    raise PortfolioFormatError(
                        f"Row {row_num}: currency must be CAD or USD, got {currency!r}"
                    )
                raw_value = row["market_value"].strip()
                try:
                    value = Decimal(raw_value)
                except InvalidOperation as exc:
                    raise PortfolioFormatError(
                        f"Row {row_num}: market_value must be numeric"
                    ) from exc
                if not value.is_finite() or value <= 0:
                    raise PortfolioFormatError(
                        f"Row {row_num}: market_value must be finite and greater than zero"
                    )
                raw_quote = (row.get("quote_currency") or "").strip().upper()
                if raw_quote and raw_quote not in _SUPPORTED_CURRENCIES:
                    raise PortfolioFormatError(
                        f"Row {row_num}: quote_currency must be CAD, USD or blank"
                    )
                holdings.append(
                    Holding(
                        symbol=symbol,
                        market_value=value,
                        currency=currency,
                        quote_currency=raw_quote or None,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PortfolioFormatError(f"Unable to read portfolio file: {exc}") from exc

    if not holdings:
        raise PortfolioFormatError("Portfolio has no holdings")
    return PortfolioInput(tuple(holdings), source_file=str(file_path))
