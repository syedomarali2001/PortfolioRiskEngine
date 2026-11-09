"""UI-side portfolio input validation. The engine owns all finance mathematics."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

SUPPORTED_CURRENCIES = ("USD", "CAD")
MAX_HOLDINGS = max(1, min(200, int(os.environ.get("PORTFOLIO_RISK_UI_MAX_HOLDINGS", "40"))))
_SYMBOL = re.compile(r"^[A-Z0-9^][A-Z0-9.^=_\-]{0,24}$")


class PortfolioError(ValueError):
    pass


def _number(value: object, label: str) -> float:
    try:
        x = float(Decimal(str(value)))
    except (TypeError, ValueError, InvalidOperation) as e:
        raise PortfolioError(f"{label} must be a valid number") from e
    if not math.isfinite(x):
        raise PortfolioError(f"{label} must be finite")
    return x


def normalized_symbol(symbol: object) -> str:
    text = str(symbol or "").strip().upper()
    if not _SYMBOL.fullmatch(text):
        raise PortfolioError(f"Invalid ticker symbol: {symbol!r}")
    return text


def validate_holdings(rows: list[dict], mode: str, *, max_holdings: int = MAX_HOLDINGS) -> list[dict]:
    """Never silently normalize weights, infer FX, or drop invalid rows."""
    if not rows:
        raise PortfolioError("Add at least one investment")
    if len(rows) > max_holdings:
        raise PortfolioError(f"The configured limit is {max_holdings} holdings")
    if mode not in ("values", "weights"):
        raise PortfolioError("Unknown portfolio entry method")
    clean = []
    seen = set()
    for row in rows:
        symbol = normalized_symbol(row.get("symbol"))
        if symbol in seen:
            raise PortfolioError(f"Duplicate ticker: {symbol}")
        seen.add(symbol)
        currency = str(row.get("currency", "USD")).strip().upper()
        if currency not in SUPPORTED_CURRENCIES:
            raise PortfolioError(f"{symbol}: currency must be USD or CAD")
        field = "market_value" if mode == "values" else "weight"
        v = _number(row.get(field), f"{symbol}: {field}")
        if v <= 0:
            raise PortfolioError(f"{symbol}: {field} must be greater than zero")
        item = {"symbol": symbol, field: v, "currency": currency}
        # Trading currency is independent of the denomination of an entered value.
        # Only attach it when explicitly supplied; the engine can use provider
        # metadata when it is available. Never infer USD from a USD market value.
        quote = str(row.get("quote_currency") or "").strip().upper()
        if quote and quote not in SUPPORTED_CURRENCIES:
            raise PortfolioError(f"{symbol}: market quote currency must be USD, CAD or Auto")
        if quote:
            item["quote_currency"] = quote
        clean.append(item)
    if mode == "weights":
        total = sum(row["weight"] for row in clean)
        if abs(total - 100.0) > 0.0001:
            raise PortfolioError(f"Target weights must add to 100% (currently {total:.4f}%)")
    return clean


def engine_rows(rows: list[dict], mode: str, base: str, *, total_value: float | None = None) -> list[dict]:
    if base not in SUPPORTED_CURRENCIES:
        raise PortfolioError("Base currency must be USD or CAD")
    clean = validate_holdings(rows, mode)
    if mode == "weights":
        if total_value is None or _number(total_value, "Investment amount") <= 0:
            raise PortfolioError("Enter a positive hypothetical investment amount")
        # Keep one exact total after float rounding to avoid weight drift.
        v = _number(total_value, "Investment amount")
        vals = [round(v * r["weight"] / 100.0, 8) for r in clean]
        vals[-1] += v - sum(vals)
        return [dict({"symbol": r["symbol"], "market_value": vals[i], "currency": base},
                     **({"quote_currency": r["quote_currency"]} if r.get("quote_currency") else {}))
                for i, r in enumerate(clean)]
    return clean


def update_holding_amount(rows: list[dict], symbol: str, field: str, value: object) -> bool:
    """Commit a Streamlit widget edit to the actual source holdings, by ticker.

    Updating the session holding itself (rather than only a widget's display
    value) ensures that report fingerprints, allocations and totals all change.
    """
    if field not in {"market_value", "weight"}:
        raise PortfolioError("Unsupported holding field")
    ticker = normalized_symbol(symbol)
    number = _number(value, f"{ticker}: {field}")
    if number <= 0 or (field == "weight" and number > 100):
        raise PortfolioError("Holding value must be positive; allocations must be between 0% and 100%")
    for holding in rows:
        if holding.get("symbol") == ticker:
            if float(holding.get(field, 0)) == number:
                return False
            holding[field] = number
            return True
    raise PortfolioError(f"Investment {ticker} is no longer in your portfolio")


def to_csv(rows: list[dict]) -> str:
    sink = io.StringIO(newline="")
    columns = ["symbol", "market_value", "currency"]
    if any(row.get("quote_currency") for row in rows):
        columns.append("quote_currency")
    writer = csv.DictWriter(sink, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in columns})
    return sink.getvalue()


def import_values_csv(data: bytes, max_holdings: int = MAX_HOLDINGS) -> list[dict]:
    if len(data) > 1_000_000:
        raise PortfolioError("CSV is too large (maximum 1 MB)")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise PortfolioError("Please upload a UTF-8 CSV file") from e
    if not text.strip():
        raise PortfolioError("CSV is empty")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"symbol", "market_value", "currency"}.issubset(set(reader.fieldnames)):
        raise PortfolioError("CSV requires symbol, market_value and currency columns")
    rows = []
    for row in reader:
        if None in row or all(not str(v or "").strip() for v in row.values()):
            if None in row:
                raise PortfolioError("Malformed CSV: unexpected extra columns")
            continue
        rows.append(row)
        if len(rows) > max_holdings:
            raise PortfolioError(f"The configured limit is {max_holdings} holdings")
    return validate_holdings(rows, "values", max_holdings=max_holdings)


def fingerprint(rows: list[dict], base: str, start: str, end: str) -> str:
    payload = json.dumps({"rows": rows, "base": base, "start": start, "end": end}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def exact_money(value: object, currency: str = "USD") -> str:
    if value is None:
        return "Unavailable"
    return f"{currency} {float(value):,.2f}"


def pct(value: object, digits: int = 2) -> str:
    return "Unavailable" if value is None else f"{float(value) * 100:.{digits}f}%"


def safe_portfolio_nominal_total(rows: list[dict], base: str) -> float | None:
    """Mixed-currency portfolios need actual FX; never sum them unconverted."""
    if not rows or any(row["currency"] != base for row in rows):
        return None
    return sum(float(row["market_value"]) for row in rows)


def dividend_summary(report: dict) -> dict:
    return report.get("income_summary") or {}


def show_complete_dividend_yield(summary: dict) -> bool:
    return (summary.get("income_coverage_fraction") == 1.0
            and summary.get("fully_covered_portfolio_yield") is not None)
