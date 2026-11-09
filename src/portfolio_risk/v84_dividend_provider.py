"""Best-effort Yahoo *unadjusted* cash-distribution snapshots and dated local cache."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from portfolio_risk.v84_dividends import DividendError


def fetch_yahoo_distribution_snapshot(
    symbol: str, *, as_of: date, expected_currency: str | None = None, ticker_factory=None
) -> dict:
    """Fetch raw close + cash events, never return an adjusted-price backtest series.

    Expected currency is optional explicit user validation, not an inference from
    the currency of user-entered market values. Missing metadata fails closed.
    """
    try:
        if ticker_factory is None:
            import yfinance as yf

            ticker_factory = yf.Ticker
        ticker = ticker_factory(symbol)
        first_requested = as_of - timedelta(days=3 * 366)
        frame = ticker.history(
            start=first_requested.isoformat(),
            end=(as_of + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=True,
            timeout=20,
        )
        if frame is None or frame.empty or "Close" not in frame.columns:
            raise DividendError("No unadjusted daily closing prices")
        if "Dividends" not in frame.columns:
            raise DividendError("Provider did not return a verified Dividends event column")
        import pandas as pd

        prices = pd.to_numeric(frame["Close"], errors="coerce")
        dates = pd.to_datetime(prices.index)
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.normalize()
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise DividendError("Non-unique or unsorted provider dates")
        rows = [
            (d.date(), float(value), raw)
            for d, value, raw in zip(
                dates, prices, pd.to_numeric(frame["Dividends"], errors="coerce")
            , strict=False)
        ]
        if not rows or any(d > as_of or d < first_requested for d, _, _ in rows):
            raise DividendError("Provider returned dates outside requested interval")
        valid_close = [(d, value) for d, value, _ in rows if value > 0 and pd.notna(value)]
        if not valid_close:
            raise DividendError("No positive unadjusted close available")
        currency = ""
        try:
            metadata = ticker.get_history_metadata()
            if isinstance(metadata, dict):
                currency = str(metadata.get("currency") or "").upper().strip()
        except Exception:
            pass
        if currency not in {"CAD", "USD"}:
            try:
                info = ticker.fast_info
                raw = info.get("currency") if hasattr(info, "get") else info["currency"]
                currency = str(raw or "").strip().upper()
            except Exception:
                pass
        if expected_currency:
            expected_currency = expected_currency.strip().upper()
            if expected_currency not in {"CAD", "USD"}:
                raise DividendError("Expected currency must be CAD or USD")
            if currency in {"CAD", "USD"} and currency != expected_currency:
                raise DividendError(
                    f"Provider currency {currency} disagrees with requested {expected_currency}"
                )
            if currency not in {"CAD", "USD"}:
                currency = expected_currency
        if currency not in {"CAD", "USD"}:
            raise DividendError(
                "Quote currency unavailable; supply quote_currency in portfolio file"
            )
        events = []
        for d, _, amount in rows:
            if not pd.notna(amount) or float(amount) < 0:
                raise DividendError("Dividend event column contains missing/invalid entries")
            if float(amount) > 0:
                events.append({"date": d.isoformat(), "amount": float(amount)})
        return {
            "symbol": symbol,
            "quote_currency": currency,
            "close_price": valid_close[-1][1],
            "price_date": valid_close[-1][0].isoformat(),
            "history_first_date": rows[0][0].isoformat(),
            "history_last_date": rows[-1][0].isoformat(),
            "dividends": events,
            "source": "Yahoo Finance via yfinance; unadjusted close and cash distributions",
        }
    except DividendError:
        raise
    except Exception as exc:
        raise DividendError(f"Yahoo distribution retrieval failed: {exc}") from exc


def get_cached_dividends(
    symbol: str,
    *,
    as_of: date,
    cache_dir: Path,
    expected_currency: str | None = None,
    refresh: bool = False,
    ticker_factory=None,
) -> dict:
    """One cache record per requested as-of date; never silently substitute another date."""
    if not symbol or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=_" for ch in symbol):
        raise DividendError("Invalid dividend symbol")
    path = Path(cache_dir) / f"{symbol}_{as_of.isoformat()}.json"
    if path.is_file() and not refresh:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise DividendError("Unreadable dividend cache; try --refresh") from exc
        if not isinstance(payload, dict) or payload.get("symbol") != symbol:
            raise DividendError("Dividend cache identity mismatch; try --refresh")
        if expected_currency and payload.get("quote_currency") != expected_currency:
            raise DividendError("Cached dividend currency mismatch; try --refresh")
        return payload
    result = fetch_yahoo_distribution_snapshot(
        symbol, as_of=as_of, expected_currency=expected_currency, ticker_factory=ticker_factory
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    except OSError as exc:
        raise DividendError(f"Cannot persist dividend cache: {exc}") from exc
    return result
