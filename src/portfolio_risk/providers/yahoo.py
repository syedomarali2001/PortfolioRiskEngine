"""Yahoo Finance research adapter; no API credentials needed.

The provider interface keeps this replaceable if we later switch to a licensed
vendor. Do not redistribute data without checking the provider's conditions.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from portfolio_risk.models import PriceHistory
from portfolio_risk.providers.base import MarketDataError


class YahooFinanceProvider:
    def get_history(self, symbol: str, start: date, end: date) -> PriceHistory:
        if start >= end:
            raise ValueError("start must be earlier than end (end is exclusive)")
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            frame = ticker.history(
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=True,
                actions=False,
                raise_errors=True,
                timeout=20,
            )
            if frame is None or frame.empty or "Close" not in frame.columns:
                raise MarketDataError(f"{symbol}: no daily close prices for selected dates")

            prices = pd.to_numeric(frame["Close"], errors="coerce").copy()
            idx = pd.to_datetime(prices.index)
            if idx.tz is not None:
                # Keep exchange-local calendar dates; don't shift to UTC first.
                idx = idx.tz_localize(None)
            prices.index = idx.normalize()
            prices.name = "adjusted_close"

            currency = None
            try:
                metadata = ticker.get_history_metadata()
                raw_currency = metadata.get("currency") if isinstance(metadata, dict) else None
                if isinstance(raw_currency, str) and raw_currency.strip():
                    currency = raw_currency.strip().upper()
            except Exception:
                # Missing metadata isn't a reason to discard valid price history.
                pass

            return PriceHistory(
                symbol=symbol,
                prices=prices,
                start=start,
                end=end,
                source="Yahoo Finance (via yfinance)",
                fetched_at=datetime.now(timezone.utc),
                quote_currency=currency,
                adjusted=True,
            )
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError(f"Unable to retrieve {symbol}: {exc}") from exc
