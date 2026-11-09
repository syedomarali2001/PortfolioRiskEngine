"""Small, human-auditable JSON cache with atomic writes and a TTL."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from portfolio_risk.models import PriceHistory


class HistoryCache:
    def __init__(self, root: str | Path = "data/cache/v1", ttl_hours: int = 24) -> None:
        if ttl_hours <= 0:
            raise ValueError("ttl_hours must be positive")
        self.root = Path(root)
        self.ttl = timedelta(hours=ttl_hours)

    def _path(self, symbol: str, start: date, end: date) -> Path:
        key = f"v1|daily-adjusted|{symbol}|{start.isoformat()}|{end.isoformat()}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.root / f"{digest}.json"

    def get(self, symbol: str, start: date, end: date) -> PriceHistory | None:
        path = self._path(symbol, start, end)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(payload["fetched_at"])
            if fetched.tzinfo is None or datetime.now(timezone.utc) - fetched > self.ttl:
                return None
            if payload["symbol"] != symbol or payload["start"] != start.isoformat() or payload["end"] != end.isoformat():
                return None
            idx = pd.to_datetime([row[0] for row in payload["observations"]])
            prices = pd.Series([float(row[1]) for row in payload["observations"]], index=idx, name="adjusted_close", dtype=float)
            return PriceHistory(
                symbol=symbol,
                prices=prices,
                start=start,
                end=end,
                source=payload["source"],
                fetched_at=fetched,
                quote_currency=payload.get("quote_currency"),
                adjusted=True,
            )
        except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError):
            return None

    def put(self, history: PriceHistory) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "symbol": history.symbol,
            "start": history.start.isoformat(),
            "end": history.end.isoformat(),
            "source": history.source,
            "fetched_at": history.fetched_at.isoformat(),
            "quote_currency": history.quote_currency,
            "adjusted": history.adjusted,
            "observations": [[ts.date().isoformat(), float(value)] for ts, value in history.prices.items()],
        }
        output = self._path(history.symbol, history.start, history.end)
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix="tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, allow_nan=False)
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
