"""TickerInside US fund-data adapter (CC BY 4.0, attribution required).

No key. Holds are percent of fund assets, NOT fractions. The index discovers
US-listed ETF/ETP tickers; non-US listings and non-funds are never guessed.
Cache is optional/automatic, never a user-maintained holdings database.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from portfolio_risk.exposure import Constituent, ExposureError, FundSnapshot
from portfolio_risk.input import normalize_symbol

BASE_URL = "https://tickerinside.com/api/v1"
USER_AGENT = "PortfolioRiskEngine/0.7.3 (educational ETF exposure audit)"


class TickerInsideError(ValueError):
    pass


class FundUnavailable(TickerInsideError):
    pass


@dataclass(frozen=True)
class FundRecord:
    symbol: str
    name: str
    holdings: dict[str, float]   # PERCENT of fund NAV, not 0..1
    holdings_as_of: date | None
    source_layer: str
    filed: date | None
    positions_reported: int | None
    holdings_kind: str
    sector_pct: dict[str, float]
    country_pct: dict[str, float]
    listed_pct: float | None
    expense_ratio_pct: float | None
    source_url: str
    cite_as: str
    fetched_from_cache: bool = False
    category_asset_class: str = ""
    category_basis: str = ""
    category_as_of: date | None = None
    category_stock_count: int | None = None
    sector_coverage_pct: float | None = None
    other_funds_pct: float | None = None
    non_equity_positions: tuple[str, ...] = ()
    cash_holdings: dict[str, float] | None = None

    @property
    def equity_validation(self) -> str:
        """Explicit stock kind OR corroborated issuer category, never ticker guessing."""
        if self.holdings_kind in {"bond", "derivative", "commodity", "crypto", "multi-asset"}:
            return "non_equity"
        if self.non_equity_positions:
            return "mixed_positions"
        if self.holdings_kind == "stock":
            return "explicit_stock"
        if (self.holdings_kind == "unknown"
            and self.category_asset_class == "equity"
            and self.category_basis == "holdings"
            and self.category_as_of == self.holdings_as_of
            and self.category_stock_count is not None and self.category_stock_count > 0
            and self.sector_coverage_pct is not None and self.sector_coverage_pct >= 95
            and self.other_funds_pct is not None and self.other_funds_pct <= 0.5
        ):
            return "issuer_evidence"
        return "unverified"

    @property
    def is_metal_or_spot_crypto(self) -> bool:
        if self.holdings_kind not in {"commodity", "crypto"} or not self.holdings:
            return False
        if self.non_equity_positions or self.known_pct > 100.00001:
            return False
        return set(self.holdings).issubset({"GOLD", "SILVER", "PLATINUM", "PALLADIUM", "BITCOIN", "ETHEREUM", "BTC", "ETH"})

    @property
    def exposure_support(self) -> str:
        if self.equity_validation in {"explicit_stock", "issuer_evidence"}:
            return "equity"
        if self.is_metal_or_spot_crypto:
            return "spot_asset"
        return "price_only"

    @property
    def known_pct(self) -> float:
        return sum(self.holdings.values()) + sum((self.cash_holdings or {}).values())

    @property
    def is_equity_holdings(self) -> bool:
        return self.equity_validation in {"explicit_stock", "issuer_evidence"}

    def to_snapshot(self) -> FundSnapshot:
        if self.exposure_support == "price_only":
            raise TickerInsideError(
                f"{self.symbol}: holdings type {self.holdings_kind} cannot be safely expanded "
                "(bond/derivative/unknown or mixed positions); price-risk analysis only")
        if not self.holdings_as_of:
            raise TickerInsideError(f"{self.symbol}: no verified holdings disclosure date")
        if not self.holdings:
            raise TickerInsideError(f"{self.symbol}: no holdings to expand")
        if self.known_pct > 100.00001:
            raise TickerInsideError(f"{self.symbol}: gross reported weights exceed 100%; unsuitable for long-only look-through")
        return FundSnapshot(
            self.symbol,
            (tuple(Constituent(s, w / 100.0,
                "equity" if self.exposure_support == "equity" else
                "crypto" if s in {"BITCOIN", "ETHEREUM", "BTC", "ETH"} else "commodity")
                for s, w in self.holdings.items())
             + tuple(Constituent(s, w / 100.0, "cash") for s, w in (self.cash_holdings or {}).items())),
            f"TickerInside {self.source_layer}; {self.source_url}; CC BY 4.0",
            self.holdings_as_of,
            date.today(),
        )


def _percent_map(value: object, field: str, *, total_limit: float = 105.0) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TickerInsideError(f"{field}: expected mapping")
    result = {}
    for raw, weight in value.items():
        if not isinstance(raw, str) or not raw.strip():
            raise TickerInsideError(f"{field}: invalid key")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise TickerInsideError(f"{field}/{raw}: invalid percent weight")
        if weight > 0:
            key = raw.strip().upper() if field == "holdings" else raw.strip()
            if key in result:
                raise TickerInsideError(f"{field}: duplicate {key}")
            result[key] = float(weight)
    if sum(result.values()) > total_limit:
        raise TickerInsideError(f"{field}: reported weights exceed {total_limit}% (possibly leveraged / gross weights); not usable as net exposure")
    return result


# Month-coded futures: conservative exclusion from stock-look-through.
_FUTURE = re.compile(r"^(?:ES|MES|NQ|MNQ|YM|RTY|M2K|CL|MCL|GC|MGC|SI|SIL|ZN|ZB|ZF|ZQ|6E|6B|DX|NG|RB|HO|HG|ZC|ZW|ZS|VX|MFS|MSF)[FGHJKMNQUVXZ]\d{1,2}$", re.I)
_CASH_WORDS = {"CASH", "USD", "CAD", "USD CASH", "CAD CASH", "US DOLLAR", "US DOLLARS", "DOLLARS", "CASH&OTHER"}
_NON_EQUITY_WORDS = {"FUTURES", "SWAP", "MARGIN"}


def _non_equity_position(name: str) -> bool:
    value = name.strip().upper()
    return bool(_FUTURE.fullmatch(value) or value in _NON_EQUITY_WORDS or
                value.startswith(("FUTURE ", "SWAP ", "MARGIN ")))


def _parse_holdings(value: object) -> tuple[dict[str, float], tuple[str, ...], dict[str, float]]:
    """Preserve valid stock rows without misrepresenting signed derivative weights.

    Excluded instruments are recorded; any excluded position blocks equity-only
    look-through rather than causing the whole fund record to vanish.
    """
    if value is None:
        return {}, (), {}
    if not isinstance(value, dict):
        raise TickerInsideError("holdings: expected mapping")
    included: dict[str, float] = {}
    cash: dict[str, float] = {}
    skipped = []
    for raw, weight in value.items():
        if not isinstance(raw, str) or not raw.strip():
            raise TickerInsideError("holdings: invalid security identifier")
        name = raw.strip().upper()
        if name in included or name in cash:
            raise TickerInsideError(f"holdings: duplicate {name}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise TickerInsideError(f"holdings/{name}: invalid percent weight")
        if name in _CASH_WORDS or name.startswith("CASH "):
            if 0 <= weight <= 100 and weight > 0:
                cash[name] = float(weight)
            elif weight < 0 or weight > 100:
                skipped.append(f"{name} ({weight:+.4g}%)")
            continue
        if _non_equity_position(name):
            skipped.append(f"{name} ({weight:+.4g}%)")
            continue
        if weight < 0 or weight > 100:
            raise TickerInsideError(f"holdings/{name}: invalid stock percent weight")
        if weight > 0:
            included[name] = float(weight)
    if sum(included.values()) + sum(cash.values()) > 105:
        raise TickerInsideError("holdings: reported stock/cash weights exceed 105%; not a valid long-only snapshot")
    return included, tuple(skipped), cash


def _optional_percent(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) if 0 <= value <= 105 else None


def _date(raw: object, field: str) -> date | None:
    if raw in (None, ""):
        return None
    try:
        return date.fromisoformat(str(raw))
    except ValueError as e:
        raise TickerInsideError(f"{field}: invalid date") from e


def parse_fund_record(payload: object, expected_symbol: str) -> FundRecord:
    if not isinstance(payload, dict):
        raise TickerInsideError("TickerInside fund response is not an object")
    symbol = normalize_symbol(str(payload.get("ticker", "")))
    if symbol != normalize_symbol(expected_symbol):
        raise TickerInsideError(f"TickerInside returned {symbol}, expected {expected_symbol}")
    breakdown = payload.get("breakdown") or {}
    if not isinstance(breakdown, dict):
        raise TickerInsideError("breakdown: expected object")
    holdings, excluded, cash = _parse_holdings(payload.get("holdings"))
    category = payload.get("category") or {}
    if not isinstance(category, dict):
        category = {}
    measured = category.get("measured") or {}
    if not isinstance(measured, dict):
        measured = {}
    stocks = measured.get("stocks")
    if not isinstance(stocks, int) or isinstance(stocks, bool) or stocks < 0:
        stocks = None
    kind = str(payload.get("holdings_kind") or ("stock" if payload.get("stock_pct") is not None else "unknown")).lower()
    listed = payload.get("listed_pct")
    if listed is not None:
        if isinstance(listed, bool) or not isinstance(listed, (int, float)) or not math.isfinite(listed) or listed < 0 or listed > 105:
            raise TickerInsideError("listed_pct invalid")
        listed = float(listed)
    expense = payload.get("expense_ratio")
    if isinstance(expense, (int, float)) and not isinstance(expense, bool) and math.isfinite(expense) and 0 <= expense <= 100:
        expense = float(expense)
    else:
        expense = None
    positions = payload.get("positions")
    if not isinstance(positions, int) or isinstance(positions, bool) or positions < 0:
        positions = None
    url = payload.get("page_url")
    if not isinstance(url, str) or not url.startswith("https://tickerinside.com/"):
        url = f"https://tickerinside.com/api/v1/fund/{quote(symbol.lower(),safe='.')}.json"
    return FundRecord(
        symbol=symbol,
        name=str(payload.get("name") or symbol),
        holdings=holdings,
        holdings_as_of=_date(payload.get("holdings_as_of"), "holdings_as_of"),
        source_layer=str(payload.get("source_layer") or "unknown"),
        filed=_date(payload.get("filed"), "filed"),
        positions_reported=positions,
        holdings_kind=kind,
        sector_pct=_percent_map(breakdown.get("sectors_pct"), "sectors_pct"),
        country_pct=_percent_map(breakdown.get("countries_pct"), "countries_pct"),
        listed_pct=listed,
        expense_ratio_pct=expense,
        source_url=url,
        cite_as=str(payload.get("cite_as") or f"TickerInside, {url}, CC BY 4.0"),
        category_asset_class=str(category.get("asset_class") or "").lower(),
        category_basis=str(category.get("basis") or "").lower(),
        category_as_of=_date(category.get("as_of"), "category.as_of"),
        category_stock_count=stocks,
        sector_coverage_pct=_optional_percent(breakdown.get("sector_coverage_pct")),
        other_funds_pct=_optional_percent(breakdown.get("other_funds_pct")),
        non_equity_positions=excluded,
        cash_holdings=cash,
    )


class TickerInsideClient:
    """Injectable HTTP getter for deterministic unit tests; no request without a caller."""

    def __init__(self, cache_dir: str | Path = "data/cache/tickerinside", ttl_hours: int = 24,
                 getter=None):
        if ttl_hours < 0:
            raise ValueError("ttl_hours cannot be negative")
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.getter = getter or self._get_json

    @staticmethod
    def _get_json(url: str) -> dict:
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(req, timeout=20) as response:
                raw = response.read(12_000_001)
            if len(raw) > 12_000_000:
                raise TickerInsideError("Provider response exceeded 12 MB limit")
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise TickerInsideError("Provider returned non-object JSON")
            return obj
        except HTTPError as e:
            if e.code == 404:
                raise FundUnavailable("Ticker not covered by TickerInside") from e
            raise TickerInsideError(f"TickerInside HTTP {e.code}") from e
        except (URLError, TimeoutError, OSError, ValueError) as e:
            raise TickerInsideError(f"TickerInside request failed: {e}") from e

    def _read(self, key: str, url: str, refresh: bool = False) -> tuple[dict, bool]:
        path = self.cache_dir / f"{key}.json"
        if not refresh and self.ttl.total_seconds() > 0 and path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
                stored = datetime.fromisoformat(doc["cached_at"])
                if stored.tzinfo and datetime.now(timezone.utc) - stored < self.ttl and isinstance(doc["payload"], dict):
                    return doc["payload"], True
            except (OSError, ValueError, KeyError, TypeError):
                pass
        data = self.getter(url)
        if not isinstance(data, dict):
            raise TickerInsideError("Invalid API response type")
        if self.ttl.total_seconds() > 0:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                fd, temp = tempfile.mkstemp(dir=self.cache_dir, prefix="fetch_", suffix=".json")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump({"cached_at": datetime.now(timezone.utc).isoformat(), "payload": data}, f)
                    os.replace(temp, path)
                finally:
                    if os.path.exists(temp):
                        os.unlink(temp)
            except OSError:
                pass  # cache failure must never invalidate successful API data
        return data, False

    def fund(self, symbol: str, *, refresh: bool = False) -> FundRecord:
        sym = normalize_symbol(symbol)
        if sym.endswith(".TO") or sym.endswith("-USD"):
            raise FundUnavailable("Non-US ETF ticker is outside TickerInside fund scope")
        url = f"{BASE_URL}/fund/{quote(sym.lower(), safe='.')}.json"
        doc, cached = self._read(f"fund_{sym.lower().replace('.', '_')}", url, refresh)
        record = parse_fund_record(doc, sym)
        return FundRecord(**{**record.__dict__, "fetched_from_cache": cached})

    def index(self, *, refresh: bool = False) -> tuple[set[str], dict]:
        data, _ = self._read("index", f"{BASE_URL}/index.json", refresh)
        # API's index uses multiple ticker families; read ONLY documented tickers_* lists
        # plus 'tickers', not arbitrary strings elsewhere in the payload.
        names = set()
        for key, val in data.items():
            if key == "tickers" or key.startswith("tickers_"):
                if isinstance(val, list):
                    names.update(x.upper() for x in val if isinstance(x, str))
        if not names:
            raise TickerInsideError("TickerInside index returned no fund tickers; cannot discover coverage safely")
        return names, {"as_of": data.get("as_of"), "coverage": data.get("coverage", {}), "count": len(names)}
