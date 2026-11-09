"""V7 ingestion: strict, source-attributed fund snapshots and asset metadata.

This module never downloads holdings data. CSVs from ETF providers or user
research are normalized to a common schema without guessing omitted weights.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

from portfolio_risk.exposure import (
    ASSET_CLASSES,
    UNKNOWN,
    Constituent,
    ExposureError,
    FundSnapshot,
    SecurityMetadata,
    validate_snapshots,
)
from portfolio_risk.input import PortfolioFormatError, normalize_symbol


def _rows(path: str | Path, required: set[str]) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ExposureError(f"Metadata file not found: {file_path}")
    if file_path.suffix.lower() not in {".csv", ".txt", ".tsv"}:
        raise ExposureError("Metadata must be CSV, TXT or TSV")
    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as file:
            header = file.readline()
            delimiter = "\t" if "\t" in header and "," not in header else ","
            file.seek(0)
            reader = csv.DictReader(file, delimiter=delimiter)
            if not reader.fieldnames:
                raise ExposureError("Metadata file requires a header")
            keys = [name.strip().lower() for name in reader.fieldnames]
            if len(keys) != len(set(keys)) or not required.issubset(keys):
                raise ExposureError(f"CSV must have unique columns including {sorted(required)}")
            reader.fieldnames = keys
            output = []
            for index, row in enumerate(reader, 2):
                if row is None or not any(str(v or "").strip() for v in row.values()):
                    continue
                if None in row or any(v is None for v in row.values()):
                    raise ExposureError(f"Row {index}: incorrect column count")
                output.append({k: v.strip() for k, v in row.items()})
    except (UnicodeError, OSError, csv.Error) as exc:
        raise ExposureError(f"Unable to read metadata file: {exc}") from exc
    if not output:
        raise ExposureError(f"No data rows in {file_path}")
    return output


def _symbol(raw: str) -> str:
    try:
        return normalize_symbol(raw)
    except PortfolioFormatError as exc:
        raise ExposureError(str(exc)) from exc


def _class(raw: str, *, default: str = UNKNOWN) -> str:
    value = raw.strip().lower() if raw else default
    if value.lower() == "unknown":
        return UNKNOWN
    if value not in ASSET_CLASSES:
        raise ExposureError(
            f"Unrecognized asset_class {raw!r}; use equity, etf, bond, cash, commodity, crypto, other"
        )
    return value


def _label(raw: str) -> str:
    return raw.strip() or UNKNOWN


def load_asset_metadata(path: str | Path) -> dict[str, SecurityMetadata]:
    """CSV: symbol,asset_class,sector,country (sector/country optional)."""
    result = {}
    for row in _rows(path, {"symbol", "asset_class"}):
        symbol = _symbol(row["symbol"])
        if symbol in result:
            raise ExposureError(f"Duplicate asset classification: {symbol}")
        result[symbol] = SecurityMetadata(
            symbol,
            _class(row["asset_class"]),
            _label(row.get("sector", "")),
            _label(row.get("country", "")),
        )
    return result


def load_fund_snapshots(path: str | Path) -> dict[str, FundSnapshot]:
    """CSV schema: fund_symbol,holding_symbol,weight_pct,asset_class,sector,country,as_of,source.

    weights are PERCENT (e.g. 25 = 25%, NOT 0.25); row sums <=100.
    Consistency of source and disclosure date within each fund is required.
    """
    grouped: dict[str, list[Constituent]] = defaultdict(list)
    sources: dict[str, tuple[date, str]] = {}
    for row in _rows(
        path, {"fund_symbol", "holding_symbol", "weight_pct", "asset_class", "as_of", "source"}
    ):
        fund = _symbol(row["fund_symbol"])
        constituent = _symbol(row["holding_symbol"])
        try:
            pct = float(row["weight_pct"])
        except ValueError as exc:
            raise ExposureError(f"{fund}/{constituent}: invalid weight_pct") from exc
        try:
            as_of = date.fromisoformat(row["as_of"])
        except ValueError as exc:
            raise ExposureError(f"{fund}: as_of must be YYYY-MM-DD disclosure date") from exc
        src = row["source"].strip()
        if not src:
            raise ExposureError(f"{fund}: source attribution is required")
        if fund in sources and sources[fund] != (as_of, src):
            raise ExposureError(
                f"{fund}: mixed disclosure dates or sources; use one snapshot per fund"
            )
        sources[fund] = (as_of, src)
        grouped[fund].append(
            Constituent(
                constituent,
                pct / 100.0,
                _class(row["asset_class"]),
                _label(row.get("sector", "")),
                _label(row.get("country", "")),
            )
        )
    output = {
        fund: FundSnapshot(fund, tuple(parts), sources[fund][1], sources[fund][0])
        for fund, parts in grouped.items()
    }
    validate_snapshots(output)
    return output
