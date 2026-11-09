from __future__ import annotations

from datetime import date

import pytest

from portfolio_risk.exposure import UNKNOWN, ExposureError
from portfolio_risk.exposure_data import load_asset_metadata, load_fund_snapshots


def path(tmp_path, text, name="input.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_fund_snapshot_csv_round_trip(tmp_path):
    p = path(
        tmp_path,
        "fund_symbol,holding_symbol,weight_pct,asset_class,as_of,source,sector,country\n"
        "QQQ,AAPL,35,equity,2026-11-01,Invesco,Technology,US\n"
        "QQQ,MSFT,25,equity,2026-11-01,Invesco,Technology,US\n",
    )
    f = load_fund_snapshots(p)["QQQ"]
    assert len(f.holdings) == 2
    assert f.as_of == date(2026, 11, 1)
    assert f.holdings[0].fraction == pytest.approx(0.35)


def test_asset_class_and_unknown_sector(tmp_path):
    p = path(tmp_path, "symbol,asset_class,sector,country\nGLD,commodity,,\n")
    meta = load_asset_metadata(p)
    assert meta["GLD"].sector == UNKNOWN
    assert meta["GLD"].asset_class == "commodity"


def test_duplicate_asset_metadata_rejected(tmp_path):
    p = path(tmp_path, "symbol,asset_class\nAAPL,equity\nAAPL,equity\n")
    with pytest.raises(ExposureError, match="Duplicate"):
        load_asset_metadata(p)


@pytest.mark.parametrize(
    "header", ["fund_symbol,holding_symbol,weight_pct", "fund_symbol,as_of,source"]
)
def test_fund_csv_missing_columns_fails(tmp_path, header):
    p = path(tmp_path, header + "\nFOO,BAR,5\n")
    with pytest.raises(ExposureError, match="columns"):
        load_fund_snapshots(p)


def test_mixed_asof_rejected(tmp_path):
    p = path(
        tmp_path,
        "fund_symbol,holding_symbol,weight_pct,asset_class,as_of,source\n"
        "QQQ,AAPL,25,equity,2026-11-01,issuer\n"
        "QQQ,MSFT,25,equity,2026-10-01,issuer\n",
    )
    with pytest.raises(ExposureError, match="mixed"):
        load_fund_snapshots(p)


def test_mixed_source_rejected(tmp_path):
    p = path(
        tmp_path,
        "fund_symbol,holding_symbol,weight_pct,asset_class,as_of,source\n"
        "QQQ,AAPL,25,equity,2026-11-01,issuer\n"
        "QQQ,MSFT,25,equity,2026-11-01,blog\n",
    )
    with pytest.raises(ExposureError, match="mixed"):
        load_fund_snapshots(p)


def test_bad_date_rejected(tmp_path):
    p = path(
        tmp_path,
        "fund_symbol,holding_symbol,weight_pct,asset_class,as_of,source\n"
        "QQQ,AAPL,25,equity,not-date,issuer\n",
    )
    with pytest.raises(ExposureError, match="as_of"):
        load_fund_snapshots(p)


def test_bad_weight_rejected(tmp_path):
    p = path(
        tmp_path,
        "fund_symbol,holding_symbol,weight_pct,asset_class,as_of,source\n"
        "QQQ,AAPL,XX,equity,2026-11-01,issuer\n",
    )
    with pytest.raises(ExposureError, match="invalid weight"):
        load_fund_snapshots(p)


def test_zero_weight_rejected(tmp_path):
    p = path(
        tmp_path,
        "fund_symbol,holding_symbol,weight_pct,asset_class,as_of,source\n"
        "QQQ,AAPL,0,equity,2026-11-01,issuer\n",
    )
    with pytest.raises(ExposureError, match="weight"):
        load_fund_snapshots(p)


def test_bad_class_rejected(tmp_path):
    p = path(tmp_path, "symbol,asset_class\nAAPL,tech\n")
    with pytest.raises(ExposureError, match="asset_class"):
        load_asset_metadata(p)


def test_nonexistent_file_rejected(tmp_path):
    with pytest.raises(ExposureError, match="not found"):
        load_fund_snapshots(tmp_path / "nope.csv")


def test_empty_data_rows_rejected(tmp_path):
    p = path(tmp_path, "symbol,asset_class\n")
    with pytest.raises(ExposureError, match="No data"):
        load_asset_metadata(p)


def test_tab_delimited_txt(tmp_path):
    p = path(tmp_path, "symbol\tasset_class\nAAPL\tequity\n", name="tickers.txt")
    assert load_asset_metadata(p)["AAPL"].asset_class == "equity"
