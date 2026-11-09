from __future__ import annotations
import json
from pathlib import Path

from portfolio_risk.cli import main


PROJECT = Path(__file__).resolve().parents[1] / "examples"


def arguments():
    return ["exposure", str(PROJECT / "exposure_demo_portfolio.csv"),
            "--holdings-file", str(PROJECT / "exposure_demo_holdings.csv"),
            "--assets-file", str(PROJECT / "exposure_demo_assets.csv"),
            "--base", "CAD", "--as-of", "2026-11-07"]


def test_cli_offline_demo(capsys):
    assert main(arguments()) == 0
    out = capsys.readouterr().out
    assert "V7 exposure intelligence" in out
    assert "SYNTHETIC" in out
    assert "overlap" in out.lower()
    assert "36.00%" in out


def test_cli_json_and_coverage(capsys):
    assert main(arguments() + ["--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["unknown_underlying"] > .2
    assert output["coverage_by_fund"]["DEMOA"]["as_of"] == "2026-11-01"
    assert output["pairwise_overlap"][0]["identified_overlap_lower_bound"] == .36 or abs(
        output["pairwise_overlap"][0]["identified_overlap_lower_bound"] - .36) < 1e-12


def test_cli_rejects_future_snapshot(capsys):
    assert main(arguments()[:-1] + ["2026-01-01"]) != 0
    assert "EXPOSURE ERROR" in capsys.readouterr().err


def test_cli_with_no_snapshots_has_explicit_unknown(capsys):
    arg = ["exposure", str(PROJECT / "exposure_demo_portfolio.csv"), "--as-of", "2026-11-07"]
    assert main(arg) == 0
    out = capsys.readouterr().out
    assert "no classification or fund snapshot" in out
    assert "No ETF/fund snapshots identified" in out


def test_cli_rejects_mismatched_yahoo_symbol(capsys):
    args = arguments() + ["--yahoo-top-holdings", "UNKNOWNFUND"]
    assert main(args) == 1
    assert "not a direct holding" in capsys.readouterr().err


def test_cli_rejects_duplicate_manual_yahoo_symbol(capsys):
    args = arguments() + ["--yahoo-top-holdings", "DEMOA"]
    assert main(args) == 1
    assert "already supplied" in capsys.readouterr().err
