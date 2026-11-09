from __future__ import annotations

import json

from portfolio_risk.cli import main


def test_manual_fx_analyze_works_without_internet(capsys):
    code = main(["analyze", "examples/sample_portfolio.csv", "--base", "CAD", "--usd-cad", "1.35"])
    out = capsys.readouterr()
    assert code == 0
    assert "24,200.00 CAD" in out.out
    assert "Top three:" in out.out
    assert "USD/CAD: 1.35" in out.out
    assert "V2 is accounting" in out.out


def test_json_output_is_pure_json(capsys):
    code = main(["analyze", "examples/sample_portfolio.csv", "--usd-cad", "1.35", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    parsed = json.loads(captured.out)
    assert parsed["total_value"] == "24200.00"
    assert len(parsed["holdings"]) == 6
    assert parsed["base_currency"] == "CAD"


def test_invalid_rate_exits_at_argparse(capsys):
    import pytest
    with pytest.raises(SystemExit, match="2"):
        main(["analyze", "examples/sample_portfolio.csv", "--usd-cad", "0"])


def test_future_fx_cutoff_is_rejected(capsys):
    code = main(["analyze", "examples/sample_portfolio.csv", "--usd-cad", "1.4",
                 "--as-of", "2099-01-01"])
    assert code == 2
    assert "future" in capsys.readouterr().err


def test_no_fx_needed_if_all_holdings_in_base(tmp_path, capsys):
    input_file = tmp_path / "same.csv"
    input_file.write_text("symbol,market_value,currency\nQQQ,10,USD\nSCHD,30,USD\n")
    code = main(["analyze", str(input_file), "--base", "USD"])
    assert code == 0
    output = capsys.readouterr().out
    assert "40.00 USD" in output
    assert "FX conversion not required" in output
