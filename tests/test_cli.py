from __future__ import annotations

from portfolio_risk.cli import main


def test_validate_command_works_offline(capsys):
    code = main(["validate", "examples/sample_portfolio.csv"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Validated 6 positions" in captured.out
    assert "No API requests" in captured.out


def test_validate_command_fails_bad_file(capsys):
    code = main(["validate", "missing.csv"])
    captured = capsys.readouterr()
    assert code == 2
    assert "INPUT ERROR" in captured.err
