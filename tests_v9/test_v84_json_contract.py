"""Contract-check an actual V8.4 example report; no provider request needed."""
import json
from pathlib import Path
import pytest

from portfolio_risk_ui.core import show_complete_dividend_yield


@pytest.fixture(scope='module')
def report():
    return json.loads((Path(__file__).parent/'fixtures'/'v84_dividends_example.json').read_text())


def test_report_version_and_status(report):
    assert report['version']=='8.4'
    assert report['income_summary']['status']=='evaluated'


def test_income_coverage_and_numbers(report):
    summary=report['income_summary']
    assert show_complete_dividend_yield(summary)
    assert summary['estimated_annual_gross_income_covered_holdings']==pytest.approx(382.49692963786)
    assert summary['estimated_monthly_average_gross_income_covered_holdings']==pytest.approx(382.49692963786/12)


def test_portfolio_income_is_sum_of_holdings(report):
    assert sum(h['estimated_annual_gross_income'] for h in report['holdings'])==pytest.approx(report['income_summary']['estimated_annual_gross_income_covered_holdings'])


def test_simulator_has_correct_reconciliation(report):
    for year in report['simulation']['annual_results']:
        cash=year['cash']
        assert cash['invested_market_value']+cash['cumulative_net_cash_withdrawn']==pytest.approx(cash['total_wealth_including_withdrawn_cash'])
        assert year['drip']['invested_market_value']==pytest.approx(year['drip']['total_wealth'])


def test_year_ten_cash_vs_drip(report):
    end=report['simulation']['end_comparison']
    assert end['year']==10
    assert end['cash']['total_wealth_including_withdrawn_cash']==pytest.approx(41391.005761335146)
    assert end['drip']['total_wealth']==pytest.approx(42804.55826567192)


def test_tax_drag_is_explicit(report):
    assert report['simulation']['assumed_effective_distribution_tax_drag']==0
