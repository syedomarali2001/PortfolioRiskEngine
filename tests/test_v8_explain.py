from datetime import date
from decimal import Decimal

import pytest

from portfolio_risk.accounting import value_portfolio
from portfolio_risk.models import PortfolioInput, Holding
from portfolio_risk.exposure import (SecurityMetadata, FundSnapshot, Constituent, analyze_exposure)
from portfolio_risk.v8_explain import explain_exposure


def test_drilldown_sectors_top_stocks_overlap_and_missing_data():
    p=PortfolioInput(tuple(Holding(s,Decimal(str(v)),'USD','USD') for s,v in
                           [('SPY',40),('QQQ',30),('AAPL',10),('XEQT.TO',20)]),'x.csv')
    snaps={
        'SPY':FundSnapshot('SPY',(Constituent('AAPL',.6,'equity'),
                                  Constituent('MSFT',.4,'equity')), 'fixture SPY',date(2026,10,31)),
        'QQQ':FundSnapshot('QQQ',(Constituent('AAPL',.5,'equity'),
                                  Constituent('MSFT',.5,'equity')),'fixture QQQ',date(2026,10,31)),
    }
    classes={'AAPL':SecurityMetadata('AAPL','equity','Technology','US'),
             'XEQT.TO':SecurityMetadata('XEQT.TO','etf')}
    agg={'SPY':({'technology':60,'health_care':40},{'US':100}),
         'QQQ':({'technology':50,'health_care':50},{'US':100})}
    valuation=value_portfolio(p,base_currency='USD')
    report=analyze_exposure(valuation,snapshots=snaps,classifications=classes,
                            fund_aggregates=agg,analysis_date=date(2026,11,7))
    x=explain_exposure(report,snapshots=snaps,classifications=classes,fund_aggregates=agg)
    assert x['unknown_holdings_by_fund']['XEQT.TO']['unknown_portfolio_fraction']==pytest.approx(.2)
    assert x['fund_top_holdings']['SPY']['top_holdings'][0]['portfolio_fraction']==pytest.approx(.24)
    assert x['sector_drilldown']['technology']['portfolio_fraction']==pytest.approx(.4*.6+.3*.5+.1)
    assert x['sector_drilldown']['technology']['known_company_contributions']['AAPL']==pytest.approx(.4*.6+.3*.5+.1)
    assert x['overlap_contributors'][0]['overlap_lower_bound']==pytest.approx(.9)
    assert x['overlap_contributors'][0]['largest_contributors'][0]['symbol'] in ('AAPL','MSFT')
    assert x['fund_top_holdings']['QQQ']['classified_constituent_fraction']==pytest.approx(.5)
    assert x['sector_drilldown']['health_care']['known_company_total_fraction']==0
    assert x['sector_drilldown']['health_care']['unattributed_company_fraction']==pytest.approx(.4*.4+.3*.5)


def test_invalid_drilldown_top_limit():
    p=PortfolioInput((Holding('AAPL',Decimal('100'),'USD'),),'x')
    report=analyze_exposure(value_portfolio(p,base_currency='USD'))
    with pytest.raises(ValueError):explain_exposure(report,snapshots={},top_n=0)
