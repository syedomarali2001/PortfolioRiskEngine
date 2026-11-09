from portfolio_risk.capability_report import build_capability_report


def test_matrix_uses_independent_dimensions():
    etfs={'summary':{'total':20,'indexed':18,'with_dated_holdings':17,
                     'near_full_weight_95pct':13,'with_sector_breakdown':10,
                     'with_country_breakdown':10,'current_120d':4,
                     'equity_lookthrough_safe':12},
          'assets':[{}], 'by_theme':{'Bonds':{'total':5}}}
    stocks={'summary':{'total':4}, 'assets':[{}],
            'by_market':{'US_STOCK':{'total':2,'sector_available':1,'company_country_available':1},
                         'CAD_STOCK':{'total':2,'sector_available':2,'company_country_available':2}}}
    mixed={'summary':{'by_category':{
       key:{'tickers':5,'price_available':5,'holdings_available':0}
       for key in ('US_ETF','CAD_ETF','US_STOCK','CAD_STOCK','COMMODITY_ETP','CRYPTO')
    }},'assets':[{}]}
    out=build_capability_report(etfs,stocks,mixed)
    assert out['capabilities']['US_ETF']['funds_with_dated_holdings']==17
    assert out['capabilities']['CAD_STOCK']['sampled_prices']['available']==5
    assert out['capabilities']['CRYPTO']['sampled_prices']['available']==5
    assert out['capabilities']['CAD_ETF']['sampled_holdings_available']==0
    assert 'curated' in out['scope_note'].lower()


def test_matrix_rejects_missing_inputs():
    import pytest
    with pytest.raises(ValueError):
        build_capability_report({}, {}, {})
