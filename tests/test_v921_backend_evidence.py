"""Backend integration: independent evidence ratings without optimizer changes."""
import numpy as np
import pandas as pd

from portfolio_risk.v918_three_strategies import analyze_three_strategies
from portfolio_risk_ui.v921_evidence import status_label, benchmark_status_label
import portfolio_risk.v918_three_strategies as engine


def test_reliable_original_improvement_is_not_vetoed_by_market_benchmarks(monkeypatch):
    rng = np.random.default_rng(921)
    days = pd.bdate_range('2024-01-02', periods=280)
    frame = pd.DataFrame(rng.normal(.0004, .007, (280,2)), index=days, columns=['A','B'])
    dates = pd.DatetimeIndex([days[0]-pd.Timedelta(days=1),*days])
    reference = np.array([.85,.15])
    benchmarks = {'SPY':pd.Series(.0004,index=days), 'RSP':pd.Series(.0003,index=days)}
    monkeypatch.setattr(engine,'_select',lambda *_args,**_kwargs: (np.array([.5,.5]),{'method':'fake test-only selector'}))
    positive=[.03,.05,-.01,.02]
    def fake_fold(_frame,_dates,_ref,_bench,_rules,_rf,_goal,end):
        position = {156:0,186:1,216:2,246:3}[end]
        return {'primary_better':positive[position]>0,
                'primary_improvement':positive[position],
                'beats_benchmarks_on_goal':{'SPY':position < 2,'RSP':position != 2}}
    monkeypatch.setattr(engine,'_fold_metrics',fake_fold)
    result = analyze_three_strategies(frame,dates,reference,benchmarks,rules={
        'train_periods':126, 'test_periods':30, 'step_periods':30,
        'rebalance':'quarterly','max_weight':1.0,
    },max_folds=4)
    for goal in ('growth','balanced','defensive'):
        item=result['strategies'][goal]
        h=item['holdout']
        assert h['better'] == 3 and h['tests']==4
        assert h['baseline_consistent'] is True
        assert h['benchmarks_consistent'] is False
        assert h['portfolio_evidence']=='consistent'
        assert h['benchmark_evidence']=='consistent_one'
        assert h['status']=='mixed'  # compatibility with old combined code
        assert status_label(item)[0]=='★'
        assert benchmark_status_label(item)=='Consistently ahead of one benchmark'
