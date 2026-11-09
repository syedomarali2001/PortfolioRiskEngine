"""V9.15 status must communicate historical evidence, not guarantee trading returns."""
from pathlib import Path
from portfolio_risk_ui.v913_strategy_paths import path_signal
from portfolio_risk_ui.v911_presentation import candidate_label

APP=(Path(__file__).resolve().parents[1]/'portfolio_risk_ui'/'app.py').read_text()


def summary(better=0, worse=0, *, supported=False, own_passed=False, available=True):
    return {'supported':supported,'own_passed':own_passed,
            'primary_row':{'Better':better,'Worse':worse},
            'review':{'status':'evaluated' if available else 'unavailable'},
            'example':{'status':'evaluated' if available else 'unavailable'}}


def test_star_requires_full_validation_not_just_some_improvement():
    assert path_signal(summary(better=18,worse=2))[0]=='○'
    assert path_signal(summary(better=2,worse=18))[0]=='✕'
    assert path_signal(summary(better=20,worse=0,own_passed=True))[0]=='○'
    assert path_signal(summary(better=20,worse=0,own_passed=True,supported=True))[0]=='★'
    assert path_signal(summary(available=False))[0]=='○'


def test_three_strategy_tabs_show_evidence_and_method_labels():
    from pathlib import Path
    view=(Path(__file__).resolve().parents[1]/'portfolio_risk_ui'/'v918_three_views.py').read_text(encoding='utf-8')
    assert 'Consistent past improvements' in view
    assert 'Mixed historical evidence' in view
    assert 'Not supported by past tests' in view
    assert "('growth', 'Growth'" in view
    assert "('balanced', 'Balanced'" in view
    assert "('defensive', 'Defensive'" in view
    assert 'continuous search' in candidate_label('continuous_sharpe').lower()

