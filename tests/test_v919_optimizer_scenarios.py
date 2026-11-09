"""V9.19: offline illustrative portfolio stress tests (NOT market-data claims)."""

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.v918_three_strategies import analyze_three_strategies


def scenarios(n=882):
    idx = pd.bdate_range("2021-01-04", periods=n)
    dates = idx.insert(0, idx[0] - pd.Timedelta(days=1))

    def assemble(name, assets, ref, bench, extra=""):
        f = pd.DataFrame(assets, index=idx)
        b = {sym: pd.Series(a, index=idx) for sym, a in bench.items()}
        return name, f, dates, np.array(ref, float), b

    rng = np.random.default_rng(91021)
    def noise(scale):
        return rng.normal(0, scale, n)
    # Deliberately weak concentrated holding; strong, calm equity competitor and modest broad market.
    common = noise(0.004)
    yield assemble(
        "Weak 90% CCL-like single-stock concentration",
        {
            "CCL_like": -0.0015 + noise(0.021) + 0.25 * common,
            "AAPL_like": 0.0019 + noise(0.003) + 0.2 * common,
            "MSFT_like": 0.0008 + noise(0.009) + 0.4 * common,
            "SPY_like": 0.00025 + noise(0.009) + common,
        },
        [0.9, 0.033, 0.033, 0.034],
        {"SPY": 0.00025 + noise(0.009) + common, "RSP": 0.00020 + noise(0.009) + common},
    )
    # Concentrated in fast-growing, high-beta; a diversified mix may have better Sharpe and lower risk, but growth unclear.
    market = noise(0.009)
    yield assemble(
        "90% fast-growing but volatile NVDA-like holding",
        {
            "NVDA_like": 0.0016 + 1.9 * market + noise(0.015),
            "AAPL_like": 0.0010 + 1.0 * market + noise(0.007),
            "MSFT_like": 0.00085 + 0.95 * market + noise(0.006),
            "GLD_like": 0.00045 - 0.3 * market + noise(0.006),
        },
        [0.9, 0.04, 0.03, 0.03],
        {"SPY": 0.0007 + market + noise(0.002), "RSP": 0.00055 + 0.95 * market + noise(0.004)},
    )
    # Already strongly allocated, efficient portfolio; should not fabricate improvement.
    equity = noise(0.008)
    defensive = noise(0.004)
    yield assemble(
        "Already efficient multi-asset portfolio",
        {
            "AAPL_like": 0.0012 + equity + noise(0.003),
            "MSFT_like": 0.00105 + 0.95 * equity + noise(0.003),
            "SPY_like": 0.0007 + 0.9 * equity + noise(0.002),
            "GLD_like": 0.0004 - 0.2 * equity + defensive,
        },
        [0.35, 0.25, 0.25, 0.15],
        {"SPY": 0.0007 + 0.9 * equity + noise(0.003), "RSP": 0.0006 + 0.9 * equity + noise(0.003)},
    )
    # Strong momentum-ETF-like holding is included explicitly in the user universe.
    equity = noise(0.007)
    yield assemble(
        "Momentum ETF candidate versus weak concentrated stock",
        {
            "CCL_like": -0.0010 + noise(0.017) + 0.5 * equity,
            "AAPL_like": 0.0006 + noise(0.008) + equity,
            "GLD_like": 0.0002 + noise(0.006) - 0.1 * equity,
            "SPMO_like": 0.0016 + noise(0.003) + 0.2 * equity,
        },
        [0.9, 0.04, 0.03, 0.03],
        {"SPY": 0.00045 + equity, "RSP": 0.00042 + noise(0.006) + 0.6 * equity},
    )
    # Optimizer tempting based on initially strong returns, flips after halfway. Don't falsify validation.
    market = noise(0.007)
    early = np.arange(n) < n // 2
    shifts = np.where(early, 0.0016, -0.0010)
    yield assemble(
        "Regime reversal / historical winner stops winning",
        {
            "Momentum_like": shifts + noise(0.010) + market,
            "Stable_like": 0.00055 + noise(0.004) + 0.2 * market,
            "Index_like": 0.0006 + market,
            "Gold_like": 0.0003 + noise(0.006) - 0.15 * market,
        },
        [0.8, 0.05, 0.10, 0.05],
        {"SPY": 0.0006 + market, "RSP": 0.0005 + noise(0.007) + market * 0.5},
    )


def test_default_ui_has_quarterly_rebalancing_and_unconstrained_weight_search():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "portfolio_risk_ui" / "app.py").read_text()
    assert "from .v920_presentation import REBALANCE_CHOICES" in text
    assert 'key="v920_cadence"' in text
    assert "max_cap = 100.0" in text
    assert 'key="v919_use_cap"' in text
    assert 'value=False, key="v919_use_cap"' in text
    assert "default_max" not in text[text.index("def page_strategy():") : text.index("def main():")]


def test_five_distinct_scenarios_prove_when_stars_do_and_do_not_appear():
    cases = {}
    for name, frame, dates, original, benchmark in scenarios():
        result = analyze_three_strategies(
            frame,
            dates,
            original,
            benchmark,
            rules={
                "max_weight": 1.0,
                "rebalance": "buy_hold",
                "cost_bps": 10.0,
                "train_periods": 252,
                "test_periods": 126,
                "step_periods": 126,
            },
        )
        cases[name] = result
        for goal in ("growth", "balanced", "defensive"):
            item = result["strategies"][goal]
            assert item["status"] == "evaluated"
            assert sum(item["weights"].values()) == pytest.approx(1)
            assert item["holdout"]["tests"] >= 3
            assert set(item["benchmarks_training_metrics"]) == {"SPY", "RSP"}
            if item["holdout"]["status"] == "historically_consistent":
                h = item["holdout"]
                assert h["better"] >= np.ceil(0.75 * h["tests"])
                assert h["mean_primary_improvement"] > 0
                assert all(v >= np.ceil(0.75 * h["tests"]) for v in h["benchmark_counts"].values())

    ccl = cases["Weak 90% CCL-like single-stock concentration"]
    winning = ccl["strategies"]["growth"]
    assert winning["weights"]["AAPL_like"] > 0.99
    assert winning["holdout"]["status"] == "historically_consistent"
    assert winning["holdout"]["better"] == 4 and winning["holdout"]["worse"] == 1

    nvda = cases["90% fast-growing but volatile NVDA-like holding"]
    assert nvda["strategies"]["growth"]["holdout"]["status"] != "historically_consistent"
    assert nvda["strategies"]["defensive"]["holdout"]["status"] == "historically_consistent"

    efficient = cases["Already efficient multi-asset portfolio"]
    assert efficient["strategies"]["balanced"]["holdout"]["status"] != "historically_consistent"

    momentum = cases["Momentum ETF candidate versus weak concentrated stock"]
    assert momentum["strategies"]["growth"]["weights"]["SPMO_like"] > 0.99
    assert momentum["strategies"]["growth"]["holdout"]["status"] == "historically_consistent"

    reversal = cases["Regime reversal / historical winner stops winning"]
    assert reversal["strategies"]["growth"]["holdout"]["status"] != "historically_consistent"


def test_unconstrained_growth_training_optimum_is_not_worse_than_40_percent_cap():
    from portfolio_risk.v83_robustness import _validated_rules
    from portfolio_risk.v918_three_strategies import _blocks, _fast_growth, _select

    cases = list(scenarios())
    name, frame, dates, orig, benchmark = cases[3]
    rbase = {
        "rebalance": "buy_hold",
        "cost_bps": 10.0,
        "train_periods": 252,
        "test_periods": 126,
        "step_periods": 126,
    }
    cap_rules = _validated_rules(list(frame), orig, {"max_weight": 0.4, **rbase})
    full_rules = _validated_rules(list(frame), orig, {"max_weight": 1.0, **rbase})
    capped, _ = _select(frame, orig, cap_rules, 0.0, "growth")
    full, _ = _select(frame, orig, full_rules, 0.0, "growth")
    assert max(capped) <= 0.400001
    assert max(full) > 0.99
    blocks = _blocks(frame, "buy_hold")
    capped_cagr = _fast_growth(blocks, capped, orig, 10.0, len(frame))
    full_cagr = _fast_growth(blocks, full, orig, 10.0, len(frame))
    assert full_cagr >= capped_cagr - 1e-6
