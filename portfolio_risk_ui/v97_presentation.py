"""Plain-language display helpers for V9.7. Financial outputs remain those of V8.4.

No inputs are altered or new valuation classifications invented here.
"""
from __future__ import annotations

import math
import re


def is_known_fund(symbol: str, name: str) -> bool:
    """Conservative UI hint, not a security-type determination for the backend."""
    text = f"{symbol} {name}".upper()
    return bool(re.search(r"\b(ETF|EXCHANGE.TRADED FUND|INDEX FUND|ETN|MUTUAL FUND)\b", text)) or symbol.upper() in {
        "SPY", "RSP", "GLD", "SLV", "SCHD", "SPMO", "QQQ", "VOO", "VTI", "XEQT.TO", "VGRO.TO"
    }


MULTIPLE_INFO = {
    "price_to_earnings": (
        "Price to earnings (P/E)",
        "How much investors pay for each $1 of yearly company earnings. For example, 25× means $25 of stock-market value for $1 of annual earnings."
        " High values can reflect growth expectations; low values can reflect risks or slower growth. Compare with similar companies, not a universal cutoff."
    ),
    "price_to_fcf": (
        "Price to free cash flow (P/FCF)",
        "How much investors pay for each $1 of annual cash remaining after the business pays for its operations and capital investment."
        " A lower multiple can be attractive, but business quality, investment cycles, and peers matter."
    ),
    "ev_to_ebitda": (
        "Enterprise value to EBITDA (EV/EBITDA)",
        "Compares the value of the entire business, including net debt, with earnings before interest, taxes, depreciation and amortization."
        " Useful when comparing similar companies, but it leaves out some important costs and is not a standalone buy signal."
    ),
}


def multiple_reading(key: str, value: object) -> str:
    """Describe meaning without presenting unsupported cheap/expensive cutoffs."""
    label, _ = MULTIPLE_INFO[key]
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return f"{label} is unavailable from the verified financial data."
    if key == "price_to_earnings":
        return f"At {value:.2f}×, investors pay about {value:.2f} units of market value for each 1 unit of annual net income."
    if key == "price_to_fcf":
        return f"At {value:.2f}×, the market values the company at about {value:.2f} times its annual free cash flow."
    return f"At {value:.2f}×, the full business is valued at about {value:.2f} times annual EBITDA."


def valuation_scenarios(report: dict) -> list[dict]:
    """Use only backend-computed per-share values, no invented estimates."""
    items = (report.get("dcf") or {}).get("scenarios") or {}
    results = []
    display_names = {"conservative": "Cautious assumptions", "base": "Middle assumptions", "optimistic": "Optimistic assumptions"}
    for key in ("conservative", "base", "optimistic"):
        case = items.get(key)
        if not isinstance(case, dict):
            continue
        value = case.get("intrinsic_value_per_share")
        if isinstance(value, (int, float)) and math.isfinite(value):
            results.append({"Scenario": display_names[key], "Estimated value per share": float(value), "scenario_key": key})
    # Preserve other provider/engine supported scenarios without guessing.
    for key, case in items.items():
        if key in display_names or not isinstance(case, dict):
            continue
        value = case.get("intrinsic_value_per_share")
        if isinstance(value, (int, float)) and math.isfinite(value):
            results.append({"Scenario": key.replace("_", " ").title(), "Estimated value per share": float(value), "scenario_key": key})
    return results


def simulation_axis_range(cash_wealth: list[float], drip_wealth: list[float]) -> tuple[float, float] | None:
    """Readable nonzero baseline for close series; callers must disclose the zoom."""
    values = [float(v) for v in cash_wealth + drip_wealth if isinstance(v,(int,float)) and math.isfinite(v)]
    if not values:
        return None
    low, high = min(values), max(values)
    spread = high - low
    pad = max(spread * 0.12, abs(high) * 0.002, 1.)
    return low - pad, high + pad



def close_simulation_lines(cash_wealth: list[float], drip_wealth: list[float], threshold: float = 0.10) -> bool:
    """Check whether the two cumulative-wealth scenarios visually overlap.

    This only determines chart presentation. It does not change the model.
    """
    if len(cash_wealth) != len(drip_wealth) or not cash_wealth:
        return False
    pairs = [(float(a), float(b)) for a,b in zip(cash_wealth, drip_wealth)
             if isinstance(a,(int,float)) and isinstance(b,(int,float))
             and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) != len(cash_wealth):
        return False
    return max(abs(b-a)/max(abs(a),abs(b),1.) for a,b in pairs) < threshold


def holdout_summary_rows(tally: dict) -> list[dict]:
    """Display metric-specific validation counts, never replace status with invented judgment."""
    names = {"sharpe": "Return relative to total risk (Sharpe)", "volatility": "Portfolio volatility",
             "drawdown": "Largest historical decline", "cagr": "Annual growth",
             "sortino": "Return relative to downside risk (Sortino)"}
    result = []
    for key, counts in (tally or {}).items():
        if not isinstance(counts, dict):
            continue
        result.append({"Priority": names.get(str(key), str(key).replace("_", " ").title()),
                       "Better": int(counts.get("positive") or 0),
                       "Similar": int(counts.get("neutral") or 0),
                       "Worse": int(counts.get("negative") or 0),
                       "Not assessed": int(counts.get("unavailable") or 0)})
    return result
