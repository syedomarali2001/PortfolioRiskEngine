"""Independent educational calculator math. Does not call or modify the portfolio engine.

Rates are nominal APRs expressed as percentages. Monthly deposits are made
at the end of each month. Interest is credited according to the selected
compounding frequency and converted to an effective monthly rate.
"""
from __future__ import annotations

from math import isfinite


def _valid_nonnegative(value: float, label: str) -> float:
    x = float(value)
    if not isfinite(x) or x < 0:
        raise ValueError(f"{label} must be a nonnegative finite number")
    return x


def effective_monthly_rate(apr_pct: float, compounds_per_year: int = 12) -> float:
    """Convert nominal annual interest to an equivalent monthly effective rate."""
    n = int(compounds_per_year)
    if n not in (1, 4, 12, 365):
        raise ValueError("Choose annual, quarterly, monthly or daily compounding")
    apr = float(apr_pct) / 100.0
    if not isfinite(apr) or apr <= -100.0 * n / 100.0:
        raise ValueError("The rate is not valid for this compounding schedule")
    return (1.0 + apr / n) ** (n / 12.0) - 1.0


def future_value(principal: float, monthly_contribution: float, apr_pct: float,
                 years: int, compounds_per_year: int = 12) -> float:
    principal = _valid_nonnegative(principal, "Initial investment")
    contribution = float(monthly_contribution)
    if not isfinite(contribution):
        raise ValueError("Monthly contribution must be finite")
    months = 12 * int(years)
    if years < 0 or months > 1200:
        raise ValueError("Years must be between 0 and 100")
    r = effective_monthly_rate(apr_pct, compounds_per_year)
    growth = (1.0 + r) ** months
    series = months if abs(r) < 1e-12 else (growth - 1.0) / r
    return principal * growth + contribution * series


def monthly_contribution_for_goal(target: float, principal: float, apr_pct: float,
                                  years: int, compounds_per_year: int = 12) -> float:
    target = _valid_nonnegative(target, "Savings goal")
    principal = _valid_nonnegative(principal, "Initial savings")
    months = int(years) * 12
    if years <= 0 or months > 1200:
        raise ValueError("Savings horizon must be between 1 and 100 years")
    r = effective_monthly_rate(apr_pct, compounds_per_year)
    growth = (1.0 + r) ** months
    series = months if abs(r) < 1e-12 else (growth - 1.0) / r
    return max(0.0, (target - principal * growth) / series)


def college_future_cost(annual_cost_today: float, years_until: int,
                        years_of_study: int, annual_inflation_pct: float) -> float:
    """Sum future annual costs in nominal dollars, referenced to enrollment year.

    This budget is deliberately conservative: no investment earnings during
    study, no aid, and no additional savings once studies start.
    """
    annual = _valid_nonnegative(annual_cost_today, "Annual college cost")
    if not 0 <= years_until <= 80 or not 1 <= years_of_study <= 10:
        raise ValueError("Choose a valid horizon and 1 to 10 study years")
    inflation = float(annual_inflation_pct) / 100.0
    if not isfinite(inflation) or inflation <= -1:
        raise ValueError("Annual inflation rate must exceed -100%")
    return sum(annual * (1 + inflation) ** (years_until + offset)
               for offset in range(int(years_of_study)))


# IRS Publication 590-B (2025), Appendix B, Table III: Uniform Lifetime.
# Standard owner case ONLY. Source: https://www.irs.gov/publications/p590b
_IRS_FACTORS = [
    26.5, 25.5, 24.6, 23.7, 22.9, 22.0, 21.1, 20.2, 19.4, 18.5,
    17.7, 16.8, 16.0, 15.2, 14.4, 13.7, 12.9, 12.2, 11.5, 10.8,
    10.1, 9.5, 8.9, 8.4, 7.8, 7.3, 6.8, 6.4, 6.0, 5.6,
    5.2, 4.9, 4.6, 4.3, 4.1, 3.9, 3.7, 3.5, 3.4, 3.3,
    3.1, 3.0, 2.9, 2.8, 2.7, 2.5, 2.3, 2.0,
]
IRS_UNIFORM_LIFETIME = {age: factor for age, factor in enumerate(_IRS_FACTORS, start=73)}


def us_rmd_standard(account_balance_previous_dec31: float, age_in_year: int) -> tuple[float, float]:
    """Return (RMD, divisor), if standard owner Uniform Lifetime Table applies.

    This intentionally does not determine legal eligibility, including changing
    required beginning ages, inherited plans or a younger sole spouse beneficiary.
    """
    balance = _valid_nonnegative(account_balance_previous_dec31, "Prior Dec 31 balance")
    age = int(age_in_year)
    if age not in IRS_UNIFORM_LIFETIME:
        raise ValueError("The standard example supports ages 73 to 120")
    factor = IRS_UNIFORM_LIFETIME[age]
    return balance / factor, factor


# Four-function calculator. No dynamic code evaluation is used.
from decimal import Decimal, InvalidOperation

def initial_calculator_state() -> dict[str, object]:
    return {"display": "0", "stored": None, "operator": None, "fresh": False}

def _number_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ArithmeticError("Non-finite calculation")
    if abs(value) > Decimal('1e30'):
        raise ArithmeticError("Result too large")
    return format(value.normalize(), 'f') if value != 0 else "0"

def calculator_press(current: dict, key: str) -> dict[str, object]:
    """Pure calculator state transition. Handles numbers as Decimal, no eval()."""
    state = dict(current)
    if key == "C":
        return initial_calculator_state()
    display = str(state["display"])
    if display == "Error":
        state = initial_calculator_state()
        display = "0"
    if key.isdigit() and len(key) == 1:
        if state["fresh"] or display == "0":
            display = key
        elif len(display.replace("-", "").replace(".", "")) < 18:
            display += key
        state.update(display=display, fresh=False)
    elif key == ".":
        if state["fresh"]:
            display = "0"
        if "." not in display:
            display += "."
        state.update(display=display, fresh=False)
    elif key == "⌫":
        state.update(display=(display[:-1] if len(display)>1 else "0"), fresh=False)
        if state["display"] in ("", "-"):
            state["display"] = "0"
    elif key == "±":
        if Decimal(display) != 0:
            state["display"] = display[1:] if display.startswith("-") else "-"+display
    elif key == "%":
        state["display"] = _number_text(Decimal(display) / Decimal(100))
    elif key in ("+", "−", "×", "÷"):
        if state["operator"] is not None and not state["fresh"]:
            state = calculator_press(state, "=")
            if state["display"] == "Error":
                return state
        state.update(stored=str(state["display"]), operator=key, fresh=True)
    elif key == "=":
        if not state["operator"] or state["stored"] is None:
            return state
        try:
            a, b = Decimal(str(state["stored"])), Decimal(display)
            result = {"+": lambda: a+b, "−": lambda: a-b,
                      "×": lambda: a*b, "÷": lambda: a/b}[str(state["operator"])]()
            state.update(display=_number_text(result), stored=None, operator=None, fresh=True)
        except (ZeroDivisionError, InvalidOperation, ArithmeticError, OverflowError):
            state.update(display="Error", stored=None, operator=None, fresh=True)
    else:
        raise ValueError("Unsupported calculator button")
    return state
