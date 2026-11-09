"""Educational standalone calculators. They do not require a portfolio or save trades."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from .v10_calculators import (future_value, monthly_contribution_for_goal,
                              calculator_press, initial_calculator_state)
from .v10_design import tune_plotly


def _money(value: float, currency: str) -> str:
    return f"${float(value):,.2f} {currency}"


def _intro(title: str, description: str):
    st.title(title)
    st.caption(description)
    st.caption("Educational estimates only. Returns are hypothetical, not investment forecasts.")


def _ccy() -> str:
    return st.session_state.get("base", "USD")


def _year_chart(points: list[dict], currency: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["Year"] for p in points],
                             y=[p["Value"] for p in points],
                             mode="lines+markers", name="Projected value",
                             line=dict(width=3), marker=dict(size=6)))
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=18, b=12),
                      xaxis_title="Years", yaxis_title=f"Value ({currency})",
                      showlegend=False)
    for trace in fig.data:
        trace.hovertemplate = f"Year %{{x}}: $%{{y:,.2f}} {currency}<extra></extra>"
    tune_plotly(fig, st.session_state.get("v10_theme"))
    st.plotly_chart(fig, width="stretch", theme=None)


def page_compound():
    _intro("Compound Interest Calculator", "See how an initial investment and monthly deposits can grow over time.")
    c1, c2 = st.columns(2)
    with c1:
        initial = st.number_input(f"Initial investment ($, {_ccy()})", 0.0, 1e10, 10000.0, 500.0, key="calc_ci_initial")
        monthly = st.number_input(f"Monthly contribution ($, {_ccy()})", 0.0, 1e8, 250.0, 25.0, key="calc_ci_monthly")
        years = st.number_input("Years to invest", 1, 100, 20, key="calc_ci_years")
    with c2:
        rate = st.number_input("Estimated annual interest (%)", -25.0, 40.0, 6.0, 0.5, key="calc_ci_rate")
        variance = st.number_input("Optional rate variation (percentage points)", 0.0, 15.0, 2.0, 0.5, key="calc_ci_variance")
        frequency = st.selectbox("Compound frequency", ["Monthly", "Annually", "Quarterly", "Daily"], key="calc_ci_freq")
    freq = {"Annually": 1, "Quarterly": 4, "Monthly": 12, "Daily": 365}[frequency]
    if st.button("Calculate growth", type="primary", key="calc_ci_run", width="stretch"):
        try:
            final = future_value(initial, monthly, rate, years, freq)
            invested = initial + monthly * 12 * years
            conservative = future_value(initial, monthly, rate-variance, years, freq)
            optimistic = future_value(initial, monthly, rate+variance, years, freq)
        except (ValueError, OverflowError) as exc:
            st.error(str(exc)); return
        st.session_state.calc_ci_results = (final, invested, conservative, optimistic,
                                            [(year, future_value(initial, monthly, rate, year, freq))
                                             for year in range(years + 1)], _ccy())
    result = st.session_state.get("calc_ci_results")
    if result:
        final, invested, conservative, optimistic, points, currency = result
        a,b,c = st.columns(3)
        a.metric("Projected value", _money(final, currency))
        b.metric("Total contributed", _money(invested, currency))
        c.metric("Investment growth", _money(final-invested, currency))
        _year_chart([{"Year": year, "Value": value} for year, value in points], currency)
        st.caption(f"Lower-rate case: {_money(conservative, currency)}. Higher-rate case: {_money(optimistic, currency)}. Both use the same deposit schedule.")
        st.caption("Assumes end-of-month deposits, a constant nominal rate and no taxes or fees. Results are not guaranteed.")


def page_savings():
    _intro("Savings Goal Calculator", "Estimate the monthly deposit needed to reach a chosen target.")
    a,b = st.columns(2)
    with a:
        goal = st.number_input(f"Savings target ($, {_ccy()})", 1.0, 1e10, 100000.0, 1000.0, key="calc_s_goal")
        initial = st.number_input(f"Already saved ($, {_ccy()})", 0.0, 1e10, 10000.0, 500.0, key="calc_s_initial")
    with b:
        years = st.number_input("Years until your goal", 1, 100, 10, key="calc_s_years")
        apr = st.number_input("Estimated annual return (%)", -25.0, 40.0, 5.0, 0.5, key="calc_s_rate")
        freq = st.selectbox("Compound frequency", ["Monthly", "Annually", "Quarterly", "Daily"], key="calc_s_freq")
    n = {"Annually":1,"Quarterly":4,"Monthly":12,"Daily":365}[freq]
    if st.button("Calculate monthly savings", type="primary", key="calc_s_run", width="stretch"):
        try:
            payment = monthly_contribution_for_goal(goal, initial, apr, years, n)
            reached = future_value(initial, payment, apr, years, n)
        except (ValueError, OverflowError) as exc:
            st.error(str(exc)); return
        st.session_state.calc_s_results = (payment, reached, _ccy())
    if "calc_s_results" in st.session_state:
        monthly, reached, currency = st.session_state.calc_s_results
        c1,c2 = st.columns(2)
        c1.metric("Monthly deposit needed", _money(monthly, currency))
        c2.metric("Estimated balance at your goal", _money(reached, currency))
        if monthly <= 0:
            st.success("Your existing savings may already be enough under the selected growth assumptions.")
        st.caption("Monthly deposits are assumed at month-end. Adjust your goal, time horizon or return assumptions and recalculate.")



def page_basic():
    """A familiar four-function keypad, independent of the investment engine."""
    st.title("Calculator")
    st.caption("A simple calculator for quick arithmetic. No investment analysis is run.")
    if "v102_basic_calculator" not in st.session_state:
        st.session_state.v102_basic_calculator = initial_calculator_state()
    state = st.session_state.v102_basic_calculator
    # Use Streamlit's built-in readable metric surface instead of an input with
    # edit restrictions, so display contrast follows the active theme.
    st.metric("Result", state["display"])
    if state["operator"]:
        st.caption(f"Pending operation: {state['operator']}")
    rows = [
        ("C", "⌫", "%", "÷"),
        ("7", "8", "9", "×"),
        ("4", "5", "6", "−"),
        ("1", "2", "3", "+"),
        ("±", "0", ".", "="),
    ]
    for row_index, row in enumerate(rows):
        cols = st.columns(4, gap="small")
        for index, symbol in enumerate(row):
            with cols[index]:
                if st.button(symbol, key=f"calc_basic_{row_index}_{index}",
                             type="primary" if symbol == "=" else "secondary",
                             width="stretch"):
                    st.session_state.v102_basic_calculator = calculator_press(state, symbol)
                    st.rerun()
    st.caption("Use C to clear, ⌫ to erase, and = to calculate. Division by zero is not allowed.")
