"""Seven-page, evidence-first Streamlit UI for the existing V8.4 engine.

This code never computes financial risk metrics. Backend JSON is the contract.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .v10_design import THEMES, theme_css, tune_plotly, display_table
from .backend import BackendError, run_fundamentals, run_report
from .v10_branding import GITHUB_URL
from .v10_calculator_pages import (page_compound, page_savings, page_basic)
from .core import (MAX_HOLDINGS, PortfolioError, SUPPORTED_CURRENCIES,
                   engine_rows, exact_money, fingerprint, import_values_csv,
                   normalized_symbol, pct, safe_portfolio_nominal_total, update_holding_amount,
                   show_complete_dividend_yield)
from .search import lookup
from .example_portfolios import EXAMPLES, get_example, load_example
from .quotes import verified_currency
from .fx_display import display_portfolio_value, get_fx_quote
from .v95_presentation import sort_sectors_by_exposure, coverage_summary
from .v97_presentation import (is_known_fund, MULTIPLE_INFO, multiple_reading,
                               valuation_scenarios, simulation_axis_range, close_simulation_lines, holdout_summary_rows)
from .v98_presentation import (STRATEGY_GOALS, market_valuation_gap, scenario_allocation_check,
                               format_group_cap, holdout_chart_rows, holdout_takeaway)
from .v99_presentation import (goal_from_priorities, goal_metric_labels,
                               metric_decision, validation_summary)
from .v910_presentation import all_goal_review, tested_candidate_warning
from .v911_presentation import latest_candidate_example, status_group, format_metric_value
from .v912_presentation import (income_ranked, strategy_tradeoff, status_for_summary,
                                 strategy_goal_name, strategy_option_label)
from .v913_strategy_paths import PATHS, strategy_report_args, path_summary
from .v96_presentation import historical_window_text, HISTORY_NOTE, composition_data_with_unknown
from .periods import requested_window, actual_window_label
from .scenarios import prepare_what_if, parse_group_limits
from .presentation import (inspect_risk_contribution, read_exposure_breakdowns,
                           holding_support_rows, overlap_rows, what_if_flag, finding_display,
                           allocation_pie_rows, overview_notes)
from .v94_presentation import (
    RISK_METRIC_HELP, benchmark_status, composition_rows, exposure_notes,
    finding_for_people, select_overview_findings, simplify_insights_warnings,
    snapshot_rows, support_for_people,
)


def display_money(value: object, currency: str = "USD") -> str:
    """Render a clear currency symbol without changing the numeric engine values."""
    if value is None:
        return "Unavailable"
    return f"${float(value):,.2f} {currency}"


def _history_dates() -> tuple[dt.date, dt.date]:
    return requested_window(st.session_state.get("history_auto", True),
                            st.session_state.search_date_start,
                            st.session_state.search_date_end, dt.date.today())


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_quote_currency(symbol: str):
    return verified_currency(symbol)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_display_fx(as_of: dt.date):
    """Only for UI display: use the dated V8.4 FX source, not a guessed rate."""
    try:
        quote = get_fx_quote(as_of)
        return str(quote.usd_cad), quote.observed_on.isoformat(), quote.source
    except Exception:
        return None


def _resolve_auto_quotes() -> list[str]:
    """Persist only issuer-verified or explicit provider metadata; never assume FX."""
    unresolved = []
    changed = False
    quote_sources = st.session_state.setdefault("quote_sources", {})
    for holding in st.session_state.holdings:
        if holding.get("quote_currency") in ("USD", "CAD"):
            continue
        evidence = _cached_quote_currency(holding["symbol"])
        if evidence:
            holding["quote_currency"] = evidence.currency
            quote_sources[holding["symbol"]] = evidence.source
            changed = True
        else:
            unresolved.append(holding["symbol"])
    if changed:
        # Re-key the holding editor after an Auto value becomes verified; old
        # selectbox widget state must not misleadingly keep showing 'Auto'.
        st.session_state.editor_version += 1
    return unresolved


def _init_state():
    defaults = {
        "v10_theme": "Dark Navy", "holdings": [], "entry_mode": "values", "base": "USD", "strategy_cache": {},
        "hypothetical_total": 25000.0, "editor_version": 0,
        "ticker_version": 0, "selected_ticker": None, "last_search_query": "", "reports": {}, "pending_quote_symbol": None, "history_auto": True, "quote_sources": {},
        "search_date_start": dt.date.today()-dt.timedelta(days=365*3),
        "search_date_end": dt.date.today() + dt.timedelta(days=1),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _style():
    """One consistent appearance on every page, selected from the sidebar."""
    # Existing cancel-selection control is styled by
    # .st-key-cancel_selected_ticker button in v10_design.theme_css.
    st.markdown(theme_css(st.session_state.get("v10_theme")), unsafe_allow_html=True)


def _portfolio_signature():
    try:
        rows = engine_rows(st.session_state.holdings, st.session_state.entry_mode,
                           st.session_state.base,
                           total_value=st.session_state.hypothetical_total)
    except PortfolioError:
        return None
    start, end = _history_dates()
    return fingerprint(rows, st.session_state.base, str(start), str(end))


def _get_inputs():
    return engine_rows(st.session_state.holdings, st.session_state.entry_mode,
                       st.session_state.base,
                       total_value=st.session_state.hypothetical_total)


def _save_report(name: str, data: dict, args: list[str]):
    st.session_state.reports[name] = {
        "signature": _portfolio_signature(), "report": data, "args": list(args),
        "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _current_report(name: str) -> dict | None:
    record = st.session_state.reports.get(name)
    if not record or record.get("signature") != _portfolio_signature():
        return None
    return record["report"]


def _run(command: str, args: list[str], *, key: str | None = None):
    name = key or command
    try:
        if command in {"insights", "dividends"}:
            # Best-effort Auto lookup; missing provider metadata is NOT a
            # reason to invent currency. V8.4 still validates explicit data.
            _resolve_auto_quotes()
        rows = _get_inputs()
        result = run_report(command, rows, st.session_state.base, args)
        _save_report(name, result, args)
        if command == "insights":
            st.session_state.pending_quote_symbol = None
        return True
    except (PortfolioError, BackendError) as exc:
        message = str(exc)
        if "quote currency unavailable" in message.lower():
            import re
            match = re.search(r"([A-Z][A-Z0-9.=_^\-]*):\s*quote currency unavailable", message, re.I)
            ticker = match.group(1).upper() if match else "a holding"
            st.session_state.pending_quote_symbol = ticker if ticker != "a holding" else None
            st.error(f"Trading currency was not provided for **{ticker}**. "
                     "On Portfolio Builder, set that holding's **Trading currency** to USD or CAD "
                     "according to its actual trading currency, then analyze again. "
                     "The amount currency can be different.")
            with st.expander("Technical details"):
                st.code(message)
        else:
            st.error(message)
        return False


def _run_strategy_cached(args: list[str]) -> bool:
    """Reuse exact existing historical tests when revisiting a selection focus.

    The cache key includes both the portfolio/history signature and all backend
    arguments, so no result can silently be reused for different assumptions.
    """
    key = json.dumps({"signature": _portfolio_signature(), "args": args,
                      "schema": "v913_same_window_spy"},
                     sort_keys=True, default=str)
    cache = st.session_state.setdefault("strategy_cache", {})
    cached = cache.get(key)
    if cached and cached.get("signature") == _portfolio_signature():
        st.session_state.reports["strategy"] = cached
        return True
    if not _run("insights", args, key="strategy"):
        return False
    cache[key] = st.session_state.reports["strategy"]
    if len(cache) > 18:
        # Keep only the newest few combinations, avoiding long-lived stale
        # reports from earlier portfolios in this browser session.
        cache.pop(next(iter(cache)))
    return True


def _warnings(report: dict):
    for warning in report.get("warnings") or []:
        if isinstance(warning, dict):
            warning = warning.get("message") or str(warning)
        st.warning(str(warning))


def _report_download(report: dict, filename: str):
    """Quiet, end-of-page source download; never crowd out the analysis."""
    _, tools_col = st.columns([5, 1])
    with tools_col:
        with st.popover("Research data"):
            st.caption("Download the original results and technical notes, including source dates and limitations.")
            st.download_button("Download JSON", json.dumps(report, indent=2, default=str),
                               file_name=filename, mime="application/json", type="secondary")


def _current_data_warning():
    if not st.session_state.holdings:
        st.info("Build a portfolio first. Use **Portfolio Builder** to add securities.")
        return True
    if _portfolio_signature() is None:
        st.warning("Your portfolio needs valid positive values or target weights totaling 100% before analysis.")
        return True
    if _history_dates()[0] >= _history_dates()[1]:
        st.warning("Choose a historical start date before the end date.")
        return True
    return False


def _header(title: str, caption: str):
    st.title(title)
    st.caption(caption)


@st.cache_data(ttl=600, show_spinner=False)
def _search(q: str):
    return lookup(q)


def _add_ticker(symbol: str, name: str, amount: float, weight: float,
                currency: str, quote_currency: str = ""):
    symbol = normalized_symbol(symbol)
    if symbol in {r["symbol"] for r in st.session_state.holdings}:
        raise PortfolioError(f"{symbol} is already in your portfolio")
    if len(st.session_state.holdings) >= MAX_HOLDINGS:
        raise PortfolioError(f"Maximum {MAX_HOLDINGS} holdings reached")
    if amount <= 0 or weight <= 0:
        raise PortfolioError("Enter a positive holding value and target weight")
    candidate = {"symbol": symbol, "name": name,
                 "market_value": float(amount), "weight": float(weight), "currency": currency}
    if quote_currency in ("", "Auto", None):
        evidence = _cached_quote_currency(symbol)
        if evidence:
            quote_currency = evidence.currency
            st.session_state.setdefault("quote_sources", {})[symbol] = evidence.source
    if quote_currency not in ("", "Auto", None):
        if quote_currency not in SUPPORTED_CURRENCIES:
            raise PortfolioError("Market quote currency must be USD or CAD")
        candidate["quote_currency"] = quote_currency
    st.session_state.holdings.append(candidate)
    st.session_state.editor_version += 1
    st.session_state.ticker_version += 1
    st.session_state.selected_ticker = None
    st.session_state.last_search_query = ""
    st.session_state.reports.clear()


def _commit_holding_widget(symbol: str, field: str, widget_key: str):
    """On-change callback runs before Streamlit rerenders derived allocations.

    Unlike reading a `number_input` after other widgets rerun, this explicitly
    writes the changed number into the portfolio's source-of-truth state.
    """
    if update_holding_amount(st.session_state.holdings, symbol, field,
                             st.session_state[widget_key]):
        st.session_state.reports.clear()


def page_builder():
    _header("Portfolio Builder", "Add investments once, then explore your portfolio on any analysis page.")
    st.caption(f"Portfolio totals and reports use **{st.session_state.base}**. Change your reporting currency in the sidebar. Individual investments can still be entered in CAD or USD.")
    choice = st.segmented_control(
        "Entry method", options=["Actual holdings", "Target weights"],
        default="Actual holdings" if st.session_state.entry_mode == "values" else "Target weights",
        help="Actual holdings: money invested in each security. Target weights: hypothetical percentages totaling 100%."
    )
    st.session_state.entry_mode = "values" if choice == "Actual holdings" else "weights"
    if st.session_state.entry_mode == "weights":
        st.session_state.hypothetical_total = st.number_input(
            f"Illustrative investment amount ({st.session_state.base} $)", min_value=0.01,
            max_value=1_000_000_000.0, step=1000.0,
            value=float(st.session_state.hypothetical_total), format="%.2f",
            help="This amount is used for dollar-level projections; the percentages determine the allocation."
        )

    st.subheader("1. Choose an investment")
    # Native Streamlit live input responds after a short typing pause. It avoids
    # the old silent searchbox dependency fallback that required Enter/blur.
    query = st.text_input(
        "Search ticker or company name", placeholder="Try SCHD, Apple, Microsoft or XEQT.TO",
        key=f"ticker_live_{st.session_state.ticker_version}", live="350ms"
    ).strip()
    if query != st.session_state.last_search_query:
        st.session_state.last_search_query = query
        st.session_state.selected_ticker = None
    if query and st.session_state.selected_ticker is None:
        matches = _search(query)[:6]
        if matches:
            st.caption("Select an investment below")
            with st.container(border=True):
                for label, info in matches:
                    if st.button(label, key=f"suggest_{st.session_state.ticker_version}_{info['symbol']}",
                                 width="stretch"):
                        st.session_state.selected_ticker = dict(info)
                        st.rerun()
        else:
            st.caption("No suggestions available. Check the spelling or use the exact-symbol option below.")

    selected = st.session_state.selected_ticker
    if selected:
        selected_info, cancel_col = st.columns([5, 1], vertical_alignment="center")
        with selected_info:
            st.success(f"Selected: **{selected['symbol']}** | {selected.get('name', selected['symbol'])}")
        with cancel_col:
            if st.button("Cancel selection", key="cancel_selected_ticker", width="stretch"):
                st.session_state.selected_ticker = None
                st.session_state.last_search_query = ""
                st.session_state.ticker_version += 1
                st.rerun()
    with st.expander("Can't find the investment? Enter its exact ticker"):
        manual = st.text_input("Exact ticker symbol", placeholder="e.g. XEQT.TO",
                               key=f"manual_ticker_{st.session_state.ticker_version}")

    if not selected and not manual.strip():
        st.caption("Select a matching investment above to enter its value or allocation.")
        _render_portfolio_list()
        _render_sample_portfolios()
        return

    st.subheader("2. Set amount and add to portfolio")
    left, middle, right = st.columns([1.5, 1, 1])
    with left:
        if st.session_state.entry_mode == "values":
            amount = st.number_input("Holding value ($)", min_value=0.01, value=1000.0, step=100.0)
            weight = 10.0
        else:
            weight = st.number_input("Allocation (%)", min_value=0.01, max_value=100.0,
                                     value=10.0, step=1.0)
            amount = 1000.0
    with middle:
        curr = st.selectbox("Amount currency", SUPPORTED_CURRENCIES,
                            index=SUPPORTED_CURRENCIES.index(st.session_state.base),
                            disabled=st.session_state.entry_mode == "weights",
                            help="Currency of the entered amount (not necessarily the security's trading currency).")
    with right:
        quote_choice = st.selectbox("Trading currency", ["Detect automatically", "USD", "CAD"],
                    help="The currency the investment actually trades in. This may differ "
                         "from the currency of the amount you entered.")
        quote = "Auto" if quote_choice == "Detect automatically" else quote_choice
        if quote == "Auto" and selected and selected.get("quote_currency") in SUPPORTED_CURRENCIES:
            quote = selected["quote_currency"]
    if st.button("Add to portfolio", type="primary", disabled=len(st.session_state.holdings) >= MAX_HOLDINGS):
        try:
            symbol = (manual.strip() if manual else "") or ((selected or {}).get("symbol") or "")
            if not symbol:
                raise PortfolioError("Select a search result or enter an exact ticker first")
            label = ((selected or {}).get("name") if symbol.upper() == (selected or {}).get("symbol") else None) or symbol
            _add_ticker(symbol, label, amount, weight, curr, quote)
            st.rerun()
        except PortfolioError as exc:
            st.error(str(exc))

    _render_portfolio_list()
    _render_sample_portfolios()


def _render_sample_portfolios():
    """Show sample portfolios at the end of Builder, regardless of ticker selection."""
    st.divider()
    # Test fixture chooser, available for both empty and populated portfolios.
    # Loading a fixture replaces session holdings only; it does not place trades.
    with st.expander("Sample portfolios for testing", expanded=False):
        demo_key = st.selectbox(
            "Choose a portfolio", [x.key for x in EXAMPLES],
            format_func=lambda key: get_example(key).label,
            key="builder_example_choice",
        )
        demo = get_example(demo_key)
        st.write(f"**{demo.purpose}**")
        if st.button("Load this example (replace current portfolio)",
                     key="load_builder_example", width="stretch",
                     help="Loading replaces the current portfolio in this session; no trades are placed."):
            try:
                load_example(st.session_state, demo_key)
                st.rerun()
            except PortfolioError as exc:
                st.error(str(exc))


def _render_portfolio_list():
    st.divider()
    count = len(st.session_state.holdings)
    st.subheader(f"Your portfolio · {count}/{MAX_HOLDINGS} investments")
    if not count:
        st.markdown("<div class='v10-soft-info'>Your portfolio is empty. Add a ticker above or load an example to see how the dashboard works.</div>", unsafe_allow_html=True)
        st.caption("Open **Sample portfolios for testing** below to load a ready-made example, including the starter portfolio and edge cases.")
    else:
        rows = st.session_state.holdings
        nominal = None
        share_by_ticker = {}
        fx_date = None
        if st.session_state.entry_mode == "values":
            fx_required = any(r["currency"] != st.session_state.base for r in rows)
            fx_info = _cached_display_fx(dt.date.today()) if fx_required else None
            try:
                value = display_portfolio_value(
                    rows, st.session_state.base,
                    usd_cad=fx_info[0] if fx_info else None)
                nominal = float(value.total)
                share_by_ticker = value.shares
                fx_date = fx_info[1] if fx_info else None
            except (ValueError, ArithmeticError):
                pass
        head = st.columns([2.3, 1.3, 1, 1.1, .5])
        for column, label in zip(head, ["INVESTMENT",
                   "HOLDING VALUE ($)" if st.session_state.entry_mode == "values" else "WEIGHT (%)",
                   "AMOUNT CURRENCY", "TRADING CURRENCY", ""]):
            with column:
                st.caption(label)
        for index, holding in enumerate(list(rows)):
            symbol = holding["symbol"]
            widget_key = f"{symbol}_{st.session_state.entry_mode}_{st.session_state.editor_version}"
            with st.container(border=True, key=f"portfolio_holding_card_{index}"):
                name_col, val_col, currency_col, quote_col, delete_col = st.columns(
                    [2.3, 1.3, 1, 1.1, .5], vertical_alignment="center"
                )
                with name_col:
                    st.markdown(f"**{symbol}**")
                    st.caption(str(holding.get("name") or symbol))
                    if st.session_state.entry_mode == "values" and nominal:
                        share = share_by_ticker.get(symbol)
                        if share is not None:
                            st.caption(f"{share:.2f}% of portfolio")
                with val_col:
                    key = "market_value" if st.session_state.entry_mode == "values" else "weight"
                    title = "Holding value ($)" if key == "market_value" else "Allocation (%)"
                    amount_widget = f"amount_{widget_key}"
                    st.number_input(title, min_value=0.01,
                        max_value=100.0 if key == "weight" else 1_000_000_000.0,
                        value=float(holding[key]), step=1.0 if key == "weight" else 100.0,
                        key=amount_widget, format="%.2f", label_visibility="collapsed",
                        on_change=_commit_holding_widget,
                        args=(symbol, key, amount_widget))
                with currency_col:
                    ccy = st.selectbox("Amount currency", SUPPORTED_CURRENCIES,
                        index=SUPPORTED_CURRENCIES.index(holding["currency"]),
                        disabled=st.session_state.entry_mode == "weights",
                        key=f"currency_{widget_key}", label_visibility="collapsed",
                        help="Currency used for the entered holding value.")
                    if st.session_state.entry_mode == "values" and ccy != holding["currency"]:
                        st.session_state.holdings[index]["currency"] = ccy
                        st.session_state.reports.clear()
                with quote_col:
                    existing_quote = holding.get("quote_currency") or "Auto"
                    quote_options = ["Detect automatically", "USD", "CAD"]
                    selection = st.selectbox("Trading currency", quote_options,
                        index=quote_options.index("Detect automatically" if existing_quote == "Auto" else existing_quote),
                        key=f"quote_{widget_key}", label_visibility="collapsed",
                        help="Trading currency of this ticker. "
                        "If Yahoo omits this metadata, choose its correct trading currency here.")
                    if ("" if selection == "Detect automatically" else selection) != (holding.get("quote_currency") or ""):
                        if selection == "Detect automatically":
                            st.session_state.holdings[index].pop("quote_currency", None)
                        else:
                            st.session_state.holdings[index]["quote_currency"] = selection
                        st.session_state.reports.clear()
                with delete_col:
                    if st.button("✕", key=f"remove_{widget_key}", help=f"Remove {symbol}"):
                        st.session_state.holdings.pop(index)
                        st.session_state.editor_version += 1
                        st.session_state.reports.clear()
                        st.rerun()

        st.caption("Changes to holding values are saved when you press Enter or leave the field. Re-run an analysis to use the updated portfolio.")
        with st.expander("How do the two currency settings work?"):
            st.write("**Amount currency** is what you entered your investment value in. **Trading currency** is what the stock or fund trades in. For example, $5,000 CAD invested in a US-listed fund can have CAD as the amount currency and USD as the trading currency. Your overall reporting currency is chosen in the sidebar.")
        if st.session_state.entry_mode == "weights":
            total_w = sum(float(r.get("weight") or 0) for r in st.session_state.holdings)
            if abs(total_w - 100) > 0.0001:
                st.warning(f"Allocations total {total_w:.2f}%. Adjust them to exactly 100% before analysis.")
            else:
                st.success("Allocations total 100%")
            st.metric("Hypothetical investment amount", display_money(
                st.session_state.hypothetical_total, st.session_state.base))
        elif nominal is not None:
            st.metric(f"Estimated portfolio value in {st.session_state.base}",
                      display_money(nominal, st.session_state.base))
            if fx_date:
                st.caption(f"Converted using the available USD/CAD closing rate from {fx_date}. The analysis separately checks the appropriate historical exchange rates.")
        else:
            st.caption("To show a combined portfolio value, we need a recent CAD/USD rate. The quote is unavailable right now. You can still edit your holdings and retry later.")
        a, b = st.columns(2)
        with a:
            if st.button("Clear portfolio", type="secondary"):
                st.session_state.holdings = []
                st.session_state.reports.clear()
                st.session_state.editor_version += 1
                st.rerun()
        try:
            from .core import to_csv
            with b:
                st.download_button("Export portfolio CSV", to_csv(_get_inputs()),
                                   file_name="my_portfolio.csv", mime="text/csv")
        except PortfolioError:
            pass

    with st.expander("Import an existing CSV (optional)"):
        uploaded = st.file_uploader("CSV with symbol,market_value,currency[,quote_currency]",type="csv",key="portfolio_csv")
        if uploaded and st.button("Import CSV holdings"):
            try:
                st.session_state.holdings = [dict(x, name=x["symbol"], weight=0.0)
                                             for x in import_values_csv(uploaded.getvalue())]
                st.session_state.entry_mode = "values"
                st.session_state.reports.clear()
                st.session_state.editor_version += 1
                st.rerun()
            except PortfolioError as exc:
                st.error(str(exc))
    if st.session_state.holdings and _portfolio_signature():
        st.caption("Portfolio ready · Open **Overview** to run analysis.")

def _sentiment_card(item: dict):
    """Metric-specific benchmark interpretation; V8.4 supplies the classification."""
    finding = finding_for_people(item)
    colors = {"positive": "green", "negative": "red", "neutral": "gray"}
    with st.container(border=True):
        st.badge(finding["status_label"], color=colors.get(finding["status"], "gray"))
        st.markdown(f"**{finding['headline']}**")
        st.write(finding["comparison"])
        st.caption(finding["reason"])
        if finding["explanation"]:
            metric_name = finding['headline'].split(' compared with ',1)[0]
            with st.popover(f"About {metric_name}"):
                st.write(finding["explanation"])


def _diagnostics(report: dict, *, maximum: int = 6):
    obj = report.get("diagnostics") or {}
    if not isinstance(obj, dict):
        return
    all_items = (obj.get("negative") or []) + (obj.get("positive") or []) + (obj.get("neutral") or [])
    if not all_items:
        st.info("Not enough information for meaningful benchmark comparisons.")
        return
    items = select_overview_findings(obj, maximum=maximum) if maximum <= 4 else all_items[:maximum]
    for item in items:
        _sentiment_card(item)
    if len(all_items) > len(items):
        st.caption(f"Showing {len(items)} distinct takeaways of {len(all_items)} comparisons. Open Risk & Performance for the full list.")


def _findings_tabs(report: dict):
    """Plain language, consistent with the Strengths / Similar / Watch labels."""
    obj = report.get("diagnostics") or {}
    positives = obj.get("positive") or []
    neutrals = obj.get("neutral") or []
    negatives = obj.get("negative") or []
    st.caption("Strengths mean a favorable historical comparison, areas to watch mean an unfavorable one, "
               "and similar results mean the gap was too small for this screening rule. "
               "These are comparisons, not predictions.")
    a, b, c = st.tabs([f"Strengths ({len(positives)})", f"Similar results ({len(neutrals)})", f"Areas to watch ({len(negatives)})"])
    for tab, items, label in ((a, positives, "strengths"), (b, neutrals, "similar results"), (c, negatives, "areas to watch")):
        with tab:
            if not items:
                st.caption(f"No {label} found against these comparisons.")
                continue
            show_all = (st.toggle(f"Show all {len(items)} {label}", key=f"findings_all_{label}")
                        if len(items) > 3 else True)
            for item in items if show_all else items[:3]:
                _sentiment_card(item)


def _metric_row(report: dict):
    p = report.get("portfolio") or {}
    defs = [
       ("Historical CAGR", "cagr", pct),
       ("Volatility", "annualized_volatility", pct),
       ("Sharpe ratio", "sharpe", lambda x: "Unavailable" if x is None else f"{x:.2f}"),
       ("Max drawdown", "drawdown", pct),
    ]
    cols = st.columns(4, vertical_alignment="top")
    for col, (label, metric, formatter) in zip(cols, defs):
        with col:
            st.metric(label, formatter(p.get(metric)),
                      help=RISK_METRIC_HELP[metric][1])
            status = benchmark_status(report, metric)
            if status is not None:
                badge, color = {"positive": ("Strength vs SPY", "green"),
                                "negative": ("Area to watch vs SPY", "red"),
                                "neutral": ("Similar to SPY", "gray")}[status]
                st.badge(badge, color=color)

def _income_compact(report: dict):
    summary = report.get("income_summary") or {}
    if not summary:
        return
    st.subheader("Dividend Income")
    fully = show_complete_dividend_yield(summary)
    rate = summary.get("fully_covered_portfolio_yield") if fully else None
    c1,c2,c3 = st.columns(3)
    c1.metric("Portfolio dividend yield", pct(rate) if fully else "Incomplete coverage")
    c2.metric("Estimated annual income", display_money(summary.get("estimated_annual_gross_income_covered_holdings"),report.get("base_currency","USD")))
    c3.metric("Monthly average", display_money(summary.get("estimated_monthly_average_gross_income_covered_holdings"),report.get("base_currency","USD")))
    coverage = summary.get("income_coverage_fraction")
    if coverage is not None and coverage < 1:
        st.warning(f"Dividend coverage: {pct(coverage)}. These income totals describe evaluated holdings only; missing data is not zero income.")
    st.caption("Based on payments over the last 12 months, before taxes. Monthly average = annual income ÷ 12; payments may not arrive monthly.")


def page_overview():
    _header("Portfolio Overview", "One clear view of your portfolio's performance, risks and evidence-based findings.")
    if _current_data_warning():
        return
    rows = _get_inputs()
    report = _current_report("insights")
    if report:
        st.subheader(f"Historical period analyzed: {historical_window_text(report)}")
        st.caption(HISTORY_NOTE)
        st.caption(f"{len(rows)} investments, results shown in {st.session_state.base}.")
    elif st.session_state.history_auto:
            st.caption(f"{len(rows)} holdings, shown in {st.session_state.base}. The analysis will use the longest shared price history available.")
    else:
        start, end = _history_dates()
        st.caption(f"{len(rows)} holdings · {st.session_state.base} · Requested dates: {start} to {end} (end exclusive)")
    if st.button("Analyze portfolio", type="primary", width="stretch"):
        with st.spinner("Analyzing portfolio and dividend income using available market data..."):
            start, end = _history_dates()
            args = ["--start", str(start), "--end", str(end)]
            insight_ok = _run("insights", args)
            st.session_state.last_insights_failure_sig = None if insight_ok else _portfolio_signature()
            # Income is useful on Overview; a failure does not erase valid risk analysis.
            _run("dividends", ["--years","10","--price-growth","4","--dividend-growth","3","--tax-drag","0"])
    pending = st.session_state.get("pending_quote_symbol")
    if pending and pending in {h["symbol"] for h in st.session_state.holdings}:
        st.caption(f"If you know where **{pending}** trades, confirm its trading currency here "
                   "without returning to Portfolio Builder. Do not choose based only on your amount currency.")
        usd, cad = st.columns(2)
        for column, currency in ((usd, "USD"), (cad, "CAD")):
            with column:
                if st.button(f"Confirm {pending} trades in {currency}", key=f"confirm_quote_{pending}_{currency}"):
                    for holding in st.session_state.holdings:
                        if holding["symbol"] == pending:
                            holding["quote_currency"] = currency
                    st.session_state.reports.clear()
                    st.session_state.pending_quote_symbol = None
                    st.session_state.editor_version += 1
                    st.rerun()
    insights = _current_report("insights")
    income = _current_report("dividends")
    if insights:
        st.subheader("Performance at a glance")
        _metric_row(insights)
        st.caption("More indicators and risk explanations are available on Risk & Performance.")
        st.divider()
        if isinstance(insights.get("input_weights"), dict):
            st.subheader("Portfolio allocation")
            _allocation_donut(insights["input_weights"])
        st.divider()
        st.subheader("Key findings")
        _benchmark_explainer(compact=True)
        _diagnostics(insights, maximum=4)
        summary, additional = simplify_insights_warnings(insights.get("warnings") or [])
        critical = [note for note in additional if any(word in note.upper() for word in
                    ("STALE_END", "LARGE_GAP", "ERROR", "FAILURE"))]
        if critical:
            st.caption(f"Market-data note: {len(critical)} potentially important provider issue(s). Details below.")
        # The historical basis is already explained next to the dates.
        # Keep raw notices in the downloadable report rather than repeating them.
        # Keep data export below the visible income summary.
    else:
        if st.session_state.get("last_insights_failure_sig") == _portfolio_signature():
            st.warning("Risk and performance analysis could not be completed. "
                       "Correct the issue above and analyze again. Income may still be available independently.")
        else:
            st.info("Click **Analyze portfolio** to populate the historical metrics, benchmark comparisons and findings from the real backend. No illustrative risk scores are shown.")
    st.divider()
    if income:
        _income_compact(income)
        _warnings(income)
    elif insights:
        st.caption("Dividend information unavailable for this run; see Income & Dividends to retry.")
    if insights:
        _report_download(insights, "v9_portfolio_insights.json")


def _allocation_donut(weights: dict):
    values = allocation_pie_rows(weights)
    if not values:
        st.caption("Allocation unavailable for this report.")
        return
    fig = go.Figure(go.Pie(
        labels=[row["Holding"] for row in values],
        values=[row["Weight (%)"] for row in values],
        hole=0.55, sort=False, domain=dict(x=[0.06, 0.72]),
        textinfo="percent" if len(values) <= 8 else "none",
        textposition="inside",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        marker=dict(line=dict(color="#0c1422", width=1)),
    ))
    fig.update_layout(height=318, margin=dict(l=12,r=12,t=5,b=5),
                      legend=dict(orientation="v", y=0.5, yanchor="middle", x=0.77,
                                  font=dict(size=13)),
                      showlegend=True)
    tune_plotly(fig, st.session_state.get("v10_theme"))
    st.plotly_chart(fig, width="stretch", theme=None)


def _benchmark_explainer(*, compact: bool = False):
    st.markdown("#### Your comparison benchmarks" if compact else "#### Compare against the U.S. stock market")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**SPY · S&P 500 ETF**")
            st.caption("A widely used U.S. stock-market benchmark. Tracks roughly 500 large U.S. "
                       "companies, with the biggest companies carrying the most weight.")
    with c2:
        with st.container(border=True):
            st.markdown("**RSP · Equal-weight S&P 500 ETF**")
            st.caption("A second perspective on the same 500 companies. Gives each company roughly "
                       "equal weight instead of emphasizing the largest firms.")


def _risk_extra_metrics(report: dict):
    p = report.get("portfolio") or {}
    relative = (report.get("relative_to_benchmarks") or {}).get("SPY") or {}
    st.markdown("**Additional indicators**")
    cols = st.columns(4)
    items = [
        ("Sortino ratio", "sortino", p.get("sortino"), "number"),
        ("Calmar ratio", "calmar", p.get("calmar"), "number"),
        ("Expected shortfall (95%)", "es95", p.get("es95"), "percent"),
        ("Beta versus SPY", "beta", relative.get("beta"), "number"),
    ]
    for col,(label,metric,value,kind) in zip(cols,items):
        with col:
            shown = pct(value) if kind == "percent" else ("Unavailable" if value is None else f"{value:.2f}")
            st.metric(label, shown, help=RISK_METRIC_HELP[metric][1])
            status = benchmark_status(report,metric) if metric in {"sortino", "es95"} else None
            if status:
                badge, color = {"positive": ("Strength vs SPY", "green"),
                                "negative": ("Area to watch vs SPY", "red"),
                                "neutral": ("Similar to SPY", "gray")}[status]
                st.badge(badge, color=color)
            elif metric == "calmar":
                st.caption("Context only. Higher means more growth relative to the worst historical fall.")
            elif metric == "beta":
                if isinstance(value, (int, float)):
                    st.caption(f"SPY sensitivity: {value:.2f}×. Around 1 means similar market-linked movements, not a grade.")
                else:
                    st.caption("Market sensitivity unavailable for this sample.")

def _benchmark_bars(report: dict, benchmark: str = "SPY"):
    """Compare portfolio versus selected ETF(s); never imply three benchmarks."""
    p = report.get("portfolio") or {}
    benchmarks = report.get("benchmarks") or {}
    chosen = ("SPY", "RSP") if benchmark=="Both" else (benchmark,)
    data = []
    for name,row in [("Your portfolio",p), *[(label,benchmarks.get(label, {})) for label in chosen]]:
        if not isinstance(row,dict):
            continue
        for metric,label in [("cagr","Annual growth"),("annualized_volatility","Volatility"),("drawdown","Max drawdown")]:
            value = row.get(metric)
            if isinstance(value,(int,float)):
                data.append({"Portfolio":name,"Metric":label,"Value":float(value)*100})
    if not data:
        st.info("Benchmark comparisons are unavailable for this time period.")
        return
    fig = px.bar(pd.DataFrame(data),x="Metric",y="Value",color="Portfolio",barmode="group",
                 text="Value",labels={"Value":"Percent (%)","Metric":"Historical measure"},
                 color_discrete_sequence=["#78A9EB","#A5B4FC","#A7C7B7"])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside",cliponaxis=False,
                      hovertemplate="%{fullData.name}: %{y:.2f}%<extra></extra>")
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=5,r=10,t=25,b=15),legend=dict(orientation="h",y=1.17),
                      yaxis=dict(rangemode="tozero"),height=355)
    tune_plotly(fig, st.session_state.get("v10_theme"))
    st.plotly_chart(fig,width="stretch", theme=None)
    st.caption("**Higher annual growth** is favorable; **lower volatility and smaller drawdowns** are generally preferable. This is past performance, not a prediction.")


def _render_risk_attribution(report: dict):
    items = inspect_risk_contribution(report)
    if not items:
        st.info("We couldn't measure each holding's contribution to portfolio risk for this time period.")
        return
    st.subheader("Which investments drive the most ups and downs?")
    st.write("See how much of your portfolio is invested in each holding, and how much it contributed to historical ups and downs. The larger the risk share compared with its weight, the greater its contribution to volatility.")
    df = pd.DataFrame(items).copy()
    df["Allocation (%)"] = df["Allocation (%)"].round(2)
    df["Share of volatility risk (%)"] = df["Share of volatility risk (%)"].round(2)
    long = df.melt(id_vars="Ticker",value_vars=["Allocation (%)","Share of volatility risk (%)"],
                   var_name="What is measured",value_name="Share (%)")
    fig = px.bar(long, y="Ticker",x="Share (%)",color="What is measured",orientation="h",barmode="group",
                 text="Share (%)",color_discrete_sequence=["#4A9BCD", "#E2A25C"])
    fig.update_traces(texttemplate="%{text:.2f}%",textposition="outside",cliponaxis=False)
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=5,r=10,t=40,b=0),yaxis={"autorange":"reversed"},height=max(320,len(df)*70),
                      legend=dict(orientation="h",y=1.12),xaxis=dict(rangemode="tozero"))
    tune_plotly(fig, st.session_state.get("v10_theme"))
    st.plotly_chart(fig,width="stretch", theme=None)
    first=items[0]
    name=first["Ticker"]
    share=first["Share of volatility risk (%)"]
    allocation=first["Allocation (%)"]
    st.caption(f"For example, **{name}** is **{allocation:.2f}%** of your portfolio but contributed **{share:.2f}%** of its measured volatility. This isn't a forecast of future losses.")


def _historical_research(report: dict):
    st.subheader("Explore historical risks and possible outcomes")
    st.caption("Investigate market crashes, modeled one-year scenarios and how results change across time periods.")
    crises, simulations, sensitivity = st.tabs(["Past market crashes", "One-year scenarios", "Different time periods"])
    with crises:
        st.write("**How would today's portfolio mix have done during major market downturns?**")
        st.caption("The first column shows the portfolio's gain or loss between the event's starting and ending dates. The second is the biggest fall from a high to a later low during that event.")
        events = (report.get("historical_events") or {}).get("portfolio") or []
        evaluated = [e for e in events if e.get("status")=="evaluated"]
        if evaluated:
            rows=[{"Event":e.get("event"),"Total change from start to finish":pct(e.get("total_return")),
                   "Largest drop during event":pct(e.get("maximum_drawdown"))} for e in evaluated]
            display_table(pd.DataFrame(rows),hide_index=True,width="stretch")
        else:
            st.info("No complete crisis period was available for the shared history of these investments. We won't invent missing results.")
        missing=[e.get("event") for e in events if e.get("status")!="evaluated"]
        if missing:
            st.caption("Events without enough historical data: " + ", ".join(str(x) for x in missing))
    with simulations:
        st.write("**Illustrative one-year return scenarios**")
        st.caption("We run a historical block-bootstrap simulation: it stitches together short blocks of past daily returns to create many imaginary one-year market paths. These are examples, not predicted returns or guaranteed odds.")
        mc = report.get("block_bootstrap_monte_carlo") or {}
        rets = mc.get("terminal_return_p05_p50_p95") or []
        if mc.get("status")=="evaluated" and len(rets)==3:
            a,b,c=st.columns(3)
            a.metric("Downside scenario (5th percentile)",pct(rets[0]),help="A poorer hypothetical one-year outcome. About 5% of sampled paths fell below this, but actual losses could be worse.")
            b.metric("Middle scenario (median)",pct(rets[1]),help="The middle of the simulated one-year returns, not a predicted or recommended base case.")
            c.metric("Upside scenario (95th percentile)",pct(rets[2]),help="A stronger hypothetical one-year outcome. About 5% of sampled paths exceeded this, but future returns are not known.")
            st.caption(f"Model: historical block bootstrap, {mc.get('paths','?')} simulated paths of {mc.get('horizon_periods','?')} trading days each. The high and low examples reflect sampled historical patterns, not boundaries or probabilities for future returns. With only a few hundred paths the extremes can change noticeably if the sample changes. Past data cannot tell us how likely future returns are.")
        else:
            st.caption("Not enough consistent historical returns to show the simulation.")
    with sensitivity:
        st.write("**Do the results change when you only consider a shorter recent period?**")
        st.caption("Compare annual growth and volatility using the available last 1, 3 or 5 years. A shorter history may show different behavior; unavailable periods are not filled with guesses.")
        info = report.get("starting_date_sensitivity") or {}
        rows=[]
        missing=[]
        for window,item in info.items():
            if not isinstance(item,dict):continue
            if item.get("status")=="evaluated":
                metrics=item.get("metrics") or {}
                rows.append({"History used":str(window),"Annual growth":pct(metrics.get("cagr")),
                             "Volatility":pct(metrics.get("annualized_volatility"))})
            else:missing.append(str(window))
        if rows:
            display_table(pd.DataFrame(rows),hide_index=True,width="stretch")
        else:st.caption("A shorter-date comparison could not be calculated for this portfolio.")
        if missing:
            st.caption("Insufficient full history for: " + ", ".join(missing))

def page_risk():
    _header("Risk & Performance", "Understand returns, market swings, benchmarks and what is driving your portfolio's risk.")
    if _current_data_warning(): return
    r = _current_report("insights")
    if r is None:
        st.info("Run **Analyze portfolio** on Overview first. This page uses the same validated results.")
        return
    st.subheader(f"Historical period analyzed: {historical_window_text(r)}")
    st.caption(HISTORY_NOTE)
    st.divider()
    st.subheader("Risk and return")
    _metric_row(r)
    _risk_extra_metrics(r)
    st.divider()
    st.subheader("How does your portfolio compare?")
    _benchmark_explainer()
    selected = st.segmented_control("Compare against", ["SPY", "RSP", "Both"],
                                    default="SPY", key="benchmark_choice")
    _benchmark_bars(r, selected or "SPY")
    st.divider()
    _render_risk_attribution(r)
    st.divider()
    st.subheader("Strengths and areas to watch")
    _findings_tabs(r)
    st.divider()
    _historical_research(r)
    summary, other = simplify_insights_warnings((r.get("warnings") or []) + (r.get("data_quality_warnings") or []))
    serious = [x for x in other if any(key in x.upper() for key in ("STALE_END", "LARGE_GAP", "ERROR", "FAILURE"))]
    if serious:
        st.caption(f"Data note: {len(serious)} potentially important provider issue(s). See details below.")
    with st.expander("Analysis method and data notes", expanded=False):
        st.write("This is a historical illustration of today's portfolio weights. It does not use your personal transaction history or predict future prices.")
        st.caption(f"Shared price history: {r.get('first_date', '?')} to {r.get('last_date', '?')}. Only dates with comparable observations were used.")
        for message in summary:
            st.caption(message)
        if serious:
            st.warning(f"{len(serious)} potentially important market-data issue(s). Open the downloadable research report for the details.")
        st.caption("Full technical notices are preserved in the downloadable JSON report.")
    _report_download(r,"v9_portfolio_risk.json")

def _bar_from_weights(weights: dict, title: str, max_rows: int = 12):
    rows = [{"Holding": str(k), "Weight (%)": float(v)*100} for k,v in weights.items()
            if isinstance(v,(float,int)) and float(v)>0]
    if not rows:
        st.caption("No verified holdings available for this chart.")
        return
    df = pd.DataFrame(rows).sort_values("Weight (%)",ascending=False).head(max_rows)
    fig = px.bar(df, x="Weight (%)", y="Holding", orientation="h", title=title,
                 text="Weight (%)", color_discrete_sequence=["#78A9EB"])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside",cliponaxis=False,
                      hovertemplate="%{y}: %{x:.2f}% of portfolio<extra></extra>")
    fig.update_layout(yaxis={"autorange":"reversed", "title":None},template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      height=max(260,min(650,len(df)*35+90)),
                      margin=dict(l=5,r=50,t=50,b=5), xaxis=dict(rangemode="tozero",title="Share of portfolio (%)"))
    tune_plotly(fig, st.session_state.get("v10_theme"))
    st.plotly_chart(fig,width="stretch", theme=None)


def _composition_chart(weights: dict, label: str):
    rows, unknown = composition_data_with_unknown(weights,label)
    if rows:
        df = pd.DataFrame(rows)
        fig = px.bar(df, x="Share of portfolio (%)",y="Category",orientation="h",
                     text="Share of portfolio (%)", color="Coverage",
                     color_discrete_map={"Identified":"#78A9EB", "Unclassified":"#66758C"})
        fig.update_traces(texttemplate="%{text:.2f}%",textposition="outside",cliponaxis=False,
                          hovertemplate="%{y}: %{x:.2f}% of portfolio<extra></extra>")
        fig.update_layout(yaxis=dict(autorange="reversed",title=None),
                          xaxis=dict(rangemode="tozero", title="Share of portfolio (%)"),
                          template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=5,r=55,t=10,b=10),height=max(250,min(620,70+len(df)*35)),
                          showlegend=False)
        tune_plotly(fig, st.session_state.get("v10_theme"))
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.caption("No verified categories available for this breakdown.")
    if unknown > .005:
        headline, description = coverage_summary(max(0.0, 100.0-unknown), unknown)
        if label == "Sectors":
            note = description
        elif label == "Countries (company domicile)":
            note = f"{unknown:.2f}% could not be assigned to a verified company domicile or does not have one."
        else:
            note = f"{unknown:.2f}% could not be placed in a verified asset type."
        st.caption(f"**{headline}.** {note} The unclassified share stays separate "
                   "and is not added to the other bars.")

def page_exposure():
    _header("Holdings & Exposure", "Find out which stocks, sectors and assets you own directly or through ETFs, and where they overlap.")
    if _current_data_warning():
        return
    # Fixed, bounded metadata enrichment without confusing users with provider limits.
    enrich = 10
    if st.button("Analyze underlying exposure",type="primary"):
        args=["--as-of",str(dt.date.today()),"--tickerinside",
              "--auto-stock-metadata","--drilldown"]
        if enrich:
            args += ["--enrich-underlyings",str(enrich)]
        with st.spinner("Looking up available ETF holdings and company details..."):
            _run("exposure", args)
    r=_current_report("exposure")
    if not r:
        st.info("Select **Analyze underlying exposure** to explore the stocks beneath your ETFs. Missing provider data stays marked as unknown.")
        return
    unknown = r.get("unknown_underlying")
    if isinstance(unknown, (int,float)):
        a,b = st.columns(2)
        a.metric("Underlying holdings identified", pct(max(0,1-unknown)),
                 help="Share of your portfolio we can map to direct positions or reported ETF constituents. This does not mean that every company sector is known.")
        b.metric("Underlying holdings not identified", pct(unknown),
                 help="Share we cannot map to a named position using current available fund data. This is different from a known position with an unknown sector.")
    st.divider()
    st.subheader("Explore what's inside your portfolio")
    nav_descriptions = [
        ("Portfolio makeup", "Sectors, assets and countries"),
        ("Explore sectors", "Which holdings create concentration"),
        ("ETF overlap", "Companies shared by your funds"),
        ("Underlying investments", "The stocks inside ETFs"),
    ]
    if "xray_active_section" not in st.session_state:
        st.session_state.xray_active_section = "Portfolio makeup"
    for col, (name, explanation) in zip(st.columns(4), nav_descriptions):
        with col:
            with st.container(border=True):
                if st.button(name, key=f"exposure_nav_{name.replace(' ', '_')}",
                             type="primary" if st.session_state.xray_active_section == name else "secondary",
                             width="stretch", help=explanation):
                    st.session_state.xray_active_section = name
                    st.rerun()
                st.caption(explanation)
    view = st.session_state.xray_active_section
    st.divider()
    breakdowns = read_exposure_breakdowns(r)
    drill=r.get("drilldown") or {}
    if view == "Portfolio makeup":
        st.markdown("**Where is your money concentrated?**")
        chosen = st.segmented_control("View breakdown", list(breakdowns),
                                      default="Sectors", key="xray_composition_mode")
        label = chosen or "Sectors"
        if label == "Countries (company domicile)":
            st.caption("Shows where companies are legally based, **not** where they earn revenue or how much currency exposure you have.")
        elif label == "Asset classes":
            st.caption("Broad asset types such as stocks or commodities. A gold holding is a commodity, not a company sector.")
        else:
            st.caption("Business sectors behind your direct stocks and supported ETF holdings. Larger bars mean more of your invested money is exposed to that sector.")
        _composition_chart(breakdowns.get(label) or {},label)
    if view == "Explore sectors":
        st.markdown("**Which investments create your sector exposure?**")
        sectors = drill.get("sector_drilldown") or {}
        if not sectors:
            st.info("Individual sector drill-down is unavailable for this portfolio; the provider may not supply the needed classifications.")
        else:
            selected = st.selectbox("Choose a sector", sort_sectors_by_exposure(sectors), key="xray_sector",
                                    format_func=lambda x: x.replace("_"," ").title())
            detail=sectors[selected]
            a,b,c = st.columns(3)
            a.metric("Portfolio in this sector",pct(detail.get("portfolio_fraction")),
                     help="Total portfolio share classified to this sector, including fund-level sector reports.")
            b.metric("Named companies confirmed",pct(detail.get("known_company_total_fraction")),
                     help="Portfolio share where the particular underlying company's sector was independently verified.")
            c.metric("Sector share without named firms",pct(detail.get("unattributed_company_fraction")),
                     help="Sector allocation supplied by funds but not yet attributed to individually verified companies. This is not zero exposure.")
            by=detail.get("by_investment") or {}
            if by:
                st.markdown("**Which of your investments add this exposure?**")
                _bar_from_weights(by,f"{selected.replace('_',' ').title()}: investments adding this exposure",20)
            companies=detail.get("known_company_contributions") or {}
            if companies:
                with st.expander("Show individually verified underlying companies"):
                    _bar_from_weights(companies,f"{selected.replace('_',' ').title()}: identified companies",20)
            st.caption("Fund sector totals and individually verified company sectors come from different evidence; we show them separately so missing classifications don't look like zero exposure.")
    if view == "ETF overlap":
        st.markdown("**Do your ETFs own many of the same stocks?**")
        st.caption("Two ETFs can both hold a company. That creates repeated exposure, even if the ETF names sound different. An overlap is not automatically bad, but it may reduce diversification.")
        overlaps=overlap_rows(r)
        if overlaps:
            df=pd.DataFrame(overlaps).rename(columns={
                "Identified common equity overlap (min %)":"Identified overlap (at least %)",
                "Identified overlapping companies":"Shared stocks identified"
            })
            df["Identified overlap (at least %)"]=df["Identified overlap (at least %)"].round(2)
            display_table(df,hide_index=True,width="stretch",alt="Pairs of ETFs and the minimum equity overlap identified from provider holdings")
            st.caption("Overlap estimates compare the funds' known stock weights, **not** the percentage of your total portfolio. They are minimum estimates; missing holdings and different reporting dates can change the real overlap.")
        else:
            st.info("No verified ETF-to-ETF stock overlap was available for these holdings.")
    if view == "Underlying investments":
        st.markdown("**What securities sit underneath your holdings?**")
        st.caption("Shows large direct positions and identified holdings inside your ETFs; a metal like gold can appear here even though it has no company sector.")
        weights=r.get("identified_underlying") or {}
        if weights:
            _bar_from_weights(weights,"Largest identified underlying investments",15)
        else:
            st.info("No underlying constituents could be verified from the available provider data.")
    routine, caution = exposure_notes(r.get("warnings") or [])
    with st.expander("Data sources and coverage (optional)",expanded=False):
        st.write("ETF holdings come from the latest available fund reports. Some funds or sectors may have incomplete information. The underlying stock lists are not a record of what ETFs owned in earlier years.")
        support=holding_support_rows(r)
        if support:
            with st.expander("Coverage by investment"):
                display_table(pd.DataFrame(support_for_people(support)),hide_index=True,width="stretch",
                             alt="Data coverage by portfolio investment")
        fund=drill.get("fund_top_holdings") or r.get("coverage_by_fund") or {}
        dated=snapshot_rows(fund)
        if dated:
            with st.expander("When did ETFs last publish their holdings?"):
                display_table(pd.DataFrame(dated),hide_index=True,width="stretch",
                             alt="ETF disclosure dates, holdings coverage, and sources")
        if caution:
            st.markdown("**Provider limitations worth reviewing**")
            for message in list(dict.fromkeys(caution)):
                st.caption("• " + message)
        if routine:
            st.caption("Additional methodology details are included in the downloadable report.")
    _report_download(r,"v9_portfolio_exposure.json")

def _income_table(r: dict):
    rows = []
    currency = r.get("base_currency", "USD")
    for holding in income_ranked(r.get("holdings") or []):
        status = holding.get("status")
        rows.append({
            "Investment": holding.get("symbol"),
            "Annual income": display_money(holding.get("estimated_annual_gross_income"), currency),
            "Dividend yield": pct(holding.get("trailing_distribution_yield")),
            "Dividend growth, historical": pct(holding.get("historical_trailing_year_distribution_growth")),
            "Payments recorded, past year": holding.get("observed_trailing_distribution_events"),
            "Data status": "Available" if status == "evaluated" else str(status or "Unavailable").replace("_", " ").title(),
        })
    if rows:
        st.caption("Sorted by estimated annual dividend income, highest first. Amounts with unavailable data appear last.")
        display_table(pd.DataFrame(rows), hide_index=True, width="stretch",
                     alt="Dividend estimates and historical payout information by investment, sorted by annual income")
        st.caption("Historical payout growth describes past distributions. It does not predict next year's payments.")


def _drip_chart(r: dict):
    sim = r.get("simulation") or {}
    annual = sim.get("annual_results") or []
    points = []
    for point in annual:
        cash = (point.get("cash") or {}).get("total_wealth_including_withdrawn_cash")
        reinvested = (point.get("drip") or {}).get("total_wealth")
        if (isinstance(cash, (int, float)) and isinstance(reinvested, (int, float))
                and point.get("year") is not None):
            points.append({"Year": point["year"], "Withdraw and keep cash": float(cash),
                           "Reinvest dividends (DRIP)": float(reinvested)})
    if not points:
        st.info("A projected comparison is unavailable with these inputs.")
        return
    ccy = r.get("base_currency", "USD")
    st.subheader("Taking dividends as cash vs. reinvesting them")
    st.caption("Both lines include the money you own. The cash option counts withdrawn dividends, even though they leave the investment account.")
    long = pd.DataFrame(points).melt(id_vars="Year", var_name="Choice", value_name="Total wealth")
    fig = px.line(long, x="Year", y="Total wealth", color="Choice", markers=True,
                  color_discrete_map={"Withdraw and keep cash": "#F5A45B",
                                      "Reinvest dividends (DRIP)": "#4BD4E5"},
                  line_dash="Choice", line_dash_map={"Withdraw and keep cash": "dash",
                      "Reinvest dividends (DRIP)": "solid"},
                  symbol="Choice", symbol_map={"Withdraw and keep cash": "diamond",
                      "Reinvest dividends (DRIP)": "circle"})
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=390,
                      legend=dict(orientation="h", y=1.12),
                      margin=dict(l=10, r=25, t=30, b=10),
                      yaxis_title=f"Total wealth ({ccy})", xaxis_title="Years")
    for trace in fig.data:
        trace.update(line=dict(width=3.5), marker=dict(size=9))
        trace.hovertemplate = f"{trace.name}<br>Year %{{x}}: $%{{y:,.2f}} {ccy}<extra></extra>"
    fig.update_layout(hoverlabel=dict(bgcolor="#19283F", font=dict(color="#FFFFFF", size=15),
                                      bordercolor="#8DA3BF"))
    axis = simulation_axis_range([p["Withdraw and keep cash"] for p in points],
                                 [p["Reinvest dividends (DRIP)"] for p in points])
    if axis:
        fig.update_yaxes(range=list(axis), tickformat=",.0f")
    tune_plotly(fig, st.session_state.get("v10_theme"))
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("The vertical scale is zoomed to show small differences, so it does not start at zero.")
    cash_values = [p["Withdraw and keep cash"] for p in points]
    drip_values = [p["Reinvest dividends (DRIP)"] for p in points]
    if close_simulation_lines(cash_values, drip_values):
        # This second view makes small differences legible without exaggerating
        # the wealth chart's scale or suggesting guaranteed reinvestment gains.
        difference_df = pd.DataFrame({
            "Year": [p["Year"] for p in points],
            "Reinvesting minus taking cash": [b-a for a,b in zip(cash_values,drip_values)],
        })
        st.markdown("**A closer look at the difference**")
        st.caption("These totals are close together, so this chart shows the gap separately. Values above zero favor reinvesting under these assumptions; values below zero favor cash.")
        difference_chart = px.bar(difference_df, x="Year", y="Reinvesting minus taking cash",
                                  color_discrete_sequence=["#78B8F5"])
        difference_chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                       plot_bgcolor="rgba(0,0,0,0)", height=225,
                                       margin=dict(l=10, r=25, t=10, b=10),
                                       yaxis_title=f"Difference ({ccy})")
        difference_chart.update_traces(hovertemplate=f"Year %{{x}}: $%{{y:,.2f}} {ccy}<extra></extra>")
        difference_chart.update_layout(hoverlabel=dict(bgcolor="#19283F",
            font=dict(color="#FFFFFF", size=15), bordercolor="#8DA3BF"))
        tune_plotly(difference_chart, st.session_state.get("v10_theme"))
        st.plotly_chart(difference_chart, width="stretch", theme=None)
    cash_end = (sim.get("end_comparison") or {}).get("cash") or {}
    drip_end = (sim.get("end_comparison") or {}).get("drip") or {}
    a, b, c = st.columns(3)
    withdraw_value = cash_end.get("total_wealth_including_withdrawn_cash")
    drip_value = drip_end.get("total_wealth")
    a.metric("Total wealth if you take cash", display_money(withdraw_value, ccy))
    b.metric("Total wealth if you reinvest", display_money(drip_value, ccy))
    if isinstance(withdraw_value, (int, float)) and isinstance(drip_value, (int, float)):
        difference = drip_value - withdraw_value
        c.metric("DRIP difference", display_money(difference, ccy),
                 help="Reinvested scenario minus cash-withdrawal scenario. A negative number would favor cash under these assumptions.")
    else:
        c.metric("DRIP difference", "Unavailable")
    st.caption("The model assumes fractional shares and no reinvestment fees. Cash withdrawn earns no further interest in this comparison.")


def page_dividends():
    _header("Income & Dividends", "See what your investments distribute and compare taking payments with reinvesting them.")
    if _current_data_warning():
        return
    r = _current_report("dividends")
    if r:
        _income_compact(r)
        st.divider()
    st.subheader("Explore future dividend scenarios")
    st.caption("Adjust the assumptions below and compare outcomes. These are examples, not promised returns or dividend payments.")
    a, b, c = st.columns(3)
    with a:
        years = st.selectbox("Years to compare", [5, 10, 15, 20, 25, 30, 40], index=1,
                             help="How long the illustration runs. Longer periods magnify any assumptions.")
    with b:
        price = st.number_input("Annual share-price growth (%)", value=4.0, min_value=-30.0,
                                max_value=30.0, step=1.0, format="%.2f",
                                help="Your assumed yearly change in the market prices of the investments. For example, 4% means each share price rises by 4% per year; it is not based on a forecast.")
    with c:
        growth = st.number_input("Annual dividend-per-share growth (%)", value=3.0, min_value=-30.0,
                                 max_value=30.0, step=1.0, format="%.2f",
                                 help="Your assumed yearly change in dividend paid per share. For example, 3% means next year's dividend per share is 3% higher. Payouts can fall or stop.")
    with st.expander("Optional tax or withholding adjustment", expanded=False):
        taxed = st.toggle("Reduce modeled dividends for estimated taxes or withholding", value=False)
        tax = (st.number_input("Your effective reduction on dividends (%)", 0.0, 100.0, 15.0, 1.0,
                               disabled=not taxed, help="Only affects distributions in this comparison. It does not calculate your personal tax obligations.")
               if taxed else 0.0)
    expected_dividend_args = ["--years", str(years), "--price-growth", str(price),
                              "--dividend-growth", str(growth), "--tax-drag", str(tax)]
    if st.button("Compare dividend income and reinvestment", type="primary", width="stretch"):
        args = expected_dividend_args
        with st.spinner("Calculating the dividend scenarios..."):
            _run("dividends", args)
        r = _current_report("dividends")
    if not r:
        st.caption("Choose your assumptions and select Compare. No cash flows are invented when the source data is missing.")
        return
    saved_dividend_args = (st.session_state.reports.get("dividends") or {}).get("args") or []
    if saved_dividend_args != expected_dividend_args:
        st.caption("The results below use the previous simulation settings. Select Compare to apply your new choices.")
    st.divider()
    _drip_chart(r)
    st.divider()
    st.subheader("Which investments generate your income?")
    _income_table(r)
    # Complete data-quality notices and tax assumptions remain in the downloadable original report.
    _report_download(r, "v9_portfolio_dividends.json")


def page_valuation():
    _header("Stock Valuation", "Understand company valuations with plain-language financial ratios and scenario estimates.")
    if _current_data_warning():
        return
    stocks = [h for h in st.session_state.holdings if not is_known_fund(h["symbol"], str(h.get("name") or ""))]
    if not stocks:
        st.info("This tool analyzes individual company shares. Your current portfolio appears to contain only funds or other non-company assets. Add a company stock in Portfolio Builder to use Stock Valuation.")
        return
    choices = [h["symbol"] for h in stocks]
    symbol = st.selectbox("Choose a company stock", options=choices,
                          format_func=lambda sym: next((f"{sym} | {h.get('name') or sym}" for h in stocks if h["symbol"] == sym), sym))
    st.caption("ETFs and gold funds are not companies with their own operating earnings, so they are not eligible for this analysis.")
    key = "fundamentals_" + symbol
    if st.button("Analyze company", type="primary"):
        try:
            with st.spinner("Retrieving company accounts and estimated valuation scenarios..."):
                report = run_fundamentals(symbol)
            _save_report(key, report, [])
        except (BackendError, PortfolioError) as exc:
            details = str(exc)
            if any(term in details.lower() for term in ("require an individual equity", "not an etf", "not an individual equity")):
                st.info("Select an individual company stock. ETFs and funds do not have the company financial statements required for this valuation.")
            else:
                st.warning("Company financial data could not be verified for this ticker. Try another company or retry later.")
                with st.expander("More details"):
                    st.code(details)
    r = _current_report(key)
    if not r:
        st.caption("Select a company and choose Analyze to view its reported valuation ratios and scenarios.")
        return
    currency = r.get("statement_currency") or "USD"
    multiple_data = r.get("multiples") or {}
    st.divider()
    st.subheader("How is the company priced?")
    st.caption("These three ratios show what investors pay relative to different kinds of earnings or cash flow. There is no universal 'good' or 'bad' number: compare with similar companies and the business's growth and risks.")
    columns = st.columns(3)
    for column, metric in zip(columns, MULTIPLE_INFO):
        name, definition = MULTIPLE_INFO[metric]
        value = multiple_data.get(metric)
        with column:
            with st.container(border=True, key=f"valuation_multiple_{metric}"):
                st.metric(name, f"{value:.2f}×" if isinstance(value, (int, float)) else "Unavailable",
                          help=definition)
                st.caption(multiple_reading(metric, value))
    st.caption("EBITDA stands for **earnings before interest, taxes, depreciation and amortization**. It is a rough operating-profit measure, not cash left over after all expenses. For reliable comparisons, use similar companies from the same industry.")
    st.divider()
    from portfolio_risk.v8_fundamentals import simple_dcf_limitation
    model_warning = simple_dcf_limitation(symbol, r.get("sector"), r.get("industry"))
    if model_warning:
        st.subheader(f"Can we estimate a fair value for {symbol} from cash flow alone?")
        st.caption("Not reliably with this simplified approach. The stock price is shown for context, but no DCF fair-value bars are displayed.")
    else:
        st.subheader(f"What is one {symbol} share worth under different assumptions?")
        st.caption("Three sets of assumed business growth and required returns produce different values. They are sensitivity examples, not price targets or forecasts.")
    if model_warning:
        st.info(model_warning + " The general cash-flow model is not a reliable standalone fair-value estimate here. Reported company ratios above remain visible.")
    data = [] if model_warning else valuation_scenarios(r)
    price = (r.get("reported") or {}).get("price")
    if isinstance(price, (int, float)):
        st.metric(f"{symbol} market price used for comparison", display_money(price, currency),
                  help="Provider-supplied price at the financial-data snapshot. It may be delayed. This is not a model estimate or a live quote.")
        if not model_warning:
            st.caption(f"The {symbol} price is an observed comparison point; the bars below are model-implied estimates per share under different assumptions, not expected future trading prices.")
    if data:
        df = pd.DataFrame(data)
        fig = px.bar(df, x="Scenario", y="Estimated value per share", text="Estimated value per share",
                     color="Scenario", color_discrete_sequence=["#8599B0", "#367BC0", "#D9A35C"])
        fig.update_traces(texttemplate=f"$%{{text:,.2f}} {currency}", textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}<br>Estimated value: %{y:,.2f}<extra></extra>")
        if isinstance(price, (int, float)):
            fig.add_hline(y=price, line_dash="dot", line_color="#D8DEE8",
                          annotation_text=f"{symbol} market price: ${price:,.2f} {currency}",
                          annotation_position="top left")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=380, showlegend=False,
                          margin=dict(l=5,r=20,t=30,b=15), yaxis_title=f"Value per share ({currency})", xaxis_title="Assumptions",
                          hoverlabel=dict(bgcolor="#132238", font=dict(color="#FFFFFF", size=15),
                                          bordercolor="#B3C3D9"))
        tune_plotly(fig, st.session_state.get("v10_theme"))
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("Cautious assumes slower cash-flow growth and a higher required return. Optimistic assumes faster growth and a lower required return. The middle case lies between them, but is not necessarily most likely.")
        middle = next((item for item in data if item["scenario_key"] == "base"), None)
        if middle and isinstance(price, (int, float)):
            estimate = middle["Estimated value per share"]
            comparison = market_valuation_gap(price, estimate)
            if comparison:
                st.markdown(f"**{symbol}: middle scenario vs. market price**")
                st.write(f"The middle model case is ${estimate:,.2f} {currency} per share, "
                         f"{abs(comparison['percent']):.1f}% {comparison['direction']} the "
                         f"provider's market price of ${price:,.2f} {currency}. "
                         "This is a difference between an assumption-based estimate and a market quote, **not a price forecast**.")
                if comparison["large"]:
                    st.caption("Why the gap can be large: Market prices reflect expectations for future growth, business quality, risk and investor sentiment. This simplified cash-flow model depends heavily on its inputs, and a gap is not a forecast or evidence the share price must move toward the estimate. Check the assumptions below before interpreting it.")
        st.caption("Peer multiples are not currently verified in this report. We cannot responsibly label a stock 'cheap' or 'expensive' relative to competitors without comparable peer data.")
        with st.expander("What assumptions drive these estimates?"):
            for case in data:
                params = (r.get("dcf") or {}).get("scenarios", {}).get(case["scenario_key"], {})
                st.write(f"**{case['Scenario']}**: "
                         f"cash-flow growth {pct(params.get('growth'))}, "
                         f"required return {pct(params.get('discount_rate'))}, "
                         f"long-term growth {pct(params.get('terminal_growth'))}.")
    else:
        if not model_warning:
            st.info((r.get("dcf") or {}).get("reason") or "Estimated values could not be calculated from the available financial statements. We won't invent cash-flow or valuation inputs.")
    _report_download(r, f"v9_{symbol}_fundamentals.json")


def _strategy_test_chart(rows: list[dict], count: int, *, chart_key: str = "v914_legacy_test_counts"):
    """Plot backend evidence only; distinct visual groups are explained explicitly."""
    if not rows:
        return
    chart_df = pd.DataFrame(rows)
    fig = go.Figure()
    for label, color in (("Better", "#48B99E"), ("Similar", "#92A3B8"),
                         ("Worse", "#EA777F"), ("Not assessed", "#677385")):
        fig.add_trace(go.Bar(name=label, y=chart_df["Goal"], x=chart_df[label],
                             orientation="h", marker_color=color,
                             text=[str(v) if v else "" for v in chart_df[label]], textposition="inside",
                             hovertemplate="%{y}: %{x} tests<extra>" + label + "</extra>"))
    fig.update_layout(barmode="stack", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=max(200, 115 + len(chart_df) * 66),
                      margin=dict(l=10, r=10, t=35, b=5),
                      legend=dict(orientation="h", y=1.24),
                      xaxis_title="Historical tests", yaxis_title=None)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(1, count)], dtick=max(1, count // 4))
    fig.update_layout(hoverlabel=dict(bgcolor="#19283F", font=dict(color="#FFFFFF", size=14)))
    tune_plotly(fig, st.session_state.get("v10_theme"))
    st.plotly_chart(fig, width="stretch", key=chart_key, theme=None)


def _strategy_allocation_chart(rows: list[dict], *, chart_key: str = "v914_legacy_allocation"):
    """Compare original versus EXACT latest test weights, not an invented optimum."""
    displayed = rows[:min(10, len(rows))]
    if not displayed:
        return
    labels = [row["Ticker"] for row in displayed]
    fig = go.Figure()
    for label, key, color in (("Current portfolio", "Original (%)", "#91A4C0"),
                              ("Latest tested allocation", "Tested (%)", "#4AB99D")):
        values = [row[key] for row in displayed]
        fig.add_trace(go.Bar(name=label, y=labels, x=values, orientation="h",
                             marker_color=color, textfont=dict(color="#F2F7FF"),
                             text=[f"{v:.2f}%" for v in values], textposition="outside",
                             hovertemplate=label + "<br>%{y}: %{x:.2f}%<extra></extra>"))
    fig.update_layout(barmode="group", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      height=max(270, len(displayed)*56 + 100),
                      margin=dict(l=15, r=70, t=45, b=25),
                      legend=dict(orientation="h", y=1.15),
                      xaxis_title="Portfolio allocation (%)", yaxis_title=None,
                      hoverlabel=dict(bgcolor="#19283F", font=dict(color="#FFFFFF", size=14)))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(5, max(max(row["Original (%)"], row["Tested (%)"]) for row in displayed)*1.17)])
    tune_plotly(fig, st.session_state.get("v10_theme"))
    st.plotly_chart(fig, width="stretch", key=chart_key, theme=None)


def _strategy_allocation_example(report: dict):
    """Make allocation changes actionable to inspect, never automatic trade advice."""
    example = latest_candidate_example(report)
    st.subheader("What weights did the model actually test?")
    if example["status"] != "evaluated":
        st.caption(example.get("reason") or "The last tested allocation is unavailable.")
        return
    st.write("**Example from the most recent completed historical test**")
    st.caption(f"Selected approach: {example['selected_label']}. "
               f"Chosen using data through {example['training_end']}; checked on later data through "
               f"{example['holdout_end']}. This is **one historical example**, not a "
               "validated recommendation for today's allocation.")
    rows = example["rows"]
    changes = [r for r in rows if abs(r["Change (percentage points)"]) >= .005]
    if not changes:
        st.info("The latest test selected your existing weights. No allocation changes were proposed in this window.")
    else:
        biggest = changes[0]
        action = "Reduce" if biggest["Change (percentage points)"] < 0 else "Increase"
        st.markdown(f"**Largest proposed change:** {action.lower()} **{biggest['Ticker']}** from "
                    f"**{biggest['Original (%)']:.2f}% to {biggest['Tested (%)']:.2f}%** "
                    f"({biggest['Change (percentage points)']:+.2f} percentage points).")
        _strategy_allocation_chart(rows)
        st.caption(f"Moving from today's weights to this example changes approximately "
                   f"{example['allocation_turnover_pct']:.2f}% of portfolio weight one way, "
                   "before fees and market movement. No transaction is performed.")
    # Three related details live behind TABS rather than three stacked drop-downs.
    weights_tab, outcome_tab, history_tab = st.tabs([
        "Exact weight changes", "How this example performed", "Other historical tests"
    ])
    with weights_tab:
        full = [{"Investment": r["Ticker"],
                 "Current": f"{r['Original (%)']:.2f}%",
                 "Tested": f"{r['Tested (%)']:.2f}%",
                 "Change": f"{r['Change (percentage points)']:+.2f} pp"} for r in rows]
        display_table(pd.DataFrame(full), hide_index=True, width="stretch")
    with outcome_tab:
        st.caption(f"One later historical test ending {example['holdout_end']}, "
                   "not a prediction or the average across tests.")
        compared = [{"Measure": r["Metric"],
                     "Current portfolio": format_metric_value(r["Key"], r["Your portfolio"]),
                     "Tested allocation": format_metric_value(r["Key"], r["Tested allocation"]),
                     "Result": r["Result"]} for r in example["measures"]]
        if compared:
            display_table(pd.DataFrame(compared), hide_index=True, width="stretch")
        else:
            st.caption("This historical report did not include a complete metric comparison.")
    with history_tab:
        st.caption(f"Each of the {example['fold_count']} tests selected its own allocation. "
                   "The middle value below is a median of weights, not another backtested portfolio.")
        if example["historical_ranges"]:
            historical = [{"Investment": r["Ticker"],
                           "Lowest tested": f"{r['Lowest tested (%)']:.2f}%",
                           "Median tested": f"{r['Median tested (%)']:.2f}%",
                           "Highest tested": f"{r['Highest tested (%)']:.2f}%"}
                          for r in example["historical_ranges"]]
            display_table(pd.DataFrame(historical), hide_index=True, width="stretch")


def _strategy_plain_summary(report: dict):
    """Put financial trade-offs above the historical charts and nested evidence."""
    st.subheader("What would this change improve, and what might it cost?")
    evidence = strategy_tradeoff(report)
    if evidence["status"] != "evaluated":
        st.caption("The latest tested portfolio could not be compared in enough detail to show the trade-off.")
        return
    st.caption(f"These are ANNUALIZED results in the most recent historical test ending "
               f"{evidence['latest_end']}, not forecasts of what you will earn next year. "
               "The candidate in other tests may have used different weights.")
    measures = {item["key"]: item for item in evidence["measures"]}
    examples = [("Annual growth", "cagr"), ("Portfolio volatility", "volatility"),
                ("Largest decline", "drawdown")]
    cols = st.columns(3)
    for col, (name, key) in zip(cols, examples):
        measure = measures.get(key)
        if measure is None:
            continue
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.markdown(f"**{measure['original']:.2%} → {measure['tested']:.2%}**")
                st.badge(measure["result"], color={"Better": "green", "Worse": "red",
                                                        "Similar": "gray"}.get(measure["result"], "gray"))
                st.caption("Current mix to latest tested mix")
    if evidence.get("interpretation"):
        st.markdown("**Main trade-off:** " + evidence["interpretation"])
    if evidence["positives"]:
        st.markdown("**What helped in this example:** " + ", ".join(evidence["positives"]) + ".")
    if evidence["negatives"]:
        st.markdown("**What got worse in this example:** " + ", ".join(evidence["negatives"]) + ".")
    st.markdown("**Across all historical tests:**")
    important = [row for row in evidence["all_fold_counts"]
                 if row["key"] in ("cagr", "volatility", "drawdown")]
    if important:
        st.write(" · ".join(f"**{row['label']}:** better in {row['better']}/{row['total']}, "
                            f"worse in {row['worse']}/{row['total']}"
                            for row in important))
    st.info(status_for_summary(report))


def _strategy_run_all(base_args: list[str]) -> None:
    """Run six REAL independent selection objectives, not six renamed verdicts.

    The historical work may be substantial. Reuse each previously completed
    exact calculation, show real progress and do not fabricate missing paths.
    """
    _resolve_auto_quotes()
    results = {}
    errors = {}
    with st.status("Analyzing six portfolio strategies using historical market data", expanded=True) as status:
        st.write("This can take several minutes. Each strategy runs a separate historical analysis. You can leave this page open while the results are calculated.")
        progress = st.progress(0, text="Preparing six separate strategy tests")
        for index, (goal, short, _metric, _explanation) in enumerate(PATHS, start=1):
            st.write(f"{index} of {len(PATHS)}: {short}. Testing allocations against later historical periods...")
            args = strategy_report_args(base_args, STRATEGY_GOALS[goal][0])
            try:
                if not _run_strategy_cached(args):
                    errors[goal] = "The historical analysis could not be completed."
                    continue
                results[goal] = {
                    "report": st.session_state.reports["strategy"]["report"],
                    "args": list(args),
                }
            except (BackendError, PortfolioError, ValueError) as exc:
                errors[goal] = str(exc)
            finally:
                progress.progress(index / len(PATHS), text=f"Checked {index} of {len(PATHS)}: {short}")
        progress.progress(1.0, text=f"Finished checking all six paths. {len(results)} produced a report.")
        status.update(label=f"Historical strategy tests complete: {len(results)} of {len(PATHS)} profiles available",
                      state="complete" if len(results) == len(PATHS) else "error",
                      expanded=False)
    st.session_state.strategy_profiles = {
        "signature": _portfolio_signature(), "base_args": list(base_args),
        "results": results, "errors": errors,
    }


def _strategy_profile_panel(goal: str, data: dict):
    """Lead with allocation and decision, then keep detailed audit evidence quiet."""
    summary = path_summary(data["report"], goal)
    example = summary["example"]
    st.subheader(summary["short_name"])
    st.caption(summary["detail"])
    if example.get("status") != "evaluated":
        st.warning(summary["explanation"])
        return
    from .v913_strategy_paths import path_signal
    signal, meaning = path_signal(summary)
    if signal == "★":
        st.success("★ VALIDATED HISTORICAL EVIDENCE: " + summary["explanation"])
    elif signal == "○":
        st.info("○ MIXED EVIDENCE: " + summary["explanation"])
    else:
        st.warning("✕ NOT SUPPORTED: " + summary["explanation"])

    st.markdown("#### Proposed allocation from the latest historical test")
    st.caption(f"Illustrative target weights selected using data through {example['training_end']}, "
               f"then checked through {example['holdout_end']}. Earlier tests selected other weights. "
               "This is not a live trading recommendation.")
    rows = sorted(example["rows"], key=lambda r: (-r["Tested (%)"], r["Ticker"]))
    allocation = [{"Investment": row["Ticker"],
                   "Your portfolio": f"{row['Original (%)']:.2f}%",
                   "Tested weight": f"{row['Tested (%)']:.2f}%",
                   "Change": f"{row['Change (percentage points)']:+.2f} pp"}
                  for row in rows]
    display_table(pd.DataFrame(allocation), hide_index=True, width="stretch",
                 height=min(390, 65 + len(allocation) * 36))
    if rows:
        added = [r for r in rows if r["Change (percentage points)"] > .01]
        reduced = [r for r in rows if r["Change (percentage points)"] < -.01]
        highlights = []
        if added:
            row = max(added, key=lambda r: r["Change (percentage points)"])
            highlights.append(f"Increase {row['Ticker']} from {row['Original (%)']:.2f}% to {row['Tested (%)']:.2f}%")
        if reduced:
            row = min(reduced, key=lambda r: r["Change (percentage points)"])
            highlights.append(f"reduce {row['Ticker']} from {row['Original (%)']:.2f}% to {row['Tested (%)']:.2f}%")
        if highlights:
            st.write("**Largest weight changes:** " + "; ".join(highlights) + ".")

    st.markdown("#### Why these weights, and what did they change?")
    st.write(f"The test selected **{example['selected_label'].lower()}** from candidate allocations "
             f"using earlier prices, aiming to {summary['detail'][0].lower() + summary['detail'][1:]}. "
             "It did not select weights using the later period's outcomes.")
    trade = summary["trade"]
    measures = {r["key"]: r for r in trade.get("measures", [])}
    metrics = [("Annual growth", "cagr"), ("Fluctuations", "volatility"),
               ("Worst decline", "drawdown")]
    columns = st.columns(3)
    for col, (label, key) in zip(columns, metrics):
        metric = measures.get(key)
        with col:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                if metric is not None:
                    st.write(f"{metric['original']:.2%} → **{metric['tested']:.2%}**")
                    st.caption({"Better": "Improved", "Worse": "Got worse", "Similar": "About the same"}
                               .get(metric["result"], "Not assessed"))
                else:
                    st.caption("Not available")
    audit = ((data["report"].get("rolling_robustness") or {})
             .get("latest_asset_audit") or {})
    if audit:
        with st.expander("See the returns the optimizer actually saw (data check)"):
            st.write("The **earlier period** determined which weights were chosen. The **later period** was used only to evaluate them. This shows why a stock that looked weak earlier could outperform after the decision was made.")
            evidence = []
            for ticker, values in audit.items():
                evidence.append({"Investment": ticker,
                                 "Earlier annualized growth": f"{values['training_cagr']:.2%}",
                                 "Later period total return": f"{values['holdout_total_return']:.2%}",
                                 "Earlier volatility": f"{values['training_volatility']:.2%}"})
            evidence.sort(key=lambda v: next((i for i, r in enumerate(rows) if r["Ticker"] == v["Investment"]), 999))
            display_table(pd.DataFrame(evidence), hide_index=True, width="stretch")
            st.caption("These numbers use the same date-aligned, reporting-currency adjusted returns supplied to Portfolio Optimizer. Annualizing a short test period can be misleading, so later returns are shown as the actual gain or loss over that period. Historical returns do not predict future winners.")

    st.markdown("**All six measures in this historical example**")
    status_label = {"Better": "✓ Improved", "Similar": "○ Similar", "Worse": "✕ Worsened"}
    status_rank = {"Better": 0, "Similar": 1, "Worse": 2}
    comparisons = sorted(trade.get("measures", []), key=lambda item: (status_rank.get(item["result"], 3), item["label"]))
    if comparisons:
        comparison_rows = [{"Measure": item["label"],
                            "Current portfolio": format_metric_value(item["key"], item["original"]),
                            "Tested allocation": format_metric_value(item["key"], item["tested"]),
                            "Result": status_label.get(item["result"], "Not assessed")}
                           for item in comparisons]
        display_table(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
        st.caption("Improved and worsened are relative to your current portfolio in this ONE later historical test. Lower is better for volatility, drawdown and worst-day losses; higher is better for growth and risk-adjusted returns.")
    if trade.get("interpretation"):
        st.write("**Main trade-off:** " + trade["interpretation"])
    if trade.get("positives"):
        st.caption("Better in the latest test: " + ", ".join(trade["positives"]))
    if trade.get("negatives"):
        st.caption("Worse in the latest test: " + ", ".join(trade["negatives"]))

    review = summary["review"]
    item = summary["primary_row"]
    st.markdown("#### Did the improvement hold up over time?")
    if item:
        import math
        total = int(review["count"])
        better = int(item["Better"])
        similar = int(item["Similar"])
        worse = int(item["Worse"])
        needed = math.ceil(0.75 * total)
        st.write(f"**{item['Goal']}:** Improved in **{better} of {total} tests ({better / total:.0%})**, "
                 f"was similar in {similar}, and worsened in {worse}.")
        st.caption("A test is one separate later historical period. For EACH test, the model chooses "
                   "a portfolio using only earlier data, then checks that choice against your original "
                   "portfolio in the later period. It does not test the exact same weights 30 times.")
        if better >= needed and worse == 0:
            st.success(f"This measure met the historical consistency rule: at least {needed} improved tests and no meaningful worsening.")
        else:
            st.caption(f"The strict consistency rule needs at least {needed} improved tests out of {total} and zero meaningfully worse tests. "
                       f"This measure had {better} improved and {worse} worse, so it did not meet both conditions.")
    spy = summary["spy"]
    if spy.get("status") == "evaluated":
        primary = spy["metrics"][summary["primary"]]
        growth = spy["metrics"]["cagr"]
        st.write(f"**Compared with SPY on the same dates:** This strategy's main measure matched or "
                 f"exceeded SPY in **{primary['matched']} of {primary['total']}** test periods. "
                 f"Annualized growth matched or exceeded SPY in **{growth['matched']}** periods. "
                 "These are observations, not predicted returns.")
        last_growth = growth["last"]
        if isinstance(last_growth.get("candidate"), (int, float)) and isinstance(last_growth.get("spy"), (int, float)):
            st.caption(f"Latest period's annualized growth: tested portfolio "
                       f"{last_growth['candidate']:.2%}; SPY {last_growth['spy']:.2%}. "
                       "SPY is the same-currency, same-date ETF market reference.")
        latest = primary["last"]
        if isinstance(latest.get("candidate"), (int, float)) and isinstance(latest.get("spy"), (int, float)):
            st.caption(f"Latest period, main measure vs SPY: "
                       f"{format_metric_value(summary['primary'], latest['candidate'])} vs "
                       f"{format_metric_value(summary['primary'], latest['spy'])}. "
                       "Both use the same historical trading dates and reporting currency.")
    else:
        st.caption("Same-window SPY comparison is unavailable for this report, so no claim "
                   "of market outperformance is made.")

    with st.expander("Show all six measures and additional historical evidence"):
        _strategy_test_chart(review["rows"], review["count"], chart_key=f"v914_{summary['primary']}_validation")
        st.caption("Green means better, gray means a small difference and red means worse "
                   "than your existing portfolio in that period. Each period chose its own weights.")
        _strategy_allocation_chart(example["rows"], chart_key=f"v914_{summary['primary']}_allocation")
        if example.get("historical_ranges"):
            st.markdown("**How the chosen weights differed across historical periods**")
            display_table(pd.DataFrame(example["historical_ranges"]), hide_index=True, width="stretch")


def _strategy_path_tabs(profiles: dict):
    """Six goal-specific views. Each one requires its OWN optimization report."""
    st.subheader("Explore six portfolio strategies")
    st.caption("Each tab uses a separately optimized and historically tested portfolio. "
               "Pick a goal to see the proposed holdings first, then the trade-offs and evidence. "
               "Different goals can sometimes choose the same holdings because the historical "
               "optimizer evaluates a limited set of candidate portfolios. These are not automatic trades.")
    results = profiles.get("results", {})
    # A live provider can update between the six subprocess runs. Never imply
    # cross-path comparability when their observed histories differ.
    histories = {
        (data["report"].get("first_date"), data["report"].get("last_date"),
         data["report"].get("shared_close_dates"), data["report"].get("base_currency"))
        for data in results.values()
    }
    if len(histories) > 1:
        st.warning("Some strategy paths used different market-data cutoffs. Compare each "
                   "path with its own original portfolio and SPY reference, not directly "
                   "with other paths until all paths are run on identical price histories.")
    names = [p[0] for p in PATHS]
    summaries = {goal: path_summary(results[goal]["report"], goal) for goal in names if goal in results}
    from .v913_strategy_paths import path_signal
    signal_words = {"★": "VALIDATED", "○": "MIXED", "✕": "NOT SUPPORTED"}
    st.markdown("**Historical evidence by strategy**")
    legend = st.columns(3)
    with legend[0]:
        st.caption("★ **Validated**: passed strict historical and SPY checks")
    with legend[1]:
        st.caption("○ **Mixed**: evidence is incomplete or inconsistent")
    with legend[2]:
        st.caption("✕ **Not supported**: no convincing improvement for this goal")
    st.caption("Historical research only. A label does not constitute investment advice.")
    for start in range(0, len(PATHS), 3):
        columns = st.columns(3)
        for col, (goal, short, primary, _detail) in zip(columns, PATHS[start:start + 3]):
            with col:
                with st.container(border=True):
                    s = summaries.get(goal)
                    icon, meaning = path_signal(s) if s else ("○", "Not evaluated")
                    st.markdown(f"**{short}**")
                    status_title = signal_words[icon]
                    st.badge(f"{icon} {status_title}", color={"★": "green", "○": "gray", "✕": "red"}[icon])
                    row = s.get("primary_row") if s else None
                    count = (s.get("review") or {}).get("count") if s else None
                    if row and count:
                        st.caption(f"Goal improved in {row['Better']} of {count} historical tests")
                    else:
                        st.caption("No complete historical comparison")
    st.caption("Select a strategy below to see its proposed weights, trade-offs and full explanation.")
    tabs = st.tabs([f"{path_signal(summaries[goal])[0]} {short}"
                    if goal in summaries else f"○ {short}"
                    for goal, short, *_ in PATHS])
    for tab, goal in zip(tabs, names):
        with tab:
            data = results.get(goal)
            if data:
                _strategy_profile_panel(goal, data)
            else:
                st.warning("This strategy could not be calculated: " +
                           profiles.get("errors", {}).get(goal, "No result was provided."))
    st.caption("The strict historical approval compares performance against your original portfolio "
               "and SPY on the same later trading dates. A result can improve one goal while making "
               "another worse. No single latest-test allocation is proven optimal for the future.")


def _strategy_diagnostics(report: dict, tested_goal: str | None = None):
    rob = report.get("rolling_robustness") or {}
    review = all_goal_review(report)
    if review["status"] != "evaluated":
        st.subheader("What did the strategy tests find?")
        st.info("There were not enough comparable historical tests to evaluate this approach.")
        if rob.get("reason"):
            st.caption(str(rob["reason"]))
        return
    count = review["count"]
    _strategy_allocation_example(report)
    st.divider()
    st.subheader("What improved and what got worse?")
    st.caption(f"In each of {count} historical test windows, one allocation was chosen using "
               "earlier data, then compared on six measures using later prices. "
               "These are not six different portfolios in any one window. "
               "The model did not keep the same allocation in all windows.")
    focus = goal_from_priorities(tested_goal)
    st.markdown(f"**Selection focus:** {focus}. **Evaluation:** all six measures.")

    findings = review["findings"]
    for offset in range(0, len(findings), 3):
        columns = st.columns(3)
        for column, item in zip(columns, findings[offset:offset+3]):
            with column:
                with st.container(border=True):
                    status, color = status_group(item)
                    st.markdown(f"**{item['goal']}**")
                    # Text statuses are provided in addition to color for accessibility.
                    st.caption(status)
                    matched = next((row for row in review["rows"] if row["Goal"] == item["goal"]), None)
                    if matched:
                        st.write(f"**{matched['Better']} better**  ·  {matched['Similar']} similar  ·  {matched['Worse']} worse")
                        st.caption(f"Across {count} separate historical tests")
    _strategy_test_chart(review["rows"], count)
    st.caption("Green means meaningfully better than the current portfolio, gray means "
               "a small difference and red means meaningfully worse, in that historical test. "
               "These observations do not predict future performance.")
    if review["passed"]:
        st.success("The tested selection method passed its required historical checks. "
                   "This does not establish one optimal portfolio or guarantee future returns.")
    elif review["strong"]:
        st.info("Some measures improved consistently, but the selection method did not "
                "pass every required test. Explore the exact weights above to understand "
                "the observed trade-offs, not as an automatic trade recommendation.")
    else:
        st.info("No strategy passed the required historical checks. The results still "
                "show which measures improved and which worsened.")
    warning = tested_candidate_warning(report)
    if warning:
        st.caption(warning)
    with st.expander("What is required for an overall approval?"):
        st.write("The original risk rules require at least 75% of historical windows to show an "
                 "improvement **and zero windows with material worsening** for each measure used "
                 "to select candidates. Other measures are displayed for context, not silently "
                 "added to the approval requirements. This strict rule can reject some useful "
                 "trade-offs; it is not proof that your portfolio cannot be improved.")
        existing = validation_summary(report)
        if existing.get("status") != "unavailable":
            st.write(existing["description"])
            for row in [existing["primary"], *existing["supporting"]]:
                if row:
                    decision = metric_decision(row, count)
                    st.write(f"**{row['Goal']}:** {decision['message']}")
    with st.expander("See the individual historical tests"):
        folds = rob.get("folds") or []
        if folds:
            rows = []
            for i, fold in enumerate(folds, 1):
                row = {"Test": i, "Earlier data used through": fold.get("train_end"),
                       "Later data tested through": fold.get("holdout_end"),
                       "Allocation tested": fold.get("selected")}
                row.update({k.replace("_", " ").title(): v.get("sentiment", "unavailable").title()
                            for k, v in (fold.get("all_metric_verdicts") or fold.get("verdicts") or {}).items()})
                rows.append(row)
            display_table(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("Individual test windows were not available in this report.")
    diagnostics = report.get("actionable_diagnostics") or {}
    if diagnostics.get("concentration_findings"):
        with st.expander("Concentration areas worth reviewing"):
            for item in diagnostics["concentration_findings"][:3]:
                st.write(f"**{item.get('asset') or 'Investment'}:** "
                         f"{item.get('evidence') or 'Concentration deserves closer review.'}")


def _compare_scenarios(report: dict):
    outcomes=report.get("user_scenarios") or {}
    if not outcomes:
        return
    st.subheader("Your custom allocation compared with the original")
    st.caption("Your manual mix, original portfolio, SPY and RSP use the same historical dates and reporting currency. "
               "The two portfolios follow the selected rebalancing schedule and estimated trading costs. "
               "This is a retrospective comparison, not separate multi-window validation of your custom mix.")
    three = report.get('three_strategy_analysis') or {}
    growth = (three.get('strategies') or {}).get('growth') or {}
    # V8 report.portfolio is legacy daily-constant-mix data. Use the
    # selected-cadence original from the three-strategy analysis instead.
    original = growth.get('original_training_metrics') or report.get('portfolio') or {}
    benchmarks = growth.get('benchmarks_training_metrics') or report.get('benchmarks') or {}
    rows=[{"Portfolio":"Your starting portfolio", "CAGR":original.get("cagr"),
           "Volatility":original.get("annualized_volatility"),
           "Max drawdown":original.get("drawdown"), "Sharpe":original.get("sharpe")}]
    for name, outcome in outcomes.items():
        metric=outcome.get("metrics") or {}
        rows.append({"Portfolio":name.replace("GUI_WhatIf", "Your what-if allocation"),
                     "CAGR":metric.get("cagr"),
                     "Volatility":metric.get("annualized_volatility"),
                     "Max drawdown":metric.get("drawdown"), "Sharpe":metric.get("sharpe")})
    for symbol in ('SPY', 'RSP'):
        benchmark = benchmarks.get(symbol) or {}
        if benchmark:
            rows.append({'Portfolio':symbol, 'CAGR':benchmark.get('cagr'),
                         'Volatility':benchmark.get('annualized_volatility'),
                         'Max drawdown':benchmark.get('drawdown'), 'Sharpe':benchmark.get('sharpe')})
    display_table(pd.DataFrame([{"Portfolio":r["Portfolio"],"CAGR":pct(r["CAGR"]),
                               "Volatility":pct(r["Volatility"]),
                               "Max drawdown":pct(r["Max drawdown"]),
                               "Sharpe":f"{r['Sharpe']:.2f}" if r["Sharpe"] is not None else "Unavailable"}
                              for r in rows]),hide_index=True,width="stretch")
    with st.expander("Allocation assumptions used in this comparison"):
        for name, outcome in outcomes.items():
            st.write(name.replace("GUI_WhatIf", "Your what-if allocation"))
            _bar_from_weights(outcome.get("weights") or {},"Hypothetical weights")
    st.caption("This compares past performance using current hypothetical weights on historical prices; "
               "it is not proof that the alternate portfolio will perform better in the future.")


def page_strategy():
    """V9.18: one market-data run, three mathematically distinct objectives."""
    from .v918_three_views import show_three_results

    _header("Portfolio Optimizer", "Optimize your current holdings for growth, balance or capital protection, then compare each with SPY and RSP.")
    if _current_data_warning():
        return
    st.subheader("How should the target weights be maintained?")
    from .v920_presentation import REBALANCE_CHOICES, rebalance_explanation
    left, right = st.columns(2)
    with left:
        cadence = st.selectbox("Keep the selected allocation on target", list(REBALANCE_CHOICES),
                               index=0, key="v920_cadence",
                               help="Quarterly is the default. It resets to target weights every three months, not every day. Daily maintains nearly constant weights but entails much more trading; buy and hold permits drift.")
    with right:
        fee = st.number_input("Estimated trading cost per trade (%)", min_value=0., max_value=5.,
                              value=.1, step=.05, format="%.2f", key="v918_cost",
                              help="Applied to the simulated amount traded at each rebalance. Excludes taxes, bid/ask spread and market impact.")
    rebalance = REBALANCE_CHOICES[cadence]
    st.info(rebalance_explanation(rebalance))
    n = len(st.session_state.holdings)
    max_cap = 100.0
    extra_args = []
    group_rule = None
    scenario_rule = None
    scenario_invalid = False
    with st.expander("Optional investment limits (none applied by default)"):
        use_cap = st.checkbox("Set a maximum weight per investment", value=False, key="v919_use_cap",
                              help="Optional diversification constraint; not a mathematical requirement.")
        if use_cap:
            max_cap = st.number_input("Maximum percentage in a single investment", 
                                      min_value=math.ceil(10000/max(n,1))/100,
                                      max_value=100., value=max(40., math.ceil(10000/max(n,1))/100),
                                      step=5., key="v919_max_cap",
                                      help="A restriction you choose. It may prevent the mathematically highest-growth allocation.")
            st.caption(f"Optional cap enabled: at most {max_cap:.0f}% per investment.")
        else:
            st.caption("No cap: any one holding may receive between 0% and 100% of the proposed portfolio.")
        for key, field in (("asset_caps", "Maximum for a specific investment"), ("asset_floors", "Minimum for a specific investment")):
            values = st.text_input(field, placeholder="AAPL=25" if key == 'asset_caps' else 'SPY=10',
                                   key=f"v918_{key}", help="Enter ticker=percent. Separate multiple investments with commas.")
            if values.strip():
                for token in values.split(','):
                    if token.strip():
                        extra_args.extend(['--asset-cap' if key == 'asset_caps' else '--asset-floor', token.strip()])
        turnover = st.number_input("Maximum one-way trading turnover (%)", value=0.,
                                   min_value=0., max_value=100., step=5., key="v918_turnover",
                                   help="0 means no extra cap. E.g. 25 limits total one-way allocation changes to 25 percentage points.")
        if turnover:
            extra_args.extend(['--turnover-cap', str(turnover)])
        st.markdown('**Limit a group of investments**')
        group_label = st.text_input('Name your investment group', value='My_group', key='v918_group_name',
                                    help='This is a label you choose, not a predefined financial category.')
        group_members = st.multiselect('Choose investments for the group',
                        options=sorted(h['symbol'] for h in st.session_state.holdings),
                        key='v918_group_members')
        if group_members:
            group_max = st.number_input('Maximum combined group allocation (%)',
                        min_value=0., max_value=100., value=40., key='v918_group_max')
            try:
                group_rule = format_group_cap(group_label, group_members, group_max,
                                  [h['symbol'] for h in st.session_state.holdings])
                extra_args.extend(['--group-cap', group_rule])
            except PortfolioError as exc:
                st.error(str(exc))
    with st.expander('Test my own allocation (optional)', expanded=False):
        enabled = st.toggle('Compare a custom allocation', value=False, key='v920_include_custom')
        if enabled:
            from .v920_presentation import custom_start_weights, editable_percentages
            prior = (_current_report('three_strategy') or _current_report('insights')
                     or _current_report('strategy'))
            baseline_weights = (prior or {}).get('input_weights') or {}
            if not baseline_weights:
                # No initial Overview or Portfolio Optimizer run required. Convert the
                # Builder holdings with provider-verified dated FX when needed.
                try:
                    rows = _get_inputs()
                    needs_fx = any(row['currency'] != st.session_state.base for row in rows)
                    fx_result = _cached_display_fx(dt.date.today()) if needs_fx else None
                    if needs_fx and not fx_result:
                        raise PortfolioError('A verified CAD/USD quote is unavailable. FX weights cannot be estimated safely.')
                    baseline_weights = custom_start_weights(
                        rows, st.session_state.base,
                        usd_cad=fx_result[0] if fx_result else None)
                    if needs_fx:
                        st.caption(f"Starting weights use verified CAD/USD FX observed {fx_result[1]}.")
                except (PortfolioError, ValueError) as exc:
                    st.warning(f'Cannot initialize a currency-accurate manual comparison: {exc}')
            if baseline_weights:
                baseline_weights = dict(baseline_weights)
                st.caption('Edit the percentages and compare them with your real starting allocation, SPY and RSP. '
                           'Only investments already in Portfolio Builder can be used. This does not place trades.')
                c1, c2, c3 = st.columns(3)
                if c1.button('Start with current weights', key='v920_fill_current', width='stretch'):
                    st.session_state.v920_custom_prefill = editable_percentages(baseline_weights)
                    st.session_state.v920_custom_edit_version = st.session_state.get('v920_custom_edit_version', 0) + 1
                    st.rerun()
                if c2.button('Start equally weighted', key='v920_fill_equal', width='stretch'):
                    st.session_state.v920_custom_prefill = editable_percentages(
                        {key:1/len(baseline_weights) for key in baseline_weights})
                    st.session_state.v920_custom_edit_version = st.session_state.get('v920_custom_edit_version', 0) + 1
                    st.rerun()
                if c3.button('Use Growth suggestion', key='v920_fill_growth', width='stretch',
                             disabled=not ((prior or {}).get('three_strategy_analysis') or {}).get('strategies', {}).get('growth', {}).get('weights')):
                    st.session_state.v920_custom_prefill = editable_percentages(
                        prior['three_strategy_analysis']['strategies']['growth']['weights'])
                    st.session_state.v920_custom_edit_version = st.session_state.get('v920_custom_edit_version', 0) + 1
                    st.rerun()
                # Re-keying on changes to the Builder portfolio avoids keeping
                # previous ticker mixes in the dataframe editor.
                signature = (_portfolio_signature() or 'empty')[:12]
                active_sig = st.session_state.get('v920_custom_portfolio_sig')
                if active_sig != signature:
                    st.session_state.v920_custom_portfolio_sig = signature
                    st.session_state.v920_custom_prefill = editable_percentages(baseline_weights)
                    st.session_state.v920_custom_edit_version = st.session_state.get('v920_custom_edit_version', 0) + 1
                prefill = st.session_state.get('v920_custom_prefill') or editable_percentages(baseline_weights)
                custom = st.data_editor(pd.DataFrame([
                        {'Ticker':sym, 'Weight (%)': weight}
                        for sym, weight in prefill.items()]), hide_index=True,
                        column_config={'Ticker':st.column_config.TextColumn('Ticker', disabled=True),
                                       'Weight (%)':st.column_config.NumberColumn('Weight (%)', min_value=0.,max_value=100.,format='%.2f')},
                        num_rows='fixed', width='stretch',
                        key=f'v920_custom_editor_{signature}_{st.session_state.get("v920_custom_edit_version",0)}')
                custom_rows=[{'Ticker':row['Ticker'],'Target weight (%)':row['Weight (%)']}
                             for row in custom.to_dict(orient='records')]
                total=sum(row['Target weight (%)'] for row in custom_rows)
                st.metric('Your custom allocation total', f'{total:.2f}%')
                if abs(total-100)>0.0001:
                    scenario_invalid=True
                    st.error(f'Your weights total {total:.2f}%. Adjust them to 100.00% before running.')
                else:
                    scenario_rule=prepare_what_if(custom_rows,list(baseline_weights))
                    st.success('Ready: your custom mix will be compared when you select Optimize my portfolio.')
            else:
                scenario_invalid=True
    st.divider()
    st.subheader("Optimize and compare")
    st.write("One analysis finds three mathematical optima **among the investments you already hold**: "
             "**maximum historical growth**, **highest Sharpe ratio** and **minimum volatility**. "
             "Each is compared with your current portfolio, SPY and RSP on matching dates.")
    st.caption("Optimizes only the investments added in Portfolio Builder. Results are historical comparisons, not forecasts.")
    args = ["--start", str(_history_dates()[0]), "--end", str(_history_dates()[1]),
            "--three-strategies", "--max-weight", str(max_cap/100),
            "--rebalance", rebalance, "--cost-bps", str(round(fee*100)),
            "--train-periods", "252", "--test-periods", "126", "--step-periods", "126",
            *extra_args]
    if scenario_rule is not None:
        args.extend(['--scenario', scenario_rule])
    record = st.session_state.reports.get('three_strategy') or {}
    if st.button('Optimize my portfolio', type='primary', width='stretch', key='v918_run_three', disabled=scenario_invalid):
        with st.spinner('Optimizing the three portfolios and checking them against later historical periods...'):
            if _run('insights', args, key='three_strategy'):
                st.rerun()
    if record.get('signature') != _portfolio_signature():
        st.caption('Run the analysis to explore the three optimized allocations.')
        return
    if record.get('args') != args:
        st.info('Management settings changed. Run the analysis again to refresh all three portfolios.')
        return
    st.divider()
    show_three_results(record['report'])
    if record['report'].get('user_scenarios'):
        with st.expander('Your custom allocation: historical comparison', expanded=True):
            _compare_scenarios(record['report'])
    _report_download(record['report'], 'three_strategy_research.json')

def main():
    st.set_page_config(page_title="Portfolio Risk Engine · V10",page_icon="📊",layout="wide",
                       initial_sidebar_state="expanded")
    _init_state()
    _style()
    with st.sidebar:
        st.markdown("## Portfolio Risk Engine")
        count = len(st.session_state.holdings)
        portfolio_ready = _portfolio_signature() is not None
        if portfolio_ready:
            st.caption(f"Portfolio ready: {count} investment{'s' if count != 1 else ''}")
        # A saved Light preference from V10.3/V10.4 must not leave the selector invalid.
        if st.session_state.get("v10_theme") not in THEMES:
            st.session_state["v10_theme"] = "Dark Navy"
        st.selectbox("Appearance", options=list(THEMES), key="v10_theme",
                     help="Choose from four coordinated dark themes. Charts, panels and controls update together.")
        st.selectbox("Display your results in", options=SUPPORTED_CURRENCIES,
                     key="base",help="Choose CAD or USD for your displayed portfolio totals and research results. Mixed-currency holdings are converted using dated market exchange rates.")
        with st.expander("Historical time period"):
            st.toggle("Use longest shared history", key="history_auto",
                      help="The engine aligns all holdings and both S&P 500 ETF comparisons to shared dates. An asset with a later launch limits the earliest usable date.")
            if st.session_state.history_auto:
                st.caption("Automatic: request history back to 2000; analyze all common available dates. Older holdings may have longer individual histories.")
            else:
                st.date_input("Start",key="search_date_start")
                st.date_input("End (exclusive)",key="search_date_end",
                              help="Last day excluded from the historical price window.")
                if _history_dates()[0] >= _history_dates()[1]:
                    st.warning("Start date must be before the end date")
        st.divider()
        github = (os.environ.get("PORTFOLIO_GITHUB_URL") or GITHUB_URL).strip()
        if github.startswith(("https://github.com/", "https://www.github.com/")):
            st.link_button("View project on GitHub", github, icon=":material/code:")
        else:
            st.caption("Portfolio analytics · Python + Streamlit")
    # The page list stays visible, but shows a lock until the portfolio can be analyzed.
    # Existing _current_data_warning checks prevent the locked pages from calculating.
    builder_icon = ":material/account_balance_wallet:" if portfolio_ready else ":material/arrow_circle_right:"
    def nav_page(func, title, icon):
        return st.Page(func, title=title, icon=icon if portfolio_ready else ":material/lock:")
    nav=st.navigation({
        "YOUR PORTFOLIO": [
             st.Page(page_builder,title="Portfolio Builder",icon=builder_icon,default=True),
             nav_page(page_overview,"Overview",":material/dashboard:")],
        "ANALYSIS": [
             nav_page(page_risk,"Risk & Performance",":material/monitoring:"),
             nav_page(page_exposure,"Holdings & Exposure",":material/layers:"),
             nav_page(page_dividends,"Income & Dividends",":material/payments:"),
             nav_page(page_valuation,"Stock Valuation",":material/query_stats:")],
        "OPTIMIZATION": [nav_page(page_strategy,"Portfolio Optimizer",":material/tune:")],
        "CALCULATORS": [
            st.Page(page_compound, title="Compound Interest", icon=":material/trending_up:"),
            st.Page(page_savings, title="Savings Goal", icon=":material/savings:"),
            st.Page(page_basic, title="Calculator", icon=":material/calculate:"),
        ],
    },position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()
