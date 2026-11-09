"""Plain-language views for V9.18 three-objective portfolio research.

Never computes security returns or reinterprets missing market data as zero.
"""
from __future__ import annotations

from typing import Mapping
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .v10_design import tune_plotly, theme_for, display_table
from .v911_presentation import format_metric_value

_ORDER = (
    ('growth', 'Growth', 'Find the strongest historical growth'),
    ('balanced', 'Balanced', 'Seek more return per unit of risk'),
    ('defensive', 'Defensive', 'Reduce historical portfolio fluctuations'),
)
_METRICS = (
    ('cagr', 'Annual growth'), ('sharpe', 'Sharpe ratio'),
    ('annualized_volatility', 'Volatility'), ('drawdown', 'Largest decline'),
    ('sortino', 'Downside-adjusted return'), ('es95', 'Expected shortfall'),
)


# Keep the established public names in this view; presentation delegates to
# independently testable evidence helpers instead of recomputing market metrics.
from .v921_evidence import status_label, benchmark_status_label, average_objective_advantage
from .v922_result_explanations import (goal_improvement, benchmark_insights,
    portfolio_validation_explanation, rating_text, benchmark_three_metric_summary)

# User-facing original-portfolio statuses:
_PORTFOLIO_LABELS = ('Consistent past improvements', 'Mixed historical evidence',
                     'Not supported by past tests')


def format_value(metric: str, number) -> str:
    if not isinstance(number, (float, int)):
        return 'Unavailable'
    if metric in {'cagr', 'annualized_volatility', 'drawdown', 'es95'}:
        return f'{number:.2%}'
    return f'{number:.2f}'


def comparisons(report: Mapping) -> pd.DataFrame:
    rows = []
    cand = report['current_training_metrics']
    orig = report['original_training_metrics']
    benchmarks = report['benchmarks_training_metrics']
    for key, label in _METRICS:
        rows.append({'Measure': label,
                     'Your portfolio': format_value(key, orig.get(key)),
                     'Proposed': format_value(key, cand.get(key)),
                     'SPY': format_value(key, benchmarks['SPY'].get(key)),
                     'RSP': format_value(key, benchmarks['RSP'].get(key))})
    return pd.DataFrame(rows)


from .v924_metric_colors import color_proposed_metrics


def styled_comparisons(report: Mapping, light: bool = False) -> pd.io.formats.style.Styler:
    return color_proposed_metrics(comparisons(report), report,
                                  tuple(metric for metric, _ in _METRICS), light=light)


def weights_rows(report: Mapping, original_weights: Mapping) -> pd.DataFrame:
    rows = []
    for sym, now in report['weights'].items():
        previous = float(original_weights.get(sym, 0))
        rows.append({'Investment': sym, 'Current (%)': previous*100,
                     'Proposed (%)': now*100, 'Change (pp)': (now-previous)*100})
    return pd.DataFrame(rows).sort_values('Proposed (%)', ascending=False)


def _verdict_text(goal: str, report: dict) -> str:
    cand = report['current_training_metrics']
    orig = report['original_training_metrics']
    if goal == 'growth':
        return (f"This mix had {cand['cagr']:.2%} annualized growth over the historical optimization period, "
                f"compared with {orig['cagr']:.2%} for your current weights. "
                "It was selected to maximize historical growth, not to minimize risk.")
    if goal == 'balanced':
        return (f"This mix had a Sharpe ratio of {cand['sharpe']:.2f}, versus "
                f"{orig['sharpe']:.2f} for your current weights. Higher means more historical "
                "return per unit of volatility, but it does not guarantee higher growth.")
    return (f"This mix had {cand['annualized_volatility']:.2%} historical annual volatility, "
            f"compared with {orig['annualized_volatility']:.2%} for your current weights. "
            "It may sacrifice growth in exchange for smaller fluctuations.")


def show_three_results(report: dict):
    analysis = report.get('three_strategy_analysis') or {}
    if analysis.get('status') != 'evaluated':
        st.info('Three-strategy results are not yet available for this portfolio.')
        return
    results = analysis['strategies']
    st.subheader('Your three optimized portfolios')
    st.caption(f"Historical prices: {analysis['history_start']} to {analysis['history_end']}. "
               'The proposed weights are the best fit we found looking backward, not forecasts or trading instructions.')
    st.caption('**How we tested the approach:** We went back to earlier dates, selected weights using only '
               'prices known then, and checked how they performed afterward. This is a replay of the '
               'past, not a simulation of what will happen in the future.')
    cards = st.columns(3)
    for col, (key, title, description) in zip(cards, _ORDER):
        with col:
            item = results[key]
            with st.container(border=True):
                st.markdown(f'**{title}**')
                if item.get('status') == 'evaluated':
                    st.markdown('**' + rating_text(item) + '**')
                    st.caption('With hindsight: ' + goal_improvement(key, item)['headline'])
                else:
                    st.caption('Insufficient data for optimization')
    # Full text labels make the outcome intelligible without having to decode
    # a tiny star/cross beside the name.
    tabs = st.tabs([f'{name} · {rating_text(results[key])}'
                    if results[key].get('status') == 'evaluated' else name
                    for key, name, _ in _ORDER])
    for tab, (goal, title, desc) in zip(tabs, _ORDER):
        with tab:
            item = results[goal]
            if item.get('status') != 'evaluated':
                st.info(item.get('reason', 'This allocation was unavailable under the selected limits.'))
                continue
            icon, label = status_label(item)
            st.subheader(f'{title} portfolio')
            st.markdown('### ' + rating_text(item))
            st.caption(desc)
            st.markdown('**Proposed allocation**')
            if analysis['rebalancing'] == 'buy_hold':
                st.caption('These weights are starting allocations. Actual weights drift as prices change.')
            wdf = weights_rows(item, report['input_weights'])
            fig = go.Figure()
            fig.add_bar(name='Current', x=wdf['Investment'], y=wdf['Current (%)'],
                        marker_color='#7C879C', hovertemplate='%{x}: %{y:.2f}%<extra>Current</extra>')
            fig.add_bar(name='Optimized', x=wdf['Investment'], y=wdf['Proposed (%)'],
                        marker_color='#46BED8', hovertemplate='%{x}: %{y:.2f}%<extra>Optimized</extra>')
            fig.update_layout(barmode='group', height=300, margin=dict(l=5,r=5,t=20,b=15),
                              yaxis_title='Share of portfolio (%)', xaxis_title='',
                              legend=dict(orientation='h',y=1.17), template='plotly_dark')
            tune_plotly(fig, st.session_state.get('v10_theme'))
            st.plotly_chart(fig, width='stretch', key=f'v918_weights_{goal}', theme=None)
            display_table(wdf.style.format({'Current (%)':'{:.2f}%',
                    'Proposed (%)':'{:.2f}%', 'Change (pp)':'{:+.2f}'}),
                    width='stretch', hide_index=True)
            largest = wdf.iloc[0]
            if float(largest['Proposed (%)']) >= 80.0:
                st.markdown(
                    f"<div class='v10-concentration-warning'><strong>Concentration risk:</strong> "
                    f"{largest['Investment']} receives {largest['Proposed (%)']:.1f}% of this allocation. "
                    "Uncapped historical optimization can favor a single past winner; "
                    "it does not establish that concentrating here is prudent or will work in the future.</div>",
                    unsafe_allow_html=True)
            if analysis['rebalancing'] == 'buy_hold' and item.get('ending_weights'):
                st.markdown('**How the weights drifted by the end of the historical buy-and-hold simulation**')
                ending = item['ending_weights']
                drift_rows = pd.DataFrame([{'Investment':sym,
                    'Starting weight (%)':100*float(weight),
                    'Ending weight (%)':100*float(ending.get(sym,0))}
                    for sym,weight in item['weights'].items()])
                display_table(drift_rows.style.format({'Starting weight (%)':'{:.2f}%',
                    'Ending weight (%)':'{:.2f}%'}), hide_index=True, width='stretch')
                st.caption('Ending weights describe the hypothetical historical simulation. They are not '
                           "today's recommended weights and do not represent a real holdings account.")
            improvements = goal_improvement(goal, item)
            st.markdown('**How the best historical fit performed (with hindsight)**')
            st.markdown('**' + improvements['headline'] + '**')
            st.caption(improvements['tradeoff'])
            # styled_comparisons(item) is the original dark-mode metric presentation.
            display_table(styled_comparisons(item, light=theme_for(st.session_state.get('v10_theme')).is_light), width='stretch', hide_index=True)
            st.caption('Proposed values: green improves on your current portfolio; '
                       'red is worse. These colors describe the full historical period, not future results.')
            hold = item['holdout']
            st.markdown('**How convincing are these results?**')
            # Full-width stacked evidence cards: no half-empty left column.
            with st.container(border=True, key=f"optimizer_evidence_portfolio_{goal}"):
                st.markdown('**Did it improve your portfolio?**')
                st.markdown('**' + rating_text(item) + '**')
                st.write(portfolio_validation_explanation(goal, item))
            with st.container(border=True, key=f"optimizer_evidence_benchmarks_{goal}"):
                st.markdown('**How does it compare with SPY and RSP?**')
                _, later = benchmark_insights(goal, item)
                st.caption('Compared with each benchmark over the full history (proposed allocation):')
                for comparison in benchmark_three_metric_summary(item):
                    st.markdown(f"**{comparison['ticker']}: {comparison['summary']}**")
                    for metric in comparison['lines']:
                        color = {'better': 'green', 'worse': 'red', 'similar': 'gray'}[metric['status']]
                        st.markdown(f":{color}[**{metric['label']}:** {metric['description']}]")
                st.caption(later)
            with st.expander('Technical details: individual historical tests'):
                n = hold['tests']
                st.write(f"Selected goal improved in {hold['better']} of {n} later tests, "
                         f"was similar in {hold['similar']}, and worsened in {hold['worse']}.")
                st.write(f"Beat SPY in {hold['benchmark_counts']['SPY']}/{n} later tests "
                         f"and RSP in {hold['benchmark_counts']['RSP']}/{n}.")
                st.caption('Improvement evidence requires at least 75% positive later tests '
                           'and a positive average. Each test refits the weights using only earlier data. '
                           'These later historical periods are not statistical significance and do not validate the exact full-history weights.')
                if hold['folds']:
                    display_table(pd.DataFrame([{
                        'Training ends': x['train_end'], 'Later test ends': x['test_end'],
                        'Goal improved': 'Yes' if x['primary_better'] else 'No',
                        'Goal change': x['primary_improvement'],
                    } for x in hold['folds']]).style.format({'Goal change':'{:+.4f}'}),
                        hide_index=True, width='stretch')
    with st.expander('Technical details: data and assumptions'):
        st.caption('SPY is the capitalization-weighted S&P 500 ETF; RSP is the equal-weighted S&P 500 ETF. '
                   'Both use matched dates and reporting currency.')
        st.caption(f"Rebalancing: {analysis['rebalancing']}. Trading costs: "
                   f"{analysis['trading_cost_bps']:.2f} basis points on amounts traded. "
                   f"Individual weight cap: {'none' if analysis['candidate_cap'] >= .999999 else f'{analysis["candidate_cap"]:.0%}'}. "
                   f"Test windows: {analysis['validation']['tested_folds']}, "
                   f"with {analysis['validation']['holdout_periods']} days per later period.")
        st.caption('A star means the optimization method improved its selected goal in at least 75% '
                   'of later windows with positive average improvement versus your original allocation. '
                   'Benchmark superiority is evaluated separately. Neither establishes investment suitability.')
