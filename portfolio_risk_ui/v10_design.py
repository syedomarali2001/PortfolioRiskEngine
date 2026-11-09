"""V10 visual design system. No finance, portfolio, or optimization logic lives here.

Four user-selectable dark themes. Styles are scoped to Streamlit's
app chrome and ordinary widgets; financial status coloring remains semantic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    sidebar: str
    panel: str
    elevated: str
    text: str
    muted: str
    border: str
    accent: str
    accent_text: str
    accent_soft: str
    current: str
    chart: str
    grid: str
    positive: str
    negative: str
    is_light: bool = False


THEMES: Mapping[str, Theme] = {
    "Dark Navy": Theme("Dark Navy", "#0B111B", "#18263A", "#172437", "#1E3048",
                       "#F0F5FC", "#ACBDD1", "#30435A", "#78A9EB", "#071725",
                       "#263C5B", "#8897AE", "#4CBFDA", "#33475D", "#60D6AA", "#F48A98"),
    "Ocean Blue": Theme("Ocean Blue", "#081A2B", "#112B44", "#143450", "#1A4160",
                        "#F0F8FD", "#ACCBDD", "#315771", "#53B9E9", "#072030",
                        "#224C69", "#8BA4B8", "#4CBFDA", "#365B73", "#64D9AE", "#FF99A1"),
    "Forest Green": Theme("Forest Green", "#0B1B18", "#13342B", "#193B33", "#23493F",
                          "#EEF9F4", "#B5D0C4", "#355B4E", "#63CBA6", "#0B251B",
                          "#275449", "#92ADA0", "#56CCAC", "#42675A", "#74E2B9", "#F99AA0"),
    "Burgundy Red": Theme("Burgundy Red", "#1C111A", "#30202C", "#392635", "#493143",
                          "#FCF2F7", "#DAC0CF", "#624657", "#E69AB4", "#301121",
                          "#593448", "#AC95A5", "#E0A0BD", "#66485A", "#80DBB4", "#FF8B91"),
}


def theme_for(name: str | None) -> Theme:
    """Unknown preference gracefully falls back to the original dark theme."""
    return THEMES.get(name or "Dark Navy", THEMES["Dark Navy"])


# Separate categorical colors from the accent color. Adjacent chart marks
# should remain distinct even when the theme is predominantly green/red/blue.
CHART_PALETTES: Mapping[str, tuple[str, ...]] = {
    "Dark Navy": ("#4FB9E8", "#F0B76D", "#B3A1F1", "#55CBA0", "#EC8496", "#9AAEC4", "#DFCC75", "#C6A0DC"),
    "Ocean Blue": ("#53C1F5", "#FFBF6D", "#B5A0FF", "#65D6B6", "#FF8B9D", "#A6BED4", "#F1D474", "#DDA2E9"),
    "Forest Green": ("#46C7AD", "#F5B15C", "#9687E0", "#E77A89", "#70A6EA", "#DBC56A", "#BA86C7", "#A4BCC5"),
    "Burgundy Red": ("#E68FB6", "#62C8BF", "#EDB25C", "#83B1F4", "#A68DE0", "#C8CA76", "#94BCC9", "#E58D6F"),
}


def categorical_colors(name: str | None, count: int) -> list[str]:
    palette = CHART_PALETTES.get(name or "Dark Navy", CHART_PALETTES["Dark Navy"])
    return [palette[i % len(palette)] for i in range(max(0, count))]


def theme_css(name: str | None) -> str:
    """Shared sizing, typography and surfaces, plus the active palette.

    No application calculations or Streamlit widget state are modified.
    """
    t = theme_for(name)
    return f"""
<style>
:root, [data-testid="stApp"] {{
  --v10-bg:{t.background}; --v10-sidebar:{t.sidebar};
  --v10-panel:{t.panel}; --v10-elevated:{t.elevated};
  --v10-fg:{t.text}; --v10-muted:{t.muted}; --v10-border:{t.border};
  --v10-accent:{t.accent}; --v10-accent-text:{t.accent_text};
  --v10-accent-soft:{t.accent_soft};
  --background-color:{t.background}; --secondary-background-color:{t.panel};
  --text-color:{t.text}; --primary-color:{t.accent}; --border-color:{t.border};
}}
/* A wide but controlled reading canvas: more room for tables without huge lines. */
[data-testid="stMainBlockContainer"], .block-container {{
  width:100%; max-width:1520px !important;
  padding:1.75rem clamp(1.25rem,3.0vw,2.75rem) 3rem !important;
}}
[data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{ background-color:var(--v10-bg) !important; color:var(--v10-fg); }}
[data-testid="stHeader"] {{ background-color:transparent !important; }}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
  background-color:var(--v10-sidebar) !important; color:var(--v10-fg) !important;
}}
[data-testid="stApp"] h1, [data-testid="stApp"] h2, [data-testid="stApp"] h3,
[data-testid="stApp"] h4, [data-testid="stApp"] p,
[data-testid="stApp"] label, [data-testid="stApp"] li,
[data-testid="stMarkdownContainer"] {{ color:var(--v10-fg); }}
[data-testid="stApp"] h1 {{ letter-spacing:-.035em; font-weight:750; }}
[data-testid="stApp"] h2, [data-testid="stApp"] h3 {{ letter-spacing:-.027em; font-weight:690; }}
/* Caption and quiet explanatory text must still be readable on 1080p screens. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] small {{ color:var(--v10-muted) !important; font-size:.875rem !important; line-height:1.55 !important; }}
[data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebarNav"] span {{ font-size:.91rem !important; }}
[data-testid="stSidebarNav"] a {{ border-radius:8px; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background-color:var(--v10-accent-soft) !important; }}
/* Same metric system on Overview, Risk, Income and Valuation. */
[data-testid="stMetric"] {{
  background:var(--v10-panel) !important; border:1px solid var(--v10-border) !important;
  border-radius:12px !important; padding:1rem 1.12rem !important;
  min-height:112px;
}}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
  opacity:1 !important; color:var(--v10-muted) !important;
  font-size:.9rem !important;
}}
[data-testid="stMetricValue"] {{ color:var(--v10-fg) !important; }}
/* This outer surface replaces the nested box inside each valuation ratio card. */
[class*="st-key-valuation_multiple_"] [data-testid="stMetric"] {{
  border:0 !important; background:transparent !important;
  padding:0 !important; min-height:0 !important;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:var(--v10-border); border-radius:12px; }}
[data-testid="stExpander"] {{ border-color:var(--v10-border) !important; border-radius:10px; }}
[data-testid="stAlert"] {{ border-radius:10px !important; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius:10px; overflow:hidden; }}
/* Native inputs and primary actions follow the selected appearance. */
[data-baseweb="input"] > div, [data-baseweb="select"] > div,
[data-baseweb="textarea"] > div {{
  background-color:var(--v10-panel) !important;
  border-color:var(--v10-border) !important;
  color:var(--v10-fg) !important;
}}
[data-baseweb="input"] input, [data-baseweb="select"] input,
[data-baseweb="textarea"] textarea {{ color:var(--v10-fg) !important; }}
[data-testid="stApp"] button[kind="primary"] {{
  background-color:var(--v10-accent) !important;
  color:var(--v10-accent-text) !important;
  border-color:var(--v10-accent) !important;
  font-weight:650;
}}
[data-testid="stApp"] button[kind="secondary"] {{
  background:var(--v10-panel) !important;
  color:var(--v10-fg) !important; border-color:var(--v10-border) !important;
}}
[data-testid="stApp"] button:focus-visible, [data-testid="stApp"] a:focus-visible {{
  outline:2px solid var(--v10-accent) !important; outline-offset:3px;
}}
[data-testid="stTabs"] button[role="tab"] {{
  font-size:.95rem !important; font-weight:650; min-height:2.9rem;
  padding-left:1rem; padding-right:1rem;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
  border-bottom:3px solid var(--v10-accent) !important;
}}
/* V10.1: consistent visible input surfaces and selected states. */
[data-testid="stApp"] [data-baseweb="input"],
[data-testid="stApp"] [data-baseweb="input"] > div,
[data-testid="stApp"] [data-baseweb="select"],
[data-testid="stApp"] [data-baseweb="select"] > div,
[data-testid="stApp"] [data-baseweb="textarea"],
[data-testid="stApp"] [data-baseweb="textarea"] > div,
[data-testid="stApp"] [role="combobox"] {{
  background:var(--v10-panel) !important; color:var(--v10-fg) !important;
  border-color:var(--v10-border) !important;
}}
[data-testid="stApp"] input, [data-testid="stApp"] textarea,
[data-testid="stApp"] [role="combobox"] *,
[data-testid="stApp"] [data-baseweb="select"] * {{ color:var(--v10-fg) !important; }}
[data-testid="stApp"] input::placeholder {{ color:var(--v10-muted) !important; opacity:1; }}
[data-testid="stApp"] [data-testid="stSegmentedControl"] button,
[data-testid="stApp"] [data-testid="stSegmentedControl"] [role="radio"] {{
  background:var(--v10-panel) !important; color:var(--v10-fg) !important;
  border:1px solid var(--v10-border) !important;
}}
[data-testid="stApp"] [data-testid="stSegmentedControl"] [aria-checked="true"],
[data-testid="stApp"] [data-testid="stSegmentedControl"] [aria-pressed="true"] {{
  background:var(--v10-accent-soft) !important; color:var(--v10-fg) !important;
  border-color:var(--v10-accent) !important;
}}
[data-testid="stApp"] button:disabled {{ opacity:.62 !important; color:var(--v10-muted) !important; }}
/* Restore visible borders around holdings, expanders and report cards. */
[data-testid="stApp"] [data-testid="stVerticalBlockBorderWrapper"] {{
  border-color:var(--v10-border) !important;
}}
[data-testid="stApp"] [data-testid="stExpander"] {{
  background:var(--v10-panel) !important; border:1px solid var(--v10-border) !important;
}}
[data-testid="stApp"] [data-testid="stExpander"] summary,
[data-testid="stApp"] [data-testid="stExpander"] summary * {{ color:var(--v10-fg) !important; }}
[data-testid="stApp"] hr {{ border-color:var(--v10-border) !important; opacity:1; }}
/* Alert surfaces and labels use the selected dark palette. */
[data-testid="stApp"] [data-testid="stAlert"] {{
  background:{t.elevated} !important;
  border:1px solid {t.border} !important;
}}
[data-testid="stApp"] [data-testid="stAlert"] *,
[data-testid="stApp"] [data-testid="stAlert"] p {{
  color:{t.text} !important;
}}
/* Keep readable borders on any static report tables. */
[data-testid="stApp"] [data-testid="stTable"] {{
  background:var(--v10-panel); border:1px solid var(--v10-border);
  border-radius:10px; overflow:auto; max-height:580px;
}}
[data-testid="stApp"] [data-testid="stTable"] th,
[data-testid="stApp"] [data-testid="stTable"] td {{
  border-color:var(--v10-border); padding:.48rem .65rem;
}}

/* V10.2: Make inactive tabs and text controls readable consistently. */
[data-testid="stTabs"] button[role="tab"] {{ color:var(--v10-muted) !important; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color:var(--v10-fg) !important; }}
[data-testid="stApp"] [data-testid="stSidebarNav"] svg,
[data-testid="stApp"] [data-testid="stSidebarNav"] [data-testid="stIconMaterial"] {{
  color:var(--v10-fg) !important; opacity:1 !important;
}}


/* Purpose-built messages use readable colors independent of Streamlit alert internals. */
.v10-soft-info {{
  background:{t.elevated} !important;
  border:1px solid {t.border} !important;
  color:{t.text} !important;
  padding:.85rem 1rem; border-radius:9px; line-height:1.5;
}}
.v10-concentration-warning {{
  background:#3B3018 !important;
  border:1px solid #8A6A30 !important;
  color:#FFE6A8 !important;
  padding:.85rem 1rem; border-radius:9px; line-height:1.5;
}}

/* Preserve a distinct cancel/dismiss visual across themes. */
.st-key-cancel_selected_ticker button {{
  color:{t.negative} !important; background:var(--v10-panel) !important;
  border:1px solid {t.negative} !important;
}}
</style>
"""


def tune_plotly(fig, name: str | None):
    """Reskin an existing chart. Data, traces and metric calculations are untouched."""
    t = theme_for(name)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t.text, family="Arial, sans-serif", size=13),
        colorway=categorical_colors(name, 8),
        hoverlabel=dict(bgcolor=t.elevated, font=dict(color=t.text, size=13),
                        bordercolor=t.border),
    )
    fig.update_xaxes(tickfont=dict(color=t.muted, size=12), title_font=dict(color=t.muted, size=13),
                     gridcolor=t.grid, zerolinecolor=t.border)
    fig.update_yaxes(tickfont=dict(color=t.muted, size=12), title_font=dict(color=t.muted, size=13),
                     gridcolor=t.grid, zerolinecolor=t.border)
    # Retint the previous default blue without destroying categorical or semantic colors.
    old_accents = {"#78A9EB", "#78B8F5", "#46BED8"}
    for trace in fig.data:
        if getattr(trace, "type", "") == "pie":
            labels = getattr(trace, "labels", None)
            n = len(labels) if labels is not None else 0
            if n:
                trace.marker.colors = categorical_colors(name, n)
                trace.marker.line = dict(color=t.background, width=2)
                trace.insidetextfont = dict(color=t.background, size=13)
        marker = getattr(trace, "marker", None)
        if marker is not None and isinstance(getattr(marker, "color", None), str):
            if marker.color.upper() in old_accents:
                marker.color = t.chart
        line = getattr(trace, "line", None)
        if line is not None and isinstance(getattr(line, "color", None), str):
            if line.color.upper() in old_accents:
                line.color = t.chart
    return fig


def display_table(data, **kwargs):
    """Keep all four dark themes on the interactive dataframe used in V10.3."""
    import streamlit as st
    return st.dataframe(data, **kwargs)
