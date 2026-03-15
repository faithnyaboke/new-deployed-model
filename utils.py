"""utils.py — Shared constants, helpers, preprocessing for SSA Trade Analysis App"""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
COUNTRIES = [
    "Kenya", "Uganda", "Tanzania",
    "Cote d'Ivoire", "Ghana", "Senegal",
    "South Africa", "Zambia", "Zimbabwe"
]
BLOCS = {
    "EAC":    ["Kenya", "Uganda", "Tanzania"],
    "ECOWAS": ["Cote d'Ivoire", "Ghana", "Senegal"],
    "SADC":   ["South Africa", "Zambia", "Zimbabwe"],
}
COUNTRY_BLOC = {c: b for b, cs in BLOCS.items() for c in cs}
BLOC_COLORS  = {"EAC": "#1E88E5", "ECOWAS": "#FF7043", "SADC": "#43A047"}
COUNTRY_COLOR = {c: BLOC_COLORS[COUNTRY_BLOC[c]] for c in COUNTRIES}
# Distinct line colors for 1st / 2nd / 3rd country in bloc comparison charts (same order in every bloc)
LINE_COLORS_3 = ["#2196F3", "#FF9800", "#4CAF50"]

# Five shock scenarios used in Monte Carlo
SCENARIOS = {
    "Baseline":         {"gdp": 0.00, "inf": 0.00, "exch": 0.00},
    "Mild Shock":       {"gdp":-0.05, "inf": 0.10, "exch": 0.05},
    "Moderate Shock":   {"gdp":-0.10, "inf": 0.20, "exch": 0.15},
    "Severe Shock":     {"gdp":-0.20, "inf": 0.40, "exch": 0.30},
    "COVID-like Shock": {"gdp":-0.15, "inf": 0.05, "exch": 0.20},
}
SCEN_COLORS = {
    "Baseline":         "#1E88E5",
    "Mild Shock":       "#FDD835",
    "Moderate Shock":   "#FB8C00",
    "Severe Shock":     "#E53935",
    "COVID-like Shock": "#8E24AA",
}

TARGET = "Trade (% of GDP)"

# Feature names (after engineering) used for modelling
FEATURES = [
    "log_GDP",
    "log_Exchange",
    "Inflation (annual %)",
    "Trade (% of GDP)_lag1",
    "Trade (% of GDP)_lag2",
    "Trade_rolling3",
    "GDP_growth",
    "Trade_x_GDP",
    "Year_norm",
    "Inflation_rolling3",
    "log_GDP_lag1",
]
FEAT_DISPLAY = {
    "log_GDP":                  "Log GDP",
    "log_Exchange":             "Log Exchange Rate",
    "Inflation (annual %)":     "Inflation (%)",
    "Trade (% of GDP)_lag1":   "Trade Lag-1",
    "Trade (% of GDP)_lag2":   "Trade Lag-2",
    "Trade_rolling3":           "Trade Rolling Mean (3yr)",
    "GDP_growth":               "GDP Growth Rate (%)",
    "Trade_x_GDP":              "Trade × Log GDP",
    "Year_norm":                "Year (Normalized)",
    "Inflation_rolling3":       "Inflation Rolling (3yr)",
    "log_GDP_lag1":             "Log GDP Lag-1",
}
DISP_FEATURES = [FEAT_DISPLAY[f] for f in FEATURES]

# Clustering features — country-level aggregates
CLUSTER_FEATS = [
    "Trade (% of GDP)",
    "log_GDP",
    "Inflation (annual %)",
    "log_Exchange",
    "GDP_growth",
]

CLUSTER_PAL = ["#EF5350", "#42A5F5", "#66BB6A", "#FFA726", "#AB47BC"]

# ── DATA PIPELINE ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_bytes: bytes) -> pd.DataFrame:
    """Load CSV, filter to 9 SSA countries, interpolate missing values."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df[df["Country Name"].isin(COUNTRIES)].copy()
    df = df.sort_values(["Country Name", "Year"]).reset_index(drop=True)
    df.drop(columns=["FDI (% of GDP)"], inplace=True, errors="ignore")
    for col in [
        "Trade (% of GDP)", "GDP (current US$)",
        "Exchange Rate (LCU/USD)", "Inflation (annual %)",
    ]:
        if col in df.columns:
            df[col] = df.groupby("Country Name")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )
    df["Bloc"] = df["Country Name"].map(COUNTRY_BLOC)
    return df


@st.cache_data(show_spinner=False)
def engineer_features(_df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df = _df.copy()
    # Log transforms
    df["log_GDP"]      = np.log1p(df["GDP (current US$)"])
    df["log_Exchange"] = np.log1p(df["Exchange Rate (LCU/USD)"])
    # GDP growth rate
    df["GDP_growth"] = (
        df.groupby("Country Name")["GDP (current US$)"]
        .pct_change() * 100
    )
    # Lag features (1 & 2 year)
    for col in ["Trade (% of GDP)", "Inflation (annual %)", "log_GDP"]:
        df[f"{col}_lag1"] = df.groupby("Country Name")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("Country Name")[col].shift(2)
    # Rolling means (3-year)
    df["Trade_rolling3"] = df.groupby("Country Name")["Trade (% of GDP)"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["Inflation_rolling3"] = df.groupby("Country Name")["Inflation (annual %)"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    # Interaction term
    df["Trade_x_GDP"] = df["Trade (% of GDP)_lag1"] * df["log_GDP"]
    # Year normalised to [0, 1]
    df["Year_norm"] = (df["Year"] - 1970) / (2024 - 1970)
    return df


# ── MATPLOTLIB DARK THEME ─────────────────────────────────────────────────────
BG    = "#0D0F18"
BG2   = "#13151F"
EDGE  = "#1E2235"
TICK  = "#7A85A0"
WHITE = "#E8EAF0"


def dark_fig(nrows=1, ncols=1, figsize=(12, 5), **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG, **kwargs)
    axlist = np.array(axes).flatten() if hasattr(axes, "__len__") else [axes]
    for ax in axlist:
        ax.set_facecolor(BG2)
        ax.tick_params(colors=TICK, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(EDGE)
    return fig, axes


def style_ax(ax, title=None, xlabel=None, ylabel=None, fontsize=11):
    if title:  ax.set_title(title,  color=WHITE, fontweight="bold", fontsize=fontsize)
    if xlabel: ax.set_xlabel(xlabel, color=TICK,  fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TICK,  fontsize=9)


def show(fig):
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_plotly(fig):
    """Render an interactive Plotly figure (hover/click to see values)."""
    try:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=BG2,
            font=dict(color=WHITE, size=11),
            margin=dict(t=50, b=50, l=50, r=50),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": True})
    except Exception:
        st.plotly_chart(fig, width="stretch")


# ── CSS ───────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
code, pre { font-family: 'DM Mono', monospace !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #13151F; padding: 6px 8px;
    border-radius: 12px; border: 1px solid #1E2235; flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 16px; font-weight: 500;
    font-size: 0.85rem; color: #7A85A0 !important;
    background: transparent; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1A2540 !important;
    color: #5B9BF5 !important; font-weight: 600 !important;
}
[data-testid="stMetric"] {
    background: #13151F; border: 1px solid #1E2235;
    border-radius: 10px; padding: 14px 18px;
}
[data-testid="stMetricLabel"] { font-size:0.78rem; color:#7A85A0 !important; }
[data-testid="stMetricValue"] { font-size:1.4rem; font-weight:700; }
[data-testid="stFileUploadDropzone"] {
    background:#13151F !important; border:2px dashed #2A3A5C !important;
    border-radius:12px !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1565C0,#2196F3) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-weight:600 !important; padding:0.55rem 1.6rem !important;
}
.stDownloadButton > button {
    background:#1A1D27 !important; color:#5B9BF5 !important;
    border:1px solid #2A3A5C !important; border-radius:8px !important;
    font-size:0.85rem !important;
}
hr { border-color: #1E2235 !important; }
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def metric_card(label, value, color="#1E88E5", sub=None):
    sub_html = f"<div style='font-size:.78rem;color:#7A85A0;margin-top:4px'>{sub}</div>" if sub else ""
    st.markdown(
        f"""<div style='background:#13151F;border-top:3px solid {color};
        border:1px solid #1E2235;border-top:3px solid {color};
        border-radius:10px;padding:14px 18px;'>
        <div style='font-size:.78rem;color:#7A85A0'>{label}</div>
        <div style='font-size:1.4rem;font-weight:700;color:#E8EAF0;margin-top:4px'>{value}</div>
        {sub_html}</div>""",
        unsafe_allow_html=True,
    )


def guard(keys: list, message: str):
    """Stop rendering and show a warning if session_state keys are missing."""
    missing = [k for k in keys if k not in st.session_state]
    if missing:
        st.warning(message)
        st.stop()
