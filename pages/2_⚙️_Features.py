"""2_Features.py — Feature Engineering"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import (
    inject_css, COUNTRIES, COUNTRY_BLOC, BLOC_COLORS,
    FEATURES, FEAT_DISPLAY, DISP_FEATURES, TARGET,
    dark_fig, style_ax, show, guard,
)

st.set_page_config(page_title="Features · SSA Trade", page_icon="⚙️", layout="wide")
inject_css()
guard(["df"], "⚠️ Please upload your dataset on the **Home** page first.")
df = st.session_state["df"]

st.markdown("## ⚙️ Feature Engineering")
st.caption("All transformations applied to raw data before training. Features were selected "
           "based on gravity model theory and prior SSA trade literature.")

# ── Feature Reference Table ───────────────────────────────────────────────────
st.markdown("### Feature Definitions")
feat_info = [
    ("Log GDP",             "log(1 + GDP_USD)",                  "Gravity model — stabilise right-skewed GDP"),
    ("Log Exchange Rate",   "log(1 + ExchRate_LCU/USD)",         "Reduce exchange rate skewness"),
    ("Inflation (%)",       "raw Inflation (annual %)",          "Price-level competitiveness proxy"),
    ("Trade Lag-1",         "Trade(t-1)",                        "First-order autocorrelation in trade openness"),
    ("Trade Lag-2",         "Trade(t-2)",                        "Second-order autoregressive effect"),
    ("Inflation Lag-1",     "Inflation(t-1)",                    "Lagged price effect on trade"),
    ("Log GDP Lag-1",       "log_GDP(t-1)",                      "Lagged economic size"),
    ("Trade Rolling (3yr)", "rolling_mean(Trade, 3)",            "Smooth business-cycle noise"),
    ("Inflation Rolling (3yr)","rolling_mean(Inflation, 3)",     "Smooth inflationary noise"),
    ("Trade × Log GDP",     "Trade_lag1 × log_GDP",              "Openness-size interaction (gravity)"),
    ("Year (Normalized)",   "(Year − 1970) / 54",                "Linear time trend [0, 1]"),
]
feat_df = pd.DataFrame(feat_info, columns=["Feature", "Formula", "Economic Rationale"])
st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.divider()

# ── Country-level inspection ──────────────────────────────────────────────────
st.markdown("### Feature Distributions by Country")
fe_country = st.selectbox("Select country", COUNTRIES, key="fe_country")
sub = df[df["Country Name"] == fe_country]
c   = BLOC_COLORS[COUNTRY_BLOC[fe_country]]

plot_feats = [
    ("log_GDP",            "Log GDP"),
    ("log_Exchange",       "Log Exchange Rate"),
    ("GDP_growth",         "GDP Growth (%)"),
    ("Trade_rolling3",     "Trade Rolling Mean (3yr)"),
    ("Trade_x_GDP",        "Trade × Log GDP"),
    ("Inflation_rolling3", "Inflation Rolling (3yr)"),
]

fig, axes = dark_fig(2, 3, figsize=(15, 8), constrained_layout=True)
fig.suptitle(f"Engineered Feature Distributions — {fe_country}",
             color="#E8EAF0", fontsize=12, fontweight="bold")
for ax, (feat, label) in zip(np.array(axes).flat, plot_feats):
    if feat not in sub.columns:
        continue
    vals = sub[feat].dropna()
    ax.hist(vals, bins=20, color=c, alpha=0.82, edgecolor="#0D0F18")
    ax.axvline(vals.mean(),   color="white",   lw=1.5, ls="--", alpha=0.9, label=f"Mean={vals.mean():.1f}")
    ax.axvline(vals.median(), color="#FDD835", lw=1.2, ls=":",  alpha=0.8, label=f"Med={vals.median():.1f}")
    style_ax(ax, title=label)
    ax.legend(fontsize=8, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
show(fig)

# ── Lag Correlation Chart ─────────────────────────────────────────────────────
st.markdown("### Lag & Rolling Feature Correlations with Trade Volume")
lag_feats = [f for f in FEATURES if "lag" in f or "rolling" in f]
corrs, labels = [], []
for feat in lag_feats:
    if feat in sub.columns and TARGET in sub.columns:
        pair = sub[[feat, TARGET]].dropna()
        if len(pair) > 5:
            r = pair.corr().iloc[0, 1]
            corrs.append(r)
            labels.append(FEAT_DISPLAY.get(feat, feat))

c1, c2 = st.columns(2)
with c1:
    fig, ax = dark_fig(figsize=(8, 5))
    colors = ["#43A047" if v >= 0 else "#EF5350" for v in corrs]
    sorted_pairs = sorted(zip(corrs, labels), key=lambda x: x[0])
    s_corrs, s_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])
    ax.barh(list(s_labels), list(s_corrs), color=["#43A047" if v >= 0 else "#EF5350" for v in s_corrs],
            edgecolor="#0D0F18", height=0.62)
    ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.4)
    style_ax(ax, title=f"Lag/Rolling Correlation with {TARGET}",
             xlabel="Pearson r")
    show(fig)
with c2:
    corr_df = pd.DataFrame({"Feature": s_labels, "Pearson r": [round(v, 3) for v in s_corrs]})
    corr_df["Strength"] = corr_df["Pearson r"].abs().apply(
        lambda x: "Strong" if x > 0.7 else ("Moderate" if x > 0.4 else "Weak")
    )
    st.dataframe(corr_df.sort_values("Pearson r", ascending=False),
                 use_container_width=True, hide_index=True)

# ── Time Series All Countries ─────────────────────────────────────────────────
st.markdown("### Feature Time Series — All Countries")
feat_choice = st.selectbox("Select feature", DISP_FEATURES, key="fe_feat")
feat_key    = FEATURES[DISP_FEATURES.index(feat_choice)]
fig, ax = dark_fig(figsize=(15, 4.5))
for country in COUNTRIES:
    sc = df[df["Country Name"] == country]
    if feat_key in sc.columns:
        ax.plot(sc["Year"], sc[feat_key],
                color=BLOC_COLORS[COUNTRY_BLOC[country]],
                lw=1.8, label=country, alpha=0.85)
style_ax(ax, title=f"{feat_choice} — All Countries", xlabel="Year", ylabel=feat_choice)
import matplotlib.patches as mpatches
from utils import BLOCS
bloc_patches = [mpatches.Patch(color=v, label=k) for k, v in BLOC_COLORS.items()]
ax.legend(handles=bloc_patches, fontsize=9,
          facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
show(fig)
