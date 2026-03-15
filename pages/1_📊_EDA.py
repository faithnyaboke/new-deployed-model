"""1_EDA.py — Exploratory Data Analysis"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
from utils import (
    inject_css, COUNTRIES, BLOCS, BLOC_COLORS, LINE_COLORS_3, COUNTRY_BLOC,
    dark_fig, style_ax, show, guard, TARGET,
    show_plotly,
)

st.set_page_config(page_title="EDA · SSA Trade", page_icon="📊", layout="wide")
inject_css()
guard(["df"], "⚠️ Please upload your dataset on the **Home** page first.")
df = st.session_state["df"]

st.markdown("## 📊 Exploratory Data Analysis")
st.caption("Understand the structure, distributions, and trends in the data before modelling.")

# ── Filters ───────────────────────────────────────────────────────────────────
fc1, fc2 = st.columns([2.5, 1])
with fc1:
    sel = st.multiselect("Countries", COUNTRIES, default=COUNTRIES, key="eda_c")
with fc2:
    yr = st.slider("Year range", 1970, 2024, (1970, 2024), key="eda_yr")
dfe = df[(df["Country Name"].isin(sel)) & (df["Year"].between(yr[0], yr[1]))]

t1, t2, t3, t4, t5, t6 = st.tabs([
    "📈 Trade Trends", "🌡️ Inflation", "💰 GDP",
    "💱 Exchange Rate", "🔥 Correlation", "📋 Summary Stats"
])

# ── Tab 1: Trade Trends ───────────────────────────────────────────────────────
with t1:
    st.markdown("#### Trade Volume (% of GDP) Over Time")
    st.caption("**Interactive:** hover or click on each chart to see exact values. One chart per country (unchanged). Dashed red = COVID-19 (2020) · Dotted orange = GFC (2008)")
    if go is not None:
        from plotly.subplots import make_subplots
        fig_ply = make_subplots(rows=3, cols=3, subplot_titles=[f"{c} ({COUNTRY_BLOC[c]})" for c in COUNTRIES],
                                vertical_spacing=0.08, horizontal_spacing=0.06)
        for idx, country in enumerate(COUNTRIES):
            row, col = idx // 3 + 1, idx % 3 + 1
            sub = dfe[dfe["Country Name"] == country]
            if not sub.empty:
                c = BLOC_COLORS[COUNTRY_BLOC[country]]
                fig_ply.add_trace(
                    go.Scatter(x=sub["Year"], y=sub[TARGET], name=country, line=dict(color=c, width=2.2),
                               mode="lines+markers", marker=dict(size=4),
                               hovertemplate="Year: %{x}<br>Trade % GDP: %{y:.2f}<extra></extra>"),
                    row=row, col=col)
        for r in range(1, 4):
            for c in range(1, 4):
                fig_ply.add_vline(x=2020, line_dash="dash", line_color="#EF5350", opacity=0.8, row=r, col=c)
                fig_ply.add_vline(x=2008, line_dash="dot", line_color="#FFA726", opacity=0.7, row=r, col=c)
        fig_ply.update_layout(
            paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
            title_text="Trade Volume (% of GDP) — 1970–2024", showlegend=False, margin=dict(t=60))
        fig_ply.update_xaxes(title_text="Year", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_ply.update_yaxes(title_text="Trade % GDP", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_ply)
    else:
        fig, axes = dark_fig(3, 3, figsize=(17, 11), constrained_layout=True)
        fig.suptitle("Trade Volume (% of GDP) — 1970–2024", color="#E8EAF0", fontsize=13, fontweight="bold", y=1.01)
        for ax, country in zip(np.array(axes).flat, COUNTRIES):
            sub = dfe[dfe["Country Name"] == country]
            c   = BLOC_COLORS[COUNTRY_BLOC[country]]
            if not sub.empty:
                ax.plot(sub["Year"], sub[TARGET], color=c, lw=2.2)
                ax.fill_between(sub["Year"], sub[TARGET], alpha=0.13, color=c)
            ax.axvline(2020, color="#EF5350", ls="--", lw=1.4, alpha=0.8)
            ax.axvline(2008, color="#FFA726", ls=":",  lw=1.0, alpha=0.7)
            style_ax(ax, title=f"{country}  ({COUNTRY_BLOC[country]})", xlabel="Year", ylabel="Trade % GDP")
        patches = [mpatches.Patch(color=v, label=k) for k, v in BLOC_COLORS.items()]
        patches += [mpatches.Patch(color="#EF5350", label="COVID-19 2020"), mpatches.Patch(color="#FFA726", label="GFC 2008")]
        fig.legend(handles=patches, loc="lower center", ncol=5, frameon=False, labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.03))
        show(fig)

    # Descriptive stats for trade (exclude countries with no valid trade data)
    st.markdown("**Trade Volume Summary Statistics**")
    ts = dfe.groupby("Country Name")[TARGET].agg(
        Mean="mean", Std="std", Min="min", Max="max", Median="median"
    ).round(2)
    ts = ts.dropna(how="all")  # omit countries with no trade data (e.g. if source had none)
    st.dataframe(ts, use_container_width=True)
    st.caption(
        "ECOWAS uses Côte d'Ivoire (Nigeria has no Trade % of GDP in World Bank source for 1970–2024)."
    )

# ── Tab 2: Inflation ──────────────────────────────────────────────────────────
with t2:
    st.markdown("#### Inflation Distribution by Country")
    st.caption("**Interactive:** hover or click on boxes to see quartiles and values.")
    c1, c2 = st.columns([2, 1])
    with c1:
        order = (dfe.groupby("Country Name")["Inflation (annual %)"]
                 .median().sort_values().index.tolist())
        order = [c for c in order if c in sel]
        if go is not None:
            fig_box = go.Figure()
            for c in order:
                vals = dfe.loc[dfe["Country Name"] == c, "Inflation (annual %)"].dropna()
                if len(vals) > 0:
                    fig_box.add_trace(go.Box(
                        y=vals, name=c, marker_color=BLOC_COLORS[COUNTRY_BLOC[c]],
                        boxpoints="outliers", hovertemplate="%{y:.2f}%<extra></extra>"
                    ))
            fig_box.update_layout(
                paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                title="Inflation Distribution (1970–2024)", yaxis_title="Inflation (annual %)",
                xaxis=dict(tickangle=-30, tickfont=dict(color="#7A85A0")),
                yaxis=dict(gridcolor="#1E2235", tickfont=dict(color="#7A85A0")),
                showlegend=False, margin=dict(t=50))
            show_plotly(fig_box)
        else:
            fig, ax = dark_fig(figsize=(12, 5))
            sns.boxplot(
                data=dfe, x="Country Name", y="Inflation (annual %)",
                order=order,
                palette=[BLOC_COLORS[COUNTRY_BLOC[c]] for c in order],
                ax=ax, width=0.55, fliersize=3
            )
            style_ax(ax, title="Inflation Distribution (1970–2024)",
                     ylabel="Inflation (annual %)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color="#C5CAD6")
            ax.legend(handles=[mpatches.Patch(color=v, label=k) for k,v in BLOC_COLORS.items()],
                      facecolor="#13151F", edgecolor="#1E2235", labelcolor="white", title="Bloc")
            show(fig)
    with c2:
        st.markdown("**Inflation Statistics**")
        inf_s = (dfe.groupby("Country Name")["Inflation (annual %)"]
                 .agg(Mean="mean", Std="std", Max="max").round(1)
                 .sort_values("Mean", ascending=False))
        st.dataframe(inf_s, use_container_width=True)
        st.caption("Zimbabwe's hyperinflation (2007–2009) dominates the distribution. "
                   "This extreme outlier is retained to preserve methodological integrity.")

    st.markdown("#### Inflation Over Time by Bloc")
    st.caption("**Interactive:** hover or click to see values. One chart per bloc (EAC, ECOWAS, SADC).")
    if go is not None:
        from plotly.subplots import make_subplots
        fig_ply = make_subplots(rows=1, cols=3, subplot_titles=list(BLOCS.keys()),
                                horizontal_spacing=0.08)
        for col, (bloc, members) in enumerate(BLOCS.items(), 1):
            for i, country in enumerate([m for m in members if m in sel]):
                sub = dfe[dfe["Country Name"] == country]
                if not sub.empty:
                    fig_ply.add_trace(
                        go.Scatter(x=sub["Year"], y=sub["Inflation (annual %)"], name=country,
                                  line=dict(color=LINE_COLORS_3[i % 3], width=2), mode="lines+markers", marker=dict(size=3),
                                  hovertemplate="Year: %{x}<br>Inflation %%: %{y:.2f}<extra></extra>"),
                        row=1, col=col)
        fig_ply.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              title_text="Inflation Over Time by Bloc", hovermode="x unified", margin=dict(t=50),
                              showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5))
        fig_ply.update_xaxes(title_text="Year", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_ply.update_yaxes(title_text="Inflation (annual %)", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_ply)
    else:
        fig, axes = dark_fig(1, 3, figsize=(16, 5))
        for ax, (bloc, members) in zip(axes, BLOCS.items()):
            for i, country in enumerate([m for m in members if m in sel]):
                sub = dfe[dfe["Country Name"] == country]
                ax.plot(sub["Year"], sub["Inflation (annual %)"], label=country, color=LINE_COLORS_3[i % 3], lw=2, alpha=0.85)
            style_ax(ax, title=bloc, xlabel="Year", ylabel="Inflation (%)")
            ax.title.set_color(BLOC_COLORS[bloc])
            ax.legend(fontsize=9, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
        show(fig)

# ── Tab 3: GDP ────────────────────────────────────────────────────────────────
with t3:
    st.markdown("#### GDP (Current USD) Trends")
    st.caption("**Interactive:** hover or click to see values. One chart per bloc (EAC, ECOWAS, SADC).")
    if go is not None:
        from plotly.subplots import make_subplots
        fig_ply = make_subplots(rows=1, cols=3, subplot_titles=list(BLOCS.keys()),
                                horizontal_spacing=0.08)
        for col, (bloc, members) in enumerate(BLOCS.items(), 1):
            for i, country in enumerate([m for m in members if m in sel]):
                sub = dfe[dfe["Country Name"] == country]
                if not sub.empty:
                    fig_ply.add_trace(
                        go.Scatter(x=sub["Year"], y=sub["GDP (current US$)"] / 1e9, name=country,
                                  line=dict(color=LINE_COLORS_3[i % 3], width=2), mode="lines+markers", marker=dict(size=3),
                                  hovertemplate="Year: %{x}<br>GDP (B USD): %{y:.2f}<extra></extra>"),
                        row=1, col=col)
        fig_ply.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              title_text="GDP (Current USD) Trends", hovermode="x unified", margin=dict(t=50),
                              showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5))
        fig_ply.update_xaxes(title_text="Year", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_ply.update_yaxes(title_text="GDP (Billion USD)", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_ply)
    else:
        fig, axes = dark_fig(1, 3, figsize=(16, 5))
        for ax, (bloc, members) in zip(axes, BLOCS.items()):
            for i, country in enumerate([m for m in members if m in sel]):
                sub = dfe[dfe["Country Name"] == country]
                ax.plot(sub["Year"], sub["GDP (current US$)"] / 1e9, label=country, color=LINE_COLORS_3[i % 3], lw=2)
            style_ax(ax, title=bloc, xlabel="Year", ylabel="GDP (Billion USD)")
            ax.title.set_color(BLOC_COLORS[bloc])
            ax.legend(fontsize=9, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
        show(fig)

    st.markdown("#### Log-Transformed GDP (stabilised for modelling)")
    st.caption("**Interactive:** hover or click to see values. One chart per bloc (EAC, ECOWAS, SADC).")
    if go is not None:
        from plotly.subplots import make_subplots
        fig_log = make_subplots(rows=1, cols=3, subplot_titles=list(BLOCS.keys()),
                                horizontal_spacing=0.08)
        for col, (bloc, members) in enumerate(BLOCS.items(), 1):
            for i, country in enumerate([m for m in members if m in sel]):
                sub = dfe[dfe["Country Name"] == country]
                if not sub.empty:
                    fig_log.add_trace(
                        go.Scatter(x=sub["Year"], y=sub["log_GDP"], name=country,
                                  line=dict(color=LINE_COLORS_3[i % 3], width=2), mode="lines+markers", marker=dict(size=3),
                                  hovertemplate="Year: %{x}<br>log(1+GDP): %{y:.3f}<extra></extra>"),
                        row=1, col=col)
        fig_log.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              title_text="Log-Transformed GDP", hovermode="x unified", margin=dict(t=50),
                              showlegend=True, legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5))
        fig_log.update_xaxes(title_text="Year", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_log.update_yaxes(title_text="log(1+GDP)", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_log)
    else:
        fig, axes = dark_fig(1, 3, figsize=(16, 5))
        for ax, (bloc, members) in zip(axes, BLOCS.items()):
            for i, country in enumerate([m for m in members if m in sel]):
                sub = dfe[dfe["Country Name"] == country]
                ax.plot(sub["Year"], sub["log_GDP"], label=country, color=LINE_COLORS_3[i % 3], lw=2)
            style_ax(ax, title=bloc, xlabel="Year", ylabel="log(1+GDP)")
            ax.title.set_color(BLOC_COLORS[bloc])
            ax.legend(fontsize=9, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
        show(fig)

# ── Tab 4: Exchange Rate ──────────────────────────────────────────────────────
with t4:
    st.markdown("#### Exchange Rate (LCU/USD) Over Time")
    st.caption("**Interactive:** hover or click to see values. One chart per country.")
    if go is not None:
        from plotly.subplots import make_subplots
        fig_ply = make_subplots(rows=3, cols=3, subplot_titles=[f"{c} ({COUNTRY_BLOC[c]})" for c in COUNTRIES],
                                vertical_spacing=0.08, horizontal_spacing=0.06)
        for idx, country in enumerate(COUNTRIES):
            row, col = idx // 3 + 1, idx % 3 + 1
            sub = dfe[dfe["Country Name"] == country]
            if not sub.empty:
                c = BLOC_COLORS[COUNTRY_BLOC[country]]
                fig_ply.add_trace(
                    go.Scatter(x=sub["Year"], y=sub["Exchange Rate (LCU/USD)"], name=country,
                              line=dict(color=c, width=2), mode="lines+markers", marker=dict(size=4),
                              hovertemplate="Year: %{x}<br>LCU/USD: %{y:.2f}<extra></extra>"),
                    row=row, col=col)
        fig_ply.update_layout(
            paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
            title_text="Exchange Rate (LCU per USD)", showlegend=False, margin=dict(t=60))
        fig_ply.update_xaxes(title_text="Year", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_ply.update_yaxes(title_text="LCU/USD", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_ply)
    else:
        fig, axes = dark_fig(3, 3, figsize=(17, 10), constrained_layout=True)
        fig.suptitle("Exchange Rate (LCU per USD)", color="#E8EAF0", fontsize=13, fontweight="bold", y=1.01)
        for ax, country in zip(np.array(axes).flat, COUNTRIES):
            sub = dfe[dfe["Country Name"] == country]
            c   = BLOC_COLORS[COUNTRY_BLOC[country]]
            if not sub.empty:
                ax.plot(sub["Year"], sub["Exchange Rate (LCU/USD)"], color=c, lw=2)
            style_ax(ax, title=country, xlabel="Year", ylabel="LCU/USD")
        patches = [mpatches.Patch(color=v, label=k) for k, v in BLOC_COLORS.items()]
        fig.legend(handles=patches, loc="lower center", ncol=3, frameon=False, labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.03))
        show(fig)

# ── Tab 5: Correlation ────────────────────────────────────────────────────────
with t5:
    st.markdown("#### Correlation Matrix — Macroeconomic Variables")
    st.caption("**Interactive:** hover or click cells to see correlation values.")
    feat_cols = [TARGET, "log_GDP", "log_Exchange",
                 "Inflation (annual %)", "GDP_growth",
                 "Trade_rolling3", "Trade (% of GDP)_lag1"]
    feat_cols = [c for c in feat_cols if c in dfe.columns]
    corr = dfe[feat_cols].corr()
    if go is not None:
        fig_ply = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}", textfont={"size": 10},
            hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>"
        ))
        fig_ply.update_layout(
            paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
            title="Pearson Correlation Matrix", xaxis=dict(tickangle=-30, tickfont=dict(color="#7A85A0")),
            yaxis=dict(tickfont=dict(color="#7A85A0")), margin=dict(t=50))
        show_plotly(fig_ply)
    else:
        fig, ax = dark_fig(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True, linewidths=0.4,
                    annot_kws={"color": "white", "fontsize": 10}, cbar_kws={"shrink": 0.8})
        style_ax(ax, title="Pearson Correlation Matrix")
        ax.tick_params(colors="#C5CAD6")
        show(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Key correlations with Trade (% of GDP)**")
        trade_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3)
        for feat, val in trade_corr.items():
            color = "#43A047" if val > 0 else "#EF5350"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1E2235;'>"
                f"<span style='color:#C5CAD6'>{feat}</span>"
                f"<span style='color:{color};font-weight:700'>{val:+.3f}</span>"
                f"</div>", unsafe_allow_html=True
            )
    with c2:
        st.markdown("**Interpretation**")
        st.markdown("""
        - **Trade Lag-1** shows the highest correlation with current trade, 
          confirming strong serial autocorrelation in trade openness.
        - **Log GDP** and **Trade Rolling Mean** are positively correlated 
          with trade volumes, reflecting openness scaling with economic size.
        - **Inflation** shows a weak/negative relationship, consistent with 
          Dutch disease effects in commodity exporters.
        """)

# ── Tab 6: Summary Stats ─────────────────────────────────────────────────────
with t6:
    st.markdown("#### Descriptive Statistics by Country")
    desc = df.groupby("Country Name")[
        [TARGET, "GDP (current US$)", "Exchange Rate (LCU/USD)", "Inflation (annual %)"]
    ].agg(["mean", "std", "min", "max"]).round(2)
    st.dataframe(desc, use_container_width=True)
    st.download_button("⬇️ Download Stats", desc.to_csv(), "desc_stats.csv", "text/csv")

    st.markdown("#### Missing Data Summary (before interpolation)")
    st.markdown("""
    | Column | Missing % | Treatment |
    |---|---|---|
    | FDI (% of GDP) | 88.9% | Dropped entirely |
    | Trade (% of GDP) | 21.0% | Linear interpolation per country |
    | Inflation (annual %) | 16.6% | Linear interpolation per country |
    | Exchange Rate (LCU/USD) | 2.2% | Linear interpolation per country |
    | GDP (current US$) | 0.0% | None required |
    """)
