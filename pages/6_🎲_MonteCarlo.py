"""6_MonteCarlo.py — Monte Carlo Shock Simulations"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as scipy_stats
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None
from utils import (
    inject_css, COUNTRIES, COUNTRY_BLOC, BLOCS, BLOC_COLORS,
    SCENARIOS, SCEN_COLORS, FEATURES, TARGET,
    dark_fig, style_ax, show, guard, show_plotly,
)

st.set_page_config(page_title="Monte Carlo · SSA Trade", page_icon="🎲", layout="wide")
inject_css()
guard(["best_model", "scaler"],
      "⚠️ Train the regression model on the **Regression** page first.")

best_model = st.session_state["best_model"]
scaler     = st.session_state["scaler"]
best_name  = st.session_state["reg_best"]
df         = st.session_state["df"]

st.markdown("## 🎲 Monte Carlo Simulations")
st.markdown(f"""
**Objective**: Assess how exogenous macroeconomic shocks propagate through trade volumes.

**Methodology**:  
1. Start from each country's most recent (last-available) feature vector.  
2. Apply deterministic shock to GDP, Inflation, and Exchange Rate.  
3. Add stochastic Gaussian noise (σ calibrated to historical volatility).  
4. Predict trade volume using the **{best_name}** model.  
5. Repeat **N** times to build a distribution of outcomes per country per scenario.

**Shock Scenarios**: Baseline · Mild · Moderate · Severe · COVID-like
""")

st.divider()

# ── Scenario table ────────────────────────────────────────────────────────────
st.markdown("### Shock Scenario Definitions")
scen_df = pd.DataFrame([
    {
        "Scenario":        s,
        "GDP Shock":       f"{v['gdp']*100:+.0f}%",
        "Inflation Shock": f"{v['inf']*100:+.0f}%",
        "Exchange Shock":  f"{v['exch']*100:+.0f}%",
        "Severity":        ("None" if v['gdp']==0
                            else "Mild" if v['gdp']>=-0.05
                            else "Moderate" if v['gdp']>=-0.10
                            else "Severe"),
    }
    for s, v in SCENARIOS.items()
])
st.dataframe(scen_df, use_container_width=True, hide_index=True)

# ── Simulation settings ───────────────────────────────────────────────────────
st.markdown("### Simulation Settings")
sc1, sc2, sc3 = st.columns(3)
n_sims    = sc1.select_slider("Number of simulations", [500, 1000, 2000, 5000], 1000, key="mc_n")
add_custom = sc2.checkbox("Add custom shock scenario", key="mc_custom")
seed       = sc3.number_input("Random seed", 1, 9999, 42, key="mc_seed")

custom_scenario = {}
if add_custom:
    st.markdown("#### Custom Shock Parameters")
    cx1, cx2, cx3 = st.columns(3)
    cg = cx1.slider("GDP shock %",       -50, 10, -12) / 100
    ci = cx2.slider("Inflation shock %",   0, 150, 25)  / 100
    ce = cx3.slider("Exchange shock %",    0,  60, 18)  / 100
    custom_scenario = {"gdp": cg, "inf": ci, "exch": ce}

if st.button("🚀  Run Monte Carlo Simulations", key="mc_run"):
    best_name = st.session_state.get("reg_best", "")
    # Build scenario set
    scenarios = dict(SCENARIOS)
    sc_colors = dict(SCEN_COLORS)
    if add_custom and custom_scenario:
        scenarios["Custom Shock"] = custom_scenario
        sc_colors["Custom Shock"] = "#00BCD4"

    # Use last-available year per country as base
    base = (
        df.groupby("Country Name")
        .last()
        .reset_index()
        .pipe(lambda d: d[d["Country Name"].isin(COUNTRIES)])
    )
    # Ensure all features present
    for feat in FEATURES:
        if feat not in base.columns:
            base[feat] = 0.0

    mc = {s: {c: [] for c in COUNTRIES} for s in scenarios}
    np.random.seed(int(seed))

    # Calibrate noise from historical data
    hist_std = {
        "gdp":  float(df.groupby("Country Name")["GDP_growth"].std().mean()) / 100,
        "inf":  float(df.groupby("Country Name")["Inflation (annual %)"].pct_change().std().mean()),
        "exch": float(df.groupby("Country Name")["log_Exchange"].diff().std().mean()),
    }

    prog = st.progress(0, "Initialising…")
    n_scen = len(scenarios)

    for si, (scenario, shocks) in enumerate(scenarios.items()):
        for sim_i in range(n_sims):
            sim = base.copy()
            n   = len(sim)

            # GDP shock + calibrated noise
            gdp_noise  = np.random.normal(0, max(hist_std["gdp"], 0.02), n)
            inf_noise  = np.random.normal(0, max(hist_std["inf"], 0.05), n)
            exch_noise = np.random.normal(0, max(hist_std["exch"], 0.03), n)

            sim["log_GDP"]     = sim["log_GDP"]     * (1 + shocks["gdp"]  + gdp_noise)
            sim["log_Exchange"]= sim["log_Exchange"] * (1 + shocks["exch"] + exch_noise)
            sim["Inflation (annual %)"] = sim["Inflation (annual %)"] * \
                                          (1 + shocks["inf"] + inf_noise)

            # Propagate to lag / rolling features (attenuated)
            sim["log_GDP_lag1"]          = sim["log_GDP_lag1"] * (1 + shocks["gdp"] * 0.8)
            sim["Trade (% of GDP)_lag1"] = sim["Trade (% of GDP)_lag1"] * (1 + shocks["gdp"] * 0.45)
            sim["Trade (% of GDP)_lag2"] = sim["Trade (% of GDP)_lag2"] * (1 + shocks["gdp"] * 0.30)
            sim["Trade_rolling3"]        = sim["Trade_rolling3"] * (1 + shocks["gdp"] * 0.40)
            sim["Inflation_rolling3"]    = sim["Inflation_rolling3"] * (1 + shocks["inf"]  * 0.85)
            sim["GDP_growth"]            = sim["GDP_growth"] + shocks["gdp"] * 100 + \
                                           np.random.normal(0, 1.5, n)
            sim["Trade_x_GDP"]           = sim["Trade (% of GDP)_lag1"] * sim["log_GDP"]

            # LSTM expects 3D (n_samples, seq_len, n_features): use base sequences with last row shocked
            use_lstm = best_name == "LSTM" and "reg_base_sequences" in st.session_state
            if use_lstm:
                base_seqs = st.session_state["reg_base_sequences"]
                seq_len = st.session_state["reg_seq_len"]
                X_list = []
                for country in sim["Country Name"].values:
                    seq = base_seqs.get(country)
                    if seq is not None:
                        row = sim[sim["Country Name"] == country][FEATURES].fillna(sim[FEATURES].median()).values
                        last_row = scaler.transform(row)
                        seq_new = np.array(seq, copy=True)
                        seq_new[-1] = last_row[0]
                        X_list.append(seq_new)
                    else:
                        X_list.append(np.zeros((seq_len, len(FEATURES))))
                X_sim_3d = np.stack(X_list)
                preds = best_model.predict(X_sim_3d, verbose=0).flatten()
            else:
                X_sim = scaler.transform(
                    sim[FEATURES].fillna(sim[FEATURES].median())
                )
                preds = best_model.predict(X_sim)

            for j, country in enumerate(sim["Country Name"].values):
                mc[scenario][country].append(float(preds[j]))

            if sim_i % 200 == 0:
                frac = (si * n_sims + sim_i) / (n_scen * n_sims)
                prog.progress(frac, f"{scenario} — sim {sim_i}/{n_sims}")

    prog.progress(1.0, "Done!")
    mc_arrays = {s: {c: np.array(v) for c, v in d.items()} for s, d in mc.items()}

    st.session_state["mc_results"]    = mc_arrays
    st.session_state["mc_scenarios"]  = scenarios
    st.session_state["mc_sc_colors"]  = sc_colors
    st.success(f"✅ {n_sims:,} simulations × {len(scenarios)} scenarios complete!")

# ── Results ───────────────────────────────────────────────────────────────────
if "mc_results" not in st.session_state:
    st.info("👆 Click **Run Monte Carlo Simulations** above to begin.")
    st.stop()

mc         = st.session_state["mc_results"]
scenarios  = st.session_state["mc_scenarios"]
sc_colors  = st.session_state["mc_sc_colors"]

# Build summary DataFrames
mc_mean = pd.DataFrame({s: {c: mc[s][c].mean() for c in COUNTRIES} for s in scenarios}).T
mc_std  = pd.DataFrame({s: {c: mc[s][c].std()  for c in COUNTRIES} for s in scenarios}).T
mc_pct  = ((mc_mean.sub(mc_mean.loc["Baseline"])) / mc_mean.loc["Baseline"] * 100).round(2)
mc_ci95 = mc_std * 1.96

st.divider()

mct1, mct2, mct3, mct4, mct5 = st.tabs([
    "📊 Distributions",
    "🌡️ Heatmap",
    "📉 % Change",
    "🔬 Country Deep-Dive",
    "📋 Tables & Export",
])

# ── Tab 1: Distributions ──────────────────────────────────────────────────────
with mct1:
    st.markdown("#### Predicted Trade Volume Distribution Under Each Shock Scenario")
    st.caption("**Interactive:** hover to see counts and values. One chart per country (unchanged).")
    if go is not None and make_subplots is not None:
        fig_ply = make_subplots(rows=3, cols=3,
                               subplot_titles=[f"{c}  ({COUNTRY_BLOC[c]})" for c in COUNTRIES],
                               vertical_spacing=0.08, horizontal_spacing=0.06)
        for idx, country in enumerate(COUNTRIES):
            row, col = idx // 3 + 1, idx % 3 + 1
            for si, scenario in enumerate(scenarios):
                vals = mc[scenario][country]
                fig_ply.add_trace(
                    go.Histogram(x=vals, nbinsx=40, opacity=0.38, name=scenario, histnorm="probability density",
                                 marker_color=sc_colors.get(scenario, "#AAAAAA"),
                                 showlegend=(idx == 0),
                                 hovertemplate="Trade % GDP: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>"),
                    row=row, col=col)
            for scenario in scenarios:
                mean_val = float(mc[scenario][country].mean())
                fig_ply.add_vline(x=mean_val, row=row, col=col, line_dash="dash",
                                  line_color=sc_colors.get(scenario, "#AAA"), line_width=1.5)
        fig_ply.update_layout(barmode="overlay", title_text="Monte Carlo: Trade Volume Distribution (% of GDP)",
                              paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              margin=dict(t=60, b=80), showlegend=True,
                              legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="center", x=0.5))
        fig_ply.update_xaxes(title_text="Trade % GDP", gridcolor="#1E2235")
        fig_ply.update_yaxes(title_text="Density", gridcolor="#1E2235")
        show_plotly(fig_ply)
    else:
        fig, axes = dark_fig(3, 3, figsize=(17, 12), constrained_layout=True)
        fig.suptitle("Monte Carlo: Trade Volume Distribution (% of GDP)",
                     color="#E8EAF0", fontsize=13, fontweight="bold", y=1.01)
        for ax, country in zip(np.array(axes).flat, COUNTRIES):
            for scenario in scenarios:
                col  = sc_colors.get(scenario, "#AAAAAA")
                vals = mc[scenario][country]
                ax.hist(vals, bins=40, alpha=0.38, color=col, density=True)
                ax.axvline(vals.mean(), color=col, lw=1.8, ls="--")
            style_ax(ax, title=f"{country}  ({COUNTRY_BLOC[country]})",
                     xlabel="Trade % GDP", ylabel="Density")
        patches = [mpatches.Patch(color=sc_colors.get(s, "#AAA"), alpha=0.75, label=s)
                   for s in scenarios]
        fig.legend(handles=patches, loc="lower center", ncol=min(len(scenarios), 5),
                   frameon=False, labelcolor="white", fontsize=9,
                   bbox_to_anchor=(0.5, -0.03))
        show(fig)

    # Key stats box
    st.markdown("**Key Statistics: Baseline vs Severe Shock**")
    stats_rows = []
    for country in COUNTRIES:
        b_vals = mc["Baseline"][country]
        s_vals = mc["Severe Shock"][country]
        stats_rows.append({
            "Country":       country,
            "Bloc":          COUNTRY_BLOC[country],
            "Baseline Mean": round(b_vals.mean(), 2),
            "Baseline CI95": f"±{b_vals.std()*1.96:.2f}",
            "Severe Mean":   round(s_vals.mean(), 2),
            "Severe CI95":   f"±{s_vals.std()*1.96:.2f}",
            "Δ (pp)":        round(s_vals.mean() - b_vals.mean(), 2),
            "Δ (%)":         round((s_vals.mean()-b_vals.mean())/b_vals.mean()*100, 1),
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

# ── Tab 2: Heatmap ────────────────────────────────────────────────────────────
with mct2:
    st.markdown("#### Mean Trade Volume by Scenario (% of GDP)")
    st.caption("**Interactive:** hover to see scenario, country, and mean value.")
    if go is not None:
        fig_ply = go.Figure(data=go.Heatmap(
            z=mc_mean.values, x=mc_mean.columns.tolist(), y=mc_mean.index.tolist(),
            colorscale="YlOrRd_r", text=np.round(mc_mean.values, 1).astype(str),
            texttemplate="%{text}", textfont=dict(color="white", size=10),
            hovertemplate="Scenario: %{y}<br>Country: %{x}<br>Mean: %{z:.2f} (% GDP)<extra></extra>"))
        fig_ply.update_layout(title="Mean Predicted Trade Volume Under Shock Scenarios",
                              paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              xaxis=dict(tickangle=-20), yaxis=dict(autorange="reversed"),
                              margin=dict(t=50))
        show_plotly(fig_ply)
    else:
        fig, ax = dark_fig(figsize=(14, 5))
        sns.heatmap(mc_mean, annot=True, fmt=".1f", cmap="YlOrRd_r",
                    ax=ax, linewidths=0.4,
                    annot_kws={"color": "white", "fontsize": 9},
                    cbar_kws={"label": "Mean Trade (% GDP)"})
        style_ax(ax, title="Mean Predicted Trade Volume Under Shock Scenarios")
        ax.tick_params(colors="#C5CAD6")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", color="#C5CAD6")
        show(fig)

    # Bar chart
    st.markdown("#### Grouped Bar Chart")
    st.caption("**Interactive:** hover to see mean trade (% GDP) per scenario and country.")
    if go is not None:
        fig_bar = go.Figure()
        x = np.arange(len(COUNTRIES))
        w = 0.8 / len(scenarios)
        for i, scenario in enumerate(scenarios):
            fig_bar.add_trace(go.Bar(
                x=x + i * w, y=mc_mean.loc[scenario].values, name=scenario,
                marker_color=sc_colors.get(scenario, "#AAA"), width=w,
                customdata=np.column_stack([COUNTRIES, [scenario] * len(COUNTRIES)]),
                hovertemplate="Country: %{customdata[0]}<br>Scenario: %{customdata[1]}<br>Mean: %{y:.2f} (% GDP)<extra></extra>"))
        fig_bar.update_layout(barmode="group", xaxis=dict(tickvals=x + w * (len(scenarios) - 1) / 2, ticktext=COUNTRIES, tickangle=-20),
                              title="Mean Trade Volume by Scenario", yaxis_title="Mean Predicted Trade (% of GDP)",
                              paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"))
        show_plotly(fig_bar)
    else:
        fig, ax = dark_fig(figsize=(15, 6))
        x = np.arange(len(COUNTRIES))
        w = 0.8 / len(scenarios)
        for i, scenario in enumerate(scenarios):
            ax.bar(x + i * w, mc_mean.loc[scenario], width=w, label=scenario,
                   color=sc_colors.get(scenario, "#AAA"), alpha=0.85, edgecolor="#0D0F18")
        ax.set_xticks(x + w * (len(scenarios) - 1) / 2)
        ax.set_xticklabels(COUNTRIES, rotation=20, ha="right", color="#C5CAD6")
        style_ax(ax, title="Mean Trade Volume by Scenario",
                 ylabel="Mean Predicted Trade (% of GDP)")
        ax.legend(fontsize=9, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
        show(fig)

# ── Tab 3: % Change ───────────────────────────────────────────────────────────
with mct3:
    st.markdown("#### % Change in Trade Volume from Baseline")
    non_base = [s for s in scenarios if s != "Baseline"]
    if non_base:
        st.caption("**Interactive:** hover to see scenario, country, and % change.")
        if go is not None:
            df_pct = mc_pct.loc[non_base]
            fig_ply = go.Figure(data=go.Heatmap(
                z=df_pct.values, x=df_pct.columns.tolist(), y=df_pct.index.tolist(),
                colorscale="RdYlGn", zmid=0, text=np.round(df_pct.values, 1).astype(str),
                texttemplate="%{text}", textfont=dict(color="white", size=10),
                hovertemplate="Scenario: %{y}<br>Country: %{x}<br>% Change: %{z:.2f}%<extra></extra>"))
            fig_ply.update_layout(title="% Change in Trade Volume from Baseline",
                                 paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                                 xaxis=dict(tickangle=-30), yaxis=dict(autorange="reversed"), margin=dict(t=50))
            show_plotly(fig_ply)
        else:
            fig, ax = dark_fig(figsize=(13, 5))
            sns.heatmap(
                mc_pct.loc[non_base],
                annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.4,
                annot_kws={"color": "white", "fontsize": 9},
                cbar_kws={"label": "% Change from Baseline"}
            )
            style_ax(ax, title="% Change in Trade Volume from Baseline")
            ax.tick_params(colors="#C5CAD6")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color="#C5CAD6")
            show(fig)

    # Most vulnerable countries
    if "Severe Shock" in mc_pct.index:
        vuln = mc_pct.loc["Severe Shock"].sort_values()
        st.markdown("**Country Vulnerability Under Severe Shock (↓ worst)**")
        st.caption("**Interactive:** hover to see country and % change.")
        if go is not None:
            colors = ["#EF5350" if v < 0 else "#43A047" for v in vuln]
            fig_ply = go.Figure(data=go.Bar(y=vuln.index.tolist(), x=vuln.values, orientation="h",
                                            marker_color=colors,
                                            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
                                            text=[f"{v:+.1f}%" for v in vuln], textposition="outside", textfont=dict(color="white")))
            fig_ply.update_layout(title="% Change Under Severe Shock", xaxis_title="% Change from Baseline",
                                 paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                                 yaxis=dict(autorange="reversed"), margin=dict(t=50), showlegend=False)
            fig_ply.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
            show_plotly(fig_ply)
        else:
            fig, ax = dark_fig(figsize=(12, 4))
            colors  = ["#EF5350" if v < 0 else "#43A047" for v in vuln]
            ax.barh(vuln.index, vuln, color=colors, edgecolor="#0D0F18", height=0.65)
            ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.4)
            for i, (country, v) in enumerate(vuln.items()):
                ax.text(v - 0.3 if v < 0 else v + 0.1, i,
                        f"{v:+.1f}%", va="center", ha="right" if v < 0 else "left",
                        color="white", fontsize=9, fontweight="bold")
            style_ax(ax, title="% Change Under Severe Shock", xlabel="% Change from Baseline")
            show(fig)

# ── Tab 4: Country Deep-Dive ──────────────────────────────────────────────────
with mct4:
    sel_country = st.selectbox("Select country", COUNTRIES, key="mc_country")
    st.markdown(f"#### {sel_country} — Full Scenario Analysis")

    # Distribution overlay
    st.caption("**Interactive:** hover to see density and values.")
    if go is not None:
        fig_ply = go.Figure()
        for scenario in scenarios:
            vals = mc[scenario][sel_country]
            color = sc_colors.get(scenario, "#AAA")
            fig_ply.add_trace(go.Histogram(x=vals, nbinsx=50, opacity=0.4, name=scenario,
                                          marker_color=color, histnorm="probability density",
                                          hovertemplate="Trade % GDP: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>"))
            mean_val = float(vals.mean())
            fig_ply.add_vline(x=mean_val, line_dash="dash", line_color=color, line_width=2)
            kde = scipy_stats.gaussian_kde(vals)
            x_kde = np.linspace(float(vals.min()), float(vals.max()), 200)
            fig_ply.add_trace(go.Scatter(x=x_kde, y=kde(x_kde), mode="lines", name=f"{scenario} KDE",
                                         line=dict(color=color, width=1.5), showlegend=False,
                                         hovertemplate="Trade % GDP: %{x:.2f}<br>KDE: %{y:.3f}<extra></extra>"))
        fig_ply.update_layout(barmode="overlay", title=f"{sel_country} — Trade Volume Distribution",
                              xaxis_title="Trade % GDP", yaxis_title="Density",
                              paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              legend=dict(orientation="h"))
        show_plotly(fig_ply)
    else:
        fig, ax = dark_fig(figsize=(12, 5))
        for scenario in scenarios:
            vals = mc[scenario][sel_country]
            color = sc_colors.get(scenario, "#AAA")
            ax.hist(vals, bins=50, alpha=0.4, color=color, density=True, label=scenario)
            ax.axvline(vals.mean(), color=color, lw=2, ls="--",
                       label=f"{scenario}: {vals.mean():.1f}±{vals.std():.1f}")

            # Add KDE
            kde = scipy_stats.gaussian_kde(vals)
            x_kde = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(x_kde, kde(x_kde), color=color, lw=1.5, alpha=0.8)

        style_ax(ax, title=f"{sel_country} — Trade Volume Distribution",
                 xlabel="Trade % GDP", ylabel="Density")
        ax.legend(fontsize=8, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white",
                  ncol=2)
        show(fig)

    # Numerical summary table
    rows = []
    b_mean = mc["Baseline"][sel_country].mean()
    for scenario in scenarios:
        v = mc[scenario][sel_country]
        ci_lo = np.percentile(v, 2.5)
        ci_hi = np.percentile(v, 97.5)
        rows.append({
            "Scenario":   scenario,
            "Mean":       round(v.mean(), 3),
            "Std Dev":    round(v.std(),  3),
            "95% CI Lo":  round(ci_lo, 3),
            "95% CI Hi":  round(ci_hi, 3),
            "Δ Mean (pp)":round(v.mean() - b_mean, 3),
            "Δ % from Baseline": round((v.mean()-b_mean)/b_mean*100, 2),
            "Prob < Baseline": round(float((v < b_mean).mean())*100, 1),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Scenario fan chart over simulations
    st.markdown("#### Simulation Fan Chart — First 200 Draws")
    st.caption("**Interactive:** hover to see simulation index and predicted trade % GDP.")
    sims_to_show = min(200, len(mc["Baseline"][sel_country]))
    if go is not None:
        fig_ply = go.Figure()
        for scenario in scenarios:
            vals = mc[scenario][sel_country][:sims_to_show]
            color = sc_colors.get(scenario, "#AAA")
            fig_ply.add_trace(go.Scatter(x=list(range(sims_to_show)), y=vals.tolist(), mode="markers",
                                         name=f"{scenario}: μ={vals.mean():.1f}", marker=dict(size=4, color=color, opacity=0.5),
                                         hovertemplate="Sim: %{x}<br>Trade % GDP: %{y:.2f}<extra></extra>"))
            fig_ply.add_hline(y=float(vals.mean()), line_dash="dash", line_color=color, line_width=1.5)
        fig_ply.update_layout(title=f"{sel_country} — Simulation Fan Chart",
                              xaxis_title="Simulation #", yaxis_title="Predicted Trade % GDP",
                              paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              legend=dict(orientation="h"))
        show_plotly(fig_ply)
    else:
        fig, ax = dark_fig(figsize=(12, 4))
        for si, scenario in enumerate(scenarios):
            vals = mc[scenario][sel_country][:sims_to_show]
            ax.plot(range(sims_to_show), vals, ".", color=sc_colors.get(scenario, "#AAA"),
                    alpha=0.4, markersize=3)
            ax.axhline(vals.mean(), color=sc_colors.get(scenario, "#AAA"),
                       lw=1.8, ls="--", alpha=0.9, label=f"{scenario}: μ={vals.mean():.1f}")
        style_ax(ax, title=f"{sel_country} — Simulation Fan Chart",
                 xlabel="Simulation #", ylabel="Predicted Trade % GDP")
        ax.legend(fontsize=8, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white",
                  ncol=2)
        show(fig)

# ── Tab 5: Export ─────────────────────────────────────────────────────────────
with mct5:
    st.markdown("#### Mean ± 95% CI by Scenario and Country")
    ci_df = pd.DataFrame({
        s: {c: f"{mc[s][c].mean():.2f} ± {mc[s][c].std()*1.96:.2f}"
            for c in COUNTRIES}
        for s in scenarios
    }).T
    st.dataframe(ci_df, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button("⬇️ Mean trade volumes",
                           mc_mean.round(3).to_csv(), "mc_means.csv", "text/csv")
    with col_b:
        st.download_button("⬇️ % Change from baseline",
                           mc_pct.round(3).to_csv(), "mc_pct_change.csv", "text/csv")
    with col_c:
        st.download_button("⬇️ 95% CI table",
                           ci_df.to_csv(), "mc_ci95.csv", "text/csv")
