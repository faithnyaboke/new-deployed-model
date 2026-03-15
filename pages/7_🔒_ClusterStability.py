"""7_ClusterStability.py — Cluster Stability Under Shocks"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
from utils import (
    inject_css, COUNTRIES, COUNTRY_BLOC, BLOC_COLORS, BLOCS,
    CLUSTER_FEATS, CLUSTER_PAL,
    dark_fig, style_ax, show, guard, show_plotly,
)

st.set_page_config(page_title="Cluster Stability · SSA Trade", page_icon="🔒", layout="wide")
inject_css()
guard(
    ["km_final", "sc_clust", "country_agg", "best_k", "mc_results", "mc_scenarios"],
    "⚠️ Complete **Clustering** and **Monte Carlo** pages first."
)

km_final    = st.session_state["km_final"]
sc_clust    = st.session_state["sc_clust"]
country_agg = st.session_state["country_agg"]
best_k      = st.session_state["best_k"]
mc          = st.session_state["mc_results"]
scenarios   = st.session_state["mc_scenarios"]
sc_colors   = st.session_state.get("mc_sc_colors", {})

st.markdown("## 🔒 Cluster Stability Analysis")
st.markdown("""
**Research Question**: Do countries maintain their economic cluster membership when exposed to 
external macroeconomic shocks, or do shocks cause structural regime shifts?

**Methodology**:
1. Take the K-Means cluster assignments from the baseline.  
2. For each shock scenario, replace each country's mean trade volume with the Monte Carlo 
   mean prediction under that shock.  
3. Re-apply K-Means prediction (predict cluster, do not re-fit).  
4. Track whether each country's cluster assignment changes across scenarios.

A country is **structurally stable** if its cluster assignment never changes across any scenario.
""")

st.divider()

# ── Re-cluster under each scenario ────────────────────────────────────────────
stab_records = []
for scenario in scenarios:
    cd = country_agg[CLUSTER_FEATS].copy()
    for c in COUNTRIES:
        if c in cd.index:
            cd.loc[c, "Trade (% of GDP)"] = mc[scenario][c].mean()
    labels = km_final.predict(sc_clust.transform(cd))
    for country, label in zip(cd.index, labels):
        stab_records.append({
            "Country":  country,
            "Scenario": scenario,
            "Cluster":  int(label) + 1,
            "Bloc":     COUNTRY_BLOC[country],
        })

stab_df = pd.DataFrame(stab_records)
pivot   = stab_df.pivot(index="Country", columns="Scenario", values="Cluster")
pivot   = pivot[list(scenarios.keys())]
pivot["Stable?"] = pivot.apply(lambda r: r.nunique() == 1, axis=1)

stable   = [c for c in pivot.index if pivot.loc[c, "Stable?"]]
unstable = [c for c in pivot.index if not pivot.loc[c, "Stable?"]]

# ── Summary metrics ───────────────────────────────────────────────────────────
sm1, sm2, sm3, sm4, sm5 = st.columns(5)
sm1.metric("Countries Analysed", len(pivot))
sm2.metric("Structurally Stable", len(stable),
           f"{len(stable)/len(pivot)*100:.0f}% of total")
sm3.metric("Cluster-Shifting",    len(unstable),
           f"{len(unstable)/len(pivot)*100:.0f}% of total")
sm4.metric("Shock Scenarios",     len(scenarios))
sm5.metric("Clusters (K)",        best_k)

st.divider()

# ── Stability Heatmap ─────────────────────────────────────────────────────────
st.markdown("### Cluster Assignment Heatmap")
st.caption("Each cell = cluster assigned under that shock scenario. "
           "Red border = at least one cluster shift detected. **Interactive:** hover to see country, scenario, and cluster.")

hmap_data = pivot.drop(columns="Stable?").astype(int)
cmap      = mcolors.ListedColormap(CLUSTER_PAL[:best_k])

if go is not None:
    # Discrete colors for clusters 1..best_k
    z = hmap_data.values
    if best_k <= 1:
        scale = [[0, CLUSTER_PAL[0]], [1, CLUSTER_PAL[0]]]
    else:
        scale = [[(i - 1) / (best_k - 1), CLUSTER_PAL[(i - 1) % len(CLUSTER_PAL)]] for i in range(1, best_k + 1)]
    fig_ply = go.Figure(data=go.Heatmap(
        z=z, x=hmap_data.columns.tolist(), y=hmap_data.index.tolist(),
        colorscale=scale,
        text=z.astype(int).astype(str), texttemplate="%{text}", textfont=dict(color="white", size=12, family="bold"),
        hovertemplate="Country: %{y}<br>Scenario: %{x}<br>Cluster: %{z}<extra></extra>",
        zmin=1, zmax=best_k))
    shapes = []
    for i, country in enumerate(hmap_data.index):
        if country in unstable:
            shapes.append(dict(type="rect", x0=-0.5, x1=len(scenarios) - 0.5, y0=i - 0.5, y1=i + 0.5,
                               line=dict(color="#EF5350", width=3), fillcolor="rgba(0,0,0,0)"))
    fig_ply.update_layout(title="Cluster Stability Across Shock Scenarios  (red border = shift)",
                          xaxis_title="Shock Scenario", yaxis_title="Country",
                          paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                          xaxis=dict(tickangle=-20), yaxis=dict(autorange="reversed"),
                          margin=dict(t=50), shapes=shapes)
    show_plotly(fig_ply)
else:
    fig, ax = dark_fig(figsize=(15, 6))
    sns.heatmap(
        hmap_data, annot=True, fmt="d", cmap=cmap, ax=ax,
        linewidths=0.8, linecolor="#0D0F18",
        vmin=1, vmax=best_k,
        annot_kws={"color": "white", "fontsize": 12, "fontweight": "bold"},
        cbar_kws={"label": "Cluster", "ticks": list(range(1, best_k + 1))}
    )
    for i, country in enumerate(hmap_data.index):
        bloc = COUNTRY_BLOC[country]
        ax.text(-0.25, i + 0.5, bloc, ha="right", va="center",
                fontsize=9, color=BLOC_COLORS[bloc], fontweight="bold",
                transform=ax.get_yaxis_transform())
    for i, country in enumerate(hmap_data.index):
        if country in unstable:
            ax.add_patch(plt.Rectangle(
                (0, i), len(scenarios), 1,
                fill=False, edgecolor="#EF5350", lw=3.5, zorder=5
            ))
    style_ax(ax, title="Cluster Stability Across Shock Scenarios  (red border = shift)",
             xlabel="Shock Scenario", ylabel="Country")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", color="#C5CAD6")
    ax.tick_params(colors="#C5CAD6")
    show(fig)

st.divider()

# ── Stable / Unstable panels ──────────────────────────────────────────────────
st.markdown("### Country Stability Assessment")
cs1, cs2 = st.columns(2)

with cs1:
    st.markdown("#### ✅ Structurally Stable Countries")
    st.caption("Cluster assignment unchanged across all shock scenarios.")
    if stable:
        for country in sorted(stable):
            bloc = COUNTRY_BLOC[country]
            cl   = int(pivot.loc[country, "Baseline"])
            clr  = CLUSTER_PAL[(cl - 1) % len(CLUSTER_PAL)]
            st.markdown(
                f"<div style='background:#0D1F0D;border:1px solid #43A047;"
                f"border-radius:8px;padding:10px 16px;margin-bottom:6px;"
                f"display:flex;align-items:center;gap:12px;'>"
                f"<span style='font-size:1.3rem'>🟢</span>"
                f"<div><b style='color:#E8EAF0'>{country}</b>"
                f"<div style='color:#7A85A0;font-size:.82rem'>"
                f"{bloc} · Remains in Cluster {cl} across all scenarios</div></div></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No countries are structurally stable — all shift clusters under at least one scenario.")

with cs2:
    st.markdown("#### ⚠️ Cluster-Shifting Countries")
    st.caption("Cluster assignment changes under at least one shock scenario.")
    if unstable:
        for country in sorted(unstable):
            bloc  = COUNTRY_BLOC[country]
            cl_changes = list(pivot.loc[country, list(scenarios.keys())].unique())
            cl_changes_str = " → ".join([f"C{c}" for c in sorted(set(cl_changes))])
            worst = (
                [s for s in scenarios
                 if s != "Baseline"
                 and pivot.loc[country, s] != pivot.loc[country, "Baseline"]]
            )
            st.markdown(
                f"<div style='background:#1F0D0D;border:1px solid #EF5350;"
                f"border-radius:8px;padding:10px 16px;margin-bottom:6px;"
                f"display:flex;align-items:center;gap:12px;'>"
                f"<span style='font-size:1.3rem'>🔴</span>"
                f"<div><b style='color:#E8EAF0'>{country}</b>"
                f"<div style='color:#7A85A0;font-size:.82rem'>"
                f"{bloc} · {cl_changes_str}</div>"
                f"<div style='color:#EF9A9A;font-size:.78rem'>"
                f"Shifts under: {', '.join(worst)}</div></div></div>",
                unsafe_allow_html=True
            )
    else:
        st.success("✅ All countries are structurally stable — no cluster shifts detected under any scenario.")

st.divider()

# ── Trajectory chart ──────────────────────────────────────────────────────────
st.markdown("### Cluster Trajectory by Country")
st.caption("Lines show each country's cluster assignment as shock severity increases. **Interactive:** hover to see scenario, country, and cluster.")

scen_list = list(scenarios.keys())
xp = np.arange(len(scen_list))
if go is not None:
    fig_ply = go.Figure()
    for country in COUNTRIES:
        if country in pivot.index:
            vals = [int(pivot.loc[country, s]) for s in scen_list]
            color = BLOC_COLORS[COUNTRY_BLOC[country]]
            fig_ply.add_trace(go.Scatter(
                x=scen_list, y=vals, mode="lines+markers+text", name=country,
                line=dict(color=color, width=2), marker=dict(size=10),
                text=[country if i == len(scen_list) - 1 else "" for i in range(len(scen_list))],
                textposition="middle right", textfont=dict(size=9, color=color),
                customdata=[country] * len(scen_list),
                hovertemplate="Country: %{customdata}<br>Scenario: %{x}<br>Cluster: %{y}<extra></extra>"))
    fig_ply.update_layout(title="Cluster Trajectory Under Progressive Shock Severity",
                          xaxis_title="Shock Scenario", yaxis_title="Cluster Assignment",
                          yaxis=dict(tickmode="array", tickvals=list(range(1, best_k + 1)),
                                     ticktext=[f"Cluster {i}" for i in range(1, best_k + 1)]),
                          paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                          xaxis=dict(tickangle=-18), margin=dict(t=50),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    show_plotly(fig_ply)
else:
    import matplotlib.patches as mpatches
    fig, ax = dark_fig(figsize=(15, 5))
    for country in COUNTRIES:
        if country in pivot.index:
            vals = [int(pivot.loc[country, s]) for s in scen_list]
            ax.plot(xp, vals, "o-",
                    color=BLOC_COLORS[COUNTRY_BLOC[country]],
                    lw=2, markersize=9, label=country, alpha=0.85)
            ax.annotate(country, (xp[-1], vals[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=BLOC_COLORS[COUNTRY_BLOC[country]],
                        va="center")
    ax.set_xticks(xp)
    ax.set_xticklabels(scen_list, rotation=18, ha="right", color="#C5CAD6")
    ax.set_yticks(range(1, best_k + 1))
    ax.set_yticklabels([f"Cluster {i}" for i in range(1, best_k + 1)], color="#C5CAD6")
    style_ax(ax, title="Cluster Trajectory Under Progressive Shock Severity",
             ylabel="Cluster Assignment")
    bloc_patches = [mpatches.Patch(color=v, label=k) for k, v in BLOC_COLORS.items()]
    ax.legend(handles=bloc_patches, fontsize=9, ncol=3,
              facecolor="#13151F", edgecolor="#1E2235", labelcolor="white",
              loc="upper right")
    show(fig)

# ── Cluster membership change analysis ───────────────────────────────────────
st.divider()
st.markdown("### Cluster Membership Flow")
st.caption("How many countries move between clusters under each shock scenario?")

baseline_labels = hmap_data["Baseline"]
flow_rows = []
for scenario in [s for s in scen_list if s != "Baseline"]:
    shock_labels = hmap_data[scenario]
    changes = (baseline_labels != shock_labels).sum()
    flow_rows.append({
        "Scenario": scenario,
        "Countries Changed": int(changes),
        "% Changed":         round(changes / len(COUNTRIES) * 100, 1),
        "Countries Stable":  int(len(COUNTRIES) - changes),
    })
flow_df = pd.DataFrame(flow_rows)
st.dataframe(flow_df, use_container_width=True, hide_index=True)

if not flow_df.empty:
    st.caption("**Interactive:** hover to see scenario and number of countries shifted.")
    if go is not None:
        colors_bar = [sc_colors.get(s, "#42A5F5") for s in flow_df["Scenario"]]
        fig_ply = go.Figure(data=go.Bar(
            x=flow_df["Scenario"], y=flow_df["Countries Changed"],
            marker_color=colors_bar,
            text=[f"{v} countries" for v in flow_df["Countries Changed"]],
            textposition="outside", textfont=dict(color="white", size=10, family="bold"),
            hovertemplate="Scenario: %{x}<br>Countries shifted: %{y}<extra></extra>"))
        fig_ply.update_layout(title="Number of Countries Changing Cluster vs Baseline",
                             xaxis_title="Shock Scenario", yaxis_title="# Countries Shifted",
                             paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                             xaxis=dict(tickangle=-15), margin=dict(t=50))
        show_plotly(fig_ply)
    else:
        fig, ax = dark_fig(figsize=(10, 4))
        colors_bar = [sc_colors.get(s, "#42A5F5") for s in flow_df["Scenario"]]
        bars = ax.bar(flow_df["Scenario"], flow_df["Countries Changed"],
                      color=colors_bar, edgecolor="#0D0F18", width=0.55)
        for bar, v in zip(bars, flow_df["Countries Changed"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{v} countries", ha="center", va="bottom",
                    color="white", fontsize=9, fontweight="bold")
        style_ax(ax, title="Number of Countries Changing Cluster vs Baseline",
                 xlabel="Shock Scenario", ylabel="# Countries Shifted")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", color="#C5CAD6")
        show(fig)

# ── Research Implications ─────────────────────────────────────────────────────
st.divider()
st.markdown("### 📝 Research Implications")
n_stable_pct = round(len(stable) / len(pivot) * 100)
st.markdown(f"""
**Key Finding**: {len(stable)} of {len(pivot)} countries ({n_stable_pct}%) remain in the same 
cluster across all {len(scenarios)} shock scenarios, suggesting **structural resilience** 
in SSA trade regime groupings.

**Policy Implications**:
- Cluster membership primarily reflects long-run structural characteristics 
  (GDP size, trade openness, exchange rate regime), not short-term shocks.
- Countries identified as cluster-shifting are more **structurally vulnerable** 
  — policy interventions should target their macro stabilisation.
- Regionally coordinated trade policy within stable clusters (EAC/ECOWAS/SADC) 
  may be more effective than one-size-fits-all approaches.

**Methodological Note**: Cluster predictions use the frozen K-Means model trained on 
baseline data. If shocks were persistent and re-training were allowed, more instability 
might emerge — this represents a conservative bound on structural resilience.
""")

# ── Export ────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📋 Full stability table"):
    st.dataframe(pivot, use_container_width=True)
    st.download_button("⬇️ Download stability table",
                       pivot.to_csv(), "cluster_stability.csv", "text/csv")
st.download_button("⬇️ Download flow summary",
                   flow_df.to_csv(index=False), "cluster_flow.csv", "text/csv")
