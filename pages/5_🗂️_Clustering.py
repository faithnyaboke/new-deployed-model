"""5_Clustering.py — K-Means & DBSCAN Country Clustering"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None
from utils import (
    inject_css, COUNTRIES, COUNTRY_BLOC, BLOCS, BLOC_COLORS,
    CLUSTER_FEATS, CLUSTER_PAL,
    dark_fig, style_ax, show, guard, show_plotly,
)

st.set_page_config(page_title="Clustering · SSA Trade", page_icon="🗂️", layout="wide")
inject_css()
guard(["df"], "⚠️ Please upload your dataset on the **Home** page first.")
df = st.session_state["df"]

st.markdown("## 🗂️ Country Clustering")
st.markdown("""
Group the 9 SSA countries into economic clusters based on aggregate macroeconomic profiles.
Two algorithms are applied: **K-Means** (partition-based) and **DBSCAN** (density-based).
Cluster validity is assessed using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz.
""")

# ── Cluster feature aggregation ────────────────────────────────────────────────
country_agg = df.groupby("Country Name")[CLUSTER_FEATS].mean().dropna()
st.markdown(
    f"**Clustering features** ({len(CLUSTER_FEATS)}): "
    + " · ".join(CLUSTER_FEATS)
)
with st.expander("📋 Country aggregate profiles"):
    display_df = country_agg.copy().round(3)
    display_df["Bloc"] = [COUNTRY_BLOC[c] for c in display_df.index]
    st.dataframe(display_df, use_container_width=True)

st.divider()

# ── Parameters ────────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Algorithm Parameters")
pc1, pc2, pc3 = st.columns(3)
k_max    = pc1.slider("Max K to evaluate", 3, 8, 6, key="cl_kmax")
eps_val  = pc2.slider("DBSCAN ε (epsilon)", 0.3, 3.0, 1.2, step=0.1, key="cl_eps")
min_samp = pc3.slider("DBSCAN min_samples", 1, 4, 2, key="cl_minsamp")

sc = StandardScaler()
Xc = sc.fit_transform(country_agg)

# ── K-selection plots ─────────────────────────────────────────────────────────
st.markdown("### Optimal K Selection")
st.caption("**Interactive:** hover to see K and metric value. Four panels unchanged.")
inertias, silhouettes, db_scores, ch_scores = [], [], [], []
from sklearn.metrics import calinski_harabasz_score

for k in range(2, k_max + 1):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(Xc)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(Xc, lbl))
    db_scores.append(davies_bouldin_score(Xc, lbl))
    ch_scores.append(calinski_harabasz_score(Xc, lbl))

best_k = 2 + int(np.argmax(silhouettes))

if go is not None and make_subplots is not None:
    k_vals = list(range(2, k_max + 1))
    plot_data = [
        ("Elbow — Inertia ↓",     inertias,    "#42A5F5"),
        ("Silhouette Score ↑",    silhouettes, "#66BB6A"),
        ("Davies-Bouldin Index ↓", db_scores,  "#EF5350"),
        ("Calinski-Harabasz ↑",   ch_scores,   "#FFA726"),
    ]
    fig_ply = make_subplots(rows=1, cols=4, subplot_titles=[t for t, _, _ in plot_data],
                            horizontal_spacing=0.06)
    for col, (title, vals, color) in enumerate(plot_data, 1):
        fig_ply.add_trace(
            go.Scatter(x=k_vals, y=vals, mode="lines+markers", line=dict(color=color, width=2),
                       marker=dict(size=8), showlegend=False,
                       hovertemplate="K: %{x}<br>Value: %{y:.4f}<extra></extra>"),
            row=1, col=col)
        fig_ply.add_vline(x=best_k, row=1, col=col, line_dash="dash", line_color="#FDD835", line_width=1.5)
    fig_ply.update_layout(title_text="K-Selection Metrics", paper_bgcolor="#0D0F18", plot_bgcolor="#13151F",
                         font=dict(color="#E8EAF0"), margin=dict(t=60))
    fig_ply.update_xaxes(title_text="K", gridcolor="#1E2235")
    fig_ply.update_yaxes(gridcolor="#1E2235")
    show_plotly(fig_ply)
else:
    fig, axes = dark_fig(1, 4, figsize=(18, 4))
    plot_data = [
        ("Elbow — Inertia ↓",       inertias,    "#42A5F5", "bo-"),
        ("Silhouette Score ↑",       silhouettes, "#66BB6A", "gs-"),
        ("Davies-Bouldin Index ↓",   db_scores,   "#EF5350", "r^-"),
        ("Calinski-Harabasz ↑",      ch_scores,   "#FFA726", "D-"),
    ]
    for ax, (title, vals, col, mk) in zip(axes, plot_data):
        ax.plot(range(2, k_max + 1), vals, mk, lw=2, markersize=8, color=col)
        ax.axvline(best_k, color="#FDD835", lw=1.5, ls="--", alpha=0.8)
        style_ax(ax, title=title, xlabel="K")
        ax.text(best_k + 0.05, ax.get_ylim()[1] * 0.95,
                f"K={best_k}", color="#FDD835", fontsize=9, fontweight="bold")
    fig.suptitle("K-Selection Metrics", color="#E8EAF0", fontsize=12, fontweight="bold")
    show(fig)

km1, km2, km3, km4 = st.columns(4)
km1.metric("Optimal K",     best_k,                       "by Silhouette")
km2.metric("Silhouette",    f"{max(silhouettes):.3f}",     "> 0.5 = good")
km3.metric("Davies-Bouldin",f"{db_scores[best_k-2]:.3f}",  "↓ = better")
km4.metric("CH Score",      f"{ch_scores[best_k-2]:.1f}",  "↑ = better")

st.divider()

# ── Final K-Means ─────────────────────────────────────────────────────────────
km_final  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_final.fit_predict(Xc)
country_agg["KMeans_Cluster"] = km_labels

# ── DBSCAN ────────────────────────────────────────────────────────────────────
dbscan    = DBSCAN(eps=eps_val, min_samples=min_samp)
db_labels = dbscan.fit_predict(Xc)
country_agg["DBSCAN_Cluster"] = db_labels
n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = int((db_labels == -1).sum())

# Quality metrics
sil_km = silhouette_score(Xc, km_labels)
db_km  = davies_bouldin_score(Xc, km_labels)
ch_km  = calinski_harabasz_score(Xc, km_labels)

# PCA for 2D vis
pca  = PCA(n_components=2, random_state=42)
Xpca = pca.fit_transform(Xc)
country_agg["PC1"] = Xpca[:, 0]
country_agg["PC2"] = Xpca[:, 1]

# ── Store for Cluster Stability tab ───────────────────────────────────────────
st.session_state.update({
    "km_final":      km_final,
    "sc_clust":      sc,
    "country_agg":   country_agg,
    "best_k":        best_k,
    "CLUSTER_FEATS": CLUSTER_FEATS,
})

# ── Side-by-side PCA plots ────────────────────────────────────────────────────
st.markdown("### Cluster Visualisation (PCA 2D Projection)")
st.caption(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance, "
           f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}%. **Interactive:** hover to see country, PC1, PC2, and cluster.")

pc1_pct = pca.explained_variance_ratio_[0] * 100
pc2_pct = pca.explained_variance_ratio_[1] * 100

if go is not None and make_subplots is not None:
    titles = [
        f"K-Means  K={best_k}  Sil={sil_km:.3f}  DB={db_km:.3f}  CH={ch_km:.1f}",
        f"DBSCAN  ε={eps_val}  {n_db_clusters} clusters  {n_noise} noise pts"
    ]
    fig_ply = make_subplots(rows=1, cols=2, subplot_titles=titles, horizontal_spacing=0.08)
    for col, (col_label, title) in enumerate(zip(["KMeans_Cluster", "DBSCAN_Cluster"], titles), 1):
        for label in sorted(country_agg[col_label].unique()):
            sub = country_agg[country_agg[col_label] == label]
            color = "#888888" if label == -1 else CLUSTER_PAL[int(label) % len(CLUSTER_PAL)]
            lbl = "Noise" if label == -1 else f"Cluster {int(label) + 1}"
            fig_ply.add_trace(
                go.Scatter(
                    x=sub["PC1"], y=sub["PC2"], mode="markers+text", name=lbl,
                    text=sub.index.tolist(), textposition="top center", textfont=dict(size=10, color="white"),
                    marker=dict(size=14, color=color, line=dict(width=1.5, color="white")),
                    hovertemplate="Country: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>" + lbl + "<extra></extra>",
                    showlegend=True),
                row=1, col=col)
    fig_ply.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                          margin=dict(t=60), legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig_ply.update_xaxes(title_text=f"PC1 ({pc1_pct:.1f}%)", gridcolor="#1E2235")
    fig_ply.update_yaxes(title_text=f"PC2 ({pc2_pct:.1f}%)", gridcolor="#1E2235")
    show_plotly(fig_ply)
else:
    fig, axes = dark_fig(1, 2, figsize=(16, 7))
    for ax, col_label, title in zip(
        axes,
        ["KMeans_Cluster", "DBSCAN_Cluster"],
        [f"K-Means  K={best_k}  Sil={sil_km:.3f}  DB={db_km:.3f}  CH={ch_km:.1f}",
         f"DBSCAN  ε={eps_val}  {n_db_clusters} clusters  {n_noise} noise pts"]
    ):
        for label in sorted(country_agg[col_label].unique()):
            sub   = country_agg[country_agg[col_label] == label]
            color = "#888888" if label == -1 else CLUSTER_PAL[label % len(CLUSTER_PAL)]
            lbl   = "Noise" if label == -1 else f"Cluster {label + 1}"
            ax.scatter(sub["PC1"], sub["PC2"], c=color, s=220,
                       label=lbl, zorder=3, edgecolors="white", lw=1.5)
            for c_name, row in sub.iterrows():
                ax.annotate(
                    c_name, (row["PC1"], row["PC2"]),
                    fontsize=9, ha="center", va="bottom",
                    xytext=(0, 12), textcoords="offset points",
                    fontweight="bold", color="white"
                )
        style_ax(ax, title=title,
                 xlabel=f"PC1 ({pc1_pct:.1f}%)",
                 ylabel=f"PC2 ({pc2_pct:.1f}%)")
        ax.legend(fontsize=9, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
    show(fig)

st.divider()

# ── Cluster profiles ──────────────────────────────────────────────────────────
clt1, clt2, clt3 = st.tabs(["🏷️ Assignments", "📊 Profiles", "🔍 Comparison"])

with clt1:
    km_col, db_col = st.columns(2)
    with km_col:
        st.markdown("#### K-Means Assignments")
        for c in sorted(country_agg["KMeans_Cluster"].unique()):
            members = country_agg[country_agg["KMeans_Cluster"] == c].index.tolist()
            color   = CLUSTER_PAL[c % len(CLUSTER_PAL)]
            blocs   = list(set(COUNTRY_BLOC[m] for m in members))
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:10px 14px;"
                f"background:#13151F;border-radius:0 10px 10px 0;margin-bottom:8px;'>"
                f"<b style='color:{color}'>Cluster {c+1}</b>"
                f"<span style='color:#7A85A0;font-size:.8rem;margin-left:8px'>"
                f"({', '.join(blocs)})</span><br>"
                f"<span style='color:#C5CAD6;font-size:.9rem'>"
                f"{' · '.join(members)}</span></div>",
                unsafe_allow_html=True
            )
    with db_col:
        st.markdown("#### DBSCAN Assignments")
        for c in sorted(country_agg["DBSCAN_Cluster"].unique()):
            members = country_agg[country_agg["DBSCAN_Cluster"] == c].index.tolist()
            color   = "#888888" if c == -1 else CLUSTER_PAL[c % len(CLUSTER_PAL)]
            label   = "Noise / Outliers" if c == -1 else f"Cluster {c+1}"
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:10px 14px;"
                f"background:#13151F;border-radius:0 10px 10px 0;margin-bottom:8px;'>"
                f"<b style='color:{color}'>{label}</b><br>"
                f"<span style='color:#C5CAD6;font-size:.9rem'>"
                f"{' · '.join(members)}</span></div>",
                unsafe_allow_html=True
            )
        st.caption(f"DBSCAN identified {n_db_clusters} core cluster(s) "
                   f"and {n_noise} noise point(s) at ε={eps_val}.")

with clt2:
    st.markdown("#### K-Means Cluster Profiles — Mean Feature Values")
    cp = country_agg.groupby("KMeans_Cluster")[CLUSTER_FEATS].mean().round(2)
    fig, axes = dark_fig(1, len(CLUSTER_FEATS), figsize=(16, 4.5),
                         constrained_layout=True)
    for ax, feat in zip(axes, CLUSTER_FEATS):
        vals = cp[feat]
        bars = ax.bar(
            [f"C{i+1}" for i in vals.index], vals,
            color=[CLUSTER_PAL[i % len(CLUSTER_PAL)] for i in vals.index],
            edgecolor="#0D0F18", width=0.6
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * abs(bar.get_height()),
                    f"{v:.1f}", ha="center", va="bottom",
                    color="white", fontsize=8)
        style_ax(ax, title=feat.split("(")[0].strip())
    show(fig)

    st.markdown("**Radar chart of cluster profiles**")
    angles = np.linspace(0, 2 * np.pi, len(CLUSTER_FEATS), endpoint=False).tolist()
    angles += angles[:1]
    # Normalise each feature to [0,1]
    cp_norm = (cp - cp.min()) / (cp.max() - cp.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True},
                           facecolor="#0D0F18")
    ax.set_facecolor("#13151F")
    ax.tick_params(colors="#7A85A0", labelsize=8)
    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [f.split("(")[0].strip() for f in CLUSTER_FEATS],
                      color="#C5CAD6", fontsize=9)
    ax.spines["polar"].set_edgecolor("#1E2235")
    for c in sorted(cp_norm.index):
        vals_r = cp_norm.loc[c].tolist() + [cp_norm.loc[c].iloc[0]]
        ax.plot(angles, vals_r, color=CLUSTER_PAL[c % len(CLUSTER_PAL)],
                lw=2, label=f"Cluster {c+1}")
        ax.fill(angles, vals_r, color=CLUSTER_PAL[c % len(CLUSTER_PAL)], alpha=0.13)
    ax.legend(loc="upper right", facecolor="#13151F",
              edgecolor="#1E2235", labelcolor="white", fontsize=9,
              bbox_to_anchor=(1.35, 1.15))
    ax.set_title("Cluster Radar Profile (normalised)", color="white",
                 fontsize=11, fontweight="bold", pad=20)
    st.pyplot(fig, use_container_width=True)
    plt.close()

with clt3:
    st.markdown("#### Algorithm Comparison")
    st.markdown("""
    | Metric | K-Means | DBSCAN |
    |--------|---------|--------|
    | Algorithm type | Partition-based | Density-based |
    | Requires K upfront | ✅ Yes | ❌ No |
    | Handles noise/outliers | ❌ No | ✅ Yes |
    | Cluster shape | Spherical (assumed) | Arbitrary |
    | Scalability | High | Medium |
    | Best for | Well-separated blobs | Irregular / noisy data |
    """)

    # Heatmap of cluster membership vs bloc
    bloc_cluster = pd.crosstab(
        [COUNTRY_BLOC[c] for c in country_agg.index],
        country_agg["KMeans_Cluster"],
        rownames=["Bloc"], colnames=["Cluster"]
    )
    fig, ax = dark_fig(figsize=(8, 4))
    sns.heatmap(bloc_cluster, annot=True, fmt="d", cmap="Blues",
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"color": "white", "fontsize": 11})
    style_ax(ax, title="Bloc × K-Means Cluster Cross-tabulation")
    ax.tick_params(colors="#C5CAD6")
    show(fig)

# ── Download ─────────────────────────────────────────────────────────────────
st.download_button(
    "⬇️ Download cluster assignments",
    country_agg[["KMeans_Cluster", "DBSCAN_Cluster"]].to_csv(),
    "cluster_assignments.csv", "text/csv"
)
