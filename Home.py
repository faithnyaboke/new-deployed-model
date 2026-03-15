"""
Home.py — SSA Trade Volume Analysis Dashboard
Landing page: upload data + navigation overview
"""
import streamlit as st
from utils import (
    inject_css, BLOCS, BLOC_COLORS, COUNTRIES,
    load_and_preprocess, engineer_features,
)

st.set_page_config(
    page_title="SSA Trade Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:28px 0 4px 0">
  <div style="font-size:2.3rem;font-weight:700;line-height:1.2;
    background:linear-gradient(135deg,#1E88E5,#42A5F5,#43A047);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    Predicting Trade Volumes in Sub-Saharan Africa
  </div>
  <div style="color:#7A85A0;margin-top:8px;font-size:1rem;">
    A Machine Learning &amp; Simulation-Based Approach &nbsp;·&nbsp;
    Strathmore University &nbsp;·&nbsp; MSc Data Science &amp; Analytics
  </div>
  <div style="color:#5A6070;font-size:.88rem;margin-top:4px;">
    Faith Nyaboke Orucho &nbsp;·&nbsp; 2025
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── DATA UPLOAD ───────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Dataset")
up_col, info_col = st.columns([1.3, 1], gap="large")

with up_col:
    st.markdown("Upload **`final_economic_dataset_with_inflation.csv`** to begin.")
    uploaded = st.file_uploader(
        "Drop CSV here", type=["csv"], label_visibility="collapsed",
        key="home_upload"
    )
    if uploaded:
        with st.spinner("Loading and preprocessing…"):
            raw = load_and_preprocess(uploaded.read())
            df  = engineer_features(raw)
            st.session_state["df"]  = df
            st.session_state["raw"] = raw
        st.success(
            f"✅ **{raw.shape[0]} rows** loaded — "
            f"**{raw['Country Name'].nunique()} countries**, "
            f"**{raw['Year'].min()}–{raw['Year'].max()}**"
        )
        with st.expander("📋 Data preview"):
            st.dataframe(raw.head(20), use_container_width=True)

with info_col:
    st.markdown("**Expected columns**")
    st.code(
        "Country Name, Country Code, Year\n"
        "Trade (% of GDP)\nGDP (current US$)\n"
        "Exchange Rate (LCU/USD)\nInflation (annual %)\n"
        "FDI (% of GDP)   ← auto-dropped", language="text"
    )
    st.markdown("**Countries & Blocs**")
    for bloc, members in BLOCS.items():
        col = BLOC_COLORS[bloc]
        st.markdown(
            f"<span style='color:{col};font-weight:600'>{bloc}</span>: "
            + ", ".join(members),
            unsafe_allow_html=True,
        )

# ── NAVIGATION GUIDE ──────────────────────────────────────────────────────────
st.divider()
st.markdown("### Navigation Guide")
st.caption("Use the sidebar to move between analysis sections in order.")

steps = [
    ("📊", "1_EDA",             "Exploratory Data Analysis",
     "Trade trends · Inflation · GDP · Correlation heatmap"),
    ("⚙️", "2_Features",        "Feature Engineering",
     "Log transforms · Lag features · Rolling means · Interaction terms"),
    ("🤖", "3_Regression",      "Regression Modelling",
     "Random Forest · Gradient Boosting · HistGBM · 5-fold CV · Hyperparameter tuning"),
    ("🔍", "4_SHAP",            "SHAP & Feature Importance",
     "Tree SHAP · Beeswarm · Dependence plots · Permutation importance"),
    ("🗂️", "5_Clustering",      "Country Clustering",
     "K-Means · DBSCAN · Elbow/Silhouette/Davies-Bouldin · PCA visualization"),
    ("🎲", "6_MonteCarlo",      "Monte Carlo Simulations",
     "5 shock scenarios · 1,000 stochastic runs · Trade distribution analysis"),
    ("🔒", "7_ClusterStability","Cluster Stability",
     "Re-cluster under each shock · stability heatmap · trajectory chart"),
]
cols = st.columns(4)
for i, (icon, _, title, desc) in enumerate(steps):
    with cols[i % 4]:
        st.markdown(
            f"""<div style="background:#13151F;border:1px solid #1E2235;
            border-radius:10px;padding:14px;margin-bottom:10px;height:130px;">
            <div style="font-size:1.5rem">{icon}</div>
            <div style="font-weight:700;color:#E8EAF0;font-size:.95rem;margin-top:6px">{title}</div>
            <div style="font-size:.78rem;color:#7A85A0;margin-top:4px">{desc}</div>
            </div>""",
            unsafe_allow_html=True,
        )

if "df" not in st.session_state:
    st.info("⬆️ Upload your dataset above to unlock all analysis pages.")
else:
    df = st.session_state["df"]
    st.divider()
    st.markdown("### Dataset Overview")
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Countries",   len(df["Country Name"].unique()))
    m2.metric("Blocs",       3, "EAC · ECOWAS · SADC")
    m3.metric("Year Range",  f"{df['Year'].min()}–{df['Year'].max()}")
    m4.metric("Rows",        df.shape[0])
    m5.metric("Features",    11, "after engineering")
    m6.metric("Target",      "Trade % GDP")
    st.caption("Navigate to **1_EDA** in the sidebar to begin the analysis.")
