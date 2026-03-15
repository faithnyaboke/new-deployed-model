"""3_Regression.py — Regression Modelling (RF · GBM · HistGBM · LSTM)"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from utils import (
    inject_css, COUNTRIES, COUNTRY_BLOC, BLOC_COLORS,
    FEATURES, FEAT_DISPLAY, DISP_FEATURES, TARGET,
    dark_fig, style_ax, show, guard, show_plotly,
)

# Optional: Plotly for interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Optional: Keras/TensorFlow for LSTM (sequential relationships over time)
try:
    from tensorflow import keras
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False

st.set_page_config(page_title="Regression · SSA Trade", page_icon="🤖", layout="wide")
inject_css()
guard(["df"], "⚠️ Please upload your dataset on the **Home** page first.")
df = st.session_state["df"]

st.markdown("## 🤖 Regression Modelling")
st.markdown("""
Predict trade volume (% of GDP) using tree-based ensemble models and **LSTM** (to capture sequential relationships over time, per methodology).
All models use the same 11 engineered features, with 80/20 train-test split
and 5-fold cross-validation for tree models. LSTM uses a sliding-window sequence of past years.
""")

# ── Dataset summary ───────────────────────────────────────────────────────────
df_model = df[FEATURES + [TARGET, "Country Name", "Year"]].dropna()
st.markdown(
    f"**Modelling dataset:** {df_model.shape[0]} observations × {len(FEATURES)} features"
    f" (dropped {len(df) - df_model.shape[0]} rows with NaN)"
)

def build_sequences(df_m, features, target, seq_len):
    """Build LSTM sequences per country: (X_seq, y, X_last, last_row_indices) for each sample."""
    X_seqs, ys, X_last, last_idx = [], [], [], []
    for country in df_m["Country Name"].unique():
        sub = df_m[df_m["Country Name"] == country].sort_values("Year")
        if len(sub) <= seq_len:
            continue
        arr = sub[features].values
        y_arr = sub[target].values
        inds = sub.index.tolist()
        for i in range(seq_len, len(sub)):
            X_seqs.append(arr[i - seq_len : i])
            ys.append(y_arr[i])
            X_last.append(arr[i])
            last_idx.append(inds[i])
    return np.array(X_seqs), np.array(ys), np.array(X_last), last_idx

# ── Settings ──────────────────────────────────────────────────────────────────
st.markdown("### ⚙️ Hyperparameters")
c1, c2, c3, c4, c5, c6 = st.columns(6)
test_size  = c1.slider("Test split %", 10, 30, 20, key="reg_test") / 100
n_trees    = c2.slider("n_estimators", 100, 600, 300, step=50, key="reg_n")
max_depth  = c3.slider("max_depth",    3,  12,   5,          key="reg_d")
lr         = c4.select_slider("GBM learning rate", [0.01, 0.05, 0.1, 0.2], 0.05, key="reg_lr")
run_cv     = c5.checkbox("Run 5-fold CV", value=True, key="reg_cv")
include_lstm = c6.checkbox("Include LSTM", value=HAS_KERAS, key="reg_lstm", disabled=not HAS_KERAS)
if not HAS_KERAS and include_lstm is False:
    st.caption("Install **tensorflow** to enable LSTM (sequential model per methodology).")
seq_len = 5
if include_lstm:
    seq_len = st.slider("LSTM sequence length (years)", 3, 15, 5, key="reg_seq")

if st.button("🚀  Train All Models", key="reg_train"):
    # Align samples: use sequence indices so tree and LSTM see same train/test split
    if include_lstm and HAS_KERAS:
        X_seq_all, y_all, X_last_all, seq_last_indices = build_sequences(df_model, FEATURES, TARGET, seq_len)
        n_seq = len(y_all)
        idx = np.arange(n_seq)
        i_tr, i_te = train_test_split(idx, test_size=test_size, random_state=42)
        X_tr_2d = X_last_all[i_tr]
        X_te_2d = X_last_all[i_te]
        X_tr_seq = X_seq_all[i_tr]
        X_te_seq = X_seq_all[i_te]
        y_tr = y_all[i_tr]
        y_te = y_all[i_te]
        st.session_state["reg_seq_test_indices"] = [seq_last_indices[i] for i in i_te]  # df_model row indices for country-level tab
    else:
        st.session_state.pop("reg_seq_test_indices", None)
        X = df_model[FEATURES].values
        y = df_model[TARGET].values
        X_tr_2d, X_te_2d, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_tr_seq = X_te_seq = None

    scaler = StandardScaler()
    scaler.fit(X_tr_2d)
    X_tr_s = scaler.transform(X_tr_2d)
    X_te_s = scaler.transform(X_te_2d)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=n_trees, max_depth=max_depth,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=n_trees, learning_rate=lr,
            max_depth=max_depth, subsample=0.8, random_state=42
        ),
        "HistGradient Boosting": HistGradientBoostingRegressor(
            max_iter=n_trees, learning_rate=lr,
            max_depth=max_depth, random_state=42
        ),
    }

    results = {}
    n_models = len(models) + (1 if include_lstm and HAS_KERAS else 0)
    prog = st.progress(0, "Training models…")
    for i, (name, mdl) in enumerate(models.items()):
        prog.progress(int(i / n_models * 80), f"Training {name}…")
        mdl.fit(X_tr_s, y_tr)
        y_pred = mdl.predict(X_te_s)
        res = {
            "model":  mdl,
            "y_pred": y_pred,
            "R2":     r2_score(y_te, y_pred),
            "RMSE":   float(np.sqrt(mean_squared_error(y_te, y_pred))),
            "MAE":    float(mean_absolute_error(y_te, y_pred)),
        }
        if run_cv:
            Xa = StandardScaler().fit_transform(df_model[FEATURES].values)
            ya = df_model[TARGET].values
            cv = cross_val_score(mdl, Xa, ya, cv=5, scoring="r2")
            res["CV_mean"] = float(cv.mean())
            res["CV_std"]  = float(cv.std())
        results[name] = res

    if include_lstm and HAS_KERAS and X_tr_seq is not None:
        prog.progress(85, "Training LSTM…")
        # Scale 3D sequences (same scaler as 2D)
        flat_tr = X_tr_seq.reshape(-1, X_tr_seq.shape[2])
        flat_te = X_te_seq.reshape(-1, X_te_seq.shape[2])
        scaler.fit(X_tr_2d)  # already fitted; use for consistency
        X_tr_seq_s = scaler.transform(flat_tr).reshape(X_tr_seq.shape)
        X_te_seq_s = scaler.transform(flat_te).reshape(X_te_seq.shape)
        n_feat = X_tr_seq.shape[2]
        lstm_model = Sequential([
            LSTM(64, activation="tanh", return_sequences=False, input_shape=(seq_len, n_feat)),
            Dropout(0.2),
            Dense(1),
        ])
        lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        lstm_model.fit(
            X_tr_seq_s, y_tr, validation_split=0.15, epochs=80, batch_size=16, callbacks=[early], verbose=0
        )
        y_pred_lstm = lstm_model.predict(X_te_seq_s, verbose=0).flatten()
        results["LSTM"] = {
            "model": lstm_model,
            "y_pred": y_pred_lstm,
            "R2":     r2_score(y_te, y_pred_lstm),
            "RMSE":   float(np.sqrt(mean_squared_error(y_te, y_pred_lstm))),
            "MAE":    float(mean_absolute_error(y_te, y_pred_lstm)),
            "CV_mean": None,
            "CV_std":  None,
        }
        st.session_state["reg_X_te_seq"] = X_te_seq_s  # for SHAP when best is LSTM
        # Base sequences (last seq_len rows per country, scaled) for Monte Carlo
        base_seqs = {}
        for country in COUNTRIES:
            sub = df_model[df_model["Country Name"] == country].sort_values("Year")
            if len(sub) >= seq_len:
                arr = sub[FEATURES].values[-seq_len:]
                base_seqs[country] = scaler.transform(arr)
        st.session_state["reg_base_sequences"] = base_seqs
        st.session_state["reg_seq_len"] = seq_len

    prog.progress(100, "Done!")
    best = max(results, key=lambda k: results[k]["R2"])
    st.session_state.update({
        "reg_results": results,
        "reg_y_test":  y_te,
        "reg_X_test":  X_te_s,
        "reg_best":    best,
        "best_model":  results[best]["model"],
        "scaler":      scaler,
        "df_model":    df_model,
        "include_lstm": include_lstm and HAS_KERAS,
        "seq_len":     seq_len if (include_lstm and HAS_KERAS) else None,
    })
    st.success(f"✅ Training complete. Best model: **{best}**  (R²={results[best]['R2']:.4f})")

# ── Results ───────────────────────────────────────────────────────────────────
if "reg_results" not in st.session_state:
    st.info("👆 Configure hyperparameters above and click **Train All Models**.")
    st.stop()

results   = st.session_state["reg_results"]
y_te      = st.session_state["reg_y_test"]
X_te_s    = st.session_state["reg_X_test"]
best      = st.session_state["reg_best"]
MCOL      = {
    "Random Forest":         "#1E88E5",
    "Gradient Boosting":     "#FF7043",
    "HistGradient Boosting": "#43A047",
    "LSTM":                  "#9C27B0",
}

st.divider()

# ── Metric cards ──────────────────────────────────────────────────────────────
st.markdown("### Model Performance Summary")
n_res = len(results)
cols = st.columns(n_res)
for col, (name, res) in zip(cols, results.items()):
    is_best = name == best
    c = MCOL.get(name, "#7A85A0")
    cv_html = (
        f"<div style='margin-top:8px;font-size:.8rem;color:#7A85A0;'>"
        f"5-fold CV R²: {res['CV_mean']:.3f} ± {res['CV_std']:.3f}</div>"
        if res.get("CV_mean") is not None else ""
    )
    with col:
        st.markdown(
            f"<div style='background:#13151F;"
            f"border:1px solid {'#43A047' if is_best else '#1E2235'};"
            f"border-top:4px solid {c};border-radius:10px;padding:18px;'>"
            f"<div style='font-weight:700;color:{c};font-size:1rem;'>"
            f"{name} {'🏆' if is_best else ''}</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
            f"gap:10px;margin-top:14px;text-align:center;'>"
            f"<div><div style='font-size:1.6rem;font-weight:700;color:#E8EAF0'>{res['R2']:.3f}</div>"
            f"<div style='font-size:.72rem;color:#7A85A0'>R²</div></div>"
            f"<div><div style='font-size:1.6rem;font-weight:700;color:#E8EAF0'>{res['RMSE']:.2f}</div>"
            f"<div style='font-size:.72rem;color:#7A85A0'>RMSE</div></div>"
            f"<div><div style='font-size:1.6rem;font-weight:700;color:#E8EAF0'>{res['MAE']:.2f}</div>"
            f"<div style='font-size:.72rem;color:#7A85A0'>MAE</div></div>"
            f"</div>{cv_html}</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Tabs for detailed charts ───────────────────────────────────────────────────
rt1, rt2, rt3, rt4, rt5 = st.tabs([
    "📈 Actual vs Predicted",
    "📉 Residuals",
    "📊 Metrics Comparison",
    "🌐 Country-Level",
    "📋 Full Metrics Table",
])

with rt1:
    st.caption("**Interactive:** hover or click points to see actual vs predicted values.")
    n_mod = len(results)
    if HAS_PLOTLY:
        fig_ply = make_subplots(rows=1, cols=n_mod, subplot_titles=[f"{name}" for name in results],
                                horizontal_spacing=0.06)
        for idx, (name, res) in enumerate(results.items()):
            fig_ply.add_trace(
                go.Scatter(x=y_te, y=res["y_pred"], mode="markers", name=name,
                           marker=dict(color=MCOL.get(name, "#7A85A0"), size=8, opacity=0.6),
                           hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"),
                row=1, col=idx + 1)
            mn, mx = float(y_te.min()), float(y_te.max())
            fig_ply.add_trace(
                go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(dash="dash", color="white", width=1.2)),
                row=1, col=idx + 1)
        fig_ply.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              showlegend=False, margin=dict(t=50))
        fig_ply.update_xaxes(title_text="Actual Trade % GDP", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_ply.update_yaxes(title_text="Predicted", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_ply)
    else:
        fig, axes = dark_fig(1, n_mod, figsize=(5 * n_mod, 5))
        axes = np.atleast_1d(axes)
        for ax, (name, res) in zip(axes.flat, results.items()):
            ax.scatter(y_te, res["y_pred"], alpha=0.4, color=MCOL.get(name, "#7A85A0"), s=22, zorder=3)
            mn, mx = y_te.min(), y_te.max()
            ax.plot([mn, mx], [mn, mx], "white", lw=1.3, ls="--", alpha=0.5)
            style_ax(ax, title=f"{name}\nR²={res['R2']:.3f}  RMSE={res['RMSE']:.2f}",
                     xlabel="Actual Trade % GDP", ylabel="Predicted")
        show(fig)

with rt2:
    st.caption("**Interactive:** hover to see counts and residual values.")
    n_mod = len(results)
    if HAS_PLOTLY:
        fig_ply = make_subplots(rows=1, cols=n_mod, subplot_titles=[f"{name} Residuals" for name in results],
                                horizontal_spacing=0.06)
        for idx, (name, res) in enumerate(results.items()):
            resid = y_te - res["y_pred"]
            fig_ply.add_trace(go.Histogram(x=resid, nbinsx=30, name=name, marker_color=MCOL.get(name, "#7A85A0"),
                                          hovertemplate="Residual: %{x:.2f}<br>Count: %{y}<extra></extra>"), row=1, col=idx + 1)
        fig_ply.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"), showlegend=False, margin=dict(t=50))
        fig_ply.update_xaxes(title_text="Residual", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_ply.update_yaxes(title_text="Count", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_ply)
        st.markdown("**Residuals vs Fitted Values**")
        fig_ply2 = make_subplots(rows=1, cols=n_mod, subplot_titles=list(results.keys()), horizontal_spacing=0.06)
        for idx, (name, res) in enumerate(results.items()):
            resid = y_te - res["y_pred"]
            fig_ply2.add_trace(go.Scatter(x=res["y_pred"], y=resid, mode="markers", name=name,
                                          marker=dict(color=MCOL.get(name, "#7A85A0"), size=6, opacity=0.5),
                                          hovertemplate="Fitted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>"), row=1, col=idx + 1)
        fig_ply2.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"), showlegend=False, margin=dict(t=50))
        fig_ply2.update_xaxes(title_text="Fitted", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_ply2.update_yaxes(title_text="Residual", gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_ply2)
    else:
        fig, axes = dark_fig(1, n_mod, figsize=(5 * n_mod, 4))
        axes = np.atleast_1d(axes)
        for ax, (name, res) in zip(axes.flat, results.items()):
            resid = y_te - res["y_pred"]
            ax.hist(resid, bins=30, color=MCOL.get(name, "#7A85A0"), alpha=0.82, edgecolor="#0D0F18")
            ax.axvline(0, color="white", lw=1.5, ls="--", alpha=0.6)
            ax.axvline(resid.mean(), color="#FDD835", lw=1.2, ls=":", alpha=0.8, label=f"mean={resid.mean():.2f}")
            style_ax(ax, title=f"{name} Residuals\nSkew={float(pd.Series(resid).skew()):.3f}", xlabel="Residual", ylabel="Count")
            ax.legend(fontsize=8, facecolor="#13151F", edgecolor="#1E2235", labelcolor="white")
        show(fig)
        st.markdown("**Residuals vs Fitted Values**")
        fig, axes = dark_fig(1, n_mod, figsize=(5 * n_mod, 4))
        axes = np.atleast_1d(axes)
        for ax, (name, res) in zip(axes.flat, results.items()):
            resid = y_te - res["y_pred"]
            ax.scatter(res["y_pred"], resid, alpha=0.35, color=MCOL.get(name, "#7A85A0"), s=18)
            ax.axhline(0, color="white", lw=1.3, ls="--", alpha=0.5)
            style_ax(ax, title=name, xlabel="Fitted Values", ylabel="Residual")
        show(fig)

with rt3:
    st.caption("**Interactive:** hover bars to see exact values.")
    names = list(results.keys())
    if HAS_PLOTLY:
        fig_ply = make_subplots(rows=1, cols=3, subplot_titles=["R²", "RMSE", "MAE"], horizontal_spacing=0.08)
        for col, metric in enumerate(["R2", "RMSE", "MAE"], 1):
            vals = [results[n][metric] for n in names]
            fig_ply.add_trace(
                go.Bar(x=names, y=vals, marker_color=[MCOL.get(n, "#7A85A0") for n in names],
                       text=[f"{v:.3f}" for v in vals], textposition="outside",
                       hovertemplate="%{x}<br>" + metric + ": %{y:.3f}<extra></extra>"),
                row=1, col=col)
        fig_ply.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"), showlegend=False, margin=dict(t=50))
        fig_ply.update_xaxes(tickangle=-15, gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        fig_ply.update_yaxes(gridcolor="#1E2235", tickfont=dict(color="#7A85A0"))
        show_plotly(fig_ply)
    else:
        fig, axes = dark_fig(1, 3, figsize=(14, 5))
        for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
            vals   = [results[n][metric] for n in names]
            colors = [MCOL.get(n, "#7A85A0") for n in names]
            bars   = ax.bar(names, vals, color=colors, width=0.5, edgecolor="#0D0F18")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals), f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
            style_ax(ax, title=metric, ylabel=metric)
            ax.set_xticklabels(names, rotation=15, ha="right", color="#C5CAD6")
            if results[names[0]].get("CV_mean") is not None:
                cv_vals = [results[n]["CV_mean"] for n in names]
                ax2 = ax.twinx()
                ax2.plot(names, cv_vals, "o--", color="#FDD835", lw=1.5, markersize=8, label="CV R²")
                ax2.set_ylabel("CV R²", color="#FDD835", fontsize=8)
                ax2.tick_params(colors="#FDD835", labelsize=8)
        show(fig)

with rt4:
    st.markdown("**Per-country prediction analysis (test-set observations)**")
    df_model_local = st.session_state["df_model"].copy()
    # Rebuild test indices (when LSTM used, test set is sequence-based so use stored row indices)
    if st.session_state.get("reg_seq_test_indices") is not None:
        idx_te = st.session_state["reg_seq_test_indices"]
        df_te = df_model_local.loc[idx_te].copy()
    else:
        from sklearn.model_selection import train_test_split as tts
        _, idx_te = tts(range(len(df_model_local)), test_size=test_size, random_state=42)
        df_te = df_model_local.iloc[list(idx_te)].copy()
    df_te["Predicted"] = results[best]["y_pred"]
    df_te["Residual"]  = df_te[TARGET] - df_te["Predicted"]
    df_te["AbsError"]  = df_te["Residual"].abs()

    country_err = df_te.groupby("Country Name")["AbsError"].agg(
        MAE_test="mean", Std="std"
    ).round(3).sort_values("MAE_test")
    st.dataframe(country_err, use_container_width=True)

    st.caption("**Interactive:** hover bars to see MAE and standard deviation.")
    if HAS_PLOTLY:
        fig_ply = go.Figure()
        fig_ply.add_trace(go.Bar(
            x=country_err.index, y=country_err["MAE_test"],
            error_y=dict(type="data", array=country_err["Std"], color="white"),
            marker_color=[BLOC_COLORS[COUNTRY_BLOC[c]] for c in country_err.index],
            hovertemplate="%{x}<br>MAE: %{y:.3f}<br>Std: %{customdata:.3f}<extra></extra>",
            customdata=country_err["Std"]
        ))
        fig_ply.update_layout(paper_bgcolor="#0D0F18", plot_bgcolor="#13151F", font=dict(color="#E8EAF0"),
                              title=f"Mean Absolute Error by Country — {best}", xaxis_title="Country", yaxis_title="MAE (pp of GDP)",
                              xaxis=dict(tickangle=-30, gridcolor="#1E2235", tickfont=dict(color="#7A85A0")),
                              yaxis=dict(gridcolor="#1E2235", tickfont=dict(color="#7A85A0")), margin=dict(t=50))
        show_plotly(fig_ply)
    else:
        fig, ax = dark_fig(figsize=(12, 4))
        ax.bar(country_err.index, country_err["MAE_test"],
               color=[BLOC_COLORS[COUNTRY_BLOC[c]] for c in country_err.index],
               edgecolor="#0D0F18", width=0.6)
        ax.errorbar(range(len(country_err)), country_err["MAE_test"],
                    yerr=country_err["Std"], fmt="none",
                    color="white", capsize=4, lw=1.2)
        style_ax(ax, title=f"Mean Absolute Error by Country — {best}",
                 xlabel="Country", ylabel="MAE (pp of GDP)")
        ax.set_xticks(range(len(country_err)))
        ax.set_xticklabels(country_err.index, rotation=30, ha="right", color="#C5CAD6")
        show(fig)

with rt5:
    rows = []
    for name, res in results.items():
        row = {"Model": name, "R²": round(res["R2"],4),
               "RMSE": round(res["RMSE"],3), "MAE": round(res["MAE"],3)}
        if res.get("CV_mean") is not None and res.get("CV_std") is not None:
            row["CV R² (mean)"] = round(res["CV_mean"], 4)
            row["CV R² (std)"]  = round(res["CV_std"], 4)
        rows.append(row)
    mdf = pd.DataFrame(rows)
    st.dataframe(mdf, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download metrics", mdf.to_csv(index=False),
                       "model_metrics.csv", "text/csv")
