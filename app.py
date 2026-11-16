import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Drinking-Water Potability Screening",
    page_icon="💧",
    layout="wide"
)

# --------------------------------------------------
# PATHS & DATA LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "water_potability.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

import numpy as np

@st.cache_data
def load_processed_data():
    try:
        X_tr = np.load(PROCESSED_DIR / "X_train.npy")
        y_tr = np.load(PROCESSED_DIR / "y_train.npy")
        X_va = np.load(PROCESSED_DIR / "X_val.npy")
        y_va = np.load(PROCESSED_DIR / "y_val.npy")
        X_te = np.load(PROCESSED_DIR / "X_test.npy")
        y_te = np.load(PROCESSED_DIR / "y_test.npy")
        
        feature_file = PROCESSED_DIR / "features.txt"
        if feature_file.exists():
            feature_names = [line.strip() for line in feature_file.read_text().splitlines() if line.strip()]
        else:
            feature_names = [f"Feature_{i}" for i in range(X_tr.shape[1])]
            
        return {
            "X_train": X_tr, "y_train": y_tr,
            "X_val": X_va, "y_val": y_va,
            "X_test": X_te, "y_test": y_te,
            "features": feature_names,
            "available": True
        }
    except Exception:
        return {"available": False}

try:
    df = load_dataset()
    proc_data = load_processed_data()
except Exception as e:
    st.error("Unable to load project dataset.")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("AquaMatrix: Water Potability ML Evaluation")
st.caption("Machine Learning Benchmark & Evaluation Dashboard")

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Dataset Inspection",
        "Model Performance",
        "Model Comparison",
        "India CPCB Case Study"
    ],
    index=0
)

# --------------------------------------------------
# PAGE 1: DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Samples", f"{len(df):,}")
    with c2:
        st.metric("Processed Features", "10 (9 Raw + 1 Ratio)")
    with c3:
        st.metric("Best Model (Val AUC)", "Tuned Random Forest")
    with c4:
        test_csv = OUTPUTS_DIR / "final_test_metrics.csv"
        if test_csv.exists():
            _tdf = pd.read_csv(test_csv)
            _prop = _tdf[_tdf["Model"] == "Proposed Model - Tuned Random Forest"]
            _acc_val = f"{float(_prop['Accuracy'].iloc[0]) * 100:.2f}%" if not _prop.empty else "66.67%"
            _spec_val = f"Unsafe Recall: {float(_prop['Unsafe_Recall_Spec'].iloc[0]) * 100:.2f}%" if not _prop.empty else ""
            st.metric("Test Accuracy (n=492)", _acc_val, _spec_val)
        else:
            st.metric("Test Accuracy (n=492)", "66.67%")

    st.markdown("---")

    st.subheader("All Evaluation Matrices")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Final Unseen Test Metrics (All Models)**")
        metrics_path = OUTPUTS_DIR / "final_test_metrics.csv"
        if metrics_path.exists():
            st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)

        st.markdown("**Confusion Matrix (Proposed Model)**")
        cm_csv_path = OUTPUTS_DIR / "final_confusion_matrix.csv"
        if cm_csv_path.exists():
            st.dataframe(pd.read_csv(cm_csv_path, index_col=0), use_container_width=True)

    with col_right:
        st.markdown("**Validation Results (491 Samples)**")
        validation_path = OUTPUTS_DIR / "validation_results.csv"
        if validation_path.exists():
            st.dataframe(pd.read_csv(validation_path), use_container_width=True, hide_index=True)

        st.markdown("**5-Fold Cross-Validation (Mean ROC-AUC)**")
        cv_path = OUTPUTS_DIR / "cross_validation_results.csv"
        if cv_path.exists():
            cv_df = pd.read_csv(cv_path).sort_values(by="CV_ROC_AUC_Mean", ascending=False)
            st.dataframe(cv_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Workflow Pipeline")
    st.write(
        "1. Dataset Collection → 2. Stratified 70/15/15 Split → "
        "3. Multivariate ExtraTrees Imputation → 4. Domain Feature Engineering (pH / Sulfate) → "
        "5. StandardScaler Normalization → 6. 5-Fold Stratified GridSearchCV → 7. Multi-Model Evaluation"
    )

# --------------------------------------------------
# PAGE 2: DATASET INSPECTION
# --------------------------------------------------

elif page == "Dataset Inspection":

    st.subheader("Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with c2:
        st.metric("Raw Features", df.shape[1] - 1)
    with c3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with c4:
        st.metric("Duplicates", int(df.duplicated().sum()))

    st.markdown("---")

    tab_raw, tab_mod, tab_split = st.tabs([
        "Raw Dataset",
        "Modified Dataset",
        "Split Partitions"
    ])

    with tab_raw:
        col_data, col_target = st.columns([1.3, 1])

        with col_data:
            st.markdown("**Missingness by Feature**")
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)

            if len(missing) > 0:
                missing_df = pd.DataFrame({
                    "Feature": missing.index,
                    "Missing Samples": missing.values,
                    "Percentage": [f"{(v / len(df) * 100):.2f}%" for v in missing.values]
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)

            st.markdown("**Raw Data Preview (First 10 Rows)**")
            st.dataframe(df.head(10), use_container_width=True)

        with col_target:
            st.markdown("**Target Class Distribution**")
            target_counts = df["Potability"].value_counts().sort_index()
            target_df = pd.DataFrame({
                "Class": ["0: Non-Potable", "1: Potable"],
                "Count": [int(target_counts.get(0, 0)), int(target_counts.get(1, 0))],
                "Proportion": [
                    f"{(target_counts.get(0, 0) / len(df) * 100):.2f}%",
                    f"{(target_counts.get(1, 0) / len(df) * 100):.2f}%"
                ]
            })
            st.dataframe(target_df, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.bar(["Non-Potable", "Potable"], target_df["Count"], color=["#4b5563", "#0284c7"])
            ax.set_ylabel("Count")
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 30, f"{yval:,}", ha="center", fontsize=9)
            ax.set_ylim(0, 2400)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_mod:
        if proc_data.get("available", False):
            st.markdown("**Modified & Imputed Feature Matrix (First 15 Rows)**")
            preview_proc_df = pd.DataFrame(
                proc_data["X_train"][:15],
                columns=proc_data["features"]
            ).round(4)
            preview_proc_df.insert(0, "Potability", proc_data["y_train"][:15])
            st.dataframe(preview_proc_df, use_container_width=True)

            st.markdown("**Dimension Comparison**")
            comp_dim_df = pd.DataFrame({
                "Property": ["Features", "Engineered Feature", "Missing NaNs", "Total Samples"],
                "Raw Dataset": ["9 Features", "None", "1,434 NaNs", "3,276 Samples"],
                "Modified Dataset": ["10 Features", "1 (ph_sulfate_ratio)", "0 NaNs (Fully Imputed)", "3,276 Samples"]
            })
            st.dataframe(comp_dim_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Processed dataset not found.")

    with tab_split:
        if proc_data.get("available", False):
            st.markdown("**Dataset Split Partitions**")
            split_df = pd.DataFrame({
                "Partition": ["Training Set (X_train)", "Validation Set (X_val)", "Unseen Test Set (X_test)", "Combined"],
                "Samples": [
                    f"{len(proc_data['y_train']):,} (70%)",
                    f"{len(proc_data['y_val']):,} (15%)",
                    f"{len(proc_data['y_test']):,} (15%)",
                    "3,276 (100%)"
                ],
                "Features Shape": [
                    f"{proc_data['X_train'].shape}",
                    f"{proc_data['X_val'].shape}",
                    f"{proc_data['X_test'].shape}",
                    "(3276, 10)"
                ],
                "Target Shape": [
                    f"{proc_data['y_train'].shape}",
                    f"{proc_data['y_val'].shape}",
                    f"{proc_data['y_test'].shape}",
                    "(3276,)"
                ]
            })
            st.dataframe(split_df, use_container_width=True, hide_index=True)

# --------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.subheader("Final Test Evaluation (492 Samples)")

    metrics_path = OUTPUTS_DIR / "final_test_metrics.csv"
    if metrics_path.exists():
        test_df = pd.read_csv(metrics_path)
        st.markdown("**Test Set Metrics (All Models)**")
        st.dataframe(test_df, use_container_width=True, hide_index=True)

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix (Proposed Model)**")
        cm_path = OUTPUTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        cm_csv_path = OUTPUTS_DIR / "final_confusion_matrix.csv"
        if cm_csv_path.exists():
            st.dataframe(pd.read_csv(cm_csv_path, index_col=0), use_container_width=True)

    with col_roc:
        st.markdown("**ROC Curve (Proposed Model)**")
        roc_path = OUTPUTS_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)

# --------------------------------------------------
# PAGE 4: MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.subheader("Model Comparison & Objectives Benchmark")

    tab_acc, tab_f1, tab_auc, tab_all, tab_imp, tab_shap = st.tabs([
        "Accuracy Comparison",
        "F1-Score Comparison",
        "ROC-AUC Comparison",
        "Combined Overview",
        "Imputation Strategies (Objective 2)",
        "SHAP Explainability (Objective 3)"
    ])

    with tab_acc:
        acc_path = OUTPUTS_DIR / "accuracy_comparison.png"
        if acc_path.exists():
            st.image(str(acc_path), use_container_width=True)

    with tab_f1:
        f1_path = OUTPUTS_DIR / "f1_comparison.png"
        if f1_path.exists():
            st.image(str(f1_path), use_container_width=True)

    with tab_auc:
        auc_path = OUTPUTS_DIR / "roc_auc_comparison.png"
        if auc_path.exists():
            st.image(str(auc_path), use_container_width=True)

    with tab_all:
        all_path = OUTPUTS_DIR / "model_comparison.png"
        if all_path.exists():
            st.image(str(all_path), use_container_width=True)

    with tab_imp:
        st.markdown("**Objective 2: Impact of Imputation Strategies on Unsafe Water Detection (N = 492)**")
        st.caption(
            "Controlled benchmark comparing imputation techniques under identical 70/15/15 leak-free splits "
            "to assess sensitivity in detecting contaminated/unsafe water."
        )
        imp_csv_path = OUTPUTS_DIR / "imputation_ablation_study.csv"
        if imp_csv_path.exists():
            st.dataframe(pd.read_csv(imp_csv_path), use_container_width=True, hide_index=True)
            st.info(
                "Health & Practical Impact: Simple median imputation collapses missing values to the neutral safe zone, "
                "leading to dangerous false negatives. Multivariate ExtraTrees MICE preserves feature covariance, achieving the "
                "highest test Recall (52.08%) and F1-score (55.71%) for identifying unsafe water."
            )

    with tab_shap:
        st.markdown("**Objective 3: Explainable AI (SHAP) & Alignment with WHO Health Guidance**")
        st.caption("Interpreting global physicochemical feature importance and verifying non-linear interactions against WHO standards.")
        
        c_imp, c_bar = st.columns([1, 1.2])
        with c_imp:
            st.markdown("**Mean Absolute SHAP Feature Ranking**")
            shap_csv = OUTPUTS_DIR / "shap_feature_importance.csv"
            if shap_csv.exists():
                st.dataframe(pd.read_csv(shap_csv), use_container_width=True, hide_index=True)
            st.info(
                "Key Domain Finding: pH (0.0427) and Sulfate (0.0413) are the primary drivers of potability. "
                "Crucially, the engineered interaction feature (ph_sulfate_ratio, 0.0307) ranks 3rd overall, "
                "surpassing Hardness, Chloramines, and Solids in predictive attribution."
            )
        
        with c_bar:
            shap_bar = OUTPUTS_DIR / "shap_bar_plot.png"
            if shap_bar.exists():
                st.image(str(shap_bar), use_container_width=True)

        st.markdown("---")
        c_sum, c_dep = st.columns([1.2, 1])
        with c_sum:
            st.markdown("**SHAP Beeswarm Summary Plot**")
            shap_sum = OUTPUTS_DIR / "shap_summary_plot.png"
            if shap_sum.exists():
                st.image(str(shap_sum), use_container_width=True)
        with c_dep:
            st.markdown("**SHAP Dependence Plot (pH-to-Sulfate Ratio)**")
            shap_dep = OUTPUTS_DIR / "shap_dependence_ratio.png"
            if shap_dep.exists():
                st.image(str(shap_dep), use_container_width=True)

# --------------------------------------------------
# PAGE 5: INDIA CPCB CASE STUDY
# --------------------------------------------------

elif page == "India CPCB Case Study":

    st.subheader("Separate Regional Case Study: Indian CPCB Surface Water Screening")
    st.caption(
        "Regional surface-water screening case study on official Central Pollution Control Board (CPCB) lentic water bodies "
        "(683 stations across 26 Indian States/UTs; distinct from the Kaggle drinking-water potability task)."
    )

    st.info(
        "**Methodological Disclosures & Provenance:**\n"
        "1. **Separate Problem (Ambient Surface Water vs. Drinking Potability):** Consumer drinking water potability in India is defined strictly by "
        "BIS IS 10500:2012 (requiring 0 coliforms/100mL). This CPCB dataset monitors ambient lentic surface water (Lakes, Tanks, Ponds) "
        "under the National Water Quality Monitoring Programme (NWMP). Its classes reflect ambient Water Quality Index (WQI) grades, "
        "so this is a separate ambient surface-water case study—not a generalization test of the drinking-water potability model.\n"
        "2. **Circularity (18 Features) vs. Partial-Formula Overlap (11 Features):** When trained on all 18 laboratory parameters, the 98.06% score "
        "(95% CI: [93.19%, 99.47%], 101/103) is circular WQI formula-emulation fidelity. The **Instant Field-Probe model (11 features)** "
        "achieves 92.23% [95% CI: 85.42%–96.01%] on n=103 without 5-day BOD or coliforms, but still uses DO, pH, EC, and Nitrate (which are WQI inputs — partial-formula overlap). "
        "Under 5-fold State-Grouped CV (`GroupKFold`, n=683), RF achieves 87.78% ± 3.17% Acc (0.8854 ± 0.0719 AUC) vs. a Logistic Regression subset-of-formula baseline at 85.15% ± 3.68% Acc (0.7918 ± 0.1408 AUC)."
    )

    india_csv_path = BASE_DIR / "data" / "india_water_quality.csv"
    india_df = pd.read_csv(india_csv_path) if india_csv_path.exists() else None

    if india_df is not None:
        total_stations = len(india_df)
        total_states = india_df["state"].nunique()
        acceptable_count = int(india_df["class"].isin(["Excellent", "Good"]).sum())
        degraded_count = total_stations - acceptable_count
    else:
        total_stations, total_states, acceptable_count, degraded_count = 683, 26, 369, 314

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Unique Stations (0 Duplicates)", f"{total_stations}")
    c2.metric("Indian States & UTs", f"{total_states}")
    c3.metric("Full Suite Binary Acc", "98.06%", "WQI Surrogate: 0.9977 AUC")
    c4.metric("4-Class WQI Acc", "93.20%", "Macro-F1: 0.9330")
    c5.metric("Instant Field-Probe Acc", "92.23%", "True Predictive: No 5-Day BOD")

    st.markdown(
        f"**Leak-Free Experimental Design:** `{total_stations}` distinct monitoring stations across 26 states "
        f"(`Acceptable Quality` = Excellent [156] + Good [213] = `{acceptable_count}`; "
        f"`Degraded Quality` = Poor [142] + Very Poor [172] = `{degraded_count}`). Zero repeated measures across stations "
        "(zero station-level leakage). Partitioned strictly into **70/15/15 stratified split** (478 Train / 102 Validation / 103 Unseen Test) "
        "with all domain ratios and scalers fitted on training folds only."
    )

    tab_bench, tab_probe, tab_states, tab_xai = st.tabs([
        "Binary & Multi-Class Benchmarks",
        "Instant Field-Probe vs. Lab Screening",
        "State-Wise Hotspots",
        "CPCB SHAP Explainability"
    ])

    with tab_bench:
        col_bin, col_multi = st.columns(2)
        with col_bin:
            st.markdown("**Binary Classification (Acceptable vs. Degraded — Unseen Test, n=103)**")
            bin_path = OUTPUTS_DIR / "india_binary_metrics.csv"
            if bin_path.exists():
                st.dataframe(pd.read_csv(bin_path), use_container_width=True, hide_index=True)
        with col_multi:
            st.markdown("**4-Class WQI Grade (Excellent/Good/Poor/Very Poor — Test, n=103)**")
            multi_path = OUTPUTS_DIR / "india_multiclass_metrics.csv"
            if multi_path.exists():
                st.dataframe(pd.read_csv(multi_path), use_container_width=True, hide_index=True)

        st.info(
            "Methodological Context: Because Indian CPCB stations exhibit pronounced biochemical oxygen demand and conductivity gradients, "
            "the Random Forest surrogate model reproduces the laboratory WQI boundary with 98.06% accuracy (0.9977 ROC-AUC) on the test split."
        )

    with tab_probe:
        st.markdown("**Instant Field-Probe Screening (11 Features) vs. Full Laboratory Suite (18 Features)**")
        probe_path = OUTPUTS_DIR / "india_probe_vs_lab.csv"
        if probe_path.exists():
            st.dataframe(pd.read_csv(probe_path), use_container_width=True, hide_index=True)

        st.info(
            "Field Screening Advantage: Measuring BOD and Coliform bacteria in the laboratory requires a 5-day incubation period. "
            "By relying strictly on instant field-probe sensors (DO, pH, EC, Nitrate, Temperature, and the pH-to-EC ratio), "
            "environmental officers achieve 92.23% binary screening accuracy and 0.9837 ROC-AUC on-site without any laboratory delay."
        )

    with tab_states:
        st.markdown("**State-Wise Surface Water Pollution Burden across India**")
        state_img = OUTPUTS_DIR / "india_state_pollution.png"
        if state_img.exists():
            st.image(str(state_img), use_container_width=True)

        col_st_tbl, col_wb_tbl = st.columns([1.6, 1])
        with col_st_tbl:
            st.markdown("**State-by-State Summary (Sorted by Monitoring Volume)**")
            state_csv = OUTPUTS_DIR / "india_state_summary.csv"
            if state_csv.exists():
                st.dataframe(pd.read_csv(state_csv), use_container_width=True, hide_index=True)
        with col_wb_tbl:
            if india_df is not None:
                st.markdown("**Water Body Type Breakdown**")
                wb_df = (
                    india_df.assign(Acceptable=india_df["class"].isin(["Excellent", "Good"]).astype(int))
                    .groupby("water_body_type", as_index=False)
                    .agg(Stations=("class", "count"), Acceptable_Count=("Acceptable", "sum"))
                )
                wb_df["Degraded_Count"] = wb_df["Stations"] - wb_df["Acceptable_Count"]
                wb_df["Degraded_Pct"] = (wb_df["Degraded_Count"] / wb_df["Stations"] * 100).round(2)
                st.dataframe(wb_df.sort_values("Stations", ascending=False), use_container_width=True, hide_index=True)

    with tab_xai:
        st.markdown("**Exact TreeExplainer SHAP Attribution on Indian CPCB Tuned Random Forest**")
        c_bar, c_bee = st.columns(2)
        with c_bar:
            st.markdown("**Global Mean |SHAP| Importance (Full Lab Suite)**")
            bar_img = OUTPUTS_DIR / "india_shap_bar.png"
            if bar_img.exists():
                st.image(str(bar_img), use_container_width=True)
        with c_bee:
            st.markdown("**SHAP Beeswarm Summary (Full Lab Suite)**")
            bee_img = OUTPUTS_DIR / "india_shap_summary.png"
            if bee_img.exists():
                st.image(str(bee_img), use_container_width=True)

        st.markdown("---")
        c_pimg, c_ptbl = st.columns([1.2, 1])
        with c_pimg:
            st.markdown("**SHAP Summary — Instant Field-Probe Model (11 Features)**")
            p_img = OUTPUTS_DIR / "india_probe_shap_summary.png"
            if p_img.exists():
                st.image(str(p_img), use_container_width=True)
        with c_ptbl:
            st.markdown("**Top SHAP Rankings (Full Suite vs. Instant Probe)**")
            shap_full_csv = OUTPUTS_DIR / "india_shap_importance.csv"
            shap_probe_csv = OUTPUTS_DIR / "india_probe_shap_importance.csv"
            if shap_full_csv.exists():
                st.caption("Full Laboratory Suite (DO_BOD_ratio ranks #1: 0.17527)")
                st.dataframe(pd.read_csv(shap_full_csv).head(10), use_container_width=True, hide_index=True)
            if shap_probe_csv.exists():
                st.caption("Instant Field-Probe Suite (pH_EC_ratio ranks #4: 0.04765)")
                st.dataframe(pd.read_csv(shap_probe_csv).head(6), use_container_width=True, hide_index=True)
