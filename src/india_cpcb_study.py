import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from scipy.stats import binomtest


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "india_water_quality.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


FULL_RAW_FEATURES = [
    "Temp_min", "DO_min", "pH_min", "EC_min", "BOD_min", "Nitrate_min", "FC_min", "TC_min",
    "Temp_max", "DO_max", "pH_max", "EC_max", "BOD_max", "Nitrate_max", "FC_max", "TC_max"
]
FULL_FEATURE_NAMES = FULL_RAW_FEATURES + ["DO_BOD_ratio", "pH_EC_ratio"]

PROBE_RAW_FEATURES = [
    "Temp_min", "DO_min", "pH_min", "EC_min", "Nitrate_min",
    "Temp_max", "DO_max", "pH_max", "EC_max", "Nitrate_max"
]
PROBE_FEATURE_NAMES = PROBE_RAW_FEATURES + ["pH_EC_ratio"]

CLASS_ORDER = ["Excellent", "Good", "Poor", "Very Poor"]
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_ORDER)}


def add_full_ratios(arr: np.ndarray) -> np.ndarray:
    """Append DO_BOD_ratio and pH_EC_ratio to the 16-feature CPCB matrix."""
    do_min = arr[:, 1]
    bod_max = arr[:, 12]
    ph_max = arr[:, 10]
    ec_max = arr[:, 11]
    do_bod_ratio = (do_min / (bod_max + 0.1)).reshape(-1, 1)
    ph_ec_ratio = (ph_max / (ec_max + 1.0)).reshape(-1, 1)
    return np.hstack([arr, do_bod_ratio, ph_ec_ratio])


def add_probe_ratio(arr: np.ndarray) -> np.ndarray:
    """Append pH_EC_ratio to the 10-feature instant field-probe matrix."""
    ph_max = arr[:, 7]
    ec_max = arr[:, 8]
    ph_ec_ratio = (ph_max / (ec_max + 1.0)).reshape(-1, 1)
    return np.hstack([arr, ph_ec_ratio])


def build_preprocessor(imputer, ratio_fn) -> Pipeline:
    """Create a leak-free preprocessing pipeline: impute -> domain ratios -> scale."""
    return Pipeline(
        steps=[
            ("imputer", imputer),
            ("ratios", FunctionTransformer(ratio_fn, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )


def get_base_models() -> dict:
    """Return the five standard comparison classifiers with consistent regularization."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=4,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    }


def evaluate_binary(model, X_eval: np.ndarray, y_eval: np.ndarray) -> dict:
    """Evaluate binary classification metrics."""
    preds = model.predict(X_eval)
    probs = model.predict_proba(X_eval)[:, 1]
    return {
        "Accuracy": round(accuracy_score(y_eval, preds), 4),
        "Precision": round(precision_score(y_eval, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_eval, preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_eval, preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_eval, probs), 4),
    }


def evaluate_multiclass(model, X_eval: np.ndarray, y_eval: np.ndarray) -> dict:
    """Evaluate 4-class classification metrics."""
    preds = model.predict(X_eval)
    return {
        "Accuracy": round(accuracy_score(y_eval, preds), 4),
        "Macro-Precision": round(precision_score(y_eval, preds, average="macro", zero_division=0), 4),
        "Macro-Recall": round(recall_score(y_eval, preds, average="macro", zero_division=0), 4),
        "Macro-F1": round(f1_score(y_eval, preds, average="macro", zero_division=0), 4),
        "Weighted-F1": round(f1_score(y_eval, preds, average="weighted", zero_division=0), 4),
    }


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray, scoring: str) -> tuple:
    """Tune Random Forest via 5-fold Stratified GridSearchCV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=4,
    )
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [8, 12],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    gs = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=4,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def extract_positive_shap(model, X_eval: np.ndarray) -> np.ndarray:
    """Extract class-1 (Safe) SHAP values reliably across SHAP versions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)
    if isinstance(shap_values, list):
        return shap_values[1]
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def main() -> None:
    print("=" * 72)
    print("INDIAN CPCB SURFACE WATER QUALITY EXTERNAL VALIDATION STUDY")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    print(f"Loaded dataset: {df.shape[0]} stations, {df.shape[1]} columns, {df['state'].nunique()} states/UTs")
    print("Class distribution:\n", df["class"].value_counts().reindex(CLASS_ORDER))

    df["y_bin"] = df["class"].isin(["Excellent", "Good"]).astype(int)
    df["y_multi"] = df["class"].map(CLASS_TO_INT)

    # Strict Leak-Free 70% / 15% / 15% Stratified Split (stratified on 4-class grade)
    idx_all = df.index.to_numpy()
    idx_train, idx_temp = train_test_split(
        idx_all, test_size=0.30, random_state=42, stratify=df["y_multi"]
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=42, stratify=df.loc[idx_temp, "y_multi"]
    )

    print(f"\nStratified Split Sizes -> Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    y_train_bin = df.loc[idx_train, "y_bin"].to_numpy()
    y_val_bin = df.loc[idx_val, "y_bin"].to_numpy()
    y_test_bin = df.loc[idx_test, "y_bin"].to_numpy()

    y_train_multi = df.loc[idx_train, "y_multi"].to_numpy()
    y_val_multi = df.loc[idx_val, "y_multi"].to_numpy()
    y_test_multi = df.loc[idx_test, "y_multi"].to_numpy()

    # ---------------------------------------------------------
    # Preprocess Full Laboratory Suite (16 raw + 2 domain ratios)
    # ---------------------------------------------------------
    full_preproc = build_preprocessor(
        IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=4),
            max_iter=10,
            random_state=42,
        ),
        add_full_ratios,
    )
    X_train_full = full_preproc.fit_transform(df.loc[idx_train, FULL_RAW_FEATURES].to_numpy())
    X_val_full = full_preproc.transform(df.loc[idx_val, FULL_RAW_FEATURES].to_numpy())
    X_test_full = full_preproc.transform(df.loc[idx_test, FULL_RAW_FEATURES].to_numpy())

    # ---------------------------------------------------------
    # Experiment A1: Binary Classification Benchmark (Safe vs. Polluted)
    # ---------------------------------------------------------
    print("\n--- Experiment A1: Binary Water Safety Benchmark (Test Split) ---")
    tuned_rf_bin, best_params_bin, cv_auc_bin = tune_random_forest(
        X_train_full, y_train_bin, scoring="roc_auc"
    )
    print(f"Tuned RF Binary Best Params: {best_params_bin} | 5-Fold CV ROC-AUC: {cv_auc_bin:.4f}")

    binary_models = {"Proposed Model - Tuned Random Forest": tuned_rf_bin}
    for name, clf in get_base_models().items():
        clf.fit(X_train_full, y_train_bin)
        binary_models[name] = clf

    binary_rows = []
    for name, clf in binary_models.items():
        metrics = evaluate_binary(clf, X_test_full, y_test_bin)
        binary_rows.append({"Model": name, **metrics})

    binary_df = pd.DataFrame(binary_rows).sort_values(by="ROC-AUC", ascending=False)
    binary_df.to_csv(os.path.join(OUTPUT_DIR, "india_binary_metrics.csv"), index=False)
    print(binary_df.to_string(index=False))

    # ---------------------------------------------------------
    # Experiment A2: 4-Class Multi-Class WQI Benchmark
    # ---------------------------------------------------------
    print("\n--- Experiment A2: 4-Class Water Quality Benchmark (Test Split) ---")
    tuned_rf_multi, best_params_multi, cv_f1_multi = tune_random_forest(
        X_train_full, y_train_multi, scoring="f1_macro"
    )
    print(f"Tuned RF 4-Class Best Params: {best_params_multi} | 5-Fold CV Macro-F1: {cv_f1_multi:.4f}")

    multi_models = {"Proposed Model - Tuned Random Forest": tuned_rf_multi}
    for name, clf in get_base_models().items():
        clf.fit(X_train_full, y_train_multi)
        multi_models[name] = clf

    multi_rows = []
    for name, clf in multi_models.items():
        metrics = evaluate_multiclass(clf, X_test_full, y_test_multi)
        multi_rows.append({"Model": name, **metrics})

    multi_df = pd.DataFrame(multi_rows).sort_values(by="Macro-F1", ascending=False)
    multi_df.to_csv(os.path.join(OUTPUT_DIR, "india_multiclass_metrics.csv"), index=False)
    print(multi_df.to_string(index=False))

    # ---------------------------------------------------------
    # Experiment A3: Instant Field-Probe vs. Full Laboratory Suite
    # ---------------------------------------------------------
    print("\n--- Experiment A3: Instant Field-Probe Screening vs. Full Lab Suite ---")
    probe_preproc = build_preprocessor(
        IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=4),
            max_iter=10,
            random_state=42,
        ),
        add_probe_ratio,
    )
    X_train_probe = probe_preproc.fit_transform(df.loc[idx_train, PROBE_RAW_FEATURES].to_numpy())
    X_test_probe = probe_preproc.transform(df.loc[idx_test, PROBE_RAW_FEATURES].to_numpy())

    tuned_rf_probe_bin, _, _ = tune_random_forest(X_train_probe, y_train_bin, scoring="roc_auc")
    tuned_rf_probe_multi, _, _ = tune_random_forest(X_train_probe, y_train_multi, scoring="f1_macro")

    full_bin_m = evaluate_binary(tuned_rf_bin, X_test_full, y_test_bin)
    probe_bin_m = evaluate_binary(tuned_rf_probe_bin, X_test_probe, y_test_bin)
    full_multi_m = evaluate_multiclass(tuned_rf_multi, X_test_full, y_test_multi)
    probe_multi_m = evaluate_multiclass(tuned_rf_probe_multi, X_test_probe, y_test_multi)

    probe_vs_lab_df = pd.DataFrame([
        {
            "Task": "Binary (Safe vs. Polluted)",
            "Feature Suite": "Full Laboratory Suite (18 Features)",
            "Accuracy": full_bin_m["Accuracy"],
            "Precision / Macro-Prec": full_bin_m["Precision"],
            "Recall / Macro-Rec": full_bin_m["Recall"],
            "F1 / Macro-F1": full_bin_m["F1-Score"],
            "ROC-AUC / Weighted-F1": full_bin_m["ROC-AUC"],
        },
        {
            "Task": "Binary (Safe vs. Polluted)",
            "Feature Suite": "Instant Field-Probe Only (11 Features)",
            "Accuracy": probe_bin_m["Accuracy"],
            "Precision / Macro-Prec": probe_bin_m["Precision"],
            "Recall / Macro-Rec": probe_bin_m["Recall"],
            "F1 / Macro-F1": probe_bin_m["F1-Score"],
            "ROC-AUC / Weighted-F1": probe_bin_m["ROC-AUC"],
        },
        {
            "Task": "4-Class Grade (Excellent/Good/Poor/Very Poor)",
            "Feature Suite": "Full Laboratory Suite (18 Features)",
            "Accuracy": full_multi_m["Accuracy"],
            "Precision / Macro-Prec": full_multi_m["Macro-Precision"],
            "Recall / Macro-Rec": full_multi_m["Macro-Recall"],
            "F1 / Macro-F1": full_multi_m["Macro-F1"],
            "ROC-AUC / Weighted-F1": full_multi_m["Weighted-F1"],
        },
        {
            "Task": "4-Class Grade (Excellent/Good/Poor/Very Poor)",
            "Feature Suite": "Instant Field-Probe Only (11 Features)",
            "Accuracy": probe_multi_m["Accuracy"],
            "Precision / Macro-Prec": probe_multi_m["Macro-Precision"],
            "Recall / Macro-Rec": probe_multi_m["Macro-Recall"],
            "F1 / Macro-F1": probe_multi_m["Macro-F1"],
            "ROC-AUC / Weighted-F1": probe_multi_m["Weighted-F1"],
        },
    ])
    probe_vs_lab_df.to_csv(os.path.join(OUTPUT_DIR, "india_probe_vs_lab.csv"), index=False)
    print(probe_vs_lab_df.to_string(index=False))

    # ---------------------------------------------------------
    # Experiment B: 20% Sensor Dropout Imputation Ablation
    # ---------------------------------------------------------
    print("\n--- Experiment B: 20% Sensor Dropout Imputation Ablation (Instant Field-Probe) ---")
    dropout_cols = ["DO_min", "pH_min", "EC_max", "Nitrate_max"]
    rng = np.random.default_rng(42)

    X_train_masked = df.loc[idx_train, PROBE_RAW_FEATURES].copy()
    X_test_masked = df.loc[idx_test, PROBE_RAW_FEATURES].copy()

    for col in dropout_cols:
        tr_mask = rng.random(len(X_train_masked)) < 0.20
        te_mask = rng.random(len(X_test_masked)) < 0.20
        X_train_masked.loc[tr_mask, col] = np.nan
        X_test_masked.loc[te_mask, col] = np.nan

    imputers = {
        "1. Mean Imputation": SimpleImputer(strategy="mean"),
        "2. Median Imputation": SimpleImputer(strategy="median"),
        "3. KNN Imputation (k=5)": KNNImputer(n_neighbors=5),
        "4. MICE (BayesianRidge)": IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42),
        "5. MICE (ExtraTrees - Proposed)": IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=4),
            max_iter=10,
            random_state=42,
        ),
    }

    ablation_rows = []
    ablation_preds = {}
    for imp_name, imp in imputers.items():
        pipe = build_preprocessor(imp, add_probe_ratio)
        X_tr_imp = pipe.fit_transform(X_train_masked.to_numpy())
        X_te_imp = pipe.transform(X_test_masked.to_numpy())

        rf_ab = RandomForestClassifier(
            n_estimators=best_params_bin.get("n_estimators", 200),
            max_depth=best_params_bin.get("max_depth", 8),
            min_samples_split=best_params_bin.get("min_samples_split", 2),
            min_samples_leaf=best_params_bin.get("min_samples_leaf", 1),
            max_features=best_params_bin.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=42,
            n_jobs=4,
        )
        rf_ab.fit(X_tr_imp, y_train_bin)
        ablation_preds[imp_name] = rf_ab.predict(X_te_imp)
        m = evaluate_binary(rf_ab, X_te_imp, y_test_bin)
        ablation_rows.append({"Imputation Strategy": imp_name, **m})

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(os.path.join(OUTPUT_DIR, "india_imputation_ablation.csv"), index=False)
    print(ablation_df.to_string(index=False))

    prop_p = ablation_preds["5. MICE (ExtraTrees - Proposed)"]
    med_p = ablation_preds["2. Median Imputation"]
    b = int(np.sum((prop_p == y_test_bin) & (med_p != y_test_bin)))
    c = int(np.sum((prop_p != y_test_bin) & (med_p == y_test_bin)))
    p_val = float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue) if (b + c) > 0 else 1.0
    print(f"McNemar (ExtraTrees vs Median under 20% sensor dropout): b={b}, c={c}, p={p_val:.4f}")

    # ---------------------------------------------------------
    # Experiment C: State-Wise & Water-Body-Type Hotspot Analysis
    # ---------------------------------------------------------
    print("\n--- Experiment C: State-Wise & Water-Body-Type Pollution Hotspots ---")
    df["is_polluted"] = (df["y_bin"] == 0).astype(int)
    df["is_safe"] = (df["y_bin"] == 1).astype(int)

    state_summary = (
        df.groupby("state", as_index=False)
        .agg(
            Total_Stations=("station_id", "count"),
            Safe_Count=("is_safe", "sum"),
            Polluted_Count=("is_polluted", "sum"),
            Mean_BOD_max=("BOD_max", "mean"),
            Mean_DO_min=("DO_min", "mean"),
            Mean_FC_max=("FC_max", "mean"),
        )
    )
    state_summary["Polluted_Pct"] = (
        state_summary["Polluted_Count"] / state_summary["Total_Stations"] * 100
    ).round(2)
    state_summary["Mean_BOD_max"] = state_summary["Mean_BOD_max"].round(2)
    state_summary["Mean_DO_min"] = state_summary["Mean_DO_min"].round(2)
    state_summary["Mean_FC_max"] = state_summary["Mean_FC_max"].round(1)
    state_summary = state_summary.sort_values(
        by=["Total_Stations", "Polluted_Pct"], ascending=[False, False]
    ).reset_index(drop=True)

    state_cols_order = [
        "state", "Total_Stations", "Safe_Count", "Polluted_Count", "Polluted_Pct",
        "Mean_BOD_max", "Mean_DO_min", "Mean_FC_max"
    ]
    state_summary = state_summary[state_cols_order]
    state_summary.to_csv(os.path.join(OUTPUT_DIR, "india_state_summary.csv"), index=False)
    print(state_summary.head(12).to_string(index=False))

    wb_summary = (
        df.groupby("water_body_type", as_index=False)
        .agg(
            Total_Stations=("station_id", "count"),
            Safe_Count=("is_safe", "sum"),
            Polluted_Count=("is_polluted", "sum"),
        )
    )
    wb_summary["Polluted_Pct"] = (
        wb_summary["Polluted_Count"] / wb_summary["Total_Stations"] * 100
    ).round(2)
    print("\nWater Body Type Breakdown:")
    print(wb_summary.to_string(index=False))

    # Plot Top 10 States by Station Count (Safe vs. Polluted stacked horizontal bar)
    top10 = state_summary.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top10))

    ax.barh(y_pos, top10["Safe_Count"], color="#0284c7", label="Safe (Excellent / Good)")
    ax.barh(
        y_pos,
        top10["Polluted_Count"],
        left=top10["Safe_Count"],
        color="#dc2626",
        label="Polluted (Poor / Very Poor)",
    )

    for idx_bar, (_, row) in enumerate(top10.iterrows()):
        total = int(row["Total_Stations"])
        pct = row["Polluted_Pct"]
        ax.text(
            total + 2.5,
            idx_bar,
            f"{total} stations ({pct:.1f}% polluted)",
            va="center",
            fontsize=9,
            color="#1f2937",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10["state"], fontsize=9.5)
    ax.set_xlabel("Number of CPCB Monitoring Stations", fontsize=10)
    ax.set_title("Indian CPCB Surface Water Quality Hotspots (Top 10 States)", fontsize=12, pad=12)
    ax.set_xlim(0, top10["Total_Stations"].max() * 1.32)
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    state_plot_path = os.path.join(OUTPUT_DIR, "india_state_pollution.png")
    fig.savefig(state_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved state pollution hotspot plot: {state_plot_path}")

    # ---------------------------------------------------------
    # Experiment D: TreeExplainer SHAP on Indian CPCB Data
    # ---------------------------------------------------------
    print("\n--- Experiment D: TreeExplainer SHAP Analysis (Full & Probe Suites) ---")
    shap_full_vals = extract_positive_shap(tuned_rf_bin, X_test_full)
    shap_probe_vals = extract_positive_shap(tuned_rf_probe_bin, X_test_probe)

    full_imp = pd.DataFrame({
        "Feature": FULL_FEATURE_NAMES,
        "Mean_Absolute_SHAP": np.round(np.abs(shap_full_vals).mean(axis=0), 5),
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False)
    full_imp.to_csv(os.path.join(OUTPUT_DIR, "india_shap_importance.csv"), index=False)

    probe_imp = pd.DataFrame({
        "Feature": PROBE_FEATURE_NAMES,
        "Mean_Absolute_SHAP": np.round(np.abs(shap_probe_vals).mean(axis=0), 5),
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False)
    probe_imp.to_csv(os.path.join(OUTPUT_DIR, "india_probe_shap_importance.csv"), index=False)

    print("Top SHAP Features (Full Suite):")
    print(full_imp.head(10).to_string(index=False))
    print("\nTop SHAP Features (Instant Field-Probe Suite):")
    print(probe_imp.head(10).to_string(index=False))

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_full_vals, X_test_full, feature_names=FULL_FEATURE_NAMES, show=False)
    plt.title("India CPCB SHAP Summary (Safe vs. Polluted Surface Water)", fontsize=12, pad=14)
    plt.tight_layout()
    shap_sum_path = os.path.join(OUTPUT_DIR, "india_shap_summary.png")
    plt.savefig(shap_sum_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    shap.summary_plot(
        shap_full_vals, X_test_full, feature_names=FULL_FEATURE_NAMES, plot_type="bar", show=False
    )
    plt.title("India CPCB Global Feature Attribution (Mean |SHAP|)", fontsize=12, pad=14)
    plt.tight_layout()
    shap_bar_path = os.path.join(OUTPUT_DIR, "india_shap_bar.png")
    plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9.5, 5.5))
    shap.summary_plot(shap_probe_vals, X_test_probe, feature_names=PROBE_FEATURE_NAMES, show=False)
    plt.title("India CPCB Instant Field-Probe SHAP Summary (No Lab Culture Tests)", fontsize=12, pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "india_probe_shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 72)
    print("INDIAN CPCB EXTERNAL VALIDATION STUDY COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()
