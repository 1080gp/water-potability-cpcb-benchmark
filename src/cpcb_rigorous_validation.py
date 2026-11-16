import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize

BASE_DIR = Path(__file__).resolve().parent.parent
KAGGLE_PATH = BASE_DIR / "data" / "water_potability.csv"
CPCB_PATH = BASE_DIR / "data" / "india_water_quality.csv"

# -------------------------------------------------------------------------
# 1. EMPIRICAL DEMONSTRATION OF TARGET LEAKAGE (KAGGLE)
# -------------------------------------------------------------------------
def run_leakage_experiment():
    print("=" * 70)
    print("EXPERIMENT 1: EMPIRICAL AUDIT OF TARGET LEAKAGE VS. LEAK-FREE PIPELINE")
    print("=" * 70)
    df = pd.read_csv(KAGGLE_PATH)
    features = [c for c in df.columns if c != "Potability"]
    
    # LEAKY PIPELINE: Class-conditional imputation BEFORE splitting
    df_leaky = df.copy()
    for col in ["Sulfate", "ph", "Trihalomethanes"]:
        df_leaky[col] = df_leaky.groupby("Potability")[col].transform(lambda x: x.fillna(x.mean()))
    
    X_leaky = df_leaky[features].values
    y_leaky = df_leaky["Potability"].values
    
    X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
        X_leaky, y_leaky, test_size=0.15, random_state=42, stratify=y_leaky
    )
    scaler_l = StandardScaler()
    X_tr_l_sc = scaler_l.fit_transform(X_tr_l)
    X_te_l_sc = scaler_l.transform(X_te_l)
    
    rf_leaky = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=4)
    rf_leaky.fit(X_tr_l_sc, y_tr_l)
    preds_l = rf_leaky.predict(X_te_l_sc)
    probs_l = rf_leaky.predict_proba(X_te_l_sc)[:, 1]
    
    acc_l = accuracy_score(y_te_l, preds_l)
    auc_l = roc_auc_score(y_te_l, probs_l)
    f1_l = f1_score(y_te_l, preds_l)
    
    print(f"LEAKY PIPELINE (Class-conditional mean impute before split):")
    print(f"-> Test Accuracy: {acc_l * 100:.2f}% | F1: {f1_l:.4f} | ROC-AUC: {auc_l:.4f}")
    
    # LEAK-FREE PIPELINE: Standard SimpleImputer fitted strictly on X_train
    X_raw = df[features].values
    y_raw = df["Potability"].values
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=42, stratify=y_raw
    )
    imp_c = SimpleImputer(strategy="mean")
    X_tr_c_imp = imp_c.fit_transform(X_tr_c)
    X_te_c_imp = imp_c.transform(X_te_c)
    scaler_c = StandardScaler()
    X_tr_c_sc = scaler_c.fit_transform(X_tr_c_imp)
    X_te_c_sc = scaler_c.transform(X_te_c_imp)
    
    rf_clean = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=4)
    rf_clean.fit(X_tr_c_sc, y_tr_c)
    preds_c = rf_clean.predict(X_te_c_sc)
    probs_c = rf_clean.predict_proba(X_te_c_sc)[:, 1]
    
    acc_c = accuracy_score(y_te_c, preds_c)
    auc_c = roc_auc_score(y_te_c, probs_c)
    f1_c = f1_score(y_te_c, preds_c)
    
    print(f"\nLEAK-FREE PIPELINE (Imputer fitted strictly on X_train only):")
    print(f"-> Test Accuracy: {acc_c * 100:.2f}% | F1: {f1_c:.4f} | ROC-AUC: {auc_c:.4f}")
    print(f"-> EMPIRICAL LEAKAGE INFLATION GAP: +{(acc_l - acc_c) * 100:.2f}% Accuracy | +{(auc_l - auc_c):.4f} ROC-AUC")
    return {
        "leaky_acc": round(acc_l * 100, 2),
        "leaky_f1": round(f1_l, 4),
        "leaky_auc": round(auc_l, 4),
        "clean_acc": round(acc_c * 100, 2),
        "clean_f1": round(f1_c, 4),
        "clean_auc": round(auc_c, 4),
        "gap_acc": round((acc_l - acc_c) * 100, 2),
        "gap_auc": round(auc_l - auc_c, 4),
    }

# -------------------------------------------------------------------------
# 2. WILSON SCORE CONFIDENCE INTERVALS (N = 103 CPCB TEST SET)
# -------------------------------------------------------------------------
def wilson_score_interval(successes, total, conf=0.95):
    z = 1.96 if conf == 0.95 else 2.576
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return round((centre - margin) * 100, 2), round((centre + margin) * 100, 2)

def run_confidence_intervals():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: 95% WILSON SCORE CONFIDENCE INTERVALS (DIRECTLY FROM PREDICTIONS, n=103)")
    print("=" * 70)
    df = pd.read_csv(CPCB_PATH)
    df["DO_BOD_ratio"] = df["DO_min"] / (df["BOD_max"] + 1e-6)
    df["pH_EC_ratio"] = df["pH_max"] / (df["EC_max"] + 1e-6)

    full_feats = [
        "Temp_min", "DO_min", "pH_min", "EC_min", "BOD_min", "Nitrate_min", "FC_min", "TC_min",
        "Temp_max", "DO_max", "pH_max", "EC_max", "BOD_max", "Nitrate_max", "FC_max", "TC_max",
        "DO_BOD_ratio", "pH_EC_ratio"
    ]
    probe_feats = [
        "Temp_min", "DO_min", "pH_min", "EC_min", "Nitrate_min",
        "Temp_max", "DO_max", "pH_max", "EC_max", "Nitrate_max", "pH_EC_ratio"
    ]
    y_bin = df["class"].isin(["Excellent", "Good"]).astype(int).values
    y_multi = df["class"].values

    idx = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y_bin)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=42, stratify=y_bin[idx_tmp])

    def eval_split_ci(X_arr, y_arr, clf):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_arr[idx_tr])
        X_te_s = sc.transform(X_arr[idx_te])
        clf.fit(X_tr_s, y_arr[idx_tr])
        preds = clf.predict(X_te_s)
        correct = int((preds == y_arr[idx_te]).sum())
        total = len(idx_te)
        lo, hi = wilson_score_interval(correct, total)
        return round(correct / total * 100, 2), lo, hi, correct, total

    acc_bf, lo_bf, hi_bf, c_bf, n_bf = eval_split_ci(
        df[full_feats].values, y_bin, RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=4)
    )
    acc_bp, lo_bp, hi_bp, c_bp, n_bp = eval_split_ci(
        df[probe_feats].values, y_bin, RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=4)
    )
    acc_mf, lo_mf, hi_mf, c_mf, n_mf = eval_split_ci(
        df[full_feats].values, y_multi, RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=4)
    )
    acc_mp, lo_mp, hi_mp, c_mp, n_mp = eval_split_ci(
        df[probe_feats].values, y_multi, RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=4)
    )

    print(f"Full Suite (18 Features) Binary Fidelity: {acc_bf:.2f}% [95% CI: {lo_bf}% - {hi_bf}%] ({c_bf}/{n_bf})")
    print(f"Instant Probe (11 Features) Binary Screening: {acc_bp:.2f}% [95% CI: {lo_bp}% - {hi_bp}%] ({c_bp}/{n_bp})")
    print(f"Full Suite (18 Features) 4-Class Fidelity: {acc_mf:.2f}% [95% CI: {lo_mf}% - {hi_mf}%] ({c_mf}/{n_mf})")
    print(f"Instant Probe (11 Features) 4-Class Screening: {acc_mp:.2f}% [95% CI: {lo_mp}% - {hi_mp}%] ({c_mp}/{n_mp})")


# -------------------------------------------------------------------------
# 3. SPATIAL GROUPED & LEAVE-ONE-STATE-OUT (LOSO) CROSS-VALIDATION (CPCB)
# -------------------------------------------------------------------------
def run_spatial_validation():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: SPATIAL GROUPKFOLD & LEAVE-ONE-STATE-OUT (LOSO) VALIDATION")
    print("=" * 70)
    df = pd.read_csv(CPCB_PATH)
    PROBE_FEATURES = [
        "Temp_min", "DO_min", "pH_min", "EC_min", "Nitrate_min",
        "Temp_max", "DO_max", "pH_max", "EC_max", "Nitrate_max"
    ]
    df_p = df.copy()
    df_p["pH_EC_ratio"] = df_p["pH_max"] / (df_p["EC_max"] + 1e-6)
    all_probe_feats = PROBE_FEATURES + ["pH_EC_ratio"]

    y_bin = df_p["class"].isin(["Excellent", "Good"]).astype(int).values
    y_multi = df_p["class"].values
    groups = df_p["state"].values
    X = df_p[all_probe_feats].values
    do_vals = df_p["DO_min"].values
    n_total = len(y_bin)

    spatial_rows = []

    for cv_name, splitter in [("5-Fold GroupKFold", GroupKFold(n_splits=5)), ("Leave-One-State-Out (26 States)", LeaveOneGroupOut())]:
        oof_rf_pred = np.zeros(n_total, dtype=int)
        oof_rf_prob = np.zeros(n_total, dtype=float)
        oof_lr_pred = np.zeros(n_total, dtype=int)
        oof_lr_prob = np.zeros(n_total, dtype=float)
        oof_do_pred = np.zeros(n_total, dtype=int)

        oof_m_pred = np.empty(n_total, dtype=object)
        class_order = np.array(["Excellent", "Good", "Poor", "Very Poor"])
        oof_m_prob = np.zeros((n_total, len(class_order)), dtype=float)

        rf_fold_accs, lr_fold_accs, do_fold_accs = [], [], []

        for tr_idx, te_idx in splitter.split(X, y_bin, groups=groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])

            # Binary RF
            clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=4)
            clf.fit(X_tr, y_bin[tr_idx])
            oof_rf_pred[te_idx] = clf.predict(X_te)
            oof_rf_prob[te_idx] = clf.predict_proba(X_te)[:, 1]

            # Binary Logistic Regression (subset-of-formula baseline)
            lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
            lr_clf.fit(X_tr, y_bin[tr_idx])
            oof_lr_pred[te_idx] = lr_clf.predict(X_te)
            oof_lr_prob[te_idx] = lr_clf.predict_proba(X_te)[:, 1]

            # Identical-fold DO_min >= 4.0 heuristic
            oof_do_pred[te_idx] = (do_vals[te_idx] >= 4.0).astype(int)

            rf_fold_accs.append(accuracy_score(y_bin[te_idx], oof_rf_pred[te_idx]))
            lr_fold_accs.append(accuracy_score(y_bin[te_idx], oof_lr_pred[te_idx]))
            do_fold_accs.append(accuracy_score(y_bin[te_idx], oof_do_pred[te_idx]))

            # 4-Class Spatial RF
            clf_m = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=4)
            clf_m.fit(X_tr, y_multi[tr_idx])
            oof_m_pred[te_idx] = clf_m.predict(X_te)
            probs_m = clf_m.predict_proba(X_te)
            for j, c_lbl in enumerate(clf_m.classes_):
                col_idx = int(np.where(class_order == c_lbl)[0][0])
                oof_m_prob[te_idx, col_idx] = probs_m[:, j]

        # Pooled OOF metrics + 95% Wilson CIs directly from predictions
        for m_label, preds, probs, f_accs in [
            ("Binary RF (11 Probe)", oof_rf_pred, oof_rf_prob, rf_fold_accs),
            ("Binary Logistic Reg (11 Probe)", oof_lr_pred, oof_lr_prob, lr_fold_accs),
            ("Binary Heuristic (DO_min >= 4.0)", oof_do_pred, None, do_fold_accs),
        ]:
            acc = accuracy_score(y_bin, preds)
            lo, hi = wilson_score_interval(int((preds == y_bin).sum()), n_total)
            f1 = f1_score(y_bin, preds, zero_division=0)
            auc = roc_auc_score(y_bin, probs) if probs is not None else np.nan
            spatial_rows.append({
                "CV_Scheme": cv_name,
                "Model": m_label,
                "Pooled_OOF_Acc": round(acc, 4),
                "OOF_Acc_95_CI": f"[{lo}%, {hi}%]",
                "Fold_Mean_Acc": round(float(np.mean(f_accs)), 4),
                "Fold_Std_Acc": round(float(np.std(f_accs)), 4),
                "Pooled_F1_or_MacroF1": round(f1, 4),
                "Pooled_ROC_AUC_or_MacroOvR": round(float(auc), 4) if not np.isnan(auc) else np.nan,
            })

        # 4-Class Pooled OOF
        acc_m = accuracy_score(y_multi, oof_m_pred)
        lo_m, hi_m = wilson_score_interval(int((oof_m_pred == y_multi).sum()), n_total)
        macro_f1_m = f1_score(y_multi, oof_m_pred, average="macro", zero_division=0)
        y_m_bin = label_binarize(y_multi, classes=class_order)
        macro_ovr_m = roc_auc_score(y_m_bin, oof_m_prob, multi_class="ovr", average="macro")
        spatial_rows.append({
            "CV_Scheme": cv_name,
            "Model": "4-Class Regularized RF (11 Probe)",
            "Pooled_OOF_Acc": round(acc_m, 4),
            "OOF_Acc_95_CI": f"[{lo_m}%, {hi_m}%]",
            "Fold_Mean_Acc": np.nan,
            "Fold_Std_Acc": np.nan,
            "Pooled_F1_or_MacroF1": round(macro_f1_m, 4),
            "Pooled_ROC_AUC_or_MacroOvR": round(float(macro_ovr_m), 4),
        })

    spatial_df = pd.DataFrame(spatial_rows)
    spatial_out = BASE_DIR / "outputs" / "cpcb_spatial_validation.csv"
    spatial_df.to_csv(spatial_out, index=False)
    print(spatial_df.to_string(index=False))
    print(f"\nSaved spatial validation table to: {spatial_out}")
    return spatial_rows

# -------------------------------------------------------------------------
# 4. TRUE MACRO ONE-VS-REST ROC-AUC FOR 4-CLASS MODEL
# -------------------------------------------------------------------------
def run_multiclass_ovr_auc():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: TRUE MACRO ONE-VS-REST (OvR) ROC-AUC (CPCB 4-CLASS)")
    print("=" * 70)
    from sklearn.preprocessing import label_binarize
    
    df = pd.read_csv(CPCB_PATH)
    FULL_FEATURES = [
        "Temp_min", "DO_min", "pH_min", "EC_min", "BOD_min", "Nitrate_min", "FC_min", "TC_min",
        "Temp_max", "DO_max", "pH_max", "EC_max", "BOD_max", "Nitrate_max", "FC_max", "TC_max"
    ]
    df["DO_BOD_ratio"] = df["DO_min"] / (df["BOD_max"] + 1e-6)
    df["pH_EC_ratio"] = df["pH_max"] / (df["EC_max"] + 1e-6)
    full_feats = FULL_FEATURES + ["DO_BOD_ratio", "pH_EC_ratio"]
    
    CLASS_ORDER = ["Excellent", "Good", "Poor", "Very Poor"]
    y_str = np.array(df["class"].tolist())
    X = np.array(df[full_feats].values, dtype=float)
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_str, test_size=0.15, random_state=42, stratify=y_str
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    clf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=4)
    clf.fit(X_tr_sc, y_tr)
    probs = clf.predict_proba(X_te_sc)
    
    # Calculate true Macro One-vs-Rest ROC-AUC
    y_te_bin = label_binarize(y_te, classes=clf.classes_)
    macro_ovr_auc = roc_auc_score(y_te_bin, probs, multi_class="ovr", average="macro")
    weighted_ovr_auc = roc_auc_score(y_te_bin, probs, multi_class="ovr", average="weighted")
    
    print(f"4-Class Regularized Random Forest (Full Laboratory Suite, 400 trees, max_depth=12):")
    print(f"-> True Macro OvR ROC-AUC: {macro_ovr_auc:.4f}")
    print(f"-> True Weighted OvR ROC-AUC: {weighted_ovr_auc:.4f}")

    # 4-Class Instant Probe (11 features)
    probe_feats = [
        "Temp_min", "DO_min", "pH_min", "EC_min", "Nitrate_min",
        "Temp_max", "DO_max", "pH_max", "EC_max", "Nitrate_max", "pH_EC_ratio"
    ]
    X_p = np.array(df[probe_feats].values, dtype=float)
    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_p, y_str, test_size=0.15, random_state=42, stratify=y_str
    )
    sc_p = StandardScaler()
    X_tr_p_sc = sc_p.fit_transform(X_tr_p)
    X_te_p_sc = sc_p.transform(X_te_p)
    clf_p = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=4)
    clf_p.fit(X_tr_p_sc, y_tr_p)
    probs_p = clf_p.predict_proba(X_te_p_sc)
    y_te_p_bin = label_binarize(y_te_p, classes=clf_p.classes_)
    macro_ovr_auc_p = roc_auc_score(y_te_p_bin, probs_p, multi_class="ovr", average="macro")
    print(f"\n4-Class Regularized Random Forest (Instant Field-Probe, 11 features, 400 trees, max_depth=12):")
    print(f"-> True Macro OvR ROC-AUC: {macro_ovr_auc_p:.4f}")
    return {
        "full_4class_macro_ovr_auc": round(float(macro_ovr_auc), 4),
        "full_4class_weighted_ovr_auc": round(float(weighted_ovr_auc), 4),
        "probe_4class_macro_ovr_auc": round(float(macro_ovr_auc_p), 4),
    }

if __name__ == "__main__":
    leak_res = run_leakage_experiment()
    run_confidence_intervals()
    spatial_res = run_spatial_validation()
    ovr_res = run_multiclass_ovr_auc()
    summary_path = BASE_DIR / "outputs" / "cpcb_rigorous_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"leakage_experiment": leak_res, "ovr_auc": ovr_res, "spatial_validation": spatial_res}, f, indent=2)
    print(f"\nSaved rigorous summary JSON to: {summary_path}")
