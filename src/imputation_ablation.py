import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
)

from preprocessing import build_preprocessor, get_splits

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("IMPUTATION ABLATION ON 5-FOLD TRAINING CV & VALIDATION (NO TEST PEEKING)")
print("=" * 75)

# 1. Load Data Splits (X_test remains untouched)
X_train_raw, X_val_raw, _, y_train, y_val, _, _, _ = get_splits()

# Imputation strategies to benchmark
imputers = {
    "1. Mean Imputation": SimpleImputer(strategy="mean"),
    "2. Median Imputation": SimpleImputer(strategy="median"),
    "3. KNN Imputation (k=5)": KNNImputer(n_neighbors=5),
    "4. MICE (BayesianRidge)": IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42),
    "5. MICE (ExtraTrees - Proposed)": IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=4),
        max_iter=10,
        random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "unsafe_recall": make_scorer(recall_score, pos_label=0, zero_division=0),
    "potable_recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "potable_precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "f1_macro": "f1_macro"
}

results = []
val_predictions_dict = {}
cv_auc_folds = {}

for name, imputer in imputers.items():
    print(f"\nEvaluating inside Pipeline CV: {name}...")

    pipe = Pipeline([
        ("prep", build_preprocessor(imputer)),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=4
        ))
    ])

    # 1. 5-Fold Stratified CV on X_train_raw (imputer refitted inside every fold)
    cv_res = cross_validate(pipe, X_train_raw, y_train, cv=cv, scoring=scoring, n_jobs=2)
    cv_auc_folds[name] = cv_res["test_roc_auc"]

    # 2. Fit on full X_train_raw, evaluate on Validation set (X_val_raw, n=491)
    pipe.fit(X_train_raw, y_train)
    val_preds = pipe.predict(X_val_raw)
    val_probs = pipe.predict_proba(X_val_raw)[:, 1]
    val_predictions_dict[name] = val_preds

    val_acc = accuracy_score(y_val, val_preds)
    val_unsafe_rec = recall_score(y_val, val_preds, pos_label=0, zero_division=0)
    val_false_safe = 1.0 - val_unsafe_rec
    val_pot_rec = recall_score(y_val, val_preds, pos_label=1, zero_division=0)
    val_pot_prec = precision_score(y_val, val_preds, pos_label=1, zero_division=0)
    val_f1_pot = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
    val_auc = roc_auc_score(y_val, val_probs)

    results.append({
        "Imputation Strategy": name,
        "CV_Acc_Mean": round(cv_res["test_accuracy"].mean(), 4),
        "CV_AUC_Mean": round(cv_res["test_roc_auc"].mean(), 4),
        "CV_AUC_Std": round(cv_res["test_roc_auc"].std(), 4),
        "CV_Unsafe_Recall": round(cv_res["test_unsafe_recall"].mean(), 4),
        "Val_Accuracy": round(val_acc, 4),
        "Val_Unsafe_Recall_Spec": round(val_unsafe_rec, 4),
        "Val_False_Safe_Rate": round(val_false_safe, 4),
        "Val_Potable_Prec": round(val_pot_prec, 4),
        "Val_Potable_Recall": round(val_pot_rec, 4),
        "Val_Potable_F1": round(val_f1_pot, 4),
        "Val_ROC_AUC": round(val_auc, 4),
    })
    print(f"-> 5-Fold CV AUC: {cv_res['test_roc_auc'].mean():.4f} (+/- {cv_res['test_roc_auc'].std():.4f}) | "
          f"Val Unsafe Recall (Spec): {val_unsafe_rec:.4f} | Val False-Safe Rate: {val_false_safe:.4f} | Val AUC: {val_auc:.4f}")

ablation_df = pd.DataFrame(results)

# Paired t-test across CV folds & McNemar on Validation set (ExtraTrees vs Median)
proposed_val = val_predictions_dict["5. MICE (ExtraTrees - Proposed)"]
median_val = val_predictions_dict["2. Median Imputation"]
b = int(np.sum((proposed_val == y_val) & (median_val != y_val)))
c = int(np.sum((proposed_val != y_val) & (median_val == y_val)))
p_mcnemar = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue if (b + c) > 0 else 1.0
t_stat, p_cv_ttest = ttest_rel(cv_auc_folds["5. MICE (ExtraTrees - Proposed)"], cv_auc_folds["2. Median Imputation"])

print("\n" + "=" * 75)
print("STATISTICAL COMPARISON (ExtraTrees MICE vs. Median Imputation)")
print(f"5-Fold CV ROC-AUC Paired t-test: t = {t_stat:.4f}, p = {p_cv_ttest:.4f}")
print(f"Validation McNemar Exact Binomial (b={b}, c={c}): p = {p_mcnemar:.4f}")
print("=" * 75)

out_path = os.path.join(OUTPUT_DIR, "imputation_ablation_study.csv")
ablation_df.to_csv(out_path, index=False)
print(f"\nAblation benchmark saved to: {out_path}")
print(ablation_df.to_string(index=False))
