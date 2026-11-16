import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix, make_scorer
)

from preprocessing import build_preprocessor, get_splits

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Raw Splits (preprocessing is inside Pipeline so every CV fold refits it!)
X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, _, _ = get_splits()
print("Raw splits loaded. Training:", X_train_raw.shape, "Val:", X_val_raw.shape, "Test:", X_test_raw.shape)


def evaluate_metrics(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    unsafe_rec = recall_score(y, preds, pos_label=0, zero_division=0)
    return {
        "Accuracy": accuracy_score(y, preds),
        "Unsafe_Recall_Spec": unsafe_rec,
        "False_Safe_Rate": 1.0 - unsafe_rec,
        "Unsafe_Precision_NPV": precision_score(y, preds, pos_label=0, zero_division=0),
        "Precision": precision_score(y, preds, pos_label=1, zero_division=0),
        "Recall": recall_score(y, preds, pos_label=1, zero_division=0),
        "F1-Score": f1_score(y, preds, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probs),
        "Brier_Score": brier_score_loss(y, probs),
        "Confusion Matrix": confusion_matrix(y, preds)
    }


base_classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=4
    ),
    "SVM": SVC(
        kernel="rbf", probability=True, random_state=42, class_weight="balanced"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=5, random_state=42, class_weight="balanced"
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=9, weights="distance"
    )
}

trained_pipelines = {}
validation_results = []

# Train base pipelines on X_train_raw and evaluate on X_val_raw
for name, clf in base_classifiers.items():
    print(f"\nTraining Pipeline: {name}...")
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", clf)
    ])
    pipe.fit(X_train_raw, y_train)
    trained_pipelines[name] = pipe

    res = evaluate_metrics(pipe, X_val_raw, y_val)
    validation_results.append({
        "Model": name,
        "Accuracy": res["Accuracy"],
        "Unsafe_Recall_Spec": res["Unsafe_Recall_Spec"],
        "False_Safe_Rate": res["False_Safe_Rate"],
        "Unsafe_Precision_NPV": res["Unsafe_Precision_NPV"],
        "Precision": res["Precision"],
        "Recall": res["Recall"],
        "F1-Score": res["F1-Score"],
        "ROC-AUC": res["ROC-AUC"],
        "Brier_Score": res["Brier_Score"]
    })
    print(f"-> Val Acc: {res['Accuracy']:.4f} | Unsafe Recall (Spec): {res['Unsafe_Recall_Spec']:.4f} | "
          f"False-Safe Rate: {res['False_Safe_Rate']:.4f} | AUC: {res['ROC-AUC']:.4f}")

# Proposed Model: Tuned Random Forest with preprocessing INSIDE GridSearchCV folds
print("\n" + "=" * 75)
print("TUNING PROPOSED MODEL INSIDE LEAK-FREE PIPELINE CV")
print("=" * 75)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tuned_pipe = Pipeline([
    ("prep", build_preprocessor()),
    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=4))
])

param_grid = {
    "clf__n_estimators": [300, 400],
    "clf__max_depth": [10, 12],
    "clf__min_samples_split": [4, 5],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=tuned_pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=inner_cv,
    n_jobs=2,
    verbose=1
)
grid_search.fit(X_train_raw, y_train)

best_rf_pipe = grid_search.best_estimator_
trained_pipelines["Proposed Model - Tuned Random Forest"] = best_rf_pipe

print("\nBest Parameters:", grid_search.best_params_)
print("GridSearch Inner-CV Best ROC-AUC (optimistic):", round(grid_search.best_score_, 4))

res_prop = evaluate_metrics(best_rf_pipe, X_val_raw, y_val)
validation_results.append({
    "Model": "Proposed Model - Tuned Random Forest",
    "Accuracy": res_prop["Accuracy"],
    "Unsafe_Recall_Spec": res_prop["Unsafe_Recall_Spec"],
    "False_Safe_Rate": res_prop["False_Safe_Rate"],
    "Unsafe_Precision_NPV": res_prop["Unsafe_Precision_NPV"],
    "Precision": res_prop["Precision"],
    "Recall": res_prop["Recall"],
    "F1-Score": res_prop["F1-Score"],
    "ROC-AUC": res_prop["ROC-AUC"],
    "Brier_Score": res_prop["Brier_Score"]
})

results_df = pd.DataFrame(validation_results).sort_values(by="ROC-AUC", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "validation_results.csv"), index=False)
print("\nValidation Results:")
print(results_df.to_string(index=False))

# Threshold Sweep on Validation Set for Best Model (Cost-Sensitive Operating Point)
val_probs = best_rf_pipe.predict_proba(X_val_raw)[:, 1]
thresh_rows = []
for thr in [0.30, 0.40, 0.50, 0.60, 0.70]:
    t_preds = (val_probs >= thr).astype(int)
    u_rec = recall_score(y_val, t_preds, pos_label=0, zero_division=0)
    p_rec = recall_score(y_val, t_preds, pos_label=1, zero_division=0)
    p_prec = precision_score(y_val, t_preds, pos_label=1, zero_division=0)
    thresh_rows.append({
        "Threshold": thr,
        "Accuracy": round(accuracy_score(y_val, t_preds), 4),
        "Unsafe_Recall_Spec": round(u_rec, 4),
        "False_Safe_Rate": round(1.0 - u_rec, 4),
        "Potable_Precision": round(p_prec, 4),
        "Potable_Recall": round(p_rec, 4)
    })
thresh_df = pd.DataFrame(thresh_rows)
thresh_df.to_csv(os.path.join(OUTPUT_DIR, "validation_threshold_sweep.csv"), index=False)
print("\nValidation Threshold Sweep (Operating Point Analysis):")
print(thresh_df.to_string(index=False))

# Repeated Stratified 5x3 CV (15 folds) with preprocessing refitted per fold
print("\n" + "=" * 75)
print("REPEATED STRATIFIED 5x3 CV (PREPROCESSING REFITTED INSIDE EVERY FOLD)")
print("=" * 75)

rep_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scoring_cv = {
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "unsafe_recall": make_scorer(recall_score, pos_label=0, zero_division=0)
}

cv_results = []
fold_auc_dict = {}

for name, pipe in trained_pipelines.items():
    print(f"Cross-validating Pipeline: {name}...")
    # Clone fresh pipeline with fixed hyperparams so CV doesn't reuse GridSearch folds
    fresh_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", pipe.named_steps["clf"])
    ])
    scores = cross_validate(fresh_pipe, X_train_raw, y_train, cv=rep_cv, scoring=scoring_cv, n_jobs=2)
    fold_auc_dict[name] = scores["test_roc_auc"]
    cv_results.append({
        "Model": name,
        "CV_ROC_AUC_Mean": round(scores["test_roc_auc"].mean(), 4),
        "CV_ROC_AUC_Std": round(scores["test_roc_auc"].std(), 4),
        "CV_Accuracy_Mean": round(scores["test_accuracy"].mean(), 4),
        "CV_Unsafe_Recall_Mean": round(scores["test_unsafe_recall"].mean(), 4)
    })

cv_df = pd.DataFrame(cv_results).sort_values(by="CV_ROC_AUC_Mean", ascending=False)
cv_df.to_csv(os.path.join(OUTPUT_DIR, "cross_validation_results.csv"), index=False)
print("\nRepeated Stratified 5x3 CV Results:")
print(cv_df.to_string(index=False))

# Nadeau-Bengio (2003) corrected resampled t-test across all models vs Proposed Tuned RF
def nadeau_bengio_ttest(scores_a, scores_b, n_train_fold=4, n_test_fold=1):
    """Corrected resampled t-test accounting for overlapping training folds in Repeated CV."""
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    k_total = len(diffs)
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))
    corr_var = ((1.0 / k_total) + (n_test_fold / n_train_fold)) * var_diff
    if corr_var <= 0:
        return 0.0, 1.0
    t_stat = mean_diff / np.sqrt(corr_var)
    p_val = float(2.0 * student_t.sf(np.abs(t_stat), df=k_total - 1))
    return round(t_stat, 4), round(p_val, 4)


rf_name = "Proposed Model - Tuned Random Forest"
stat_rows = []
for name in cv_df["Model"]:
    if name == rf_name:
        continue
    t_nb, p_nb = nadeau_bengio_ttest(fold_auc_dict[name], fold_auc_dict[rf_name], n_train_fold=4, n_test_fold=1)
    stat_rows.append({
        "Comparison": f"{name} vs. {rf_name}",
        "Mean_AUC_Diff": round(float(np.mean(fold_auc_dict[name] - fold_auc_dict[rf_name])), 4),
        "Nadeau_Bengio_t": t_nb,
        "Nadeau_Bengio_p": p_nb,
        "Significant_0.05": bool(p_nb < 0.05)
    })
stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(os.path.join(OUTPUT_DIR, "cv_statistical_tests.csv"), index=False)
print("\nNadeau-Bengio Corrected Resampled t-tests (5x3 Repeated CV, df=14):")
print(stat_df.to_string(index=False))

# Ensemble & Weighting Ablation on Validation Set (n=491) & 5-Fold Stratified CV (NO test-set tuning)
print("\n" + "=" * 75)
print("ENSEMBLE & WEIGHTING ABLATION (5-FOLD CV + VALIDATION SET ONLY)")
print("=" * 75)

best_rf_params = grid_search.best_params_
rf_kw = {k.replace("clf__", ""): v for k, v in best_rf_params.items()}

ablation_clfs = {
    "Tuned RF (Balanced)": RandomForestClassifier(**rf_kw, class_weight="balanced", random_state=42, n_jobs=4),
    "Unweighted Tuned RF (No Weighting)": RandomForestClassifier(**rf_kw, class_weight=None, random_state=42, n_jobs=4),
    "Extra Trees (Balanced)": ExtraTreesClassifier(
        n_estimators=400, max_depth=12, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=4
    ),
    "SVM (RBF, Balanced)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "Soft-Voting Ensemble (RF+ET+SVM)": VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(**rf_kw, class_weight="balanced", random_state=42, n_jobs=4)),
            ("et", ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=4)),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ],
        voting="soft",
        n_jobs=2
    ),
    "Stacking Ensemble (Meta-LR)": StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(**rf_kw, class_weight="balanced", random_state=42, n_jobs=4)),
            ("et", ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=4)),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        cv=5,
        n_jobs=2
    ),
}

ablation_rows = []
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for arch_name, clf_obj in ablation_clfs.items():
    pipe_ab = Pipeline([("prep", build_preprocessor()), ("clf", clf_obj)])
    cv_ab = cross_validate(pipe_ab, X_train_raw, y_train, cv=cv5, scoring="roc_auc", n_jobs=2)
    pipe_ab.fit(X_train_raw, y_train)
    m_ab = evaluate_metrics(pipe_ab, X_val_raw, y_val)
    ablation_rows.append({
        "Architecture": arch_name,
        "CV_ROC_AUC_Mean": round(float(cv_ab["test_score"].mean()), 4),
        "Val_Accuracy": round(m_ab["Accuracy"], 4),
        "Val_Unsafe_Recall_Spec": round(m_ab["Unsafe_Recall_Spec"], 4),
        "Val_False_Safe_Rate": round(m_ab["False_Safe_Rate"], 4),
        "Val_Potable_Prec": round(m_ab["Precision"], 4),
        "Val_Potable_Recall": round(m_ab["Recall"], 4),
        "Val_Potable_F1": round(m_ab["F1-Score"], 4),
        "Val_ROC_AUC": round(m_ab["ROC-AUC"], 4),
        "Val_Brier": round(m_ab["Brier_Score"], 4),
    })

ablation_df = pd.DataFrame(ablation_rows).sort_values(by="Val_ROC_AUC", ascending=False)
ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_ablation_study.csv"), index=False)
print(ablation_df.to_string(index=False))

# Save all trained pipelines + single source-of-truth best pipeline (compressed)
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_pipelines[best_model_name]

joblib.dump(trained_pipelines, os.path.join(MODEL_DIR, "all_trained_pipelines.joblib"), compress=3)
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model_pipeline.joblib"), compress=3)
with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_model_name)

print(f"\nSaved compressed pipelines & single source-of-truth best_model_pipeline.joblib: {best_model_name}")
