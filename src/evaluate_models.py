import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.calibration import CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from preprocessing import get_splits

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def wilson_ci(successes, total, z=1.95996):
    p = successes / total
    denom = 1.0 + z**2 / total
    centre = (p + z**2 / (2.0 * total)) / denom
    margin = (z * np.sqrt(p * (1.0 - p) / total + z**2 / (4.0 * total**2))) / denom
    return round((centre - margin) * 100, 2), round((centre + margin) * 100, 2)


# 1. Load splits & pre-trained pipelines (no retraining!)
_, X_val_raw, X_test_raw, y_train, y_val, y_test, _, _ = get_splits()
all_pipelines = joblib.load(os.path.join(MODEL_DIR, "all_trained_pipelines.joblib"))
with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "r") as f:
    best_model_name = f.read().strip()

best_model = all_pipelines[best_model_name]

print("=" * 75)
print("FINAL EVALUATION ON HELD-OUT TEST SPLIT (n = 492)")
print("=" * 75)

rows = []
n_test = len(y_test)
for name, pipe in all_pipelines.items():
    preds = pipe.predict(X_test_raw)
    probs_pos1 = pipe.predict_proba(X_test_raw)[:, 1]
    acc = accuracy_score(y_test, preds)
    ci_low, ci_high = wilson_ci(int(round(acc * n_test)), n_test)
    unsafe_rec = recall_score(y_test, preds, pos_label=0, zero_division=0)
    rows.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Acc_95_CI": f"[{ci_low}%, {ci_high}%]",
        "Unsafe_Recall_Spec": round(unsafe_rec, 4),
        "False_Safe_Rate": round(1.0 - unsafe_rec, 4),
        "Unsafe_Precision_NPV": round(precision_score(y_test, preds, pos_label=0, zero_division=0), 4),
        "Precision": round(precision_score(y_test, preds, pos_label=1, zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, pos_label=1, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, preds, pos_label=1, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, probs_pos1), 4),
        "Brier_Score": round(brier_score_loss(y_test, probs_pos1), 4)
    })

metrics_df = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False)
print("\nTest Set Evaluation Metrics (All Models):")
print(metrics_df.to_string(index=False))

metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_evaluation.csv"), index=False)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"), index=False)

# Best Model Confusion Matrix
predictions = best_model.predict(X_test_raw)
probabilities = best_model.predict_proba(X_test_raw)[:, 1]
roc_auc = roc_auc_score(y_test, probabilities)
cm = confusion_matrix(y_test, predictions)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-Potable (Unsafe)", "Actual Potable (Safe)"],
    columns=["Predicted Non-Potable (Unsafe)", "Predicted Potable (Safe)"]
)
cm_df.to_csv(os.path.join(OUTPUT_DIR, "final_confusion_matrix.csv"))

print(f"\nSelected Best Model ({best_model_name}) Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(7, 5.5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Potable (0)", "Potable (1)"])
disp.plot(ax=ax, cmap="Blues")
ax.set_title(f"Confusion Matrix - {best_model_name}")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, probabilities)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(fpr, tpr, linewidth=2, label=f"{best_model_name} (ROC-AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
ax.set_xlabel("False Positive Rate (False-Safe Rate)")
ax.set_ylabel("True Positive Rate (Potable Recall)")
ax.set_title(f"ROC Curve - {best_model_name}")
ax.legend(loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Precision-Recall Curve for Unsafe Class (Class 0: Non-Potable)
y_unsafe = (y_test == 0).astype(int)
probs_unsafe = best_model.predict_proba(X_test_raw)[:, 0]
prec_u, rec_u, _ = precision_recall_curve(y_unsafe, probs_unsafe)
ap_u = average_precision_score(y_unsafe, probs_unsafe)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(rec_u, prec_u, linewidth=2, color="#DC2626", label=f"Unsafe Class (0) PR Curve (AP = {ap_u:.4f})")
ax.axhline(y=y_unsafe.mean(), linestyle="--", color="gray", label=f"Baseline Prevalence ({y_unsafe.mean():.2f})")
ax.set_xlabel("Unsafe Recall / Specificity (Catching Unsafe Water)")
ax.set_ylabel("Unsafe Precision / NPV")
ax.set_title(f"Unsafe Water (Class 0) Precision-Recall Curve - {best_model_name}")
ax.legend(loc="lower left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "unsafe_pr_curve.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 2. Frozen Threshold Evaluation on Held-Out Test Split (0.50 Default vs. 0.60 Safety-First frozen on Val)
thr_rows = []
for label, thr in [("Default Score Cutoff (0.50)", 0.50), ("Frozen Safety-First Cutoff (0.60, from Val)", 0.60)]:
    t_preds = (probabilities >= thr).astype(int)
    u_rec = recall_score(y_test, t_preds, pos_label=0, zero_division=0)
    p_rec = recall_score(y_test, t_preds, pos_label=1, zero_division=0)
    thr_rows.append({
        "Operating_Point": label,
        "Score_Threshold": thr,
        "Test_Accuracy": round(accuracy_score(y_test, t_preds), 4),
        "Unsafe_Recall_Spec": round(u_rec, 4),
        "False_Safe_Rate": round(1.0 - u_rec, 4),
        "Potable_Precision": round(precision_score(y_test, t_preds, pos_label=1, zero_division=0), 4),
        "Potable_Recall": round(p_rec, 4),
        "Safe_Water_Rejected_Rate": round(1.0 - p_rec, 4),
        "Potable_F1": round(f1_score(y_test, t_preds, pos_label=1, zero_division=0), 4),
    })
thr_comp_df = pd.DataFrame(thr_rows)
thr_comp_df.to_csv(os.path.join(OUTPUT_DIR, "test_threshold_comparison.csv"), index=False)
print("\nTest Set Operating-Point Comparison (Frozen Thresholds):")
print(thr_comp_df.to_string(index=False))

# 3. Probability Calibration Audit (Base-Rate Brier Skill + Isotonic Calibration on Validation Split)
prevalence_train = float(np.mean(y_train))
brier_base = brier_score_loss(y_test, np.full_like(y_test, prevalence_train, dtype=float))
brier_raw = brier_score_loss(y_test, probabilities)
bss_raw = 1.0 - (brier_raw / brier_base)

val_probs = best_model.predict_proba(X_val_raw)[:, 1]
iso_cal = IsotonicRegression(out_of_bounds="clip")
iso_cal.fit(val_probs, y_val)
probs_cal = iso_cal.predict(probabilities)
brier_cal = brier_score_loss(y_test, probs_cal)
bss_cal = 1.0 - (brier_cal / brier_base)

cal_summary_df = pd.DataFrame([
    {"Model_Calibration": "Always-Predict-Prevalence Baseline", "Brier_Score": round(brier_base, 4), "Brier_Skill_Score": 0.0},
    {"Model_Calibration": "Raw Balanced RF Scores (Uncalibrated)", "Brier_Score": round(brier_raw, 4), "Brier_Skill_Score": round(bss_raw, 4)},
    {"Model_Calibration": "Isotonic-Calibrated RF (Fit on Val)", "Brier_Score": round(brier_cal, 4), "Brier_Skill_Score": round(bss_cal, 4)},
])
cal_summary_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_summary.csv"), index=False)
print("\nProbability Calibration & Brier Skill Score:")
print(cal_summary_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 5.5))
CalibrationDisplay.from_predictions(
    y_test, probabilities, n_bins=10, name=f"Raw Balanced RF (Brier={brier_raw:.4f})", ax=ax
)
CalibrationDisplay.from_predictions(
    y_test, probs_cal, n_bins=10, name=f"Isotonic Calibrated (Brier={brier_cal:.4f})", ax=ax
)
ax.set_title("Reliability / Calibration Curve (Held-Out Test Split, n=492)")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "calibration_curve.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nAll test metrics, threshold comparisons, and calibration figures saved successfully.")
