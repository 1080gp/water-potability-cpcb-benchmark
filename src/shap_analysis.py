import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("COMPUTING SHAP EXPLAINABILITY VALUES & PLOTS")
print("=" * 70)

# 1. Load Single Source-of-Truth Pipeline & Raw Test Data
best_pipeline = joblib.load(os.path.join(MODEL_DIR, "best_model_pipeline.joblib"))
best_model = best_pipeline.named_steps["clf"]
X_test_raw = np.load(os.path.join(PROCESSED_DIR, "X_test_raw.npy"))
X_test = best_pipeline.named_steps["prep"].transform(X_test_raw)

feature_file = os.path.join(PROCESSED_DIR, "features.txt")
if os.path.exists(feature_file):
    with open(feature_file, "r") as f:
        feature_names = [line.strip() for line in f if line.strip()]
else:
    feature_names = [
        "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
        "ph_sulfate_ratio"
    ]

print(f"Features: {feature_names}")
print(f"Model: {type(best_model).__name__}")
print(f"Analyzing {len(X_test)} unseen test samples...")

# 2. Compute TreeExplainer SHAP Values
explainer = shap.TreeExplainer(best_model)
# For binary classification, slice for class 1 (Potable)
shap_values = explainer.shap_values(X_test)

# Check shap_values structure (some versions return list of [N, M] per class, or [N, M, 2])
if isinstance(shap_values, list):
    shap_vals_potable = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_potable = shap_values[:, :, 1]
else:
    shap_vals_potable = shap_values

print("SHAP values calculated successfully. Shape:", shap_vals_potable.shape)

# Convert to DataFrame for tabular reporting
mean_abs_shap = np.abs(shap_vals_potable).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": np.round(mean_abs_shap, 5)
}).sort_values(by="Mean_Absolute_SHAP", ascending=False)

importance_csv_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
shap_importance_df.to_csv(importance_csv_path, index=False)
print(f"\nSaved feature importance to:\n{importance_csv_path}")
print(shap_importance_df.to_string(index=False))

# 3. Generate SHAP Summary / Beeswarm Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals_potable,
    X_test,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Feature Attribution Summary (Potability Prediction)", fontsize=13, pad=15)
plt.tight_layout()
summary_plot_path = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved SHAP summary plot to: {summary_plot_path}")

# 4. Generate SHAP Bar Plot
plt.figure(figsize=(9, 5))
shap.summary_plot(
    shap_vals_potable,
    X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("Mean Absolute SHAP Value (Global Impact on Model Output)", fontsize=12, pad=15)
plt.tight_layout()
bar_plot_path = os.path.join(OUTPUT_DIR, "shap_bar_plot.png")
plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved SHAP bar plot to: {bar_plot_path}")

# 5. Generate SHAP Dependence Plot for ph_sulfate_ratio vs ph
ratio_idx = feature_names.index("ph_sulfate_ratio")
ph_idx = feature_names.index("ph")

plt.figure(figsize=(8, 5))
shap.dependence_plot(
    ratio_idx,
    shap_vals_potable,
    X_test,
    feature_names=feature_names,
    interaction_index=ph_idx,
    show=False
)
plt.title("SHAP Interaction: pH-to-Sulfate Ratio vs. pH", fontsize=12, pad=15)
plt.tight_layout()
dep_plot_path = os.path.join(OUTPUT_DIR, "shap_dependence_ratio.png")
plt.savefig(dep_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved SHAP dependence plot to: {dep_plot_path}")

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)
