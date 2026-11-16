import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_results.csv"
)

df = pd.read_csv(RESULTS_PATH)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\nValidation Results:")
print(df.to_string(index=False))


def save_bar_chart(metric_name, title, filename, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["Model"], df[metric_name], color=color, edgecolor="black", alpha=0.88)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# Accuracy, F1-Score, ROC-AUC Comparison Charts
# ------------------------------------------------------------

accuracy_path = save_bar_chart(
    "Accuracy",
    "Model Accuracy Comparison (Validation Set)",
    "accuracy_comparison.png",
    color="#2b6cb0"
)

f1_path = save_bar_chart(
    "F1-Score",
    "Model F1-Score Comparison (Validation Set)",
    "f1_comparison.png",
    color="#2f855a"
)

roc_path = save_bar_chart(
    "ROC-AUC",
    "Model ROC-AUC Comparison (Validation Set)",
    "roc_auc_comparison.png",
    color="#6b46c1"
)


# ------------------------------------------------------------
# Combined Metrics
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

x = range(len(df))

width = 0.15

for i, metric in enumerate(metrics):

    positions = [
        value + (i - 2) * width
        for value in x
    ]

    ax.bar(
        positions,
        df[metric],
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    df["Model"],
    rotation=25,
    ha="right"
)

ax.set_ylabel("Score")
ax.set_title("Machine Learning Model Performance Comparison", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.25)
ax.legend()

fig.tight_layout()

combined_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

fig.savefig(
    combined_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print("\nGraphs saved successfully:")

print(accuracy_path)
print(f1_path)
print(roc_path)
print(combined_path)

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)
