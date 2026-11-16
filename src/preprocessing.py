import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "water_potability.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]
PROCESSED_FEATURES = FEATURES + ["ph_sulfate_ratio"]
TARGET = "Potability"


def safe_reciprocal(x):
    """Compute element-wise reciprocal with a small epsilon guard against zero division."""
    x_arr = np.asarray(x, dtype=float)
    safe_x = np.where(np.abs(x_arr) < 1e-6, 1e-6, x_arr)
    return np.reciprocal(safe_x)


def build_preprocessor(imputer=None):
    """Build a self-contained sklearn ColumnTransformer (imputer -> ph/Sulfate ratio -> StandardScaler)."""
    if imputer is None:
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=4
            ),
            max_iter=10,
            random_state=42
        )

    ph_sulfate_ratio_branch = Pipeline(
        steps=[
            (
                "select_ph_inv_sulf",
                ColumnTransformer(
                    transformers=[
                        ("ph", "passthrough", [0]),
                        ("inv_sulf", FunctionTransformer(safe_reciprocal), [4])
                    ]
                )
            ),
            (
                "interact",
                PolynomialFeatures(
                    degree=2,
                    interaction_only=True,
                    include_bias=False
                )
            ),
            (
                "keep_ratio_only",
                ColumnTransformer(
                    transformers=[
                        ("ratio", "passthrough", [2])
                    ]
                )
            )
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        ("raw", FunctionTransformer(None)),
                        ("ph_sulfate_ratio", ph_sulfate_ratio_branch)
                    ]
                )
            ),
            ("scaler", StandardScaler())
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(range(len(FEATURES))))
        ]
    )


def get_splits(data_path=DATA_PATH):
    """Load dataset, drop any exact duplicates, and return stratified 70/15/15 splits."""
    df = pd.read_csv(data_path)
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, duplicates, df.shape


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test, duplicates, shape = get_splits()
    print("Dataset loaded successfully")
    print("Shape:", shape)
    print("Duplicate rows:", duplicates)
    print("\nDataset Split:")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    # Save raw splits for leak-free Pipeline CV
    np.save(os.path.join(OUTPUT_DIR, "X_train_raw.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val_raw.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test_raw.npy"), X_test)

    # Fit ONLY on Training Data for standalone processed arrays / SHAP / app.py
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_processed)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val_processed)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_processed)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    with open(os.path.join(OUTPUT_DIR, "features.txt"), "w") as f:
        for feature in PROCESSED_FEATURES:
            f.write(feature + "\n")

    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("\nPreprocessing completed successfully!")
