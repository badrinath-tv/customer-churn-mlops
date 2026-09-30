"""
Final Training Script
AI-Powered Customer Churn Prediction

Selected benchmark configuration:
    CatBoost + SMOTE

Input:
    data/processed/train.csv
    data/processed/test.csv

Output:
    models/final_model/churn_pipeline.pkl
    models/final_model/final_metrics.csv
    models/final_model/feature_preprocessor.pkl
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "final_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PATH = MODEL_DIR / "churn_pipeline.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "feature_preprocessor.pkl"
METRICS_PATH = MODEL_DIR / "final_metrics.csv"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.csv"
INFO_PATH = MODEL_DIR / "model_info.json"

RANDOM_STATE = 42
TARGET = "Exited"


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def load_data():
    print("\n" + "=" * 80)
    print("LOADING FINAL TRAINING DATA")
    print("=" * 80)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    if TARGET not in train_df.columns:
        raise ValueError(f"{TARGET} missing from train.csv")

    if TARGET not in test_df.columns:
        raise ValueError(f"{TARGET} missing from test.csv")

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------
# PREPROCESSOR
# ---------------------------------------------------------------------

def create_preprocessor(X_train):
    drop_columns = [
        c for c in ["RowNumber", "CustomerId", "Surname"]
        if c in X_train.columns
    ]

    X_reference = X_train.drop(
        columns=drop_columns,
        errors="ignore"
    )

    categorical_columns = [
        c for c in ["Geography", "Gender"]
        if c in X_reference.columns
    ]

    numerical_columns = [
        c for c in X_reference.columns
        if c not in categorical_columns
    ]

    print("\nDropped identifier columns:", drop_columns)
    print("Categorical columns:", categorical_columns)
    print("Numerical columns:", numerical_columns)

    numerical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="passthrough",
    )

    return preprocessor, drop_columns


# ---------------------------------------------------------------------
# FINAL MODEL
# ---------------------------------------------------------------------

def create_model():
    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        task_type="CPU",
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("AI-POWERED CUSTOMER CHURN PREDICTION")
    print("FINAL MODEL TRAINING")
    print("=" * 80)
    print("Selected configuration: CatBoost + SMOTE")
    print("Random state:", RANDOM_STATE)

    # Load
    X_train, y_train, X_test, y_test = load_data()

    # Remove identifiers before pipeline
    preprocessor, drop_columns = create_preprocessor(X_train)

    X_train_model = X_train.drop(
        columns=drop_columns,
        errors="ignore"
    )

    X_test_model = X_test.drop(
        columns=drop_columns,
        errors="ignore"
    )

    # Final pipeline:
    # preprocessing -> SMOTE -> CatBoost
    model = create_model()

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )

    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)

    print("Pipeline:")
    print("  1. Missing-value handling")
    print("  2. Standard scaling")
    print("  3. One-hot encoding")
    print("  4. SMOTE")
    print("  5. CatBoost")

    start_time = time.time()

    pipeline.fit(
        X_train_model,
        y_train
    )

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds.")

    # Prediction
    print("\n" + "=" * 80)
    print("FINAL MODEL EVALUATION")
    print("=" * 80)

    y_pred = pipeline.predict(X_test_model)
    y_prob = pipeline.predict_proba(X_test_model)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Train Time: {training_time:.2f} sec")

    print("\nConfusion Matrix:")
    print(cm)

    # Save complete deployable pipeline
    joblib.dump(
        pipeline,
        PIPELINE_PATH
    )

    # Save the fitted preprocessor separately for inspection
    fitted_preprocessor = pipeline.named_steps["preprocessor"]

    joblib.dump(
        fitted_preprocessor,
        PREPROCESSOR_PATH
    )

    # Save metrics
    metrics_df = pd.DataFrame(
        [
            {
                "Model": "CatBoost",
                "Strategy": "SMOTE",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC_AUC": roc_auc,
                "Training_Time": training_time,
            }
        ]
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    )

    cm_df.to_csv(
        CONFUSION_PATH
    )

    # Save model information
    model_info = {
        "model": "CatBoost",
        "imbalance_strategy": "SMOTE",
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_time_seconds": float(training_time),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        },
        "pipeline": [
            "Identifier column removal",
            "Median imputation",
            "Standard scaling",
            "One-hot encoding",
            "SMOTE",
            "CatBoost classifier",
        ],
    }

    with open(
        INFO_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            model_info,
            file,
            indent=4
        )

    print("\n" + "=" * 80)
    print("FINAL MODEL SAVED SUCCESSFULLY")
    print("=" * 80)

    print(f"\nDeployable pipeline:")
    print(PIPELINE_PATH)

    print(f"\nPreprocessor:")
    print(PREPROCESSOR_PATH)

    print(f"\nMetrics:")
    print(METRICS_PATH)

    print(f"\nConfusion matrix:")
    print(CONFUSION_PATH)

    print(f"\nModel information:")
    print(INFO_PATH)

    print("\n" + "=" * 80)
    print("READY FOR DEPLOYMENT")
    print("=" * 80)


if __name__ == "__main__":
    main()
