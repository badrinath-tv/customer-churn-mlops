"""
AI-Powered Customer Churn Prediction
Final Model Benchmark

Uses the existing project train/test split:

    data/processed/train.csv
    data/processed/test.csv

Experiments:
    1. Original
    2. Class Weighting
    3. SMOTE

Models:
    Logistic Regression
    Decision Tree
    Random Forest
    XGBoost
    LightGBM
    CatBoost
    KNN
    Naive Bayes
    SVM

Author:
    Badrinath T V
    Abdul Azim M
"""

import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

warnings.filterwarnings("ignore")

# =============================================================================
# MACHINE LEARNING IMPORTS
# =============================================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "final_benchmark"
MODELS_DIR = PROJECT_ROOT / "models" / "final_benchmark"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TARGET = "Exited"

TEST_SIZE = 0.20

print("\n" + "=" * 80)
print("AI-POWERED CUSTOMER CHURN PREDICTION")
print("FINAL MODEL BENCHMARK")
print("=" * 80)

print("\nConfiguration:")
print("Python environment : CPU-compatible")
print(f"Random state       : {RANDOM_STATE}")
print(f"Existing split     : Train = 80% / Test = 20%")
print(f"Target             : {TARGET}")


# =============================================================================
# LOAD EXISTING TRAIN / TEST DATA
# =============================================================================

def load_dataset():

    print("\n" + "=" * 80)
    print("LOADING EXISTING TRAIN / TEST DATA")
    print("=" * 80)

    print(f"\nTrain path: {TRAIN_PATH}")
    print(f"Test path : {TEST_PATH}")

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"\nTrain file not found:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"\nTest file not found:\n{TEST_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    if TARGET not in train_df.columns:
        raise ValueError(
            f"\nTarget column '{TARGET}' not found in train.csv.\n"
            f"Available columns:\n{train_df.columns.tolist()}"
        )

    if TARGET not in test_df.columns:
        raise ValueError(
            f"\nTarget column '{TARGET}' not found in test.csv.\n"
            f"Available columns:\n{test_df.columns.tolist()}"
        )

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    print("\nClass percentages in training data:")
    print(
        (y_train.value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )

    return X_train, y_train, X_test, y_test


# =============================================================================
# PREPROCESSING
# =============================================================================

def create_preprocessor(X_train):

    print("\n" + "=" * 80)
    print("CREATING PREPROCESSING PIPELINE")
    print("=" * 80)

    # Remove identifiers / non-predictive columns
    columns_to_drop = [
        "RowNumber",
        "CustomerId",
        "Surname",
    ]

    existing_drop_columns = [
        col for col in columns_to_drop
        if col in X_train.columns
    ]

    if existing_drop_columns:
        print(
            "\nDropping identifier columns:",
            existing_drop_columns
        )

        X_train = X_train.drop(
            columns=existing_drop_columns
        )

    # Identify categorical columns
    categorical_candidates = [
        "Geography",
        "Gender",
    ]

    categorical_columns = [
        col
        for col in categorical_candidates
        if col in X_train.columns
    ]

    # Everything else is treated as numerical
    numerical_columns = [
        col
        for col in X_train.columns
        if col not in categorical_columns
    ]

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nNumerical columns:")
    print(numerical_columns)

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
        ],
        remainder="passthrough"
    )

    return preprocessor, existing_drop_columns


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_original_models():

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            device="cpu"
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        ),

        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
            task_type="CPU"
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Naive Bayes": GaussianNB(),

        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
    }


# =============================================================================
# CLASS-WEIGHT MODELS
# =============================================================================

def get_class_weight_models():

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            device="cpu"
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        ),

        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=[1, 3.91],
            random_seed=RANDOM_STATE,
            verbose=False,
            task_type="CPU"
        ),

        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
    }


# =============================================================================
# PREPROCESS DATA
# =============================================================================

def prepare_data(
    X_train,
    X_test,
    drop_columns
):

    X_train = X_train.copy()
    X_test = X_test.copy()

    if drop_columns:

        X_train = X_train.drop(
            columns=drop_columns,
            errors="ignore"
        )

        X_test = X_test.drop(
            columns=drop_columns,
            errors="ignore"
        )

    return X_train, X_test


# =============================================================================
# METRIC CALCULATION
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )
    except Exception:
        roc_auc = np.nan

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
    }


# =============================================================================
# GET PROBABILITY
# =============================================================================

def get_probability(model, X):

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)

        if probabilities.ndim == 2:
            return probabilities[:, 1]

        return probabilities

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(X)

        # Convert decision scores to probability-like values
        scores = np.asarray(scores)

        probabilities = (
            1 / (1 + np.exp(-scores))
        )

        return probabilities

    else:

        return model.predict(X)


# =============================================================================
# RUN SINGLE EXPERIMENT
# =============================================================================

def run_experiment(
    model_name,
    strategy,
    model,
    X_train_processed,
    y_train,
    X_test_processed,
    y_test
):

    print(
        f"\n{'-' * 80}"
    )

    print(
        f"Experiment: {model_name} + {strategy}"
    )

    print(
        f"{'-' * 80}"
    )

    # -------------------------------------------------------------------------
    # SMOTE
    # -------------------------------------------------------------------------

    if strategy == "SMOTE":

        print("Applying SMOTE...")

        smote = SMOTE(
            random_state=RANDOM_STATE
        )

        X_fit, y_fit = smote.fit_resample(
            X_train_processed,
            y_train
        )

        print(
            f"Before SMOTE: {len(y_train)} samples"
        )

        print(
            f"After SMOTE : {len(y_fit)} samples"
        )

    else:

        X_fit = X_train_processed
        y_fit = y_train

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------

    print("Training model...")

    start_time = time.time()

    model.fit(
        X_fit,
        y_fit
    )

    training_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    print("Generating predictions...")

    y_pred = model.predict(
        X_test_processed
    )

    y_probability = get_probability(
        model,
        X_test_processed
    )

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability
    )

    metrics["Training_Time"] = training_time

    # -------------------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------------------

    print(
        f"Accuracy  : {metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['Precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score  : {metrics['F1']:.4f}"
    )

    print(
        f"ROC-AUC   : {metrics['ROC_AUC']:.4f}"
    )

    print(
        f"Train Time: {metrics['Training_Time']:.2f} sec"
    )

    # -------------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------------

    safe_model_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    safe_strategy = (
        strategy
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_filename = (
        f"{safe_model_name}_{safe_strategy}.pkl"
    )

    model_path = MODELS_DIR / model_filename

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Saved model: {model_path}"
    )

    # -------------------------------------------------------------------------
    # RESULT DICTIONARY
    # -------------------------------------------------------------------------

    result = {
        "Model": model_name,
        "Strategy": strategy,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "ROC_AUC": metrics["ROC_AUC"],
        "Training_Time": metrics["Training_Time"],
        "Model_File": str(model_path.relative_to(PROJECT_ROOT)),
    }

    return result


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def main():

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    X_train, y_train, X_test, y_test = load_dataset()

    # =========================================================================
    # CREATE PREPROCESSOR
    # =========================================================================

    preprocessor, drop_columns = create_preprocessor(
        X_train
    )

    # =========================================================================
    # REMOVE IDENTIFIER COLUMNS
    # =========================================================================

    X_train, X_test = prepare_data(
        X_train,
        X_test,
        drop_columns
    )

    print("\nFinal feature columns:")
    print(
        X_train.columns.tolist()
    )

    # =========================================================================
    # FIT PREPROCESSOR ONLY ON TRAIN DATA
    # =========================================================================

    print("\n" + "=" * 80)
    print("FITTING PREPROCESSOR")
    print("=" * 80)

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        f"\nProcessed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed testing shape : "
        f"{X_test_processed.shape}"
    )

    # =========================================================================
    # SAVE PREPROCESSOR
    # =========================================================================

    preprocessor_path = (
        MODELS_DIR / "preprocessor.pkl"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    print(
        f"\nSaved preprocessor:"
        f"\n{preprocessor_path}"
    )

    # =========================================================================
    # RESULTS LIST
    # =========================================================================

    all_results = []

    # =========================================================================
    # EXPERIMENT 1: ORIGINAL
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("EXPERIMENT GROUP 1: ORIGINAL")
    print("=" * 80)

    original_models = get_original_models()

    for model_name, model in original_models.items():

        try:

            result = run_experiment(
                model_name=model_name,
                strategy="Original",
                model=model,
                X_train_processed=X_train_processed,
                y_train=y_train,
                X_test_processed=X_test_processed,
                y_test=y_test
            )

            all_results.append(result)

        except Exception as e:

            print(
                f"\nERROR in {model_name} + Original:"
            )

            print(e)

    # =========================================================================
    # EXPERIMENT 2: CLASS WEIGHT
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("EXPERIMENT GROUP 2: CLASS WEIGHTING")
    print("=" * 80)

    class_weight_models = get_class_weight_models()

    for model_name, model in class_weight_models.items():

        try:

            result = run_experiment(
                model_name=model_name,
                strategy="ClassWeight",
                model=model,
                X_train_processed=X_train_processed,
                y_train=y_train,
                X_test_processed=X_test_processed,
                y_test=y_test
            )

            all_results.append(result)

        except Exception as e:

            print(
                f"\nERROR in {model_name} + ClassWeight:"
            )

            print(e)

    # =========================================================================
    # EXPERIMENT 3: SMOTE
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("EXPERIMENT GROUP 3: SMOTE")
    print("=" * 80)

    smote_models = get_original_models()

    for model_name, model in smote_models.items():

        try:

            result = run_experiment(
                model_name=model_name,
                strategy="SMOTE",
                model=model,
                X_train_processed=X_train_processed,
                y_train=y_train,
                X_test_processed=X_test_processed,
                y_test=y_test
            )

            all_results.append(result)

        except Exception as e:

            print(
                f"\nERROR in {model_name} + SMOTE:"
            )

            print(e)

    # =========================================================================
    # CREATE RESULTS DATAFRAME
    # =========================================================================

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:

        raise RuntimeError(
            "No benchmark results were generated."
        )

    # =========================================================================
    # SAVE COMPLETE RESULTS
    # =========================================================================

    results_path = (
        RESULTS_DIR / "final_benchmark_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"\nSaved complete results:"
        f"\n{results_path}"
    )

    # =========================================================================
    # MODEL RANKING
    # =========================================================================

    ranking_df = results_df.sort_values(
        by=["F1", "ROC_AUC"],
        ascending=[False, False]
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "Rank",
        range(1, len(ranking_df) + 1)
    )

    ranking_path = (
        RESULTS_DIR / "model_ranking.csv"
    )

    ranking_df.to_csv(
        ranking_path,
        index=False
    )

    print(
        f"\nSaved ranking:"
        f"\n{ranking_path}"
    )

    # =========================================================================
    # TOP 5
    # =========================================================================

    top5_df = ranking_df.head(5)

    top5_path = (
        RESULTS_DIR / "top_5_models.csv"
    )

    top5_df.to_csv(
        top5_path,
        index=False
    )

    print(
        f"\nSaved Top 5:"
        f"\n{top5_path}"
    )

    # =========================================================================
    # DISPLAY TOP 10
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("TOP 10 BENCHMARK RESULTS")
    print("=" * 80)

    display_columns = [
        "Rank",
        "Model",
        "Strategy",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "Training_Time",
    ]

    print(
        ranking_df[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    # =========================================================================
    # CATBOOST + CLASS WEIGHT RESULT
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("CATBOOST + CLASS WEIGHTING")
    print("=" * 80)

    selected_rows = results_df[
        (results_df["Model"] == "CatBoost")
        &
        (results_df["Strategy"] == "ClassWeight")
    ]

    if not selected_rows.empty:

        selected = selected_rows.iloc[0]

        print(
            f"\nAccuracy  : "
            f"{selected['Accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{selected['Precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{selected['Recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{selected['F1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{selected['ROC_AUC']:.4f}"
        )

        print(
            f"Train Time: "
            f"{selected['Training_Time']:.2f} sec"
        )

    else:

        print(
            "\nCatBoost + ClassWeight result "
            "was not generated."
        )

    # =========================================================================
    # BEST MODEL ACCORDING TO F1
    # =========================================================================

    best = ranking_df.iloc[0]

    print("\n")
    print("=" * 80)
    print("BEST BENCHMARK CONFIGURATION")
    print("=" * 80)

    print(
        f"\nModel     : {best['Model']}"
    )

    print(
        f"Strategy  : {best['Strategy']}"
    )

    print(
        f"Accuracy  : {best['Accuracy']:.4f}"
    )

    print(
        f"Precision : {best['Precision']:.4f}"
    )

    print(
        f"Recall    : {best['Recall']:.4f}"
    )

    print(
        f"F1 Score  : {best['F1']:.4f}"
    )

    print(
        f"ROC-AUC   : {best['ROC_AUC']:.4f}"
    )

    print(
        f"Train Time: {best['Training_Time']:.2f} sec"
    )

    print(
        "\nNOTE:"
        "\nThe benchmark ranking is based primarily on F1-score,"
        "\nwith ROC-AUC used as the secondary sorting metric."
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n")
    print("=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated files:")

    print(
        f"\nResults:"
        f"\n  {results_path}"
        f"\n  {ranking_path}"
        f"\n  {top5_path}"
    )

    print(
        f"\nModels:"
        f"\n  {MODELS_DIR}"
    )

    print(
        f"\nPreprocessor:"
        f"\n  {preprocessor_path}"
    )

    print("\n" + "=" * 80)


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()