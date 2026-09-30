import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

RESULTS_DIR = "results"
MODELS_DIR = "models"

TARGET = "Exited"

DROP_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname"
]


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = train_df.drop(columns=DROP_COLUMNS)
    test_df = test_df.drop(columns=DROP_COLUMNS)

    X_train = train_df.drop(TARGET, axis=1)
    y_train = train_df[TARGET]

    X_test = test_df.drop(TARGET, axis=1)
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test


def get_preprocessor():

    numeric_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary"
    ]

    categorical_features = [
        "Geography",
        "Gender"
    ]

    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat",
             OneHotEncoder(
                 drop="first",
                 handle_unknown="ignore"
             ),
             categorical_features)
        ],
        remainder="passthrough"
    )


def evaluate(name, strategy, model,
             X_test, y_test):

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "Strategy": strategy,
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC_AUC": roc_auc_score(y_test, probs)
    }


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    preprocessor = get_preprocessor()

    results = []

    # ---------- ORIGINAL ----------

    original_models = {
        "LogisticRegression":
            LogisticRegression(max_iter=1000),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=42
            )
    }

    for name, model in original_models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        results.append(
            evaluate(
                name,
                "Original",
                pipeline,
                X_test,
                y_test
            )
        )

    # ---------- CLASS WEIGHT ----------

    scale_pos = (
        (y_train == 0).sum()
        /
        (y_train == 1).sum()
    )

    weighted_models = {
        "LogisticRegression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                scale_pos_weight=scale_pos
            )
    }

    for name, model in weighted_models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        results.append(
            evaluate(
                name,
                "ClassWeight",
                pipeline,
                X_test,
                y_test
            )
        )

    # ---------- SMOTE ----------

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)

    X_smote, y_smote = smote.fit_resample(
        X_train_processed,
        y_train
    )

    smote_models = {
        "LogisticRegression":
            LogisticRegression(max_iter=1000),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=42
            )
    }

    best_model = None
    best_f1 = 0

    for name, model in smote_models.items():

        model.fit(X_smote, y_smote)

        result = evaluate(
            name,
            "SMOTE",
            model,
            X_test_processed,
            y_test
        )

        results.append(result)

        if result["F1"] > best_f1:
            best_f1 = result["F1"]
            best_model = model

    results_df = pd.DataFrame(results)

    results_path = os.path.join(
        RESULTS_DIR,
        "experiment_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n")
    print(results_df)

    print("\nSaved:")
    print(results_path)

    print("\nBest F1:", best_f1)

    if best_model:
        joblib.dump(
            best_model,
            os.path.join(
                MODELS_DIR,
                "best_experiment_model.pkl"
            )
        )


if __name__ == "__main__":
    main()