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


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_DIR = "models"

TARGET = "Exited"

DROP_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname"
]


def get_data():

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

    preprocessor = ColumnTransformer([
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough")

    return preprocessor


def evaluate_model(name, model, X_test, y_test):

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    return f1


def main():

    X_train, X_test, y_train, y_test = get_data()

    preprocessor = get_preprocessor()

    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Random Forest":
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
    best_score = 0
    best_name = None

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        score = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test
        )

        if score > best_score:
            best_score = score
            best_model = pipeline
            best_name = name

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )

    joblib.dump(best_model, model_path)

    print("\n" + "=" * 50)
    print("BEST MODEL")
    print("=" * 50)

    print(best_name)
    print(f"F1 Score: {best_score:.4f}")

    print("\nSaved:")
    print(model_path)


if __name__ == "__main__":
    main()