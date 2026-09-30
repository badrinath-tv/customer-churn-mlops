import os
import time
import warnings
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

RESULTS_DIR = "results"
MODELS_DIR = "models/benchmark_models"

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

    preprocessor = ColumnTransformer(
        [
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
        remainder="passthrough"
    )

    return preprocessor


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(
            y_test,
            preds,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            preds,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            preds,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probs
        )
    }


def get_original_models():

    return {

        "LogisticRegression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "DecisionTree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=200,
                n_jobs=-1
            ),

        "LightGBM":
            LGBMClassifier(
                n_estimators=200,
                random_state=42,
                verbose=-1
            ),

        "CatBoost":
            CatBoostClassifier(
                iterations=200,
                verbose=0,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier(
                n_neighbors=5
            ),

        "SVM":
            SVC(
                probability=True,
                random_state=42
            ),

        "NaiveBayes":
            GaussianNB()
    }


def get_classweight_models(scale_pos):

    return {

        "LogisticRegression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),

        "DecisionTree":
            DecisionTreeClassifier(
                class_weight="balanced",
                random_state=42
            ),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos,
                random_state=42,
                n_estimators=200,
                n_jobs=-1
            ),

        "LightGBM":
            LGBMClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                verbose=-1
            ),

        "CatBoost":
            CatBoostClassifier(
                iterations=200,
                class_weights=[1, scale_pos],
                verbose=0,
                random_state=42
            ),

        "SVM":
            SVC(
                probability=True,
                class_weight="balanced",
                random_state=42
            )
    }

def save_model(model, name, strategy):

    os.makedirs(MODELS_DIR, exist_ok=True)

    path = os.path.join(
        MODELS_DIR,
        f"{name}_{strategy}.pkl"
    )

    joblib.dump(model, path)


def run_models(
    strategy,
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    results
):

    for name, model in models.items():

        print(f"\n[{strategy}] {name}")

        start = time.time()

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        train_time = round(
            time.time() - start,
            2
        )

        row = {
            "Strategy": strategy,
            "Model": name,
            **metrics,
            "Train_Time": train_time
        }

        results.append(row)

        save_model(
            model,
            name,
            strategy
        )

        print(
            f"F1={metrics['F1']:.4f}"
        )


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    results = []

    X_train, X_test, y_train, y_test = (
        load_data()
    )

    preprocessor = get_preprocessor()

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    scale_pos = (
        (y_train == 0).sum()
        /
        (y_train == 1).sum()
    )

    print("\n" + "=" * 60)
    print("RUNNING ORIGINAL EXPERIMENTS")
    print("=" * 60)

    run_models(
        "Original",
        get_original_models(),
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        results
    )

    print("\n" + "=" * 60)
    print("RUNNING CLASS WEIGHT EXPERIMENTS")
    print("=" * 60)

    run_models(
        "ClassWeight",
        get_classweight_models(scale_pos),
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        results
    )

    results.append({
        "Strategy": "ClassWeight",
        "Model": "KNN",
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "ROC_AUC": None,
        "Train_Time": None
    })

    results.append({
        "Strategy": "ClassWeight",
        "Model": "NaiveBayes",
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "ROC_AUC": None,
        "Train_Time": None
    })

    print("\n" + "=" * 60)
    print("APPLYING SMOTE")
    print("=" * 60)

    smote = SMOTE(
        random_state=42
    )

    X_smote, y_smote = (
        smote.fit_resample(
            X_train_processed,
            y_train
        )
    )

    print(
        f"SMOTE Shape: {X_smote.shape}"
    )

    print("\n" + "=" * 60)
    print("RUNNING SMOTE EXPERIMENTS")
    print("=" * 60)

    run_models(
        "SMOTE",
        get_original_models(),
        X_smote,
        y_smote,
        X_test_processed,
        y_test,
        results
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/full_benchmark_results.csv",
        index=False
    )

    ranking_df = (
        results_df
        .dropna(subset=["F1"])
        .sort_values(
            by="F1",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking_df.index += 1

    ranking_df.to_csv(
        "results/model_ranking.csv",
        index_label="Rank"
    )

    ranking_df.head(5).to_csv(
        "results/top_5_models.csv",
        index_label="Rank"
    )

    winner = ranking_df.iloc[0]

    print("\n")
    print("=" * 70)
    print("TOP 10 MODELS")
    print("=" * 70)

    print(
        ranking_df[
            [
                "Strategy",
                "Model",
                "F1",
                "ROC_AUC",
                "Train_Time"
            ]
        ].head(10)
    )

    print("\n")
    print("=" * 70)
    print("WINNER")
    print("=" * 70)

    print(
        f"Strategy : {winner['Strategy']}"
    )
    print(
        f"Model    : {winner['Model']}"
    )
    print(
        f"F1 Score : {winner['F1']:.4f}"
    )
    print(
        f"ROC_AUC  : {winner['ROC_AUC']:.4f}"
    )

    print("\nSaved Files:")
    print(
        "results/full_benchmark_results.csv"
    )
    print(
        "results/model_ranking.csv"
    )
    print(
        "results/top_5_models.csv"
    )


if __name__ == "__main__":
    main()

