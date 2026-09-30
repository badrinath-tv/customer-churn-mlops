import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Paths
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_DIR = "models"

TARGET_COLUMN = "Exited"

# Columns to remove
DROP_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname"
]


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def build_preprocessor():

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

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(drop="first",
                                  handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="passthrough"
    )

    return preprocessor


def preprocess_data():

    train_df, test_df = load_data()

    # Remove unwanted columns
    train_df = train_df.drop(columns=DROP_COLUMNS)
    test_df = test_df.drop(columns=DROP_COLUMNS)

    # Split X and y
    X_train = train_df.drop(TARGET_COLUMN, axis=1)
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(TARGET_COLUMN, axis=1)
    y_test = test_df[TARGET_COLUMN]

    # Build transformer
    preprocessor = build_preprocessor()

    # Fit transform train
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test
    X_test_processed = preprocessor.transform(X_test)

    # Save pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        preprocessor,
        os.path.join(MODEL_DIR, "preprocessor.pkl")
    )

    print("=" * 50)
    print("Preprocessing Complete")
    print("=" * 50)

    print("X_train:", X_train_processed.shape)
    print("X_test :", X_test_processed.shape)

    print("\nSaved:")
    print("models/preprocessor.pkl")

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    )


if __name__ == "__main__":
    preprocess_data()