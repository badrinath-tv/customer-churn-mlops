import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd

from research.data.preprocessing import DataPreprocessor
from research.data.splitting import DataSplitter

df = pd.read_csv("data/raw/Churn_Modelling.csv")

preprocessor = DataPreprocessor(

    target_column="Exited",

    drop_columns=[
        "RowNumber",
        "CustomerId",
        "Surname",
    ],
)

X, y, _ = preprocessor.preprocess(df)

splitter = DataSplitter()

X_train, X_test, y_train, y_test = splitter.split(
    X,
    y,
)

print()

print(X_train.shape)

print(X_test.shape)