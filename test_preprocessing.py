import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd

from research.data.preprocessing import DataPreprocessor

df = pd.read_csv(
    "data/raw/Churn_Modelling.csv"
)

preprocessor = DataPreprocessor(
    target_column="Exited",
    drop_columns=[
        "RowNumber",
        "CustomerId",
        "Surname",
    ],
)

X, y, encoders = preprocessor.preprocess(df)

print("\nFinal Features")

print(X.head())

print("\nTarget")

print(y.head())