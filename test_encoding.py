import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd

from research.data.encoding import FeatureEncoder

df = pd.read_csv("data/raw/Churn_Modelling.csv")

encoder = FeatureEncoder()

encoded_df, encoders = encoder.fit_transform(df)

encoder.summary()

print("\nShape:", encoded_df.shape)

print("\nData Types:")
print(encoded_df.dtypes)