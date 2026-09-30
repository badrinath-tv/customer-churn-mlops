import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/Churn_Modelling.csv.xls"
PROCESSED_DIR = "data/processed"


def load_data():
    print("=" * 50)
    print("Loading dataset...")
    print("=" * 50)

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


def split_data(df):

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Exited"]
    )

    train_df.to_csv(
        os.path.join(PROCESSED_DIR, "train.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(PROCESSED_DIR, "test.csv"),
        index=False
    )

    print("\nTrain Shape:", train_df.shape)
    print("Test Shape :", test_df.shape)

    print("\nFiles saved to data/processed/")


if __name__ == "__main__":
    df = load_data()
    split_data(df)