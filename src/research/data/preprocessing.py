"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Data Preprocessing Module

Author : Badrinath & Abdul Azim

Description
-----------
Performs dataset preprocessing before model training.

Responsibilities
----------------
1. Remove identifier columns
2. Separate features and target
3. Encode categorical features
4. Return processed dataset

=========================================================================== 
"""

from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from research.data.encoding import FeatureEncoder


class DataPreprocessor:
    """
    Dataset preprocessing pipeline.
    """

    def __init__(
        self,
        target_column: str,
        drop_columns: List[str],
    ) -> None:

        self.target_column = target_column
        self.drop_columns = drop_columns

        self.encoder = FeatureEncoder()

    # ------------------------------------------------------------------ #

    def remove_identifier_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove identifier columns.
        """

        existing_columns = [
            column
            for column in self.drop_columns
            if column in dataframe.columns
        ]

        dataframe = dataframe.drop(
            columns=existing_columns
        )

        print("\nRemoved Identifier Columns")
        print(existing_columns)

        return dataframe

    # ------------------------------------------------------------------ #

    def split_features_target(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target.
        """

        X = dataframe.drop(
            columns=[self.target_column]
        )

        y = dataframe[
            self.target_column
        ]

        return X, y

    # ------------------------------------------------------------------ #

    def encode_features(
        self,
        X: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        Dict[str, LabelEncoder],
    ]:
        """
        Encode categorical features.
        """

        X, encoders = self.encoder.fit_transform(
            X
        )

        self.encoder.summary()

        return X, encoders

    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.Series,
        Dict[str, LabelEncoder],
    ]:
        """
        Complete preprocessing pipeline.
        """

        dataframe = self.remove_identifier_columns(
            dataframe
        )

        X, y = self.split_features_target(
            dataframe
        )

        X, encoders = self.encode_features(
            X
        )

        print("\nProcessed Feature Shape")
        print(X.shape)

        print("Target Shape")
        print(y.shape)

        return (
            X,
            y,
            encoders,
        )

    # ------------------------------------------------------------------ #

    def get_encoder(self) -> FeatureEncoder:
        """
        Returns FeatureEncoder instance.
        """

        return self.encoder