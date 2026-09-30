"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Categorical Feature Encoding

Author : Badrinath & Abdul Azim
===========================================================================
"""

from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEncoder:
    """
    Encodes categorical features using LabelEncoder.
    """

    def __init__(self):

        self.encoders: Dict[str, LabelEncoder] = {}

    # ------------------------------------------------------------------ #

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Fit and transform categorical columns.
        """

        dataframe = dataframe.copy()

        categorical_columns = dataframe.select_dtypes(
            include=["object"]
        ).columns

        for column in categorical_columns:

            encoder = LabelEncoder()

            dataframe[column] = encoder.fit_transform(
                dataframe[column].astype(str)
            )

            self.encoders[column] = encoder

        return dataframe, self.encoders

    # ------------------------------------------------------------------ #

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform new data using fitted encoders.
        """

        dataframe = dataframe.copy()

        for column, encoder in self.encoders.items():

            dataframe[column] = encoder.transform(
                dataframe[column].astype(str)
            )

        return dataframe

    # ------------------------------------------------------------------ #

    def get_encoders(self):

        return self.encoders

    # ------------------------------------------------------------------ #

    def summary(self):

        print("\nEncoded Columns")
        print("-" * 40)

        for column in self.encoders:

            print(column)

        print("-" * 40)