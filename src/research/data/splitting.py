"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Dataset Splitting Module

Author : Badrinath & Abdul Azim
===========================================================================
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:

        self.test_size = test_size
        self.random_state = random_state

    # ------------------------------------------------------------------ #

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple:

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=y,
        )

        print("\nTrain/Test Split")
        print("-" * 40)

        print(f"Train : {X_train.shape}")
        print(f"Test  : {X_test.shape}")

        return (

            X_train,

            X_test,

            y_train,

            y_test,
        )