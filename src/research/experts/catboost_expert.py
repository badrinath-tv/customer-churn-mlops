"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)
CatBoost Expert

Author : Badrinath & Abdul Azim
Project: AI-Powered Customer Churn Prediction with Adaptive Mixture of
         Experts and MLOps Framework

Description
-----------
Concrete implementation of the BaseExpert interface using CatBoost.

Features
--------
• GPU acceleration (automatic fallback to CPU)
• Probability prediction
• Model persistence
• Unified interface
=========================================================================== 
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .base_expert import BaseExpert


class CatBoostExpert(BaseExpert):
    """
    CatBoost implementation of BaseExpert.
    """

    def __init__(
        self,
        random_state: int = 42,
        use_gpu: bool = True,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
    ) -> None:
        """
        Initialize CatBoost Expert.

        Parameters
        ----------
        random_state : int
            Random seed.

        use_gpu : bool
            Use GPU if available.

        iterations : int
            Number of boosting iterations.

        learning_rate : float
            Learning rate.

        depth : int
            Maximum tree depth.
        """

        super().__init__("CatBoost")

        self.random_state = random_state
        self.use_gpu = use_gpu

        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False,
            task_type="GPU" if use_gpu else "CPU",
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Train CatBoost model.
        """

        try:

            self.model.fit(
                X_train,
                y_train,
                verbose=False,
            )

        except Exception:

            print(
                "\nGPU unavailable."
                "\nFalling back to CPU...\n"
            )

            self.model = CatBoostClassifier(
                iterations=self.model.get_param("iterations"),
                learning_rate=self.model.get_param("learning_rate"),
                depth=self.model.get_param("depth"),
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=self.random_state,
                verbose=False,
                task_type="CPU",
            )

            self.model.fit(
                X_train,
                y_train,
                verbose=False,
            )

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict class labels.
        """

        return self.model.predict(X)

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict churn probabilities.

        Returns
        -------
        ndarray

        Probability of positive class.
        """

        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(
        self,
        model_path: Path,
    ) -> None:
        """
        Save trained model.
        """

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.model,
            model_path,
        )

    def load(
        self,
        model_path: Path,
    ) -> None:
        """
        Load trained model.
        """

        self.model = joblib.load(model_path)

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def feature_importance(
        self,
        feature_names: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Return feature importance.

        Parameters
        ----------
        feature_names : list, optional

        Returns
        -------
        pd.DataFrame
        """

        importance = self.model.get_feature_importance()

        if feature_names is None:
            feature_names = [
                f"Feature_{i}"
                for i in range(len(importance))
            ]

        return (
            pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": importance,
                }
            )
            .sort_values(
                by="Importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    def __repr__(self) -> str:
        """
        String representation.
        """

        return (
            "CatBoostExpert("
            f"GPU={self.use_gpu}, "
            f"RandomState={self.random_state})"
        )