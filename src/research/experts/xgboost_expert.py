"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)
XGBoost Expert

Author : Badrinath & Abdul Azim
Project: AI-Powered Customer Churn Prediction with Adaptive Mixture of
         Experts and MLOps Framework

Description
-----------
Concrete implementation of the BaseExpert interface using XGBoost.

Features
--------
• GPU acceleration (automatic CPU fallback)
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
from xgboost import XGBClassifier

from .base_expert import BaseExpert


class XGBoostExpert(BaseExpert):
    """
    XGBoost implementation of BaseExpert.
    """

    def __init__(
        self,
        random_state: int = 42,
        use_gpu: bool = True,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
    ) -> None:

        super().__init__("XGBoost")

        self.random_state = random_state
        self.use_gpu = use_gpu

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:

        try:

            self.model.fit(
                X_train,
                y_train,
            )

        except Exception:

            print(
                "\nGPU unavailable."
                "\nFalling back to CPU...\n"
            )

            self.model = XGBClassifier(
                n_estimators=self.model.n_estimators,
                learning_rate=self.model.learning_rate,
                max_depth=self.model.max_depth,
                random_state=self.random_state,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                device="cpu",
            )

            self.model.fit(
                X_train,
                y_train,
            )

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict(X)

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:

        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(
        self,
        model_path: Path,
    ) -> None:

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

        self.model = joblib.load(model_path)

    # ------------------------------------------------------------------ #
    # Feature Importance
    # ------------------------------------------------------------------ #

    def feature_importance(
        self,
        feature_names: Optional[list] = None,
    ) -> pd.DataFrame:

        importance = self.model.feature_importances_

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

        return (
            "XGBoostExpert("
            f"GPU={self.use_gpu}, "
            f"RandomState={self.random_state})"
        )