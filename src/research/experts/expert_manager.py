"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)
Expert Manager

Author : Badrinath & Abdul Azim
Project: AI-Powered Customer Churn Prediction with Adaptive Mixture of
         Experts and MLOps Framework

Description
-----------
The Expert Manager is responsible for managing all expert models used in
the Adaptive Mixture of Experts framework.

Responsibilities
----------------
• Register experts
• Train all experts
• Save all experts
• Load all experts
• Generate prediction matrix
=========================================================================== 
"""

from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from .base_expert import BaseExpert


class ExpertManager:
    """
    Manager for all expert models.
    """

    def __init__(self) -> None:

        self.experts: List[BaseExpert] = []

    # ------------------------------------------------------------------ #
    # Expert Registration
    # ------------------------------------------------------------------ #

    def register(self, expert: BaseExpert) -> None:
        """
        Register an expert model.

        Parameters
        ----------
        expert : BaseExpert
        """

        self.experts.append(expert)

    # ------------------------------------------------------------------ #

    def get_expert_names(self) -> List[str]:
        """
        Returns names of registered experts.
        """

        return [
            expert.get_name()
            for expert in self.experts
        ]

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Train every registered expert.
        """

        print("\nTraining Expert Models")
        print("-" * 50)

        for expert in self.experts:

            print(f"Training {expert.get_name()}...")

            expert.fit(
                X_train,
                y_train,
            )

        print("\nAll experts trained successfully.")

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """
        Predict class labels using every expert.

        Returns
        -------
        dict
        """

        predictions = {}

        for expert in self.experts:

            predictions[
                expert.get_name()
            ] = expert.predict(X)

        return predictions

    # ------------------------------------------------------------------ #

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate prediction matrix.

        Returns
        -------
        ndarray

        Shape

        (n_samples, n_experts)
        """

        probability_matrix = []

        for expert in self.experts:

            probability_matrix.append(
                expert.predict_proba(X)
            )

        probability_matrix = np.column_stack(
            probability_matrix
        )

        return probability_matrix

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_all(
        self,
        save_directory: Path,
    ) -> None:
        """
        Save every expert.
        """

        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for expert in self.experts:

            filename = (
                expert.get_name().lower()
                + ".pkl"
            )

            expert.save(
                save_directory / filename
            )

    # ------------------------------------------------------------------ #

    def load_all(
        self,
        save_directory: Path,
    ) -> None:
        """
        Load every expert.
        """

        for expert in self.experts:

            filename = (
                expert.get_name().lower()
                + ".pkl"
            )

            expert.load(
                save_directory / filename
            )

    # ------------------------------------------------------------------ #
    # Feature Importance
    # ------------------------------------------------------------------ #

    def feature_importance(
        self,
        feature_names=None,
    ):
        """
        Return feature importance from every expert.

        Returns
        -------
        dict
        """

        importance = {}

        for expert in self.experts:

            if hasattr(
                expert,
                "feature_importance",
            ):

                importance[
                    expert.get_name()
                ] = expert.feature_importance(
                    feature_names
                )

        return importance

    # ------------------------------------------------------------------ #

    def summary(self) -> None:
        """
        Print registered experts.
        """

        print("\nRegistered Experts")
        print("-" * 50)

        for i, expert in enumerate(
            self.experts,
            start=1,
        ):

            print(
                f"{i}. {expert.get_name()}"
            )

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """
        Number of experts.
        """

        return len(self.experts)

    # ------------------------------------------------------------------ #

    def __iter__(self):
        """
        Iterate through experts.
        """

        return iter(self.experts)

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:

        return (
            f"ExpertManager("
            f"{len(self.experts)} Experts)"
        )