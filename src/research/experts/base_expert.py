"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)
Base Expert Interface

Author : Badrinath & Abdul Azim
Project: AI-Powered Customer Churn Prediction with Adaptive Mixture of
         Experts and MLOps Framework

Description
-----------
This module defines the abstract interface that every expert model in the
Adaptive Mixture of Experts (A-MoE) framework must implement.

Every expert (CatBoost, LightGBM, XGBoost, and future models) must inherit
from this class to ensure a common API throughout the framework.

Design Principles
-----------------
• Single Responsibility Principle
• Open-Closed Principle
• Dependency Inversion Principle

Required Methods
----------------
fit()
predict()
predict_proba()
save()
load()

=========================================================================== 
"""

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseExpert(ABC):
    """
    Abstract base class for every expert model.

    Every expert used in the Adaptive Mixture of Experts framework must
    implement this interface.

    This allows the Expert Manager and Adaptive MoE Pipeline to interact
    with every expert identically, regardless of the underlying ML library.
    """

    def __init__(self, model_name: str) -> None:
        """
        Parameters
        ----------
        model_name : str
            Human-readable expert name.
        """

        self.model_name = model_name
        self.model = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Train the expert model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.

        y_train : pd.Series
            Training labels.
        """
        pass

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        numpy.ndarray
        """
        pass

    @abstractmethod
    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        numpy.ndarray
        """
        pass

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @abstractmethod
    def save(
        self,
        model_path: Path,
    ) -> None:
        """
        Save trained model.
        """
        pass

    @abstractmethod
    def load(
        self,
        model_path: Path,
    ) -> None:
        """
        Load trained model.
        """
        pass

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def get_name(self) -> str:
        """
        Returns
        -------
        str
            Expert model name.
        """
        return self.model_name

    def __repr__(self) -> str:
        """
        String representation of expert.
        """

        return (
            f"{self.__class__.__name__}"
            f"(model_name='{self.model_name}')"
        )