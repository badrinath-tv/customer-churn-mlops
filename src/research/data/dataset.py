"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Dataset Loader

Author : Badrinath & Abdul Azim

Description
-----------
Master data loading pipeline.

Pipeline
--------
Configuration
      │
      ▼
Validation
      │
      ▼
Preprocessing
      │
      ▼
Encoding
      │
      ▼
Train/Test Split
      │
      ▼
Ready for Training

===========================================================================
"""

from typing import Dict

from research.configs.dataset_config import DATASET_CONFIG
from research.data.preprocessing import DataPreprocessor
from research.data.splitting import DataSplitter
from research.data.validation import DatasetValidator


class DatasetLoader:
    """
    Master dataset loader.
    """

    def __init__(
        self,
        dataset_name: str,
    ) -> None:

        if dataset_name not in DATASET_CONFIG:

            raise ValueError(
                f"Unknown dataset : {dataset_name}"
            )

        self.config = DATASET_CONFIG[dataset_name]

    # ------------------------------------------------------------------ #

    def load(self) -> Dict:
        """
        Execute complete data pipeline.

        Returns
        -------
        dict
        """

        # -------------------------------------------------------------- #
        # Validation
        # -------------------------------------------------------------- #

        validator = DatasetValidator(

            dataset_path=self.config["path"],

            target_column=self.config["target"],
        )

        dataframe = validator.validate()

        # -------------------------------------------------------------- #
        # Preprocessing
        # -------------------------------------------------------------- #

        preprocessor = DataPreprocessor(

            target_column=self.config["target"],

            drop_columns=self.config["drop_columns"],
        )

        X, y, encoders = preprocessor.preprocess(
            dataframe
        )

        # -------------------------------------------------------------- #
        # Split
        # -------------------------------------------------------------- #

        splitter = DataSplitter()

        X_train, X_test, y_train, y_test = splitter.split(
            X,
            y,
        )

        # -------------------------------------------------------------- #
        # Return everything
        # -------------------------------------------------------------- #

        return {

            "X_train": X_train,

            "X_test": X_test,

            "y_train": y_train,

            "y_test": y_test,

            "encoders": encoders,

            "feature_names": list(X.columns),

            "config": self.config,
        }


# ---------------------------------------------------------------------- #

def load_dataset(
    dataset_name: str,
) -> Dict:
    """
    Convenience function.

    Example
    -------

    data = load_dataset("dataset_1")
    """

    loader = DatasetLoader(
        dataset_name
    )

    return loader.load()