"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Dataset Validation Module

Author : Badrinath & Abdul Azim
=========================================================================== 
"""

from pathlib import Path

import pandas as pd


class DatasetValidator:
    """
    Performs dataset validation before preprocessing.
    """

    def __init__(
        self,
        dataset_path: Path,
        target_column: str,
    ) -> None:

        self.dataset_path = Path(dataset_path)
        self.target_column = target_column

    # ------------------------------------------------------------------ #

    def validate_file_exists(self) -> None:
        """
        Verify dataset file exists.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

    # ------------------------------------------------------------------ #

    def validate_target_column(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify target column exists.
        """

        if self.target_column not in dataframe.columns:

            raise ValueError(
                f"Target column '{self.target_column}' "
                "not found."
            )

    # ------------------------------------------------------------------ #

    def validate_empty_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Verify dataset is not empty.
        """

        if dataframe.empty:

            raise ValueError(
                "Dataset is empty."
            )

    # ------------------------------------------------------------------ #

    def validate_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Returns number of duplicate rows.
        """

        return dataframe.duplicated().sum()

    # ------------------------------------------------------------------ #

    def validate_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Returns missing values per column.
        """

        return dataframe.isnull().sum()

    # ------------------------------------------------------------------ #

    def validate(self) -> pd.DataFrame:
        """
        Complete validation pipeline.

        Returns
        -------
        pandas.DataFrame
        """

        self.validate_file_exists()

        dataframe = pd.read_csv(
            self.dataset_path
        )

        self.validate_empty_dataset(
            dataframe
        )

        self.validate_target_column(
            dataframe
        )

        duplicates = self.validate_duplicate_rows(
            dataframe
        )

        missing = self.validate_missing_values(
            dataframe
        )

        print("\n" + "=" * 70)
        print("DATASET VALIDATION")
        print("=" * 70)

        print(f"Dataset : {self.dataset_path.name}")
        print(f"Shape   : {dataframe.shape}")

        print(f"\nDuplicate Rows : {duplicates}")

        if missing.sum() == 0:
            print("Missing Values : None")
        else:
            print("\nMissing Values")
            print(missing[missing > 0])

        print("\nValidation Successful.\n")

        return dataframe