"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Dataset Configuration

Author : Badrinath & Abdul Azim
===========================================================================
"""

from pathlib import Path

DATASET_CONFIG = {

    "dataset_1": {

        "path": Path("data/raw/Churn_Modelling.csv"),

        "target": "Exited",

        "drop_columns": [
            "RowNumber",
            "CustomerId",
            "Surname",
        ],
    },

    "dataset_2": {

        "path": Path("data/raw/Churn_Dataset_2.csv"),

        "target": "exit",

        "drop_columns": [
            "customer_id",
            "name",
            "email",
            "phone",
        ],
    },

    "dataset_3": {

        "path": Path("data/raw/churn_dataset_3.csv"),

        "target": "exit",

        "drop_columns": [
            "id",
            "full_name",
        ],
    },
}