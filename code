"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Ensemble Evaluation

Author : Badrinath & Abdul Azim

Description
-----------
Compares individual expert models and ensemble strategies for customer
churn prediction.

Compared Strategies
-------------------
1. CatBoost
2. LightGBM
3. XGBoost
4. Average Ensemble
5. Weighted Ensemble
6. Adaptive MoE

Metrics
-------
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
===========================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# -------------------------------------------------------------------------
# Project Path
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT / "src")
)

# -------------------------------------------------------------------------
# Project Imports
# -------------------------------------------------------------------------

from research.data.dataset import load_dataset

from research.experts.expert_manager import ExpertManager
from research.experts.catboost_expert import CatBoostExpert
from research.experts.lightgbm_expert import LightGBMExpert
from research.experts.xgboost_expert import XGBoostExpert

from research.adaptive_moe.model import AdaptiveMoE

from research.configs.trainer_config import TrainerConfig


# =========================================================================
# Configuration
# =========================================================================

DATASET_NAME = "dataset_1"

EXPERT_MODEL_PATH = Path(
    "models/experts/dataset_1"
)

GATING_MODEL_PATH = Path(
    "models/gating/gating_network.pt"
)

RANDOM_STATE = 42


# =========================================================================
# Utility Functions
# =========================================================================

def calculate_metrics(
    y_true,
    probabilities,
):
    """
    Calculate classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.

    probabilities : array-like
        Predicted churn probabilities.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
    }


# =========================================================================
# Main Evaluation
# =========================================================================

def main():

    print("=" * 70)
    print("AHDHE ENSEMBLE EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------------------------

    print("\nLoading dataset...")

    data = load_dataset(
        DATASET_NAME
    )

    X_test = data["X_test"]
    y_test = data["y_test"]

    feature_names = data[
        "feature_names"
    ]

    print(
        f"Testing Samples : {X_test.shape}"
    )

    print(
        f"Features        : {len(feature_names)}"
    )

    # ---------------------------------------------------------------------
    # Create Expert Manager
    # ---------------------------------------------------------------------

    print("\nLoading expert models...")

    manager = ExpertManager()

    manager.register(
        CatBoostExpert()
    )

    manager.register(
        LightGBMExpert()
    )

    manager.register(
        XGBoostExpert()
    )

    manager.load_all(
        EXPERT_MODEL_PATH
    )

    print(
        "Expert models loaded successfully."
    )

    print(
        "\nExperts:"
    )

    for name in manager.get_expert_names():

        print(
            f"  - {name}"
        )

    # ---------------------------------------------------------------------
    # Generate Expert Predictions
    # ---------------------------------------------------------------------

    print("\nGenerating expert predictions...")

    expert_predictions = (
        manager.predict_proba(
            X_test
        )
    )

    print(
        "Prediction Matrix Shape : "
        f"{expert_predictions.shape}"
    )

    # ---------------------------------------------------------------------
    # Verify Expert Count
    # ---------------------------------------------------------------------

    n_experts = len(
        manager
    )

    if n_experts != 3:

        raise ValueError(
            "Expected exactly 3 experts, "
            f"but found {n_experts}."
        )

    # ---------------------------------------------------------------------
    # Individual Expert Evaluation
    # ---------------------------------------------------------------------

    results = {}

    expert_names = (
        manager.get_expert_names()
    )

    print("\n")
    print("=" * 70)
    print("INDIVIDUAL EXPERT EVALUATION")
    print("=" * 70)

    for index, name in enumerate(
        expert_names
    ):

        probability = (
            expert_predictions[:, index]
        )

        metrics = calculate_metrics(
            y_test,
            probability,
        )

        results[name] = metrics

        print(
            f"\n{name}"
        )

        print(
            f"Accuracy  : "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{metrics['Precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{metrics['Recall']:.4f}"
        )

        print(
            f"F1 Score  : "
            f"{metrics['F1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{metrics['ROC-AUC']:.4f}"
        )

    # ---------------------------------------------------------------------
    # Average Ensemble
    # ---------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("AVERAGE ENSEMBLE")
    print("=" * 70)

    average_probability = (
        expert_predictions.mean(
            axis=1
        )
    )

    average_metrics = calculate_metrics(
        y_test,
        average_probability,
    )

    results[
        "Average Ensemble"
    ] = average_metrics

    print(
        f"Accuracy  : "
        f"{average_metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{average_metrics['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{average_metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{average_metrics['F1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{average_metrics['ROC-AUC']:.4f}"
    )

    # ---------------------------------------------------------------------
    # Weighted Ensemble
    # ---------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WEIGHTED ENSEMBLE")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # IMPORTANT
    #
    # These are initial weights for comparison only.
    # Later we will replace them with weights selected using validation
    # performance instead of manually assigning them.
    # ---------------------------------------------------------------------

    weights = np.array(
        [
            1 / 3,
            1 / 3,
            1 / 3,
        ],
        dtype=np.float32,
    )

    weights = (
        weights
        / weights.sum()
    )

    print(
        "Weights:"
    )

    for name, weight in zip(
        expert_names,
        weights,
    ):

        print(
            f"{name:<15} : "
            f"{weight:.4f}"
        )

    weighted_probability = (
        expert_predictions
        * weights
    ).sum(
        axis=1
    )

    weighted_metrics = calculate_metrics(
        y_test,
        weighted_probability,
    )

    results[
        "Weighted Ensemble"
    ] = weighted_metrics

    print(
        f"\nAccuracy  : "
        f"{weighted_metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{weighted_metrics['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{weighted_metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{weighted_metrics['F1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{weighted_metrics['ROC-AUC']:.4f}"
    )

    # ---------------------------------------------------------------------
    # Adaptive MoE
    # ---------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ADAPTIVE MoE")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device : {device}"
    )

    model = AdaptiveMoE(
        input_dim=len(
            feature_names
        ),
        n_experts=3,
    ).to(device)

    # ---------------------------------------------------------------------
    # Load trained gating model
    # ---------------------------------------------------------------------

    if not GATING_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Gating model not found:\n"
            f"{GATING_MODEL_PATH}\n\n"
            "Train the Adaptive MoE first."
        )

    checkpoint = torch.load(
        GATING_MODEL_PATH,
        map_location=device,
    )

    # ---------------------------------------------------------------------
    # Support both checkpoint formats:
    #
    # 1. Direct state_dict
    # 2. Dictionary containing model_state_dict
    # ---------------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    # ---------------------------------------------------------------------
    # Tensor Conversion
    # ---------------------------------------------------------------------

    X_tensor = torch.tensor(
        X_test.values,
        dtype=torch.float32,
        device=device,
    )

    expert_tensor = torch.tensor(
        expert_predictions,
        dtype=torch.float32,
        device=device,
    )

    # ---------------------------------------------------------------------
    # Adaptive Prediction
    # ---------------------------------------------------------------------

    with torch.no_grad():

        adaptive_probability, weights_tensor = (
            model(
                X_tensor,
                expert_tensor,
            )
        )

    adaptive_probability = (
        adaptive_probability
        .cpu()
        .numpy()
    )

    adaptive_weights = (
        weights_tensor
        .cpu()
        .numpy()
    )

    adaptive_metrics = calculate_metrics(
        y_test,
        adaptive_probability,
    )

    results[
        "Adaptive MoE"
    ] = adaptive_metrics

    print(
        f"Accuracy  : "
        f"{adaptive_metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{adaptive_metrics['Precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{adaptive_metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{adaptive_metrics['F1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{adaptive_metrics['ROC-AUC']:.4f}"
    )

    # ---------------------------------------------------------------------
    # Show Adaptive Weights
    # ---------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SAMPLE ADAPTIVE WEIGHTS")
    print("=" * 70)

    for index in range(
        min(5, len(adaptive_weights))
    ):

        print(
            f"Sample {index + 1}: "
            f"{adaptive_weights[index]}"
        )

    # ---------------------------------------------------------------------
    # Results Table
    # ---------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL ENSEMBLE COMPARISON")
    print("=" * 70)

    results_dataframe = pd.DataFrame(
        results
    ).T

    print(
        results_dataframe.to_string(
            float_format=lambda value:
            f"{value:.4f}"
        )
    )

    # ---------------------------------------------------------------------
    # Determine Best Strategy
    # ---------------------------------------------------------------------

    best_strategy = (
        results_dataframe[
            "ROC-AUC"
        ].idxmax()
    )

    best_auc = (
        results_dataframe.loc[
            best_strategy,
            "ROC-AUC",
        ]
    )

    print("\n")
    print("=" * 70)
    print("BEST STRATEGY")
    print("=" * 70)

    print(
        f"Strategy : {best_strategy}"
    )

    print(
        f"ROC-AUC  : {best_auc:.4f}"
    )

    # ---------------------------------------------------------------------
    # Save Results
    # ---------------------------------------------------------------------

    output_directory = Path(
        "reports/evaluation"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "ensemble_comparison.csv"
    )

    results_dataframe.to_csv(
        output_path
    )

    print("\n")
    print(
        "Results saved -> "
        f"{output_path}"
    )

    print("\n")
    print("=" * 70)
    print("ENSEMBLE EVALUATION COMPLETED")
    print("=" * 70)


# =========================================================================
# Entry Point
# =========================================================================

if __name__ == "__main__":

    main()