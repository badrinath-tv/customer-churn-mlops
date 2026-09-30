"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Adaptive Mixture of Experts Model

Author : Badrinath & Abdul Azim
===========================================================================
"""

import torch
import torch.nn as nn

from research.fusion.weighted_fusion import WeightedFusion
from research.gating.gating_network import GatingNetwork


class AdaptiveMoE(nn.Module):
    """
    Adaptive Mixture of Experts.

    Inputs
    ------
    features
        Customer feature matrix.

    expert_predictions
        Probability predictions from expert models.

    Output
    ------
    Final churn probability.
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int = 3,
    ):

        super().__init__()

        self.gating_network = GatingNetwork(
            input_dim=input_dim,
            n_experts=n_experts,
        )

        self.fusion = WeightedFusion()

    # ------------------------------------------------------------ #

    def forward(
        self,
        features: torch.Tensor,
        expert_predictions: torch.Tensor,
    ):

        expert_weights = self.gating_network(
            features
        )

        final_prediction = self.fusion(
            expert_predictions,
            expert_weights,
        )

        return final_prediction, expert_weights

    # ------------------------------------------------------------ #

    def predict(
        self,
        features,
        expert_predictions,
    ):

        self.eval()

        with torch.no_grad():

            probabilities, _ = self.forward(
                features,
                expert_predictions,
            )

        return probabilities