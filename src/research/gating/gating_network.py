"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)
Adaptive Gating Network

Author : Badrinath & Abdul Azim

Description
-----------
PyTorch implementation of the Adaptive Gating Network.

The Gating Network receives customer features and learns how much weight
should be assigned to each expert model.

Input
-----
Customer Feature Matrix

Shape:
    (batch_size, n_features)

Example

[[619,0,1,42,2,0,1,1,101348.88,1],
 [608,2,0,41,1,83807.86,1,0,112542.58,0]]

Output
------
Expert Weights

[[0.12,0.21,0.67],
 [0.54,0.17,0.29]]

Each row sums to 1 because of the Softmax activation.

=========================================================================== 
"""

import torch
import torch.nn as nn


class GatingNetwork(nn.Module):
    """
    Adaptive Gating Network.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    n_experts : int
        Number of expert models.

    hidden_dim1 : int
        First hidden layer size.

    hidden_dim2 : int
        Second hidden layer size.
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int = 3,
        hidden_dim1: int = 32,
        hidden_dim2: int = 16,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.n_experts = n_experts

        self.network = nn.Sequential(

            # Input Layer
            nn.Linear(
                input_dim,
                hidden_dim1,
            ),

            nn.ReLU(),

            # Hidden Layer
            nn.Linear(
                hidden_dim1,
                hidden_dim2,
            ),

            nn.ReLU(),

            # Output Layer
            nn.Linear(
                hidden_dim2,
                n_experts,
            ),

            # Expert weights
            nn.Softmax(dim=1),
        )

    # ------------------------------------------------------------------ #

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        features : torch.Tensor

            Customer feature matrix.

            Shape

            (batch_size, input_dim)

        Returns
        -------
        torch.Tensor

            Expert weights.

            Shape

            (batch_size, n_experts)
        """

        return self.network(features)

    # ------------------------------------------------------------------ #

    def get_num_experts(self) -> int:
        """
        Returns
        -------
        int
            Number of expert models.
        """

        return self.n_experts

    # ------------------------------------------------------------------ #

    def get_input_dimension(self) -> int:
        """
        Returns
        -------
        int
            Number of input features.
        """

        return self.input_dim

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:

        return (
            f"GatingNetwork("
            f"InputFeatures={self.input_dim}, "
            f"Experts={self.n_experts})"
        )