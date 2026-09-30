"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Trainer Configuration

Author : Badrinath & Abdul Azim
===========================================================================
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TrainerConfig:
    """
    Configuration for Gating Network Training.
    """

    # ------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------ #

    dataset_name: str = "dataset_1"

    # ------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------ #

    epochs: int = 50

    batch_size: int = 256

    learning_rate: float = 1e-3

    # ------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------ #

    early_stopping_patience: int = 10

    # ------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------ #

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------ #

    model_directory: Path = Path(
        "models/gating"
    )

    model_name: str = "gating_network.pt"

    # ------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------ #

    verbose: bool = True

    # ------------------------------------------------------------ #

    @property
    def model_path(self) -> Path:

        return self.model_directory / self.model_name