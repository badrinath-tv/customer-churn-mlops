"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Training Callbacks

Author : Badrinath & Abdul Azim
===========================================================================
"""

import torch


class EarlyStopping:
    """
    Early stopping callback.
    """

    def __init__(
        self,
        patience=10,
        min_delta=0.0,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

    # ------------------------------------------------------------ #

    def step(
        self,
        validation_loss,
    ):

        if validation_loss < (
            self.best_loss - self.min_delta
        ):

            self.best_loss = validation_loss

            self.counter = 0

            return False

        self.counter += 1

        return self.counter >= self.patience


# ------------------------------------------------------------------ #


class ModelCheckpoint:
    """
    Saves best gating network.
    """

    def __init__(
        self,
        path,
    ):

        self.path = path

        self.best_loss = float("inf")

    # ------------------------------------------------------------ #

    def step(
        self,
        validation_loss,
        model,
    ):

        if validation_loss < self.best_loss:

            self.best_loss = validation_loss

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(

                model.state_dict(),

                self.path,

            )

            print(
                f"Best model saved -> {self.path}"
            )