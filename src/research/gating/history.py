"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Training History

===========================================================================
"""


class TrainingHistory:

    def __init__(self):

        self.train_loss = []

        self.validation_loss = []

        self.validation_accuracy = []

    # ------------------------------------------------------------ #

    def update(

        self,

        train_loss,

        validation_loss,

        validation_accuracy,

    ):

        self.train_loss.append(
            train_loss
        )

        self.validation_loss.append(
            validation_loss
        )

        self.validation_accuracy.append(
            validation_accuracy
        )