"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Evaluation Metrics

===========================================================================
"""

import numpy as np

from sklearn.metrics import accuracy_score


def compute_accuracy(
    y_true,
    y_probability,
):

    predictions = (
        np.array(
            y_probability
        )
        >= 0.5
    ).astype(int)

    return accuracy_score(
        y_true,
        predictions,
    )