"""
===========================================================================
Adaptive Mixture of Experts (A-MoE)

Loss Functions

===========================================================================
"""

import torch.nn as nn


def binary_cross_entropy():

    return nn.BCELoss()