#!/usr/bin/env python3
"""This module will create the 12_reg_cost function"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Inputs:
    cost - cost of the network without L2 regularization
    lambtha - regularization parameter
    weights - dictionary of the weights and biases (numpy.ndarrays) of the neural network
    L - number of layers in the neural network
    m - number of data points used

    Returns:
    The cost of the network accounting for L2 regularization
    """
    normal = []
    for key, value in weights.items():
        if key[0] == 'W':
            normal.append(np.sum(value * value))

    return cost + (lambtha / (2 * m) * np.sum(normal))
