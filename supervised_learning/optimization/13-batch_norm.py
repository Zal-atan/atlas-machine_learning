#!/usr/bin/env python3
""" This module creates batch_norm(Z, gamma, beta, epsilon) function
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using
    batch normalization

    Inputs:
    Z - numpy.ndarray of shape (m, n) that should be normalized
        m - number of data points
        n - number of features in Z
    gamma - numpy.ndarray of shape (1, n) containing the scales used
            for batch normalization
    beta - numpy.ndarray of shape (1, n) containing the offsets used
            for batch normalization
    epsilon - small number used to avoid division by zero

    Returns:
    The normalized Z matrix
    """

    mean = Z.mean(axis=0)
    var = Z.var(axis=0)

    Z_normalized = (Z - mean) / ((var + epsilon) ** (1/2))
    Z_matrix_norm = gamma * Z_normalized + beta

    return Z_matrix_norm
