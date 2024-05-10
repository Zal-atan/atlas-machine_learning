#!/usr/bin/env python3
"""This module contains the function pca(X, ndim)"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset

    Inputs:
    X: np.ndarray, shape(n, d) - dataset
        n: number of data points
        d: number of dimensions
    ndim: new dimensionality of the transformed X

    Returns:
    T, a numpy.ndarray of shape (n, ndim) containing the
        transformed version of X
    """
    # Compute mean
    X_mean = np.mean(X, axis=0, keepdims=True)

    # Center the data
    A = X - X_mean

    # Singular Value Decomposition (SVD)
    _, _, Vt = np.linalg.svd(A)

    # Extract top 'ndim' components
    W = Vt.T[:, :ndim]

    # Project the centered data onto the new basis
    T = np.matmul(A, W)

    return T
