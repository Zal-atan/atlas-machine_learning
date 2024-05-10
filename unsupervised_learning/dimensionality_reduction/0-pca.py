#!/usr/bin/env python3
"""This module contains the function pca(X, var=0.95)"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset

    Inputs:
    X: np.ndarray, shape(n, d) - dataset
        n: number of data points
        d: number of dimensions
    var: float - fraction of variance that PCA transformation should maintain

    Returns:
    W: np.ndarray, shape(d, nd) - weights matrix which maintains var fraction
    of X‘s original variance
        nd: new dimensions
    """

    # Singular Value Decomposition (SVD)
    _, Sigma, Vt = np.linalg.svd(X, full_matrices=False)

    # Calculate cumulative explained variance ratio
    cumu_var = np.cumsum(Sigma) / np.sum(Sigma)

    # Find dimensions for mainting variance
    dimens = np.argwhere(cumu_var >= var)[0, 0]

    # Select Correct Eigenvectors to form weights
    W = W[:, :dimens + 1]

    return W
