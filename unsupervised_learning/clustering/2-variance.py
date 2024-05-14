#!/usr/bin/env python3
""" This module creates the variance function"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n - number of data points
        d - number of dimensions for each data point
    C - numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster

    Returns:
    var, or None on failure
    var - total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    n, d1 = X.shape
    k, d2 = C.shape

    if d1 != d2:
        return None

    # Extend centroids to match distance calculation to the shape of X
    extended_Cents = C[:, np.newaxis]

    # Calculate distances from each point to each centroid
    distances = np.sqrt(((X - extended_Cents) ** 2).sum(axis=2))

    # Find the minimum distance to each point to any centroid
    min_distance = np.min(distances, axis=0)

    variance = np.sum(min_distance ** 2)

    return variance
