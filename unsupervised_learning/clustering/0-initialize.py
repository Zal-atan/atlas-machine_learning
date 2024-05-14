#!/usr/bin/env python3
""" This module creates the initialize function"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n - number of data points
        d - number of dimensions for each data point
    k - positive integer containing the number of clusters

    Returns:
    numpy.ndarray of shape (k, d) containing the initialized centroids for
    each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min_val = X.min(axis=0)
    max_val = X.max(axis=0)

    centroid = np.random.uniform(min_val, max_val, size=(k, X.shape[1]))

    return centroid
