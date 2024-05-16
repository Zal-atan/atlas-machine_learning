#!/usr/bin/env python3
""" This module creates the initialize function"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Tests for the optimum number of clusters by variance

    Inputs:
    X - numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n - number of data points
        d - number of dimensions for each data point
    k - positive integer containing the number of clusters

    Returns: pi, m, S, or None, None, None on failure
    pi - numpy.ndarray of shape (k,) containing the priors for each cluster,
    initialized evenly
    m - numpy.ndarray of shape (k, d) containing the centroid means for each
    cluster, initialized with K-means
    S - numpy.ndarray of shape (k, d, d) containing the covariance matrices
    for each cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None

    n, d = X.shape

    # 1d matrix of ones divided by k
    pi = np.ones(k) / k

    # Centroid means using kmeans
    m, _ = kmeans(X, k)

    # Covariance matrices for each cluster, initialized as identity matices
    S = np.zeros((k, d, d))
    S[:] = np.eye(d)

    return (pi, m, S)
